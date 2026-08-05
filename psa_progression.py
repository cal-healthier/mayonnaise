"""
psa_progression.py -- how LONG does hormone therapy keep working?

Not "does PSA drop" (mostly settled by where it started) but "when does it
start climbing again" -- the onset of castration resistance. That is the
question that changes management, and it is not determined by baseline.

PCWG3 definition of PSA progression:
  rise of >=25% AND >=2 ng/mL above the running nadir, CONFIRMED by a second
  qualifying value >=21 days later.

Time zero = treatment start. Men who never progress are censored at their last
PSA -- proper time-to-event, same C-index machinery as the survival work.

Label diagnostics print BEFORE any model.
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200)

L0 = pd.read_parquet("psa_label.parquet")
labs = globals().get("labs")
if labs is not None:
    labs.to_parquet("psa_labs.parquet")          # cache so a kernel restart is survivable
else:
    try:
        labs = pd.read_parquet("psa_labs.parquet")
    except Exception:
        raise SystemExit("no `labs` in kernel and no psa_labs.parquet -- run psa_model.py first")
print(f"lab panel: {labs.shape[1]} columns, {labs.shape[0]:,} men")

# ---------------------------------------------------------------- 1. long-window PSA
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
psa_c = cc[cc["name_l"].str.contains("prostate specific|prostate-specific", na=False)
           | cc["code"].astype(str).isin(["2857-1", "83112-3", "35741-8", "19195-7"])]
PSA_SQL = ",".join(str(int(x)) for x in psa_c.sort_values("n_persons", ascending=False).head(6)["cid"])

ids = "','".join(L0.index.astype(str))
print("\npulling full PSA history (no 1-year cap this time) ...")
pv = C.query(f"""
  WITH ppl AS (
    SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, MIN(pe.person_id) AS person_id
    FROM {D}.DIM_PATIENT p
    JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                        = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
    WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
    GROUP BY 1)
  SELECT ppl.clinic, DATE(m.measurement_date) AS d, m.value_as_number AS psa
  FROM ppl JOIN {D}.measurement m ON m.person_id = ppl.person_id
  WHERE m.measurement_concept_id IN ({PSA_SQL})
    AND m.value_as_number IS NOT NULL AND m.value_as_number >= 0
""").to_dataframe()

tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
pv["d"] = pd.to_datetime(pv["d"])
pv = pv.join(tx.rename("tx"), on="clinic").dropna(subset=["tx"])
pv["day"] = (pv["d"] - pv["tx"]).dt.days
print(f"  PSA rows: {len(pv):,}   men: {pv['clinic'].nunique():,}")

post = pv[pv["day"] > 0].sort_values(["clinic", "day"]).copy()
fu = post.groupby("clinic")["day"].max()
print(f"  median follow-up after treatment: {fu.median()/365.25:.1f} years "
      f"(p90 {fu.quantile(.9)/365.25:.1f})")
print(f"  median PSA draws after treatment: {post.groupby('clinic').size().median():.0f}")

# ---------------------------------------------------------------- 2. PCWG3 progression
post["nadir_run"] = post.groupby("clinic")["psa"].cummin()
post["flag"] = (post["psa"] >= 1.25 * post["nadir_run"]) & (post["psa"] >= post["nadir_run"] + 2)

rows = []
for clinic, g in post.groupby("clinic", sort=False):
    days = g["day"].values
    flags = g["flag"].values
    last = days[-1]
    ev, t = 0, last
    fd = days[flags]
    if len(fd) >= 2:
        for i, day0 in enumerate(fd[:-1]):
            if (fd[i + 1:] >= day0 + 21).any():       # confirmed >=21 days later
                ev, t = 1, day0
                break
    rows.append({"clinic": clinic, "prog": ev, "time": t,
                 "n_post": len(g), "nadir": g["psa"].min(),
                 "last_day": last})
E = pd.DataFrame(rows).set_index("clinic")
E = E.join(L0[["baseline"]], how="inner")
E = E[(E["n_post"] >= 3) & (E["last_day"] >= 90) & (E["baseline"] > 0)]
E.to_parquet("psa_progression.parquet")

print("\n" + "=" * 74)
print("THE LABEL: time to PSA progression (castration resistance)")
print("=" * 74)
print(f"  men with >=3 follow-up PSAs and >=90 days:   {len(E):,}")
print(f"  progressed:                                  {int(E['prog'].sum()):,} "
      f"({E['prog'].mean():.1%})   <-- THE NUMBER")
print(f"  censored (still responding at last PSA):     {int((1-E['prog']).sum()):,}")
print(f"  median time to progression (progressors):    "
      f"{E.loc[E['prog']==1,'time'].median()/365.25:.2f} years")
print(f"  median follow-up (censored):                 "
      f"{E.loc[E['prog']==0,'time'].median()/365.25:.2f} years")

print("\n  progression rate by starting PSA (is this baseline-driven too?):")
E["bl_bin"] = pd.cut(E["baseline"], [0, 0.5, 1, 2, 4, 10, 20, 100, 1e9],
                     labels=["<0.5", "0.5-1", "1-2", "2-4", "4-10", "10-20", "20-100", ">100"])
for b, r in E.groupby("bl_bin", observed=True).agg(
        n=("prog", "size"), rate=("prog", "mean"),
        med=("time", "median")).iterrows():
    print(f"    {str(b):<8} n={int(r['n']):>5}  progressed {r['rate']:6.1%}  "
          f"median time {r['med']/365.25:.2f} yr")

# ---------------------------------------------------------------- 3. models
def cindex(t, e, risk, n_pairs=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(risk, float)
    n = len(t); conc = perm = 0
    for _ in range(0, n_pairs, 500_000):
        i = rng.integers(0, n, 500_000); j = rng.integers(0, n, 500_000)
        ei = (t[i] < t[j]) & e[i]; ej = (t[j] < t[i]) & e[j]
        ok = ei | ej
        if not ok.any():
            continue
        ri, rj = r[i[ok]], r[j[ok]]
        hi = np.where(ei[ok], ri, rj); lo = np.where(ei[ok], rj, ri)
        conc += np.sum(hi > lo) + 0.5 * np.sum(hi == lo); perm += ok.sum()
    return conc / perm

def prep(df, idx):
    X = df.select_dtypes(include=[np.number]).reindex(idx)
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def run(X, name, folds=5):
    s, oof = [], pd.Series(index=X.index, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        m = LogisticRegression(class_weight="balanced", max_iter=4000, C=0.5)
        m.fit(sc.transform(imp.transform(X.iloc[tr])), E["prog"].iloc[tr])
        p = m.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]
        oof.iloc[te] = p
        s.append(cindex(E["time"].iloc[te], E["prog"].iloc[te], p))
    a = np.array(s)
    print(f"  {name:<44}C-index {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} feat)")
    return a.mean(), oof

B = pd.DataFrame({"baseline": E["baseline"], "log_baseline": np.log1p(E["baseline"])})
LB = prep(labs, E.index)
print("\n" + "=" * 74)
print(f"WHO STOPS RESPONDING, AND WHEN?   n={len(E):,}, "
      f"{int(E['prog'].sum()):,} events")
print("=" * 74)
c_psa, _ = run(prep(B, E.index), "PSA at treatment start only")
c_lab, _ = run(LB, "routine bloodwork only (no PSA)")
c_all, oof = run(prep(B.join(LB, how="left"), E.index), "PSA + bloodwork")

# ---------------------------------------------------------------- 4. KM chart
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def km(t, e):
        d = pd.DataFrame({"t": t, "e": e}).sort_values("t")
        ts, ss, s, n = [0.0], [1.0], 1.0, len(d)
        for tt, g in d.groupby("t"):
            if n <= 0:
                break
            k = int(g["e"].sum())
            if k:
                s *= (1 - k / n); ts.append(tt); ss.append(s)
            n -= len(g)
        return ts, ss

    grp = pd.qcut(oof, 3, labels=["Low risk", "Medium risk", "High risk"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g, col in zip(["Low risk", "Medium risk", "High risk"],
                      ["#1f9e5a", "#e8a33d", "#c0392b"]):
        sel = grp == g
        t, s = km(E.loc[sel, "time"].values / 365.25, E.loc[sel, "prog"].values)
        ax.step(t, s, where="post", lw=2.5, color=col, label=f"{g} (n={int(sel.sum()):,})")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    ax.set_xlabel("Years since starting hormone therapy")
    ax.set_ylabel("Still responding")
    ax.set_title("Predicted risk separates how long hormone therapy keeps working",
                 fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("psa_progression_km.png", dpi=200, bbox_inches="tight")
    print("\nwrote psa_progression_km.png")
except Exception as e:
    print(f"\n(chart skipped: {type(e).__name__} {str(e)[:80]})")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"psa_progression | n={len(E)} | events={int(E['prog'].sum())} "
      f"({E['prog'].mean():.0%}) | c_psa={c_psa:.3f} | c_blood={c_lab:.3f} "
      f"| c_both={c_all:.3f} | med_yrs={E.loc[E['prog']==1,'time'].median()/365.25:.2f}")

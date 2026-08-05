"""
ovarian_study.py -- the prostate study, ported to ovarian cancer.

Same question, different cancer, different standard marker:
  does routine bloodwork predict when platinum chemotherapy stops working
  better than CA-125, the marker oncologists actually use?

If yes, the prostate finding generalises. If no, it was a prostate quirk.
That is the whole point of running this.

Endpoint: time to CA-125 progression, GCIG (Rustin) criteria --
  CA-125 >= 2x the running nadir, or >= 2x the upper limit of normal (35 U/mL)
  if the nadir normalised, CONFIRMED by a second qualifying value >=7 days later.
  threshold = 2 * max(running_nadir, 35) captures both arms cleanly.

Everything is cached to parquet as it goes, so a crash never costs a re-query.
Drug search is NULL-safe (the `||` bug returned 0 platinum patients).
`sign_height` is excluded -- it is mismapped (values 0.02-0.05).
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200)
ULN = 35.0                      # CA-125 upper limit of normal, U/mL

# ---------------------------------------------------------------- setup
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
ca = cc[cc["name_l"].str.contains("ca 125|ca-125|cancer ag 125|cancer antigen 125", na=False)]
ca = ca[~ca["name_l"].str.contains("cortisol", na=False)]
CA_SQL = ",".join(str(int(x)) for x in ca.sort_values("n_persons", ascending=False).head(4)["cid"])
print(f"CA-125 concepts: {CA_SQL}")

medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
PLAT = ["CARBOPLATIN", "CISPLATIN", "OXALIPLATIN"]
like = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in PLAT)

CTES = f"""
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3)) AS site3
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND (CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C56%'
      OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C57%'
      OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C48%')
  GROUP BY 1),
plat AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
           FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, r.site3, pl.tx_date
  FROM reg r JOIN plat pl USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK) JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND pl.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY pl.tx_date) = 1)
"""

# ---------------------------------------------------------------- 1. CA-125 history
if os.path.exists("ov_ca125.parquet"):
    cv = pd.read_parquet("ov_ca125.parquet")
    print("loaded cached ov_ca125.parquet")
else:
    print("pulling CA-125 history ...")
    cv = C.query(f"""
    WITH {CTES}
    SELECT c.clinic, c.site3, c.tx_date, DATE(m.measurement_date) AS d,
           m.value_as_number AS ca
    FROM cohort c JOIN {D}.measurement m ON m.person_id = c.person_id
    WHERE m.measurement_concept_id IN ({CA_SQL})
      AND m.value_as_number IS NOT NULL AND m.value_as_number >= 0
    """).to_dataframe()
    cv["tx_date"] = pd.to_datetime(cv["tx_date"]); cv["d"] = pd.to_datetime(cv["d"])
    cv["day"] = (cv["d"] - cv["tx_date"]).dt.days
    cv.to_parquet("ov_ca125.parquet")
print(f"  CA-125 rows {len(cv):,}   women {cv['clinic'].nunique():,}")

# ---------------------------------------------------------------- 2. GCIG progression
pre = cv[(cv["day"] <= 0) & (cv["day"] >= -90)].sort_values("day").groupby("clinic").tail(1)
pre = pre.set_index("clinic")["ca"].rename("baseline")
post = cv[cv["day"] > 0].sort_values(["clinic", "day"]).copy()
post["nadir_run"] = post.groupby("clinic")["ca"].cummin()
post["thresh"] = 2.0 * np.maximum(post["nadir_run"], ULN)
post["flag"] = post["ca"] >= post["thresh"]

rows = []
for clinic, g in post.groupby("clinic", sort=False):
    days, flags = g["day"].values, g["flag"].values
    last = days[-1]
    ev, t = 0, last
    fd = days[flags]
    if len(fd) >= 2:
        for i, d0 in enumerate(fd[:-1]):
            if (fd[i + 1:] >= d0 + 7).any():          # GCIG: confirmed >=1 week later
                ev, t = 1, d0
                break
    rows.append({"clinic": clinic, "prog": ev, "time": t, "n_post": len(g),
                 "nadir": g["ca"].min(), "last_day": last,
                 "site3": g["site3"].iloc[0]})
E = pd.DataFrame(rows).set_index("clinic").join(pre, how="inner")
E = E[(E["n_post"] >= 3) & (E["last_day"] >= 90) & (E["baseline"] > 0)]
E.to_parquet("ov_label.parquet")

print("\n" + "=" * 74)
print("THE LABEL: time to CA-125 progression (GCIG criteria)")
print("=" * 74)
print(f"  women with >=3 follow-up CA-125 and >=90 days: {len(E):,}")
print(f"  progressed:                                    {int(E['prog'].sum()):,} "
      f"({E['prog'].mean():.1%})")
print(f"  median time to progression:                    "
      f"{E.loc[E['prog']==1,'time'].median()/365.25:.2f} years")
print(f"  median follow-up (censored):                   "
      f"{E.loc[E['prog']==0,'time'].median()/365.25:.2f} years")
print(f"  median CA-125 draws after treatment:           {E['n_post'].median():.0f}")

print("\n  progression rate by starting CA-125 (baseline-driven, like prostate?):")
E["bl_bin"] = pd.cut(E["baseline"], [0, 35, 100, 300, 1000, 3000, 1e9],
                     labels=["<=35 (normal)", "35-100", "100-300", "300-1000",
                             "1000-3000", ">3000"])
for b, r in E.groupby("bl_bin", observed=True).agg(
        n=("prog", "size"), rate=("prog", "mean"), med=("time", "median")).iterrows():
    print(f"    {str(b):<16} n={int(r['n']):>4}  progressed {r['rate']:6.1%}  "
          f"median {r['med']/365.25:.2f} yr")

# ---------------------------------------------------------------- 3. pre-treatment labs
if os.path.exists("ov_labs.parquet"):
    labs = pd.read_parquet("ov_labs.parquet")
    print("\nloaded cached ov_labs.parquet")
else:
    fm = pd.read_csv("feature_map_final.csv")
    CID_SQL = ",".join(str(int(x)) for x in fm["concept_id"])
    cid2feat = dict(zip(fm["concept_id"].astype(int), fm["feature"]))
    cid2fac = dict(zip(fm["concept_id"].astype(int), fm["to_std_factor"].fillna(1.0)))
    ids = "','".join(E.index.astype(str))
    print("\npulling pre-treatment bloodwork ...")
    meas = C.query(f"""
      WITH ppl AS (
        SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, MIN(pe.person_id) AS person_id
        FROM {D}.DIM_PATIENT p
        JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                            = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
        WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}') GROUP BY 1)
      SELECT ppl.clinic, DATE(m.measurement_date) AS d,
             m.measurement_concept_id AS cid, m.value_as_number AS value
      FROM ppl JOIN {D}.measurement m ON m.person_id = ppl.person_id
      WHERE m.measurement_concept_id IN ({CID_SQL}) AND m.value_as_number IS NOT NULL
    """).to_dataframe()
    meas["feature"] = meas["cid"].map(cid2feat)
    meas["value"] = meas["value"] * meas["cid"].map(cid2fac)
    meas["d"] = pd.to_datetime(meas["d"])
    tx = cv.groupby("clinic")["tx_date"].first()
    meas = meas.join(tx.rename("tx"), on="clinic")
    pre_l = meas[(meas["d"] <= meas["tx"]) & (meas["d"] >= meas["tx"] - pd.Timedelta(days=365))]
    pre_l = pre_l.dropna(subset=["feature"])
    g = pre_l.groupby(["clinic", "feature"])["value"].agg(["mean", "min", "max", "last"])
    labs = g.unstack()
    labs.columns = [f"{f}__{s}" for s, f in labs.columns]
    labs = labs[[c for c in labs.columns if "height" not in c.lower()]]   # mismapped
    labs.to_parquet("ov_labs.parquet")
print(f"  lab panel: {labs.shape[1]} columns, {labs.shape[0]:,} women "
      f"({labs.shape[0]/len(E):.0%} of cohort)")

# ---------------------------------------------------------------- 4. models
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

def prep(df):
    X = df.select_dtypes(include=[np.number]).reindex(E.index)
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def run(X, name, model="gbm"):
    y, t = E["prog"], E["time"]
    s, oof = [], pd.Series(index=X.index, dtype=float)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        if model == "lr":
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
            sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
            m = LogisticRegression(class_weight="balanced", max_iter=4000, C=0.5)
            m.fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
            p = m.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]
        else:
            m = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                random_state=0).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
        oof.iloc[te] = p
        s.append(cindex(t.iloc[te], y.iloc[te], p))
    a = np.array(s)
    print(f"  {name:<44}C-index {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} feat)")
    return a.mean(), oof

B = pd.DataFrame({"baseline": E["baseline"], "log_baseline": np.log1p(E["baseline"])})
LB = prep(labs)
BOTH = prep(B.join(LB, how="left"))

print("\n" + "=" * 74)
print(f"WHO STOPS RESPONDING TO PLATINUM?  n={len(E):,}, {int(E['prog'].sum()):,} events")
print("=" * 74)
print("  logistic regression:")
l1, _ = run(prep(B), "CA-125 at treatment start only", "lr")
l2, _ = run(LB, "routine bloodwork only (no CA-125)", "lr")
l3, _ = run(BOTH, "CA-125 + bloodwork", "lr")
print("  gradient boosting:")
g1, _ = run(prep(B), "CA-125 at treatment start only")
g2, _ = run(LB, "routine bloodwork only (no CA-125)")
g3, oof = run(BOTH, "CA-125 + bloodwork")

print(f"\n  PROSTATE, for comparison:   marker 0.637 | bloodwork 0.725 | both 0.756")
print(f"  OVARIAN:                    marker {g1:.3f} | bloodwork {g2:.3f} | both {g3:.3f}")
print(f"  -> bloodwork beats the standard marker by {g2-g1:+.3f} "
      f"(prostate: +0.088)")

# ---------------------------------------------------------------- 5. chart
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def km(t, e):
        d = pd.DataFrame({"t": t, "e": e}).sort_values("t")
        ts, ss, s, n = [0.0], [1.0], 1.0, len(d)
        for tt, gg in d.groupby("t"):
            if n <= 0:
                break
            k = int(gg["e"].sum())
            if k:
                s *= (1 - k / n); ts.append(tt); ss.append(s)
            n -= len(gg)
        return ts, ss
    grp = pd.qcut(oof, 3, labels=["Low risk", "Medium risk", "High risk"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gname, col in zip(["Low risk", "Medium risk", "High risk"],
                          ["#1f9e5a", "#e8a33d", "#c0392b"]):
        sel = grp == gname
        t, s = km(E.loc[sel, "time"].values / 365.25, E.loc[sel, "prog"].values)
        ax.step(t, s, where="post", lw=2.5, color=col, label=f"{gname} (n={int(sel.sum()):,})")
    ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    ax.set_xlabel("Years since starting platinum chemotherapy")
    ax.set_ylabel("Still responding")
    ax.set_title("Predicted risk separates platinum response duration", fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("ovarian_km.png", dpi=200, bbox_inches="tight")
    print("\nwrote ovarian_km.png")
except Exception as e:
    print(f"\n(chart skipped: {type(e).__name__} {str(e)[:80]})")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"ovarian_study | n={len(E)} | events={int(E['prog'].sum())} ({E['prog'].mean():.0%}) "
      f"| gbm_marker={g1:.3f} | gbm_blood={g2:.3f} | gbm_both={g3:.3f} "
      f"| lr_both={l3:.3f} | med_yrs={E.loc[E['prog']==1,'time'].median()/365.25:.2f}")

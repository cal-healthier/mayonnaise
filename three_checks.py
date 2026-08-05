"""
three_checks.py

1. AUDIT THE BLOOD TESTS. The feature map was built by automatic name matching.
   `height` is provably wrong (values ~0.04). Check every feature's median
   against the range that test should plausibly read. Runs FIRST because if the
   inputs are wrong, nothing downstream means anything.
2. IMMUNOTHERAPY SURVIVAL. Bloodwork did not predict thyroid side effects.
   Does it predict who actually lives longer? 3,198 deaths, data already cached.
3. TRAIN EARLY, TEST LATE. Random cross-validation flatters a model when
   practice changes over time. Real use is predicting forward.

Cached files only. No BigQuery, no cost.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

pd.set_option("display.width", 220)

# ============================================================ 1. AUDIT
print("=" * 78)
print("1. ARE WE USING THE BLOOD TESTS WE THINK WE ARE?")
print("=" * 78)
fm = pd.read_csv("feature_map_final.csv")
cc = pd.read_parquet("measurement_concepts.parquet")[["cid", "name", "p50", "unit", "n_persons"]]
A = fm.merge(cc, left_on="concept_id", right_on="cid", how="left")
A["adj_median"] = A["p50"] * A["to_std_factor"].fillna(1.0)

# what each test should plausibly read (substring -> low, high, units)
EXPECT = [
    ("alp",      30,   170,  "U/L"),      ("alk",   30,   170,  "U/L"),
    ("alt",      5,    70,   "U/L"),      ("ast",   5,    70,   "U/L"),
    ("alb",      2.5,  5.5,  "g/dL"),     ("bili",  0.1,  2.0,  "mg/dL"),
    ("creat",    0.4,  2.0,  "mg/dL"),    ("bun",   5,    30,   "mg/dL"),
    ("gluc",     60,   200,  "mg/dL"),    ("prot",  5.5,  9.0,  "g/dL"),
    ("sodium",   125,  150,  "mmol/L"),   ("_na",   125,  150,  "mmol/L"),
    ("potas",    3.0,  5.8,  "mmol/L"),   ("chlor", 92,   115,  "mmol/L"),
    ("calc",     7.5,  11.5, "mg/dL"),    ("co2",   18,   34,   "mmol/L"),
    ("ldh",      90,   350,  "U/L"),      ("magnes",1.4,  2.6,  "mg/dL"),
    ("hgb",      9,    18,   "g/dL"),     ("hemog", 9,    18,   "g/dL"),
    ("hct",      28,   52,   "%"),        ("hemat", 28,   52,   "%"),
    ("rbc",      3.0,  6.2,  "M/uL"),     ("wbc",   2.5,  14,   "K/uL"),
    ("plt",      100,  450,  "K/uL"),     ("platel",100,  450,  "K/uL"),
    ("mcv",      75,   102,  "fL"),       ("mchc",  29,   38,   "g/dL"),
    ("mch",      24,   36,   "pg"),       ("rdw",   11,   18,   "%"),
    ("mpv",      6,    13,   "fL"),       ("_perc", 0,    100,  "%"),
    ("neut",     0,    100,  "% or K/uL"),("lymph", 0,    100,  "% or K/uL"),
    ("mono",     0,    100,  "% or K/uL"),("eos",   0,    100,  "% or K/uL"),
    ("baso",     0,    100,  "% or K/uL"),
    ("height",   140,  200,  "cm (or 1.4-2.0 m)"),
    ("weight",   40,   160,  "kg"),       ("bmi",   15,   45,   "kg/m2"),
    ("pulse",    50,   110,  "bpm"),      ("heart", 50,   110,  "bpm"),
    ("breath",   8,    28,   "/min"),     ("resp",  8,    28,   "/min"),
    ("temp",     35,   39,   "C"),        ("sat",   85,   100,  "%"),
    ("systol",   85,   185,  "mmHg"),     ("diastol",45,  110,  "mmHg"),
    ("inr",      0.8,  3.0,  "ratio"),    ("tsh",   0.3,  6.0,  "mIU/L"),
]
def expected_for(feat):
    f = str(feat).lower()
    for key, lo, hi, u in EXPECT:
        if key in f:
            return lo, hi, u
    return None, None, None

bad, unknown = [], []
print(f"  {'feature':<22}{'raw median':>11}{'x factor':>10}{'adjusted':>11}   "
      f"{'expected':<22}verdict")
print("  " + "-" * 92)
for _, r in A.sort_values("feature").iterrows():
    lo, hi, u = expected_for(r["feature"])
    adj = r["adj_median"]
    if lo is None:
        verdict, note = "?", "no rule - eyeball it"
        unknown.append(r["feature"])
    elif pd.isna(adj):
        verdict, note = "!", "NO MEDIAN - concept may be non-numeric"
        bad.append(r["feature"])
    # allow height in metres as well as cm
    elif "height" in str(r["feature"]).lower() and 1.3 <= adj <= 2.1:
        verdict, note = "ok", f"{lo}-{hi} {u}"
    elif lo <= adj <= hi:
        verdict, note = "ok", f"{lo}-{hi} {u}"
    else:
        verdict, note = "**BAD**", f"{lo}-{hi} {u}"
        bad.append(r["feature"])
    print(f"  {str(r['feature']):<22}{r['p50']:>11.3g}{r['to_std_factor']:>10.4g}"
          f"{adj:>11.3g}   {note:<22}{verdict}")

print(f"\n  {len(A)} features checked | {len(bad)} implausible | {len(unknown)} no rule")
if bad:
    print(f"  IMPLAUSIBLE: {', '.join(map(str, bad))}")
    print("  -> exclude these, or re-map them, before trusting any result that used them.")

# ============================================================ shared helpers
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
        hi_ = np.where(ei[ok], ri, rj); lo_ = np.where(ei[ok], rj, ri)
        conc += np.sum(hi_ > lo_) + 0.5 * np.sum(hi_ == lo_); perm += ok.sum()
    return conc / perm

def gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)

def cv(X, y, t, name):
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = gbm().fit(X.iloc[tr], y.iloc[tr])
        s.append(cindex(t.iloc[te], y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(s)
    print(f"  {name:<48}C-index {a.mean():.3f} ± {a.std():.3f}  ({X.shape[1]} feat)")
    return a.mean()

def clean(df, idx, drop_bad=True):
    X = df.select_dtypes(include=[np.number]).reindex(idx)
    if drop_bad:
        X = X[[c for c in X.columns
               if not any(str(b).lower() in c.lower() for b in bad)]]
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

# ============================================================ 2. IMMUNOTHERAPY SURVIVAL
print("\n" + "=" * 78)
print("2. DOES BLOODWORK PREDICT WHO LIVES LONGER ON IMMUNOTHERAPY?")
print("=" * 78)
if os.path.exists("ici_thyroid_label.parquet") and os.path.exists("ici_labs.parquet"):
    I = pd.read_parquet("ici_thyroid_label.parquet")
    IL = pd.read_parquet("ici_labs.parquet")
    I = I[I["os_days"] > 0]
    print(f"  {len(I):,} patients, {int(I['died'].sum()):,} deaths "
          f"({I['died'].mean():.0%}), median follow-up "
          f"{I['os_days'].median()/365.25:.2f} yr")
    site = pd.get_dummies(I["site3"].astype(str), prefix="site").astype(float)
    y, t = I["died"], I["os_days"]
    c_site = cv(clean(site, I.index), y, t, "cancer type only")
    c_blood = cv(clean(IL, I.index), y, t, "routine bloodwork only")
    c_all = cv(clean(site.join(IL, how="left"), I.index), y, t, "cancer type + bloodwork")
    print(f"\n  bloodwork adds {c_all - c_site:+.3f} over knowing the cancer type alone")
    print(f"  (thyroid side-effect model was 0.528 — chance. Survival is a different"
          f" question.)")
else:
    c_site = c_blood = c_all = float("nan")
    print("  cached immunotherapy files not found — run checkpoint_tki.py first")

# ============================================================ 3. TRAIN EARLY, TEST LATE
print("\n" + "=" * 78)
print("3. TRAIN ON OLDER PATIENTS, TEST ON NEWER ONES  (prostate study)")
print("=" * 78)
P = pd.read_parquet("psa_progression.parquet")
PL = pd.read_parquet("psa_labs.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
P["year"] = pd.to_datetime(tx.reindex(P.index)).dt.year
P = P[P["year"].notna()]
print("  patients per treatment year:")
vc = P["year"].value_counts().sort_index()
for yr, n in vc.items():
    print(f"    {int(yr)}  {'#' * max(1, int(n / max(vc.max(), 1) * 46))} {n:,}")

cut = int(P["year"].quantile(0.70))
tr_i = P.index[P["year"] <= cut]
te_i = P.index[P["year"] > cut]
print(f"\n  split at {cut}: train {len(tr_i):,} (<= {cut})  "
       f"test {len(te_i):,} (> {cut})")
print(f"  events: train {int(P.loc[tr_i,'prog'].sum()):,}  "
      f"test {int(P.loc[te_i,'prog'].sum()):,}")

B = pd.DataFrame({"baseline": P["baseline"], "log_baseline": np.log1p(P["baseline"])})
X = clean(B.join(PL, how="left"), P.index)
rand = cv(X, P["prog"], P["time"], "random split (what we reported: ~0.756)")
m = gbm().fit(X.loc[tr_i], P.loc[tr_i, "prog"])
p = m.predict_proba(X.loc[te_i])[:, 1]
fwd = cindex(P.loc[te_i, "time"], P.loc[te_i, "prog"], p)
print(f"  {'train on the past, predict the future':<48}C-index {fwd:.3f}")
print(f"\n  drop from random to forward-in-time: {fwd - rand:+.3f}")
print("  (a small drop is normal and expected; a large one means the model was")
print("   partly learning how medicine was practised in a particular era)")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"three_checks | bad_feats={len(bad)} | ici_site={c_site:.3f} "
      f"| ici_blood={c_blood:.3f} | ici_all={c_all:.3f} "
      f"| prost_random={rand:.3f} | prost_forward={fwd:.3f} | cut={cut}")

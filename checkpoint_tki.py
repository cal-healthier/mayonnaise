"""
checkpoint_tki.py -- strip the tyrosine-kinase-inhibitor confound, then model.

The label passed its CTLA-4 positive control (1.42x) but irAE rate varies
11%-40% across cancers, and the two highest are kidney (36.7%) and endometrial
(39.9%) -- exactly where checkpoint inhibitors are routinely combined with
lenvatinib / axitinib / cabozantinib. Those TKIs cause hypothyroidism directly
in 60-80% of patients. That is drug toxicity, not immune-related thyroiditis.

1. Find concurrent TKI exposure.
2. Re-run the diagnostics on the TKI-free cohort: does site variation collapse?
   does the CTLA-4 control still pass?
3. Then model: pre-treatment bloodwork -> time to thyroid irAE.
   There is NO standard clinical predictor for this, so any signal is new.
4. Bonus: do patients who develop an irAE live longer? (irAEs are reported to
   correlate with response -- a cheap check that costs nothing extra.)
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
pd.set_option("display.width", 220)

E = pd.read_parquet("ici_thyroid_label.parquet")
print(f"label: {len(E):,} patients, {int(E['irae'].sum()):,} events "
      f"({E['irae'].mean():.1%})")

medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
# TKIs with well-documented direct hypothyroidism
TKI = ["LENVATINIB", "LENVIMA", "AXITINIB", "INLYTA", "CABOZANTINIB", "CABOMETYX",
       "COMETRIQ", "SUNITINIB", "SUTENT", "SORAFENIB", "NEXAVAR", "PAZOPANIB",
       "VOTRIENT", "VANDETANIB", "CAPRELSA", "REGORAFENIB", "STIVARGA"]
tki_like = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in TKI)

# ---------------------------------------------------------------- 1. who got a TKI?
if os.path.exists("ici_tki.parquet"):
    tki = pd.read_parquet("ici_tki.parquet")
else:
    ids = "','".join(E.index.astype(str))
    print("finding concurrent TKI exposure ...")
    tki = C.query(f"""
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             MIN(DATE(t.TREATMENT_DTM)) AS tki_first
      FROM {D}.DIM_PATIENT p
      JOIN {D}.FACT_TREATMENT_DETAIL t ON t.PATIENT_DK = p.PATIENT_DK
      JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
        AND ({tki_like}) AND t.TREATMENT_DTM IS NOT NULL
      GROUP BY 1
    """).to_dataframe()
    tki.to_parquet("ici_tki.parquet")
tki["tki_first"] = pd.to_datetime(tki["tki_first"])
tset = tki.set_index("clinic")["tki_first"]
E["tki_date"] = tset.reindex(E.index)
# concurrent = TKI started before or during the checkpoint course
E["on_tki"] = (E["tki_date"].notna() &
               (E["tki_date"] <= E["ici_date"] + pd.Timedelta(days=90))).astype(int)

print("\n" + "=" * 76)
print("IS THE THYROID SIGNAL ACTUALLY TKI TOXICITY?")
print("=" * 76)
for f, lbl in [(1, "on a thyroid-toxic TKI"), (0, "checkpoint inhibitor only")]:
    s = E[E["on_tki"] == f]
    if len(s):
        print(f"  {lbl:<30} n={len(s):>6,}   thyroid event {s['irae'].mean():7.1%}")
r_t = E.loc[E["on_tki"] == 1, "irae"].mean() if (E["on_tki"] == 1).any() else np.nan
r_n = E.loc[E["on_tki"] == 0, "irae"].mean()
print(f"\n  ratio {r_t/r_n:.2f}x  -> TKI users are {'MUCH ' if r_t/r_n > 1.4 else ''}"
      f"more likely; excluding them.")

CLEAN = E[E["on_tki"] == 0].copy()
print(f"\n  clean cohort (checkpoint only): {len(CLEAN):,}, "
      f"{int(CLEAN['irae'].sum()):,} events ({CLEAN['irae'].mean():.1%})")

print("\n  irAE rate by site, BEFORE vs AFTER excluding TKI users:")
for s in E["site3"].value_counts().head(8).index:
    a = E[E["site3"] == s]; b = CLEAN[CLEAN["site3"] == s]
    print(f"    {s}  before {a['irae'].mean():6.1%} (n={len(a):>5,})   "
          f"after {b['irae'].mean() if len(b) else float('nan'):6.1%} (n={len(b):>5,})")
sp_b = E.groupby("site3")["irae"].mean()
sp_a = CLEAN.groupby("site3")["irae"].mean()
big = E["site3"].value_counts().head(8).index
print(f"\n  spread across the top 8 sites: before "
      f"{sp_b[big].max()-sp_b[big].min():.1%}, after "
      f"{sp_a.reindex(big).max()-sp_a.reindex(big).min():.1%}  (smaller = confound removed)")

ipi = CLEAN.loc[CLEAN["got_ctla4"] == 1, "irae"].mean()
pd1 = CLEAN.loc[CLEAN["got_ctla4"] == 0, "irae"].mean()
print(f"\n  CTLA-4 positive control on the clean cohort: "
      f"{ipi:.1%} vs {pd1:.1%} = {ipi/pd1:.2f}x  "
      f"({'still validated' if ipi/pd1 >= 1.15 else 'WEAKENED - tell Claude'})")

# ---------------------------------------------------------------- 2. bloodwork
if os.path.exists("ici_labs.parquet"):
    labs = pd.read_parquet("ici_labs.parquet")
else:
    fm = pd.read_csv("feature_map_final.csv")
    CID_SQL = ",".join(str(int(x)) for x in fm["concept_id"])
    c2f = dict(zip(fm["concept_id"].astype(int), fm["feature"]))
    c2x = dict(zip(fm["concept_id"].astype(int), fm["to_std_factor"].fillna(1.0)))
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
    meas["feature"] = meas["cid"].map(c2f)
    meas["value"] = meas["value"] * meas["cid"].map(c2x)
    meas["d"] = pd.to_datetime(meas["d"])
    meas = meas.join(E["ici_date"].rename("ici"), on="clinic")
    pre = meas[(meas["d"] <= meas["ici"]) & (meas["d"] >= meas["ici"] - pd.Timedelta(days=365))]
    pre = pre.dropna(subset=["feature"])
    g = pre.groupby(["clinic", "feature"])["value"].agg(["mean", "min", "max", "last"])
    labs = g.unstack()
    labs.columns = [f"{f}__{s}" for s, f in labs.columns]
    labs = labs[[c for c in labs.columns if "height" not in c.lower()]]
    labs.to_parquet("ici_labs.parquet")
print(f"  lab panel: {labs.shape[1]} columns, {labs.shape[0]:,} patients")

# ---------------------------------------------------------------- 3. model
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

Z = CLEAN
def prep(df):
    X = df.select_dtypes(include=[np.number]).reindex(Z.index)
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def run(X, name, model="gbm"):
    y, t = Z["irae"], Z["time"]
    s = []
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
        s.append(cindex(t.iloc[te], y.iloc[te], p))
    a = np.array(s)
    print(f"  {name:<44}C-index {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} feat)")
    return a.mean()

BASE = pd.DataFrame({"base_tsh": Z["base_tsh"], "ctla4": Z["got_ctla4"]})
LB = prep(labs)
print("\n" + "=" * 76)
print(f"WHO DEVELOPS IMMUNE THYROID TOXICITY?  n={len(Z):,}, "
      f"{int(Z['irae'].sum()):,} events")
print("  (no standard clinical predictor exists for this -- any signal is new)")
print("=" * 76)
print("  logistic regression:")
run(prep(BASE), "baseline TSH + drug class only", "lr")
run(LB, "routine bloodwork only", "lr")
run(prep(BASE.join(LB, how="left")), "everything", "lr")
print("  gradient boosting:")
g0 = run(prep(BASE), "baseline TSH + drug class only")
g1 = run(LB, "routine bloodwork only")
g2 = run(prep(BASE.join(LB, how="left")), "everything")

# ---------------------------------------------------------------- 4. does irAE track response?
print("\n" + "=" * 76)
print("DO PATIENTS WHO DEVELOP THYROID TOXICITY LIVE LONGER?")
print("=" * 76)
for f, lbl in [(1, "developed thyroid irAE"), (0, "no thyroid irAE")]:
    s = Z[Z["irae"] == f]
    print(f"  {lbl:<26} n={len(s):>6,}  died {s['died'].mean():6.1%}  "
          f"median OS {s['os_days'].median()/365.25:.2f} yr")
print("  (NB: immortal-time bias -- you must survive long enough to develop one.")
print("   Suggestive only; a landmark analysis is needed to claim anything.)")

print("\n" + "-" * 72)
print("FINAL LINE:")
print(f"checkpoint_tki | clean_n={len(Z)} | events={int(Z['irae'].sum())} "
      f"({Z['irae'].mean():.0%}) | tki_ratio={r_t/r_n:.2f} | ctla4_clean={ipi/pd1:.2f} "
      f"| gbm_base={g0:.3f} | gbm_blood={g1:.3f} | gbm_all={g2:.3f}")

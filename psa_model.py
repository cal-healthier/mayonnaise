"""
psa_model.py -- Oncoformer Fig 6F, step 2.

(a) finish the label diagnostics that crashed (pct_change collided with the
    pandas method of the same name -- bracket notation everywhere now),
(b) check whether "non-response" is real or just lab noise at low baseline PSA,
(c) pull pre-treatment bloodwork for these men,
(d) fit plain regression, nested:  baseline PSA -> + tumor -> + all labs.

The question: does routine bloodwork say anything about whether hormone therapy
will work, BEYOND the PSA number the oncologist already has in front of them?
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA.COLUMNS"
pd.set_option("display.width", 200)

L = pd.read_parquet("psa_label.parquet")
L = L.rename(columns={"pct_change": "drop_pct"})
print(f"labeled patients: {len(L):,}   responders {L['responder'].mean():.1%}")
print(f"  median drop {L['drop_pct'].median():.0f}%  "
      f"median days to nadir {L['days_to_nadir'].median():.0f}  "
      f"median follow-up PSAs {L['n_post'].median():.0f}")
print("\n  spread of PSA change:")
for q in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    p{q:<3} {L['drop_pct'].quantile(q/100):>8.1f}%")

# ---------------------------------------------------------------- is non-response real?
print("\n" + "=" * 72)
print("IS 'NON-RESPONSE' REAL, OR LAB NOISE AT LOW BASELINE PSA?")
print("=" * 72)
L["bl_bin"] = pd.cut(L["baseline"], [0, 0.5, 1, 2, 4, 10, 20, 100, 1e9],
                     labels=["<0.5", "0.5-1", "1-2", "2-4", "4-10", "10-20", "20-100", ">100"])
tab = L.groupby("bl_bin", observed=True).agg(
    n=("responder", "size"), resp_rate=("responder", "mean"),
    med_nadir=("nadir", "median"))
for b, r in tab.iterrows():
    print(f"  baseline {str(b):<8} n={int(r['n']):>5}  respond {r['resp_rate']:6.1%}  "
          f"median nadir {r['med_nadir']:.2f}")
low = L[L["baseline"] < 2]
print(f"\n  men starting at PSA < 2: {len(low):,} ({len(low)/len(L):.0%} of cohort), "
      f"{1-low['responder'].mean():.0%} of them 'non-responders'")
print("  -> if non-response concentrates at low baseline, the label is partly noise;")
print("     restricting to baseline >= 2 is the fix.")

CLEAN = L[L["baseline"] >= 2].copy()
print(f"\n  clean subset (baseline PSA >= 2): {len(CLEAN):,} men, "
      f"{CLEAN['responder'].mean():.1%} respond, "
      f"{int((1-CLEAN['responder']).sum()):,} non-responders")

# ---------------------------------------------------------------- tumor features
tcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S} WHERE table_name='FACT_CANCER_DATA_REPOSITORY'")]
want = [c for c in tcols if any(k in c.upper() for k in
        ("GLEASON", "GRADE", "STAGE_GROUP", "DERIVED_", "TNM", "REGIONAL_NODES",
         "AGE_AT", "BIRTH"))][:14]
print(f"\nprostate tumor columns used: {want}")
sel = ", ".join(f"ANY_VALUE(CAST({c} AS STRING)) AS {c}" for c in want)
tum = C.query(f"""
  SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, {sel}
  FROM {D}.FACT_CANCER_DATA_REPOSITORY r
  JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
  WHERE CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C61%'
  GROUP BY 1
""").to_dataframe().set_index("clinic")
print(f"  tumor rows: {len(tum):,}")

# ---------------------------------------------------------------- pre-treatment labs
fm = pd.read_csv("feature_map_final.csv")
CID_SQL = ",".join(str(int(x)) for x in fm.concept_id)
cid2feat = dict(zip(fm.concept_id.astype(int), fm.feature))
cid2fac = dict(zip(fm.concept_id.astype(int), fm.to_std_factor.fillna(1.0)))

pv = pd.read_parquet("psa_values.parquet")
tx = pv.groupby("clinic")["tx_date"].first()
ids = "','".join(L.index.astype(str))
print("\npulling pre-treatment bloodwork ...")
meas = C.query(f"""
  WITH ppl AS (
    SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, MIN(pe.person_id) AS person_id
    FROM {D}.DIM_PATIENT p
    JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                        = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
    WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
    GROUP BY 1)
  SELECT ppl.clinic, DATE(m.measurement_date) AS d,
         m.measurement_concept_id AS cid, m.value_as_number AS value
  FROM ppl JOIN {D}.measurement m ON m.person_id = ppl.person_id
  WHERE m.measurement_concept_id IN ({CID_SQL}) AND m.value_as_number IS NOT NULL
""").to_dataframe()
meas["feature"] = meas["cid"].map(cid2feat)
meas["value"] = meas["value"] * meas["cid"].map(cid2fac)
meas["d"] = pd.to_datetime(meas["d"])
meas = meas.join(tx.rename("tx"), on="clinic")
pre = meas[(meas["d"] <= meas["tx"]) & (meas["d"] >= meas["tx"] - pd.Timedelta(days=365))]
pre = pre.dropna(subset=["feature"])
print(f"  lab rows before treatment: {len(pre):,} | "
      f"{pre['clinic'].nunique():,} men ({pre['clinic'].nunique()/len(L):.0%})")

g = pre.groupby(["clinic", "feature"])["value"]
labs = g.agg(["mean", "min", "max", "last"]).unstack()
labs.columns = [f"{f}__{s}" for s, f in labs.columns]
print(f"  lab feature columns: {labs.shape[1]}")

# ---------------------------------------------------------------- models
def prep(df):
    X = df.select_dtypes(include=[np.number])
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def fit_binary(X, y, name):
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        m = LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)
        m.fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
        p = m.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    a = np.array(aucs)
    print(f"  {name:<44}AUROC {a.mean():.3f} ± {a.std():.3f}")
    return a.mean()

def fit_cont(X, y, name):
    r2 = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        m = LinearRegression().fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
        r2.append(r2_score(y.iloc[te], m.predict(sc.transform(imp.transform(X.iloc[te])))))
    a = np.array(r2)
    print(f"  {name:<44}R²    {a.mean():.3f} ± {a.std():.3f}")
    return a.mean()

TUM = tum.apply(pd.to_numeric, errors="coerce")
base = CLEAN[["baseline"]].copy()
base["log_baseline"] = np.log1p(CLEAN["baseline"])
sets = {
    "PSA at treatment start only":        base,
    "+ tumor (grade/stage/nodes)":        base.join(prep(TUM), how="left"),
    "+ all routine bloodwork":            base.join(prep(TUM), how="left").join(labs, how="left"),
    "bloodwork only (no PSA, no tumor)":  labs.reindex(CLEAN.index),
}
y_bin = CLEAN["responder"]
y_con = np.log1p(CLEAN["nadir"])

print("\n" + "=" * 72)
print(f"DOES THE TREATMENT WORK?  (yes/no, n={len(CLEAN):,}, "
      f"{y_bin.mean():.0%} respond)")
print("=" * 72)
res_b = {k: fit_binary(prep(v).reindex(CLEAN.index), y_bin, k) for k, v in sets.items()}

print("\n" + "=" * 72)
print("HOW FAR DOES PSA FALL?  (continuous - closest to Oncoformer Fig 6F)")
print("=" * 72)
res_c = {k: fit_cont(prep(v).reindex(CLEAN.index), y_con, k) for k, v in sets.items()}

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"psa_model | clean_n={len(CLEAN)} | resp={y_bin.mean():.0%} "
      f"| auc_psa={res_b['PSA at treatment start only']:.3f} "
      f"| auc_all={res_b['+ all routine bloodwork']:.3f} "
      f"| r2_psa={res_c['PSA at treatment start only']:.3f} "
      f"| r2_all={res_c['+ all routine bloodwork']:.3f}")

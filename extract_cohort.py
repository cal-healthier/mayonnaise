"""
extract_cohort.py  -- the real feature extraction (v1, one cancer, capped).

Builds the training cube on the locked lane:
  person key   = PATIENT_CLINIC_NUMBER  (dedups PATIENT_DK, bridges to OMOP)
  lab values   = OMOP measurement.value_as_number  (numeric, our 25 concepts)
  cancer/tx/outcome = FACT_ tables (PATIENT_DK)

Writes:
  cohort_labels.parquet   -- one row per patient: clinic, dx, tx, label, regimen
  cohort_features.parquet -- long: clinic, visit_date, feature, value  (the cube)

Needs feature_map_final.csv (from finalize_features.py) in the working dir.
Prints tensor shapes + one example patient. Read patient detail inside only.
"""
import os
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

SITE_PREFIX = "C50"     # breast; change to probe another cancer
MAX_PATIENTS = 500      # cap for a fast, inspectable first run

# ---- feature map (concept_id -> feature label + unit conversion) ----
if not os.path.exists("feature_map_final.csv"):
    raise SystemExit("feature_map_final.csv missing -- run finalize_features.py first")
fm = pd.read_csv("feature_map_final.csv")
CIDS = [int(x) for x in fm["concept_id"].tolist()]
cid2feat = dict(zip(fm.concept_id.astype(int), fm.feature))
cid2fac = dict(zip(fm.concept_id.astype(int), fm.to_std_factor.fillna(1.0)))
print(f"features: {len(CIDS)} concepts from feature_map_final.csv")

# ---- 1. cohort: one row per person, with index date and label ----
print(f"\nbuilding cohort: site {SITE_PREFIX}*, single primary, treated, labelled ...")
cohort = C.query(f"""
WITH reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(SITE_PRIMARY_ICD_O_3 AS STRING)) AS site,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_contact,
         MAX(COALESCE(
             SAFE.PARSE_DATE('%Y-%m-%d', TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING))),
             SAFE.PARSE_DATE('%m/%d/%Y', TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING))),
             SAFE_CAST(TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING)) AS DATE))) AS recur
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL GROUP BY 1
),
tx AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE UPPER(CAST(m.MED_THERAPEUTIC_CLASS_DESCRIPTION AS STRING))='ANTINEOPLASTICS'
    AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1
),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id FROM {D}.person GROUP BY 1),
joined AS (
  SELECT b.clinic, pers.person_id, r.dx, tx.tx_date, r.site,
    CASE
      WHEN r.recur IS NOT NULL AND r.recur > tx.tx_date THEN 1
      WHEN (r.recur IS NULL OR r.recur <= tx.tx_date)
           AND DATE_DIFF(r.last_contact, tx.tx_date, DAY) >= 730 THEN 0
      ELSE NULL END AS label
  FROM reg r
  JOIN tx USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK)
  JOIN pers ON pers.clinic = b.clinic
  WHERE r.n_prim = 1 AND r.site LIKE '{SITE_PREFIX}%' AND tx.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY tx.tx_date) = 1
)
SELECT * FROM joined WHERE label IS NOT NULL
ORDER BY clinic LIMIT {MAX_PATIENTS}
""").to_dataframe()

print(f"  cohort: {len(cohort)} patients  |  recurred={int(cohort.label.sum())}  "
      f"clear={int((cohort.label==0).sum())}")
cohort.to_parquet("cohort_labels.parquet")

# ---- 2. regimen per patient (normalized ingredient, for the action channel) ----
pids = ",".join(str(int(x)) for x in cohort.person_id.tolist())
clinics = "','".join(cohort.clinic.tolist())
reg = C.query(f"""
SELECT CAST(dp.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
       LOWER(REGEXP_EXTRACT(UPPER(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING)),
             r'^([A-Z0-9][A-Z0-9\\-]{{2,}})')) AS drug
FROM {D}.FACT_TREATMENT_DETAIL t
JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
JOIN {D}.DIM_PATIENT dp USING (PATIENT_DK)
WHERE CAST(dp.PATIENT_CLINIC_NUMBER AS STRING) IN ('{clinics}')
  AND UPPER(CAST(m.MED_THERAPEUTIC_CLASS_DESCRIPTION AS STRING))='ANTINEOPLASTICS'
""").to_dataframe()
regimen = (reg.dropna().groupby("clinic")["drug"]
           .agg(lambda s: "+".join(sorted(set(s))[:4])).rename("regimen"))

# ---- 3. THE CUBE: pre-treatment lab/vital values from OMOP measurement ----
print("\npulling measurement values (the one real-cost query) ...")
meas = C.query(f"""
SELECT m.person_id, DATE(m.measurement_date) AS visit_date,
       m.measurement_concept_id AS cid, m.value_as_number AS value
FROM {D}.measurement m
WHERE m.person_id IN ({pids})
  AND m.measurement_concept_id IN ({",".join(str(c) for c in CIDS)})
  AND m.value_as_number IS NOT NULL
""").to_dataframe()

# map person_id -> clinic + tx_date, keep only pre-treatment, apply unit factor
pmap = cohort.set_index("person_id")[["clinic", "tx_date"]]
meas = meas.join(pmap, on="person_id")
meas["visit_date"] = pd.to_datetime(meas["visit_date"]).dt.date
meas = meas[meas["visit_date"] < meas["tx_date"]]
meas["feature"] = meas["cid"].map(cid2feat)
meas["value"] = meas["value"] * meas["cid"].map(cid2fac)
cube = meas[["clinic", "visit_date", "feature", "value"]].dropna(subset=["feature"])
cube.to_parquet("cohort_features.parquet")

# ---- 4. shapes + coverage + one example ----
vis = cube.groupby("clinic")["visit_date"].nunique()
cov = cube.groupby("feature")["clinic"].nunique().sort_values(ascending=False)
n_with_labs = cube["clinic"].nunique()

print("\n" + "=" * 72); print("THE CUBE"); print("=" * 72)
print(f"  patients with any pre-tx labs : {n_with_labs} / {len(cohort)}")
print(f"  pre-tx visits per patient     : median {vis.median():.0f}, "
      f"p90 {vis.quantile(.9):.0f}, max {vis.max():.0f}")
print(f"  long-format rows (the parquet): {len(cube):,}")
print("\n  feature coverage (patients with >=1 measurement):")
for f, n in cov.items():
    print(f"     {f:<16} {n:>4} ({n/len(cohort):.0%})")

# one example patient's [visit x feature] matrix (patient-level; inside only)
ex = cube["clinic"].value_counts().index[0]
grid = (cube[cube.clinic == ex]
        .pivot_table(index="visit_date", columns="feature", values="value", aggfunc="last"))
print("\n  EXAMPLE patient's [visit x feature] matrix (this is one training example):")
print(f"  shape = [{grid.shape[0]} visits x {grid.shape[1]} features], blanks = missing\n")
print(grid.round(1).to_string())

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"extract | patients={len(cohort)} | recurred={int(cohort.label.sum())} "
      f"| clear={int((cohort.label==0).sum())} | with_labs={n_with_labs} "
      f"| median_visits={vis.median():.0f} | features={cube['feature'].nunique()}/{len(CIDS)}")

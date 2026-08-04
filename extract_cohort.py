"""
extract_cohort.py -- PAN-CANCER feature extraction for survival modeling.

All single-primary, antineoplastic-treated cancer patients (capped), on the
locked lane: clinic-number person key, OMOP measurement for lab values.
Outputs the labs cube + labels (clinic, dx, tx_date, cancer site). The survival
label and tumor features are added in the analysis cell.
"""
import os
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200)

MAX_PATIENTS = 30000   # well-powered sample across cancers; raise to scale further

if not os.path.exists("feature_map_final.csv"):
    raise SystemExit("feature_map_final.csv missing -- run finalize_features.py first")
fm = pd.read_csv("feature_map_final.csv")
CIDS = [int(x) for x in fm.concept_id.tolist()]
cid2feat = dict(zip(fm.concept_id.astype(int), fm.feature))
cid2fac = dict(zip(fm.concept_id.astype(int), fm.to_std_factor.fillna(1.0)))
CID_SQL = ",".join(str(c) for c in CIDS)

COHORT_CTES = f"""
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(SITE_PRIMARY_ICD_O_3 AS STRING)) AS site
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
cohort AS (
  SELECT b.clinic, pers.person_id, r.dx, tx.tx_date, r.site
  FROM reg r
  JOIN tx USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK)
  JOIN pers ON pers.clinic = b.clinic
  WHERE r.n_prim = 1 AND r.site LIKE 'C%' AND tx.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY tx.tx_date) = 1
  ORDER BY b.clinic
  LIMIT {MAX_PATIENTS}
)
"""

print("building pan-cancer cohort ...")
cohort = C.query(f"WITH {COHORT_CTES} SELECT clinic, dx, tx_date, site FROM cohort").to_dataframe()
cohort.to_parquet("cohort_labels.parquet")
cohort["site3"] = cohort.site.str.extract(r"(C\d\d)")
print(f"  cohort: {len(cohort):,} patients across {cohort.site3.nunique()} cancer sites")
print("  top sites:")
print(cohort.site3.value_counts().head(12).to_string())

print("\npulling measurement values for the whole cohort (the cost query) ...")
meas = C.query(f"""
WITH {COHORT_CTES}
SELECT c.clinic, DATE(m.measurement_date) AS visit_date,
       m.measurement_concept_id AS cid, m.value_as_number AS value
FROM cohort c
JOIN {D}.measurement m ON m.person_id = c.person_id
WHERE m.measurement_concept_id IN ({CID_SQL})
  AND m.value_as_number IS NOT NULL
  AND DATE(m.measurement_date) < c.tx_date
""").to_dataframe()
meas["feature"] = meas.cid.map(cid2feat)
meas["value"] = meas.value * meas.cid.map(cid2fac)
cube = meas[["clinic", "visit_date", "feature", "value"]].dropna(subset=["feature"])
cube.to_parquet("cohort_features.parquet")

vis = cube.groupby("clinic")["visit_date"].nunique()
print(f"\n  cube: {len(cube):,} rows | {cube.clinic.nunique():,} patients with labs "
      f"({cube.clinic.nunique()/len(cohort):.0%}) | median {vis.median():.0f} visits")
print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"pancancer_extract | patients={len(cohort)} | sites={cohort.site3.nunique()} "
      f"| with_labs={cube.clinic.nunique()} | median_visits={vis.median():.0f} "
      f"| feat_rows={len(cube)}")

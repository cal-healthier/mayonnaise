"""
extract_cohort.py  -- the real feature extraction, SCALED (full breast cohort).

Lane: person key = PATIENT_CLINIC_NUMBER; lab values from OMOP measurement;
cancer/tx/outcome from FACT_ tables. Cohort join happens in SQL (no IN-lists),
so this scales to the whole cancer.

Writes cohort_labels.parquet (one row/patient) and cohort_features.parquet
(long cube: clinic, visit_date, feature, value). Needs feature_map_final.csv.
Read patient-level detail inside only.
"""
import os
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

SITE_PREFIX = "C50"     # breast

if not os.path.exists("feature_map_final.csv"):
    raise SystemExit("feature_map_final.csv missing -- run finalize_features.py first")
fm = pd.read_csv("feature_map_final.csv")
CIDS = [int(x) for x in fm["concept_id"].tolist()]
cid2feat = dict(zip(fm.concept_id.astype(int), fm.feature))
cid2fac = dict(zip(fm.concept_id.astype(int), fm.to_std_factor.fillna(1.0)))
CID_SQL = ",".join(str(c) for c in CIDS)
print(f"features: {len(CIDS)} concepts")

# reusable cohort CTEs -> named CTE `cohort` (clinic, person_id, dx, tx_date, label)
COHORT_CTES = f"""
reg AS (
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
cohort AS (
  SELECT * FROM (
    SELECT b.clinic, pers.person_id, r.dx, tx.tx_date,
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
  ) WHERE label IS NOT NULL
)
"""

# ---- 1. labels (cheap) ----
print("\nbuilding cohort labels ...")
cohort = C.query(f"WITH {COHORT_CTES} SELECT clinic, dx, tx_date, label FROM cohort").to_dataframe()
cohort.to_parquet("cohort_labels.parquet")
print(f"  cohort: {len(cohort):,} patients  |  recurred={int(cohort.label.sum())}  "
      f"clear={int((cohort.label==0).sum())}  ({cohort.label.mean():.1%} positive)")

# ---- 2. the cube: pre-tx measurement values (the one real-cost query) ----
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

meas["feature"] = meas["cid"].map(cid2feat)
meas["value"] = meas["value"] * meas["cid"].map(cid2fac)
cube = meas[["clinic", "visit_date", "feature", "value"]].dropna(subset=["feature"])
cube.to_parquet("cohort_features.parquet")

# ---- 3. shapes + coverage + one example ----
vis = cube.groupby("clinic")["visit_date"].nunique()
cov = cube.groupby("feature")["clinic"].nunique().sort_values(ascending=False)
n_with_labs = cube["clinic"].nunique()

print("\n" + "=" * 72); print("THE SCALED CUBE"); print("=" * 72)
print(f"  patients                      : {len(cohort):,}")
print(f"  with any pre-tx labs          : {n_with_labs:,} ({n_with_labs/len(cohort):.0%})")
print(f"  recurred / clear              : {int(cohort.label.sum())} / {int((cohort.label==0).sum())}")
print(f"  pre-tx visits per patient     : median {vis.median():.0f}, p90 {vis.quantile(.9):.0f}, max {vis.max():.0f}")
print(f"  long-format rows (parquet)    : {len(cube):,}")
print("\n  feature coverage:")
for f, n in cov.items():
    print(f"     {f:<16} {n:>6,} ({n/len(cohort):.0%})")

ex = cube["clinic"].value_counts().index[0]
grid = (cube[cube.clinic == ex]
        .pivot_table(index="visit_date", columns="feature", values="value", aggfunc="last"))
print(f"\n  EXAMPLE patient's [visit x feature] matrix "
      f"[{grid.shape[0]} x {grid.shape[1]}] (inside only):\n")
print(grid.round(1).to_string())

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"extract | patients={len(cohort)} | recurred={int(cohort.label.sum())} "
      f"| clear={int((cohort.label==0).sum())} | with_labs={n_with_labs} "
      f"| median_visits={vis.median():.0f} | features={cube['feature'].nunique()}/{len(CIDS)}")

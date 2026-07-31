"""
Find ONE well-chosen cancer patient and print their data end-to-end, so we can
see exactly what goes into the model and what we predict.

PRIVACY: this prints one patient's record. Read it inside the enclave only.
Do NOT paste the rows out. The FINAL LINE at the bottom is structural only.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)


def q(sql):
    return C.query(sql).to_dataframe()


# ---- find a lab-test name column in DIM_LAB_TEST (don't guess) ----
lab_cols = q(f"""SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS
                 WHERE table_name='DIM_LAB_TEST'""")["column_name"].tolist()
NAMECOL = next((c for c in lab_cols if "NAME" in c.upper() and "DK" not in c.upper()),
          next((c for c in lab_cols if "DESC" in c.upper()), lab_cols[1]))
print("using DIM_LAB_TEST." + NAMECOL, "for test names\n")

# ---- 1. FIND the patient (deterministic) ----
found = q(f"""
WITH tx AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
  FROM {D}.FACT_TREATMENT_DETAIL t
  JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE UPPER(CAST(m.MED_THERAPEUTIC_CLASS_DESCRIPTION AS STRING)) = 'ANTINEOPLASTICS'
    AND t.TREATMENT_DTM IS NOT NULL
  GROUP BY 1
),
labs AS (
  SELECT tx.PATIENT_DK, tx.tx_date,
         COUNT(DISTINCT DATE(l.LAB_COLLECTION_DTM)) AS pre_lab_days
  FROM tx JOIN {D}.FACT_LAB_TEST l
    ON l.PATIENT_DK = tx.PATIENT_DK AND DATE(l.LAB_COLLECTION_DTM) < tx.tx_date
  GROUP BY 1, 2
),
reg AS (
  SELECT PATIENT_DK,
         COUNT(*) AS n_primaries,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx_date,
         ANY_VALUE(CAST(SITE_PRIMARY_ICD_O_3 AS STRING)) AS site,
         ANY_VALUE(CAST(SUMMARY_STAGE_2018 AS STRING)) AS stage,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_contact,
         MAX(COALESCE(
             SAFE.PARSE_DATE('%Y-%m-%d', TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING))),
             SAFE.PARSE_DATE('%m/%d/%Y', TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING))),
             SAFE_CAST(TRIM(CAST(DATE_RECURRENCE_SUMMARY AS STRING)) AS DATE)
         )) AS recur_date
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
  GROUP BY 1
)
SELECT labs.PATIENT_DK, reg.n_primaries, reg.dx_date, labs.tx_date, reg.site, reg.stage,
       reg.recur_date, reg.last_contact, labs.pre_lab_days,
       DATE_DIFF(reg.recur_date, labs.tx_date, DAY) AS days_tx_to_recur
FROM labs JOIN reg USING (PATIENT_DK)
WHERE labs.pre_lab_days BETWEEN 8 AND 25
  AND reg.n_primaries = 1                 -- one cancer only, no cross-wiring
  AND labs.tx_date >= reg.dx_date         -- treatment after diagnosis
  AND reg.recur_date > labs.tx_date       -- recurrence AFTER treatment (temporal integrity)
ORDER BY labs.PATIENT_DK
LIMIT 1
""")

if found.empty:
    print("no patient matched"); raise SystemExit
p = found.iloc[0]
PDK = str(p.PATIENT_DK)   # opaque string key, not an int

print("=" * 78)
print("THE PATIENT  (this is the whole training example, for one person)")
print("=" * 78)
print(f"  diagnosis date   : {p.dx_date}")
print(f"  cancer site      : {p.site}")
print(f"  stage            : {p.stage}")
print(f"  TREATMENT START  : {p.tx_date}      <-- index date; we predict FROM here")
print(f"  RECURRENCE date  : {p.recur_date}   <-- the LABEL, and it is AFTER treatment ({p.days_tx_to_recur} days later)")
print(f"  last contact     : {p.last_contact}")
print(f"  primaries        : {p.n_primaries}  (1 = single cancer, no cross-wiring)")
print(f"  pre-tx lab days  : {p.pre_lab_days}  <-- how many visits feed the model")

# ---- 2. the TREATMENT (the 'action' we condition on) ----
print("\n" + "=" * 78)
print("WHAT TREATMENT THEY GOT  (the action channel)")
print("=" * 78)
tx = q(f"""
SELECT REGEXP_EXTRACT(UPPER(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING)),
                      r'^([A-Z][A-Z\\-]{{2,}})') AS drug,
       MIN(DATE(t.TREATMENT_DTM)) AS first_given, COUNT(*) AS n_admin
FROM {D}.FACT_TREATMENT_DETAIL t
JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
WHERE t.PATIENT_DK = '{PDK}'
  AND UPPER(CAST(m.MED_THERAPEUTIC_CLASS_DESCRIPTION AS STRING)) = 'ANTINEOPLASTICS'
GROUP BY 1 ORDER BY 2""")
print(tx.to_string(index=False))

# ---- 3. the INPUT: pre-treatment lab timeline as a [visit x feature] grid ----
print("\n" + "=" * 78)
print("WHAT WE FEED IN  (pre-treatment labs over time = the [visit x feature] cube)")
print("=" * 78)
raw = q(f"""
SELECT DATE(l.LAB_COLLECTION_DTM) AS visit_date,
       CAST(d.{NAMECOL} AS STRING) AS test,
       CAST(l.RESULT_TXT AS STRING) AS value
FROM {D}.FACT_LAB_TEST l
JOIN {D}.DIM_LAB_TEST d USING (LAB_TEST_DK)
WHERE l.PATIENT_DK = '{PDK}'
  AND DATE(l.LAB_COLLECTION_DTM) < DATE('{p.tx_date}')
  AND DATE(l.LAB_COLLECTION_DTM) >= DATE_SUB(DATE('{p.tx_date}'), INTERVAL 3 YEAR)
""")
top_tests = raw["test"].value_counts().head(12).index.tolist()
grid = (raw[raw["test"].isin(top_tests)]
        .pivot_table(index="visit_date", columns="test", values="value",
                     aggfunc="last")
        .reindex(columns=top_tests))
print(f"\n  rows = visits ({grid.shape[0]}), columns = features ({grid.shape[1]} shown), "
      f"blanks = not measured that day\n")
print(grid.to_string())

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"single_patient | pre_tx_visits={grid.shape[0]} | distinct_tests={raw['test'].nunique()} "
      f"| regimen={'+'.join(tx['drug'].dropna().head(3))}")

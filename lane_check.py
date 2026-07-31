"""
Decide the lab lane: native FACT_LAB_TEST (string values, PATIENT_DK) vs
OMOP measurement (numeric values + standard names, person_id).
Two questions:
  A. Is native lab naming really broken, or was my join wrong?
  B. Can we bridge registry PATIENT_DK -> OMOP person_id (so we can use measurement)?
All aggregate/structural. Safe.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def q(sql): return C.query(sql).to_dataframe()


print("=" * 72); print("A. NATIVE: is DIM_LAB_TEST usable for names?"); print("=" * 72)
print("columns:")
print(q(f"""SELECT column_name, data_type FROM {D}.INFORMATION_SCHEMA.COLUMNS
          WHERE table_name='DIM_LAB_TEST' ORDER BY ordinal_position""").to_string(index=False))
print("\n5 sample rows:")
print(q(f"SELECT * FROM {D}.DIM_LAB_TEST LIMIT 5").to_string(index=False)[:1500])
print("\ndoes FACT_LAB_TEST.LAB_TEST_DK match DIM_LAB_TEST? (200k-row sample)")
m = q(f"""
SELECT COUNT(*) AS fact_rows, COUNTIF(d.LAB_TEST_DK IS NOT NULL) AS matched
FROM (SELECT LAB_TEST_DK FROM {D}.FACT_LAB_TEST LIMIT 200000) f
LEFT JOIN {D}.DIM_LAB_TEST d USING (LAB_TEST_DK)""")
native_match = m.matched[0] / m.fact_rows[0]
print(f"  matched {m.matched[0]:,}/{m.fact_rows[0]:,} = {native_match:.0%}")

print("\n" + "=" * 72); print("B. OMOP: can we link registry patients to measurement?"); print("=" * 72)
print("person columns:")
print(q(f"""SELECT column_name, data_type FROM {D}.INFORMATION_SCHEMA.COLUMNS
          WHERE table_name='person' ORDER BY ordinal_position""").to_string(index=False))
print("\nsample person source values:")
print(q(f"SELECT person_id, CAST(person_source_value AS STRING) AS src FROM {D}.person LIMIT 5").to_string(index=False))

print("\ndoes person_source_value match PATIENT_DK or CLINIC_NUMBER? (10k person sample)")
link = q(f"""
WITH ps AS (
  SELECT DISTINCT CAST(person_source_value AS STRING) AS psv
  FROM {D}.person WHERE person_source_value IS NOT NULL LIMIT 10000
)
SELECT
  (SELECT COUNT(*) FROM ps) AS sampled,
  (SELECT COUNT(*) FROM ps WHERE psv IN (SELECT CAST(PATIENT_DK AS STRING) FROM {D}.DIM_PATIENT)) AS match_patient_dk,
  (SELECT COUNT(*) FROM ps WHERE psv IN (SELECT CAST(PATIENT_CLINIC_NUMBER AS STRING) FROM {D}.DIM_PATIENT)) AS match_clinic_number
""")
print(link.to_string(index=False))
s = int(link.sampled[0])
dk = link.match_patient_dk[0] / s if s else 0
cn = link.match_clinic_number[0] / s if s else 0
bridge = ("PATIENT_DK" if dk > 0.5 else "CLINIC_NUMBER" if cn > 0.5 else "none_found")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"lane_check | native_name_match={native_match:.0%} | omop_bridge={bridge} "
      f"(dk={dk:.0%} clinic={cn:.0%})")

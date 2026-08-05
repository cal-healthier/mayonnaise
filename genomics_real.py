"""
genomics_real.py -- three things the last run left open.

1. COUNT REAL RESULTS PROPERLY. My "usable" tag only excluded the single most
   common filler; these columns have several (NOT APPLIC + UNKNOWN + NONE +
   NOT DONE all mean "no result"). HER2_NEU_DERIVED looked like 74,766 results
   and is really ~2,000. Redo it excluding every filler pattern.
2. OPEN THE NARRATIVE GENETIC REPORTS. 73,012 patients, 44,625 rows with
   `value_source_value` populated. Is that real variant text or "not detected"?
   20x the structured panels if it is real.
3. PROBE THE PATHOLOGY SLIDES. 24.8M slide IDs bridged to DICOM, 26.8M specimen
   records. Can slides be linked to patients, and how many patients have them?
   This is potentially far bigger than the molecular data.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 130)

FILLER = (r"NOT APPLIC|UNKNOWN|NOT DOCUMENTED|^\(NONE\)|NOT DONE|"
          r"RESULTS NOT IN|NO INFORMATION|NOT PERFORMED|^<NULL>|^X{1,2}\d|^R99")

cols = [r.column_name for r in C.query(f"""
  SELECT column_name FROM {S}.COLUMNS
  WHERE table_name='FACT_CANCER_DATA_REPOSITORY'
    AND REGEXP_CONTAINS(UPPER(column_name),
      r'EGFR|BRAF|KRAS|NRAS|HER2|MSI|MICROSAT|ONCOTYPE|MULTIGENE|ESTROGEN|'
      r'PROGESTERONE|MOLECULAR|CYTOGEN|KI_?67|METHYLATION|KIT_GENE|ALLRED')
  ORDER BY column_name""")]

union = "\nUNION ALL\n".join(
    f"SELECT '{c}' AS col, IFNULL(CAST({c} AS STRING),'<NULL>') AS val, COUNT(*) AS n "
    f"FROM {D}.FACT_CANCER_DATA_REPOSITORY GROUP BY 1,2" for c in cols)
V = C.query(union).to_dataframe()
V["is_filler"] = V["val"].str.upper().str.contains(FILLER, regex=True, na=False)

print("=" * 88)
print("1. REAL RESULTS ONLY  (every filler code excluded, not just the top one)")
print("=" * 88)
real = (V[~V["is_filler"]].groupby("col")["n"].sum()
        .sort_values(ascending=False))
print(f"  {'marker column':<44}{'patients with a real result':>28}")
for k, v in real.items():
    bar = "#" * max(0, int(v / max(real.max(), 1) * 30))
    print(f"  {k:<44}{int(v):>12,}   {bar}")

print("\n  what those real values look like, for the top 6 columns:")
for c in real.head(6).index:
    g = V[(V["col"] == c) & (~V["is_filler"])].sort_values("n", ascending=False)
    vals = ", ".join(f"{r['val'][:24]}={int(r['n']):,}" for _, r in g.head(5).iterrows())
    print(f"    {c}\n        {vals}")

# ---------------------------------------------------------------- 2. narrative text
print("\n" + "=" * 88)
print("2. WHAT IS ACTUALLY IN THE 44,625 GENETIC REPORT TEXTS?")
print("=" * 88)
smp = C.query(f"""
  SELECT measurement_source_value AS test, value_source_value AS result,
         COUNT(*) AS n, COUNT(DISTINCT person_id) AS pts
  FROM {D}.measurement
  WHERE measurement_concept_id = 42529070 AND value_source_value IS NOT NULL
  GROUP BY 1,2 ORDER BY n DESC LIMIT 25
""").to_dataframe()
print(smp.to_string(index=False))
lens = C.query(f"""
  SELECT APPROX_QUANTILES(LENGTH(value_source_value), 100)[OFFSET(50)] AS med_len,
         APPROX_QUANTILES(LENGTH(value_source_value), 100)[OFFSET(90)] AS p90_len,
         MAX(LENGTH(value_source_value)) AS max_len,
         COUNT(DISTINCT value_source_value) AS distinct_results
  FROM {D}.measurement
  WHERE measurement_concept_id = 42529070 AND value_source_value IS NOT NULL
""").to_dataframe().iloc[0]
print(f"\n  result text length: median {int(lens['med_len'])}, p90 "
      f"{int(lens['p90_len'])}, max {int(lens['max_len'])} chars")
print(f"  distinct result strings: {int(lens['distinct_results']):,}")
print("  -> short + few distinct = coded statuses. long + many = real variant reports.")

# ---------------------------------------------------------------- 3. pathology slides
print("\n" + "=" * 88)
print("3. DIGITAL PATHOLOGY SLIDES -- can we reach patients?")
print("=" * 88)
sp = C.query(f"""
  SELECT COUNT(*) AS rows_,
         COUNT(DISTINCT PATIENT_CLINIC_NUMBER) AS patients,
         COUNT(DISTINCT SLIDE_ID) AS slides,
         COUNT(DISTINCT SPECIMEN_NUMBER) AS specimens,
         COUNTIF(SLIDE_ID IS NOT NULL) AS with_slide_id,
         MIN(DATE(SPECIMEN_COLLECTION_DTM)) AS first_dt,
         MAX(DATE(SPECIMEN_COLLECTION_DTM)) AS last_dt
  FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
""").to_dataframe().iloc[0]
for k in ("rows_", "patients", "slides", "specimens", "with_slide_id"):
    print(f"  {k:<18}{int(sp[k]):>14,}")
print(f"  collection dates   {sp['first_dt']} .. {sp['last_dt']}")

print("\n  do those slide IDs appear in the DICOM bridge (i.e. are images addressable)?")
ov = C.query(f"""
  SELECT COUNT(*) AS matched FROM (
    SELECT DISTINCT SLIDE_ID FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
    WHERE SLIDE_ID IS NOT NULL LIMIT 200000) a
  JOIN (SELECT DISTINCT SLIDE_ID FROM {D}.DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE) b
    USING (SLIDE_ID)
""").to_dataframe().iloc[0]
print(f"    of a 200k slide-ID sample, {int(ov['matched']):,} are in the DICOM bridge")

print("\n  stain types (H&E vs immunohistochemistry?):")
st = C.query(f"""
  SELECT CAST(BLOCK_STAIN_PART AS STRING) AS stain, COUNT(*) AS n,
         COUNT(DISTINCT PATIENT_CLINIC_NUMBER) AS pts
  FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
  GROUP BY 1 ORDER BY n DESC LIMIT 15
""").to_dataframe()
print(st.to_string(index=False))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"genomics_real | top_marker={real.index[0]}({int(real.iloc[0])}) "
      f"| markers_over_5k={int((real > 5000).sum())} "
      f"| narrative_distinct={int(lens['distinct_results'])} "
      f"| narrative_medlen={int(lens['med_len'])} "
      f"| path_patients={int(sp['patients'])} | path_slides={int(sp['slides'])}")

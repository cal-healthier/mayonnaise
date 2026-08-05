"""
pathology_probe.py -- is the digital pathology real and reachable?

The molecular data turned out modest: ER/PR/HER2 ~15k patients, everything else
low thousands, and the 44,625 "genetic reports" are ~40% redaction tokens plus
pharmacogenomics (VKORC1/DPYD/UGT1A1) rather than tumour sequencing.

But there are 24.8M slide IDs bridged to DICOM and 26.8M specimen records.
That would be the largest untapped asset here by far -- and it is what
Oncoformer did NOT have (they used chest X-rays).

Questions that decide whether it is usable:
  1. How many distinct PATIENTS have slides, and over what years?
  2. Do the slide IDs actually resolve into the DICOM bridge (addressable
     images) or is the bridge a stub?
  3. What stains? H&E is the substrate for morphology models; IHC slides carry
     ER/PR/HER2 readings and might recover markers missing from the registry.
  4. Do slide-bearing patients overlap our existing cohorts (prostate ADT,
     immunotherapy) -- i.e. could imaging be added to a model we already have?
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

print("=" * 84)
print("1. SCALE AND REACH")
print("=" * 84)
sp = C.query(f"""
  SELECT COUNT(*) AS rows_,
         COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS patients,
         COUNT(DISTINCT CAST(SLIDE_ID AS STRING)) AS slides,
         COUNT(DISTINCT CAST(SPECIMEN_NUMBER AS STRING)) AS specimens,
         COUNTIF(SLIDE_ID IS NOT NULL) AS rows_with_slide,
         MIN(DATE(SPECIMEN_COLLECTION_DTM)) AS first_dt,
         MAX(DATE(SPECIMEN_COLLECTION_DTM)) AS last_dt
  FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
""").to_dataframe().iloc[0]
for k, lbl in [("rows_", "specimen rows"), ("patients", "distinct patients"),
               ("slides", "distinct slide IDs"), ("specimens", "distinct specimens"),
               ("rows_with_slide", "rows carrying a slide ID")]:
    print(f"  {lbl:<28}{int(sp[k]):>14,}")
print(f"  collection dates            {sp['first_dt']} .. {sp['last_dt']}")

print("\n" + "=" * 84)
print("2. DO SLIDE IDS RESOLVE INTO THE DICOM BRIDGE?")
print("=" * 84)
br = C.query(f"""
  SELECT COUNT(*) AS rows_, COUNT(DISTINCT CAST(SLIDE_ID AS STRING)) AS slides
  FROM {D}.DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE
""").to_dataframe().iloc[0]
print(f"  bridge: {int(br['rows_']):,} rows, {int(br['slides']):,} distinct slide IDs")
ov = C.query(f"""
  WITH a AS (SELECT DISTINCT CAST(SLIDE_ID AS STRING) AS s
             FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION WHERE SLIDE_ID IS NOT NULL),
       b AS (SELECT DISTINCT CAST(SLIDE_ID AS STRING) AS s
             FROM {D}.DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE)
  SELECT (SELECT COUNT(*) FROM a) AS in_specimens,
         (SELECT COUNT(*) FROM b) AS in_bridge,
         (SELECT COUNT(*) FROM a JOIN b USING (s)) AS matched
""").to_dataframe().iloc[0]
m = int(ov["matched"]); a = int(ov["in_specimens"])
print(f"  specimen slide IDs {a:,} | bridge {int(ov['in_bridge']):,} | "
      f"matched {m:,} ({m/max(a,1):.1%})")
print("  -> high match = images are addressable; near-zero = the bridge is a stub")

print("\n" + "=" * 84)
print("3. WHAT STAINS?  (H&E for morphology; IHC carries ER/PR/HER2)")
print("=" * 84)
for col in ("BLOCK_STAIN_PART", "STAIN_PROCESS", "SPECIMEN_PART_TYPE_NAME"):
    try:
        t = C.query(f"""
          SELECT CAST({col} AS STRING) AS v, COUNT(*) AS n,
                 COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS pts
          FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
          GROUP BY 1 ORDER BY n DESC LIMIT 12""").to_dataframe()
        print(f"\n  {col}:")
        print(t.to_string(index=False))
    except Exception as e:
        print(f"\n  {col}: {type(e).__name__} {str(e)[:70]}")

print("\n" + "=" * 84)
print("4. DO SLIDE PATIENTS OVERLAP THE COHORTS WE ALREADY MODELLED?")
print("=" * 84)
for f, lbl in [("psa_progression.parquet", "prostate on hormone therapy"),
               ("ici_thyroid_label.parquet", "immunotherapy"),
               ("ov_label.parquet", "ovarian on platinum")]:
    try:
        idx = pd.read_parquet(f).index.astype(str)
        ids = "','".join(idx[:20000])
        n = list(C.query(f"""
          SELECT COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS n
          FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
          WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
            AND SLIDE_ID IS NOT NULL"""))[0].n
        print(f"  {lbl:<32}{int(n):>7,} of {len(idx):,} have slides "
              f"({int(n)/len(idx):.0%})")
    except Exception as e:
        print(f"  {lbl:<32}skipped ({type(e).__name__})")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"pathology_probe | patients={int(sp['patients'])} | slides={int(sp['slides'])} "
      f"| bridge_slides={int(br['slides'])} | matched={m} "
      f"| match_rate={m/max(a,1):.1%} | years={sp['first_dt']}..{sp['last_dt']}")

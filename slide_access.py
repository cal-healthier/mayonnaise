"""
slide_access.py -- can we actually GET a whole-slide image, or only its ID?

24.3M slides over 1.15M patients, 1935-2026, and 54% of slide IDs resolve into
the DICOM bridge. But an identifier proves the slide is catalogued, not that
pixels are reachable from inside the enclave.

Checks, cheapest first:
  1. what the DICOM bridge actually contains (a UID? a path? a URL?)
  2. the real stain breakdown -- BLOCK_STAIN_PART turned out to be block/section
     numbering (A1-1, B1-2), not stains. Try STAIN_PROCESS and any other column
     that might carry H&E vs immunohistochemistry.
  3. any imaging tables elsewhere in the dataset (DICOM metadata, GCS paths)
  4. whether a Google Healthcare DICOM store is visible to this project
  5. whether any GCS bucket in the project holds slide data

If pixels are unreachable, this is a Mayo conversation, not a build.
"""
import subprocess
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)

def sh(cmd, n=12):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
        out = (r.stdout or r.stderr).strip().splitlines()
        return "\n".join("    " + l for l in out[:n]) or "    (no output)"
    except Exception as e:
        return f"    {type(e).__name__}: {str(e)[:90]}"

print("=" * 84)
print("1. WHAT IS IN THE DICOM BRIDGE?")
print("=" * 84)
cols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S}.COLUMNS "
    f"WHERE table_name='DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE' ORDER BY ordinal_position")]
print(f"  columns: {cols}")
print(C.query(f"SELECT * FROM {D}.DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE LIMIT 8"
              ).to_dataframe().to_string(index=False))

print("\n" + "=" * 84)
print("2. THE REAL STAIN BREAKDOWN")
print("=" * 84)
pcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S}.COLUMNS "
    f"WHERE table_name='FACT_PATHOLOGY_SPECIMEN_INFORMATION' ORDER BY ordinal_position")]
print(f"  all columns: {pcols}")
for col in [c for c in pcols if "STAIN" in c.upper() or "PART_TYPE" in c.upper()]:
    try:
        t = C.query(f"""SELECT CAST({col} AS STRING) AS v, COUNT(*) AS n,
                        COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS pts
                        FROM {D}.FACT_PATHOLOGY_SPECIMEN_INFORMATION
                        GROUP BY 1 ORDER BY n DESC LIMIT 15""").to_dataframe()
        print(f"\n  {col}:")
        print(t.to_string(index=False))
    except Exception as e:
        print(f"\n  {col}: {type(e).__name__} {str(e)[:70]}")

print("\n" + "=" * 84)
print("3. OTHER IMAGING TABLES / PATH COLUMNS ANYWHERE")
print("=" * 84)
im = C.query(f"""
  SELECT table_name, column_name FROM {S}.COLUMNS
  WHERE REGEXP_CONTAINS(UPPER(column_name),
        r'DICOM|SOP_|STUDY_UID|SERIES|INSTANCE_UID|IMAGE|WSI|SVS|TIFF|GCS|'
        r'BUCKET|URI|URL|FILE_?PATH|OBJECT_?NAME|BLOB')
  ORDER BY table_name LIMIT 40""").to_dataframe()
print(f"  {len(im)} candidate columns")
for t, g in im.groupby("table_name"):
    print(f"    {t}: {', '.join(g['column_name'])}")

print("\n" + "=" * 84)
print("4. IS A DICOM STORE VISIBLE TO THIS PROJECT?")
print("=" * 84)
print(sh("gcloud healthcare datasets list --location=us-central1 2>&1 | head -12"))
print("  other locations:")
print(sh("gcloud healthcare datasets list --location=us 2>&1 | head -6"))

print("\n" + "=" * 84)
print("5. ANY BUCKET HOLDING SLIDE DATA?")
print("=" * 84)
print(sh("gsutil ls 2>&1 | head -20", 20))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"slide_access | bridge_cols={len(cols)} | imaging_cols={len(im)} "
      f"| imaging_tables={im['table_name'].nunique() if len(im) else 0}")

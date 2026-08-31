"""
measure_footprint.py -- MB of WRITTEN record per cancer patient (no pixels).

We get note text, radiology NARRATIVES, lab values, medication orders -- never
imaging or slide pixels. This measures the text+structured footprint per
patient for the two cohorts and extrapolates to the 313k registry.

Notes are the dominant term and reading note_text is the ~2 TB scan we do on
every note pull; radiology narrative adds a smaller text scan. Labs and orders
are ESTIMATED from corpus ratios rather than scanning 23.6B + 2.0B rows -- in
text-size terms they are minor, and the scan is not worth it. Dry-run sizes
print before anything runs.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 220)

clin = sorted(set(
    pd.read_parquet("psa_progression.parquet").index.astype(str)).union(
    pd.read_parquet("ov_label.parquet").index.astype(str)))
ids = "','".join(clin)
N_PT = len(clin)
REG_TOTAL = 313442          # cancer-registry patients
# corpus ratios for the estimated structured parts (per-patient, all patients)
LAB_PER_PT, LAB_BYTES = 23.6e9 / 3.3e6, 70      # value rows x bytes/row
ORD_PER_PT, ORD_BYTES = 2.0e9 / 3.3e6, 150
CANCER_MULT = 2.5           # treated cancer patients are heavier utilisers


def cached(sql, label, cache):
    if os.path.exists(cache):
        print(f"  {label}: cached")
        return pd.read_parquet(cache)
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"  {label}: scanning {job.total_bytes_processed/1e12:.2f} TB ...")
    df = C.query(sql).to_dataframe()
    df.to_parquet(cache)
    return df


print("=" * 74)
print(f"WRITTEN-RECORD FOOTPRINT  ({N_PT:,} treated patients, full history)")
print("=" * 74)

NOTES = cached(f"""
  WITH km AS (
    SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, MIN(pe.person_id) AS person_id
    FROM {D}.DIM_PATIENT p
    JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                        = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
    WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
    GROUP BY 1)
  SELECT km.clinic,
         COUNT(*) AS n_notes,
         SUM(LENGTH(CAST(n.note_text AS STRING))) AS note_bytes
  FROM km JOIN {D}.note n ON n.person_id = km.person_id
  WHERE n.note_text IS NOT NULL
  GROUP BY 1""", "notes (note_text)", "fp_notes.parquet").set_index("clinic")

RAD = cached(f"""
  WITH km AS (
    SELECT DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic, PATIENT_DK
    FROM {D}.DIM_PATIENT
    WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
  SELECT km.clinic,
         COUNT(*) AS n_rad,
         SUM(LENGTH(IFNULL(CAST(r.RADIOLOGY_NARRATIVE AS STRING), ''))) AS rad_bytes
  FROM km JOIN {D}.FACT_RADIOLOGY r ON r.PATIENT_DK = km.PATIENT_DK
  GROUP BY 1""", "radiology (narrative)", "fp_rad.parquet").set_index("clinic")

F = pd.DataFrame(index=clin)
F["note_mb"] = (NOTES["note_bytes"].reindex(F.index).fillna(0)) / 1e6
F["n_notes"] = NOTES["n_notes"].reindex(F.index).fillna(0)
F["rad_mb"] = (RAD["rad_bytes"].reindex(F.index).fillna(0)) / 1e6
F["labs_mb_est"] = LAB_PER_PT * CANCER_MULT * LAB_BYTES / 1e6
F["orders_mb_est"] = ORD_PER_PT * CANCER_MULT * ORD_BYTES / 1e6
F["total_mb"] = F[["note_mb", "rad_mb", "labs_mb_est", "orders_mb_est"]].sum(axis=1)

def row(name, col, unit="MB"):
    s = F[col]
    print(f"  {name:<26}{s.median():>8.2f}{s.quantile(.9):>10.2f}{s.max():>10.2f}   {unit}")

print(f"\n  {'component':<26}{'median':>8}{'p90':>10}{'max':>10}")
print("  " + "-" * 62)
row("clinical notes", "note_mb")
row("radiology narratives", "rad_mb")
print(f"  {'lab values (est)':<26}{F['labs_mb_est'].iloc[0]:>8.2f}{'':>10}{'':>10}   MB")
print(f"  {'medication orders (est)':<26}{F['orders_mb_est'].iloc[0]:>8.2f}{'':>10}{'':>10}   MB")
print("  " + "-" * 62)
row("TOTAL written record", "total_mb")
print(f"\n  notes/patient: median {F['n_notes'].median():.0f}, "
      f"p90 {F['n_notes'].quantile(.9):.0f}, max {int(F['n_notes'].max())}")
print(f"  bytes/note: {(F['note_mb'].sum()/max(F['n_notes'].sum(),1))*1e6:,.0f}")

med_gb = F["total_mb"].median() / 1000
print("\n" + "=" * 74)
print("READING IT")
print("=" * 74)
print(f"  A cancer patient's entire WRITTEN record is ~{F['total_mb'].median():.0f} MB "
      f"(median), {F['total_mb'].quantile(.9):.0f} MB at p90.")
print(f"  That is {med_gb:.3f} GB -- megabytes, not gigabytes, because there are")
print(f"  no pixels. One CT series (~50-500 MB) or one pathology slide (0.5-4 GB)")
print(f"  each outweighs the patient's whole lifetime of text.")
print(f"\n  Whole registry extrapolation: {REG_TOTAL:,} patients x "
      f"{F['total_mb'].median():.0f} MB ~= {REG_TOTAL*F['total_mb'].median()/1e6:.2f} TB")
print(f"  of written record for every cancer patient Mayo has -- fits on one disk.")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"footprint | median_mb={F['total_mb'].median():.1f} | p90_mb={F['total_mb'].quantile(.9):.1f} "
      f"| notes_median={F['n_notes'].median():.0f} | registry_tb={REG_TOTAL*F['total_mb'].median()/1e6:.2f}")

"""
full_gzip_rep.py -- compression on a REPRESENTATIVE cross-cancer sample.

No more prostate/ovarian focus. This draws a random sample from the WHOLE
cancer registry (proportional to real incidence, so it represents "a cancer
patient", not one site), then measures full-history compression per patient
and BROKEN DOWN BY CANCER TYPE.

Two scans: a cheap registry draw to pick patients, then the ~2.2 TB note pull
(same fixed cost as any note query). Both dry-run sizes print first. Results
cache, so re-runs are free.
"""
import gzip, lzma, os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
N = 1000

TYPE = {"C61": "prostate", "C50": "breast", "C34": "lung", "C18": "colon",
        "C25": "pancreas", "C71": "brain", "C43": "melanoma", "C44": "skin",
        "C56": "ovary", "C64": "kidney", "C67": "bladder", "C22": "liver",
        "C16": "stomach", "C53": "cervix", "C54": "uterus", "C73": "thyroid",
        "C20": "rectum", "C15": "esophagus", "C25.9": "pancreas", "C90": "myeloma",
        "C91": "leukemia", "C92": "leukemia", "C82": "lymphoma", "C83": "lymphoma",
        "C85": "lymphoma", "C81": "lymphoma", "C64.9": "kidney"}

# ---- 1. representative cohort from the registry (cheap) ----
coh_cache = "rep_cohort.parquet"
if os.path.exists(coh_cache):
    coh = pd.read_parquet(coh_cache)
    print(f"cohort cached: {len(coh)} patients")
else:
    sql = f"""
    WITH reg AS (
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             SUBSTR(CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING), 1, 3) AS site3,
             MIN(pe.person_id) AS person_id
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r
      JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                          = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
      WHERE CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C%'
      GROUP BY 1, 2)
    SELECT clinic, ANY_VALUE(site3) AS site3, ANY_VALUE(person_id) AS person_id
    FROM reg GROUP BY clinic
    ORDER BY RAND() LIMIT {N}
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"cohort scan: {job.total_bytes_processed/1e9:.1f} GB (registry, cheap)")
    coh = C.query(sql).to_dataframe()
    coh["clinic"] = coh["clinic"].astype(str)
    coh["type"] = coh["site3"].map(lambda s: TYPE.get(s, f"other({s})"))
    coh.to_parquet(coh_cache)

print("\n  sample composition (representative of registry incidence):")
for t, c in coh["type"].value_counts().head(15).items():
    print(f"    {t:<16}{c:>5}  ({c/len(coh):.0%})")

# ---- 2. full-history notes (the ~2.2 TB note scan) ----
ids = "','".join(coh["clinic"])
notes_cache = "fullnotes_rep.parquet"
if os.path.exists(notes_cache):
    Nt = pd.read_parquet(notes_cache)
    print(f"\nnotes cached: {len(Nt):,} notes")
else:
    sql = f"""
    WITH km AS (SELECT clinic, MIN(person_id) AS person_id FROM UNNEST([
      {",".join(f"STRUCT('{r.clinic}' AS clinic,{r.person_id} AS person_id)"
                for r in coh.itertuples())}]) GROUP BY 1)
    SELECT km.clinic, SUBSTR(CAST(n.note_text AS STRING), 1, 50000) AS txt
    FROM km JOIN {D}.note n ON n.person_id = km.person_id
    WHERE n.note_text IS NOT NULL
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    tb = job.total_bytes_processed / 1e12
    print(f"\nNOTE SCAN: {tb:.2f} TB ~= ${tb*5:.0f} at $5/TB")
    Nt = C.query(sql).to_dataframe()
    Nt["clinic"] = Nt["clinic"].astype(str)
    Nt.to_parquet(notes_cache)
    print(f"pulled {len(Nt):,} notes, {Nt['clinic'].nunique()} patients")

# ---- 3. compression per patient + per type ----
ty = coh.set_index("clinic")["type"]
rows = []
for clinic, g in Nt.groupby("clinic"):
    full = "\n".join(g["txt"].astype(str)).encode()
    if len(full) < 500:
        continue
    rows.append((ty.get(clinic, "?"), len(full), len(gzip.compress(full, 9)),
                 len(lzma.compress(full, preset=6)), len(g)))
R = pd.DataFrame(rows, columns=["type", "raw", "gz", "xz", "n"])
med = R["raw"].median()

print("\n" + "=" * 70)
print(f"COMPRESSION, REPRESENTATIVE SAMPLE   {len(R)} patients with notes")
print("=" * 70)
print(f"  raw median         {med/1024:>7.0f} KB")
print(f"  gzip-9 median      {R['gz'].median()/1024:>7.0f} KB   "
      f"{R['gz'].median()/med:>4.0%} of raw   ({R['raw'].sum()/R['gz'].sum():.1f}x)")
print(f"  lzma/xz median     {R['xz'].median()/1024:>7.0f} KB   "
      f"{R['xz'].median()/med:>4.0%} of raw   ({R['raw'].sum()/R['xz'].sum():.1f}x)")
print(f"  notes/patient      median {R['n'].median():.0f}, max {int(R['n'].max())}")

print(f"\n  BY CANCER TYPE (median):")
print(f"  {'type':<16}{'n':>5}{'raw KB':>9}{'gzip %':>9}{'ratio':>8}")
print("  " + "-" * 47)
for t, g in R.groupby("type"):
    if len(g) >= 5:
        print(f"  {t:<16}{len(g):>5}{g['raw'].median()/1024:>9.0f}"
              f"{g['gz'].median()/g['raw'].median():>8.0%}{g['raw'].sum()/g['gz'].sum():>7.1f}x")

print(f"""
{'=' * 70}
READING IT
{'=' * 70}
  This is the compression a REAL cancer patient's full record gets, across the
  registry's actual type mix -- not a prostate number. If the ratio holds
  across types, strip+compress transport generalizes. If one type (e.g. heme,
  with denser notes) compresses very differently, the payload budget must be
  set per type, not globally.""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"gzip_rep | patients={len(R)} | types={R['type'].nunique()} "
      f"| raw_kb={med/1024:.0f} | gz={R['gz'].median()/med:.0%} "
      f"| ratio={R['raw'].sum()/R['gz'].sum():.1f}x")

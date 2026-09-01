"""
recency.py -- how many cancer patients were FIRST SEEN at Mayo in 2017+?

"First seen" != "diagnosed" (Mayo is a referral centre), so measure both:
  dx        registry DATE_OF_DIAGNOSIS >= 2017
  first_note earliest note in the record >= 2017  <- true first Mayo encounter

Cheap: reads note_date + person_id (small columns), NOT note_text, so ~tens of
GB not 2.2 TB. Also sanity-checks the dates, because de-identification can shift
them -- if lots of years are implausible, year filtering is not trustworthy.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
CUT = 2021   # recency cutoff year

cache = "recency.parquet"
if os.path.exists(cache):
    R = pd.read_parquet(cache)
    print(f"cached: {len(R):,} patients")
else:
    sql = f"""
    WITH reg AS (
      SELECT PATIENT_DK, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx
      FROM {D}.FACT_CANCER_DATA_REPOSITORY
      WHERE DATE_OF_DIAGNOSIS IS NOT NULL
      GROUP BY 1),
    pt AS (
      SELECT reg.PATIENT_DK, reg.dx,
             CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic
      FROM reg JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)),
    fn AS (
      SELECT CAST(pe.person_source_value AS STRING) AS clinic,
             MIN(DATE(n.note_date)) AS first_note,
             COUNT(*) AS n_notes
      FROM {D}.note n JOIN {D}.person pe ON pe.person_id = n.person_id
      WHERE n.note_date IS NOT NULL
      GROUP BY 1)
    SELECT pt.PATIENT_DK, pt.dx, fn.first_note, fn.n_notes
    FROM pt LEFT JOIN fn USING (clinic)
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"scan: {job.total_bytes_processed/1e9:.1f} GB (note_date only, cheap)")
    R = C.query(sql).to_dataframe()
    for c in ("dx", "first_note"):
        R[c] = pd.to_datetime(R[c].astype("datetime64[ns]"), errors="coerce")
    R.to_parquet(cache)
    print(f"pulled {len(R):,} cancer patients")

R["dx_yr"] = R["dx"].dt.year
R["fn_yr"] = R["first_note"].dt.year
N = len(R)

print("\n" + "=" * 66)
print(f"DATE SANITY  (are years trustworthy?)")
print("=" * 66)
bad_dx = ((R["dx_yr"] < 1985) | (R["dx_yr"] > 2026)).sum()
bad_fn = ((R["fn_yr"] < 1995) | (R["fn_yr"] > 2026)).sum()
print(f"  implausible dx year (<1985 or >2026):     {bad_dx:,} ({bad_dx/N:.1%})")
print(f"  implausible first-note year (<1995):      {bad_fn:,} "
      f"({bad_fn/max(R['fn_yr'].notna().sum(),1):.1%})")
print(f"  patients with NO notes at all:            {R['first_note'].isna().sum():,} "
      f"({R['first_note'].isna().mean():.0%})")

print("\n" + "=" * 66)
print(f"COHORT: {N:,} cancer-registry patients with a diagnosis date")
print("=" * 66)
for label, col in [("DIAGNOSED (dx year)", "dx_yr"),
                   ("FIRST SEEN AT MAYO (first note year)", "fn_yr")]:
    print(f"\n  {label}")
    for lo, hi, name in [(0, 2015, "before 2015"), (2015, CUT, f"2015-{CUT-1}"),
                         (CUT, 2100, f"{CUT} onward")]:
        m = ((R[col] >= lo) & (R[col] < hi)).sum()
        print(f"    {name:<16}{m:>9,}  ({m/N:.0%})")
    valid = R[col].notna().sum()
    ge = (R[col] >= CUT).sum()
    print(f"    -> {CUT}+: {ge:,} of {N:,} = {ge/N:.0%} of all "
          f"(or {ge/max(valid,1):.0%} of those with a date)")

both = ((R["dx_yr"] >= CUT) | (R["fn_yr"] >= CUT)).sum()
strict = ((R["fn_yr"] >= CUT)).sum()
dxc = (R["dx_yr"] >= CUT).sum()
print("\n" + "=" * 66)
print(f"DOES {CUT}+ SIGNIFICANTLY CUT THE COHORT?")
print("=" * 66)
print(f"  full cohort                              {N:,}   100%")
print(f"  first seen at Mayo {CUT}+                 {strict:,}   {strict/N:.0%}")
print(f"  diagnosed {CUT}+                          {dxc:,}   {dxc/N:.0%}")
print(f"  diagnosed OR first seen {CUT}+            {both:,}   {both/N:.0%}")

print(f"""
  "First seen at Mayo {CUT}+" keeps ~{strict/N:.0%} of the registry. Whether that
  is "significant" is your call, but note the caveat above: first-note year is
  bounded by when the note data itself begins, so a low count can mean the EHR
  record starts later, not that the patient arrived later. If the date-sanity
  flags are high, treat the year cut as approximate.""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"recency | cut={CUT} | cohort={N} | first_seen={strict}({strict/N:.0%}) "
      f"| diagnosed={dxc}({dxc/N:.0%})")

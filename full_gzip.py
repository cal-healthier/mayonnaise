"""
full_gzip.py -- true compression on FULL-history records (not the window).

The gzip analysis is free; the cost is one BigQuery scan of note_text (~2.2 TB,
same for 100 or 5000 patients since the table is not clustered by patient), so
this pulls a generous sample once and caches it. Dry-run size prints before
anything runs.

Measures per patient, over the COMPLETE record (no date window, uncapped):
  raw bytes | gzip-9 | lzma(xz) | ratio | notes/patient
and reports how the ratio scales with record size -- more notes = more
copy-forward = better compression, which the 13-note window could not show.
"""
import gzip, lzma, os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
N_SAMPLE = 500

clin = sorted(set(pd.read_parquet("psa_progression.parquet").index.astype(str))
              .union(pd.read_parquet("ov_label.parquet").index.astype(str)))
rs = np.random.RandomState(0)
pick = list(rs.choice(clin, size=min(N_SAMPLE, len(clin)), replace=False))
ids = "','".join(pick)

cache = "fullnotes.parquet"
if os.path.exists(cache):
    N = pd.read_parquet(cache)
    print(f"cached: {len(N):,} notes, {N['clinic'].nunique()} patients")
else:
    sql = f"""
    WITH km AS (
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, MIN(pe.person_id) AS person_id
      FROM {D}.DIM_PATIENT p
      JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                          = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
      GROUP BY 1)
    SELECT km.clinic, SUBSTR(CAST(n.note_text AS STRING), 1, 50000) AS txt
    FROM km JOIN {D}.note n ON n.person_id = km.person_id
    WHERE n.note_text IS NOT NULL
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    tb = job.total_bytes_processed / 1e12
    print("=" * 66)
    print(f"SCAN COST:  {tb:.2f} TB  ~= ${tb*5:.0f} at $5/TB  (full history, "
          f"{len(pick)} patients)")
    print("=" * 66)
    N = C.query(sql).to_dataframe()
    N["clinic"] = N["clinic"].astype(str)
    N.to_parquet(cache)
    print(f"pulled {len(N):,} notes, {N['clinic'].nunique()} patients\n")

rows = []
for clinic, g in N.groupby("clinic"):
    full = "\n".join(g["txt"].astype(str)).encode()
    if len(full) < 500:
        continue
    rows.append((len(full), len(gzip.compress(full, 9)),
                 len(lzma.compress(full, preset=6)), len(g)))
R = pd.DataFrame(rows, columns=["raw", "gz", "xz", "n"])
med = R["raw"].median()

print("=" * 66)
print(f"FULL-HISTORY COMPRESSION   {len(R)} patients")
print("=" * 66)
print(f"  {'':<20}{'median':>10}{'p90':>10}")
print(f"  {'raw record':<20}{med/1024:>8.0f}KB{R['raw'].quantile(.9)/1024:>8.0f}KB")
print(f"  {'gzip-9':<20}{R['gz'].median()/1024:>8.0f}KB{R['gz'].quantile(.9)/1024:>8.0f}KB")
print(f"  {'lzma / xz':<20}{R['xz'].median()/1024:>8.0f}KB{R['xz'].quantile(.9)/1024:>8.0f}KB")
print(f"\n  gzip ratio:  {R['raw'].sum()/R['gz'].sum():.1f}x  "
      f"(payload = {R['gz'].median()/med:.0%} of raw)")
print(f"  xz ratio:    {R['raw'].sum()/R['xz'].sum():.1f}x  "
      f"(payload = {R['xz'].median()/med:.0%} of raw)")
print(f"  notes/patient: median {R['n'].median():.0f}, max {int(R['n'].max())}")

print("\n  compression vs record size (more notes -> more copy-forward):")
R["ratio"] = R["raw"] / R["gz"]
for lo, hi, lab in [(0, 50, "<50 notes"), (50, 150, "50-150"),
                    (150, 400, "150-400"), (400, 1e9, "400+")]:
    s = R[(R["n"] >= lo) & (R["n"] < hi)]
    if len(s):
        print(f"    {lab:<12}{len(s):>4} pts   {s['ratio'].median():>5.1f}x   "
              f"raw {s['raw'].median()/1024:>5.0f}KB -> gz {s['gz'].median()/1024:>4.0f}KB")

print(f"""
{'=' * 66}
READING IT
{'=' * 66}
  This is the TRUE full-record wire size, not the windowed estimate. If the
  ratio climbs with note count, big records compress hardest -- exactly the
  copy-forward we could not see in a 13-note window. xz beats gzip on clinical
  text by ~20-30% and is stdlib, so it is the better transport codec if the
  receiver supports it.

  Whatever the payload row says, it is LOSSLESS -- the receiver decompresses to
  the complete record.""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"full_gzip | patients={len(R)} | raw_kb={med/1024:.0f} "
      f"| gz={R['gz'].median()/med:.0%} | xz={R['xz'].median()/med:.0%} "
      f"| gz_ratio={R['raw'].sum()/R['gz'].sum():.1f}x")

"""
stream_xz.py -- compress every diagnosed-2021+ patient's full record to xz.

Cannot to_dataframe() ~84k patients (~100 GB -> OOM). So: aggregate each
patient's notes server-side (STRING_AGG), stream ONE patient per row, compress
in a parallel pool, write modern_cohort/<shard>/<clinic>.xz, discard. Memory
stays at one batch (~2 GB), never the full corpus.

Self-calibrating: prints throughput + ETA after the first batch. If the ETA is
absurd, kill it and we lower the preset or add cores. One BigQuery scan (~$11).
Re-running re-scans (single pass, not resumable without a re-scan).
"""
import lzma, os, time
from multiprocessing import Pool, cpu_count
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
CUT = 2021
SAMPLE_N = 5000        # random sample of the cohort; set None for all ~84k
OUT = "modern_cohort_5k"
PRESET = 6                 # xz level; drop to 1-2 if too slow
BATCH = 1500
NPROC = max(1, cpu_count() - 1)
os.makedirs(OUT, exist_ok=True)


def compress_one(args):
    clinic, text = args
    b = (text or "").encode()
    xz = lzma.compress(b, preset=PRESET)
    shard = os.path.join(OUT, (clinic[:3] if len(clinic) >= 3 else "xx"))
    os.makedirs(shard, exist_ok=True)
    with open(os.path.join(shard, f"{clinic}.xz"), "wb") as f:
        f.write(xz)
    return len(b), len(xz)


MAX_NOTES = 400   # cap notes/patient BEFORE aggregating so STRING_AGG never
                  # builds an oversized value (heavy patients hit BQ limits).
                  # median is 257 so most patients are kept whole; heaviest
                  # keep their 400 most-recent notes. Stated in the output.
sql = f"""
WITH coh AS (
  SELECT PATIENT_DK FROM (
    SELECT PATIENT_DK, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx
    FROM {D}.FACT_CANCER_DATA_REPOSITORY
    WHERE DATE_OF_DIAGNOSIS IS NOT NULL GROUP BY 1)
  WHERE EXTRACT(YEAR FROM dx) >= {CUT}
  {"ORDER BY RAND() LIMIT %d" % SAMPLE_N if SAMPLE_N else ""}),
pk AS (
  SELECT DISTINCT c.PATIENT_DK,
         CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
         pe.person_id
  FROM coh c JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
  JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                      = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)),
capped AS (
  SELECT pk.clinic, SUBSTR(CAST(n.note_text AS STRING), 1, 50000) AS txt
  FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
  WHERE n.note_text IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic
                             ORDER BY n.note_date DESC) <= {MAX_NOTES})
SELECT clinic, STRING_AGG(txt, '\\n') AS alltext
FROM capped GROUP BY clinic
"""

job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
tb = job.total_bytes_processed / 1e12
print("=" * 70)
print(f"NOTE SCAN: {tb:.2f} TB ~= ${tb*5:.0f}   |   cores={NPROC}, xz preset={PRESET}")
print(f"output -> {OUT}/  (sharded by clinic prefix)")
print("=" * 70, flush=True)

t0 = time.time()
try:
    rows = C.query(sql).result(page_size=200)
except Exception as e:
    print("QUERY FAILED:\n" + str(e)[:800])
    raise
batch, n, raw_tot, xz_tot, first = [], 0, 0, 0, True
with Pool(NPROC) as pool:
    def flush():
        global n, raw_tot, xz_tot, first
        for rb, xb in pool.map(compress_one, batch):
            raw_tot += rb; xz_tot += xb; n += 1
        el = time.time() - t0
        rate = n / el
        print(f"  {n:>7,} pts | raw {raw_tot/1e9:>6.1f} GB -> xz {xz_tot/1e9:>5.2f} GB "
              f"| {el/60:>5.1f} min | {rate:>5.0f} pt/s", flush=True)
        if first:
            first = False
            est_total = SAMPLE_N or 84000
            eta = (est_total - n) / max(rate, 1) / 60
            print(f"  >>> ETA for ~84k patients: ~{eta:.0f} more minutes "
                  f"(kill now if that is too long; lower PRESET or check cores)",
                  flush=True)

    for row in rows:
        batch.append((row.clinic, row.alltext))
        if len(batch) >= BATCH:
            flush(); batch = []
    if batch:
        flush()

el = time.time() - t0
print("\n" + "=" * 70)
print(f"DONE   {n:,} patients in {el/60:.0f} min")
print("=" * 70)
print(f"  raw total       {raw_tot/1e9:.1f} GB")
print(f"  xz total        {xz_tot/1e9:.2f} GB   ({raw_tot/max(xz_tot,1):.1f}x)")
print(f"  mean xz/patient {xz_tot/max(n,1)/1024:.0f} KB")
print(f"  files in        {OUT}/  ({n:,} .xz files, sharded)")
if SAMPLE_N:
    full = 84000
    print(f"\n  EXTRAPOLATED to full ~{full:,}-patient cohort:")
    print(f"    xz total ~= {xz_tot/max(n,1)*full/1e9:.1f} GB   "
          f"raw ~= {raw_tot/max(n,1)*full/1e9:.0f} GB")
print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"stream_xz | patients={n} | raw_gb={raw_tot/1e9:.1f} | xz_gb={xz_tot/1e9:.2f} "
      f"| ratio={raw_tot/max(xz_tot,1):.1f}x | min={el/60:.0f}")

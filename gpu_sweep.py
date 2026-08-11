"""
gpu_sweep.py -- pull up to 96 notes per patient and embed every one.

The 6-note cap was never justified, and three things now say it is costing us:

  1 note 0.546 -> 3 notes 0.580 -> 6 notes 0.627   monotonic, both truncations
  the drops at 1 and 3 were the ONLY significant results in the pooling run
  inspect_notes found ~49 notes per patient in the year before treatment

So embed per-note with the cap at 96, and sweep_test.py then evaluates
k = 1, 3, 6, 12, 24, 48, 96 for free by filtering on rn <= k and re-pooling.
The expensive step is encoding; once you hold per-note vectors, every
downstream question about how many to use costs nothing.

Sweeping beats picking 12 because the interesting possibility is that the
curve TURNS OVER. Notes from eleven months back may dilute the average once
there are forty of them, and a single point at 12 cannot show that.

SUBSTR is 4,000 chars -- 512 tokens is roughly 2,000, so that is headroom
without dragging a needless gigabyte across the wire.

The GPU is left running; call stop_all() when you are done for the day.
"""
import os, subprocess, time
import pandas as pd
from google.cloud import bigquery

CAP = 96
SEQ = (256, 512)
COHORTS = {
    "ovarian":  ("ov_label.parquet",        "ov_ca125.parquet"),
    "prostate": ("psa_progression.parquet", "psa_values.parquet"),
}

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER, KEY, VP = "calder_healthier", os.path.expanduser("~/.ssh/google_compute_engine"), "~/venv/bin"
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
BOXES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100", 3.70),
         ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4",   0.85)]
C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")


def sh(cmd, n=12, quiet=False, t=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out


def notes(tag):
    cache = f"notes{CAP}_{tag}.parquet"
    gpu = f"notes_gpu{CAP}_{tag}.parquet"
    if os.path.exists(cache):
        N = pd.read_parquet(cache)
    else:
        lab, mark = COHORTS[tag]
        E = pd.read_parquet(lab)
        tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
        E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
        E = E.dropna(subset=["tx_date"])
        E = E[(E["prog"] == 1) | (E["time"] >= 365)]
        rows = ",".join(f"('{c}',DATE '{d.date()}')"
                        for c, d in zip(E.index.astype(str), E["tx_date"]))
        sql = f"""
        WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
        pk AS (SELECT c.clinic, c.tx, pe.person_id FROM coh c
               JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
        SELECT pk.clinic,
               SUBSTR(CAST(n.note_text AS STRING),1,4000) AS txt,
               DATE_DIFF(pk.tx, DATE(n.note_date), DAY) AS days_before,
               ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) AS rn
        FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
        WHERE n.note_text IS NOT NULL
          AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 25000
          AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient','H&P')
          AND DATE(n.note_date) <= pk.tx AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
        QUALIFY rn <= {CAP}
        """
        job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
        print(f"  {tag:<9} pulling up to {CAP}/patient "
              f"({job.total_bytes_processed/1e9:.0f} GB) ...")
        N = C.query(sql).to_dataframe()
        N.to_parquet(cache)
    per = N.groupby("clinic").size()
    print(f"  {tag:<9} {len(N):,} notes | {N['clinic'].nunique():,} patients | "
          f"median {per.median():.0f}/patient, p90 {per.quantile(.9):.0f}, max {per.max()}")
    print(f"  {tag:<9} still capped at {CAP}: {(per >= CAP).mean():.1%} of patients")
    pd.DataFrame({
        "clinic":      N["clinic"].astype(str),
        "txt":         N["txt"].astype(str),
        "rn":          pd.to_numeric(N["rn"]).fillna(0).astype("int64"),
        "days_before": pd.to_numeric(N["days_before"]).fillna(0).astype("int64"),
    }).to_parquet(gpu)
    return N


def grab():
    for name, zone, gpu, cost in BOXES:
        print(f"  trying {gpu} in {zone} ...", end=" ")
        code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                       f"--project={PROJ} 2>&1", quiet=True, t=420)
        print("STARTED/already up" if code == 0
              else (out.splitlines()[0][:60] if out else "failed"))
        if code == 0:
            _, ip = sh(f"gcloud compute instances describe {name} --zone={zone} "
                       f"--project={PROJ} "
                       f"--format='value(networkInterfaces[0].networkIP)' 2>&1", quiet=True)
            return ip.strip().splitlines()[-1]
    return None


def wait_ssh(ip, timeout=420):
    t0, n = time.time(), 0
    while time.time() - t0 < timeout:
        code, _ = sh(f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
                     f"-o BatchMode=yes {USER}@{ip} 'echo up' 2>&1", quiet=True, t=40)
        if code == 0:
            print(f"    sshd answered after {time.time()-t0:.0f}s")
            return True
        n += 1
        if n % 3 == 1:
            print(f"    still booting ({time.time()-t0:.0f}s) ...")
        time.sleep(10)
    return False


def stop_all():
    for name, zone, gpu, _ in BOXES:
        sh(f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
           f"--discard-local-ssd=true --quiet 2>&1", 2, quiet=True)
    print("  both instances stopped. billing ends.")


REMOTE = r'''
import os, re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
print("cuda:", torch.cuda.is_available(), flush=True)
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
CAP = %d
for L in %s:
    m = None
    for tag in %s:
        out = f"rn{CAP}_emb_{tag}_{L}.parquet"
        if os.path.exists(out):
            print(f"  {tag} @ {L}: cached", flush=True); continue
        src = f"notes_gpu{CAP}_{tag}.parquet"
        if not os.path.exists(src):
            print(f"  {tag}: {src} missing", flush=True); continue
        N = pd.read_parquet(src)
        N["nonum"] = N["txt"].str.replace(NUMS, " ", regex=True)
        if m is None:
            m = SentenceTransformer("models/pubmedbert", device="cuda")
            m.max_seq_length = L
        t0 = time.time()
        V = m.encode(N["nonum"].tolist(), batch_size=256,
                     normalize_embeddings=True, show_progress_bar=False)
        o = pd.DataFrame(V)
        o.columns = [str(c) for c in o.columns]
        o["clinic"] = N["clinic"].values
        o["rn"] = N["rn"].values
        o["days_before"] = N["days_before"].values
        o.to_parquet(out)
        dt = time.time() - t0
        print(f"  {tag} @ {L}: {len(N):,} notes in {dt:.0f}s "
              f"({len(N)/dt:.0f}/sec)", flush=True)
    del m; torch.cuda.empty_cache()
print("done")
''' % (CAP, repr(list(SEQ)), repr(list(COHORTS)))


print("=" * 78)
print(f"NOTE-COUNT SWEEP  --  embedding up to {CAP} notes per patient")
print("=" * 78)

print("\n1. notes")
for tag in COHORTS:
    try:
        notes(tag)
    except Exception as e:
        print(f"  {tag:<9} FAILED: {type(e).__name__}: {str(e)[:160]}")

print("\n2. GPU")
ip = grab()
ok = False
if not ip:
    print("  no capacity. retry in a few minutes.")
elif not wait_ssh(ip):
    print("  unreachable.")
else:
    SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=25 {USER}@{ip}"
    SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"
    code, out = sh(f"{SSH} '{VP}/python -c \"import torch;"
                   f"print(torch.cuda.is_available())\"' 2>&1", 3, quiet=True)
    print("    venv good" if "True" in out else "    WARNING: card not visible")

    print("\n3. sending")
    for tag in COHORTS:
        f = f"notes_gpu{CAP}_{tag}.parquet"
        if os.path.exists(f):
            sh(f"{SCP} {f} {USER}@{ip}:~/ 2>&1", 2, quiet=True, t=3600)
            print(f"    {f} ({os.path.getsize(f)/1e6:.0f} MB)")

    print(f"\n4. embedding every note, {len(COHORTS)} cohorts x {len(SEQ)} truncations")
    open("_remote_sweep.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_sweep.py {USER}@{ip}:~/ 2>&1", 2, quiet=True)
    sh(f"{SSH} 'cd ~ && {VP}/python _remote_sweep.py' 2>&1", 24, t=10800)

    print("\n5. copying back")
    got = []
    for tag in COHORTS:
        for L in SEQ:
            f = f"rn{CAP}_emb_{tag}_{L}.parquet"
            c, _o = sh(f"{SCP} {USER}@{ip}:~/{f} . 2>&1", 2, quiet=True, t=3600)
            if c == 0 and os.path.exists(f):
                got.append(f)
    print(f"    {len(got)}/{len(COHORTS)*len(SEQ)}: {', '.join(got)}")
    ok = len(got) > 0
    print(f"\n  GPU LEFT RUNNING at {ip}.  stop_all() when you are done.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"gpu_sweep | cap={CAP} | cohorts={','.join(COHORTS)} | ok={ok} | gpu=RUNNING")

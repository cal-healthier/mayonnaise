"""
gpu_repr.py -- the bake-off said the encoder is not the bottleneck. Find what is.

pubmedbert 0.633 | e5 0.633 | biolord 0.632 | bge 0.629

Four encoders with completely different pretraining -- PubMed abstracts,
biomedical concept similarity, and two general web-text recipes -- landing
within 0.004 of each other is not a tie between good models. It is evidence
that whatever is limiting this number sits somewhere other than the encoder.

The obvious suspect is what we do to the text before and after encoding:

  TRUNCATION   notes cut to 2,500 chars in SQL, then max_seq_length=256
               tokens, which is roughly the first 1,000 characters. Clinical
               notes open with headers and demographics.
  POOLING      up to 6 notes averaged into ONE 768-vector. Averaging is
               destructive: one alarming note and five routine ones average
               to "slightly concerning".
  RECENCY      all 6 notes weighted equally, though the note written the week
               before treatment should matter more than one from 11 months
               earlier.

Averaging over long documents has already burned us three separate times in
this project (sentence similarity, embedding triage, MLM surprisal). Each
time the aggregate hid the signal.

This job re-embeds the ovarian notes on the A100 keeping PER-NOTE vectors --
no pooling at all -- at both 256 and 512 tokens. repr_test.py then tries
different ways of collapsing them to a patient and measures which matters.

Only pubmedbert, since the bake-off showed the choice is immaterial.
"""
import base64, hashlib, json, os, re, ssl, subprocess, time, urllib.request
import pandas as pd
from google.cloud import bigquery

COHORT = "ovarian"
WREPO = "cal-healthier/mayo-weights"
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER = "calder_healthier"
KEY = os.path.expanduser("~/.ssh/google_compute_engine")
VP = "~/venv/bin"
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
BOXES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100", 3.70),
         ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4",   0.85)]
C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")


def sh(cmd, n=12, quiet=False, t=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out


# ------------------------------------------------- notes WITH rank and date
# ovt_notes.parquet kept only (clinic, txt) and the outer SELECT had no
# ORDER BY, so row order cannot be trusted to recover recency. Re-pull with
# the rank made explicit.
def slim(N):
    """Write the GPU's copy with ONLY plain numpy dtypes.

    to_dataframe() returns note_date as BigQuery's 'dbdate' extension type.
    db-dtypes ships with google-cloud-bigquery so the bastion reads it fine,
    but the GPU venv has only torch/sentence-transformers/pandas/pyarrow and
    dies with "data type 'dbdate' not understood" on read_parquet. Never send
    an extension dtype across; the remote does not need the date anyway."""
    G = pd.DataFrame({
        "clinic":      N["clinic"].astype(str),
        "txt":         N["txt"].astype(str),
        "rn":          pd.to_numeric(N["rn"]).fillna(0).astype("int64"),
        "days_before": pd.to_numeric(N["days_before"]).fillna(0).astype("int64"),
    })
    G.to_parquet("ovt_notes_gpu.parquet")
    return G


def notes():
    if os.path.exists("ovt_notes2.parquet"):
        N = pd.read_parquet("ovt_notes2.parquet")
        print(f"  cached: {len(N):,} notes with rank")
        slim(N)
        return N
    E = pd.read_parquet("ov_label.parquet")
    cv = pd.read_parquet("ov_ca125.parquet")
    E["tx_date"] = pd.to_datetime(cv.groupby("clinic")["tx_date"].first().reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    E = E[(E["prog"] == 1) | (E["time"] >= 365)]
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(E.index.astype(str), E["tx_date"]))
    sql = f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
    pk AS (SELECT c.clinic, c.tx, pe.person_id FROM coh c
           JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
    SELECT pk.clinic,
           SUBSTR(CAST(n.note_text AS STRING),1,6000) AS txt,
           DATE(n.note_date) AS note_date,
           DATE_DIFF(pk.tx, DATE(n.note_date), DAY) AS days_before,
           ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) AS rn
    FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
    WHERE n.note_text IS NOT NULL
      AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 25000
      AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient','H&P')
      AND DATE(n.note_date) <= pk.tx AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
    QUALIFY rn <= 6
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"  re-pulling notes with rank + date ({job.total_bytes_processed/1e9:.1f} GB)")
    N = C.query(sql).to_dataframe()
    if "note_date" in N.columns:
        N["note_date"] = pd.to_datetime(N["note_date"].astype("datetime64[ns]"))
    N.to_parquet("ovt_notes2.parquet")
    slim(N)
    print(f"  {len(N):,} notes, {N['clinic'].nunique():,} women")
    return N


# ---------------------------------------------------------------- gpu plumbing
def api(path):
    url = f"https://api.github.com/repos/{WREPO}/{path}?_={int(time.time())}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=180, context=CTX))


def grab():
    for name, zone, gpu, cost in BOXES:
        print(f"  trying {gpu} in {zone} ...", end=" ")
        code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                       f"--project={PROJ} 2>&1", quiet=True, t=420)
        print("STARTED" if code == 0 else (out.splitlines()[0][:60] if out else "failed"))
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
N = pd.read_parquet("ovt_notes_gpu.parquet")
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
N["nonum"] = N["txt"].str.replace(NUMS, " ", regex=True)
print(f"{len(N):,} notes", flush=True)
for L in (256, 512):
    out = f"rn_emb_{L}.parquet"
    if os.path.exists(out):
        print(f"  {L}: cached", flush=True); continue
    m = SentenceTransformer("models/pubmedbert", device="cuda")
    m.max_seq_length = L
    t0 = time.time()
    V = m.encode(N["nonum"].tolist(), batch_size=128,
                 normalize_embeddings=True, show_progress_bar=False)
    o = pd.DataFrame(V)
    o.columns = [str(c) for c in o.columns]
    o["clinic"] = N["clinic"].values
    o["rn"] = N["rn"].values
    o["days_before"] = N["days_before"].values
    o.to_parquet(out)          # PER-NOTE, deliberately not pooled
    dt = time.time() - t0
    print(f"  {L} tokens: {len(N):,} notes in {dt:.1f}s ({len(N)/dt:.0f}/sec)", flush=True)
    del m; torch.cuda.empty_cache()
print("done")
'''


print("=" * 78)
print("PER-NOTE EMBEDDING  --  finding the real bottleneck")
print("=" * 78)

print("\n1. notes with rank and date")
N = notes()

print("\n2. starting a GPU")
ip = grab()
if not ip:
    print("  no capacity. try again shortly.")
else:
    SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=25 {USER}@{ip}"
    SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"
    print("\n3. waiting for sshd")
    if not wait_ssh(ip):
        print("  unreachable. stopping.")
        stop_all()
    else:
        print("\n4. environment")
        code, out = sh(f"{SSH} '{VP}/python -c \"import torch,sentence_transformers;"
                       f"print(torch.cuda.is_available())\"' 2>&1", 3, quiet=True)
        if "True" not in out:
            print("    installing ...")
            sh(f"{SSH} 'test -x {VP}/python || python3 -m venv ~/venv'", 2, quiet=True)
            sh(f"{SSH} '{VP}/pip install -q --upgrade pip'", 2, quiet=True, t=600)
            sh(f"{SSH} '{VP}/pip install torch 2>&1 | tail -2'", 3, t=3000)
            sh(f"{SSH} '{VP}/pip install -q sentence-transformers pandas pyarrow "
               f"2>&1 | tail -2'", 3, t=1800)
        else:
            print("    venv good, card visible")

        print("\n5. sending notes")
        sh(f"{SSH} 'mkdir -p ~/models'", 2, quiet=True)
        _, have = sh(f"{SSH} 'test -f ~/models/pubmedbert/model.safetensors && echo YES'",
                     2, quiet=True)
        if "YES" not in have:
            sh(f"{SCP} -r models/pubmedbert {USER}@{ip}:~/models/ 2>&1", 2, quiet=True, t=1800)
        sh(f"{SCP} ovt_notes_gpu.parquet {USER}@{ip}:~/ 2>&1", 2, quiet=True, t=900)
        print("    sent")

        print("\n6. embedding per-note at 256 and 512 tokens")
        open("_remote_repr.py", "w").write(REMOTE)
        sh(f"{SCP} _remote_repr.py {USER}@{ip}:~/ 2>&1", 2, quiet=True)
        sh(f"{SSH} 'cd ~ && {VP}/python _remote_repr.py' 2>&1", 16, t=3600)

        print("\n7. copying back")
        got = 0
        for L in (256, 512):
            c, _o = sh(f"{SCP} {USER}@{ip}:~/rn_emb_{L}.parquet . 2>&1", 2,
                       quiet=True, t=900)
            got += (c == 0 and os.path.exists(f"rn_emb_{L}.parquet"))
        print(f"    {got}/2 retrieved")

        print("\n8. stopping the GPU")
        stop_all()

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"gpu_repr | per_note=256,512 | ready={os.path.exists('rn_emb_256.parquet')}"
      f",{os.path.exists('rn_emb_512.parquet')} | gpu=stopped")

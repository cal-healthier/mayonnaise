"""
gpu_job2.py -- same job, but install into a venv.

The A100 is confirmed (A100-SXM4-40GB, driver 610.43.02) and the files are
already copied. The only failure was PEP 668: Debian's python3.13 refuses
system-wide pip installs on a fresh image. Fix is a virtualenv.

torch+cu124 is a ~2.5GB download, so the install step takes a few minutes.
Everything after that is seconds.

  go()        venv -> install -> embed -> fetch back
  stop_all()  ALWAYS when done -- $3.70/hr while RUNNING
"""
import os, subprocess, time

IP, USER = "10.188.56.20", "calder_healthier"
KEY = os.path.expanduser("~/.ssh/google_compute_engine")
SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=20 {USER}@{IP}"
SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"
VP = "~/venv/bin"
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")

def sh(cmd, n=14, quiet=False, t=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out

def setup():
    print("creating venv and installing (torch cu124 is ~2.5GB, be patient) ...")
    sh(f"{SSH} 'test -x {VP}/python || python3 -m venv ~/venv' 2>&1", 4)
    sh(f"{SSH} '{VP}/python -c \"import sentence_transformers\" 2>/dev/null "
       f"|| ({VP}/pip install -q --upgrade pip && "
       f"{VP}/pip install -q torch --index-url https://download.pytorch.org/whl/cu124 && "
       f"{VP}/pip install -q sentence-transformers pandas pyarrow)' 2>&1", 12, t=3000)
    print("\nCUDA visible to torch?")
    code, out = sh(f"{SSH} '{VP}/python -c \"import torch;print(torch.__version__, "
                   f"torch.cuda.is_available(), torch.cuda.get_device_name(0))\"' 2>&1", 4)
    return code == 0 and "True" in out

REMOTE = r'''
import re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
N = pd.read_parquet("tvn_notes.parquet")
print(f"{len(N):,} notes", flush=True)
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
PROG = re.compile(r"\b(prognos\w*|terminal|hospice|palliat\w*|aggressive|high[- ]risk|"
  r"advanced|metasta\w*|refractor\w*|progress\w*|deteriorat\w*|declin\w*|guarded|"
  r"grave|life expectancy|poor\w*|worsen\w*|stable|improv\w*|respond\w*|remission|"
  r"recurren\w*|relaps\w*|spread\w*|widespread|extensive|bulky|burden)\b", re.I)
N["nonum"]  = N["txt"].str.replace(NUMS, " ", regex=True)
N["noprog"] = N["txt"].str.replace(PROG, " ", regex=True)
m = SentenceTransformer("models/pubmedbert", device="cuda")
for col in ("txt", "nonum", "noprog"):
    t0 = time.time()
    V = m.encode(N[col].tolist(), batch_size=256, normalize_embeddings=True,
                 show_progress_bar=False)
    out = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(f"tvn_emb_{col}.parquet")
    dt = time.time() - t0
    print(f"  {col}: {len(N):,} notes in {dt:.0f}s ({len(N)/dt:.0f}/sec)", flush=True)
print("done")
'''

def run_embed():
    open("_remote_embed.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_embed.py {USER}@{IP}:~/ 2>&1", 2)
    print("\nembedding on the A100 ...")
    sh(f"{SSH} 'cd ~ && {VP}/python _remote_embed.py' 2>&1", 20, t=3600)

def fetch():
    print("\nfetching back ...")
    for arm in ("txt", "nonum", "noprog"):
        sh(f"{SCP} {USER}@{IP}:~/tvn_emb_{arm}.parquet . 2>&1", 2, t=900)
    sh("ls -la tvn_emb_*.parquet 2>&1", 6)

def stop_all():
    for name, zone in [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f"),
                       ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c")]:
        sh(f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
           f"--discard-local-ssd=true --quiet 2>&1", 3)
    print("  stopped. billing ends.")

def go():
    if not setup():
        print("\n  torch still not seeing CUDA -- read the output above")
        return
    run_embed(); fetch()
    print("\n" + "=" * 70)
    print("  next:  exec(open(pull('tvn_analyze.py')).read())")
    print("  then:  stop_all()      <- $3.70/hr until you do")
    print("=" * 70)

print(__doc__)
go()
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_job2 | venv fix | remember stop_all()")

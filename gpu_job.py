"""
gpu_job.py -- run the embedding on the A100, end to end.

The box is RUNNING at 10.188.56.20 and billing at ~$3.70/hr. sshd just needed
longer than my 25s wait -- cold boot is 60-90s.

  go()        wait for ssh -> install deps -> copy work -> embed -> fetch back
  stop_all()  ALWAYS when finished

Each step is idempotent, so re-running after a failure picks up where it left
off rather than starting over.
"""
import os, subprocess, time

IP, USER = "10.188.56.20", "calder_healthier"
KEY = os.path.expanduser("~/.ssh/google_compute_engine")
SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15 {USER}@{IP}"
SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")

def sh(cmd, n=14, quiet=False, t=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out

def wait_ssh(mins=4):
    print(f"waiting for sshd on {IP} ...")
    for i in range(mins * 6):
        code, _ = sh(f"{SSH} 'echo up' 2>&1", quiet=True, t=40)
        if code == 0:
            print(f"  up after {i*10}s")
            return True
        time.sleep(10)
    print("  still refusing. give it another minute and re-run go()")
    return False

def setup():
    print("\nGPU on the box:")
    sh(f"{SSH} 'nvidia-smi --query-gpu=name,memory.total,driver_version "
       f"--format=csv,noheader' 2>&1", 4)
    print("\ninstalling deps (skips if already present) ...")
    sh(f"{SSH} 'python3 -c \"import sentence_transformers,torch;"
       f"print(torch.__version__, torch.cuda.is_available())\" 2>/dev/null "
       f"|| (pip install -q torch --index-url https://download.pytorch.org/whl/cu124 "
       f"&& pip install -q sentence-transformers pandas pyarrow)' 2>&1", 10, t=2400)
    print("\nverifying CUDA is visible to torch:")
    sh(f"{SSH} 'python3 -c \"import torch;print(torch.__version__, "
       f"torch.cuda.is_available(), torch.cuda.get_device_name(0))\"' 2>&1", 4)

def push_work():
    print("\ncopying notes + model across (internal network, fast) ...")
    sh(f"{SSH} 'mkdir -p ~/models'", quiet=True)
    sh(f"{SCP} tvn_notes.parquet {USER}@{IP}:~/ 2>&1", 3, t=1800)
    code, _ = sh(f"{SSH} 'test -f ~/models/pubmedbert/model.safetensors && echo have'",
                 quiet=True)
    if code != 0:
        sh(f"{SCP} -r models/pubmedbert {USER}@{IP}:~/models/ 2>&1", 3, t=1800)
    sh(f"{SSH} 'ls -la ~/tvn_notes.parquet ~/models/pubmedbert/ | head -5' 2>&1", 6)

REMOTE = r'''
import re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
N = pd.read_parquet("tvn_notes.parquet")
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
    print(f"  {col}: {len(N):,} notes in {time.time()-t0:.0f}s "
          f"({len(N)/(time.time()-t0):.0f}/sec)")
print("done")
'''

def run_embed():
    open("_remote_embed.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_embed.py {USER}@{IP}:~/ 2>&1", 2)
    print("\nembedding on the A100 (all three arms) ...")
    sh(f"{SSH} 'cd ~ && python3 _remote_embed.py' 2>&1", 20, t=3600)

def fetch():
    print("\nfetching embeddings back ...")
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
    if not wait_ssh():
        return
    setup(); push_work(); run_embed(); fetch()
    print("\n" + "=" * 70)
    print("  embeddings are local. now run:  exec(open(pull('tvn_analyze.py')).read())")
    print("  THEN RUN  stop_all()  -- it bills at $3.70/hr while running")
    print("=" * 70)

print(__doc__)
go()
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_job | see above | remember stop_all()")

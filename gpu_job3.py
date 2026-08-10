"""
gpu_job3.py -- install torch from PyPI, not pytorch.org.

"from versions: none" means download.pytorch.org is unreachable. That matches
the enclave note: the proxy allowlist is pinned to pypi.org and
files.pythonhosted.org. pytorch.org was never on it.

But PyPI's own linux torch wheel ALREADY bundles CUDA. The +cpu build on the
bastion was somebody's choice, not a constraint. So: drop the custom index.

Also copies the bastion's pip config to the GPU box, in case that box has no
proxy settings of its own -- a likely second cause.

  diagnose()  what pip can actually see from the GPU box
  go()        fix config -> install -> embed -> fetch
  stop_all()  ALWAYS -- $3.70/hr while RUNNING
"""
import os, subprocess, glob

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

def diagnose():
    print("=" * 70); print("PIP CONFIG -- bastion (works) vs GPU box"); print("=" * 70)
    print("  bastion pip config:")
    sh("pip config list 2>&1; echo '--- files ---'; "
       "ls -la ~/.pip/pip.conf ~/.config/pip/pip.conf /etc/pip.conf 2>&1 | grep -v cannot", 10)
    print("\n  bastion proxy env:")
    sh("env | grep -i proxy || echo '    (none set)'", 6)
    print("\n  GPU box pip config:")
    sh(f"{SSH} 'pip config list 2>&1; ls -la ~/.pip/pip.conf ~/.config/pip/pip.conf "
       f"/etc/pip.conf 2>&1 | grep -v cannot' 2>&1", 10)
    print("\n  GPU box proxy env:")
    sh(f"{SSH} 'env | grep -i proxy || echo \"    (none set)\"' 2>&1", 6)
    print("\n  can the GPU box reach pypi at all?")
    sh(f"{SSH} 'curl -sS -o /dev/null -w \"pypi.org %{{http_code}}\\n\" "
       f"https://pypi.org/simple/ ; curl -sS -o /dev/null -w "
       f"\"download.pytorch.org %{{http_code}}\\n\" https://download.pytorch.org/whl/cu124/ "
       f"' 2>&1", 6)

def fix_config():
    print("\ncopying the bastion's pip config across (if any) ...")
    found = [p for p in (os.path.expanduser("~/.pip/pip.conf"),
                         os.path.expanduser("~/.config/pip/pip.conf"),
                         "/etc/pip.conf") if os.path.exists(p)]
    if found:
        sh(f"{SSH} 'mkdir -p ~/.config/pip'", quiet=True)
        sh(f"{SCP} {found[0]} {USER}@{IP}:~/.config/pip/pip.conf 2>&1", 3)
        print(f"    copied {found[0]}")
    else:
        print("    bastion has no pip.conf file; proxy must be via env vars")
    prox = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    if prox:
        exports = " ".join(f"{k}={v}" for k, v in prox.items())
        print(f"    will export: {list(prox)}")
        return exports
    return ""

def setup(envs=""):
    pre = f"{envs} " if envs else ""
    print("\ninstalling torch from PyPI (the linux wheel bundles CUDA) ...")
    sh(f"{SSH} 'test -x {VP}/python || python3 -m venv ~/venv' 2>&1", 3)
    sh(f"{SSH} '{pre}{VP}/pip install -q --upgrade pip' 2>&1", 4, t=600)
    sh(f"{SSH} '{pre}{VP}/pip install torch 2>&1 | tail -4' 2>&1", 8, t=3000)
    sh(f"{SSH} '{pre}{VP}/pip install -q sentence-transformers pandas pyarrow "
       f"2>&1 | tail -3' 2>&1", 6, t=1800)
    print("\nCUDA visible?")
    code, out = sh(f"{SSH} '{VP}/python -c \"import torch;print(torch.__version__, "
                   f"torch.cuda.is_available(), torch.cuda.get_device_name(0))\"' 2>&1", 4)
    return code == 0 and "True" in out

REMOTE = r'''
import re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
N = pd.read_parquet("tvn_notes.parquet"); print(f"{len(N):,} notes", flush=True)
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
    o = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    o.columns = [str(c) for c in o.columns]; o.to_parquet(f"tvn_emb_{col}.parquet")
    dt = time.time() - t0
    print(f"  {col}: {len(N):,} notes in {dt:.0f}s ({len(N)/dt:.0f}/sec)", flush=True)
print("done")
'''

def run_and_fetch():
    open("_remote_embed.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_embed.py {USER}@{IP}:~/ 2>&1", 2)
    print("\nembedding on the A100 ...")
    sh(f"{SSH} 'cd ~ && {VP}/python _remote_embed.py' 2>&1", 20, t=3600)
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
    envs = fix_config()
    if not setup(envs):
        print("\n  torch still unavailable -- run diagnose() and paste the output")
        return
    run_and_fetch()
    print("\n" + "=" * 70)
    print("  next:  exec(open(pull('tvn_analyze.py')).read())")
    print("  then:  stop_all()")
    print("=" * 70)

print(__doc__)
diagnose()
go()
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_job3 | torch from PyPI not pytorch.org | remember stop_all()")

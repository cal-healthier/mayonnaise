"""
gpu_start.py -- start a pre-provisioned GPU box, correctly.

This was already documented from the 2026-07-29 probe and I re-derived it the
slow way. The known facts:

  - start/stop of EXISTING instances is permitted; create is not
  - `gcloud compute ssh` FAILS -- it authenticates as the service account via
    OS Login. Use plain ssh with the key:
        ssh -i ~/.ssh/google_compute_engine calder_healthier@<internal-ip>
  - A100 a2-highgpu-1g, us-central1-f, internal IP 10.188.56.20
  - L4   g2-standard-8, us-central1-c
  - driver 610.43.02 / CUDA 13.3 already installed on them
  - stopping requires --discard-local-ssd=true (local SSD attached)

  start_l4()    ~$0.85/hr  -- ample for BERT inference
  start_a100()  ~$3.70/hr  -- only if you need the memory
  stop_all()    ALWAYS when done. Billing is per hour RUNNING, not per hour used.
"""
import os, subprocess

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER = "calder_healthier"
KEY = os.path.expanduser("~/.ssh/google_compute_engine")
L4   = ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4",   0.85)
A100 = ("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100", 3.70)

def sh(cmd, n=16, quiet=False, t=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[:n]) or "    (no output)")
    return r.returncode, out

print("=" * 74)
print("CURRENT STATE")
print("=" * 74)
sh(f"gcloud compute instances list --project={PROJ} "
   f"--format='table(name.segment(0),zone.basename(),status,"
   f"guestAccelerators[0].acceleratorType.basename():label=GPU,"
   f"networkInterfaces[0].networkIP:label=INTERNAL_IP)' 2>&1")
print(f"\n  ssh key present: {os.path.exists(KEY)}   ({KEY})")

def ip_of(name, zone):
    _, out = sh(f"gcloud compute instances describe {name} --zone={zone} "
                f"--project={PROJ} --format='value(networkInterfaces[0].networkIP)' 2>&1",
                quiet=True)
    return out.strip().splitlines()[-1] if out.strip() else None

def remote(ip, cmd, n=16):
    return sh(f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=20 "
              f"{USER}@{ip} '{cmd}' 2>&1", n)

def _start(inst):
    name, zone, gpu, cost = inst
    print(f"\nstarting {gpu} in {zone}  (~${cost:.2f}/hr while RUNNING)")
    code, _ = sh(f"gcloud compute instances start {name} --zone={zone} "
                 f"--project={PROJ} 2>&1", 8)
    if code != 0:
        print("\n  start failed. if PERMISSION_DENIED on compute.instances.start,")
        print("  that contradicts the 2026-07-29 probe and is worth re-checking.")
        return
    ip = ip_of(name, zone)
    print(f"\n  internal IP: {ip}")
    print("  waiting for sshd ...")
    sh("sleep 25", quiet=True)
    print("\n  GPU on the remote box:")
    remote(ip, "nvidia-smi --query-gpu=name,memory.total,driver_version "
               "--format=csv,noheader", 4)
    print("\n  its python/torch:")
    remote(ip, "python3 -c \"import torch;print(torch.__version__, "
               "torch.cuda.is_available())\" 2>&1 || echo 'torch not installed there'", 4)
    print(f"""
  connect with:
    ssh -i {KEY} {USER}@{ip}

  first-time setup on the box (driver + CUDA 13.3 already present):
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install sentence-transformers pandas pyarrow google-cloud-bigquery

  then copy the work over:
    scp -i {KEY} tvn_notes.parquet {USER}@{ip}:~/
    scp -i {KEY} -r models/pubmedbert {USER}@{ip}:~/models/""")

def start_l4():   _start(L4)
def start_a100(): _start(A100)

def stop_all():
    for name, zone, gpu, _ in (L4, A100):
        print(f"stopping {gpu} ...")
        sh(f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
           f"--discard-local-ssd=true --quiet 2>&1", 4)
    print("\n  (--discard-local-ssd=true is required -- these have local SSD)")

print("\n" + "=" * 74)
print("NEXT")
print("=" * 74)
print("""  start_l4()     the right choice for embedding 30k notes
  stop_all()     when finished -- do not leave it running overnight

  On an L4, that 5-hour embedding job is roughly 3-5 minutes. Which changes
  the way we should be working: stop designing experiments around what is
  cheap to compute, and just run the variants.""")
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_start | L4 + A100 provisioned and stopped | call start_l4()")

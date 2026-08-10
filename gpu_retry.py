"""
gpu_retry.py -- ZONE_RESOURCE_POOL_EXHAUSTED is capacity, not permission.

The L4 start failed because Google had no spare L4 in us-central1-c at that
moment. Permissions are fine. Capacity frees up continuously, and the A100 is
in a DIFFERENT zone with a separate pool.

  grab()            try A100 then L4, once each. usually enough.
  grab(retry=20)    keep trying both, 60s apart, until one comes free
  stop_all()        when finished (needs --discard-local-ssd=true)

For a 5-minute embedding job the A100 costs ~$0.31 and the L4 ~$0.07, so take
whichever is available rather than waiting for the cheap one.
"""
import os, subprocess, time

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER, KEY = "calder_healthier", os.path.expanduser("~/.ssh/google_compute_engine")
BOXES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100", 3.70),
         ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4",   0.85)]

def sh(cmd, n=12, quiet=False, t=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[:n]) or "    (no output)")
    return r.returncode, out

def try_start(box, quiet=True):
    name, zone, gpu, cost = box
    code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                   f"--project={PROJ} 2>&1", quiet=quiet, t=420)
    if code == 0:
        return True, ""
    if "EXHAUSTED" in out or "ZONE_RESOURCE" in out:
        return False, "no capacity"
    if "PERMISSION" in out.upper():
        return False, "PERMISSION DENIED"
    return False, out.splitlines()[0][:70] if out else "unknown"

def ready(box):
    name, zone, gpu, cost = box
    _, ip = sh(f"gcloud compute instances describe {name} --zone={zone} "
               f"--project={PROJ} --format='value(networkInterfaces[0].networkIP)' 2>&1",
               quiet=True)
    ip = ip.strip().splitlines()[-1]
    print(f"\n  {gpu} RUNNING at {ip}   (~${cost:.2f}/hr -- stop_all() when done)")
    print("  waiting for sshd ...")
    time.sleep(25)
    print("\n  the card:")
    sh(f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=25 {USER}@{ip} "
       f"'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader' 2>&1", 4)
    print(f"""
  connect:   ssh -i {KEY} {USER}@{ip}

  setup on the box (driver 610 / CUDA 13.3 already there):
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install sentence-transformers pandas pyarrow google-cloud-bigquery

  send the work over:
    scp -i {KEY} tvn_notes.parquet {USER}@{ip}:~/
    scp -i {KEY} -r models/pubmedbert {USER}@{ip}:~/models/""")
    return ip

def grab(retry=1, wait=60):
    for attempt in range(retry):
        for box in BOXES:
            name, zone, gpu, cost = box
            print(f"  trying {gpu} in {zone} ...", end=" ")
            ok, why = try_start(box)
            print("STARTED" if ok else why)
            if ok:
                return ready(box)
            if why == "PERMISSION DENIED":
                print("\n  permission problem, not capacity -- stop retrying.")
                return None
        if attempt < retry - 1:
            print(f"  both pools empty, waiting {wait}s "
                  f"(attempt {attempt+1}/{retry})\n")
            time.sleep(wait)
    print("\n  no capacity in either zone right now. Options:")
    print("    grab(retry=30)   keep trying for half an hour")
    print("    try again later  -- capacity turns over constantly")
    print("    or just run the CPU job overnight; it is free, only slow")
    return None

def stop_all():
    for name, zone, gpu, _ in BOXES:
        print(f"stopping {gpu} ...")
        sh(f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
           f"--discard-local-ssd=true --quiet 2>&1", 3)

print(__doc__)
print("=" * 72)
grab()
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_retry | capacity issue not permission | grab(retry=20) to keep trying")

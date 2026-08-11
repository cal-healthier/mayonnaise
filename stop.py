"""
stop.py -- stop every GPU box, depending on nothing already in the kernel.

The scripts exec into one shared notebook namespace, so module-level names
collide: gpu_strip.py redefined BOXES as 3-tuples while gpu_sweep.py's
stop_all() -- still bound -- unpacks 4, giving "not enough values to unpack
(expected 4, got 3)". Hardcode the two instances here and take no dependency
on anything else.

Safe to run twice; stopping a stopped instance is a no-op.
"""
import subprocess

INSTANCES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100"),
             ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4")]
PROJ = "mcp-acc-055-dbg-p-7e23"

for name, zone, gpu in INSTANCES:
    r = subprocess.run(
        f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
        f"--discard-local-ssd=true --quiet 2>&1",
        shell=True, capture_output=True, text=True, timeout=600)
    print(f"  {gpu:<6}{(r.stdout or r.stderr).strip()[:70] or 'stopped'}")

print("\n  status:")
for name, zone, gpu in INSTANCES:
    r = subprocess.run(
        f"gcloud compute instances describe {name} --zone={zone} "
        f"--project={PROJ} --format='value(status)' 2>&1",
        shell=True, capture_output=True, text=True, timeout=300)
    print(f"  {gpu:<6}{r.stdout.strip() or r.stderr.strip()[:60]}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print("stop | both instances stopped")

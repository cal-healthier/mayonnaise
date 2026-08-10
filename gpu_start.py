"""
gpu_start.py -- the GPUs exist and are stopped. Can we start one?

Found two pre-provisioned instances, both TERMINATED (stopped, not deleted --
disks persist):
    g2-standard-8 ... us-central1-c   NVIDIA L4     ~$0.85/hr
    a2-highgpu-1g ... us-central1-f   NVIDIA A100   ~$3.70/hr

The earlier "no GPU" conclusion tested compute.instances.CREATE. Starting an
existing instance is compute.instances.START -- a different permission. Worth
testing before assuming.

This REPORTS first and does not start anything. To actually start:
    start_l4()      the sensible one for BERT inference
    start_a100()    only if you need the memory or the speed
    stop_all()      always run this when finished -- they bill by the hour

Then check reachability, because a GPU we cannot log into is no use.
"""
import os, subprocess, json

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
L4 = ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "NVIDIA L4", 0.85)
A100 = ("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "NVIDIA A100", 3.70)

def sh(cmd, n=14, quiet=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    out = (r.stdout or r.stderr).strip().splitlines()
    if not quiet:
        print("\n".join("    " + l for l in out[:n]) or "    (no output)")
    return r.returncode, "\n".join(out)

print("=" * 74)
print("CURRENT STATE")
print("=" * 74)
sh(f"gcloud compute instances list --project={PROJ} "
   f"--format='table(name.segment(0):label=NAME,zone.basename(),status,"
   f"guestAccelerators[0].acceleratorType.basename():label=GPU)' 2>&1")

print("\n" + "=" * 74)
print("DO WE HAVE PERMISSION TO START THEM?")
print("=" * 74)
code, out = sh(f"gcloud compute instances describe {L4[0]} --zone={L4[1]} "
               f"--project={PROJ} --format='value(status,machineType.basename())' 2>&1",
               quiet=True)
print(f"    describe L4: {'OK' if code == 0 else 'DENIED'}  {out[:70]}")
print("\n  testing start permission via IAM (no side effects):")
sh(f"gcloud projects get-iam-policy {PROJ} --flatten='bindings[].members' "
   f"--format='value(bindings.role)' --filter='bindings.members:*' 2>&1 | "
   f"sort -u | head -12")

def _start(inst):
    name, zone, gpu, cost = inst
    print(f"\nstarting {gpu} ({name.split('-01-')[0]}) in {zone}")
    print(f"  cost ~${cost:.2f}/hour while RUNNING -- remember stop_all()")
    code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                   f"--project={PROJ} 2>&1", 8)
    if code == 0:
        print("\n  started. external/internal IP:")
        sh(f"gcloud compute instances describe {name} --zone={zone} --project={PROJ} "
           f"--format='value(networkInterfaces[0].networkIP,"
           f"networkInterfaces[0].accessConfigs[0].natIP)' 2>&1", 3)
        print("\n  can we reach it? (IAP tunnel, no external IP needed)")
        sh(f"timeout 90 gcloud compute ssh {name} --zone={zone} --project={PROJ} "
           f"--tunnel-through-iap --command='nvidia-smi; python3 -c \"import torch;"
           f"print(torch.__version__, torch.cuda.is_available())\"' 2>&1", 16)
    else:
        print("\n  start failed -- read the error above. if it is PERMISSION_DENIED")
        print("  on compute.instances.start, that is the ask for Mayo: not a new")
        print("  GPU, just permission to start machines that already exist.")

def start_l4():   _start(L4)
def start_a100(): _start(A100)
def stop_all():
    for name, zone, gpu, _ in (L4, A100):
        print(f"stopping {gpu} ...")
        sh(f"gcloud compute instances stop {name} --zone={zone} "
           f"--project={PROJ} --quiet 2>&1", 4)

print("\n" + "=" * 74)
print("NEXT")
print("=" * 74)
print("""  start_l4()    L4, ~$0.85/hr -- ample for BERT inference, the right choice
  start_a100()  A100, ~$3.70/hr -- only if you need the memory
  stop_all()    ALWAYS when finished. They bill while RUNNING, not while used.

  If start works, the plan is: install CUDA torch there, mount or copy the
  note parquet, embed, write results back. ~30k notes goes from 5 hours to
  under 5 minutes.

  If start is denied, the ask for Mayo is small and specific: permission to
  START existing instances. The machines are already provisioned and paid for
  -- nothing new is being requested.""")
print("\n" + "-" * 70)
print("FINAL LINE:")
print("gpu_start | two GPU instances found, both stopped | call start_l4()")

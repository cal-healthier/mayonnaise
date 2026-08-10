"""
find_gpu.py -- is there a GPU we can actually reach?

I previously concluded "CPU only" from torch.cuda.is_available() == False. That
was circular: the installed torch is 2.12.1+CPU, a build that reports no GPU
even when one is physically attached. So check at the SYSTEM level first.

Four places a GPU could be:
  1. attached to THIS machine, invisible to a cpu-only torch
  2. a stopped or running GCE instance with an accelerator
  3. a Vertex AI Workbench instance or custom-job quota
  4. quota that exists but cannot be instantiated (what we found before)
"""
import os, shutil, subprocess

def sh(cmd, n=16):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout or r.stderr).strip().splitlines()
        return "\n".join("    " + l for l in out[:n]) or "    (no output)"
    except Exception as e:
        return f"    {type(e).__name__}: {str(e)[:90]}"

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")

print("=" * 74)
print("1. IS A GPU ATTACHED TO THIS MACHINE?")
print("=" * 74)
print("  nvidia-smi:")
print(sh("nvidia-smi 2>&1 | head -14") if shutil.which("nvidia-smi")
      else "    nvidia-smi not on PATH")
print("\n  NVIDIA devices on the PCI bus:")
print(sh("lspci 2>/dev/null | grep -i nvidia || echo 'none found (or lspci absent)'", 6))
print("\n  /dev/nvidia* device nodes:")
print(sh("ls -la /dev/nvidia* 2>&1 | head -6"))
print("\n  driver version file:")
print(sh("cat /proc/driver/nvidia/version 2>&1 | head -3"))
print("\n  what torch thinks (remember: this build is CPU-only):")
try:
    import torch
    print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"    -> a '+cpu' build ALWAYS says False. Not evidence of absence.")
except Exception as e:
    print(f"    {type(e).__name__}")

print("\n" + "=" * 74)
print("2. GCE INSTANCES WITH ACCELERATORS")
print("=" * 74)
print(sh(f"gcloud compute instances list --project={PROJ} "
         f"--format='table(name,zone,status,machineType.basename(),"
         f"guestAccelerators[0].acceleratorType.basename(),"
         f"guestAccelerators[0].acceleratorCount)' 2>&1 | head -12"))

print("\n" + "=" * 74)
print("3. VERTEX AI WORKBENCH / NOTEBOOK INSTANCES")
print("=" * 74)
for cmd in [f"gcloud notebooks instances list --project={PROJ} --location=us-central1-a",
            f"gcloud workbench instances list --project={PROJ} --location=us-central1-a"]:
    print(f"  {cmd.split()[1]} {cmd.split()[2]}:")
    print(sh(cmd + " 2>&1 | head -6", 6))

print("\n" + "=" * 74)
print("4. QUOTA -- what are we allowed, and can we use it?")
print("=" * 74)
print(sh(f"gcloud compute regions describe us-central1 --project={PROJ} "
         f"--format='value(quotas.filter(\"metric:GPU\").list())' 2>&1 | tr ',' '\\n' "
         f"| grep -i gpu | head -10"))
print("\n  can we CREATE an instance? (this is what failed before)")
print(sh(f"gcloud compute instances create _probe_delete_me --project={PROJ} "
         f"--zone=us-central1-a --machine-type=n1-standard-1 --dry-run 2>&1 | head -5"))

print("\n" + "=" * 74)
print("5. CUSTOM TRAINING JOBS  (a GPU route that needs no instance create)")
print("=" * 74)
print("  Vertex custom jobs run on Google-managed hardware -- different")
print("  permission from compute.instances.create. Worth checking:")
print(sh(f"gcloud ai custom-jobs list --project={PROJ} --region=us-central1 2>&1 | head -6"))

print("\n" + "=" * 74)
print("WHAT TO DO WITH THE ANSWER")
print("=" * 74)
print("""  nvidia-smi shows a GPU  -> it was here all along. Reinstall torch with CUDA:
        pip install --force-reinstall torch --index-url \\
          https://download.pytorch.org/whl/cu121
      (~2GB, pip reaches PyPI). Then SentenceTransformer picks it up
      automatically and the embedding job goes from 5 hours to ~5 minutes.

  no local GPU but instances exist -> the allocation is a separate VM; we
      would move the job there rather than run it in the notebook.

  neither, and create is denied -> that matches the earlier probe: quota
      without the permission to use it. Vertex custom jobs (section 5) are the
      remaining route, since they do not require instance creation.""")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"find_gpu | nvidia_smi={'yes' if shutil.which('nvidia-smi') else 'no'} "
      f"| see sections above")

"""
hf_reach.py -- pip works. Does HuggingFace?

torch 2.12 is installed, pip pulled a wheel at 15 MB/s, and there is 453 GB
free. So the library side needs no help from Mayo at all.

The only question left: can we fetch model WEIGHTS? My notes said HF CDNs were
blocked, but that was earlier probing and PyPI being open is a surprise worth
re-checking -- proxy allowlists are usually per-host.

  1. install transformers + sentence-transformers from PyPI
  2. probe the HF hosts directly
  3. if reachable, pull a TINY model end to end and embed a sentence
  4. if not, say exactly what to request via /data/io/ingress

Nothing here needs Mayo unless step 2 fails.
"""
import subprocess, sys, os, ssl, urllib.request, importlib

def sh(cmd, n=6):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    return "\n".join("    " + l for l in (r.stdout or r.stderr).strip().splitlines()[-n:])

print("=" * 74)
print("1. INSTALL THE LIBRARIES  (pip works, so this should just go)")
print("=" * 74)
need = [m for m in ("transformers", "sentence_transformers")
        if importlib.util.find_spec(m) is None]
if need:
    pkgs = " ".join({"transformers": "transformers",
                     "sentence_transformers": "sentence-transformers"}[m] for m in need)
    print(f"  installing: {pkgs}")
    print(sh(f"{sys.executable} -m pip install --quiet --user {pkgs} 2>&1", 8))
    for m in ("transformers", "sentence_transformers"):
        importlib.invalidate_caches()
        spec = importlib.util.find_spec(m)
        print(f"  {m:<26}{'OK' if spec else 'FAILED'}")
else:
    print("  already installed")

print("\n" + "=" * 74)
print("2. ARE THE HUGGINGFACE HOSTS REACHABLE?")
print("=" * 74)
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
for url in ("https://huggingface.co/api/models/prajjwal1/bert-tiny",
            "https://cdn-lfs.huggingface.co",
            "https://cas-bridge.xethub.hf.co",
            "https://pypi.org/simple/"):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "probe"})
        with urllib.request.urlopen(req, timeout=20, context=CTX) as r:
            print(f"  {url:<52}{r.status}")
    except Exception as e:
        print(f"  {url:<52}{type(e).__name__}: {str(e)[:38]}")

print("\n" + "=" * 74)
print("3. END TO END ON A TINY MODEL  (17 MB, quick)")
print("=" * 74)
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf"))
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    mdl = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    t = tok(["Interval increase in the hepatic lesion.",
             "No evidence of metastatic disease."],
            return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = mdl(**t).last_hidden_state.mean(1)
    print(f"  WORKS. embedded 2 sentences -> {tuple(out.shape)}")
    print("  => HuggingFace is reachable. Ask Mayo for NOTHING; just download")
    print("     the model you want and go.")
    REACH = True
except Exception as e:
    REACH = False
    print(f"  {type(e).__name__}: {str(e)[:150]}")
    print("  => weights cannot be fetched directly.")

print("\n" + "=" * 74)
print("WHAT TO ASK MAYO FOR" if not REACH else "NO REQUEST NEEDED")
print("=" * 74)
if not REACH:
    print("""  Libraries are fine (pip reaches PyPI). Only the weight FILES are blocked.
  Request placement in /data/io/ingress of one model directory:

    NeuML/pubmedbert-base-embeddings      ~440 MB
      config.json
      model.safetensors
      tokenizer.json  tokenizer_config.json  vocab.txt
      special_tokens_map.json
      modules.json  1_Pooling/config.json

  Framing: read-only model weights, Apache-2.0, no network egress. Used to
  embed de-identified text that never leaves the enclave. Equivalent in
  category to the Python packages already installable from PyPI.

  Then load with: SentenceTransformer('/data/io/ingress/<dir>')""")
else:
    print("""  Nothing. Install what you need and download models directly:
    pip install --user sentence-transformers
    SentenceTransformer('NeuML/pubmedbert-base-embeddings')""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"hf_reach | hf_reachable={REACH} | pip=ok | torch=installed")

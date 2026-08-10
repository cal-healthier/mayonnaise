"""
local_model_check.py -- before asking Mayo to import weights, can we run them?

Weights are useless without a runtime. And ClinicalBERT specifically is a BERT
ENCODER, not an embedding model -- mean-pooled BERT vectors are poor for
semantic similarity because nothing trained them for it. If the goal is
clustering or retrieval, we should ask for a sentence-embedding model instead.

Checks, in the order that decides the ask:
  1. is torch / transformers / sentence-transformers installed?
  2. can pip reach PyPI (so missing pieces could be added)?
  3. is there an ingress path Mayo already uses for file transfer?
  4. how much disk is free?
Then prints exactly what to request.
"""
import os, shutil, subprocess, importlib

def sh(cmd, n=6):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90)
        return "\n".join("    " + l for l in (r.stdout or r.stderr).strip().splitlines()[:n]) \
               or "    (no output)"
    except Exception as e:
        return f"    {type(e).__name__}: {str(e)[:80]}"

print("=" * 74)
print("1. IS THERE A RUNTIME?")
print("=" * 74)
for mod in ("torch", "transformers", "sentence_transformers", "tokenizers",
            "onnxruntime", "numpy", "sklearn"):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod:<24}INSTALLED  {v}")
    except ImportError:
        print(f"  {mod:<24}--")
try:
    import torch
    print(f"\n  torch device: {'cuda' if torch.cuda.is_available() else 'cpu only'}")
except ImportError:
    pass

print("\n" + "=" * 74)
print("2. CAN PIP REACH PYPI?")
print("=" * 74)
print(sh("pip download --no-deps --dest /tmp/_piptest tokenizers 2>&1 | tail -4"))
print("\n  (if that downloaded a wheel, missing packages can be installed;")
print("   if it failed on the network, only the weights route remains)")

print("\n" + "=" * 74)
print("3. INGRESS PATH -- how does Mayo already move files in?")
print("=" * 74)
for p in ("/data/io/ingress", "/data/io", "/data/io/egress", "/data"):
    if os.path.isdir(p):
        try:
            items = sorted(os.listdir(p))[:10]
            print(f"  {p:<22}exists: {items}")
        except PermissionError:
            print(f"  {p:<22}exists, not listable")
    else:
        print(f"  {p:<22}--")

print("\n" + "=" * 74)
print("4. DISK")
print("=" * 74)
for p in ("/home", "/data", "/tmp", os.getcwd()):
    if os.path.isdir(p):
        t, u, f = shutil.disk_usage(p)
        print(f"  {p:<22}{f/1e9:>7.1f} GB free of {t/1e9:.0f} GB")

print("\n" + "=" * 74)
print("WHAT TO ASK FOR")
print("=" * 74)
print("""  If torch + transformers are present, request ONLY the model files --
  no install needed, just place them on disk:

    a sentence-embedding model (better for similarity than ClinicalBERT):
      NeuML/pubmedbert-base-embeddings        ~440 MB   biomedical, ST-trained
      pritamdeka/S-PubMedBert-MS-MARCO        ~440 MB   biomedical, retrieval
      BAAI/bge-base-en-v1.5                   ~440 MB   general, very strong

    files needed per model (all from the HF repo):
      config.json  tokenizer.json  tokenizer_config.json  vocab.txt
      special_tokens_map.json  model.safetensors  (+ 1_Pooling/config.json
      and modules.json if it is a sentence-transformers repo)

  If transformers is NOT installed and pip cannot reach PyPI, the ask has to
  include the wheels too -- which makes it a much bigger request, and TF-IDF
  locally becomes the pragmatic answer instead.

  Frame it to Mayo as: read-only model weights, no network egress, permissive
  licence, used to embed de-identified text that never leaves the enclave.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
have = [m for m in ("torch", "transformers", "sentence_transformers")
        if importlib.util.find_spec(m)]
print(f"local_model_check | runtime={','.join(have) if have else 'NONE'} "
      f"| see sections 2-3 for pip and ingress")

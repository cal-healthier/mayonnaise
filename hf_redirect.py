"""
hf_redirect.py -- huggingface.co works (200). Only the CDN hosts are blocked.

That turns the ask from "transfer 440 MB into ingress" into "allowlist one
hostname" -- and since huggingface.co itself is already permitted, the CDN
omission looks like an oversight, not a policy decision.

  1. fix the pip install (I passed --user inside a virtualenv; invalid)
  2. confirm small files come straight from huggingface.co
  3. follow the weights URL WITHOUT following redirects, to capture the exact
     hostname the download is sent to
  4. print the precise allowlist request

Aim: hand Tommy one hostname, not a project.
"""
import subprocess, sys, ssl, urllib.request, urllib.error, importlib

CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
REPO = "prajjwal1/bert-tiny"

print("=" * 74)
print("1. INSTALL (without --user this time)")
print("=" * 74)
need = [m for m in ("transformers", "sentence_transformers")
        if importlib.util.find_spec(m) is None]
if need:
    pkgs = " ".join({"transformers": "transformers",
                     "sentence_transformers": "sentence-transformers"}[m] for m in need)
    r = subprocess.run(f"{sys.executable} -m pip install --quiet {pkgs}",
                       shell=True, capture_output=True, text=True, timeout=900)
    tail = (r.stdout or r.stderr).strip().splitlines()[-5:]
    print("\n".join("    " + l for l in tail) or "    (quiet success)")
    importlib.invalidate_caches()
for m in ("transformers", "sentence_transformers"):
    print(f"  {m:<26}{'OK' if importlib.util.find_spec(m) else 'still missing'}")

print("\n" + "=" * 74)
print("2. SMALL FILES -- served directly from huggingface.co?")
print("=" * 74)
for f in ("config.json", "vocab.txt", "tokenizer_config.json"):
    url = f"https://huggingface.co/{REPO}/resolve/main/{f}"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "probe"}),
                timeout=25, context=CTX) as r:
            body = r.read(80)
            print(f"  {f:<24}{r.status}  {len(body)}+ bytes  from {r.url.split('/')[2]}")
    except Exception as e:
        print(f"  {f:<24}{type(e).__name__}: {str(e)[:50]}")

print("\n" + "=" * 74)
print("3. WHERE DOES THE WEIGHTS FILE REDIRECT TO?")
print("=" * 74)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"REDIRECT->{newurl}",
                                     headers, fp)
opener = urllib.request.build_opener(NoRedirect,
                                     urllib.request.HTTPSHandler(context=CTX))
targets = set()
for f in ("pytorch_model.bin", "model.safetensors"):
    url = f"https://huggingface.co/{REPO}/resolve/main/{f}"
    try:
        r = opener.open(urllib.request.Request(url, headers={"User-Agent": "probe"}),
                        timeout=25)
        print(f"  {f:<22}{r.status} served directly (no redirect)")
    except urllib.error.HTTPError as e:
        if "REDIRECT->" in str(e.reason):
            tgt = str(e.reason).split("REDIRECT->")[1]
            host = tgt.split("/")[2]
            targets.add(host)
            print(f"  {f:<22}{e.code} -> {host}")
        else:
            print(f"  {f:<22}HTTP {e.code}: {str(e.reason)[:50]}")
    except Exception as e:
        print(f"  {f:<22}{type(e).__name__}: {str(e)[:50]}")

print("\n  hosts the download is handed off to:")
for h in sorted(targets) or ["(none captured)"]:
    ok = "?"
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{h}",
                               headers={"User-Agent": "probe"}), timeout=15, context=CTX)
        ok = "REACHABLE"
    except urllib.error.HTTPError as e:
        ok = f"HTTP {e.code} (host resolves)"
    except Exception as e:
        ok = f"{type(e).__name__}"
    print(f"    {h:<40}{ok}")

print("\n" + "=" * 74)
print("THE ASK")
print("=" * 74)
print(f"""  huggingface.co is ALREADY permitted (200 on the API and on small files).
  Only the large-file CDN is not. Request proxy allowlisting of:

{chr(10).join('      ' + h for h in sorted(targets)) if targets else '      cdn-lfs.huggingface.co / cdn-lfs-us-1.hf.co / cas-bridge.xethub.hf.co'}

  Framing: huggingface.co is already reachable from the enclave; model weight
  files are served from a separate CDN hostname which appears to have been
  missed. Read-only downloads of Apache-2.0 model files. No egress -- data
  never leaves; only weights come in.

  FALLBACK if refused: place one model directory in /data/io/ingress
  (~440 MB, NeuML/pubmedbert-base-embeddings), then
    SentenceTransformer('/data/io/ingress/<dir>')""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"hf_redirect | transformers={'ok' if importlib.util.find_spec('transformers') else 'missing'} "
      f"| cdn_hosts={','.join(sorted(targets)) if targets else 'none_captured'}")

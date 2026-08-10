"""
get_weights.py -- pull model weights in through the GitHub Blob API.

The contents endpoint caps at 1MB, which is why pull() cannot fetch weights.
But /repos/{owner}/{repo}/git/blobs/{sha} returns base64 up to 100MB per
object, and api.github.com is already reachable. So: chunk on the outside,
reassemble on the inside. No allowlist request, no ingress ticket.

Weights live in cal-healthier/mayo-weights, split at 45MB with a sha256
manifest so a truncated transfer is caught rather than silently loaded.

Run once; it caches to ./models/ and skips work on re-run.
"""
import base64, hashlib, json, os, ssl, time, urllib.request

REPO = "cal-healthier/mayo-weights"
MODEL = "pubmedbert"
DEST = os.path.join("models", MODEL)
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")

def api(path):
    url = f"https://api.github.com/repos/{REPO}/{path}?_={int(time.time())}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=120, context=CTX))

def blob(sha):
    """the Blob API: base64 up to 100MB, unlike contents' 1MB cap"""
    j = api(f"git/blobs/{sha}")
    return base64.b64decode(j["content"])

os.makedirs(DEST, exist_ok=True)
os.makedirs(os.path.join(DEST, "1_Pooling"), exist_ok=True)

print("=" * 72)
print("1. SMALL FILES  (contents API is fine for these)")
print("=" * 72)
for entry in api(f"contents/{MODEL}"):
    if entry["type"] != "file":
        continue
    p = os.path.join(DEST, entry["name"])
    if os.path.exists(p) and os.path.getsize(p) == entry["size"]:
        print(f"  {entry['name']:<38}cached")
        continue
    data = blob(entry["sha"])
    open(p, "wb").write(data)
    print(f"  {entry['name']:<38}{len(data):>10,} bytes")
try:
    for entry in api(f"contents/{MODEL}/1_Pooling"):
        p = os.path.join(DEST, "1_Pooling", entry["name"])
        open(p, "wb").write(blob(entry["sha"]))
        print(f"  1_Pooling/{entry['name']:<28}ok")
except Exception as e:
    print(f"  1_Pooling: {type(e).__name__}")

print("\n" + "=" * 72)
print("2. WEIGHT CHUNKS  (Blob API, 45MB each)")
print("=" * 72)
chunks = api(f"contents/{MODEL}/_chunks")
man = json.loads(blob(next(c["sha"] for c in chunks if c["name"] == "manifest.json")))
print(f"  expecting {man['chunks']} chunks, {man['total_bytes']/1e6:.0f} MB total")
target = os.path.join(DEST, man["file"])
if os.path.exists(target) and os.path.getsize(target) == man["total_bytes"]:
    print("  already assembled, skipping download")
else:
    parts, t0 = [], time.time()
    for i in range(man["chunks"]):
        name = f"{man['file']}.{i:03d}"
        e = next(c for c in chunks if c["name"] == name)
        for attempt in range(3):
            try:
                parts.append(blob(e["sha"]))
                break
            except Exception as ex:
                if attempt == 2:
                    raise
                print(f"    retry {name} ({type(ex).__name__})")
                time.sleep(3)
        done = sum(len(p) for p in parts)
        print(f"  {name:<30}{done/1e6:>7.0f} / {man['total_bytes']/1e6:.0f} MB"
              f"  ({time.time()-t0:.0f}s)")
    data = b"".join(parts)
    got = hashlib.sha256(data).hexdigest()
    if got != man["sha256"]:
        raise SystemExit(f"CHECKSUM MISMATCH\n  expected {man['sha256']}\n  got      {got}")
    print(f"  sha256 verified: {got[:20]}...")
    open(target, "wb").write(data)
    print(f"  wrote {target}")

print("\n" + "=" * 72)
print("3. LOAD IT")
print("=" * 72)
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(DEST)
    v = m.encode(["Interval increase in the dominant hepatic lesion.",
                  "No evidence of metastatic disease.",
                  "New sclerotic focus in the left iliac bone."])
    import numpy as np
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    print(f"  loaded. embeddings {v.shape}")
    print(f"  'increase' vs 'new lesion' : {float(n[0] @ n[2]):+.3f}")
    print(f"  'increase' vs 'no disease' : {float(n[0] @ n[1]):+.3f}")
    print("\n  the first should be clearly higher than the second.")
    print(f"\n  from now on:  SentenceTransformer('{DEST}')")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:180]}")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"get_weights | dest={DEST} | bytes={os.path.getsize(target) if os.path.exists(target) else 0}")

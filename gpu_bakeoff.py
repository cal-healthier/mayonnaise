"""
gpu_bakeoff.py -- run the encoder bake-off on the A100 instead of the CPU.

Embedding is the only slow part and it is exactly what a GPU is for. Four
encoders over 4k notes is ~30 min on the bastion CPU and well under a minute
on the A100.

End to end this is more like 5 minutes than instant, and the embedding is not
what costs the time:

    start the instance          60-120s
    fetch 3 new models          one time, cached after
    scp models + notes across   ~1 min, internal VPC
    EMBED ALL FOUR              ~30s          <- the part we came for
    copy embeddings back        seconds
    stop the instance           immediately, before any scoring

Scoring stays on the bastion: it is 712 patients by 768 dims, seconds of work,
and there is no reason to pay $3.70/hr to fit gradient boosting.

Writes bo_emb_<name>.parquet, which bakeoff.py then finds already cached, so
the two run back to back and bakeoff.py never touches a GPU or an encoder.

  go()          everything, then stops the box for you
  stop_all()    the panic button. ALWAYS safe to call twice.

Set COHORT = "prostate" to do the confirmatory run on the big cohort later --
25k notes, which is 2h on CPU and still under a minute here.
"""
import base64, hashlib, json, os, re, ssl, subprocess, time, urllib.request

COHORT = "ovarian"
NOTES = {"ovarian": "ovt_notes.parquet", "prostate": "tvn_notes.parquet"}[COHORT]
MODELS = [("pubmedbert", ""), ("biolord", ""), ("bge", ""), ("e5", "passage: ")]

WREPO = "cal-healthier/mayo-weights"
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER = "calder_healthier"
KEY = os.path.expanduser("~/.ssh/google_compute_engine")
VP = "~/venv/bin"
BOXES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100", 3.70),
         ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4",   0.85)]


def sh(cmd, n=12, quiet=False, t=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out


# ------------------------------------------------------------------ weights
def api(path):
    url = f"https://api.github.com/repos/{WREPO}/{path}?_={int(time.time())}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=180, context=CTX))


def blob(sha):
    return base64.b64decode(api(f"git/blobs/{sha}")["content"])


def fetch(name):
    dest = os.path.join("models", name)
    if os.path.exists(os.path.join(dest, "model.safetensors")):
        print(f"    {name:<12}already local")
        return
    t0 = time.time()
    os.makedirs(dest, exist_ok=True)
    for e in api(f"contents/{name}"):
        if e["type"] == "file":
            open(os.path.join(dest, e["name"]), "wb").write(blob(e["sha"]))
        elif e["name"] != "_chunks":
            sub = os.path.join(dest, e["name"])
            os.makedirs(sub, exist_ok=True)
            for f in api(f"contents/{name}/{e['name']}"):
                open(os.path.join(sub, f["name"]), "wb").write(blob(f["sha"]))
    ch = api(f"contents/{name}/_chunks")
    man = json.loads(blob(next(c["sha"] for c in ch if c["name"] == "manifest.json")))
    parts = []
    for i in range(man["chunks"]):
        e = next(c for c in ch if c["name"] == f"{man['file']}.{i:03d}")
        for a in range(3):
            try:
                parts.append(blob(e["sha"])); break
            except Exception:
                if a == 2: raise
                time.sleep(3)
    data = b"".join(parts)
    got = hashlib.sha256(data).hexdigest()
    if got != man["sha256"]:
        raise SystemExit(f"{name}: CHECKSUM MISMATCH {got[:16]} != {man['sha256'][:16]}")
    open(os.path.join(dest, man["file"]), "wb").write(data)
    print(f"    {name:<12}{len(data)/1e6:.0f} MB verified ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------- gpu
def try_start(box):
    name, zone, gpu, _ = box
    code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                   f"--project={PROJ} 2>&1", quiet=True, t=420)
    if code == 0:
        return True, ""
    if "EXHAUSTED" in out or "ZONE_RESOURCE" in out:
        return False, "no capacity"
    if "PERMISSION" in out.upper():
        return False, "PERMISSION DENIED"
    return False, (out.splitlines()[0][:70] if out else "unknown")


def grab(retry=1, wait=60):
    for attempt in range(retry):
        for box in BOXES:
            name, zone, gpu, cost = box
            print(f"  trying {gpu} in {zone} ...", end=" ")
            ok, why = try_start(box)
            print("STARTED" if ok else why)
            if ok:
                _, ip = sh(f"gcloud compute instances describe {name} --zone={zone} "
                           f"--project={PROJ} "
                           f"--format='value(networkInterfaces[0].networkIP)' 2>&1",
                           quiet=True)
                ip = ip.strip().splitlines()[-1]
                print(f"  {gpu} at {ip}  (${cost:.2f}/hr -- stopped automatically below)")
                print("  waiting for sshd ...")
                time.sleep(25)
                return ip
            if why == "PERMISSION DENIED":
                return None
        if attempt < retry - 1:
            print(f"  both pools empty, waiting {wait}s ({attempt+1}/{retry})")
            time.sleep(wait)
    return None


def stop_all():
    for name, zone, gpu, _ in BOXES:
        sh(f"gcloud compute instances stop {name} --zone={zone} --project={PROJ} "
           f"--discard-local-ssd=true --quiet 2>&1", 2, quiet=True)
    print("  both instances stopped. billing ends.")


REMOTE = '''
import os, re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
N = pd.read_parquet("%s")
N["nonum"] = N["txt"].str.replace(re.compile(r"[0-9]+(\\.[0-9]+)?"), " ", regex=True)
print(f"{len(N):,} notes", flush=True)
for name, prefix in %s:
    out = f"bo_emb_{name}.parquet"
    if os.path.exists(out):
        print(f"  {name}: cached", flush=True); continue
    m = SentenceTransformer(f"models/{name}", device="cuda")
    m.max_seq_length = 256
    txt = (prefix + N["nonum"]).tolist() if prefix else N["nonum"].tolist()
    t0 = time.time()
    V = m.encode(txt, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    o = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    o.columns = [str(c) for c in o.columns]
    o.to_parquet(out)
    dt = time.time() - t0
    print(f"  {name}: {len(N):,} notes in {dt:.1f}s ({len(N)/dt:.0f}/sec) dim {V.shape[1]}",
          flush=True)
    del m; torch.cuda.empty_cache()
print("done")
''' % (NOTES, repr(MODELS))


def go():
    print("=" * 78)
    print(f"GPU BAKE-OFF   cohort={COHORT}   notes={NOTES}")
    print("=" * 78)

    print("\n1. weights on the bastion")
    for name, _ in MODELS:
        fetch(name)

    print("\n2. starting a GPU")
    ip = grab()
    if not ip:
        print("""
  No GPU available. Nothing is broken -- capacity turns over constantly.
    retry:  go()  again in a few minutes, or grab(retry=20) to keep trying
    or:     just run bakeoff.py, which does the same thing on CPU in ~30 min""")
        return False

    SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=25 {USER}@{ip}"
    SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"

    print("\n3. environment on the box")
    code, out = sh(f"{SSH} '{VP}/python -c \"import torch,sentence_transformers;"
                   f"print(torch.cuda.is_available())\"' 2>&1", 3, quiet=True)
    if code != 0 or "True" not in out:
        print("    installing (first run on this disk) ...")
        sh(f"{SSH} 'test -x {VP}/python || python3 -m venv ~/venv'", 2, quiet=True)
        sh(f"{SSH} '{VP}/pip install -q --upgrade pip'", 2, quiet=True, t=600)
        sh(f"{SSH} '{VP}/pip install torch 2>&1 | tail -2'", 4, t=3000)
        sh(f"{SSH} '{VP}/pip install -q sentence-transformers pandas pyarrow "
           f"2>&1 | tail -2'", 4, t=1800)
        code, out = sh(f"{SSH} '{VP}/python -c \"import torch;"
                       f"print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
                       f" 2>&1", 3)
        if "True" not in out:
            print("\n  torch cannot see the card. stopping the box so it does not bill.")
            stop_all()
            return False
    else:
        print("    venv already good, torch sees the card")

    print("\n4. sending models and notes across")
    sh(f"{SSH} 'mkdir -p ~/models'", 2, quiet=True)
    for name, _ in MODELS:
        _, have = sh(f"{SSH} 'test -f ~/models/{name}/model.safetensors && echo YES'",
                     2, quiet=True)
        if "YES" in have:
            print(f"    {name:<12}already on the box")
            continue
        t0 = time.time()
        sh(f"{SCP} -r models/{name} {USER}@{ip}:~/models/ 2>&1", 2, quiet=True, t=1800)
        print(f"    {name:<12}copied ({time.time()-t0:.0f}s)")
    sh(f"{SCP} {NOTES} {USER}@{ip}:~/ 2>&1", 2, quiet=True, t=900)
    print(f"    {NOTES} copied")

    print("\n5. embedding on the card")
    open("_remote_bakeoff.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_bakeoff.py {USER}@{ip}:~/ 2>&1", 2, quiet=True)
    code, out = sh(f"{SSH} 'cd ~ && {VP}/python _remote_bakeoff.py' 2>&1", 20, t=3600)

    print("\n6. copying embeddings back")
    got = 0
    for name, _ in MODELS:
        c, _o = sh(f"{SCP} {USER}@{ip}:~/bo_emb_{name}.parquet . 2>&1", 2,
                   quiet=True, t=900)
        got += (c == 0 and os.path.exists(f"bo_emb_{name}.parquet"))
    print(f"    {got}/{len(MODELS)} embedding files retrieved")

    print("\n7. stopping the GPU before any scoring")
    stop_all()

    if got < len(MODELS):
        print("\n  Some embeddings are missing -- the remote log above says why.")
        print("  bakeoff.py will fall back to CPU for whichever are absent.")
        return False
    print(f"""
{'=' * 78}
  Embeddings ready and the box is off. bakeoff.py will now find all four
  cached and go straight to scoring -- no GPU, no encoder, about a minute.
{'=' * 78}""")
    return True


ok = go()
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"gpu_bakeoff | cohort={COHORT} | embeddings_ready={ok} | gpu=stopped")

"""
bakeoff.py -- which sentence encoder should we be using?

The 0.842 text result came from NeuML/pubmedbert-base-embeddings, chosen
because it was biomedical and it was there. That is not a reason. There is no
settled answer in the literature either, because the model classes do not
overlap cleanly:

  clinical models   Bio_ClinicalBERT, GatorTron -- trained on real notes but
                    never trained for similarity, so the vectors are not
                    built for distance
  embedding models  BGE, E5 -- excellent vectors, general-domain text
  biomedical embed  PubMedBERT-embeddings, BioLORD -- trained for similarity,
                    but on LITERATURE, and our input is clinical notes

and there is a live finding that cuts against domain specialisation
altogether: general encoders have become strong enough to beat domain-specific
ones on specialist text. So this is an empirical question, and we now have a
real benchmark to settle it with.

Four encoders, all 12-layer / 768-dim base models so capacity is held constant:

  pubmedbert   NeuML/pubmedbert-base-embeddings   incumbent, PubMed abstracts
  biolord      FremyCompany/BioLORD-2023          biomedical concept similarity
  bge          BAAI/bge-base-en-v1.5              general, top-tier
  e5           intfloat/e5-base-v2                general, different recipe

RUN ON OVARIAN, DELIBERATELY. Prostate is where the headline number lives, and
picking the encoder on the cohort you report is a quiet form of overfitting --
with four encoders you would expect the best of four to look good by chance
alone. Choosing on ovarian and then applying the winner to prostate keeps the
headline cohort clean. It also happens to be 4k notes rather than 25k.

~7 min per encoder on CPU, so roughly half an hour total. Embeddings cache, so
a re-run is free.
"""
import base64, hashlib, json, os, re, ssl, time, urllib.request
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(os.cpu_count())
WREPO = "cal-healthier/mayo-weights"
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
rng = np.random.default_rng(0)

# e5 was trained with prefixes and degrades measurably without them; bge's
# instruction is for retrieval queries only, so plain text is correct here
MODELS = [("pubmedbert", ""), ("biolord", ""), ("bge", ""), ("e5", "passage: ")]


def api(path):
    url = f"https://api.github.com/repos/{WREPO}/{path}?_={int(time.time())}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=180, context=CTX))


def blob(sha):
    return base64.b64decode(api(f"git/blobs/{sha}")["content"])


def fetch(name):
    """pull one model in through the Blob API; skips everything already cached"""
    dest = os.path.join("models", name)
    if os.path.exists(os.path.join(dest, "model.safetensors")):
        return dest
    print(f"  fetching {name} ...")
    os.makedirs(dest, exist_ok=True)
    for e in api(f"contents/{name}"):
        if e["type"] == "file":
            open(os.path.join(dest, e["name"]), "wb").write(blob(e["sha"]))
        elif e["name"] != "_chunks":                       # 1_Pooling, 2_Normalize
            sub = os.path.join(dest, e["name"])
            os.makedirs(sub, exist_ok=True)
            for f in api(f"contents/{name}/{e['name']}"):
                open(os.path.join(sub, f["name"]), "wb").write(blob(f["sha"]))
    ch = api(f"contents/{name}/_chunks")
    man = json.loads(blob(next(c["sha"] for c in ch if c["name"] == "manifest.json")))
    parts, t0 = [], time.time()
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
    print(f"    {len(data)/1e6:.0f} MB verified in {time.time()-t0:.0f}s")
    return dest


# ------------------------------------------------------------------- cohort
E = pd.read_parquet("ov_label.parquet")
cv = pd.read_parquet("ov_ca125.parquet")
E["tx_date"] = pd.to_datetime(cv.groupby("clinic")["tx_date"].first().reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
N = pd.read_parquet("ovt_notes.parquet")
E = E[E.index.isin(N["clinic"])]
N["nonum"] = N["txt"].str.replace(re.compile(r"[0-9]+(\.[0-9]+)?"), " ", regex=True)

labs = pd.read_parquet("ov_labs.parquet")
cols = [c for c in labs.columns if c.endswith("__last")]
S = labs[cols].reindex(E.index); S["baseline_ca125"] = E["baseline"]
S = S.loc[:, S.notna().sum() > len(S) * .3]
PROX = (pd.read_parquet("prox_ovarian.parquet").reindex(E.index).fillna(0)
        if os.path.exists("prox_ovarian.parquet") else None)
y = E["y"]

print("=" * 78)
print(f"ENCODER BAKE-OFF   ovarian, {len(E):,} women, {int(y.sum())} events, "
      f"{len(N):,} notes")
print("=" * 78)


def embed(name, prefix):
    cache = f"bo_emb_{name}.parquet"
    if os.path.exists(cache):
        print(f"  {name:<12}cached")
        return pd.read_parquet(cache).reindex(E.index)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(fetch(name))
    m.max_seq_length = 256
    t0 = time.time()
    txt = (prefix + N["nonum"]).tolist() if prefix else N["nonum"].tolist()
    V = m.encode(txt, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    o = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    o.columns = [str(c) for c in o.columns]
    o.to_parquet(cache)
    print(f"  {name:<12}{len(N):,} notes in {time.time()-t0:.0f}s "
          f"({len(N)/(time.time()-t0):.0f}/sec)   dim {V.shape[1]}")
    del m
    return o.reindex(E.index)


def oof(X, repeats=3):
    X = pd.DataFrame(X).reindex(E.index)
    acc = np.zeros(len(X))
    for r in range(repeats):
        p = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            mm = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                min_samples_leaf=25, l2_regularization=1.,
                random_state=0).fit(X.iloc[tr], y.iloc[tr])
            p[te] = mm.predict_proba(X.iloc[te])[:, 1]
        acc += p
    return acc / repeats


def paired(pa, pb, boot=2000):
    yy = np.asarray(y)
    obs = roc_auc_score(yy, pa) - roc_auc_score(yy, pb)
    n, d = len(yy), []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        if yy[i].sum() in (0, len(i)):
            continue
        d.append(roc_auc_score(yy[i], pa[i]) - roc_auc_score(yy[i], pb[i]))
    d = np.array(d)
    return obs, *np.percentile(d, [2.5, 97.5])


print("\nembedding")
V = {n: embed(n, p) for n, p in MODELS}

print("\nscoring (text-only arm, identical folds)")
P = {n: oof(V[n]) for n, _ in MODELS}
A = {n: roc_auc_score(y, P[n]) for n, _ in MODELS}

print("\n" + "=" * 78)
print("TEXT-ONLY AUC        (vs incumbent, paired bootstrap over patients)")
print("=" * 78)
for n in sorted(A, key=A.get, reverse=True):
    if n == "pubmedbert":
        print(f"  {n:<14}{A[n]:.3f}     (incumbent)")
    else:
        o, lo, hi = paired(P[n], P["pubmedbert"])
        v = "BETTER" if lo > 0 else ("worse" if hi < 0 else "tie")
        print(f"  {n:<14}{A[n]:.3f}   {o:+.3f} [{lo:+.3f}, {hi:+.3f}]  {v}")

best = max(A, key=A.get)
if PROX is not None:
    print("\n" + "=" * 78)
    print("AND IN THE FULL MODEL  (structured + proxies + text)")
    print("=" * 78)
    base = oof(pd.concat([S, PROX], axis=1))
    print(f"  structured + proxies, no text      {roc_auc_score(y, base):.3f}")
    for n in ("pubmedbert", best):
        full = oof(pd.concat([S, PROX, V[n].add_prefix("t")], axis=1))
        o, lo, hi = paired(full, base)
        print(f"  + text from {n:<22}{roc_auc_score(y, full):.3f}   "
              f"{o:+.3f} [{lo:+.3f}, {hi:+.3f}]")
else:
    print("\n  (run paired.py first to cache prox_ovarian.parquet for the full model)")

print(f"""
{'=' * 78}
READING IT
{'=' * 78}
  If a general-domain encoder wins, that is a result in itself -- it says the
  useful signal in these notes is ordinary language structure, not biomedical
  vocabulary, and it points at fine-tuning on Mayo's own text as the thing
  that would actually move the number.

  If the biomedical ones win, domain pretraining is carrying weight and a
  clinical-domain base model is the next thing to try.

  Either way the winner then goes to PROSTATE as a confirmatory run, on a
  cohort no encoder was selected on.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print("bakeoff | " + " | ".join(f"{n}={A[n]:.3f}" for n, _ in MODELS) +
      f" | best={best}")

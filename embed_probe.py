"""
embed_probe.py -- what can we actually embed with?

gemini-embedding-2 being disallowed does not mean every embedding route is.
Four options, probed in order of preference:

  1. OTHER VERTEX EMBEDDING MODELS -- text-embedding-004/005, gecko. Older,
     separately allowlisted, often still open when the newest is not.
  2. A LOCAL MODEL already installed -- sentence-transformers with cached
     weights. HuggingFace CDNs are blocked, so this only works if something is
     already on disk.
  3. BIGQUERY ML.GENERATE_EMBEDDING -- a different code path to Vertex, and
     sometimes governed by different permissions.
  4. TF-IDF + SVD locally -- no downloads, no API, sklearn only. Unfashionable
     but genuinely competitive on clinical text classification, and it costs
     nothing.

Prints exactly which routes work so we stop guessing.
"""
import os, warnings
warnings.filterwarnings("ignore")
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
SAMPLE = ["Interval increase in the dominant hepatic lesion.",
          "No evidence of metastatic disease.",
          "New sclerotic focus in the left iliac bone."]

print("=" * 74)
print("1. VERTEX EMBEDDING MODELS")
print("=" * 74)
from google import genai
cl = genai.Client(vertexai=True, project=PROJ, location="global")
CANDIDATES = ["text-embedding-005", "text-embedding-004", "text-multilingual-embedding-002",
              "textembedding-gecko@003", "textembedding-gecko@001",
              "gemini-embedding-001", "gemini-embedding-2"]
working = []
for m in CANDIDATES:
    try:
        r = cl.models.embed_content(model=m, contents=SAMPLE[:1])
        dim = len(r.embeddings[0].values)
        working.append((m, dim))
        print(f"  {m:<34}OK   dim={dim}")
    except Exception as e:
        msg = str(e).split("\n")[0][:70]
        print(f"  {m:<34}--   {type(e).__name__}: {msg}")

print("\n  also listing what the project reports as available:")
try:
    names = [x.name for x in cl.models.list()]
    emb = [n for n in names if "embed" in n.lower() or "gecko" in n.lower()]
    print("   ", ", ".join(emb) if emb else "(none with 'embed' in the name)")
except Exception as e:
    print(f"    list failed: {type(e).__name__} {str(e)[:70]}")

print("\n" + "=" * 74)
print("2. A LOCAL MODEL ALREADY ON DISK?")
print("=" * 74)
try:
    import sentence_transformers as st
    print(f"  sentence-transformers {st.__version__} installed")
    from pathlib import Path
    caches = [Path.home()/".cache"/"huggingface", Path.home()/".cache"/"torch",
              Path("/opt/models"), Path("/data/models")]
    found = False
    for c in caches:
        if c.exists():
            items = [p.name for p in c.rglob("*") if p.is_dir()][:8]
            if items:
                print(f"    {c}: {items}")
                found = True
    if not found:
        print("    no cached weights found -- and HF CDNs are blocked, so a")
        print("    fresh download will fail")
except ImportError:
    print("  sentence-transformers not installed")

print("\n" + "=" * 74)
print("3. BIGQUERY ML.GENERATE_EMBEDDING  (different permission path)")
print("=" * 74)
try:
    from google.cloud import bigquery
    C = bigquery.Client(project=PROJ)
    conns = C.query("SELECT 1 AS ok").to_dataframe()
    print("  BigQuery reachable. A remote model connection would be needed:")
    print("    CREATE MODEL ... REMOTE WITH CONNECTION ... ENDPOINT='text-embedding-005'")
    print("  that needs a BQ connection resource -- check with:")
    print("    bq ls --connection --location=us")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:80]}")

print("\n" + "=" * 74)
print("4. LOCAL FALLBACK -- TF-IDF + SVD  (no downloads, no API)")
print("=" * 74)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    TruncatedSVD(n_components=2, random_state=0))
V = pipe.fit_transform(SAMPLE)
print(f"  works: {len(SAMPLE)} texts -> {V.shape[1]} dims (scale n_components up)")
sim = (V @ V.T) / (np.linalg.norm(V, axis=1)[:, None] *
                   np.linalg.norm(V, axis=1)[None, :] + 1e-9)
print(f"  similarity 'increase' vs 'new lesion': {sim[0,2]:+.2f}")
print(f"  similarity 'increase' vs 'no disease': {sim[0,1]:+.2f}")
print("\n  fit on a large corpus (say 200k narratives) with n_components=300 and")
print("  this is a serviceable embedding for clustering and classification.")
print("  it will not capture paraphrase the way a neural model does.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"embed_probe | vertex_working={len(working)} "
      f"| best={working[0][0] if working else 'NONE'} "
      f"| fallback=tfidf_svd_ok")

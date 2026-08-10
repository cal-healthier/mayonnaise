"""
embed_eval.py -- is the model actually working? My first check was a bad test.

I compared "interval increase in the dominant HEPATIC lesion" with "new
sclerotic focus in the left ILIAC BONE" and expected high similarity. Those
mean the same thing clinically but share no words and describe different
organs -- that is a semantic inference test, not a sanity check. Meanwhile
"no evidence of metastatic disease" shares vocabulary with both. 0.293 vs
0.216 tells us very little.

Proper evaluation:
  1. did the pooling layer load? (if not, ST falls back to plain mean pooling
     and you get exactly the badly-behaved geometry sentence-transformers exist
     to fix)
  2. PARAPHRASES -- same meaning, different words. should score HIGH.
  3. CONTRADICTIONS -- opposite meaning, shared vocabulary. should score LOW.
  4. UNRELATED -- different topic entirely. should score LOWEST.
  5. do those three distributions separate?

Local, free, no API calls.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

DEST = "models/pubmedbert"
m = SentenceTransformer(DEST)

print("=" * 74)
print("1. DID THE POOLING LAYER LOAD?")
print("=" * 74)
for i, mod in enumerate(m):
    print(f"  [{i}] {type(mod).__name__}")
    if type(mod).__name__ == "Pooling":
        cfg = {k: v for k, v in vars(mod).items()
               if k.startswith("pooling_mode") and v}
        print(f"      {cfg}")
print(f"  max_seq_length {m.max_seq_length} | dim {m.get_sentence_embedding_dimension()}")
print("\n  a Pooling module must be present. if the list is just Transformer,")
print("  the sentence-transformer config did not load and similarities will be poor.")

PARA = [  # same meaning, deliberately different wording
 ("The hepatic lesion has increased in size.",
  "Interval enlargement of the liver lesion."),
 ("No evidence of metastatic disease.",
  "There is no sign of metastasis."),
 ("New sclerotic focus in the left iliac bone.",
  "A new bone lesion is seen in the left ilium."),
 ("The patient has a history of seizures.",
  "Seizure disorder is documented in the past medical history."),
 ("PSA has risen since the prior study.",
  "Interval increase in prostate specific antigen."),
]
CONTRA = [  # opposite meaning, overlapping vocabulary -- the hard case
 ("The hepatic lesion has increased in size.",
  "The hepatic lesion has decreased in size."),
 ("No evidence of metastatic disease.",
  "Extensive metastatic disease is present."),
 ("Disease is stable compared to prior imaging.",
  "Disease has progressed compared to prior imaging."),
 ("The patient tolerated treatment well.",
  "The patient did not tolerate treatment."),
 ("PSA has risen since the prior study.",
  "PSA has fallen since the prior study."),
]
UNREL = [
 ("The hepatic lesion has increased in size.",
  "Blood pressure was measured in the left arm."),
 ("No evidence of metastatic disease.",
  "The patient was advised on sun protection."),
 ("New sclerotic focus in the left iliac bone.",
  "Follow-up appointment scheduled in three months."),
 ("PSA has risen since the prior study.",
  "The patient reports improved appetite."),
 ("Disease is stable compared to prior imaging.",
  "Vaccination history was reviewed."),
]

def sims(pairs):
    a = m.encode([p[0] for p in pairs], normalize_embeddings=True)
    b = m.encode([p[1] for p in pairs], normalize_embeddings=True)
    return (a * b).sum(1)

print("\n" + "=" * 74)
print("2-4. DO THE DISTRIBUTIONS SEPARATE?")
print("=" * 74)
res = {}
for name, pairs in [("PARAPHRASE (want high)", PARA),
                    ("CONTRADICTION (want low)", CONTRA),
                    ("UNRELATED (want lowest)", UNREL)]:
    s = sims(pairs)
    res[name.split()[0]] = s
    print(f"\n  {name}   mean {s.mean():+.3f}   range {s.min():+.3f} to {s.max():+.3f}")
    for (x, y), v in zip(pairs, s):
        print(f"    {v:+.3f}  {x[:34]:<36} | {y[:34]}")

p, c, u = res["PARAPHRASE"], res["CONTRADICTION"], res["UNRELATED"]
print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  paraphrase - unrelated     {p.mean()-u.mean():+.3f}   "
      f"{'GOOD' if p.mean()-u.mean() > 0.25 else 'WEAK'}")
print(f"  paraphrase - contradiction {p.mean()-c.mean():+.3f}   "
      f"{'GOOD' if p.mean()-c.mean() > 0.10 else 'WEAK (hard even for good models)'}")
print(f"  contradiction - unrelated  {c.mean()-u.mean():+.3f}   "
      f"(positive is expected: shared vocabulary)")
print("""
  The paraphrase-vs-unrelated gap is the one that matters. Contradictions
  scoring high is normal -- "increased" and "decreased" appear in near-identical
  sentences, and embedding models are famously poor at negation. That is a known
  limitation, not a broken download: do not use cosine similarity alone to tell
  progression from improvement. Use the LLM for polarity, embeddings for
  topic and retrieval.""")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"embed_eval | para={p.mean():.3f} | contra={c.mean():.3f} | unrel={u.mean():.3f} "
      f"| para_minus_unrel={p.mean()-u.mean():+.3f} "
      f"| pooling={'yes' if any(type(x).__name__=='Pooling' for x in m) else 'NO'}")

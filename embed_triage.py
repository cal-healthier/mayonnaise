"""
embed_triage.py -- use the free tool to cut the expensive tool's workload.

The evaluation showed exactly what embeddings can and cannot do here:
  paraphrase vs unrelated  +0.564   -> excellent at TOPIC
  contradiction            0.832    -> blind to POLARITY ("increased" ~ "decreased")

So: embeddings find WHICH reports discuss disease status, the LLM decides
WHICH DIRECTION. That split matters financially -- only 37% of a prostate
patient's scans mention cancer at all, and we have been paying the LLM to read
the other 63%.

Test: embed narratives locally (free), retrieve the ones nearest a set of
"disease status" queries, and measure whether retrieval concentrates the
reports that actually carry response language. If it does, the LLM only reads
the top slice and the Study 3 pipeline gets several times cheaper.

CPU-only, no API calls.
"""
import re, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
N = 8000

E = pd.read_parquet("psa_progression.parquet")
ids = "','".join(E.index.astype(str)[:2000])
print(f"pulling {N:,} narratives ...")
R = C.query(f"""
  WITH pk AS (SELECT DISTINCT PATIENT_DK FROM {D}.DIM_PATIENT
              WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
  SELECT SUBSTR(CAST(r.RADIOLOGY_NARRATIVE AS STRING), 1, 1200) AS txt
  FROM pk JOIN {D}.FACT_RADIOLOGY r
    ON CAST(r.PATIENT_DK AS STRING) = CAST(pk.PATIENT_DK AS STRING)
  WHERE r.RADIOLOGY_NARRATIVE IS NOT NULL
    AND LENGTH(CAST(r.RADIOLOGY_NARRATIVE AS STRING)) BETWEEN 200 AND 6000
  LIMIT {N}""").to_dataframe()
print(f"  {len(R):,} narratives")

# ground truth for "is this a response assessment": explicit comparison language
GOLD = re.compile(r"(interval (increase|decrease|change)|compared (to|with) the prior|"
                  r"new (lesion|metasta|nodule|focus)|no evidence of (disease|metasta|recurren)|"
                  r"stable disease|progress(ion|ed)|enlarg|since the previous)", re.I)
R["is_assessment"] = R["txt"].str.contains(GOLD, na=False)
base = R["is_assessment"].mean()
print(f"  carry response language: {R['is_assessment'].sum():,} ({base:.1%})")

m = SentenceTransformer("models/pubmedbert")
print("\nembedding (CPU) ...")
t0 = time.time()
V = m.encode(R["txt"].tolist(), batch_size=32, normalize_embeddings=True,
             show_progress_bar=False)
dt = time.time() - t0
print(f"  {len(R):,} texts in {dt:.0f}s  ({len(R)/dt:.0f}/sec)")
print(f"  extrapolated to all 285,766 prostate narratives: "
      f"{285766/(len(R)/dt)/60:.0f} min")

QUERIES = [
 "Interval increase in the size of the metastatic lesion compared to prior.",
 "New lesion identified since the previous examination.",
 "No evidence of metastatic disease on this study.",
 "Disease is stable compared to the prior imaging.",
 "Interval decrease in tumor burden compared with the previous study.",
]
Q = m.encode(QUERIES, normalize_embeddings=True)
score = (V @ Q.T).max(1)
R["score"] = score

print("\n" + "=" * 74)
print("DOES RETRIEVAL CONCENTRATE THE REPORTS THAT MATTER?")
print("=" * 74)
print(f"  base rate across everything: {base:.1%}\n")
print(f"  {'read top':>10}{'n':>8}{'% assessments':>16}{'recall':>10}{'lift':>8}")
order = R.sort_values("score", ascending=False)
tot_pos = int(R["is_assessment"].sum())
for pct in (5, 10, 20, 30, 50, 100):
    k = int(len(R) * pct / 100)
    top = order.head(k)
    prec = top["is_assessment"].mean()
    rec = top["is_assessment"].sum() / max(tot_pos, 1)
    print(f"  {pct:>9}%{k:>8,}{prec:>16.1%}{rec:>10.1%}{prec/base:>8.2f}x")

print("\n  lift > 1 means the LLM would see a richer slice than random.")
print("  recall at a given depth is what you would lose by not reading the rest.")

k30 = order.head(int(len(R) * .3))
print(f"\n  reading the top 30%: catches {k30['is_assessment'].sum()/max(tot_pos,1):.0%} "
      f"of assessments for 30% of the LLM cost")
print(f"  on 285,766 narratives that is ~{285766*0.3/5:.0f} calls instead of "
      f"~{285766/5:.0f}")

print("\n" + "=" * 74)
print("SANITY: what does the score distribution look like?")
print("=" * 74)
for lbl, s in [("assessments", R[R["is_assessment"]]["score"]),
               ("everything else", R[~R["is_assessment"]]["score"])]:
    print(f"  {lbl:<18}median {s.median():+.3f}   p10 {s.quantile(.1):+.3f}   "
          f"p90 {s.quantile(.9):+.3f}")

print("\n" + "-" * 70)
print("FINAL LINE:")
top30 = order.head(int(len(R)*.3))
print(f"embed_triage | n={len(R)} | base={base:.1%} "
      f"| top30_precision={top30['is_assessment'].mean():.1%} "
      f"| top30_recall={top30['is_assessment'].sum()/max(tot_pos,1):.0%} "
      f"| rate={len(R)/dt:.0f}/sec")

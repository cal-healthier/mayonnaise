"""
ovarian_text.py -- does the narrative work where the bloodwork failed?

The sharpest possible replication. In ovarian cancer, routine labs LOST to
CA-125 (0.569 vs 0.636) with a verified mechanism: ALP detects bone metastases,
prostate goes to bone, ovarian spreads peritoneally so there is nothing for the
labs to see.

If TEXT wins here, the narrative signal is more general than the lab signal --
a considerably sharper claim than replicating somewhere easy.

Exact mirror of the prostate design:
  label      progression within 12 months (GCIG CA-125 criteria)
  notes      pre-treatment ONLY, no leakage
  arms       structured (labs + CA-125) | +utilisation proxies | text | all

CPU embedding with the speed settings from earlier (256 tokens, batch 64, all
threads) -- roughly 5k notes, so ~10 min per arm rather than 20.
"""
import os, re, time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

torch.set_num_threads(os.cpu_count())
C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

E = pd.read_parquet("ov_label.parquet")
cv = pd.read_parquet("ov_ca125.parquet")
tx = cv.groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
print(f"ovarian: {len(E):,} women | {int(E['y'].sum())} progress within 12mo "
      f"({E['y'].mean():.0%})")
print(f"  (prostate for comparison: 4,987 men, 382 events, 8%)")

if os.path.exists("ovt_notes.parquet"):
    N = pd.read_parquet("ovt_notes.parquet")
    print(f"cached notes: {len(N):,}")
else:
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(E.index.astype(str), E["tx_date"]))
    print("pulling pre-treatment notes ...")
    N = C.query(f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
    pk AS (SELECT c.clinic, c.tx, pe.person_id FROM coh c
           JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
    SELECT pk.clinic, SUBSTR(CAST(n.note_text AS STRING),1,2500) AS txt
    FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
    WHERE n.note_text IS NOT NULL
      AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 25000
      AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient','H&P')
      AND DATE(n.note_date) <= pk.tx AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) <= 6
    """).to_dataframe()
    N.to_parquet("ovt_notes.parquet")
E = E[E.index.isin(N["clinic"])]
print(f"  {len(N):,} notes | cohort with notes: {len(E):,} women, "
      f"{int(E['y'].sum())} events")

NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
N["nonum"] = N["txt"].str.replace(NUMS, " ", regex=True)

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("models/pubmedbert")
m.max_seq_length = 256
def embed(col):
    cache = f"ovt_emb_{col}.parquet"
    if os.path.exists(cache):
        print(f"    {col}: cached"); return pd.read_parquet(cache).reindex(E.index)
    t0 = time.time()
    V = m.encode(N[col].tolist(), batch_size=64, normalize_embeddings=True,
                 show_progress_bar=False)
    o = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    o.columns = [str(c) for c in o.columns]; o.to_parquet(cache)
    print(f"    {col}: {len(N):,} notes in {time.time()-t0:.0f}s "
          f"({len(N)/(time.time()-t0):.0f}/sec)")
    return o.reindex(E.index)

print("\nembedding ...")
TXT = embed("txt"); NONUM = embed("nonum")

labs = pd.read_parquet("ov_labs.parquet")
cols = [c for c in labs.columns if c.endswith("__last")]
S = labs[cols].reindex(E.index); S["baseline_ca125"] = E["baseline"]
S = S.loc[:, S.notna().sum() > len(S)*.3]

ids = "','".join(E.index.astype(str))
P = C.query(f"""
WITH pk AS (SELECT DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic, PATIENT_DK
            FROM {D}.DIM_PATIENT WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
SELECT pk.clinic,
  COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
          r'OXYCODONE|MORPHINE|HYDROMORPHONE|FENTANYL')) AS opioid_orders,
  COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
          r'PREDNISONE|DEXAMETHASONE')) AS steroid_orders,
  COUNT(*) AS total_orders
FROM pk LEFT JOIN {D}.FACT_ORDERS o ON o.PATIENT_DK = pk.PATIENT_DK
GROUP BY 1""").to_dataframe().set_index("clinic")
PROX = P.reindex(E.index).fillna(0)

def run(X, name):
    X = pd.DataFrame(X).reindex(E.index); y = E["y"]
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06,
            max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], mm.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<46}{a.mean():.3f} ± {a.std():.3f}")
    return a.mean(), a.std()

print("\n" + "=" * 78)
print(f"OVARIAN: does the narrative work where the bloodwork failed?")
print(f"({len(E):,} women, {int(E['y'].sum())} events)")
print("=" * 78)
s, ss   = run(S, "structured: labs + CA-125")
sp, sps = run(pd.concat([S, PROX], axis=1), "structured + utilisation proxies")
t, ts   = run(NONUM, "TEXT (numerals removed)")
tf, tfs = run(TXT, "TEXT (full notes)")
b, bs   = run(pd.concat([S, PROX, NONUM.add_prefix("t")], axis=1), "everything")

def sep(a, sa, bq, sb):
    return abs(a-bq) > 1.96*np.sqrt(sa**2+sb**2)/np.sqrt(5)

print("\n" + "=" * 78)
print("VERDICT  (prostate in brackets)")
print("=" * 78)
print(f"  text minus structured           {t-s:+.3f}  "
      f"{'SEPARATED' if sep(t,ts,s,ss) else 'within noise'}   [prostate +0.064]")
print(f"  text minus structured+proxies   {t-sp:+.3f}  "
      f"{'SEPARATED' if sep(t,ts,sp,sps) else 'within noise'}   [prostate +0.019 ns]")
print(f"  everything minus struct+prox    {b-sp:+.3f}  "
      f"{'SEPARATED' if sep(b,bs,sp,sps) else 'within noise'}   [prostate +0.064]")
print(f"""
  Recall that in ovarian the LABS lost to the marker (0.569 vs 0.636) because
  ALP detects bone metastases and ovarian does not go to bone.

  If text wins here, the narrative signal is more general than the lab signal
  -- it is not tied to one disease's pattern of spread. That is the strongest
  version of the claim.

  If text also fails, you have found the boundary yourself, which is worth
  knowing before a reviewer finds it.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"ovarian_text | women={len(E)} | events={int(E['y'].sum())} "
      f"| struct={s:.3f} | struct_prox={sp:.3f} | text={t:.3f} | all={b:.3f} "
      f"| text_minus_struct={t-s:+.3f}")

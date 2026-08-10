"""
text_vs_numbers.py -- does what the doctor WROTE beat what the doctor MEASURED?

The claim worth chasing: a model reading only clinical narrative predicts
treatment failure better than the structured variables oncologists rely on --
including the tumour marker they formally monitor.

That inverts the assumed hierarchy. Labs and PSA are "objective"; prose is
"soft". If prose wins, the structured record -- which is all any registry or
competitor has -- is the weaker signal, and the notes are the asset.

Four arms, one label (progression within 12 months), one held-out split:
  1 STRUCTURED     labs + baseline PSA          (the incumbent, ~0.756)
  2 TEXT ONLY      note embeddings, nothing else
  3 BOTH
  4 TEXT STRIPPED  notes with explicit prognostic language removed
                   -- if this still works, the chart predicts through channels
                      the writer is not consciously using

Embeddings are local and free. No LLM calls.
"""
import re, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
NMEN, PER_MAN = 900, 6

E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)].head(NMEN)
print(f"{len(E):,} men | {int(E['y'].sum()):,} progress within 12 months "
      f"({E['y'].mean():.0%})")

rows = ",".join(f"('{c}',DATE '{d.date()}')"
                for c, d in zip(E.index.astype(str), E["tx_date"]))
print("pulling notes from BEFORE treatment start (no leakage) ...")
N = C.query(f"""
WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
pk AS (SELECT c.clinic, c.tx, pe.person_id FROM coh c
       JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
SELECT pk.clinic, SUBSTR(CAST(n.note_text AS STRING),1,2500) AS txt
FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
WHERE n.note_text IS NOT NULL AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 25000
  AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient','H&P')
  AND DATE(n.note_date) <= pk.tx
  AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) <= {PER_MAN}
""").to_dataframe()
print(f"  {len(N):,} notes | {N['clinic'].nunique()} men have any")

E = E[E.index.isin(N["clinic"])]
print(f"  modelling cohort: {len(E):,} men, {E['y'].mean():.0%} positive")

# strip explicit prognostic language for arm 4
PROG = re.compile(
 r"\b(prognos\w*|poor(ly)? (differentiated|prognosis)|terminal|end.of.life|"
 r"hospice|palliat\w*|aggressive disease|high.risk|advanced disease|"
 r"metastatic|refractory|progress\w*|deteriorat\w*|declin\w*|"
 r"guarded|grave|life expectancy|months to live)\b", re.I)
N["stripped"] = N["txt"].str.replace(PROG, " ", regex=True)
removed = (N["txt"].str.len() - N["stripped"].str.len()).sum()
print(f"  stripped {removed:,} chars of prognostic language "
      f"({removed/N['txt'].str.len().sum():.1%} of text)")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("models/pubmedbert")
def embed_per_man(col):
    t0 = time.time()
    V = m.encode(N[col].tolist(), batch_size=32, normalize_embeddings=True,
                 show_progress_bar=False)
    df = pd.DataFrame(V, index=N["clinic"].values)
    out = df.groupby(level=0).mean()          # one vector per man
    print(f"    {col}: {len(N):,} notes in {time.time()-t0:.0f}s")
    return out.reindex(E.index)

print("\nembedding (CPU, free) ...")
TXT = embed_per_man("txt")
STRIP = embed_per_man("stripped")

cols = [c for c in labs.columns if c.endswith("__last")]
STRUCT = labs[cols].reindex(E.index)
STRUCT["baseline_psa"] = E["baseline"]
STRUCT = STRUCT.loc[:, STRUCT.notna().sum() > len(STRUCT) * .3]
print(f"\nstructured features: {STRUCT.shape[1]}  |  text dims: {TXT.shape[1]}")

def run(X, name):
    X = pd.DataFrame(X).reindex(E.index)
    y = E["y"]
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mm = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.06, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1., random_state=0
        ).fit(X.iloc[tr], y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], mm.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<44}AUROC {a.mean():.3f} ± {a.std():.3f}  ({X.shape[1]} feat)")
    return a.mean()

print("\n" + "=" * 76)
print("WHAT THE DOCTOR MEASURED vs WHAT THE DOCTOR WROTE")
print("=" * 76)
s = run(STRUCT, "1  STRUCTURED: labs + PSA (the incumbent)")
t = run(TXT, "2  TEXT ONLY: note embeddings")
b = run(pd.concat([STRUCT, TXT.add_prefix("t")], axis=1), "3  BOTH")
st = run(STRIP, "4  TEXT, prognostic language stripped")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
print(f"  text minus structured        {t-s:+.3f}   "
      f"{'*** TEXT WINS ***' if t > s else 'structured still ahead'}")
print(f"  both minus best single       {b-max(s,t):+.3f}")
print(f"  stripped minus full text     {st-t:+.3f}   "
      f"(how much was explicit prognosis)")
print(f"  stripped minus structured    {st-s:+.3f}")
print("""
  text >= structured is the headline: prose beats the numbers oncologists
  formally monitor.
  stripped still beating structured is the paper: the chart predicts through
  channels the writer is not consciously using.
  text well below structured means the notes add colour, not signal -- worth
  knowing before building a product claim on them.""")

print("\n" + "-" * 72)
print("FINAL LINE:")
print(f"text_vs_numbers | men={len(E)} | pos={E['y'].mean():.0%} "
      f"| structured={s:.3f} | text={t:.3f} | both={b:.3f} | stripped={st:.3f} "
      f"| text_minus_struct={t-s:+.3f}")

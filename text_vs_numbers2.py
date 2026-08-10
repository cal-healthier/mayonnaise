"""
text_vs_numbers2.py -- properly powered, and with the objection closed.

First pass (900 men, 83 events) gave structured 0.746, text 0.780, both 0.838.
Direction right, but +-0.054 error bars on a +0.034 difference establishes
nothing. Two design faults as well:

  - my "strip prognostic language" regex removed 0.3% of the text, so that arm
    was not a test of anything
  - notes contain lines like "PSA 12.4, ALP 140". The text arm may simply be
    RE-ENCODING the structured variables rather than beating them. That is the
    first objection any reviewer raises and I did not control for it.

Fixes:
  1. every man with notes, not 900  -> roughly 5x the events
  2. a NUMBERS-STRIPPED arm: remove every numeral from the notes. If narrative
     with no numbers still beats the lab panel, the objection is dead and the
     claim is real.
  3. a wider prognosis strip that actually removes something

Long CPU job (embedding is ~4.6 notes/sec). Caches after each stage so it can
be resumed. Free -- no API calls.
"""
import os, re, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
PER_MAN = 6

E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index)); E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
print(f"{len(E):,} men | {int(E['y'].sum()):,} events ({E['y'].mean():.0%})")

if os.path.exists("tvn_notes.parquet"):
    N = pd.read_parquet("tvn_notes.parquet")
    print(f"cached notes: {len(N):,}")
else:
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(E.index.astype(str), E["tx_date"]))
    print("pulling notes ...")
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
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) <= {PER_MAN}
    """).to_dataframe()
    N.to_parquet("tvn_notes.parquet")
E = E[E.index.isin(N["clinic"])]
print(f"  {len(N):,} notes | modelling cohort {len(E):,} men, "
      f"{int(E['y'].sum()):,} events ({E['y'].mean():.0%})")

# --- the two strips -------------------------------------------------------
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
PROG = re.compile(
 r"\b(prognos\w*|terminal|hospice|palliat\w*|aggressive|high[- ]risk|advanced|"
 r"metasta\w*|refractor\w*|progress\w*|deteriorat\w*|declin\w*|guarded|grave|"
 r"life expectancy|poor\w*|worsen\w*|stable|improv\w*|respond\w*|remission|"
 r"recurren\w*|relaps\w*|spread\w*|widespread|extensive|bulky|burden)\b", re.I)
N["nonum"] = N["txt"].str.replace(NUMS, " ", regex=True)
N["noprog"] = N["txt"].str.replace(PROG, " ", regex=True)
for c, lbl in [("nonum", "numerals"), ("noprog", "prognostic words")]:
    d = (N["txt"].str.len() - N[c].str.len()).sum() / N["txt"].str.len().sum()
    print(f"  stripping {lbl:<18}removes {d:.1%} of characters")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("models/pubmedbert")
def embed(col):
    cache = f"tvn_emb_{col}.parquet"
    if os.path.exists(cache):
        print(f"    {col}: cached"); return pd.read_parquet(cache).reindex(E.index)
    t0 = time.time()
    V = m.encode(N[col].tolist(), batch_size=32, normalize_embeddings=True,
                 show_progress_bar=False)
    out = pd.DataFrame(V, index=N["clinic"].values).groupby(level=0).mean()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(cache)
    print(f"    {col}: {len(N):,} notes in {time.time()-t0/1:.0f}s")
    return out.reindex(E.index)

print("\nembedding (slow, cached per arm) ...")
TXT   = embed("txt")
NONUM = embed("nonum")
NOPRG = embed("noprog")

cols = [c for c in labs.columns if c.endswith("__last")]
STRUCT = labs[cols].reindex(E.index)
STRUCT["baseline_psa"] = E["baseline"]
STRUCT = STRUCT.loc[:, STRUCT.notna().sum() > len(STRUCT)*.3]

def run(X, name):
    X = pd.DataFrame(X).reindex(E.index); y = E["y"]
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        mm = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.06, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1., random_state=0
        ).fit(X.iloc[tr], y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], mm.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<46}{a.mean():.3f} ± {a.std():.3f}")
    return a.mean(), a.std()

print("\n" + "=" * 78)
print(f"WHAT WAS MEASURED vs WHAT WAS WRITTEN   ({len(E):,} men, "
      f"{int(E['y'].sum())} events)")
print("=" * 78)
s, ss   = run(STRUCT, "1  STRUCTURED: labs + PSA")
t, ts   = run(TXT,    "2  TEXT (full notes)")
b, bs   = run(pd.concat([STRUCT, TXT.add_prefix("t")], axis=1), "3  BOTH")
nn, nns = run(NONUM,  "4  TEXT, ALL NUMERALS REMOVED")
np_, nps= run(NOPRG,  "5  TEXT, prognostic words removed")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
def cmp(a, sa, bq, sb, lbl):
    d = a - bq
    sep = abs(d) > 1.96*np.sqrt(sa**2 + sb**2)/np.sqrt(5)
    print(f"  {lbl:<40}{d:+.3f}   {'SEPARATED' if sep else 'within noise'}")
cmp(t, ts, s, ss,   "text minus structured")
cmp(b, bs, max(s,t), max(ss,ts), "both minus best single")
cmp(nn, nns, s, ss, "NUMERALS-REMOVED text minus structured")
cmp(np_, nps, s, ss, "prognosis-removed text minus structured")
print(f"""
  Arm 4 is the one that matters. If narrative with every number deleted still
  beats a lab panel, the text is not merely re-encoding the labs -- it carries
  something the measurements do not. That closes the obvious objection and is
  the claim worth making.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"tvn2 | men={len(E)} | events={int(E['y'].sum())} | struct={s:.3f} "
      f"| text={t:.3f} | both={b:.3f} | nonum={nn:.3f} | noprog={np_:.3f} "
      f"| nonum_minus_struct={nn-s:+.3f}")

"""
tvn_analyze.py -- run the comparison on whatever embeddings are already cached.

text_vs_numbers2.py embeds three arms in order (txt, nonum, noprog) at ~1.8h
each. The third is the least informative. This picks up whatever exists on disk
and reports, so you can interrupt once `nonum` is done.

Also prints a speed note for future runs -- the default settings leave
throughput on the table.
"""
import os, glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

pd.set_option("display.width", 240)
E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index)); E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
N = pd.read_parquet("tvn_notes.parquet")
E = E[E.index.isin(N["clinic"])]
print(f"{len(E):,} men | {int(E['y'].sum()):,} events ({E['y'].mean():.0%})")

have = {}
for f in sorted(glob.glob("tvn_emb_*.parquet")):
    arm = f.replace("tvn_emb_", "").replace(".parquet", "")
    have[arm] = pd.read_parquet(f).reindex(E.index)
    print(f"  cached: {arm:<10}{have[arm].shape}")
if not have:
    raise SystemExit("no cached embeddings yet -- let the job run longer")

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

LABEL = {"txt": "TEXT (full notes)",
         "nonum": "TEXT, ALL NUMERALS REMOVED",
         "noprog": "TEXT, prognostic words removed"}
print("\n" + "=" * 78)
print(f"WHAT WAS MEASURED vs WHAT WAS WRITTEN   ({len(E):,} men, "
      f"{int(E['y'].sum())} events)")
print("=" * 78)
s, ss = run(STRUCT, "STRUCTURED: labs + PSA")
res = {}
for arm, V in have.items():
    res[arm] = run(V, LABEL.get(arm, arm))
if "txt" in have:
    res["both"] = run(pd.concat([STRUCT, have["txt"].add_prefix("t")], axis=1),
                      "BOTH (structured + text)")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
def cmp(a, sa, lbl):
    d = a - s
    sep = abs(d) > 1.96*np.sqrt(sa**2 + ss**2)/np.sqrt(5)
    print(f"  {lbl:<44}{d:+.3f}   {'SEPARATED' if sep else 'within noise'}")
for arm in ("txt", "nonum", "noprog", "both"):
    if arm in res:
        cmp(res[arm][0], res[arm][1], f"{LABEL.get(arm, arm)} minus structured")
if "nonum" in res:
    print(f"""
  The numerals-removed arm is the study. {res['nonum'][0]:.3f} vs {s:.3f} structured.
  If that separates, narrative with every number deleted still beats the lab
  panel -- the text is not re-encoding the measurements, it carries something
  they do not.""")

print("\n" + "-" * 74)
print("SPEED NOTE for future embedding runs:")
print("""  import torch; torch.set_num_threads(os.cpu_count())
  m.max_seq_length = 256          # from 512 -- roughly halves the time
  m.encode(..., batch_size=64)
  Most notes lose little at 256 tokens, and throughput roughly doubles.""")
print("-" * 74)
print("FINAL LINE:")
print(f"tvn_analyze | men={len(E)} | events={int(E['y'].sum())} | struct={s:.3f} | "
      + " | ".join(f"{k}={v[0]:.3f}" for k, v in res.items()))

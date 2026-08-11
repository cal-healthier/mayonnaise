"""
strip_test.py -- does anything survive once the header is gone?

The deflationary hypothesis, stated so it can be killed:

    the text arm works because the MIX OF DOCUMENT TYPES in a patient's
    record encodes where they are in the treatment pathway. Oncology phone
    messages mean active chemo; post-op and nutrition notes mean a surgical
    episode. That is prognostic, but it is care bookkeeping, and a handful of
    counts should reproduce it.

Six arms, same folds, same patients:

  structured + proxies         the reference
  + NOTE-TYPE COUNTS ONLY      ~30 integers, no text at all. If this
                               reproduces the text gain, the story is over.
  + text, raw                  what we have been reporting
  + text, header stripped      preamble removed
  + text, clinical only        plumbing removed too -- prose alone
  + counts AND clinical text   does prose add anything the counts do not?

The last comparison is the one that decides whether there is a paper. If
clinical-only text still beats structured+proxies+counts, the narrative
carries something beyond which service typed it. If it does not, the honest
finding is that note-type composition predicts progression -- a much smaller
claim, and one a reviewer would find in an afternoon.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(0)
REPEATS, BOOT, K = 2, 2000, 12
COH = {
    "ovarian":  ("ov_label.parquet", "ov_ca125.parquet", "ov_labs.parquet",
                 "baseline_ca125", "prox_ovarian"),
    "prostate": ("psa_progression.parquet", "psa_values.parquet", "psa_labs.parquet",
                 "psa", "prox_prostate"),
}


def oof(X, y, repeats=REPEATS):
    X = pd.DataFrame(X).reindex(y.index)
    acc = np.zeros(len(X))
    for r in range(repeats):
        p = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                min_samples_leaf=25, l2_regularization=1.,
                random_state=0).fit(X.iloc[tr], y.iloc[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        acc += p
    return acc / repeats


def paired(y, pa, pb, boot=BOOT):
    y = np.asarray(y)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    n, d = len(y), []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, len(i)):
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return (obs, *np.percentile(d, [2.5, 97.5]))


for tag, (lab, mark, labs_f, marker, proxf) in COH.items():
    S_txt = f"strip_{tag}.parquet"
    if not os.path.exists(S_txt):
        print(f"{S_txt} missing -- run gpu_strip.py first")
        continue
    E = pd.read_parquet(lab)
    tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
    E = E[(E["prog"] == 1) | (E["time"] >= 365)]
    y = E["y"]
    labs = pd.read_parquet(labs_f)
    cols = [c for c in labs.columns if c.endswith("__last")]
    S = labs[cols].reindex(E.index); S[marker] = E["baseline"]
    S = S.loc[:, S.notna().sum() > len(S) * .3]
    PROX = None
    for f in (f"{proxf}_pre.parquet", f"{proxf}.parquet"):
        if os.path.exists(f):
            PROX = pd.read_parquet(f).reindex(E.index).fillna(0)
            break
    base_cols = pd.concat([S, PROX], axis=1) if PROX is not None else S

    T = pd.read_parquet(S_txt)
    T["clinic"] = T["clinic"].astype(str)
    T = T[T["clinic"].isin(E.index) & (T["rn"] <= K)]

    # ---- the deflationary arm: counts of document type, no text whatsoever
    top = T["ntype"].value_counts().head(30).index
    CT = (T[T["ntype"].isin(top)]
          .pivot_table(index="clinic", columns="ntype", values="rn", aggfunc="count")
          .reindex(E.index).fillna(0))
    CT["n_notes"] = T.groupby("clinic").size().reindex(E.index).fillna(0)

    print("\n" + "=" * 78)
    print(f"{tag.upper()}   {len(E):,} patients, {int(y.sum())} events, "
          f"top-{K} notes, {CT.shape[1]} type-count features")
    print("=" * 78)

    arms, P = {}, {}
    arms["structured + proxies"] = base_cols
    arms["+ note-type COUNTS only"] = pd.concat([base_cols, CT], axis=1)
    for col, nm in (("raw", "+ text, raw (as reported)"),
                    ("stripped", "+ text, header stripped"),
                    ("clinical", "+ text, clinical prose only")):
        f = (f"rn96_emb_{tag}_512.parquet" if col == "raw"
             else f"strip_emb_{tag}_{col}.parquet")
        if not os.path.exists(f):
            continue
        V = pd.read_parquet(f)
        V["clinic"] = V["clinic"].astype(str)
        V = V[V["clinic"].isin(E.index) & (V["rn"] <= K)]
        dims = [c for c in V.columns if c.isdigit()]
        M = V.groupby("clinic")[dims].mean().reindex(E.index)
        arms[nm] = pd.concat([base_cols, M.add_prefix("t")], axis=1)
        if col == "clinical":
            arms["+ counts AND clinical text"] = pd.concat(
                [base_cols, CT, M.add_prefix("t")], axis=1)

    for nm, X in arms.items():
        P[nm] = oof(X, y)
        print(f"  {nm:<34}{roc_auc_score(y, P[nm]):.3f}")

    ref = "structured + proxies"
    print(f"\n  gain over {ref}:")
    for nm in arms:
        if nm == ref:
            continue
        o, lo, hi = paired(y, P[nm], P[ref])
        print(f"    {nm:<34}{o:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"{'  REAL' if lo > 0 else ''}")

    if "+ counts AND clinical text" in P and "+ note-type COUNTS only" in P:
        o, lo, hi = paired(y, P["+ counts AND clinical text"],
                           P["+ note-type COUNTS only"])
        print(f"\n  THE DECIDING TEST -- does prose add anything beyond the counts?")
        print(f"    clinical text on top of counts   {o:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"   {'SURVIVES' if lo > 0 else 'DOES NOT SURVIVE'}")

print("""
{}
WHAT EACH OUTCOME MEANS
{}
  counts alone reproduce the gain, prose adds nothing
      The finding is "note-type composition predicts progression". Real, but
      small, and derivable from metadata without any language model. The
      narrative claim does not stand and should be withdrawn.

  prose survives on top of counts
      The header was a confound we have now removed, and what is left is the
      actual claim -- stronger than before, because the obvious deflationary
      explanation has been tested and rejected rather than ignored.

  stripped text collapses but clinical text holds
      Message plumbing was carrying it. Worth knowing exactly which part.

  Either way this had to be run before anything is written down. The occlusion
  result is not survivable by argument.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print("strip_test | see table above")

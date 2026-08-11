"""
repr_test.py -- does HOW we collapse notes matter more than WHICH encoder?

The bake-off answered the encoder question: no, it does not matter. Four very
different pretraining corpora gave 0.633 / 0.633 / 0.632 / 0.629. So the
limit is elsewhere, and the two candidates are truncation and pooling.

Same patients, same folds, same encoder (pubmedbert). Only the collapse from
per-note vectors to one patient vector changes:

  mean            what every result so far has used
  max             per-dimension max -- survives one alarming note among five
                  routine ones, which the mean averages away
  mean+max+std    2,304 dims; keeps the average AND the extremes AND how much
                  the notes disagree with each other
  most recent     one note only, the one written closest to treatment
  recent 3 mean   compromise between recency and averaging
  weighted        exponential decay by days before treatment (30-day
                  half-life), so a note from last week outweighs one from
                  ten months ago

and at two truncation lengths, 256 vs 512 tokens.

If a pooling change moves this more than swapping the whole encoder did, then
representation is the lever and fine-tuning was the wrong next step.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(0)
REPEATS, BOOT = 3, 2000

E = pd.read_parquet("ov_label.parquet")
cv = pd.read_parquet("ov_ca125.parquet")
E["tx_date"] = pd.to_datetime(cv.groupby("clinic")["tx_date"].first().reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]

labs = pd.read_parquet("ov_labs.parquet")
cols = [c for c in labs.columns if c.endswith("__last")]
S = labs[cols].reindex(E.index); S["baseline_ca125"] = E["baseline"]
S = S.loc[:, S.notna().sum() > len(S) * .3]
PROX = (pd.read_parquet("prox_ovarian_pre.parquet").reindex(E.index).fillna(0)
        if os.path.exists("prox_ovarian_pre.parquet")
        else pd.read_parquet("prox_ovarian.parquet").reindex(E.index).fillna(0)
        if os.path.exists("prox_ovarian.parquet") else None)


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


def poolings(V):
    """V: per-note rows with clinic, rn, days_before + 768 numeric columns"""
    dims = [c for c in V.columns if c.isdigit()]
    g = V.groupby("clinic")[dims]
    out = {}
    out["mean"] = g.mean()
    out["max"] = g.max()
    out["mean+max+std"] = pd.concat(
        [g.mean(), g.max().add_prefix("x"), g.std().fillna(0).add_prefix("s")], axis=1)
    out["most recent"] = (V[V["rn"] == 1].set_index("clinic")[dims])
    out["recent 3 mean"] = V[V["rn"] <= 3].groupby("clinic")[dims].mean()
    w = np.exp(-V["days_before"].clip(lower=0) / 30.0)
    Wv = V[dims].mul(w, axis=0)
    Wv["clinic"] = V["clinic"].values
    num = Wv.groupby("clinic")[dims].sum()
    den = pd.Series(w.values, index=V["clinic"].values).groupby(level=0).sum()
    out["recency weighted"] = num.div(den, axis=0)
    return out


y = E["y"]
results = {}

for L in (256, 512):
    f = f"rn_emb_{L}.parquet"
    if not os.path.exists(f):
        print(f"\n{f} missing -- run gpu_repr.py first")
        continue
    V = pd.read_parquet(f)
    V = V[V["clinic"].isin(E.index)]
    print("\n" + "=" * 78)
    print(f"TRUNCATION {L} TOKENS   {len(V):,} notes, {V['clinic'].nunique():,} women")
    print("=" * 78)
    P = poolings(V)
    idx = y.index
    print(f"  {'pooling':<20}{'dims':>6}{'text only':>12}{'+structured':>14}")
    print("  " + "-" * 54)
    for name, M in P.items():
        M = M.reindex(idx)
        pt = oof(M, y)
        a_text = roc_auc_score(y, pt)
        if PROX is not None:
            pf = oof(pd.concat([S, PROX, M.add_prefix("t")], axis=1), y)
            a_full = roc_auc_score(y, pf)
        else:
            pf, a_full = None, float("nan")
        results[(L, name)] = (pt, pf, a_text, a_full)
        print(f"  {name:<20}{M.shape[1]:>6}{a_text:>12.3f}{a_full:>14.3f}")

base = results.get((256, "mean"))
if base:
    print("\n" + "=" * 78)
    print("EVERYTHING vs THE CURRENT CHOICE (mean @ 256), paired over patients")
    print("=" * 78)
    print(f"  for scale: swapping the entire encoder moved this -0.004 to +0.000\n")
    print(f"  {'variant':<32}{'diff':>8}  {'95% CI':>18}")
    print("  " + "-" * 60)
    for (L, name), (pt, pf, at, af) in results.items():
        if (L, name) == (256, "mean"):
            continue
        o, lo, hi = paired(y, pt, base[0])
        flag = "  <-- BEATS IT" if lo > 0 else ""
        print(f"  {f'{name} @ {L}':<32}{o:+8.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")

    if PROX is not None:
        best = max((k for k in results), key=lambda k: results[k][3])
        print(f"\n  best full model: {best[1]} @ {best[0]} = {results[best][3]:.3f}")
        o, lo, hi = paired(y, results[best][1], base[1])
        print(f"  vs mean @ 256 full model:  {o:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("""
{}
READING IT
{}
  If several pooling variants move the number more than swapping PubMedBERT
  for a general web-text encoder did, the bottleneck was never the encoder --
  it is that six notes get averaged into one vector and each note is cut to
  its first thousand characters.

  That would also downgrade the fine-tuning plan. Domain-adaptive pretraining
  changes what the encoder knows, and four encoders with entirely different
  knowledge already scored the same. Fixing the representation is cheaper and
  the bake-off says it is where the headroom is.

  If nothing moves, the opposite conclusion: 0.63 is what these notes support
  under any reasonable representation, and the honest next step is more
  events, not more model.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
if results:
    b = max(results, key=lambda k: results[k][2])
    print(f"repr_test | best={b[1]}@{b[0]} text={results[b][2]:.3f} "
          f"| mean@256={base[2]:.3f} | variants={len(results)}")
else:
    print("repr_test | no per-note embeddings found")

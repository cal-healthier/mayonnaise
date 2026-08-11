"""
sweep_test.py -- how many notes per patient is the right number?

k = 1, 3, 6, 12, 24, 48, 96, evaluated by filtering per-note vectors on
rn <= k and re-pooling. No re-encoding, so the whole curve costs one pass of
gradient boosting per point.

6 was never chosen, it was assumed. The pooling run found a clean monotonic
rise from 1 to 3 to 6 and those were its only significant results, so the
question is where the curve stops rising -- and whether it turns over, which
is the outcome a single point at 12 could never reveal.

Two things are reported at every k that matter as much as the AUC:

  patients at the cap   once this hits zero, raising k further changes nothing
                        and any wobble beyond that point is noise
  median notes used     the effective sample of text per patient

Both cohorts. Prostate has 382 events against ovarian's 170, so if the two
curves agree in shape that is worth more than either alone.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CAP = 96
KS = (1, 3, 6, 12, 24, 48, 96)
SEQ = (256, 512)
rng = np.random.default_rng(0)
REPEATS, BOOT = 2, 2000

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


def load(tag):
    lab, mark, labs_f, marker, proxf = COH[tag]
    E = pd.read_parquet(lab)
    tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
    E = E[(E["prog"] == 1) | (E["time"] >= 365)]
    labs = pd.read_parquet(labs_f)
    cols = [c for c in labs.columns if c.endswith("__last")]
    S = labs[cols].reindex(E.index); S[marker] = E["baseline"]
    S = S.loc[:, S.notna().sum() > len(S) * .3]
    P = None
    for f in (f"{proxf}_pre.parquet", f"{proxf}.parquet"):
        if os.path.exists(f):
            P = pd.read_parquet(f).reindex(E.index).fillna(0)
            if f.endswith("_pre.parquet"):
                break
    return E, S, P


summary = {}
for tag in COH:
    for L in SEQ:
        f = f"rn{CAP}_emb_{tag}_{L}.parquet"
        if not os.path.exists(f):
            print(f"{f} missing -- run gpu_sweep.py first")
            continue
        E, S, PROX = load(tag)
        V = pd.read_parquet(f)
        V = V[V["clinic"].isin(E.index)]
        dims = [c for c in V.columns if c.isdigit()]
        y = E["y"]
        per = V.groupby("clinic").size()

        print("\n" + "=" * 78)
        print(f"{tag.upper()} @ {L} TOKENS   {len(E):,} patients, {int(y.sum())} events, "
              f"{len(V):,} notes")
        print("=" * 78)
        print(f"  {'k':>4}{'at cap':>9}{'median used':>13}{'text only':>12}"
              f"{'+structured':>14}")
        print("  " + "-" * 54)
        preds = {}
        for k in KS:
            Vk = V[V["rn"] <= k]
            M = Vk.groupby("clinic")[dims].mean().reindex(y.index)
            pt = oof(M, y)
            at_cap = (per >= k).mean()
            med = Vk.groupby("clinic").size().median()
            if PROX is not None:
                pf = oof(pd.concat([S, PROX, M.add_prefix("t")], axis=1), y)
                af = roc_auc_score(y, pf)
            else:
                pf, af = None, float("nan")
            preds[k] = (pt, pf)
            print(f"  {k:>4}{at_cap:>8.0%}{med:>13.0f}"
                  f"{roc_auc_score(y, pt):>12.3f}{af:>14.3f}")

        print(f"\n  paired against k=6, the current choice:")
        for k in KS:
            if k == 6:
                continue
            o, lo, hi = paired(y, preds[k][0], preds[6][0])
            flag = ("  BETTER" if lo > 0 else ("  worse" if hi < 0 else ""))
            print(f"    k={k:<4}{o:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")

        best = max(KS, key=lambda k: roc_auc_score(y, preds[k][0]))
        summary[(tag, L)] = (best, roc_auc_score(y, preds[best][0]),
                             roc_auc_score(y, preds[6][0]))

print("\n" + "=" * 78)
print("WHERE THE CURVE PEAKS")
print("=" * 78)
print(f"  {'cohort':<12}{'tokens':>8}{'best k':>9}{'AUC':>9}{'k=6':>9}{'gain':>9}")
for (tag, L), (bk, ba, b6) in summary.items():
    print(f"  {tag:<12}{L:>8}{bk:>9}{ba:>9.3f}{b6:>9.3f}{ba-b6:>+9.3f}")
print("""
  Read the 'at cap' column before believing any peak. Once it reaches 0% the
  patients have run out of notes, k stops doing anything, and every further
  point is the same model re-fitted on identical data -- differences there are
  pure noise, not a plateau you discovered.

  If both cohorts peak in the same place, that is a real hyperparameter and it
  should be used everywhere and stated in the paper. If they disagree, the
  honest move is to pick k on one cohort and report the other, exactly as we
  did with the encoder.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print("sweep | " + " | ".join(f"{t}@{L}:best_k={b}({a:.3f}vs{s:.3f})"
                              for (t, L), (b, a, s) in summary.items()))

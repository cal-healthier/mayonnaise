"""
size_control.py -- was the gap compression TIME, or just less training data?

Forward-in-time the bloodwork-vs-PSA gap fell from +0.088 to +0.025 (876
training men) and +0.053 (1,651). But those cuts also train on a fifth of the
data -- and a 68-feature model needs far more examples than a 2-feature one, so
starving both hurts only one of them.

Decisive control: run the RANDOM split with the training fold capped at exactly
the same sizes, all three feature sets separately.

  random-at-876 gap ~ +0.025  -> compression is DATA VOLUME; the full-data
                                 +0.088 stands and my correction was wrong.
  random-at-876 gap ~ +0.088  -> compression is TIME; the claim really does
                                 need narrowing.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

pd.set_option("display.width", 200)
P = pd.read_parquet("psa_progression.parquet")
PL = pd.read_parquet("psa_labs.parquet")
hc = [c for c in PL.columns if "height" in c.lower()]
if hc and PL[hc[0]].median() < 1.0:
    PL[hc] = PL[hc] / 0.0254

def cidx(t, e, r):
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(r, float)
    conc = perm = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        k = later.sum()
        if k == 0:
            continue
        conc += np.sum(r[i] > r[later]) + 0.5 * np.sum(r[i] == r[later])
        perm += k
    return conc / perm if perm else np.nan

def gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)

def prep(df):
    X = df.reindex(P.index).select_dtypes(include=[np.number])
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

B = pd.DataFrame({"baseline": P["baseline"], "log_baseline": np.log1p(P["baseline"])})
SETS = {"PSA": prep(B), "blood": prep(PL), "both": prep(B.join(PL, how="left"))}

def cv_at(X, n_tr, seed=0):
    rng = np.random.default_rng(seed)
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        if n_tr is not None and len(tr) > n_tr:
            tr = rng.choice(tr, n_tr, replace=False)
        m = gbm().fit(X.iloc[tr], P["prog"].iloc[tr])
        s.append(cidx(P["time"].iloc[te], P["prog"].iloc[te],
                      m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(s))

FWD = {876: (0.616, 0.641, 0.685), 1651: (0.634, 0.687, 0.710)}

print("=" * 88)
print("RANDOM SPLIT AT MATCHED TRAINING SIZES  (vs forward-in-time at the same size)")
print("=" * 88)
print(f"  {'training n':<14}{'split':<12}{'PSA':>8}{'blood':>9}{'both':>8}"
      f"{'gap (blood-PSA)':>18}")
print("  " + "-" * 70)
rows = []
for n_tr in (876, 1651, 3008, None):
    r = {k: cv_at(X, n_tr) for k, X in SETS.items()}
    gap = r["blood"] - r["PSA"]
    lbl = f"{n_tr:,}" if n_tr else "all 4,509"
    print(f"  {lbl:<14}{'random':<12}{r['PSA']:>8.3f}{r['blood']:>9.3f}"
          f"{r['both']:>8.3f}{gap:>+18.3f}")
    rows.append({"n": n_tr or 4509, "split": "random", "gap": gap, **r})
    if n_tr in FWD:
        p, b, o = FWD[n_tr]
        print(f"  {'':<14}{'forward':<12}{p:>8.3f}{b:>9.3f}{o:>8.3f}{b-p:>+18.3f}")
        rows.append({"n": n_tr, "split": "forward", "PSA": p, "blood": b,
                     "both": o, "gap": b - p})

R = pd.DataFrame(rows)
print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
for n in (876, 1651):
    rr = R[(R["n"] == n) & (R["split"] == "random")]["gap"].iloc[0]
    ff = R[(R["n"] == n) & (R["split"] == "forward")]["gap"].iloc[0]
    print(f"  at {n:,} training men: random gap {rr:+.3f}  vs forward gap {ff:+.3f}"
          f"   difference {ff-rr:+.3f}")
full = R[(R["n"] == 4509)]["gap"].iloc[0]
r876 = R[(R["n"] == 876) & (R["split"] == "random")]["gap"].iloc[0]
print(f"\n  gap on full data (random):        {full:+.3f}")
print(f"  gap at 876 training men (random): {r876:+.3f}")
print(f"  -> {full - r876:+.3f} of the compression is pure DATA VOLUME\n")
f876 = R[(R["n"] == 876) & (R["split"] == "forward")]["gap"].iloc[0]
if abs(f876 - r876) < 0.02:
    print("  CONCLUSION: forward and random agree once training size is matched.")
    print("  The gap shrank because we starved the 68-feature model, not because")
    print("  of time. The full-data +0.088 stands; Claude's correction was wrong.")
else:
    print("  CONCLUSION: forward is still clearly below random at equal training")
    print("  size. Time really does erode the advantage; the narrowed claim holds.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"size_control | gap_full={full:+.3f} | gap_rand876={r876:+.3f} "
      f"| gap_fwd876={f876:+.3f} | diff={f876-r876:+.3f} "
      f"| verdict={'DATA_VOLUME' if abs(f876-r876)<0.02 else 'TIME'}")

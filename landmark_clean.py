"""
landmark_clean.py -- is 0.800 biology, or bookkeeping?

Two things in the importance ranking need testing:
  n_psa   (#2, +0.019) -- the NUMBER of PSA draws so far. That is utilisation:
          men tested often are men someone is already worried about. Same class
          of confound as the __count features, which came out clean in the
          survival work. Here it ranks second.
  psa_cur (#1, +0.158) -- eight times anything else. The model may be reading
          "PSA is high" rather than "PSA is climbing", which is a much plainer
          claim than trajectory modelling.

Four nested feature sets, same landmarks, same folds:
  A everything                       (the 0.800)
  B no utilisation/timing            (drop n_psa, mo_on_tx, mo_since_nadir)
  C trajectory shape only            (slopes and ratios; NO absolute PSA level)
  D absolute PSA level alone         (one feature -- the floor)

If D alone is ~0.78, the model is a PSA thermometer and should be described as
one. If C holds up without the level, trajectory really is carrying it.

Also reports TRUE lead time, uncapped: for each man, the first month the model
ever flagged him against his actual progression month -- scanning ALL his
landmarks, not just those inside the 6-month label window.

Reads landmarks.parquet. No BigQuery.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

pd.set_option("display.width", 240)
L = pd.read_parquet("landmarks.parquet")
E = pd.read_parquet("psa_progression.parquet")
print(f"{len(L):,} landmarks | {L['clinic'].nunique():,} men | "
      f"positive {L['y'].mean():.1%}")

fc = [c for c in L.columns if c not in ("clinic", "mo", "y", "_nadir", "_cur")]
def usable(cols):
    return [c for c in cols if L[c].nunique(dropna=True) >= 2]

UTIL = ["n_psa", "mo_on_tx", "mo_since_nadir"]
LEVEL = ["psa_cur"]
SHAPE = [c for c in fc if ("slope" in c or "_vs_" in c)]
SETS = {
    "A  everything (the 0.800)":            fc,
    "B  no utilisation / timing":           [c for c in fc if c not in UTIL],
    "C  trajectory shape only (no level)":  SHAPE,
    "D  absolute PSA level alone":          LEVEL,
    "E  shape + labs, no level, no timing": [c for c in SHAPE if c not in UTIL],
}

def run(cols, name, want_oof=False):
    cols = usable(cols)
    X, y, g = L[cols], L["y"], L["clinic"]
    a, oof = [], pd.Series(index=L.index, dtype=float)
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof.iloc[te] = p
        a.append(roc_auc_score(y.iloc[te], p))
    a = np.array(a)
    print(f"  {name:<40}AUROC {a.mean():.3f} ± {a.std():.3f}  ({len(cols)} feat)")
    return (a.mean(), oof) if want_oof else a.mean()

print("\n" + "=" * 82)
print("WHAT IS ACTUALLY CARRYING THE 0.800?")
print("=" * 82)
res = {}
for k, v in SETS.items():
    if k.startswith("B"):
        res[k], oof = run(v, k, want_oof=True)
    else:
        res[k] = run(v, k)

a_all = res["A  everything (the 0.800)"]
print(f"\n  cost of dropping utilisation/timing : "
      f"{res['B  no utilisation / timing'] - a_all:+.3f}")
print(f"  absolute PSA level ALONE gets       : "
      f"{res['D  absolute PSA level alone']:.3f}")
print(f"  trajectory shape without the level  : "
      f"{res['C  trajectory shape only (no level)']:.3f}")
print("\n  if D is close to A, this is a PSA thermometer and should be called one.")

# ---------------------------------------------------------------- true lead time
print("\n" + "=" * 82)
print("TRUE LEAD TIME (uncapped): first alert ever, vs actual progression month")
print("=" * 82)
L2 = L.copy(); L2["score"] = oof
prog_men = E.index[E["prog"] == 1]
end_mo = (E["time"] / 30.44)
for pct in (5, 10, 20):
    th = L2["score"].quantile(1 - pct / 100)
    fired = L2[(L2["score"] >= th) & (L2["clinic"].isin(prog_men))]
    if not len(fired):
        continue
    first = fired.groupby("clinic")["mo"].min()
    gap = (end_mo.reindex(first.index) - first).dropna()
    gap = gap[gap >= 0]
    print(f"\n  flagging {pct}% of visits — {len(gap):,} of {len(prog_men):,} "
          f"progressors ever flagged ({len(gap)/len(prog_men):.0%})")
    for q in (25, 50, 75, 90):
        print(f"    p{q:<3} {gap.quantile(q/100):>6.1f} months before progression")
    print(f"    share warned >= 6 months ahead: {(gap >= 6).mean():.0%}")

print("\n  this is measured over ALL a man's landmarks, so it is not capped by")
print("  the 6-month label window -- the honest warning time.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"landmark_clean | A={a_all:.3f} "
      f"| B_no_util={res['B  no utilisation / timing']:.3f} "
      f"| C_shape={res['C  trajectory shape only (no level)']:.3f} "
      f"| D_level_only={res['D  absolute PSA level alone']:.3f} "
      f"| util_cost={res['B  no utilisation / timing']-a_all:+.3f}")

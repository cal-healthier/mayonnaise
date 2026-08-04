"""
gbm_baseline.py -- establish the REAL baseline before any transformer.

0.743 came from logistic regression. A transformer beating a linear model
proves nothing. Gradient boosting on identical features, identical folds,
identical C-index is the bar that actually has to be cleared.

Also fits a trajectory-flavoured feature set (slopes + variability) to see how
much of the "sequence" signal is reachable WITHOUT a sequence model -- that
tells us how much headroom a transformer is even competing for.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- inputs
try:
    labs, T, m                       # left in the notebook by the survival run
    print("using in-memory labs / T / m")
except NameError:
    import glob, os
    print("no in-memory frames; looking on disk:", sorted(glob.glob("*.parquet"))[:12])
    raise SystemExit("re-run the survival extraction cell first, then this")

labs_nc = labs[[c for c in labs.columns if not c.endswith("__count")]]
base = labs_nc.join(T, how="outer").join(m[["surv_days", "event"]], how="inner")
base = base.dropna(axis=1, how="all")
print(f"cohort {len(base):,}   deaths {int(base.event.sum()):,} "
      f"({base.event.mean():.0%})   features {base.shape[1]-2}")

# ---------------------------------------------------------------- C-index
def cindex(t, e, risk, n_pairs=3_000_000, seed=0):
    """Harrell's C by sampling comparable pairs (full O(n^2) is too big here)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(risk, float)
    n = len(t); conc = perm = 0
    for _ in range(0, n_pairs, 500_000):
        i = rng.integers(0, n, 500_000); j = rng.integers(0, n, 500_000)
        # comparable: the earlier of the two must be an observed event
        ei = (t[i] < t[j]) & e[i]
        ej = (t[j] < t[i]) & e[j]
        ok = ei | ej
        if not ok.any():
            continue
        # whoever fails first should carry the higher risk
        first_i = ei[ok]
        ri, rj = r[i[ok]], r[j[ok]]
        hi = np.where(first_i, ri, rj)      # risk of the one who failed first
        lo = np.where(first_i, rj, ri)
        conc += np.sum(hi > lo) + 0.5 * np.sum(hi == lo)
        perm += ok.sum()
    return conc / perm

def run(df, label, model="lr", folds=5):
    fcols = [c for c in df.columns if c not in ("surv_days", "event")]
    scores = []
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(df):
        a, b = df.iloc[tr], df.iloc[te]
        if model == "lr":
            imp = SimpleImputer(strategy="median").fit(a[fcols])
            sc = StandardScaler().fit(imp.transform(a[fcols]))
            mdl = LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5)
            mdl.fit(sc.transform(imp.transform(a[fcols])), a.event)
            p = mdl.predict_proba(sc.transform(imp.transform(b[fcols])))[:, 1]
        else:
            mdl = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15, random_state=0)
            mdl.fit(a[fcols], a.event)          # handles NaN natively
            p = mdl.predict_proba(b[fcols])[:, 1]
        scores.append(cindex(b.surv_days, b.event, p))
    s = np.array(scores)
    print(f"  {label:<44}{s.mean():.3f} ± {s.std():.3f}")
    return s.mean()

print("\n" + "=" * 72)
print("SAME FEATURES, SAME FOLDS, SAME METRIC")
print("=" * 72)
lr_all  = run(base, "logistic regression  (the current 0.743)", "lr")
gbm_all = run(base, "gradient boosting", "gbm")

print("\n  component sets, gradient boosting:")
lab_only = labs_nc.join(m[["surv_days", "event"]], how="inner").dropna(axis=1, how="all")
tum_only = T.join(m[["surv_days", "event"]], how="inner").dropna(axis=1, how="all")
run(lab_only, "labs only", "gbm")
run(tum_only, "tumor + site only", "gbm")

# ------------------------------------------------- how much trajectory is reachable?
# slope + variability per feature: cheap proxies for what a sequence model sees
print("\n" + "=" * 72)
print("HOW MUCH 'SEQUENCE' SIGNAL IS REACHABLE WITHOUT A SEQUENCE MODEL?")
print("=" * 72)
stems = sorted({c.rsplit("__", 1)[0] for c in labs.columns if "__" in c})
traj = {}
for s in stems:
    last, first = f"{s}__last", f"{s}__first"
    mx, mn, mean = f"{s}__max", f"{s}__min", f"{s}__mean"
    if last in labs.columns and mean in labs.columns:
        traj[f"{s}__delta"] = labs[last] - labs[mean]        # drift from own average
    if mx in labs.columns and mn in labs.columns:
        traj[f"{s}__range"] = labs[mx] - labs[mn]            # variability
if traj:
    TR = pd.DataFrame(traj, index=labs.index)
    print(f"  derived {TR.shape[1]} trajectory features from {len(stems)} labs")
    plus = base.join(TR, how="left")
    gbm_traj = run(plus, "gbm + drift/variability features", "gbm")
    print(f"\n  gain from cheap trajectory features: {gbm_traj - gbm_all:+.3f}")
else:
    gbm_traj = gbm_all
    print("  no first/min/max columns found - skipped")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"gbm_baseline | n={len(base)} | lr={lr_all:.3f} | gbm={gbm_all:.3f} "
      f"| gbm_traj={gbm_traj:.3f} | gain={gbm_all-lr_all:+.3f}")

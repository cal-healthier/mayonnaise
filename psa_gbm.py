"""
psa_gbm.py -- does 0.714 survive a stronger model, and what is driving it?

1. Gradient boosting vs logistic regression, same features, folds, metric.
2. Which blood tests actually matter (permutation importance -- the honest
   measure for a tree model). Does alkaline phosphatase lead here the way it
   did on the nadir endpoint? Consistent biology across two different endpoints
   is real corroboration; a completely different feature set is a warning.
3. How much trajectory signal is reachable WITHOUT a sequence model -- drift
   (last - mean) and variability (max - min) per test. Sets the headroom a
   transformer would be competing for.

Cached parquet only. No BigQuery, no cost.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
E = pd.read_parquet("psa_progression.parquet")
labs = globals().get("labs")
if labs is None:
    labs = pd.read_parquet("psa_labs.parquet")
print(f"men {len(E):,}   progressed {int(E['prog'].sum()):,} "
      f"({E['prog'].mean():.1%})   lab columns {labs.shape[1]}")

def cindex(t, e, risk, n_pairs=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(risk, float)
    n = len(t); conc = perm = 0
    for _ in range(0, n_pairs, 500_000):
        i = rng.integers(0, n, 500_000); j = rng.integers(0, n, 500_000)
        ei = (t[i] < t[j]) & e[i]; ej = (t[j] < t[i]) & e[j]
        ok = ei | ej
        if not ok.any():
            continue
        ri, rj = r[i[ok]], r[j[ok]]
        hi = np.where(ei[ok], ri, rj); lo = np.where(ei[ok], rj, ri)
        conc += np.sum(hi > lo) + 0.5 * np.sum(hi == lo); perm += ok.sum()
    return conc / perm

def prep(df):
    X = df.select_dtypes(include=[np.number]).reindex(E.index)
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)

def run(X, name, model="lr"):
    y, t = E["prog"], E["time"]
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        if model == "lr":
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
            sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
            m = LogisticRegression(class_weight="balanced", max_iter=4000, C=0.5)
            m.fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
            p = m.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]
        else:
            m = gbm().fit(X.iloc[tr], y.iloc[tr])       # NaN handled natively
            p = m.predict_proba(X.iloc[te])[:, 1]
        s.append(cindex(t.iloc[te], y.iloc[te], p))
    a = np.array(s)
    print(f"  {name:<42}C-index {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} feat)")
    return a.mean()

B = pd.DataFrame({"baseline": E["baseline"], "log_baseline": np.log1p(E["baseline"])})
LB = prep(labs)
BOTH = prep(B.join(LB, how="left"))
sets = {"PSA at treatment start only": prep(B),
        "routine bloodwork only": LB,
        "PSA + bloodwork": BOTH}

print("\n" + "=" * 72)
print("LOGISTIC REGRESSION  (last run's numbers, for reference)")
print("=" * 72)
lr = {k: run(v, k, "lr") for k, v in sets.items()}

print("\n" + "=" * 72)
print("GRADIENT BOOSTING  (the real bar)")
print("=" * 72)
gb = {k: run(v, k, "gbm") for k, v in sets.items()}
print(f"\n  gain from the stronger model on PSA+bloodwork: "
      f"{gb['PSA + bloodwork'] - lr['PSA + bloodwork']:+.3f}")

# ---------------------------------------------------------------- what matters
print("\n" + "=" * 72)
print("WHICH BLOOD TESTS ACTUALLY MATTER?  (permutation importance)")
print("=" * 72)
Xtr, Xte, ytr, yte = train_test_split(BOTH, E["prog"], test_size=0.25,
                                      stratify=E["prog"], random_state=0)
m = gbm().fit(Xtr, ytr)
pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=5,
                            random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=BOTH.columns).nlargest(18)
for k, v in imp.items():
    print(f"    {k:<34}{v:+.4f}")

stems = pd.Series(pi.importances_mean, index=BOTH.columns)
bytest = stems.groupby(stems.index.str.split("__").str[0]).sum().nlargest(12)
print("\n  rolled up per blood test:")
for k, v in bytest.items():
    print(f"    {k:<34}{v:+.4f}")

# ---------------------------------------------------------------- trajectory headroom
print("\n" + "=" * 72)
print("HOW MUCH TRAJECTORY SIGNAL WITHOUT A SEQUENCE MODEL?")
print("=" * 72)
traj = {}
for s in sorted({c.rsplit("__", 1)[0] for c in labs.columns if "__" in c}):
    last, mean = f"{s}__last", f"{s}__mean"
    mx, mn = f"{s}__max", f"{s}__min"
    if last in labs.columns and mean in labs.columns:
        traj[f"{s}__drift"] = labs[last] - labs[mean]
    if mx in labs.columns and mn in labs.columns:
        traj[f"{s}__range"] = labs[mx] - labs[mn]
if traj:
    TR = pd.DataFrame(traj, index=labs.index)
    print(f"  derived {TR.shape[1]} drift/variability features")
    plus = prep(B.join(LB, how="left").join(TR, how="left"))
    g2 = run(plus, "PSA + bloodwork + drift/variability", "gbm")
    print(f"\n  gain from cheap trajectory features: {g2 - gb['PSA + bloodwork']:+.3f}")
else:
    g2 = gb["PSA + bloodwork"]
    print("  no min/max/last columns -- skipped")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"psa_gbm | n={len(E)} | events={int(E['prog'].sum())} "
      f"| lr_both={lr['PSA + bloodwork']:.3f} | gbm_both={gb['PSA + bloodwork']:.3f} "
      f"| gbm_blood={gb['routine bloodwork only']:.3f} | gbm_traj={g2:.3f} "
      f"| top={bytest.index[0]}")

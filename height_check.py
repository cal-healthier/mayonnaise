"""
height_check.py -- height is the 3rd most important predictor. That is a
red flag, not a finding.

An adult's height does not change, so height__max and height__min mattering
SEPARATELY means within-patient variation is carrying signal -- which is
measurement junk, mixed units, or zeros, not biology.

1. Look at the actual height values.
2. Refit without height. Then without all vitals. See what 0.756 depends on.

Cached parquet only. No cost.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

pd.set_option("display.width", 200)
E = pd.read_parquet("psa_progression.parquet")
labs = globals().get("labs")
if labs is None:
    labs = pd.read_parquet("psa_labs.parquet")

# ---------------------------------------------------------------- 1. what IS height?
hcols = [c for c in labs.columns if "height" in c.lower()]
print("=" * 72)
print("WHAT ARE THESE HEIGHT VALUES?")
print("=" * 72)
print(f"  height columns: {hcols}")
H = labs[hcols]
print(f"\n  {H.notna().any(axis=1).sum():,} of {len(labs):,} men have any height "
      f"({H.notna().any(axis=1).mean():.0%})")
print("\n  distribution:")
print(H.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2).to_string())

zero_like = (H <= 1).sum()
print(f"\n  values <= 1 (impossible for a person):")
for c, n in zero_like.items():
    print(f"    {c:<26}{n:>6}")

if "sign_height__max" in labs.columns and "sign_height__min" in labs.columns:
    spread = (labs["sign_height__max"] - labs["sign_height__min"]).dropna()
    print(f"\n  within-patient height spread (max - min), should be ~0:")
    for q in [50, 75, 90, 95, 99]:
        print(f"    p{q:<3} {spread.quantile(q/100):>8.2f}")
    print(f"    men whose recorded height varies by >5 units: "
          f"{(spread > 5).sum():,} ({(spread > 5).mean():.0%})")
    print(f"    men whose recorded height varies by >20 units: "
          f"{(spread > 20).sum():,} ({(spread > 20).mean():.0%})   <- mixed units?")
    print("\n  progression rate by whether height even recorded:")
    has_h = labs["sign_height__max"].reindex(E.index).notna()
    for lab, sel in [("height recorded", has_h), ("no height", ~has_h)]:
        if sel.sum():
            print(f"    {lab:<18} n={int(sel.sum()):>5}  progressed "
                  f"{E.loc[sel.values, 'prog'].mean():6.1%}")

# ---------------------------------------------------------------- 2. refit without it
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

def run(X, name):
    y, t = E["prog"], E["time"]
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        s.append(cindex(t.iloc[te], y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(s)
    print(f"  {name:<46}C-index {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} feat)")
    return a.mean()

B = pd.DataFrame({"baseline": E["baseline"], "log_baseline": np.log1p(E["baseline"])})
no_h = [c for c in labs.columns if "height" not in c.lower()]
no_vit = [c for c in labs.columns if not c.lower().startswith("sign_")]
bloods = [c for c in no_vit]

print("\n" + "=" * 72)
print("WHAT DOES 0.756 ACTUALLY DEPEND ON?")
print("=" * 72)
full = run(prep(B.join(labs, how="left")), "everything (last run's 0.756)")
noh  = run(prep(B.join(labs[no_h], how="left")), "without height")
nov  = run(prep(B.join(labs[no_vit], how="left")), "without any vitals (blood tests only)")
bo   = run(prep(labs[bloods]), "blood tests only, no PSA, no vitals")
print(f"\n  cost of dropping height:  {noh - full:+.3f}")
print(f"  cost of dropping vitals:  {nov - full:+.3f}")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"height_check | full={full:.3f} | no_height={noh:.3f} | no_vitals={nov:.3f} "
      f"| blood_only={bo:.3f} | h_spread_gt20="
      f"{int((labs.get('sign_height__max', pd.Series(dtype=float)) - labs.get('sign_height__min', pd.Series(dtype=float)) > 20).sum())}")

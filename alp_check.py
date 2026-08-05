"""
alp_check.py -- test the bone-metastasis explanation, both cohorts, same method.

Claim: bloodwork won in prostate because alkaline phosphatase detects bone
metastases, and lost in ovarian because ovarian spreads peritoneally with no
routine-lab equivalent.

Three ways this claim could be wrong, all checked here:
  1. ALP simply is not MEASURED in the ovarian women -> data artefact, not biology.
  2. ALP is measured but its VALUES do not differ -> nothing to detect, fine.
  3. ALP still ranks highly in ovarian and just does not help -> explanation wrong.

Same permutation-importance method on both cohorts, height dropped from both
so the panels are comparable. Cached parquet only, no BigQuery.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 200)

P_E = pd.read_parquet("psa_progression.parquet")
P_L = pd.read_parquet("psa_labs.parquet")
O_E = pd.read_parquet("ov_label.parquet")
O_L = pd.read_parquet("ov_labs.parquet")
for L in (P_L, O_L):
    L.drop(columns=[c for c in L.columns if "height" in c.lower()], inplace=True, errors="ignore")
print(f"prostate {len(P_E):,} men,   {int(P_E['prog'].sum()):,} events, "
      f"{P_L.shape[1]} lab cols")
print(f"ovarian  {len(O_E):,} women, {int(O_E['prog'].sum()):,} events, "
      f"{O_L.shape[1]} lab cols")

# ---------------------------------------------------------------- 1. is ALP even there?
print("\n" + "=" * 74)
print("IS ALKALINE PHOSPHATASE MEASURED IN BOTH COHORTS?")
print("=" * 74)
def alp_cols(L):
    return [c for c in L.columns if "alp" in c.lower() or "alk" in c.lower()]
for name, E, L in [("prostate", P_E, P_L), ("ovarian", O_E, O_L)]:
    ac = alp_cols(L)
    if not ac:
        print(f"  {name:<10} NO ALP COLUMN AT ALL  <- explanation would be a data artefact")
        continue
    col = [c for c in ac if c.endswith("__max")] or ac
    s = L[col[0]].reindex(E.index)
    print(f"  {name:<10} {col[0]:<20} measured in {s.notna().sum():>5,}/{len(E):>5,} "
          f"({s.notna().mean():.0%})   median {s.median():.0f}  "
          f"p90 {s.quantile(.9):.0f}  p99 {s.quantile(.99):.0f}")
    # does ALP separate progressors from non-progressors?
    hi = s[E["prog"] == 1].median(); lo = s[E["prog"] == 0].median()
    print(f"  {'':<10} median ALP: progressed {hi:.0f} vs still-responding {lo:.0f}"
          f"   (ratio {hi/lo:.2f}x)" if lo and not np.isnan(lo) else "")

# ---------------------------------------------------------------- 2. importance, both
def importance(E, L, label):
    B = pd.DataFrame({"marker": E["baseline"], "log_marker": np.log1p(E["baseline"])})
    X = B.join(L, how="left").reindex(E.index).select_dtypes(include=[np.number])
    X = X.loc[:, X.notna().sum() > len(X) * 0.05]
    y = E["prog"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0).fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=8,
                                random_state=0, n_jobs=-1)
    s = pd.Series(pi.importances_mean, index=X.columns)
    rolled = s.groupby(s.index.str.split("__").str[0]).sum().sort_values(ascending=False)
    print(f"\n  {label} — importance rolled up per test:")
    for k, v in rolled.head(12).items():
        print(f"    {k:<26}{v:+.4f}")
    return rolled

print("\n" + "=" * 74)
print("WHAT MATTERS IN EACH CANCER  (same method, same settings)")
print("=" * 74)
p_imp = importance(P_E, P_L, "PROSTATE (bloodwork WON, +0.088)")
o_imp = importance(O_E, O_L, "OVARIAN (bloodwork LOST, -0.067)")

# ---------------------------------------------------------------- 3. verdict
print("\n" + "=" * 74)
print("SIDE BY SIDE")
print("=" * 74)
def rank_of(imp, key="Chem_ALP"):
    hits = [i for i, k in enumerate(imp.index, 1) if key.lower() in str(k).lower()]
    return (hits[0], imp.iloc[hits[0] - 1]) if hits else (None, None)

for name, imp in [("prostate", p_imp), ("ovarian", o_imp)]:
    r, v = rank_of(imp)
    mr, mv = rank_of(imp, "marker")
    print(f"  {name:<10} ALP rank {str(r):<5} importance {v if v is None else f'{v:+.4f}'}"
          f"    |  marker rank {str(mr):<5} {mv if mv is None else f'{mv:+.4f}'}")

pr, pv = rank_of(p_imp); orank, ov = rank_of(o_imp)
print("\n  verdict: ", end="")
if pv is not None and ov is not None:
    if pv > 0.005 and ov < 0.002:
        print("SUPPORTED — ALP carries the prostate signal and does nothing in ovarian.")
    elif ov >= 0.002:
        print("NOT SUPPORTED — ALP matters in ovarian too; the bone-mets story is wrong.")
    else:
        print("AMBIGUOUS — ALP is weak in both; something else drove prostate.")
else:
    print("cannot judge — ALP missing from one panel")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"alp_check | prost_alp_rank={pr} val={pv if pv is None else round(float(pv),4)} "
      f"| ovar_alp_rank={orank} val={ov if ov is None else round(float(ov),4)} "
      f"| prost_top={p_imp.index[0]} | ovar_top={o_imp.index[0]}")

"""
psa_model2.py -- fix and verify.

1. Ridge instead of plain linear regression (the R² = -103 was my bug:
   ~380 collinear lab columns with no regularization -> coefficients explode).
2. Better label: nadir <= 0.2 ng/mL, the standard clinical definition of a
   complete response to hormone therapy. Independent of starting PSA, so it
   rescues the 1,557 men the 50%-drop rule mislabeled.
3. VERIFY the AUROC 0.760: is PSA itself hiding in the bloodwork panel? which
   features drive it? does it survive with far fewer features?

Reads cached parquet only -- no BigQuery, no cost.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
L = pd.read_parquet("psa_label.parquet").rename(columns={"pct_change": "drop_pct"})
labs = globals().get("labs")
if labs is None:
    raise SystemExit("run psa_model.py first in this kernel (needs `labs`)")

# ---------------------------------------------------------------- is PSA in the panel?
psa_cols = [c for c in labs.columns if "psa" in c.lower() or "prostate" in c.lower()]
print(f"lab panel: {labs.shape[1]} columns, {labs.shape[0]:,} men")
print(f"PSA-derived columns inside the panel: {psa_cols if psa_cols else 'NONE'}")

# ---------------------------------------------------------------- two labels
L["resp_drop"] = (L["nadir"] <= 0.5 * L["baseline"]).astype(int)      # old
L["resp_nadir"] = (L["nadir"] <= 0.2).astype(int)                     # clinical standard
print(f"\nfull cohort n={len(L):,}")
print(f"  50%-drop rule      responders {L['resp_drop'].mean():.1%}  "
      f"non-responders {int((1-L['resp_drop']).sum()):,}")
print(f"  nadir <= 0.2 rule  responders {L['resp_nadir'].mean():.1%}  "
      f"non-responders {int((1-L['resp_nadir']).sum()):,}")
print("\n  agreement between the two labels: "
      f"{(L['resp_drop']==L['resp_nadir']).mean():.1%}")

print("\n  nadir<=0.2 response rate by starting PSA (should be flat-ish now):")
L["bl_bin"] = pd.cut(L["baseline"], [0, 0.5, 1, 2, 4, 10, 20, 100, 1e9],
                     labels=["<0.5", "0.5-1", "1-2", "2-4", "4-10", "10-20", "20-100", ">100"])
for b, r in L.groupby("bl_bin", observed=True).agg(
        n=("resp_nadir", "size"), rate=("resp_nadir", "mean")).iterrows():
    print(f"    {str(b):<8} n={int(r['n']):>5}  respond {r['rate']:6.1%}")

# ---------------------------------------------------------------- helpers
def prep(df, idx):
    X = df.select_dtypes(include=[np.number]).reindex(idx)
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

def auc(X, y, name, C=0.5, show=False):
    scores, coefs = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        m = LogisticRegression(class_weight="balanced", max_iter=4000, C=C)
        m.fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[te],
                      m.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]))
        coefs.append(m.coef_[0])
    a = np.array(scores)
    print(f"  {name:<46}AUROC {a.mean():.3f} ± {a.std():.3f}   ({X.shape[1]} features)")
    if show:
        imp_ = pd.Series(np.abs(np.mean(coefs, axis=0)), index=X.columns).nlargest(15)
        print("\n     strongest predictors:")
        for k, v in imp_.items():
            print(f"       {k:<34}{v:.3f}")
        print()
    return a.mean()

def r2(X, y, name):
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
        m = Ridge(alpha=10.0).fit(sc.transform(imp.transform(X.iloc[tr])), y.iloc[tr])
        s.append(r2_score(y.iloc[te], m.predict(sc.transform(imp.transform(X.iloc[te])))))
    a = np.array(s)
    print(f"  {name:<46}R²    {a.mean():.3f} ± {a.std():.3f}")
    return a.mean()

# ---------------------------------------------------------------- run on FULL cohort
idx = L.index
B = pd.DataFrame({"baseline": L["baseline"], "log_baseline": np.log1p(L["baseline"])})
LB = prep(labs, idx)
sets = {"PSA at treatment start only": B,
        "routine bloodwork only (no PSA)": LB,
        "PSA + bloodwork": B.join(LB, how="left")}
y = L["resp_nadir"]

print("\n" + "=" * 74)
print(f"COMPLETE RESPONSE (nadir <= 0.2)   n={len(L):,}, {y.mean():.0%} respond, "
      f"{int((1-y).sum()):,} non-responders")
print("=" * 74)
res = {}
for k, v in sets.items():
    res[k] = auc(prep(v, idx), y, k, show=(k == "routine bloodwork only (no PSA)"))

print("  same, with heavy regularization (overfitting check):")
auc(prep(LB, idx), y, "bloodwork only, C=0.02", C=0.02)
top = prep(LB, idx).notna().sum().nlargest(20).index
auc(prep(LB, idx)[top], y, "bloodwork, 20 best-populated features only", C=0.5)

print("\n" + "=" * 74)
print("HOW LOW DOES PSA GO?  (continuous, Ridge -- fixes the -103)")
print("=" * 74)
yc = np.log1p(L["nadir"])
rc = {k: r2(prep(v, idx), yc, k) for k, v in sets.items()}

print("\n" + "=" * 74)
print("OLD 50%-DROP LABEL, same features (for comparison)")
print("=" * 74)
yd = L["resp_drop"]
for k, v in sets.items():
    auc(prep(v, idx), yd, k)

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"psa_model2 | n={len(L)} | resp_nadir={y.mean():.0%} "
      f"| auc_psa={res['PSA at treatment start only']:.3f} "
      f"| auc_blood={res['routine bloodwork only (no PSA)']:.3f} "
      f"| auc_both={res['PSA + bloodwork']:.3f} "
      f"| r2_both={rc['PSA + bloodwork']:.3f} | psa_in_panel={len(psa_cols)}")

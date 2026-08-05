"""
landmark_model.py -- the early-warning model, done as landmarks.

The lead-time curves showed ALP drifting up ~4-7% from month -5 while PSA is
still falling, and controls dead flat over the same window. Small in the median
-- but a median hides a subset rising sharply. That is what individual-level
modelling finds and a group curve cannot.

Setup (the part that matters more than the architecture):
  at EVERY month on treatment, build features from what is known UP TO THEN,
  label = progression within the next 6 months. One row per patient-month, so
  5,636 men become ~100k rows. This is the question as a clinician asks it:
  "given everything up to today, what is the risk over the next six months?"

Model: gradient boosting on named trajectory features. NOT a transformer --
929 events, and every measurement today said engineered features + GBM wins.

Strict no-leakage: at landmark m only data with day <= m is used.
Grouped CV by patient so no man appears in both train and test.
Benchmarked against the clinical rule (PCWG3) applied at the same landmarks.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

pd.set_option("display.width", 240)
HORIZON, STEP, MIN_M, MAX_M = 6, 1, 3, 48

E = pd.read_parquet("psa_progression.parquet")
pv = pd.read_parquet("psa_values.parquet")
lab = pd.read_parquet("psa_labs_during.parquet")
tx = pv.groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
lab = lab.join(E["tx_date"], on="clinic").dropna(subset=["tx_date"])
lab["mo"] = ((lab["d"] - lab["tx_date"]).dt.days / 30.44)
pv = pv.copy(); pv["mo"] = pv["day"] / 30.44
print(f"{len(E):,} men | {len(lab):,} lab rows | {len(pv):,} PSA rows")

FEATS = ["Chem_ALP", "Chem_ALT", "Chem_ALB", "CBC_Hct", "CBC_WBC",
         "CBC_RBC", "CBC_PLT", "CBC_MCV"]
lab = lab[lab["feature"].isin(FEATS)]
LW = {f: g.sort_values("mo") for f, g in lab.groupby("feature")}
PW = {c: g.sort_values("mo") for c, g in pv.groupby("clinic")}
LC = {f: {c: g for c, g in d.groupby("clinic")} for f, d in LW.items()}

def slope(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return np.nan
    return np.polyfit(x, y, 1)[0]

rows = []
for clinic, e in E.iterrows():
    end_mo = e["time"] / 30.44
    p = PW.get(clinic)
    if p is None or len(p) < 4:
        continue
    for m in np.arange(MIN_M, min(end_mo, MAX_M) + 1e-9, STEP):
        # label: progression inside the next HORIZON months, else censored/no
        if e["prog"] == 1:
            lab_y = 1 if (end_mo - m) <= HORIZON else 0
        else:
            if (end_mo - m) < HORIZON:
                break                      # not followed long enough to know
            lab_y = 0
        r = {"clinic": clinic, "mo": m, "y": lab_y}
        ph = p[p["mo"] <= m]
        if len(ph) < 3:
            continue
        cur = ph["psa"].iloc[-1]; nad = ph["psa"].min()
        r["psa_cur"] = np.log1p(cur)
        r["psa_vs_nadir"] = np.log((cur + .01) / (nad + .01))
        r["psa_vs_base"] = np.log((cur + .01) / (ph["psa"].iloc[0] + .01))
        r["mo_since_nadir"] = m - ph.loc[ph["psa"].idxmin(), "mo"]
        r["mo_on_tx"] = m
        r["n_psa"] = len(ph)
        for w in (3, 6):
            s = ph[ph["mo"] >= m - w]
            r[f"psa_slope{w}"] = slope(s["mo"].values, np.log1p(s["psa"].values))
        for f in FEATS:
            g = LC.get(f, {}).get(clinic)
            if g is None:
                continue
            h = g[g["mo"] <= m]
            if len(h) < 2:
                continue
            base = h[h["mo"] <= 3]["value"].median()
            v = h["value"].iloc[-1]
            r[f"{f}_cur"] = v
            if base and not np.isnan(base) and base != 0:
                r[f"{f}_vs_base"] = v / base
            for w in (3, 6):
                s = h[h["mo"] >= m - w]
                r[f"{f}_slope{w}"] = slope(s["mo"].values, s["value"].values)
        rows.append(r)

L = pd.DataFrame(rows)
print(f"\nlandmarks: {len(L):,} patient-months over {L['clinic'].nunique():,} men")
print(f"  positive (progression within {HORIZON}mo): {int(L['y'].sum()):,} "
      f"({L['y'].mean():.1%})")

fc = [c for c in L.columns if c not in ("clinic", "mo", "y")]
PSA_ONLY = [c for c in fc if c.startswith(("psa_", "mo_", "n_psa"))]
LAB_ONLY = [c for c in fc if c.startswith(("Chem_", "CBC_"))]
print(f"  features: {len(PSA_ONLY)} PSA/time, {len(LAB_ONLY)} lab, {len(fc)} total")

def run(cols, name):
    X, y, g = L[cols], L["y"], L["clinic"]
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=60, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=0).fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(aucs)
    print(f"  {name:<44}AUROC {a.mean():.3f} ± {a.std():.3f}  ({len(cols)} feat)")
    return a.mean()

print("\n" + "=" * 82)
print(f"WILL THIS MAN PROGRESS IN THE NEXT {HORIZON} MONTHS?")
print("=" * 82)
# the clinical rule as a predictor: PSA >=25% and >=2 above nadir, right now
rule = ((np.exp(L["psa_vs_nadir"]) >= 1.25) &
        (np.expm1(L["psa_cur"]) - np.expm1(L["psa_cur"]) / np.exp(L["psa_vs_nadir"]) >= 2))
print(f"  {'PCWG3 rule applied at the landmark':<44}"
      f"AUROC {roc_auc_score(L['y'], rule.astype(int)):.3f}  (the current standard)")
a1 = run(PSA_ONLY, "PSA trajectory only")
a2 = run(LAB_ONLY, "routine labs only")
a3 = run(fc, "PSA + labs  (the early-warning model)")
print(f"\n  labs add {a3 - a1:+.3f} over the PSA trajectory alone")

print("\n" + "=" * 82)
print("WHICH FEATURES CARRY IT?")
print("=" * 82)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(L[fc], L["y"], test_size=.25,
                                      stratify=L["y"], random_state=0)
mm = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
    l2_regularization=1.0, early_stopping=True, validation_fraction=.15,
    random_state=0).fit(Xtr, ytr)
pi = permutation_importance(mm, Xte, yte, scoring="roc_auc", n_repeats=5,
                            random_state=0, n_jobs=-1)
for k, v in pd.Series(pi.importances_mean, index=fc).nlargest(15).items():
    print(f"    {k:<26}{v:+.4f}")

L.to_parquet("landmarks.parquet")
print("\n  saved landmarks.parquet")
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"landmark_model | rows={len(L)} | pos={L['y'].mean():.1%} "
      f"| rule={roc_auc_score(L['y'], rule.astype(int)):.3f} | psa={a1:.3f} "
      f"| labs={a2:.3f} | both={a3:.3f} | gain={a3-a1:+.3f}")

"""
alert_or_risk.py -- is this an EARLY WARNING or just a RISK SCORE?

The decomposition said current PSA alone gets 0.751 of the 0.800, so the model
is largely a thermometer. And the lead-time p90 of 31.8 months is suspicious:
a man flagged three years out probably was not warned -- he just ran a high PSA
the whole time. That is standing risk stratification, not an alert.

The difference decides how this can be described:
  ALERT  = the score RISES before progression. "Something changed, look now."
  RISK   = the score is flat-high throughout. "This man was always high risk."

Three tests:
  1. Trajectory of the score itself in the 24 months before progression,
     versus men still responding over the same window.
  2. Split flagged progressors into those whose score ROSE into the alert
     versus those already above threshold from their very first landmark.
  3. Does the CHANGE in score add anything over its LEVEL? If a rise carries
     independent information, that is the alert component, isolated.

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
fc = [c for c in L.columns if c not in ("clinic", "mo", "y", "_nadir", "_cur")]
fc = [c for c in fc if L[c].nunique(dropna=True) >= 2]

oof = pd.Series(index=L.index, dtype=float)
for tr, te in GroupKFold(5).split(L[fc], L["y"], groups=L["clinic"]):
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0).fit(L[fc].iloc[tr], L["y"].iloc[tr])
    oof.iloc[te] = m.predict_proba(L[fc].iloc[te])[:, 1]
L = L.copy(); L["score"] = oof
print(f"{len(L):,} landmarks scored out-of-fold")

end_mo = E["time"] / 30.44
L["end"] = end_mo.reindex(L["clinic"]).values
L["prog"] = E["prog"].reindex(L["clinic"]).values
L["to_end"] = L["end"] - L["mo"]

# ---------------------------------------------------------------- 1. score trajectory
print("\n" + "=" * 80)
print("1. DOES THE SCORE RISE BEFORE PROGRESSION?")
print("=" * 80)
print(f"  {'months before':>14}{'progressors':>14}{'still responding':>20}")
for lo, hi in [(24, 18), (18, 12), (12, 9), (9, 6), (6, 3), (3, 1), (1, 0)]:
    a = L[(L["prog"] == 1) & (L["to_end"] < lo) & (L["to_end"] >= hi)]["score"]
    b = L[(L["prog"] == 0) & (L["to_end"] < lo) & (L["to_end"] >= hi)]["score"]
    if len(a) < 30:
        continue
    print(f"  {f'{lo}-{hi}':>14}{a.median():>14.4f}"
          f"{(b.median() if len(b) > 30 else np.nan):>20.4f}")
print("\n  a rising column = alert. a flat-high column = standing risk score.")

# ---------------------------------------------------------------- 2. rose vs always-high
print("\n" + "=" * 80)
print("2. OF MEN FLAGGED, HOW MANY ROSE INTO IT vs WERE ALWAYS ABOVE?")
print("=" * 80)
for pct in (5, 10):
    th = L["score"].quantile(1 - pct / 100)
    P = L[(L["prog"] == 1)].sort_values(["clinic", "mo"])
    rose = already = never = 0
    leads_rose = []
    for c, g in P.groupby("clinic", sort=False):
        s = g["score"].values
        if (s >= th).sum() == 0:
            never += 1
            continue
        first = int(np.argmax(s >= th))
        if first == 0:
            already += 1
        else:
            rose += 1
            leads_rose.append(g["end"].iloc[0] - g["mo"].iloc[first])
    tot = rose + already + never
    print(f"\n  flagging {pct}% of visits  (n={tot:,} progressors)")
    print(f"    rose into the alert (a real warning): {rose:>5,} ({rose/tot:.0%})")
    print(f"    above threshold from first landmark : {already:>5,} ({already/tot:.0%})")
    print(f"    never flagged                       : {never:>5,} ({never/tot:.0%})")
    if leads_rose:
        lr = pd.Series(leads_rose)
        print(f"    lead time among those who ROSE: p25 {lr.quantile(.25):.1f}  "
              f"p50 {lr.quantile(.5):.1f}  p75 {lr.quantile(.75):.1f} months")

# ---------------------------------------------------------------- 3. change vs level
print("\n" + "=" * 80)
print("3. DOES THE *CHANGE* IN SCORE ADD ANYTHING OVER ITS LEVEL?")
print("=" * 80)
L = L.sort_values(["clinic", "mo"])
g = L.groupby("clinic")["score"]
L["score_d3"] = L["score"] - g.shift(3)
L["score_d6"] = L["score"] - g.shift(6)
sub = L.dropna(subset=["score_d3", "score_d6"])
print(f"  usable landmarks: {len(sub):,}")
def auc(cols, name):
    a = []
    for tr, te in GroupKFold(5).split(sub[cols], sub["y"], groups=sub["clinic"]):
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(sub[cols].iloc[tr], sub["y"].iloc[tr])
        a.append(roc_auc_score(sub["y"].iloc[te],
                               m.predict_proba(sub[cols].iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<38}AUROC {a.mean():.3f} ± {a.std():.3f}")
    return a.mean()
l1 = auc(["score"], "score LEVEL only")
l2 = auc(["score_d3", "score_d6"], "score CHANGE only (3 and 6 mo)")
l3 = auc(["score", "score_d3", "score_d6"], "level + change")
print(f"\n  change adds {l3 - l1:+.3f} over level alone")
print("  a real gain => there is an alert component. ~zero => pure risk score.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"alert_or_risk | level={l1:.3f} | change={l2:.3f} | both={l3:.3f} "
      f"| change_gain={l3-l1:+.3f}")

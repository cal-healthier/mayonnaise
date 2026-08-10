"""
what_is_it_reading.py -- 768 anonymous dimensions beat the lab panel. What are
they picking up?

You cannot read coefficients off an embedding. So: CONCEPT PROBING. Embed ~45
named clinical ideas, measure how close each man's notes sit to each idea, and
test three things per concept:

  1. does it track the MODEL's risk score?      -> what the model uses
  2. does it track the OUTCOME directly?        -> what is actually prognostic
  3. does it survive adjustment for the labs?   -> what the TEXT contributes
                                                   that the numbers do not

(3) is the study. A concept that predicts progression and is NOT explained by
the lab panel is the thing the narrative carries and the measurements miss.

Then: section ablation (which part of a note holds it) and a reader for the
highest- and lowest-risk men.

All local, all free. Uses the cached numerals-removed embeddings -- the
headline arm.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

pd.set_option("display.width", 250)
E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
N = pd.read_parquet("tvn_notes.parquet")
V = pd.read_parquet("tvn_emb_nonum.parquet")          # the winning arm
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
E = E[E.index.isin(V.index)]
V = V.reindex(E.index)
print(f"{len(E):,} men | {int(E['y'].sum())} events | embeddings {V.shape}")

# out-of-fold risk from the text model
oof = pd.Series(index=E.index, dtype=float)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(V, E["y"]):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06,
        max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.,
        random_state=0).fit(V.iloc[tr], E["y"].iloc[tr])
    oof.iloc[te] = m.predict_proba(V.iloc[te])[:, 1]
print(f"text model OOF AUROC check: "
      f"{stats.rankdata(oof)[E['y']==1].mean()/len(E):.3f} (rank-based)")

CONCEPTS = {
 "function: walks with difficulty":      "He has difficulty walking and uses a cane for support.",
 "function: bedbound or wheelchair":     "The patient is largely confined to bed or a wheelchair.",
 "function: fully independent":          "He remains fully independent in all activities of daily living.",
 "function: still working":              "He continues to work full time without limitation.",
 "function: needs help at home":         "He requires assistance at home with daily tasks.",
 "symptom: fatigue":                     "He reports increasing fatigue and low energy.",
 "symptom: pain":                        "He describes worsening pain requiring medication.",
 "symptom: bone pain":                   "He has new pain in his back and hips.",
 "symptom: weight loss":                 "He has lost weight without trying.",
 "symptom: poor appetite":               "His appetite has been poor.",
 "symptom: feels well":                  "He feels well and has no new complaints.",
 "social: lives alone":                  "He lives alone at home.",
 "social: spouse present":               "His wife accompanies him to the visit.",
 "social: family involved":              "His family is closely involved in his care.",
 "social: travels far":                  "He travels a long distance for his appointments.",
 "social: limited support":              "He has limited support at home.",
 "care: urgent or unscheduled":          "He was seen urgently outside his usual schedule.",
 "care: recent admission":               "He was recently admitted to hospital.",
 "care: emergency visit":                "He presented to the emergency department.",
 "care: routine follow up":              "This is a routine scheduled follow-up visit.",
 "care: close monitoring planned":       "We will monitor him closely and see him back soon.",
 "care: long interval planned":          "He will return in six months for routine review.",
 "engagement: declined treatment":       "He declined the recommended treatment.",
 "engagement: missed appointments":      "He has missed several appointments.",
 "engagement: wishes to continue":       "He wishes to continue with current therapy.",
 "engagement: prefers less aggressive":  "He prefers a less aggressive approach.",
 "clinician: expresses concern":         "I am concerned about his current trajectory.",
 "clinician: reassured":                 "I am reassured by his progress.",
 "clinician: uncertain":                 "It is unclear how he will respond.",
 "clinician: considered alternatives":   "We discussed several options before deciding.",
 "clinician: guideline standard":        "Treatment is per standard of care.",
 "psych: anxious":                       "He appears anxious about his diagnosis.",
 "psych: low mood":                      "He has been feeling down and withdrawn.",
 "psych: coping well":                   "He is coping well emotionally.",
 "comorbid: cardiac":                    "He has a history of heart disease.",
 "comorbid: diabetes":                   "He has diabetes managed with medication.",
 "comorbid: renal":                      "He has reduced kidney function.",
 "comorbid: multiple problems":          "He has multiple significant medical problems.",
 "comorbid: otherwise healthy":          "He is otherwise healthy with no other conditions.",
 "disease: bone involvement":            "There is disease involving the bones.",
 "disease: nodal involvement":           "There is involvement of the lymph nodes.",
 "disease: localized":                   "The disease appears confined to the prostate.",
 "disease: extensive":                   "He has extensive disease burden.",
 "goals: palliative framing":            "The goal of treatment is comfort and quality of life.",
 "goals: curative framing":              "We are treating with curative intent.",
}
mdl = SentenceTransformer("models/pubmedbert")
Q = mdl.encode(list(CONCEPTS.values()), normalize_embeddings=True)
Vn = V.values / (np.linalg.norm(V.values, axis=1, keepdims=True) + 1e-9)
SIM = pd.DataFrame(Vn @ Q.T, index=E.index, columns=list(CONCEPTS))

cols = [c for c in labs.columns if c.endswith("__last")]
S = labs[cols].reindex(E.index); S["psa"] = E["baseline"]
S = S.loc[:, S.notna().sum() > len(S)*.3]
Sf = S.fillna(S.median())
Sz = ((Sf - Sf.mean()) / (Sf.std() + 1e-9)).values

rows = []
for c in CONCEPTS:
    x = SIM[c].values
    r_model = stats.spearmanr(x, oof.values).correlation
    r_out = stats.pointbiserialr(E["y"].values, x).correlation
    # does it add beyond the labs? logistic with labs vs labs+concept
    base = LogisticRegression(max_iter=2000).fit(Sz, E["y"])
    aug = LogisticRegression(max_iter=2000).fit(
        np.c_[Sz, stats.zscore(x)], E["y"])
    from sklearn.metrics import roc_auc_score
    d = (roc_auc_score(E["y"], aug.decision_function(np.c_[Sz, stats.zscore(x)]))
         - roc_auc_score(E["y"], base.decision_function(Sz)))
    rows.append({"concept": c, "tracks_model": r_model,
                 "predicts_outcome": r_out, "adds_beyond_labs": d})
R = pd.DataFrame(rows)

print("\n" + "=" * 92)
print("WHAT THE TEXT CONTRIBUTES THAT THE LABS DO NOT  (ranked by column 3)")
print("=" * 92)
print(f"  {'concept':<38}{'tracks model':>14}{'predicts outcome':>18}{'adds beyond labs':>18}")
for _, r in R.reindex(R["adds_beyond_labs"].abs().sort_values(ascending=False).index).head(18).iterrows():
    print(f"  {r['concept']:<38}{r['tracks_model']:>+14.3f}"
          f"{r['predicts_outcome']:>+18.3f}{r['adds_beyond_labs']:>+18.4f}")

print("\n  the third column is the study: a concept that predicts progression")
print("  and is NOT explained by the lab panel is what the narrative carries.")

print("\n" + "=" * 92)
print("MOST ASSOCIATED WITH THE MODEL'S OWN RISK SCORE")
print("=" * 92)
for lbl, asc in [("higher risk", False), ("lower risk", True)]:
    print(f"\n  {lbl}:")
    for _, r in R.sort_values("tracks_model", ascending=asc).head(8).iterrows():
        print(f"    {r['concept']:<40}{r['tracks_model']:+.3f}")

R.to_csv("concept_probe.csv", index=False)
print("\nwrote concept_probe.csv")

# ---------------------------------------------------------------- read the extremes
E2 = E.copy(); E2["risk"] = oof
hi = E2.nlargest(6, "risk").index; lo = E2.nsmallest(6, "risk").index
_pool = [("HIGHEST RISK", c) for c in hi] + [("LOWEST RISK", c) for c in lo]
_i = 0
def extreme():
    """read a note from the men the model scored most extremely"""
    global _i
    if _i >= len(_pool):
        print("  end of pool"); return
    lbl, c = _pool[_i]; _i += 1
    r = E2.loc[c]
    print("\n" + "=" * 74)
    print(f"{lbl}  risk={r['risk']:.3f}  actually progressed={'YES' if r['y'] else 'no'}")
    print("=" * 74)
    import textwrap
    t = N[N["clinic"] == c]["txt"].iloc[0][:1800]
    for para in t.split("\n"):
        for l in textwrap.wrap(para, 70) or [""]:
            print(f"  {l}")
    print("\n  -> extreme() for the next")

print("\n  call  extreme()  in a new cell to read the men the model scored")
print("  most extremely. STAYS IN THE ENCLAVE.")
print("\n" + "-" * 74)
print("FINAL LINE:")
top = R.reindex(R["adds_beyond_labs"].abs().sort_values(ascending=False).index).iloc[0]
print(f"reading | men={len(E)} | top_concept={top['concept']} "
      f"| adds={top['adds_beyond_labs']:+.4f} "
      f"| max_outcome_r={R['predicts_outcome'].abs().max():.3f}")

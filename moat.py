"""
moat.py -- does grounding in real Mayo outcomes beat an unguided LLM?

This is the head-to-head that matters commercially. A generic model can reason
about a case and cite guidelines. It cannot know what happened to 5,000 similar
patients. If that knowledge measurably improves prediction, the data is a moat
and there is a number to put on it.

Task: given everything known at treatment start, will this man progress within
12 months?

  A  LLM, patient record only            <- "a random OpenAI query"
  B  LLM + 10 similar Mayo patients and their ACTUAL outcomes
  C  gradient boosting on the same features   <- the statistical floor

Same patients, same held-out split, same metric. Retrieval draws only from the
training half, so nothing leaks.
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

N_TEST, K = 150, 10
E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]          # censored before 12mo: unknowable
print(f"{len(E):,} men | {int(E['y'].sum()):,} progress within 12 months "
      f"({E['y'].mean():.0%})")

FEATS = ["Chem_ALP", "Chem_ALT", "Chem_ALB", "CBC_Hb", "CBC_Hct", "CBC_WBC",
         "CBC_PLT", "CBC_RBC"]
cols = [c for c in labs.columns if any(c.startswith(f) for f in FEATS)
        and c.endswith("__last")]
X = labs[cols].reindex(E.index)
X["baseline_psa"] = E["baseline"]
X = X.loc[:, X.notna().sum() > len(X) * .3]
tr, te = train_test_split(E.index, test_size=N_TEST, stratify=E["y"], random_state=0)
print(f"train {len(tr):,} | test {len(te)}")

# ---------------------------------------------------------------- C: the floor
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06,
        min_samples_leaf=40, l2_regularization=1., random_state=0)
m.fit(X.loc[tr], E.loc[tr, "y"])
p_gbm = m.predict_proba(X.loc[te])[:, 1]
auc_c = roc_auc_score(E.loc[te, "y"], p_gbm)
print(f"\nC  gradient boosting: AUROC {auc_c:.3f}")

# ---------------------------------------------------------------- patient card
def card(i):
    r, l = E.loc[i], X.loc[i]
    bits = [f"PSA at treatment start: {r['baseline']:.1f} ng/mL"]
    for c in X.columns:
        if c == "baseline_psa" or pd.isna(l[c]):
            continue
        bits.append(f"{c.replace('__last','').replace('Chem_','').replace('CBC_','')}"
                    f": {l[c]:.1f}")
    return "; ".join(bits)

# ---------------------------------------------------------------- retrieval
Xn = X.fillna(X.median())
Xz = (Xn - Xn.mean()) / (Xn.std() + 1e-9)
TR = Xz.loc[tr].values
def neighbours(i, k=K):
    d = np.linalg.norm(TR - Xz.loc[i].values, axis=1)
    idx = np.argsort(d)[:k]
    out = []
    for j in idx:
        pid = tr[j]
        out.append(f"- {card(pid)}  ->  "
                   f"{'PROGRESSED within 12 months' if E.loc[pid,'y'] else 'did NOT progress within 12 months'}")
    return "\n".join(out)

# ---------------------------------------------------------------- A and B
from google import genai
from google.genai import types
cl = genai.Client(vertexai=True,
                  project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                         "mcp-acc-055-dbg-p-7e23"),
                  location="global")
cfg = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=1024),
    response_mime_type="application/json", temperature=0)

SCHEMA = ('Return ONLY {"risk": <0.0-1.0 probability he progresses within 12 '
          'months>, "reason": "<one sentence>"}')

def ask(prompt):
    try:
        r = cl.models.generate_content(model="gemini-3.5-flash",
                                       contents=prompt, config=cfg)
        return float(json.loads(r.text)["risk"])
    except Exception:
        return np.nan

pa, pb, t0 = [], [], time.time()
for n, i in enumerate(te):
    base = (f"A man with prostate cancer is starting androgen deprivation "
            f"therapy.\n\nHis values at treatment start:\n{card(i)}\n\n"
            f"Will he develop castration resistance (PSA progression) within "
            f"12 months?\n\n{SCHEMA}")
    pa.append(ask(base))
    pb.append(ask(base.replace("Will he develop",
        f"Here are {K} similar patients from this institution and what actually "
        f"happened to them:\n\n{neighbours(i)}\n\nWill he develop")))
    if n % 25 == 0:
        print(f"  {n}/{len(te)}  ({time.time()-t0:.0f}s)", end="\r")
print(f"  done in {time.time()-t0:.0f}s" + " " * 20)

y = E.loc[te, "y"].values
pa, pb = np.array(pa), np.array(pb)
ok = ~(np.isnan(pa) | np.isnan(pb))
auc_a = roc_auc_score(y[ok], pa[ok])
auc_b = roc_auc_score(y[ok], pb[ok])

print("\n" + "=" * 74)
print("DOES THE DATA MAKE THE MODEL BETTER?")
print("=" * 74)
print(f"  A  LLM alone (a generic query)         AUROC {auc_a:.3f}")
print(f"  B  LLM + real Mayo outcomes            AUROC {auc_b:.3f}")
print(f"  C  gradient boosting                   AUROC {auc_c:.3f}")
print(f"\n  grounding is worth {auc_b-auc_a:+.3f}")
print(f"  best LLM arm vs statistical model: {max(auc_a,auc_b)-auc_c:+.3f}")
print(f"  parsed {ok.sum()}/{len(te)}")

print("\n  spread of predictions (a flat model is not reasoning, it is guessing):")
for lbl, p in [("A alone", pa[ok]), ("B grounded", pb[ok])]:
    print(f"    {lbl:<14}p10 {np.percentile(p,10):.2f}  median "
          f"{np.median(p):.2f}  p90 {np.percentile(p,90):.2f}")

print("""
  B > A means the outcome base is a real moat: the same model reasons better
  because it can see what happened to comparable patients. That is the thing a
  competitor cannot replicate by querying a bigger LLM.
  B ~ A means the retrieval added nothing and the moat has to come from
  somewhere else -- worth knowing now.""")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"moat | n={ok.sum()} | llm_alone={auc_a:.3f} | llm_grounded={auc_b:.3f} "
      f"| gbm={auc_c:.3f} | grounding_gain={auc_b-auc_a:+.3f}")

"""
rules_rerun_cheap.py -- same question, ~25% of the token spend.

Thinking mode costs roughly 7x. Running all 1,055 clauses through it is wasteful
when the question is narrow: did thinking unlock criteria that were previously
BLOCKED?

  - 241 clauses already came out executable -> only need a small control sample
    to check they do not regress
  - 814 were blocked -> these are the only ones that can change, so sample them
  - thinking_budget 2048 instead of 4096 (this task is far lighter than reading
    a 4,000-character clinical note)

330 clauses instead of 1,055, at half the thinking budget. Then extrapolate to
the full set with the sampling error stated, so the number is honest about its
own precision.
"""
import json, os, time
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
old = pd.read_csv("executable_rules.csv")
old["was"] = old["solid"].astype(bool)
n_ok, n_blocked = int(old["was"].sum()), int((~old["was"]).sum())
base = old["was"].mean()
print(f"original run: {len(old):,} clauses, {n_ok} executable ({base:.0%}), "
      f"{n_blocked} blocked")

N_BLOCK, N_CTRL, THINK = 250, 80, 2048
rng = np.random.default_rng(0)
s_block = old[~old["was"]].sample(min(N_BLOCK, n_blocked), random_state=0)
s_ctrl = old[old["was"]].sample(min(N_CTRL, n_ok), random_state=0)
S = pd.concat([s_block, s_ctrl])
print(f"sampling {len(s_block)} blocked + {len(s_ctrl)} previously-executable "
      f"= {len(S)} clauses")
print(f"  ~{len(S)/len(old):.0%} of the clauses, thinking budget {THINK} (was 4096)")
print(f"  rough saving vs the full run: ~75-80% of the token spend")

FIELDS = {
    "labs": ["hemoglobin","hematocrit","platelets","wbc","neutrophils_abs",
        "lymphocytes_abs","creatinine","egfr","bun","sodium","potassium","calcium",
        "magnesium","glucose","albumin","total_protein","bilirubin_total","ast","alt",
        "alk_phos","ldh","inr","ptt","tsh","cea","ca_15_3","psa","weight","height","bmi"],
    "tumor/registry": ["primary_site","histology","stage_group","t_stage","n_stage",
        "m_stage","grade","nodes_positive","nodes_examined","tumor_size_mm",
        "date_of_diagnosis","er_status","pr_status","her2_status","oncotype_score"],
    "demographics": ["age_at_index","sex","race","ethnicity","vital_status","date_of_death"],
    "treatment": ["drug_name","drug_class","treatment_start_date","treatment_end_date",
        "surgery_type","surgery_date","radiation_flag","radiation_date"],
    "encounters": ["ecog_score (2017+ only)","encounter_date","admission_flag"],
    "MEDICATIONS (FACT_ORDERS, 2.0B orders — NEW, was unavailable last time)": [
        "any_drug_generic_name","drug_order_date","steroid_flag","anticoagulant_flag",
        "diabetes_drug_flag","immunosuppressant_flag"],
}
INV = "\n".join(f"  {k}: {', '.join(v)}" for k, v in FIELDS.items())
SCHEMA = """Return ONLY a JSON array, one object per numbered criterion:
{"i": <index>, "executable": true|false,
 "field": "<exact field from the inventory, or null>",
 "op": "<one of >=,<=,>,<,==,!=,in,exists,absent, or null>",
 "value": "<threshold as a string, or null>"}
executable=true ONLY if checkable from the listed fields AND it has a concrete
comparable value. Vague criteria with no threshold are executable=false."""

from google import genai
from google.genai import types
cl = genai.Client(vertexai=True,
                  project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                         "mcp-acc-055-dbg-p-7e23"),
                  location="global")
cfg = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=THINK),
    response_mime_type="application/json", temperature=0)

out, T, calls = [], S["clause"].tolist(), 0
t0 = time.time()
print("\nre-running the sample with thinking ...")
for i in range(0, len(T), 25):
    ch = T[i:i + 25]
    listed = "\n".join(f"{j}. {c[:350]}" for j, c in enumerate(ch))
    try:
        r = cl.models.generate_content(model="gemini-3.5-flash",
            contents=(f"You are building a clinical-trial eligibility engine.\n"
                      f"AVAILABLE PATIENT DATA FIELDS:\n{INV}\n\n{SCHEMA}\n\n"
                      f"CRITERIA:\n{listed}"), config=cfg)
        calls += 1
        got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
        out += [got.get(j, {"executable": False}) for j in range(len(ch))]
    except Exception as e:
        print(f"  batch {i}: {type(e).__name__} {str(e)[:70]}")
        out += [{"executable": False}] * len(ch)
    print(f"  ...{min(i+25, len(T))}/{len(T)}", end="\r")
    time.sleep(0.1)
R = pd.DataFrame(out)
S = S.reset_index(drop=True)
S["now"] = (R.get("executable", pd.Series([False]*len(S))).fillna(False).astype(bool)
            & R.get("field").notna() & R.get("op").notna())
print(f"  done: {calls} API calls in {time.time()-t0:.0f}s")

blk = S[~S["was"]]; ctl = S[S["was"]]
rescue = blk["now"].mean()
keep = ctl["now"].mean()
se = np.sqrt(rescue * (1 - rescue) / max(len(blk), 1))
est = (n_ok * keep + n_blocked * rescue) / len(old)
lo = (n_ok * keep + n_blocked * (rescue - 1.96*se)) / len(old)
hi = (n_ok * keep + n_blocked * (rescue + 1.96*se)) / len(old)

print("\n" + "=" * 76)
print("DID THINKING UNLOCK BLOCKED CRITERIA?")
print("=" * 76)
print(f"  previously blocked, now executable   {int(blk['now'].sum())}/{len(blk)}"
      f"  ({rescue:.0%})")
print(f"  previously executable, still is      {int(ctl['now'].sum())}/{len(ctl)}"
      f"  ({keep:.0%})   <- regression check")
print(f"\n  original executable rate            {base:>8.0%}")
print(f"  estimated with thinking             {est:>8.0%}   "
      f"[95% CI {lo:.0%}-{hi:.0%}]")
print(f"  change                              {est-base:>+8.0%}")

if len(blk[blk["now"]]):
    print("\n  which fields do the newly-executable criteria use?")
    nf = blk[blk["now"]].join(R[["field"]].rename(columns={"field": "nf"}))
    for f, n in nf["nf"].value_counts().head(8).items():
        print(f"    {str(f):<30}{int(n):>4}")
    print("\n  what were they blocked as before?")
    for b, n in blk[blk["now"]]["blocker"].value_counts().head(6).items():
        print(f"    {str(b):<30}{int(n):>4}")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
if est - base >= 0.10:
    print(f"  The 23% was substantially a configuration artefact. At ~{est:.0%},")
    print("  trial matching from structured data is more viable than I said, and")
    print("  Study 1 (the eligibility audit) deserves promoting.")
elif est - base >= 0.04:
    print(f"  Modest real gain to ~{est:.0%}. The conclusion stands directionally")
    print("  but the number should be restated.")
else:
    print(f"  ~{est:.0%}: no material change. The 23% was a property of the data,")
    print("  not my configuration, and the original conclusion holds.")
if keep < 0.9:
    print(f"\n  CAUTION: only {keep:.0%} of previously-executable clauses survived.")
    print("  Thinking made it more conservative; the gain may be partly a wash.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"rules_cheap | sampled={len(S)} | calls={calls} | rescue={rescue:.0%} "
      f"| keep={keep:.0%} | old={base:.0%} | est={est:.0%} | change={est-base:+.0%}")

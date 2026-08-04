"""
extract_rules.py -- the real test for a matching engine.

Not "what topic is this criterion" (that was the wrong question) but
"can I turn it into an EXECUTABLE rule against fields we actually have?"

Gemini gets our real field inventory and must either emit a structured rule
or say which data source it would need. Self-validating: a rule either has a
field+operator+value or it doesn't.

Public trial text only - no patient data leaves anything.
"""
import json
import os
import time
from collections import Counter

import pandas as pd

work = pd.read_csv("criteria_classified_v2.csv")
work = work[work.llm != "ERROR"].reset_index(drop=True)
print(f"{len(work)} criteria clauses from {work.nct.nunique()} trials\n")

# ------------------------------------------------- what we ACTUALLY have
FIELDS = {
    "labs (numeric, longitudinal, any date)": [
        "hemoglobin","hematocrit","platelets","wbc","neutrophils_abs","lymphocytes_abs",
        "creatinine","egfr","bun","sodium","potassium","calcium","magnesium","glucose",
        "albumin","total_protein","bilirubin_total","ast","alt","alk_phos","ldh",
        "inr","ptt","tsh","cea","ca_15_3","psa","weight","height","bmi"],
    "tumor / registry": [
        "primary_site","histology","stage_group","t_stage","n_stage","m_stage",
        "grade","nodes_positive","nodes_examined","tumor_size_mm","date_of_diagnosis",
        "er_status","pr_status","her2_status","oncotype_score"],
    "demographics": ["age_at_index","sex","race","ethnicity","vital_status","date_of_death"],
    "treatment": ["drug_name","drug_class","treatment_start_date","treatment_end_date",
                  "surgery_type","surgery_date","radiation_flag","radiation_date"],
    "encounters": ["ecog_score (2017+ only)","encounter_date","admission_flag"],
}
INV = "\n".join(f"  {k}: {', '.join(v)}" for k, v in FIELDS.items())

SCHEMA = """Return ONLY a JSON array, one object per numbered criterion:
{"i": <index>,
 "executable": true|false,
 "field": "<exact field name from the inventory, or null>",
 "op": "<one of >=,<=,>,<,==,!=,in,exists,absent, or null>",
 "value": "<threshold/value as a string, or null>",
 "blocker": "<if not executable, ONE of: needs_clinical_note, needs_imaging,
              needs_pathology_report, needs_genomics, vague_no_threshold,
              not_a_data_question, field_not_available>"}

Rules:
- executable=true ONLY if the criterion can be checked using the listed fields
  alone AND has a concrete, comparable value. A vague criterion like "adequate
  organ function" with no threshold is executable=false, blocker=vague_no_threshold.
- A section header or sub-list stem is executable=false, not_a_data_question.
- Consent/willingness/contraception are not_a_data_question."""

def extract(clauses, batch=20):
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True,
                          project=os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23"),
                          location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json", temperature=0)
    out = []
    for i in range(0, len(clauses), batch):
        chunk = clauses[i:i + batch]
        listed = "\n".join(f"{j}. {c[:400]}" for j, c in enumerate(chunk))
        prompt = (f"You are building a clinical-trial eligibility engine.\n"
                  f"AVAILABLE PATIENT DATA FIELDS:\n{INV}\n\n{SCHEMA}\n\nCRITERIA:\n{listed}")
        try:
            r = client.models.generate_content(model="gemini-3.5-flash", contents=prompt, config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {"executable": False, "blocker": "PARSE_MISS"}) for j in range(len(chunk))]
        except Exception as e:
            print(f"   batch {i}: {type(e).__name__} {str(e)[:90]}")
            out += [{"executable": False, "blocker": "ERROR"}] * len(chunk)
        print(f"   ...{min(i+batch, len(clauses))}/{len(clauses)}", end="\r")
        time.sleep(0.2)
    return out

print("extracting executable rules (a few minutes) ...")
res = extract(work.clause.tolist())
work["executable"] = [bool(r.get("executable")) for r in res]
work["field"] = [r.get("field") for r in res]
work["op"] = [r.get("op") for r in res]
work["value"] = [r.get("value") for r in res]
work["blocker"] = [r.get("blocker") for r in res]

# an "executable" rule with no field or no op is not actually executable
solid = work.executable & work.field.notna() & work.op.notna()
work["solid"] = solid

print("\n" + "=" * 72)
print("CAN WE BUILD AN EXECUTABLE RULE?")
print("=" * 72)
print(f"  claimed executable      {work.executable.sum():>5} ({work.executable.mean():.0%})")
print(f"  with a real field+op    {solid.sum():>5} ({solid.mean():.0%})   <-- the honest number")

print("\n  why the rest are not executable:")
for b, n in work.loc[~solid, "blocker"].value_counts().items():
    print(f"    {str(b):<26}{n:>5} ({n/(~solid).sum():4.0%})")

print("\n" + "=" * 72)
print("WHICH FIELDS DO TRIALS ACTUALLY DEMAND? (top 25)")
print("=" * 72)
for f, n in work.loc[solid, "field"].value_counts().head(25).items():
    trials = work[solid & (work.field == f)].nct.nunique()
    print(f"  {str(f):<26}{n:>4} rules   in {trials:>2}/{work.nct.nunique()} trials")

# per-trial: how much of each trial can we actually automate?
per = work.groupby("nct").agg(total=("clause", "size"), exec_=("solid", "sum"))
per["frac"] = per.exec_ / per.total
print("\n" + "=" * 72)
print("PER-TRIAL AUTOMATION")
print("=" * 72)
print(f"  median share of a trial's criteria we can auto-check: {per.frac.median():.0%}")
print(f"  trials where we can check >= half:  {(per.frac >= .5).sum()}/{len(per)}")
print(f"  trials where we can check nothing:  {(per.exec_ == 0).sum()}/{len(per)}")

work.to_csv("executable_rules.csv", index=False)
print("\nwrote executable_rules.csv")

print("\n  sample extracted rules:")
for _, r in work[solid].head(12).iterrows():
    print(f"    {r.field:<22} {str(r.op):<6} {str(r.value)[:18]:<20} <- {r.clause[:58]}")

print("\n" + "-" * 70)
print("FINAL LINE:")
top = work.loc[solid, "field"].value_counts()
print(f"extract_rules | clauses={len(work)} | solid={solid.mean():.0%} "
      f"| per_trial_median={per.frac.median():.0%} | zero_trials={(per.exec_==0).sum()} "
      f"| top_field={top.index[0] if len(top) else 'none'}")

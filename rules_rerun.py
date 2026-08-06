"""
rules_rerun.py -- was the 23% an artefact of my configuration?

The ablation just showed thinking_budget=0 cost 23 points on a comparable
extraction task (60% -> 83%). I ran EVERY LLM call today with thinking off,
including the trial-criteria rule extraction, which concluded:

    23% of eligibility criteria are machine-executable
    median 24% of a single trial's criteria checkable

That number drove a real strategic conclusion -- that trial matching was
unviable from structured data and note extraction was the critical path. If it
was configuration rather than a property of the data, that conclusion was wrong.

Same 1,055 clauses, same model, same field inventory, same prompt. Only the
thinking budget changes. Then compare.
"""
import json, os, time
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
old = pd.read_csv("executable_rules.csv")
print(f"{len(old):,} criteria clauses from the original run")
print(f"  original: solid={old['solid'].mean():.0%} "
      f"(thinking disabled)")

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
    "MEDICATIONS (FACT_ORDERS, 2.0B orders)": [
        "any_drug_generic_name","drug_order_date","steroid_flag","anticoagulant_flag",
        "diabetes_drug_flag","immunosuppressant_flag"],
}
INV = "\n".join(f"  {k}: {', '.join(v)}" for k, v in FIELDS.items())
SCHEMA = """Return ONLY a JSON array, one object per numbered criterion:
{"i": <index>, "executable": true|false,
 "field": "<exact field from the inventory, or null>",
 "op": "<one of >=,<=,>,<,==,!=,in,exists,absent, or null>",
 "value": "<threshold as a string, or null>",
 "blocker": "<if not executable: needs_clinical_note|needs_imaging|
              needs_pathology_report|needs_genomics|vague_no_threshold|
              not_a_data_question|field_not_available>"}
executable=true ONLY if checkable from the listed fields AND it has a concrete
comparable value. Vague criteria with no threshold are executable=false."""

def run(think, tag, batch=20):
    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=think),
        response_mime_type="application/json", temperature=0)
    out, T = [], old["clause"].tolist()
    t0 = time.time()
    for i in range(0, len(T), batch):
        ch = T[i:i + batch]
        listed = "\n".join(f"{j}. {c[:400]}" for j, c in enumerate(ch))
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash",
                contents=(f"You are building a clinical-trial eligibility engine.\n"
                          f"AVAILABLE PATIENT DATA FIELDS:\n{INV}\n\n{SCHEMA}\n\n"
                          f"CRITERIA:\n{listed}"), config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {"executable": False}) for j in range(len(ch))]
        except Exception as e:
            out += [{"executable": False}] * len(ch)
        print(f"   {tag} ...{min(i+batch, len(T))}/{len(T)}", end="\r")
        time.sleep(0.1)
    print(f"   {tag} done in {time.time()-t0:.0f}s" + " " * 20)
    return pd.DataFrame(out)

print("\nre-running WITH thinking (this is the slow one) ...")
new = run(4096, "thinking")
N = old.copy()
N["n_exec"] = [bool(r) for r in new.get("executable", pd.Series([False]*len(old)))]
N["n_field"] = new.get("field")
N["n_op"] = new.get("op")
N["n_solid"] = N["n_exec"] & N["n_field"].notna() & N["n_op"].notna()

print("\n" + "=" * 78)
print("EXECUTABLE CRITERIA: thinking off vs on")
print("=" * 78)
print(f"  {'':<34}{'thinking OFF':>16}{'thinking ON':>14}")
print(f"  {'clauses called executable':<34}{old['solid'].mean():>16.0%}"
      f"{N['n_solid'].mean():>14.0%}")
gain = N["n_solid"].mean() - old["solid"].mean()
print(f"\n  change: {gain:+.0%}")

print("\n  per-trial coverage (the number that decides the matching engine):")
for lab, col in [("thinking OFF", "solid"), ("thinking ON", "n_solid")]:
    per = N.groupby("nct")[col].mean()
    print(f"    {lab:<14}median {per.median():.0%} of a trial's criteria | "
          f">=half in {int((per>=.5).sum())}/{per.size} trials")

print("\n  criteria newly executable with thinking:")
newly = N[~N["solid"].astype(bool) & N["n_solid"]]
print(f"    {len(newly):,} clauses")
if len(newly):
    for f, n in newly["n_field"].value_counts().head(10).items():
        print(f"      {str(f):<26}{int(n):>5}")
    print("\n    they were previously blocked as:")
    for b, n in newly["blocker"].value_counts().head(6).items():
        print(f"      {str(b):<26}{int(n):>5}")

lost = N[N["solid"].astype(bool) & ~N["n_solid"]]
print(f"\n  criteria LOST with thinking (was executable, now not): {len(lost):,}")
print("  a big number here means thinking made it more conservative, not better.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"rules_rerun | clauses={len(N)} | old={old['solid'].mean():.0%} "
      f"| new={N['n_solid'].mean():.0%} | change={gain:+.0%} "
      f"| newly={len(newly)} | lost={len(lost)}")

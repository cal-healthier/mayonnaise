"""
classify_v2.py -- classify trial eligibility clauses, two ways.

  A) fixed rules  (the trailing-\\b bug broke every stem pattern; + new categories)
  B) Gemini 3.5 Flash

Then compare them. Public trial text only - no patient data.
"""
import json
import os
import re
import time

import pandas as pd

cl = pd.read_csv("trial_criteria_classified.csv")[["nct", "kind", "clause"]]
print(f"{len(cl)} clauses from {cl.nct.nunique()} trials")

# ---------------------------------------------------------------- drop headers
HEADER = re.compile(r"^(inclusion|exclusion)\s+criteria\s*:?\s*$", re.I)
def is_header(c):
    c = c.strip()
    return bool(HEADER.match(c)) or (c.endswith(":") and len(c) < 120)
cl["header"] = cl.clause.map(is_header)
print(f"section headers / sub-list stems (not real criteria): {cl.header.sum()}")
work = cl[~cl.header].reset_index(drop=True)
print(f"real criteria clauses: {len(work)}\n")

# ---------------------------------------------------------------- A) fixed rules
# NOTE: no trailing \b after word stems - that was the bug.
PATTERNS = [
    ("pregnancy_contraception", r"pregnan|breast ?feed|lactat|contracept|childbearing|abstinen"),
    ("lab_organ_function",      r"\b(anc|absolute neutrophil|platelet|h(a)?emoglobin|creatinine|"
                                r"bilirubin|ast|alt|sgot|sgpt|transaminase|gfr|albumin|inr|"
                                r"leukocyte|wbc|bone marrow function|organ function|renal function|"
                                r"hepatic function)\b|×\s*10\d|/µl|/ul"),
    ("performance_status",      r"ecog|karnofsky|performance status|zubrod"),
    ("age",                     r"\bage[ds]?\b|years of age|years old"),
    ("biomarker",               r"her2|estrogen receptor|progesterone receptor|hormone receptor|"
                                r"triple negative|egfr|\balk\b|pd-?l1|braf|kras|brca|msi|mmr|"
                                r"mutation|amplification|immunohistochem|receptor status"),
    ("prior_therapy",           r"prior |previously (treated|received)|refractory|relapsed|"
                                r"lines? of (therapy|treatment)|na(i)?ve|pretreated|progressed on|"
                                r"resistant|last dose|previous (therapy|treatment|chemo)"),
    ("concomitant_meds",        r"cyp3a|concomitant|concurrent (medication|therapy|treatment)|"
                                r"strong (inhibitor|inducer)|drug[- ]drug interaction|"
                                r"currently receiving"),
    ("surgery_procedure",       r"mastectom|lumpectom|reconstruct|resect|surger|surgical|"
                                r"radiotherap|radiation therapy|aln[dc]|biops"),
    ("second_cancer",           r"other (types? of )?(cancer|malignan)|second primary|"
                                r"prior malignan|nonmelanoma skin"),
    ("menopausal_status",       r"menopaus|amenorrhea|oophorectom"),
    ("brain_cns",               r"brain metasta|cns metasta|leptomening|cerebral metasta"),
    ("cardiac",                 r"lvef|ejection fraction|cardiac|myocardial|qtc?\b|long qt|"
                                r"heart failure|arrhythm"),
    ("comorbidity",            r"autoimmune|vascular disease|vasculiti|uncontrolled|"
                                r"cognitive impairment|psychiatric|comorbid|chronic|"
                                r"history of clinically significant"),
    ("infection",               r"\bhiv\b|hepatitis|active infection|tuberculos"),
    ("measurable_disease",      r"measurable disease|recist|evaluable disease|target lesion"),
    ("diagnosis_stage",         r"histolog|cytolog|confirm|stage [0-4ivx]|metastat|"
                                r"locally advanced|unresectable|diagnos|carcinoma|"
                                r"ajcc|\bt[0-4]\b|\bn[0-3]\b"),
    ("consent_admin",           r"informed consent|consent to|willing|able to comply|"
                                r"life expectancy|protocol|follow-?up visit|investigator|"
                                r"participate"),
]
def rules(text):
    t = text.lower()
    for name, pat in PATTERNS:
        if re.search(pat, t):
            return name
    return "other"
work["rules"] = work.clause.map(rules)
rules_other = (work.rules == "other").mean()
print(f"A) fixed rules      -> unclassified {rules_other:.0%}")

# ---------------------------------------------------------------- B) Gemini
CATS = ["lab_organ_function","performance_status","age","diagnosis_stage","biomarker",
        "prior_therapy","concomitant_meds","surgery_procedure","second_cancer",
        "menopausal_status","brain_cns","cardiac","comorbidity","infection",
        "measurable_disease","pregnancy_contraception","consent_admin","other"]

def gemini_classify(clauses, batch=25):
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True,
                          project=os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23"),
                          location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        temperature=0)
    out = []
    for i in range(0, len(clauses), batch):
        chunk = clauses[i:i + batch]
        listed = "\n".join(f"{j}. {c[:300]}" for j, c in enumerate(chunk))
        prompt = (
            "Classify each clinical-trial eligibility criterion into exactly one category.\n"
            f"Categories: {', '.join(CATS)}\n\n"
            "Return ONLY a JSON array of objects like "
            '[{"i":0,"category":"lab_organ_function"}, ...] with one entry per numbered item.\n\n'
            f"{listed}")
        try:
            r = client.models.generate_content(model="gemini-3.5-flash", contents=prompt, config=cfg)
            got = json.loads(r.text)
            lookup = {int(d["i"]): d["category"] for d in got if "i" in d}
            out += [lookup.get(j, "other") for j in range(len(chunk))]
        except Exception as e:
            print(f"   batch {i}: {type(e).__name__} {str(e)[:90]}")
            out += ["ERROR"] * len(chunk)
        print(f"   ...{min(i+batch, len(clauses))}/{len(clauses)}", end="\r")
        time.sleep(0.2)
    return out

print("\nB) Gemini classifying (this takes a couple of minutes) ...")
work["llm"] = gemini_classify(work.clause.tolist())
llm_ok = (work.llm != "ERROR")
print(f"\nB) gemini           -> unclassified "
      f"{(work.loc[llm_ok,'llm']=='other').mean():.0%}  (errors {(~llm_ok).sum()})")

# ---------------------------------------------------------------- feasibility
FEAS = {
    "lab_organ_function":"evaluable now","age":"evaluable now","diagnosis_stage":"evaluable now",
    "performance_status":"evaluable now","menopausal_status":"evaluable now",
    "biomarker":"breast only","prior_therapy":"needs episode work",
    "concomitant_meds":"needs episode work","second_cancer":"evaluable now",
    "comorbidity":"needs notes","infection":"needs notes","cardiac":"needs notes",
    "surgery_procedure":"needs notes","brain_cns":"needs imaging/notes",
    "measurable_disease":"needs imaging","pregnancy_contraception":"not a data question",
    "consent_admin":"not a data question","other":"unclassified","ERROR":"unclassified",
}
for col, name in [("rules", "RULES"), ("llm", "GEMINI")]:
    d = work[work[col] != "ERROR"].copy()
    d["feas"] = d[col].map(FEAS)
    real = d[d.feas != "not a data question"]
    ev = real.feas.str.startswith("evaluable").sum()
    print(f"\n{'='*70}\n{name}: category breakdown\n{'='*70}")
    for c, n in d[col].value_counts().items():
        print(f"  {c:<26}{n:>5} ({n/len(d):4.0%})   [{FEAS.get(c,'')}]")
    print(f"  --> of {len(real)} real data questions, evaluable today: {ev} ({ev/len(real):.0%})")

agree = (work.rules == work.llm).mean()
work.to_csv("criteria_classified_v2.csv", index=False)
print(f"\nrules vs gemini agreement: {agree:.0%}")
print("wrote criteria_classified_v2.csv")

d = work[work.llm != "ERROR"].copy(); d["feas"] = d.llm.map(FEAS)
real = d[d.feas != "not a data question"]; ev = real.feas.str.startswith("evaluable").sum()
print("\n" + "-"*70)
print("FINAL LINE:")
print(f"classify_v2 | clauses={len(work)} | rules_other={rules_other:.0%} "
      f"| llm_other={(d.llm=='other').mean():.0%} | agreement={agree:.0%} "
      f"| evaluable={ev/len(real):.0%}")

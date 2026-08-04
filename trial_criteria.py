"""
trial_criteria.py -- can we evaluate real trial eligibility criteria?

1. Verify ClinicalTrials.gov is reachable from the enclave.
2. Pull recruiting trials for one cancer.
3. Split each trial's eligibility text into individual criteria clauses.
4. Classify every clause: what KIND of criterion, and can we evaluate it
   against Mayo data today?

Output = a criterion-level feasibility map. No patient data touched.
"""
import json
import re
import ssl
import urllib.parse
import urllib.request
from collections import Counter

import pandas as pd

CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
CONDITION = "breast cancer"     # where our biomarkers are strongest
N_TRIALS = 50

# ---------------------------------------------------------------- 1. fetch
def fetch(n, cond):
    params = {
        "query.cond": cond,
        "filter.overallStatus": "RECRUITING",
        "pageSize": str(n),
    }
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return json.load(r)

print(f"fetching up to {N_TRIALS} recruiting '{CONDITION}' trials ...")
try:
    data = fetch(N_TRIALS, CONDITION)
except Exception as e:
    print(f"  BLOCKED: {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit("ClinicalTrials.gov not reachable -- stop here, tell Claude")

studies = data.get("studies", [])
print(f"  got {len(studies)} studies\n")

rows = []
for s in studies:
    ps = s.get("protocolSection", {})
    nct = ps.get("identificationModule", {}).get("nctId")
    title = ps.get("identificationModule", {}).get("briefTitle", "")
    elig = ps.get("eligibilityModule", {}) or {}
    rows.append({
        "nct": nct,
        "title": title,
        "phase": ",".join(ps.get("designModule", {}).get("phases", []) or []),
        "min_age": elig.get("minimumAge", ""),
        "sex": elig.get("sex", ""),
        "criteria": elig.get("eligibilityCriteria", "") or "",
    })
tr = pd.DataFrame(rows)
tr = tr[tr.criteria.str.len() > 50]
print(f"trials with usable criteria text: {len(tr)}")
print(f"median criteria length: {int(tr.criteria.str.len().median())} chars\n")

# ---------------------------------------------------------------- 2. split into clauses
BULLET = re.compile(r"(?:^|\n)\s*(?:[*\-•‣◦]|\d+[.)])\s+")

def clauses(text):
    parts = BULLET.split(text)
    out = []
    for p in parts:
        p = " ".join(p.split())
        if 15 < len(p) < 600:
            out.append(p)
    return out

records = []
for _, t in tr.iterrows():
    txt = t.criteria
    split_at = re.search(r"exclusion criteria", txt, re.I)
    inc_txt = txt[:split_at.start()] if split_at else txt
    exc_txt = txt[split_at.start():] if split_at else ""
    for c in clauses(inc_txt):
        records.append({"nct": t.nct, "kind": "inclusion", "clause": c})
    for c in clauses(exc_txt):
        records.append({"nct": t.nct, "kind": "exclusion", "clause": c})
cl = pd.DataFrame(records)
print(f"criteria clauses parsed: {len(cl)}  "
      f"({(cl.kind=='inclusion').sum()} inclusion / {(cl.kind=='exclusion').sum()} exclusion)")
print(f"median clauses per trial: {cl.groupby('nct').size().median():.0f}\n")

# ---------------------------------------------------------------- 3. classify
PATTERNS = [
    ("lab_organ_function", r"\b(anc|absolute neutrophil|platelet|h(a)?emoglobin|creatinine|"
                           r"bilirubin|ast\b|alt\b|sgot|sgpt|transaminase|gfr|creatinine clearance|"
                           r"albumin|inr|ptt|leukocyte|wbc|anc\b)\b"),
    ("performance_status", r"\b(ecog|karnofsky|performance status|zubrod)\b"),
    ("age",                r"\b(age[ds]?\b|years of age|\byears old\b|>= ?1[89] years)\b"),
    ("diagnosis_stage",    r"\b(histologically|cytologically|confirmed|stage [0-4ivx]|metastatic|"
                           r"locally advanced|unresectable|diagnosis of)\b"),
    ("biomarker",          r"\b(her2|estrogen receptor|\ber\+|progesterone receptor|\bpr\+|egfr|alk\b|"
                           r"pd-?l1|braf|kras|brca|msi|mmr|hormone receptor|triple negative|mutation|"
                           r"amplification|expression|immunohistochem)\b"),
    ("prior_therapy",      r"\b(prior|previously (treated|received)|refractory|relapsed|"
                           r"lines? of (therapy|treatment)|na(i)?ve|pretreated|progressed on)\b"),
    ("measurable_disease", r"\b(measurable disease|recist|evaluable disease|target lesion)\b"),
    ("brain_cns",          r"\b(brain metasta|cns metasta|leptomening|cerebral metasta)\b"),
    ("cardiac",            r"\b(lvef|ejection fraction|cardiac|myocardial|qtc|heart failure|arrhythm)\b"),
    ("infection_hiv",      r"\b(hiv|hepatitis|active infection|tuberculosis)\b"),
    ("pregnancy",          r"\b(pregnan|breast ?feed|lactat|contracept)\b"),
    ("consent_admin",      r"\b(informed consent|willing|able to comply|life expectancy|"
                           r"protocol|follow-?up visits)\b"),
]

# can we evaluate it against Mayo data TODAY?
FEASIBILITY = {
    "lab_organ_function": "evaluable now",
    "age":                "evaluable now",
    "diagnosis_stage":    "evaluable now",
    "performance_status": "evaluable now (2017+)",
    "biomarker":          "breast only",
    "prior_therapy":      "needs episode work",
    "cardiac":            "needs notes/echo",
    "infection_hiv":      "needs notes",
    "brain_cns":          "needs imaging/notes",
    "measurable_disease": "needs imaging",
    "pregnancy":          "not a data question",
    "consent_admin":      "not a data question",
    "other":              "unclassified",
}

def classify(text):
    t = text.lower()
    for name, pat in PATTERNS:
        if re.search(pat, t):
            return name
    return "other"

cl["category"] = cl.clause.map(classify)
cl["feasibility"] = cl.category.map(FEASIBILITY)

print("=" * 74)
print("WHAT KIND OF CRITERIA DO TRIALS ACTUALLY USE?")
print("=" * 74)
cat = cl.category.value_counts()
for c, n in cat.items():
    trials_using = cl[cl.category == c].nct.nunique()
    print(f"  {c:<22}{n:>5} clauses  ({n/len(cl):4.0%})   in {trials_using:>3}/{len(tr)} trials"
          f"   [{FEASIBILITY.get(c,'')}]")

print("\n" + "=" * 74)
print("CAN WE EVALUATE THEM?")
print("=" * 74)
feas = cl.feasibility.value_counts()
for f, n in feas.items():
    print(f"  {f:<26}{n:>5} clauses  ({n/len(cl):4.0%})")

evaluable = cl.feasibility.str.startswith("evaluable").sum()
not_data = (cl.feasibility == "not a data question").sum()
real = len(cl) - not_data
print(f"\n  of {real} clauses that are actual data questions:")
print(f"     evaluable today: {evaluable} ({evaluable/real:.0%})")

cl.to_csv("trial_criteria_classified.csv", index=False)
print("\nwrote trial_criteria_classified.csv")

print("\n  sample clauses per category (sanity-check my patterns):")
for c in cat.head(6).index:
    ex = cl[cl.category == c].clause.iloc[0][:95]
    print(f"     {c:<22} {ex}")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"trial_criteria | trials={len(tr)} | clauses={len(cl)} "
      f"| evaluable={evaluable/real:.0%} | top_cat={cat.index[0]} "
      f"| biomarker_share={cat.get('biomarker',0)/len(cl):.0%}")

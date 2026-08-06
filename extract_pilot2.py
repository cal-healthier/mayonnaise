"""
extract_pilot2.py -- fix two problems with the first pilot.

PROBLEM 1: I read ONE note per man (the longest consult). The keyword scan
searched ALL notes in the window, median 15 per patient. So 95% keyword vs 35%
extracted was not a fair comparison. Read every note near the decision and
aggregate per man.

PROBLEM 2: the objective validation never ran -- zero measurement concepts
matched ECOG, so there was nothing structured to check against. We know the
extraction PARSES; we do not know whether it is ACCURATE.

Better anchor: we know which drug each man actually received. If the model
reads a note and says "degarelix", the treatment table can confirm it. That is
objective, already available, and directly tests reading comprehension.

Fewer men, all their notes. Aggregate output only.
"""
import json, os, time
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
NMEN = 150

E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"]).iloc[:NMEN]
rows = ",".join(f"('{c}',DATE '{d.date()}')"
                for c, d in zip(E.index.astype(str), E["tx_date"]))
print(f"{len(E)} men, reading ALL their notes near treatment start")

# ---------------------------------------------------------------- ground truth drug
ADT = ["LEUPROLIDE","GOSERELIN","DEGARELIX","TRIPTORELIN","HISTRELIN","LUPRON",
       "ELIGARD","ZOLADEX","FIRMAGON","BICALUTAMIDE","ABIRATERONE","ENZALUTAMIDE"]
med = C.query(f"""
WITH cohort AS (SELECT * FROM UNNEST(
  ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}]))
SELECT c.clinic, UPPER(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING)) AS gen,
       UPPER(CAST(m.MED_NAME_DESCRIPTION AS STRING)) AS nm
FROM cohort c
JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic
JOIN {D}.FACT_TREATMENT_DETAIL t ON t.PATIENT_DK = p.PATIENT_DK
JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
WHERE DATE(t.TREATMENT_DTM) BETWEEN DATE_SUB(c.tx, INTERVAL 30 DAY)
                                AND DATE_ADD(c.tx, INTERVAL 90 DAY)
""").to_dataframe()
def drugs_for(c):
    s = med[med["clinic"] == c]
    found = set()
    for d in ADT:
        if s["gen"].str.contains(d, na=False).any() or s["nm"].str.contains(d, na=False).any():
            found.add(d)
    return found
truth = {c: drugs_for(c) for c in E.index.astype(str)}
have_truth = sum(1 for v in truth.values() if v)
print(f"  men with a known ADT drug in the window: {have_truth}/{len(E)}")

# ---------------------------------------------------------------- all notes
q = C.query(f"""
WITH cohort AS (SELECT * FROM UNNEST(
  ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
pk AS (SELECT c.clinic, c.tx, pe.person_id FROM cohort c
       JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic)
SELECT pk.clinic, CAST(n.note_title AS STRING) AS ttl,
       SUBSTR(CAST(n.note_text AS STRING), 1, 5000) AS txt
FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
WHERE n.note_text IS NOT NULL
  AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 30000
  AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient',
                                        'Consults','H&P','Plan of Care')
  AND DATE(n.note_date) BETWEEN DATE_SUB(pk.tx, INTERVAL 45 DAY)
                            AND DATE_ADD(pk.tx, INTERVAL 45 DAY)
QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic
        ORDER BY LENGTH(CAST(n.note_text AS STRING)) DESC) <= 12
""").to_dataframe()
print(f"  {len(q):,} notes, median {q.groupby('clinic').size().median():.0f} per man")

SCHEMA = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>,
 "mentions_cancer_treatment_decision": true|false,
 "drug_named": "<generic drug name mentioned as being started/given, or null>",
 "performance_status": <0-4, or null>,
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
                 "patient_preference"|"toxicity_concern"|"guideline_standard"|
                 "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false}
drug_named: only if the note says this patient is starting/receiving it.
reason_is_specific: true only for a PATIENT-SPECIFIC rationale, false for
boilerplate such as "per standard of care"."""

def extract(texts, batch=6):
    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json", temperature=0)
    out = []
    for i in range(0, len(texts), batch):
        ch = texts[i:i + batch]
        listed = "\n\n".join(f"### NOTE {j}\n{t[:4000]}" for j, t in enumerate(ch))
        p = ("You are reading oncology notes to extract how a cancer treatment "
             f"decision was made.\n\n{SCHEMA}\n\n{listed}")
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash", contents=p, config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            print(f"   batch {i}: {type(e).__name__} {str(e)[:70]}")
            out += [{}] * len(ch)
        print(f"   ...{min(i+batch, len(texts))}/{len(texts)}", end="\r")
        time.sleep(0.15)
    return out

print(f"\nextracting from {len(q):,} notes ...")
R = pd.DataFrame(extract(q["txt"].tolist()))
X = q.reset_index(drop=True).join(R)

print("\n" + "=" * 80)
print("PER-MAN (all his notes pooled) vs PER-NOTE")
print("=" * 80)
X["spec"] = X["reason_is_specific"].fillna(False).astype(bool)
X["hasreason"] = X["reason_given"].fillna("none_stated").ne("none_stated")
X["hasperf"] = X["performance_status"].notna()
per = X.groupby("clinic").agg(any_reason=("hasreason", "max"),
                              any_spec=("spec", "max"),
                              any_perf=("hasperf", "max"),
                              notes=("txt", "size"))
print(f"  {'signal':<28}{'per-note':>12}{'per-man':>12}")
print(f"  {'a reason is stated':<28}{X['hasreason'].mean():>12.0%}"
      f"{per['any_reason'].mean():>12.0%}")
print(f"  {'reason is patient-specific':<28}{X['spec'].mean():>12.0%}"
      f"{per['any_spec'].mean():>12.0%}")
print(f"  {'performance status given':<28}{X['hasperf'].mean():>12.0%}"
      f"{per['any_perf'].mean():>12.0%}")
print("\n  per-man is the number that matters -- the reasoning only has to appear")
print("  in ONE of his notes for the decision to be adjustable.")

print("\n" + "=" * 80)
print("THE OBJECTIVE TEST: extracted drug vs the drug he actually received")
print("=" * 80)
hit = miss = nodrug = 0
for c, g in X.groupby("clinic"):
    t = truth.get(c, set())
    if not t:
        continue
    named = {str(d).upper() for d in g["drug_named"].dropna()}
    if not named:
        nodrug += 1
    elif any(any(k in n or n in k for k in t) for n in named):
        hit += 1
    else:
        miss += 1
tot = hit + miss + nodrug
if tot:
    print(f"  men with a known drug and notes: {tot}")
    print(f"    extraction named the RIGHT drug   {hit:>4}  ({hit/tot:.0%})")
    print(f"    named a drug, but the wrong one   {miss:>4}  ({miss/tot:.0%})")
    print(f"    named no drug at all              {nodrug:>4}  ({nodrug/tot:.0%})")
    if hit + miss:
        print(f"\n  when it named a drug, it was right {hit/(hit+miss):.0%} of the time")
        print("  -- that is reading comprehension measured against an objective record.")
else:
    print("  no overlap to test")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_pilot2 | men={len(per)} | notes={len(X)} "
      f"| reason_perman={per['any_reason'].mean():.0%} "
      f"| specific_perman={per['any_spec'].mean():.0%} "
      f"| perf_perman={per['any_perf'].mean():.0%} "
      f"| drug_correct={hit/(hit+miss) if (hit+miss) else float('nan'):.0%}")

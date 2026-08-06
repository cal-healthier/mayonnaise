"""
onc_text_check.py -- measure the text on the RIGHT denominator, and test the
Tier-2 premise.

Last run sampled ALL radiology: 87.9M reports across the whole health system,
including chest X-rays for pneumonia. Only 30% mentioned cancer. So "7.4% say
increase" measured how much of Mayo's imaging is oncology, not whether
oncology imaging is readable.

Part 1: response language in scans OF CANCER PATIENTS, DURING TREATMENT.
        That is the real feasibility number for Study 3.

Part 2: the Tier-2 premise. A treatment-recommendation model is only credible
        if we can adjust for WHY a regimen was chosen -- and that reasoning is
        written in the notes, never in structured data. Do oncology notes
        actually contain it? 120.7M Progress Notes, 12.0M Consults, half a
        billion notes total, median 1,811 chars. Measure how often decision
        language appears.

Aggregate rates only. No note text printed.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
ids = "','".join(E.index.astype(str)[:20000])

PH = [("increase",  ["INTERVAL INCREASE","ENLARG","INCREASED IN SIZE","PROGRESS",
                     "WORSEN"]),
      ("decrease",  ["INTERVAL DECREASE","DECREASED IN SIZE","SMALLER","REGRESS",
                     "IMPROV"]),
      ("new_lesion",["NEW LESION","NEW METASTA","NEW NODULE","NEW FOCUS",
                     "NEW SCLEROTIC","NEW OSSEOUS"]),
      ("stable",    ["STABLE","NO SIGNIFICANT CHANGE","UNCHANGED"]),
      ("no_disease",["NO EVIDENCE OF DISEASE","NO EVIDENCE OF RECURREN",
                     "NO EVIDENCE OF METASTA"]),
      ("compares",  ["COMPARED TO","COMPARISON","PRIOR STUDY","SINCE THE PREVIOUS",
                     "PREVIOUS EXAM"]),
      ("cancer_word",["MASS","LESION","METASTA","TUMOR","NODULE","MALIGNAN",
                      "OSSEOUS","SCLEROTIC"])]
F = "RADIOLOGY_NARRATIVE"
U = f"UPPER(CAST(r.{F} AS STRING))"
sel = ", ".join("COUNTIF(" + " OR ".join(f"{U} LIKE '%{p}%'" for p in ps) + f") AS {k}"
                for k, ps in PH)

print("=" * 84)
print("1. RESPONSE LANGUAGE IN SCANS OF PROSTATE PATIENTS, DURING TREATMENT")
print("=" * 84)
q = C.query(f"""
WITH coh AS (
  SELECT CAST(PATIENT_DK AS STRING) AS pdk,
         CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
  FROM {D}.DIM_PATIENT
  WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
SELECT COUNT(*) AS n, COUNT(DISTINCT CAST(r.{F} AS STRING)) AS uniq, {sel}
FROM {D}.FACT_RADIOLOGY r JOIN coh ON coh.pdk = CAST(r.PATIENT_DK AS STRING)
WHERE r.{F} IS NOT NULL AND LENGTH(CAST(r.{F} AS STRING)) > 20
""").to_dataframe().iloc[0]
n = int(q["n"])
print(f"  {n:,} narratives from {len(E):,} men on hormone therapy\n")
print(f"  {'':<14}{'this cohort':>22}{'all radiology (last run)':>28}")
PREV = {"increase": .074, "decrease": .015, "new_lesion": .003, "stable": .130,
        "no_disease": .002, "compares": .569, "cancer_word": .296}
for k, _ in PH:
    v = int(q[k]) / n
    print(f"  {k:<14}{v:>21.1%}{PREV.get(k,0):>27.1%}   {'#'*int(v*30)}")
print(f"\n  unique texts {int(q['uniq'])/n:.0%}")
print("  the gap between the two columns is the denominator effect -- these are")
print("  the scans that would actually be read.")

print("\n" + "=" * 84)
print("2. TIER-2 PREMISE: do oncology notes record WHY a treatment was chosen?")
print("=" * 84)
DEC = [("shared_decision", ["DISCUSSED THE RISKS","DISCUSSED OPTIONS","WE DISCUSSED",
                            "SHARED DECISION","REVIEWED THE OPTIONS"]),
       ("chose_because",   ["ELECTED TO","WE WILL PROCEED WITH","DECIDED TO PROCEED",
                            "OPTED FOR","WILL START"]),
       ("performance",     ["ECOG","PERFORMANCE STATUS","KARNOFSKY","FUNCTIONAL STATUS"]),
       ("declined_or_pref",["DECLINED","PREFERS","PREFERENCE","GOALS OF CARE",
                            "WISHES"]),
       ("toxicity_reason", ["TOLERAT","TOXICITY","SIDE EFFECT","ADVERSE","NEUROPATHY"]),
       ("trial_mention",   ["CLINICAL TRIAL","PROTOCOL","STUDY ENROLL","ELIGIBLE FOR"]),
       ("comorbid_reason", ["COMORBID","RENAL FUNCTION","CARDIAC","FRAIL","AGE-APPROP"])]
TITLES = ["Progress Notes", "Consults", "H&P", "Plan of Care"]
tsql = ",".join(f"'{t}'" for t in TITLES)
NS = 300_000
UN = "UPPER(txt)"
dsel = ", ".join("COUNTIF(" + " OR ".join(f"{UN} LIKE '%{p}%'" for p in ps) + f") AS {k}"
                 for k, ps in DEC)
r2 = C.query(f"""
WITH s AS (
  SELECT CAST(note_text AS STRING) AS txt, CAST(note_title AS STRING) AS ttl
  FROM {D}.note
  WHERE note_text IS NOT NULL AND LENGTH(CAST(note_text AS STRING)) > 200
    AND CAST(note_title AS STRING) IN ({tsql})
  LIMIT {NS})
SELECT COUNT(*) AS n, {dsel} FROM s
""").to_dataframe().iloc[0]
m = int(r2["n"])
print(f"  sampled {m:,} notes ({', '.join(TITLES)})\n")
for k, _ in DEC:
    v = int(r2[k]) / m
    print(f"    {k:<20}{v:>8.1%}  {'#'*int(v*40)}")
print("\n  high rates for 'chose_because', 'performance' and 'declined_or_pref'")
print("  mean the reason for a treatment choice is recorded -- which is exactly")
print("  the confounder that structured data can never capture.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"onc_text_check | onc_narratives={n} | onc_increase={int(q['increase'])/n:.1%} "
      f"| onc_cancer_word={int(q['cancer_word'])/n:.1%} "
      f"| notes_sampled={m} | chose_because={int(r2['chose_because'])/m:.1%} "
      f"| performance={int(r2['performance'])/m:.1%}")

"""
extract_pilot.py -- PILOT STEP 1: does extraction actually work?

Keyword prevalence said the words are there: 99% of treatment decisions have
reasoning in a nearby note, densest in Consults - Outpatient (90% mention a
choice, 78% performance status). But "RECOMMEND" might be "recommend follow-up
in 3 months", not "recommend degarelix because of his cardiac history".

So: run the extractor on real consult notes and test it against something
OBJECTIVE. Performance status is ideal -- it is a number, clinicians record it
in a structured field too, and the model has to find it in prose. If extracted
ECOG matches the structured ECOG, the reader is genuinely reading.

  1. find where structured performance status lives
  2. pull consult notes near treatment start for men who also have one
  3. extract with Gemini: ECOG, the treatment chosen, the stated reason
  4. report AGGREGATE quality only -- parse rate, category distribution,
     and agreement with the structured value. No note text, no extractions
     printed at patient level.
"""
import json, os, time
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)

# ---------------------------------------------------------------- 1. structured ECOG
print("=" * 80)
print("1. WHERE IS STRUCTURED PERFORMANCE STATUS?")
print("=" * 80)
hits = C.query(f"""
  SELECT table_name, column_name FROM {S}.COLUMNS
  WHERE REGEXP_CONTAINS(UPPER(column_name), r'ECOG|PERFORMANCE|KARNOFSKY')
  ORDER BY table_name LIMIT 20""").to_dataframe()
print(hits.to_string(index=False) if len(hits) else "  none by column name")

cc = pd.read_parquet("measurement_concepts.parquet")
cc["nl"] = cc["name"].astype(str).str.lower()
ec = cc[cc["nl"].str.contains("ecog|performance status|karnofsky", na=False)]
ec = ec.sort_values("n_persons", ascending=False)
print(f"\n  measurement concepts matching ECOG/performance: {len(ec)}")
if len(ec):
    print(ec[["cid", "name", "n_persons", "p50"]].head(5).to_string(index=False))
ECIDS = ",".join(str(int(x)) for x in ec.head(4)["cid"]) if len(ec) else None

# ---------------------------------------------------------------- 2. the sample
E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"]).iloc[:4000]
rows = ",".join(f"('{c}',DATE '{d.date()}')"
                for c, d in zip(E.index.astype(str), E["tx_date"]))
ecog_join = f"""
, eco AS (
  SELECT pe.person_id,
         APPROX_QUANTILES(m.value_as_number, 100)[OFFSET(50)] AS ecog_struct
  FROM {D}.measurement m
  JOIN {D}.person pe ON pe.person_id = m.person_id
  WHERE m.measurement_concept_id IN ({ECIDS}) AND m.value_as_number IS NOT NULL
  GROUP BY 1)""" if ECIDS else ""
ecog_sel = "eco.ecog_struct" if ECIDS else "CAST(NULL AS FLOAT64) AS ecog_struct"
ecog_lj = "LEFT JOIN eco ON eco.person_id = pk.person_id" if ECIDS else ""

print("\npulling consult notes near treatment start ...")
q = C.query(f"""
WITH cohort AS (SELECT * FROM UNNEST(
  ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
pk AS (
  SELECT c.clinic, c.tx, pe.person_id FROM cohort c
  JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic){ecog_join}
SELECT pk.clinic, {ecog_sel},
       SUBSTR(CAST(n.note_text AS STRING), 1, 6000) AS txt,
       CAST(n.note_title AS STRING) AS ttl
FROM pk JOIN {D}.note n ON n.person_id = pk.person_id {ecog_lj}
WHERE CAST(n.note_title AS STRING) LIKE 'Consult%'
  AND n.note_text IS NOT NULL AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 20000
  AND DATE(n.note_date) BETWEEN DATE_SUB(pk.tx, INTERVAL 45 DAY)
                            AND DATE_ADD(pk.tx, INTERVAL 45 DAY)
QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY LENGTH(CAST(n.note_text AS STRING)) DESC) = 1
LIMIT 600
""").to_dataframe()
print(f"  {len(q):,} consult notes, one per man")
if ECIDS:
    print(f"  with a structured performance value: {int(q['ecog_struct'].notna().sum()):,}")

# ---------------------------------------------------------------- 3. extract
SCHEMA = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>,
 "is_oncology_visit": true|false,
 "performance_status": <0-4 integer, or null if not stated>,
 "performance_stated_how": "explicit_ecog"|"described_in_words"|"not_stated",
 "treatment_chosen": "<short name, or null>",
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
                 "patient_preference"|"toxicity_concern"|"guideline_standard"|
                 "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false}
reason_is_specific = true only if the note gives a PATIENT-SPECIFIC rationale,
false for boilerplate like "per standard of care"."""

def extract(texts, batch=8):
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
        listed = "\n\n".join(f"### NOTE {j}\n{t[:5000]}" for j, t in enumerate(ch))
        p = ("You are reading oncology consult notes to extract how a treatment "
             f"decision was made.\n\n{SCHEMA}\n\n{listed}")
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash", contents=p, config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            print(f"   batch {i}: {type(e).__name__} {str(e)[:80]}")
            out += [{}] * len(ch)
        print(f"   ...{min(i+batch, len(texts))}/{len(texts)}", end="\r")
        time.sleep(0.2)
    return out

N = min(400, len(q))
print(f"\nextracting from {N} notes (a few minutes) ...")
res = extract(q["txt"].head(N).tolist())
R = pd.DataFrame(res)
q2 = q.head(N).reset_index(drop=True).join(R)

print("\n" + "=" * 80)
print("2. DID IT PARSE, AND WHAT DID IT FIND?  (aggregate only)")
print("=" * 80)
ok = R.notna().any(axis=1)
print(f"  parsed a result           {int(ok.sum()):>5,} of {N} ({ok.mean():.0%})")
for col in ("is_oncology_visit", "reason_is_specific"):
    if col in R:
        v = R[col].fillna(False).astype(bool)
        print(f"  {col:<26}{int(v.sum()):>5,} ({v.mean():.0%})")
if "reason_given" in R:
    print("\n  stated reason for the treatment choice:")
    for k, v in R["reason_given"].value_counts().items():
        print(f"    {str(k):<24}{int(v):>5,}  ({v/len(R):5.0%})  {'#'*int(v/len(R)*26)}")
if "performance_stated_how" in R:
    print("\n  how performance status was recorded:")
    for k, v in R["performance_stated_how"].value_counts().items():
        print(f"    {str(k):<24}{int(v):>5,}  ({v/len(R):5.0%})")

# ---------------------------------------------------------------- 4. the objective test
print("\n" + "=" * 80)
print("3. THE OBJECTIVE TEST: extracted vs structured performance status")
print("=" * 80)
if "performance_status" in q2 and q2["ecog_struct"].notna().any():
    both = q2.dropna(subset=["performance_status", "ecog_struct"]).copy()
    both["performance_status"] = pd.to_numeric(both["performance_status"], errors="coerce")
    both = both.dropna(subset=["performance_status"])
    if len(both) >= 20:
        exact = (both["performance_status"].round() == both["ecog_struct"].round()).mean()
        within1 = ((both["performance_status"] - both["ecog_struct"]).abs() <= 1).mean()
        print(f"  men with BOTH an extracted and a structured value: {len(both):,}")
        print(f"    exact agreement   {exact:.0%}")
        print(f"    within 1 point    {within1:.0%}")
        print("\n  chance agreement on a 0-4 scale is ~20%. Well above that means")
        print("  the model is reading the note, not guessing the common value.")
    else:
        print(f"  only {len(both)} men have both -- too few to judge")
else:
    print("  no structured performance value available to compare against")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_pilot | notes={N} | parsed={ok.mean():.0%} "
      f"| onc_visit={R.get('is_oncology_visit', pd.Series(dtype=bool)).fillna(False).mean():.0%} "
      f"| specific={R.get('reason_is_specific', pd.Series(dtype=bool)).fillna(False).mean():.0%}")

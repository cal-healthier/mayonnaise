"""
extract_browse.py -- read the notes and the extractions yourself.

Aggregate rates cannot tell you whether "none_stated" means the clinician wrote
no rationale or the model missed one. Only reading can.

Runs a modest extraction, saves everything to disk, then gives you a browser
that deliberately over-samples the INFORMATIVE cases:
   - model found a specific reason      -> is it actually specific?
   - model said none_stated             -> did it miss one?
   - model named the WRONG drug         -> what did it misread?
   - model named no drug                -> was one there?

  show()            next case
  show(kind='miss') only the wrong-drug cases
  show(full=True)   the entire note rather than the relevant excerpt
  again()           reshuffle

>>> STAYS IN THE ENCLAVE. Do not screenshot or paste this output back. <<<
"""
import json, os, re, textwrap, time, random
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
NMEN, PER_MAN = 80, 8

print("!" * 74)
print("  Patient text below. Read it here; do not screenshot or paste it back.")
print("!" * 74)

if os.path.exists("browse_cache.parquet"):
    X = pd.read_parquet("browse_cache.parquet")
    print(f"\nloaded cached browse_cache.parquet ({len(X):,} notes)")
else:
    E = pd.read_parquet("psa_progression.parquet")
    tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"]).iloc[:NMEN]
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(E.index.astype(str), E["tx_date"]))
    ADT = ["LEUPROLIDE","GOSERELIN","DEGARELIX","TRIPTORELIN","HISTRELIN","LUPRON",
           "ELIGARD","ZOLADEX","FIRMAGON","BICALUTAMIDE","ABIRATERONE","ENZALUTAMIDE"]
    med = C.query(f"""
    WITH cohort AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}]))
    SELECT c.clinic, UPPER(CONCAT(IFNULL(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING),''),
           ' ', IFNULL(CAST(m.MED_NAME_DESCRIPTION AS STRING),''))) AS txt
    FROM cohort c
    JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic
    JOIN {D}.FACT_TREATMENT_DETAIL t ON t.PATIENT_DK = p.PATIENT_DK
    JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
    WHERE DATE(t.TREATMENT_DTM) BETWEEN DATE_SUB(c.tx, INTERVAL 30 DAY)
                                    AND DATE_ADD(c.tx, INTERVAL 90 DAY)""").to_dataframe()
    truth = {c: {d for d in ADT if med[med["clinic"] == c]["txt"].str.contains(d, na=False).any()}
             for c in E.index.astype(str)}

    q = C.query(f"""
    WITH cohort AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
    pk AS (SELECT c.clinic, c.tx, pe.person_id FROM cohort c
           JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic)
    SELECT pk.clinic, CAST(n.note_title AS STRING) AS ttl,
           DATE(n.note_date) AS dt, pk.tx,
           SUBSTR(CAST(n.note_text AS STRING), 1, 6000) AS txt
    FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
    WHERE n.note_text IS NOT NULL
      AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 30000
      AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient',
                                           'Consults','H&P','Plan of Care')
      AND DATE(n.note_date) BETWEEN DATE_SUB(pk.tx, INTERVAL 45 DAY)
                                AND DATE_ADD(pk.tx, INTERVAL 45 DAY)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic
            ORDER BY LENGTH(CAST(n.note_text AS STRING)) DESC) <= {PER_MAN}""").to_dataframe()
    print(f"\n{len(q):,} notes from {q['clinic'].nunique()} men")

    SCHEMA = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>, "mentions_cancer_treatment_decision": true|false,
 "drug_named": "<generic drug being started/given, or null>",
 "performance_status": <0-4 or null>,
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
   "patient_preference"|"toxicity_concern"|"guideline_standard"|
   "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false,
 "evidence_quote": "<the exact sentence you based reason_given on, or null>"}"""

    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json", temperature=0)
    out, T = [], q["txt"].tolist()
    print("extracting ...")
    for i in range(0, len(T), 6):
        ch = T[i:i + 6]
        listed = "\n\n".join(f"### NOTE {j}\n{t[:4000]}" for j, t in enumerate(ch))
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash",
                contents=f"Read these oncology notes.\n\n{SCHEMA}\n\n{listed}", config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            out += [{}] * len(ch)
        print(f"  ...{min(i+6, len(T))}/{len(T)}", end="\r")
        time.sleep(0.15)
    X = q.join(pd.DataFrame(out))
    X["truth"] = X["clinic"].map(lambda c: ", ".join(sorted(truth.get(c, []))) or "")
    def cat(r):
        named = str(r.get("drug_named") or "").upper()
        t = r["truth"]
        if not named or named in ("NONE", "NAN"):
            return "no_drug_named"
        if t and any(k in named or named in k for k in t.split(", ")):
            return "drug_correct"
        return "drug_wrong" if t else "no_truth"
    X["dcat"] = X.apply(cat, axis=1)
    X.to_parquet("browse_cache.parquet")
    print(f"\nsaved browse_cache.parquet")

X["spec"] = X["reason_is_specific"].fillna(False).astype(bool)
X["none_stated"] = X["reason_given"].fillna("none_stated").eq("none_stated")

print("\n" + "=" * 74)
print("WHAT IS IN THE SAMPLE")
print("=" * 74)
print(f"  notes {len(X):,} | men {X['clinic'].nunique()}")
print(f"  drug named & correct   {int((X['dcat']=='drug_correct').sum()):>5}")
print(f"  drug named & wrong     {int((X['dcat']=='drug_wrong').sum()):>5}")
print(f"  no drug named          {int((X['dcat']=='no_drug_named').sum()):>5}")
print(f"  specific reason found  {int(X['spec'].sum()):>5}")
print(f"  said none_stated       {int(X['none_stated'].sum()):>5}")

KEY = re.compile(r"(elected|proceed|recommend|discussed|prefer|declin|ecog|"
                 r"performance status|toleran|because|given his|due to|plan)", re.I)

def excerpt(t, n=4):
    sents = re.split(r"(?<=[.!?])\s+", str(t))
    hits = [s for s in sents if KEY.search(s)]
    return hits[:n] if hits else sents[:n]

_pool, _i = [], 0
def again(kind=None):
    """reshuffle; kind = spec | none | miss | nodrug | correct | None"""
    global _pool, _i
    f = {"spec": X[X["spec"]], "none": X[X["none_stated"]],
         "miss": X[X["dcat"] == "drug_wrong"],
         "nodrug": X[X["dcat"] == "no_drug_named"],
         "correct": X[X["dcat"] == "drug_correct"]}.get(kind)
    if f is None:
        parts = [X[X["spec"]].head(40), X[X["none_stated"]].head(40),
                 X[X["dcat"] == "drug_wrong"].head(40),
                 X[X["dcat"] == "drug_correct"].head(20)]
        f = pd.concat(parts).drop_duplicates()
    _pool = f.sample(frac=1, random_state=random.randint(0, 9999)).to_dict("records")
    _i = 0
    print(f"  {len(_pool)} cases queued ({kind or 'mixed'}). call show()")

def show(kind=None, full=False):
    global _i
    if kind is not None or not _pool:
        again(kind)
    if _i >= len(_pool):
        print("  end of pool -- call again()"); return
    r = _pool[_i]; _i += 1
    print("\n" + "=" * 74)
    print(f"CASE {_i}/{len(_pool)}   {r['ttl']}   note {r['dt']}   "
          f"therapy started {r['tx']}")
    print("=" * 74)
    print("  MODEL SAID")
    print(f"    decision note?  {r.get('mentions_cancer_treatment_decision')}")
    print(f"    drug named      {r.get('drug_named')!r}   [{r['dcat']}]")
    print(f"    actually given  {r['truth'] or '(none recorded)'}")
    print(f"    performance     {r.get('performance_status')}")
    print(f"    reason          {r.get('reason_given')}  "
          f"(specific={r.get('reason_is_specific')})")
    ev = r.get("evidence_quote")
    if ev:
        print(f"    quoted evidence:")
        for l in textwrap.wrap(str(ev), 66):
            print(f"      > {l}")
    print("\n  NOTE" + (" (full)" if full else " (sentences likely to carry rationale)"))
    body = str(r["txt"]) if full else "\n".join(excerpt(r["txt"], 6))
    for para in body.split("\n"):
        for l in textwrap.wrap(para, 70) or [""]:
            print(f"    {l}")
    print("\n  -> show() for next | show(full=True) | show(kind='miss')")

again()
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_browse | notes={len(X)} | men={X['clinic'].nunique()} "
      f"| correct={int((X['dcat']=='drug_correct').sum())} "
      f"| wrong={int((X['dcat']=='drug_wrong').sum())} "
      f"| specific={int(X['spec'].sum())}")

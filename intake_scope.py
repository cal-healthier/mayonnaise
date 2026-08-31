"""
intake_scope.py -- is there enough in the notes AT THE DECISION POINT to run
the product?

The product eats a case record -- history, pathology, staging, imaging,
prior treatment -- and produces a recommendation. Mayo is a referral centre,
so much of a patient's history happened elsewhere; the question is whether
what arrives WITH the patient (transferred history, intake H&Ps, consult
notes that re-tell the story) lands in the note table in usable form.

Measured, per patient, in the 120 days up to treatment start (the moment the
product would have to speak):

  VOLUME     how many notes, how many characters
  ELEMENTS   does the window mention: stage/TNM, histology or pathology,
             imaging results, prior treatment, an oncologic-history section,
             medications, family history, biomarkers
  RUNNABLE   the core four at once -- stage AND histology AND imaging AND
             (prior treatment OR an oncologic history section). A crude
             stand-in for "a brief could be written from this packet."
  OUTSIDE    explicit outside-records language (outside hospital/records,
             OSH, referred from, records reviewed, second opinion) -- how
             often transferred history is visibly present
  ONBASE     census of the scanned-document tables, where outside paper
             usually lands after digitisation

Regex hit = topic present, not content sufficient. This is the census that
says whether the sufficiency experiment (feed real packets to the engine,
ablate, score) has raw material -- it cannot itself prove sufficiency.
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)
WIN = 120

ELEMENTS = [
    ("stage_tnm",   r"(?i)\bstage\s+(0|I{1,3}V?|IV|[1-4])[ABC]?\b|\b(?:y?p|c)?T[0-4][a-cx]?\s?N[0-3X]\s?M[01X]\b"),
    ("extent_alt",  r"(?i)gleason\s+[0-9]|grade group|risk (group|category|stratif)|"
                    r"biochemical recurrence|castrat|\bmCRPC\b|\bM1[abc]?\b|"
                    r"metastatic|locally advanced|organ.confined"),
    ("histo_path",  r"(?i)adenocarcinoma|carcinoma|sarcoma|histolog|biops|patholog|gleason"),
    ("imaging",     r"(?i)\b(CT|MRI|PET|ultrasound|bone scan)\b"),
    ("img_read",    r"(?i)impression|no evidence of|interval (increase|decrease|change)|metastat"),
    ("prior_tx",    r"(?i)status post|\bs/p\b|underwent|neoadjuvant|adjuvant|"
                    r"prior (chemo|radiation|surgery|therapy|treatment)|"
                    r"received .{0,30}(chemo|radiation)|completed .{0,25}cycles"),
    ("onc_history", r"(?i)oncologic history|oncology history|history of present illness|\bHPI\b"),
    ("medications", r"(?i)current medications|medication list|home medications"),
    ("family_hx",   r"(?i)family history"),
    ("biomarkers",  r"(?i)\bbrca\b|\bmsi\b|\bher2\b|\btmb\b|foundation ?one|\bngs\b|mutation"),
    ("outside",     r"(?i)outside (records|hospital|facility|imaging|slides|patholog|provider)|"
                    r"\bOSH\b|records (were )?reviewed|referred (to us|here|from)|"
                    r"second opinion|per (the )?outside|records from"),
]

coh_rows = []
for tag, lab, mark in [("prostate", "psa_progression.parquet", "psa_values.parquet"),
                       ("ovarian", "ov_label.parquet", "ov_ca125.parquet")]:
    E = pd.read_parquet(lab)
    tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    coh_rows += [f"('{c}','{tag}',DATE '{d.date()}')"
                 for c, d in zip(E.index.astype(str), E["tx_date"])]
rows = ",".join(coh_rows)

sel = ",\n      ".join(
    f"LOGICAL_OR(REGEXP_CONTAINS(t, r\"{rx}\")) AS {name}" for name, rx in ELEMENTS)

SQL = f"""
WITH coh AS (
  SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, cohort STRING, tx DATE>>[{rows}])),
pk AS (
  SELECT coh.clinic, ANY_VALUE(coh.cohort) AS cohort, ANY_VALUE(coh.tx) AS tx,
         MIN(pe.person_id) AS person_id
  FROM coh JOIN {D}.person pe
    ON CAST(pe.person_source_value AS STRING) = coh.clinic
  GROUP BY 1),
w AS (
  SELECT pk.clinic, pk.cohort,
         SUBSTR(CAST(n.note_text AS STRING), 1, 12000) AS t,
         CAST(n.note_title AS STRING) AS title,
         LENGTH(CAST(n.note_text AS STRING)) AS chars
  FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
  WHERE n.note_text IS NOT NULL
    AND DATE(n.note_date) <= pk.tx
    AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL {WIN} DAY))
SELECT clinic, ANY_VALUE(cohort) AS cohort,
      COUNT(*) AS n_notes,
      SUM(chars) AS total_chars,
      LOGICAL_OR(REGEXP_CONTAINS(UPPER(IFNULL(title,'')), r'H&P|CONSULT')) AS has_intake_doc,
      {sel}
FROM w GROUP BY clinic
"""
job = C.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True))
print(f"scan: {job.total_bytes_processed/1e12:.2f} TB "
      f"({len(coh_rows):,} patients, window -{WIN}d to treatment start)")
G = C.query(SQL).to_dataframe()

TITLES = C.query(f"""
WITH coh AS (
  SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, cohort STRING, tx DATE>>[{rows}])),
pk AS (
  SELECT coh.clinic, ANY_VALUE(coh.tx) AS tx, MIN(pe.person_id) AS person_id
  FROM coh JOIN {D}.person pe
    ON CAST(pe.person_source_value AS STRING) = coh.clinic
  GROUP BY 1)
SELECT CAST(n.note_title AS STRING) AS title, COUNT(*) AS n
FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
WHERE n.note_text IS NOT NULL
  AND DATE(n.note_date) <= pk.tx
  AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL {WIN} DAY)
GROUP BY 1 ORDER BY n DESC LIMIT 12""").to_dataframe()

print("\n" + "=" * 78)
print("ONBASE (scanned-document) TABLES -- where transferred paper lands")
print("=" * 78)
try:
    tabs = C.query(
        "SELECT table_name FROM `mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1."
        "INFORMATION_SCHEMA.TABLES WHERE UPPER(table_name) LIKE '%ONBASE%'"
    ).to_dataframe()["table_name"].tolist()
    for tname in tabs:
        try:
            r = list(C.query(
                f"SELECT COUNT(*) AS n, COUNT(DISTINCT PATIENT_DK) AS pts "
                f"FROM {D}.{tname}"))[0]
            print(f"  {tname:<48}{r.n:>12,} rows {r.pts:>10,} patients")
        except Exception as e:
            print(f"  {tname:<48}(no PATIENT_DK or unreadable: {type(e).__name__})")
except Exception as e:
    print(f"  listing failed: {type(e).__name__}: {str(e)[:120]}")

names = [n for n, _ in ELEMENTS]
print("\n" + "=" * 78)
print(f"THE PACKET AT TREATMENT START  (window: {WIN} days)")
print("=" * 78)
print(f"  {'':<26}{'prostate':>12}{'ovarian':>12}")
print("  " + "-" * 52)
for tag in ("prostate", "ovarian"):
    sub = G[G["cohort"] == tag]
    pass
P_, O_ = G[G["cohort"] == "prostate"], G[G["cohort"] == "ovarian"]
tot = {"prostate": len([r for r in coh_rows if "'prostate'" in r]),
       "ovarian": len([r for r in coh_rows if "'ovarian'" in r])}
print(f"  {'patients (cohort)':<26}{tot['prostate']:>12,}{tot['ovarian']:>12,}")
print(f"  {'with >=1 note in window':<26}{len(P_):>12,}{len(O_):>12,}")
print(f"  {'median notes in window':<26}{P_['n_notes'].median():>12.0f}{O_['n_notes'].median():>12.0f}")
print(f"  {'median chars in window':<26}{P_['total_chars'].median():>12,.0f}{O_['total_chars'].median():>12,.0f}")
print(f"  {'has H&P / consult doc':<26}{P_['has_intake_doc'].mean():>12.0%}{O_['has_intake_doc'].mean():>12.0%}")
print()
for name in names:
    print(f"  {name:<26}{P_[name].mean():>12.0%}{O_[name].mean():>12.0%}")

core_p = (P_["stage_tnm"] & P_["histo_path"] & P_["imaging"]
          & (P_["prior_tx"] | P_["onc_history"]))
core_o = (O_["stage_tnm"] & O_["histo_path"] & O_["imaging"]
          & (O_["prior_tx"] | O_["onc_history"]))
# relaxed: stage OR any disease-extent vocabulary (risk group, recurrence,
# metastatic) -- prostate writes extent without the word "stage"
rel_p = ((P_["stage_tnm"] | P_["extent_alt"]) & P_["histo_path"] & P_["imaging"]
         & (P_["prior_tx"] | P_["onc_history"]))
rel_o = ((O_["stage_tnm"] | O_["extent_alt"]) & O_["histo_path"] & O_["imaging"]
         & (O_["prior_tx"] | O_["onc_history"]))
print(f"\n  {'RUNNABLE (strict stage)':<26}{core_p.mean():>12.0%}{core_o.mean():>12.0%}")
print(f"  {'RUNNABLE (any extent)':<26}{rel_p.mean():>12.0%}{rel_o.mean():>12.0%}")
print(f"  (of the whole cohort:     "
      f"{core_p.sum()/max(tot['prostate'],1):>11.0%}{core_o.sum()/max(tot['ovarian'],1):>12.0%})")

print("\n  most common documents in the window:")
for _, r in TITLES.iterrows():
    print(f"    {str(r['title'])[:44]:<46}{int(r['n']):>9,}")

print("""
{}
READING IT
{}
  RUNNABLE means the four ingredients a brief needs are all MENTIONED in the
  window. It does not mean the mention is complete enough -- "stage" could be
  one line of history. The honest next step, if these rates are high, is the
  sufficiency experiment: sample real packets, feed them to the engine,
  ablate elements, score the briefs. This census says whether that experiment
  has raw material, and for what fraction of a referral centre's patients.

  The OUTSIDE row is the referral-centre question directly: how often the
  window explicitly references transferred or external history. High outside
  + high runnable = the arriving record is enough. High outside + LOW
  runnable = the history arrives as paper (check ONBASE above) rather than
  as prose the note table can see.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"intake_scope | prostate_runnable={core_p.mean():.0%} "
      f"| ovarian_runnable={core_o.mean():.0%} "
      f"| outside_p={P_['outside'].mean():.0%} | outside_o={O_['outside'].mean():.0%}")

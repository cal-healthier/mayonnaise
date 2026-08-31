"""
concept_prevalence.py -- before spending a single Gemini token, is it there?

Regex prevalence for every concept we would want to extract, across eight
cancer types. No LLM, no embedding. One BigQuery pass with the matching done
IN SQL, so nothing large comes back -- the note column gets scanned either
way, but we return counts instead of gigabytes of text.

Two numbers per concept:

  PATIENT   share of patients with at least one note mentioning it. This is
            the one that decides whether a variable is worth extracting -- a
            feature present in 4% of patients cannot carry a model.
  NOTE      share of notes mentioning it, which tells you how hard the
            extraction is. Low patient-rate with high note-rate means it is
            concentrated in a particular document; the reverse means it is
            mentioned once and never again.

A regex hit is an UPPER BOUND on real documentation and a LOWER BOUND on what
an LLM could find. "pain" matches "denies pain". "hospice" matches "not
appropriate for hospice". So read these as "the topic is discussed", not "the
finding is present" -- which is exactly what we need to know: an LLM can judge
polarity, but it cannot extract a concept nobody ever writes about.

Concepts are grouped by why they are candidates:
  NO STRUCTURED EQUIVALENT   the study lives here
  WEAK STRUCTURED SHADOW     a field exists but the note is richer
  PHYSICIAN JUDGMENT         kept separate on purpose -- see the objective vs
                             judgment split we want to report
  OPPORTUNISTIC              things I noticed in the note dump
"""
import os
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
NOTES_PER_PT = 10
pd.set_option("display.width", 250)

SITES = {"prostate": "C61", "ovary": "C56", "breast": "C50", "lung": "C34",
         "colon": "C18", "pancreas": "C25", "brain": "C71", "melanoma": "C43"}

# RE2 syntax: no lookahead, no backreferences. (?i) must lead the pattern.
CONCEPTS = [
    # ---------------------------------------------- no structured equivalent
    ("ecog_explicit",  "NONE", r"(?i)\becog\b|\bzubrod\b"),
    ("karnofsky",      "NONE", r"(?i)karnofsky|\bkps\b"),
    ("perf_status",    "NONE", r"(?i)performance status"),
    ("ambulatory",     "NONE", r"(?i)ambulat|bed[ -]?bound|wheelchair|\bwalker\b|"
                               r"assistance with (dressing|bathing|walking)|"
                               r"activities of daily living|\badls?\b"),
    ("tx_intent",      "NONE", r"(?i)palliative intent|curative intent|with curative|"
                               r"not curable|incurable|noncurative"),
    ("goals_hospice",  "NONE", r"(?i)goals of care|hospice|advance directive|"
                               r"\bdnr\b|code status|do not resuscitate"),
    ("residual_dz",    "NONE", r"(?i)residual disease|optimal debulk|optimally debulk|"
                               r"suboptimal|no gross residual|complete cytoreduction|\br0\b"),
    ("ascites",        "NONE", r"(?i)ascites|paracentesis"),
    ("effusion",       "NONE", r"(?i)pleural effusion|thoracentesis"),
    # ------------------------------------------- weak structured shadow
    ("weight_loss",    "WEAK", r"(?i)weight loss|losing weight|cachexi|"
                               r"lost \d+ (lbs|pounds|kg)"),
    ("appetite",       "WEAK", r"(?i)appetite|anorexi|oral intake|\bpo intake\b"),
    # both, deliberately: the GAP between them is how often severity is
    # actually quantified rather than just mentioned, which is the real
    # measure of how hard the extraction is
    ("pain_any",       "WEAK", r"(?i)\bpain\b|\bache\b|\bdiscomfort\b"),
    ("pain_graded",    "WEAK", r"(?i)pain (score|scale)|pain \d+ ?/ ?10|severe pain|"
                               r"uncontrolled pain|moderate pain|mild pain"),
    ("fatigue",        "WEAK", r"(?i)fatigue|exhaust|low energy"),
    ("dyspnea",        "WEAK", r"(?i)dyspnea|short(ness)? of breath|\bsob\b"),
    ("frailty_social", "WEAK", r"(?i)lives alone|caregiver|\bfalls?\b|frail|"
                               r"no support|nursing home|assisted living"),
    ("delirium_cog",   "WEAK", r"(?i)delirium|confus|disorient|dementia|cognitive"),
    # --------------------------------------------------- physician judgment
    ("prognosis_talk", "JUDGE", r"(?i)prognos|life expectancy|months to live|"
                                r"guarded|grave"),
    ("concern_lang",   "JUDGE", r"(?i)i am concerned|concerning for|worrisome|"
                                r"unfortunately|disappoint"),
    ("doing_well",     "JUDGE", r"(?i)doing well|tolerating well|no complaints|"
                                r"asymptomatic"),
    # -------------------------------------------------------- opportunistic
    ("molecular_text", "OPP",  r"(?i)foundation ?one|\bngs\b|next.generation sequencing|"
                               r"\btmb\b|\bmsi\b|microsatellite|tumor mutational|"
                               r"\bbrca\b|amplification|\bvaf\b"),
    ("trial_mention",  "OPP",  r"(?i)clinical trial|study drug|protocol|eligib"),
    ("dose_modify",    "OPP",  r"(?i)dose reduc|dose delay|held (the )?(chemo|cycle|dose)|"
                               r"treatment delay|deferred"),
    ("line_of_tx",     "OPP",  r"(?i)first.line|second.line|third.line|salvage|"
                               r"refractory|recurren"),
    ("interpreter",    "OPP",  r"(?i)interpreter|limited english|non.english speaking"),
]

case = " ".join(f"WHEN site LIKE '{c}%' THEN '{n}'" for n, c in SITES.items())
sel_note = ",\n      ".join(
    f"REGEXP_CONTAINS(t, r\"{rx}\") AS {name}" for name, _, rx in CONCEPTS)
sel_pt = ",\n      ".join(f"LOGICAL_OR({name}) AS {name}" for name, _, _ in CONCEPTS)
agg_pt = ",\n      ".join(
    f"AVG(CAST({name} AS INT64)) AS {name}" for name, _, _ in CONCEPTS)
agg_note = ",\n      ".join(
    f"AVG(CAST({name} AS INT64)) AS {name}" for name, _, _ in CONCEPTS)

SQL = f"""
WITH reg AS (
  SELECT DISTINCT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
         CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING) AS site
  FROM {D}.FACT_CANCER_DATA_REPOSITORY r
  JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
),
pts AS (
  SELECT clinic, CASE {case} END AS cancer FROM reg
  WHERE {" OR ".join(f"site LIKE '{c}%'" for c in SITES.values())}
),
one AS (SELECT clinic, ANY_VALUE(cancer) AS cancer FROM pts GROUP BY 1),
pk AS (
  SELECT one.clinic, one.cancer, pe.person_id
  FROM one JOIN {D}.person pe
    ON CAST(pe.person_source_value AS STRING) = one.clinic
),
n AS (
  SELECT pk.cancer, pk.clinic,
         SUBSTR(CAST(nt.note_text AS STRING), 1, 8000) AS t
  FROM pk JOIN {D}.note nt ON nt.person_id = pk.person_id
  WHERE nt.note_text IS NOT NULL
    AND LENGTH(CAST(nt.note_text AS STRING)) BETWEEN 400 AND 25000
  QUALIFY ROW_NUMBER() OVER (
      PARTITION BY pk.clinic ORDER BY nt.note_date DESC) <= {NOTES_PER_PT}
),
per_note AS (
  SELECT cancer, clinic,
      {sel_note}
  FROM n
),
per_pt AS (
  SELECT cancer, clinic,
      {sel_pt}
  FROM per_note GROUP BY 1, 2
)
SELECT 'PATIENT' AS level, cancer, COUNT(*) AS n,
    {agg_pt}
FROM per_pt GROUP BY 1, 2
UNION ALL
SELECT 'NOTE' AS level, cancer, COUNT(*) AS n,
    {agg_note}
FROM per_note GROUP BY 1, 2
"""

job = C.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True))
print("=" * 78)
print("CONCEPT PREVALENCE BY CANCER TYPE  --  regex only, no LLM")
print("=" * 78)
print(f"  {len(CONCEPTS)} concepts x {len(SITES)} cancers, "
      f"up to {NOTES_PER_PT} most recent notes per patient")
print(f"  scan: {job.total_bytes_processed/1e12:.2f} TB\n")

if os.path.exists("concept_prevalence.parquet"):
    print("  (cached concept_prevalence.parquet -- skipping the scan)")
    R = pd.read_parquet("concept_prevalence.parquet")
else:
    R = C.query(SQL).to_dataframe()
names = [c for c, _, _ in CONCEPTS]
group = {c: g for c, g in ((c, g) for c, g, _ in CONCEPTS)}

for level in ("PATIENT", "NOTE"):
    S = R[R["level"] == level].set_index("cancer")
    order = [c for c in SITES if c in S.index]
    print("\n" + "=" * 78)
    if level == "PATIENT":
        print("SHARE OF PATIENTS WITH >=1 NOTE MENTIONING THE CONCEPT")
    else:
        print("SHARE OF NOTES MENTIONING THE CONCEPT")
    print("=" * 78)
    print(f"  {'concept':<17}{'grp':<6}" + "".join(f"{c[:7]:>8}" for c in order))
    print("  " + "-" * (23 + 8 * len(order)))
    last = None
    for name in names:
        if group[name] != last:
            print()
            last = group[name]
        row = "".join(f"{S.loc[c, name]:>8.0%}" for c in order)
        print(f"  {name:<17}{group[name]:<6}{row}")
    if level == "PATIENT":
        print(f"\n  {'patients':<23}" + "".join(f"{int(S.loc[c,'n']):>8,}" for c in order))
    else:
        print(f"\n  {'notes':<23}" + "".join(f"{int(S.loc[c,'n']):>8,}" for c in order))

P = R[R["level"] == "PATIENT"].set_index("cancer")
avg = {n: P[n].mean() for n in names}
rich = sorted(avg, key=avg.get, reverse=True)
print("\n" + "=" * 78)
print("READING IT")
print("=" * 78)
print("  best documented:  " + ", ".join(f"{n} {avg[n]:.0%}" for n in rich[:6]))
print("  worst documented: " + ", ".join(f"{n} {avg[n]:.0%}" for n in rich[-6:]))
print("""
  A regex hit means the TOPIC IS DISCUSSED, not that the finding is present --
  "pain" matches "denies pain", "hospice" matches "not a hospice candidate".
  So this is an upper bound on documentation and a lower bound on what an LLM
  could recover, since the LLM can also judge polarity and grade severity.

  What to do with it: anything under roughly 20% of patients is not worth
  extracting -- it cannot carry a model no matter how well it is extracted.
  Anything over 60% is a real candidate variable. The middle is worth a look
  at whether the missingness is itself informative.

  Watch for concepts that vary sharply BY CANCER. Those are the ones where a
  single extraction schema will not transfer, and they are also where the
  disease-specific findings live -- ascites and residual disease should be
  ovarian-heavy, effusion lung-heavy, and if they are not, the regex is wrong.""")

R.to_parquet("concept_prevalence.parquet")
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"prevalence | concepts={len(CONCEPTS)} | cancers={len(SITES)} | "
      f"best={rich[0]}({avg[rich[0]]:.0%}) | worst={rich[-1]}({avg[rich[-1]]:.0%})")

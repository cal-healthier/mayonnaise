"""
ecog_goldset.py -- the free half of the extraction study.

Prevalence says 21-42% of patients have an explicit ECOG mention. Where the
note literally says "ECOG 2", a regex can read the NUMBER off -- no LLM, no
budget. That gives us, for free:

  1. A GOLD SET.  Thousands of notes with a physician-stated score. This is
     the validation set for the LLM extractor later: hide the explicit
     statement, ask the model to infer the score from the rest of the note,
     compare. No hand-labelling required.
  2. A PREMISE TEST.  If note-stated ECOG stratifies survival in our
     cohorts, the whole "structure hidden in prose" study rests on measured
     ground. If a physician-documented ECOG of 3 does not predict worse
     survival here, extraction is pointless and we should know today.
  3. TRAJECTORY.  How many patients have several values over time, and how
     often the score worsens -- whether "functional decline" is even
     observable in the record.

Value parsing is deliberately strict (ECOG [PS] [of/is/was] 0-4, KPS 10-100)
so a date like "ECOG on 3/2020" cannot become a score. Strict parsing
undercounts; that is the right direction for a gold set.

Aggregates only are printed; the per-patient table stays in the enclave.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

SITES = {"prostate": "C61", "ovary": "C56", "breast": "C50", "lung": "C34",
         "colon": "C18", "pancreas": "C25", "brain": "C71", "melanoma": "C43"}
# spell the full column expression out -- bare aliases have now bitten three
# scripts in a row, each time surviving the syntax check and dying at dry-run
SITECOL = "CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING)"
case = " ".join(f"WHEN {SITECOL} LIKE '{c}%' THEN '{n}'" for n, c in SITES.items())
ECOG_RX = (r"(?i)\bECOG\b(?:\s+(?:PS|PERFORMANCE\s+STATUS))?"
           r"(?:\s+(?:OF|IS|WAS))?[\s:=-]*([0-4])\b")
KPS_RX = (r"(?i)\b(?:KARNOFSKY(?:\s+(?:PERFORMANCE\s+)?(?:STATUS|SCORE))?|KPS)"
          r"(?:\s+(?:OF|IS|WAS))?[\s:=-]*((?:100|[1-9]0))\b")

cache = "ecog_gold.parquet"
if os.path.exists(cache):
    G = pd.read_parquet(cache)
    print(f"cached: {len(G):,} patients")
else:
    SQL = f"""
    WITH reg0 AS (
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             CASE {case} END AS cancer,
             MAX(CAST(r.VITAL_STATUS AS STRING)) AS vital,
             MAX(DATE(r.DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_dt
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r
      JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      WHERE {" OR ".join(f"CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING) LIKE '{c}%'"
                         for c in SITES.values())}
      GROUP BY 1, 2
    ),
    reg AS (
      SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY clinic ORDER BY cancer) AS rn1
        FROM reg0 WHERE cancer IS NOT NULL)
      WHERE rn1 = 1
    ),
    pk AS (
      SELECT reg.clinic, ANY_VALUE(reg.cancer) AS cancer,
             ANY_VALUE(reg.vital) AS vital, ANY_VALUE(reg.last_dt) AS last_dt,
             MIN(pe.person_id) AS person_id
      FROM reg JOIN {D}.person pe
        ON CAST(pe.person_source_value AS STRING) = reg.clinic
      GROUP BY 1
    ),
    hits AS (
      SELECT pk.clinic, DATE(n.note_date) AS d,
             SAFE_CAST(REGEXP_EXTRACT(
               SUBSTR(CAST(n.note_text AS STRING),1,12000), r"{ECOG_RX}")
               AS INT64) AS ev,
             SAFE_CAST(REGEXP_EXTRACT(
               SUBSTR(CAST(n.note_text AS STRING),1,12000), r"{KPS_RX}")
               AS INT64) AS kv
      FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
      WHERE n.note_text IS NOT NULL AND n.note_date IS NOT NULL
        AND REGEXP_CONTAINS(CAST(n.note_text AS STRING),
                            r'(?i)\\becog\\b|karnofsky|\\bkps\\b')
    ),
    ev AS (
      SELECT clinic,
        COUNT(ev) AS n_vals,
        ARRAY_AGG(ev IGNORE NULLS ORDER BY d ASC  LIMIT 1)[SAFE_OFFSET(0)] AS first_v,
        ARRAY_AGG(ev IGNORE NULLS ORDER BY d DESC LIMIT 1)[SAFE_OFFSET(0)] AS last_v,
        MIN(IF(ev IS NOT NULL, d, NULL)) AS first_d,
        MAX(IF(ev IS NOT NULL, d, NULL)) AS last_d,
        COUNT(DISTINCT ev) AS n_distinct,
        COUNT(kv) AS n_kps,
        ARRAY_AGG(kv IGNORE NULLS ORDER BY d ASC LIMIT 1)[SAFE_OFFSET(0)] AS first_kps
      FROM hits GROUP BY 1
    )
    SELECT pk.cancer, pk.clinic, pk.vital, pk.last_dt,
           ev.n_vals, ev.first_v, ev.last_v, ev.first_d, ev.last_d,
           ev.n_distinct, ev.n_kps, ev.first_kps
    FROM ev JOIN pk USING (clinic)
    """
    job = C.query(SQL, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"scan: {job.total_bytes_processed/1e12:.2f} TB")
    G = C.query(SQL).to_dataframe()
    for c in ("last_dt", "first_d", "last_d"):
        G[c] = pd.to_datetime(G[c].astype("datetime64[ns]"))
    G.to_parquet(cache)
    print(f"{len(G):,} patients with an ECOG/KPS mention")

G["dead"] = G["vital"].astype(str).str.upper().str.contains("DEAD|DECEASED", na=False)
G["fu_days"] = (G["last_dt"] - G["first_d"]).dt.days
H = G[G["first_v"].notna()].copy()
H["first_v"] = H["first_v"].astype(int)

print("\n" + "=" * 78)
print("1. THE GOLD SET -- patients with a regex-parseable ECOG value")
print("=" * 78)
print(f"  {'cancer':<11}{'patients':>9}{'values/pt':>11}"
      + "".join(f"{'ECOG '+str(v):>9}" for v in range(5)))
print("  " + "-" * 74)
for cn in SITES:
    sub = H[H["cancer"] == cn]
    if not len(sub):
        continue
    dist = sub["first_v"].value_counts(normalize=True)
    print(f"  {cn:<11}{len(sub):>9,}{sub['n_vals'].median():>11.0f}"
          + "".join(f"{dist.get(v, 0):>9.0%}" for v in range(5)))
tot = len(H)
print(f"\n  total gold-set patients: {tot:,}  "
      f"(plus {int((G['n_kps'] > 0).sum()):,} with a KPS, "
      f"{int(((G['cancer']=='brain') & (G['n_kps']>0)).sum()):,} of them brain)")

print("\n" + "=" * 78)
print("2. DOES NOTE-STATED ECOG PREDICT SURVIVAL?  (from first documented value)")
print("=" * 78)
print(f"  {'first ECOG':<12}{'n':>9}{'% dead':>9}{'median d to death (dead)':>27}")
print("  " + "-" * 60)
for band, lab in [((0,), "0"), ((1,), "1"), ((2,), "2"), ((3, 4), "3-4")]:
    sub = H[H["first_v"].isin(band) & H["fu_days"].notna() & (H["fu_days"] >= 0)]
    dd = sub.loc[sub["dead"], "fu_days"]
    print(f"  {lab:<12}{len(sub):>9,}{sub['dead'].mean():>9.0%}"
          f"{dd.median() if len(dd) else float('nan'):>27.0f}")
print("""
  If % dead and time-to-death do not order cleanly by ECOG here, stop: the
  premise of extracting functional status is broken. If they do, this is the
  premise measured, not assumed.""")

print("=" * 78)
print("3. TRAJECTORY -- is functional DECLINE visible?")
print("=" * 78)
multi = H[H["n_vals"] >= 2]
print(f"  patients with >=2 documented values: {len(multi):,} ({len(multi)/max(tot,1):.0%})")
print(f"  ...spanning >=180 days:              "
      f"{int(((multi['last_d']-multi['first_d']).dt.days>=180).sum()):,}")
print(f"  value changes over time (n_distinct>1): {int((multi['n_distinct']>1).sum()):,}")
print(f"  worsens (last > first):              {int((multi['last_v']>multi['first_v']).sum()):,}")
print(f"  improves (last < first):             {int((multi['last_v']<multi['first_v']).sum()):,}")

print("""
{}
WHAT THIS SETS UP
{}
  The gold set is the validation harness for the LLM extractor: take these
  notes, MASK the explicit "ECOG n" statement, ask the model to infer the
  score from the remaining prose, and compare against the physician's own
  number. Thousands of labels, zero annotation cost.

  And the survival table is the premise test. If documented ECOG stratifies
  outcomes in this dataset, then recovering it for the ~60% of patients who
  have functional-status PROSE but no explicit score is worth doing -- and
  that is the extraction study in one sentence.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"ecog_goldset | gold_patients={tot} | multi_value={len(multi)} "
      f"| kps={int((G['n_kps']>0).sum())}")

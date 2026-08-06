"""
decision_notes.py -- ask the right question about the notes.

Twice now I measured on too broad a denominator. Last run sampled 300k notes
from ALL specialties, so "3.2% say elected to proceed" described Mayo's notes
in general, not oncology notes at a treatment decision.

Two corrections:
  1. restrict to notes belonging to CANCER PATIENTS, in a window around the
     start of their treatment
  2. ask the PER-DECISION question, not the per-note one. A man has dozens of
     notes around starting therapy; the reasoning only has to appear in ONE.
     "What share of treatment decisions have reasoning documented nearby?" is
     the number that decides whether confounding can be adjusted for.

Also breaks it down by note type, so we know WHERE to look (consult? H&P?)
rather than reading all 500M.

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
E = E.dropna(subset=["tx_date"])
sub = E.iloc[:4000]
vals = ",".join(f"('{c}', DATE '{d.date()}')"
                for c, d in zip(sub.index.astype(str), sub["tx_date"]))
print(f"{len(sub):,} men, each with a hormone-therapy start date")

DEC = [("chose_because",  ["ELECTED TO","WE WILL PROCEED","DECIDED TO PROCEED",
                           "OPTED FOR","WILL START","PLAN IS TO","RECOMMEND"]),
       ("shared_decision",["DISCUSSED THE RISKS","DISCUSSED OPTIONS","WE DISCUSSED",
                           "SHARED DECISION","REVIEWED THE OPTIONS","COUNSELED"]),
       ("performance",    ["ECOG","PERFORMANCE STATUS","KARNOFSKY","AMBULAT",
                           "ACTIVITIES OF DAILY"]),
       ("preference",     ["DECLINED","PREFERS","PREFERENCE","GOALS OF CARE",
                           "WISHES","ELECTED NOT"]),
       ("comorbid_reason",["COMORBID","RENAL FUNCTION","CARDIAC HISTORY","FRAIL",
                           "HEPATIC FUNCTION"]),
       ("trial_mention",  ["CLINICAL TRIAL","PROTOCOL","STUDY ENROLL","ELIGIBLE FOR"])]
U = "UPPER(CAST(n.note_text AS STRING))"
flags = ", ".join("MAX(CASE WHEN " + " OR ".join(f"{U} LIKE '%{p}%'" for p in ps)
                  + f" THEN 1 ELSE 0 END) AS {k}" for k, ps in DEC)
anyflag = " OR ".join(" OR ".join(f"{U} LIKE '%{p}%'" for p in ps) for _, ps in DEC)

print("pulling notes in a +/-60 day window around treatment start ...")
q = C.query(f"""
WITH coh AS (SELECT clinic, tx FROM UNNEST([
  STRUCT<clinic STRING, tx DATE>{vals[1:-1] if False else ''}
]) ) , cohort AS (
  SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{vals}])
),
pk AS (
  SELECT c.clinic, c.tx, p.PATIENT_DK, pe.person_id
  FROM cohort c
  JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic
  LEFT JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic),
nt AS (
  SELECT pk.clinic, CAST(n.note_title AS STRING) AS ttl, {flags},
         COUNT(*) AS n_notes
  FROM pk JOIN {D}.note n ON n.person_id = pk.person_id
  WHERE n.note_text IS NOT NULL AND LENGTH(CAST(n.note_text AS STRING)) > 200
    AND DATE(n.note_date) BETWEEN DATE_SUB(pk.tx, INTERVAL 60 DAY)
                              AND DATE_ADD(pk.tx, INTERVAL 60 DAY)
  GROUP BY 1,2)
SELECT * FROM nt
""").to_dataframe()

if not len(q):
    print("  NO NOTES MATCHED -- the person_id join may be the problem. Tell Claude.")
    raise SystemExit

pts = q["clinic"].nunique()
print(f"  {len(q):,} patient-notetype rows | {pts:,} of {len(sub):,} men have "
      f"any note in the window ({pts/len(sub):.0%})")
print(f"  notes in window: {int(q['n_notes'].sum()):,}, "
      f"median {q.groupby('clinic')['n_notes'].sum().median():.0f} per man")

print("\n" + "=" * 80)
print("PER-DECISION: does ANY note near treatment start record the reasoning?")
print("=" * 80)
per = q.groupby("clinic")[[k for k, _ in DEC]].max()
print(f"  {'signal':<20}{'men with it documented':>26}{'%':>8}")
for k, _ in DEC:
    v = int(per[k].sum())
    print(f"  {k:<20}{v:>26,}{v/len(per):>8.0%}  {'#'*int(v/len(per)*28)}")
anyk = per.max(axis=1)
print(f"\n  ANY of the above: {int(anyk.sum()):,} of {len(per):,} men "
      f"({anyk.mean():.0%})")
print("  this is the number that matters -- per DECISION, not per note.")

print("\n" + "=" * 80)
print("WHERE TO LOOK -- which note types carry the reasoning?")
print("=" * 80)
by = q.groupby("ttl").agg(notes=("n_notes", "sum"),
                          chose=("chose_because", "mean"),
                          perf=("performance", "mean"),
                          pref=("preference", "mean")).sort_values("notes",
                                                                  ascending=False)
print(f"  {'note type':<26}{'notes':>10}{'chose':>9}{'perf':>8}{'pref':>8}")
for t, r in by.head(10).iterrows():
    print(f"  {str(t)[:25]:<26}{int(r['notes']):>10,}{r['chose']:>9.0%}"
          f"{r['perf']:>8.0%}{r['pref']:>8.0%}")
print("\n  a concentrated signal means we read one note type, not all 500M.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"decision_notes | men={len(sub)} | with_notes={pts} | any_reasoning={anyk.mean():.0%} "
      f"| chose={per['chose_because'].mean():.0%} | perf={per['performance'].mean():.0%} "
      f"| pref={per['preference'].mean():.0%}")

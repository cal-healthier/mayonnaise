"""
decision_notes2.py -- rewritten clean. Previous version had a leftover broken
CTE in the SQL.

Question: for a treatment decision, is the REASONING documented in any note
nearby? That is what makes confounding by indication adjustable, and it is the
thing structured data can never give you.

Restricted to the prostate cohort, +/-60 days around hormone-therapy start.
Per-DECISION, not per-note. Aggregate rates only, no text printed.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"]).iloc[:4000]
rows = ",".join(f"('{c}',DATE '{d.date()}')"
                for c, d in zip(E.index.astype(str), E["tx_date"]))
print(f"{len(E):,} men with a hormone-therapy start date")

# ---------------------------------------------------------------- coverage first
cov = C.query(f"""
WITH cohort AS (SELECT * FROM UNNEST(
  ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
pk AS (
  SELECT c.clinic, c.tx, pe.person_id
  FROM cohort c
  JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic)
SELECT COUNT(*) AS matched_person, COUNT(DISTINCT clinic) AS men FROM pk
""").to_dataframe().iloc[0]
print(f"  matched to an OMOP person_id: {int(cov['men']):,} of {len(E):,}")
if int(cov["men"]) == 0:
    raise SystemExit("person join failed -- tell Claude")

DEC = [("chose_because",  ["ELECTED TO","WE WILL PROCEED","DECIDED TO PROCEED",
                           "OPTED FOR","WILL START","PLAN IS TO","RECOMMEND"]),
       ("shared_decision",["DISCUSSED THE RISK","DISCUSSED OPTION","WE DISCUSSED",
                           "SHARED DECISION","REVIEWED THE OPTION","COUNSELED"]),
       ("performance",    ["ECOG","PERFORMANCE STATUS","KARNOFSKY","AMBULAT",
                           "ACTIVITIES OF DAILY"]),
       ("preference",     ["DECLINED","PREFERS","PREFERENCE","GOALS OF CARE",
                           "WISHES","ELECTED NOT"]),
       ("comorbid",       ["COMORBID","RENAL FUNCTION","CARDIAC HISTORY","FRAIL",
                           "HEPATIC FUNCTION"]),
       ("trial",          ["CLINICAL TRIAL","PROTOCOL","STUDY ENROLL","ELIGIBLE FOR"])]
U = "UPPER(CAST(n.note_text AS STRING))"
flags = ", ".join("MAX(IF(" + " OR ".join(f"{U} LIKE '%{p}%'" for p in ps)
                  + f", 1, 0)) AS {k}" for k, ps in DEC)

print("pulling notes in the +/-60 day window ...")
q = C.query(f"""
WITH cohort AS (SELECT * FROM UNNEST(
  ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
pk AS (
  SELECT c.clinic, c.tx, pe.person_id
  FROM cohort c
  JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING) = c.clinic)
SELECT pk.clinic,
       CAST(n.note_title AS STRING) AS ttl,
       COUNT(*) AS n_notes,
       {flags}
FROM pk
JOIN {D}.note n ON n.person_id = pk.person_id
WHERE n.note_text IS NOT NULL
  AND LENGTH(CAST(n.note_text AS STRING)) > 200
  AND DATE(n.note_date) BETWEEN DATE_SUB(pk.tx, INTERVAL 60 DAY)
                            AND DATE_ADD(pk.tx, INTERVAL 60 DAY)
GROUP BY 1, 2
""").to_dataframe()

if not len(q):
    print("  no notes in the window -- widen it or check note_date")
    raise SystemExit
pts = q["clinic"].nunique()
print(f"  {len(q):,} rows | {pts:,} of {len(E):,} men have notes in the window "
      f"({pts/len(E):.0%})")
print(f"  {int(q['n_notes'].sum()):,} notes, median "
      f"{q.groupby('clinic')['n_notes'].sum().median():.0f} per man")

print("\n" + "=" * 80)
print("PER-DECISION: is the reasoning in ANY note near treatment start?")
print("=" * 80)
per = q.groupby("clinic")[[k for k, _ in DEC]].max()
print(f"  {'signal':<18}{'men with it documented':>26}{'%':>8}")
for k, _ in DEC:
    v = int(per[k].sum())
    print(f"  {k:<18}{v:>26,}{v/len(per):>8.0%}  {'#'*int(v/len(per)*26)}")
anyk = per.max(axis=1)
print(f"\n  ANY of the above:  {int(anyk.sum()):,} of {len(per):,} men "
      f"({anyk.mean():.0%})")
core = per[["chose_because", "performance", "preference"]].max(axis=1)
print(f"  the three that matter for confounding: {core.mean():.0%}")

print("\n" + "=" * 80)
print("WHERE TO LOOK -- which note type carries it?")
print("=" * 80)
by = q.groupby("ttl").agg(notes=("n_notes", "sum"), men=("clinic", "nunique"),
                          chose=("chose_because", "mean"),
                          perf=("performance", "mean"),
                          pref=("preference", "mean")).sort_values(
                          "notes", ascending=False)
print(f"  {'note type':<26}{'notes':>10}{'men':>8}{'chose':>8}{'perf':>7}{'pref':>7}")
for t, r in by.head(10).iterrows():
    print(f"  {str(t)[:25]:<26}{int(r['notes']):>10,}{int(r['men']):>8,}"
          f"{r['chose']:>8.0%}{r['perf']:>7.0%}{r['pref']:>7.0%}")
print("\n  a concentrated signal means reading one note type, not all 500M.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"decision_notes2 | men={len(E)} | with_notes={pts} "
      f"| any_reasoning={anyk.mean():.0%} | core3={core.mean():.0%} "
      f"| chose={per['chose_because'].mean():.0%} "
      f"| perf={per['performance'].mean():.0%} | pref={per['preference'].mean():.0%}")

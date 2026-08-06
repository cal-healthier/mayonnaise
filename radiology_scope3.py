"""
radiology_scope3.py -- TABLESAMPLE does not work on views. Use a LIMIT subquery.

Settled already: RADIOLOGY_NARRATIVE is the field to use --
76,395,131 reports with >20 chars (87%), median 461 chars, p90 1,878.
Impression is empty over half the time; narrative is the real prose.

Remaining questions:
  1. does the narrative contain response language, and is it dictated or
     templated?
  2. what is in the OMOP `note` and `note_nlp` tables -- if note_nlp is
     populated, extraction has ALREADY been run over these documents
  3. how many prostate/ovarian patients have narrative text AND a
     marker-based progression date (the validation set)

Aggregate statistics only. No text printed.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)
T, F = "FACT_RADIOLOGY", "RADIOLOGY_NARRATIVE"
NSAMP = 400_000

PH = [("increase",   ["INTERVAL INCREASE", "ENLARG", "INCREASED IN SIZE", "PROGRESS"]),
      ("decrease",   ["INTERVAL DECREASE", "DECREASED IN SIZE", "SMALLER", "REGRESS"]),
      ("new_lesion", ["NEW LESION", "NEW METASTA", "NEW NODULE", "NEW FOCUS"]),
      ("stable",     ["STABLE", "NO SIGNIFICANT CHANGE", "UNCHANGED"]),
      ("no_disease", ["NO EVIDENCE OF DISEASE", "NO EVIDENCE OF RECURREN",
                      "NO EVIDENCE OF METASTA"]),
      ("recist",     ["RECIST", "PARTIAL RESPONSE", "COMPLETE RESPONSE",
                      "PROGRESSIVE DISEASE", "STABLE DISEASE"]),
      ("compares",   ["COMPARED TO", "COMPARISON", "PRIOR STUDY", "SINCE THE PREVIOUS",
                      "PREVIOUS EXAM"]),
      ("cancer_word", ["MASS", "LESION", "METASTA", "TUMOR", "NODULE", "MALIGNAN"])]

print("=" * 84)
print(f"1. DOES {F} CONTAIN RESPONSE LANGUAGE?  ({NSAMP:,} sample, aggregate only)")
print("=" * 84)
sel = ", ".join("COUNTIF(" + " OR ".join(f"t LIKE '%{p}%'" for p in ps) + f") AS {k}"
                for k, ps in PH)
q = C.query(f"""
WITH s AS (
  SELECT UPPER(CAST({F} AS STRING)) AS t
  FROM {D}.{T}
  WHERE {F} IS NOT NULL AND LENGTH(CAST({F} AS STRING)) > 20
  LIMIT {NSAMP})
SELECT COUNT(*) AS n, COUNT(DISTINCT t) AS uniq, {sel} FROM s
""").to_dataframe().iloc[0]
m = int(q["n"])
print(f"  sampled {m:,} narratives\n")
for k, _ in PH:
    v = int(q[k])
    print(f"    {k:<14}{v:>10,}  ({v/m:5.1%})  {'#'*int(v/m*34)}")
print(f"\n  distinct texts {int(q['uniq']):,} of {m:,} ({int(q['uniq'])/m:.0%} unique)")
print("  high uniqueness = dictated prose (good). low = template (bad).")

print("\n" + "=" * 84)
print("2. THE OMOP `note` AND `note_nlp` TABLES")
print("=" * 84)
for tab in ("note", "note_nlp"):
    try:
        cols = [x.column_name for x in C.query(
            f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='{tab}' "
            f"ORDER BY ordinal_position")]
        print(f"\n  {tab} columns: {cols}")
        idc = "person_id" if "person_id" in cols else cols[0]
        cnt = C.query(f"SELECT COUNT(*) AS n, COUNT(DISTINCT {idc}) AS p "
                      f"FROM {D}.{tab}").to_dataframe().iloc[0]
        print(f"  {int(cnt['n']):,} rows over {int(cnt['p']):,} patients")
        if int(cnt["n"]) == 0:
            print("  -> EMPTY. no shortcut here.")
            continue
        if tab == "note":
            for c in ("note_title", "note_type_concept_id"):
                if c in cols:
                    ty = C.query(f"""SELECT CAST({c} AS STRING) AS v, COUNT(*) AS n
                      FROM {D}.note GROUP BY 1 ORDER BY n DESC LIMIT 8""").to_dataframe()
                    print(f"\n  commonest {c}:")
                    print(ty.to_string(index=False))
                    break
            if "note_text" in cols:
                ln = C.query(f"""SELECT
                  COUNTIF(note_text IS NOT NULL AND LENGTH(CAST(note_text AS STRING))>20) AS w,
                  APPROX_QUANTILES(LENGTH(CAST(note_text AS STRING)),100)[OFFSET(50)] AS med
                  FROM {D}.note""").to_dataframe().iloc[0]
                print(f"\n  notes with real text: {int(ln['w']):,}, "
                      f"median {int(ln['med']):,} chars")
        else:
            print("\n  -> note_nlp is POPULATED: extraction has already been run.")
            for c in ("lexical_variant", "note_nlp_concept_id"):
                if c in cols:
                    ty = C.query(f"""SELECT CAST({c} AS STRING) AS v, COUNT(*) AS n
                      FROM {D}.note_nlp GROUP BY 1 ORDER BY n DESC LIMIT 10
                    """).to_dataframe()
                    print(f"  commonest {c}:")
                    print(ty.to_string(index=False))
                    break
    except Exception as e:
        print(f"\n  {tab}: {type(e).__name__} {str(e)[:110]}")

print("\n" + "=" * 84)
print("3. VALIDATION SET -- narrative text AND a marker progression date")
print("=" * 84)
for f, lbl in [("psa_progression.parquet", "prostate (PSA date)"),
               ("ov_label.parquet", "ovarian (CA-125 date)")]:
    try:
        idx = pd.read_parquet(f)
        idx = idx[idx["prog"] == 1].index.astype(str)
        ids = "','".join(idx[:20000])
        q2 = C.query(f"""
          SELECT COUNT(DISTINCT CAST(r.PATIENT_DK AS STRING)) AS pts,
                 COUNT(*) AS reports
          FROM {D}.{T} r
          WHERE CAST(r.PATIENT_DK AS STRING) IN (
                  SELECT CAST(PATIENT_DK AS STRING) FROM {D}.DIM_PATIENT
                  WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
            AND r.{F} IS NOT NULL AND LENGTH(CAST(r.{F} AS STRING)) > 20
        """).to_dataframe().iloc[0]
        print(f"  {lbl:<28}{int(q2['pts']):>7,} patients, "
              f"{int(q2['reports']):>10,} reports  "
              f"({int(q2['reports'])/max(int(q2['pts']),1):.0f} each)")
    except Exception as e:
        print(f"  {lbl:<28}skipped ({type(e).__name__} {str(e)[:70]})")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"radiology_scope3 | field={F} | sampled={m} | unique={int(q['uniq'])/m:.0%} "
      f"| compares={int(q['compares'])/m:.0%} | cancer_word={int(q['cancer_word'])/m:.0%} "
      f"| recist={int(q['recist'])/m:.1%}")

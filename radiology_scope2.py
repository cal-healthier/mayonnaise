"""
radiology_scope2.py -- fixed, and widened.

Last run: 39,776,801 usable impressions over 2,880,108 patients, 1933-2026.
Crashed on the language check -- my column aliases contained spaces and slashes,
which BigQuery rejects, and the query scanned all 87.9M rows unnecessarily.

Two things it missed:
  - RADIOLOGY_IMPRESSION has median length 0, i.e. over half are EMPTY. The
    table also has RADIOLOGY_REPORT and RADIOLOGY_NARRATIVE. Content is
    probably there when impression is blank.
  - the table list contains `note` and `note_nlp` -- OMOP standard tables.
    note_nlp holds ALREADY-EXTRACTED NLP output. If populated, someone has run
    extraction over these documents and the results are just sitting there.

Aggregate statistics only. No note text printed.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)
T = "FACT_RADIOLOGY"

# ---------------------------------------------------------------- 1. which field?
print("=" * 84)
print("1. WHICH TEXT FIELD ACTUALLY HAS THE CONTENT?")
print("=" * 84)
FIELDS = ["RADIOLOGY_IMPRESSION", "RADIOLOGY_NARRATIVE", "RADIOLOGY_REPORT"]
sel = ", ".join(
    f"COUNTIF({f} IS NOT NULL AND LENGTH(CAST({f} AS STRING))>20) AS f{i}, "
    f"APPROX_QUANTILES(LENGTH(CAST({f} AS STRING)),100)[OFFSET(50)] AS m{i}, "
    f"APPROX_QUANTILES(LENGTH(CAST({f} AS STRING)),100)[OFFSET(90)] AS p{i}"
    for i, f in enumerate(FIELDS))
r = C.query(f"SELECT COUNT(*) AS n, {sel} FROM {D}.{T}").to_dataframe().iloc[0]
n = int(r["n"])
print(f"  {n:,} reports total\n")
print(f"  {'field':<26}{'with >20 chars':>16}{'%':>8}{'median len':>12}{'p90 len':>10}")
for i, f in enumerate(FIELDS):
    print(f"  {f:<26}{int(r[f'f{i}']):>16,}{int(r[f'f{i}'])/n:>8.0%}"
          f"{int(r[f'm{i}']):>12,}{int(r[f'p{i}']):>10,}")
BEST = FIELDS[int(pd.Series([r[f"f{i}"] for i in range(3)]).idxmax())]
print(f"\n  richest field: {BEST}")

# ---------------------------------------------------------------- 2. response language
print("\n" + "=" * 84)
print(f"2. DOES {BEST} CONTAIN RESPONSE LANGUAGE?  (1% sample, aggregate only)")
print("=" * 84)
PH = [("increase",   ["INTERVAL INCREASE", "ENLARG", "INCREASED IN SIZE", "PROGRESS"]),
      ("decrease",   ["INTERVAL DECREASE", "DECREASED IN SIZE", "SMALLER", "REGRESS"]),
      ("new_lesion", ["NEW LESION", "NEW METASTA", "NEW NODULE", "NEW FOCUS"]),
      ("stable",     ["STABLE", "NO SIGNIFICANT CHANGE", "UNCHANGED"]),
      ("no_disease", ["NO EVIDENCE OF DISEASE", "NO EVIDENCE OF RECURREN",
                      "NO EVIDENCE OF METASTA"]),
      ("recist",     ["RECIST", "PARTIAL RESPONSE", "COMPLETE RESPONSE",
                      "PROGRESSIVE DISEASE", "STABLE DISEASE"]),
      ("compares",   ["COMPARED TO", "COMPARISON", "PRIOR STUDY", "SINCE THE PREVIOUS"]),
      ("any_cancer", ["MASS", "LESION", "METASTA", "TUMOR", "NODULE", "MALIGNAN"])]
U = f"UPPER(CAST({BEST} AS STRING))"
sel = ", ".join("COUNTIF(" + " OR ".join(f"{U} LIKE '%{p}%'" for p in ps) + f") AS {k}"
                for k, ps in PH)
q = C.query(f"""SELECT COUNT(*) AS n, {sel},
  COUNT(DISTINCT CAST({BEST} AS STRING)) AS uniq
  FROM {D}.{T} TABLESAMPLE SYSTEM (1 PERCENT)
  WHERE {BEST} IS NOT NULL AND LENGTH(CAST({BEST} AS STRING)) > 20
""").to_dataframe().iloc[0]
m = int(q["n"])
print(f"  sampled {m:,} reports with text\n")
for k, _ in PH:
    v = int(q[k])
    print(f"    {k:<14}{v:>10,}  ({v/m:5.1%})  {'#'*int(v/m*34)}")
print(f"\n  distinct texts: {int(q['uniq']):,} of {m:,} "
      f"({int(q['uniq'])/m:.0%} unique)")
print("  low uniqueness = templated boilerplate; high = dictated prose.")

# ---------------------------------------------------------------- 3. note / note_nlp
print("\n" + "=" * 84)
print("3. THE OMOP `note` AND `note_nlp` TABLES -- never opened")
print("=" * 84)
for tab in ("note", "note_nlp"):
    try:
        cols = [x.column_name for x in C.query(
            f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='{tab}' "
            f"ORDER BY ordinal_position")]
        cnt = C.query(f"SELECT COUNT(*) n, COUNT(DISTINCT person_id) p "
                      f"FROM {D}.{tab}").to_dataframe().iloc[0]
        print(f"\n  {tab}: {int(cnt['n']):,} rows over {int(cnt['p']):,} patients")
        print(f"    columns: {cols}")
        if tab == "note":
            ty = C.query(f"""SELECT CAST(note_title AS STRING) AS t, COUNT(*) n
              FROM {D}.note GROUP BY 1 ORDER BY n DESC LIMIT 10""").to_dataframe()
            print("    commonest note titles:")
            print(ty.to_string(index=False))
        else:
            ty = C.query(f"""SELECT CAST(note_nlp_source_concept_id AS STRING) AS c,
              COUNT(*) n FROM {D}.note_nlp GROUP BY 1 ORDER BY n DESC LIMIT 6
            """).to_dataframe()
            print("    most frequent extracted concepts (already run by someone):")
            print(ty.to_string(index=False))
    except Exception as e:
        print(f"\n  {tab}: {type(e).__name__} {str(e)[:100]}")

# ---------------------------------------------------------------- 4. validation set
print("\n" + "=" * 84)
print("4. VALIDATION SET -- patients with text AND a marker progression date")
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
            AND r.{BEST} IS NOT NULL
            AND LENGTH(CAST(r.{BEST} AS STRING)) > 20""").to_dataframe().iloc[0]
        print(f"  {lbl:<30}{int(q2['pts']):>7,} patients, "
              f"{int(q2['reports']):>10,} reports")
    except Exception as e:
        print(f"  {lbl:<30}skipped ({type(e).__name__} {str(e)[:60]})")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"radiology_scope2 | best_field={BEST} | with_text={int(r[f'f{FIELDS.index(BEST)}'])} "
      f"| unique={int(q['uniq'])/m:.0%} | compares={int(q['compares'])/m:.0%} "
      f"| any_cancer={int(q['any_cancer'])/m:.0%}")

"""
glp1_scope.py -- can we look at GLP-1 agonists and liposarcoma? COUNTS ONLY.

Biologically reasonable question: liposarcoma is a malignancy of adipocytes and
GLP-1 agonists act on adipose metabolism. But liposarcoma is rare (~2-3 per
million/yr), so the intersection with GLP-1 exposure is likely tiny.

This deliberately fits NO model and computes NO p-value. With n in the tens a
significant result would more likely be noise than signal, and the confounding
is severe: GLP-1 users have diabetes and/or obesity, both of which affect cancer
outcomes independently. Immortal time bias too -- you must survive to be
prescribed the drug.

So: count first, decide after.
  1. how much liposarcoma is here (by MORPHOLOGY 8850-8858, not site)
  2. how many cancer patients ever had a GLP-1
  3. the intersection -- the number that settles it
  4. the better-powered alternative: GLP-1 exposure pan-cancer
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)

# ---------------------------------------------------------------- histology column
hist = [r.column_name for r in C.query(f"""
  SELECT column_name FROM {S}.COLUMNS
  WHERE table_name='FACT_CANCER_DATA_REPOSITORY'
    AND REGEXP_CONTAINS(UPPER(column_name), r'HISTOL|MORPHOL|ICD_O_3|BEHAV')
  ORDER BY column_name""")]
print(f"histology/morphology columns: {hist}")
HCOL = next((c for c in hist if "HISTOL" in c.upper()), None)
if HCOL is None:
    HCOL = next((c for c in hist if "MORPHOL" in c.upper()), None)
print(f"using: {HCOL}")

if HCOL:
    print("\n" + "=" * 78)
    print("LIPOSARCOMA (ICD-O-3 morphology 8850-8858)")
    print("=" * 78)
    lipo = C.query(f"""
      SELECT CAST({HCOL} AS STRING) AS histology,
             COUNT(DISTINCT PATIENT_DK) AS patients,
             MIN(EXTRACT(YEAR FROM DATE(DATE_OF_DIAGNOSIS))) AS first_yr,
             MAX(EXTRACT(YEAR FROM DATE(DATE_OF_DIAGNOSIS))) AS last_yr
      FROM {D}.FACT_CANCER_DATA_REPOSITORY
      WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'^885[0-8]')
      GROUP BY 1 ORDER BY patients DESC""").to_dataframe()
    if len(lipo):
        print(lipo.to_string(index=False))
        print(f"\n  TOTAL liposarcoma patients: {int(lipo['patients'].sum()):,}")
    else:
        print("  none found by that pattern -- histology may be coded differently")
        smp = C.query(f"""SELECT CAST({HCOL} AS STRING) v, COUNT(*) n
          FROM {D}.FACT_CANCER_DATA_REPOSITORY GROUP BY 1 ORDER BY n DESC LIMIT 12
        """).to_dataframe()
        print(smp.to_string(index=False))

# ---------------------------------------------------------------- GLP-1 exposure
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
GLP1 = ["SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS", "LIRAGLUTIDE", "VICTOZA",
        "SAXENDA", "DULAGLUTIDE", "TRULICITY", "EXENATIDE", "BYETTA", "BYDUREON",
        "TIRZEPATIDE", "MOUNJARO", "ZEPBOUND", "LIXISENATIDE", "ADLYXIN"]
like = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in GLP1)

print("\n" + "=" * 78)
print("GLP-1 AGONIST EXPOSURE, ANY CANCER PATIENT")
print("=" * 78)
mm = C.query(f"""SELECT {', '.join(namecols)} FROM {D}.DIM_MED_NAME
                 WHERE {like} LIMIT 12""").to_dataframe()
print(f"  matching drug records (sample of {len(mm)}):")
print(mm.to_string(index=False) if len(mm) else "  NONE -- no GLP-1 products found")

n_glp = C.query(f"""
  SELECT COUNT(DISTINCT t.PATIENT_DK) AS n,
         MIN(EXTRACT(YEAR FROM DATE(t.TREATMENT_DTM))) AS first_yr,
         MAX(EXTRACT(YEAR FROM DATE(t.TREATMENT_DTM))) AS last_yr
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL""").to_dataframe().iloc[0]
print(f"\n  patients ever prescribed a GLP-1: {int(n_glp['n']):,} "
      f"({n_glp['first_yr']}-{n_glp['last_yr']})")

# ---------------------------------------------------------------- the intersection
if HCOL:
    print("\n" + "=" * 78)
    print("THE NUMBER THAT SETTLES IT")
    print("=" * 78)
    x = C.query(f"""
      WITH lipo AS (
        SELECT DISTINCT PATIENT_DK, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx
        FROM {D}.FACT_CANCER_DATA_REPOSITORY
        WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'^885[0-8]') GROUP BY 1),
      g AS (
        SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS glp1_first
        FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
        WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1)
      SELECT COUNT(*) AS lipo_total,
             COUNTIF(g.PATIENT_DK IS NOT NULL) AS lipo_on_glp1,
             COUNTIF(g.glp1_first <= lipo.dx) AS glp1_before_dx,
             COUNTIF(g.glp1_first > lipo.dx) AS glp1_after_dx
      FROM lipo LEFT JOIN g USING (PATIENT_DK)""").to_dataframe().iloc[0]
    for k, lbl in [("lipo_total", "liposarcoma patients"),
                   ("lipo_on_glp1", "...ever on a GLP-1"),
                   ("glp1_before_dx", "......GLP-1 started BEFORE diagnosis"),
                   ("glp1_after_dx", "......GLP-1 started AFTER diagnosis")]:
        print(f"  {lbl:<44}{int(x[k]):>7,}")
    n = int(x["lipo_on_glp1"])
    print("\n  verdict: ", end="")
    if n < 30:
        print(f"{n} exposed patients. NOT ANSWERABLE. Report the count and stop --")
        print("           any p-value here would be noise, and the confounding")
        print("           (diabetes, obesity, immortal time) is unadjustable at this n.")
    elif n < 150:
        print(f"{n} exposed. Descriptive only -- no significance testing.")
    else:
        print(f"{n} exposed. Worth a properly designed look, WITH diabetes adjustment.")

# ---------------------------------------------------------------- better-powered
print("\n" + "=" * 78)
print("THE BETTER-POWERED QUESTION: GLP-1 exposure across ALL cancers")
print("=" * 78)
pan = C.query(f"""
  WITH reg AS (
    SELECT PATIENT_DK, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
           ANY_VALUE(SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3)) AS site3
    FROM {D}.FACT_CANCER_DATA_REPOSITORY
    WHERE DATE_OF_DIAGNOSIS IS NOT NULL
      AND CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C%' GROUP BY 1),
  g AS (
    SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS glp1_first
    FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
    WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1)
  SELECT reg.site3, COUNT(*) AS cancer_pts,
         COUNTIF(g.PATIENT_DK IS NOT NULL) AS on_glp1
  FROM reg LEFT JOIN g USING (PATIENT_DK)
  GROUP BY 1 HAVING on_glp1 >= 50 ORDER BY on_glp1 DESC LIMIT 15""").to_dataframe()
SITE = {"C50": "breast", "C61": "prostate", "C34": "lung", "C18": "colon",
        "C44": "skin", "C25": "pancreas", "C64": "kidney", "C67": "bladder",
        "C54": "uterus", "C20": "rectum", "C22": "liver", "C16": "stomach",
        "C56": "ovary", "C73": "thyroid", "C49": "soft tissue"}
print(f"  {'site':<6}{'':<14}{'cancer pts':>12}{'on GLP-1':>11}{'%':>7}")
for _, r in pan.iterrows():
    print(f"  {r['site3']:<6}{SITE.get(r['site3'],''):<14}{int(r['cancer_pts']):>12,}"
          f"{int(r['on_glp1']):>11,}{int(r['on_glp1'])/int(r['cancer_pts']):>7.1%}")
print("\n  here you would have thousands, and could actually adjust for diabetes,")
print("  BMI and immortal time. That is the answerable version of the question.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"glp1_scope | glp1_patients={int(n_glp['n'])} "
      f"| lipo_on_glp1={int(x['lipo_on_glp1']) if HCOL else 'NA'} "
      f"| pan_cancer_sites_over_50={len(pan)}")

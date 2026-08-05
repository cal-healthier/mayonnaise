"""
glp1_scope2.py -- fixed. Both zeros last run were my bugs.

  1. Wrong histology column: my selector matched HIGH_RISK_HISTOLOGIC_FEATURES
     (a staging descriptor, blank for 313,130 of 313,442). Morphology lives in
     HISTOLOGIC_TYPE_ICD_O_3.
  2. Wrong table for the drug: FACT_TREATMENT_DETAIL is the ONCOLOGY treatment
     table. GLP-1s are diabetes drugs and will not be there -- the 3 "matches"
     were IRB placebo products. General medications should be in OMOP
     drug_exposure, which we have never opened.

Still counts only. No model, no p-value.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)

# ---------------------------------------------------------------- 1. liposarcoma, properly
HCOL = "HISTOLOGIC_TYPE_ICD_O_3"
print("=" * 80)
print(f"LIPOSARCOMA via {HCOL} (morphology 8850-8858)")
print("=" * 80)
lipo = C.query(f"""
  SELECT CAST({HCOL} AS STRING) AS histology,
         COUNT(DISTINCT PATIENT_DK) AS patients
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'885[0-8]')
  GROUP BY 1 ORDER BY patients DESC""").to_dataframe()
if len(lipo):
    print(lipo.to_string(index=False))
    n_lipo = int(lipo["patients"].sum())
    print(f"\n  TOTAL liposarcoma patients: {n_lipo:,}")
else:
    n_lipo = 0
    print("  still none -- showing the most common values in this column:")
    print(C.query(f"""SELECT CAST({HCOL} AS STRING) v, COUNT(*) n
      FROM {D}.FACT_CANCER_DATA_REPOSITORY GROUP BY 1 ORDER BY n DESC LIMIT 10
    """).to_dataframe().to_string(index=False))

# for context: all soft tissue sarcoma
sts = C.query(f"""
  SELECT COUNT(DISTINCT PATIENT_DK) AS n FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'^(88[0-9]{{2}}|9(0[4-6]|1[0-2])[0-9])')
""").to_dataframe().iloc[0]["n"]
print(f"  (all sarcoma-range morphologies for context: {int(sts):,})")

# ---------------------------------------------------------------- 2. where are drugs?
print("\n" + "=" * 80)
print("WHERE ARE GENERAL (NON-ONCOLOGY) MEDICATIONS RECORDED?")
print("=" * 80)
tabs = C.query(f"""
  SELECT table_name, COUNT(*) AS cols FROM {S}.COLUMNS
  WHERE REGEXP_CONTAINS(LOWER(table_name),
        r'drug|medic|pharm|prescri|order|admin')
  GROUP BY 1 ORDER BY table_name""").to_dataframe()
print(tabs.to_string(index=False))

GLP1 = ["SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS", "LIRAGLUTIDE", "VICTOZA",
        "SAXENDA", "DULAGLUTIDE", "TRULICITY", "EXENATIDE", "BYETTA", "BYDUREON",
        "TIRZEPATIDE", "MOUNJARO", "ZEPBOUND", "LIXISENATIDE"]

def try_table(tab, idcol_hint=("PERSON_ID", "PATIENT_DK", "PATIENT_CLINIC_NUMBER")):
    try:
        cols = [r.column_name for r in C.query(
            f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='{tab}' "
            f"ORDER BY ordinal_position")]
    except Exception:
        return None
    txt = [c for c in cols if any(k in c.upper() for k in
           ("NAME", "DESCRIPTION", "SOURCE_VALUE", "CONCEPT_NAME", "GENERIC"))]
    ids = [c for c in cols if c.upper() in idcol_hint]
    if not txt or not ids:
        return None
    expr = ("UPPER(CONCAT(" + ", ' ', ".join(
        f"IFNULL(CAST({c} AS STRING),'')" for c in txt[:5]) + "))")
    like = " OR ".join(f"{expr} LIKE '%{d}%'" for d in GLP1)
    try:
        r = C.query(f"""SELECT COUNT(*) AS rows_,
              COUNT(DISTINCT CAST({ids[0]} AS STRING)) AS pts
              FROM {D}.{tab} WHERE {like}""").to_dataframe().iloc[0]
        return tab, ids[0], txt[:5], int(r["rows_"]), int(r["pts"]), like, expr
    except Exception as e:
        return None

print("\n  searching each candidate table for GLP-1 products:")
best = None
for t in list(tabs["table_name"]) + ["drug_exposure"]:
    got = try_table(t)
    if got:
        tab, idc, txt, rows, pts, like, expr = got
        print(f"    {tab:<38}{rows:>10,} rows  {pts:>8,} patients")
        if best is None or pts > best[4]:
            best = got
    else:
        print(f"    {t:<38}{'(no searchable name+id columns)':>30}")

if best is None:
    print("\n  NO table found with GLP-1 products. Tell Claude.")
    raise SystemExit

tab, idc, txt, rows, pts, like, expr = best
print(f"\n  best source: {tab} ({pts:,} patients, id column {idc})")
print(f"  searched columns: {txt}")
sample = C.query(f"""SELECT {', '.join(txt[:3])}, COUNT(*) n
  FROM {D}.{tab} WHERE {like} GROUP BY {', '.join(str(i+1) for i in range(len(txt[:3])))}
  ORDER BY n DESC LIMIT 12""").to_dataframe()
print("\n  what matched:")
print(sample.to_string(index=False))

# ---------------------------------------------------------------- 3. the intersection
print("\n" + "=" * 80)
print("THE NUMBER THAT SETTLES IT")
print("=" * 80)
if idc.upper() == "PERSON_ID":
    join = f"""
      JOIN {D}.person pe ON pe.person_id = g.{idc}
      JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
                              = CAST(pe.person_source_value AS STRING)"""
    pk = "p.PATIENT_DK"
elif idc.upper() == "PATIENT_CLINIC_NUMBER":
    join = f""" JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
                                        = CAST(g.{idc} AS STRING)"""
    pk = "p.PATIENT_DK"
else:
    join = ""; pk = f"g.{idc}"

res = C.query(f"""
  WITH lipo AS (
    SELECT DISTINCT PATIENT_DK FROM {D}.FACT_CANCER_DATA_REPOSITORY
    WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'885[0-8]')),
  glp AS (
    SELECT DISTINCT {pk} AS PATIENT_DK FROM {D}.{tab} g {join}
    WHERE {like})
  SELECT (SELECT COUNT(*) FROM lipo) AS lipo_n,
         (SELECT COUNT(*) FROM glp) AS glp_n,
         (SELECT COUNT(*) FROM lipo JOIN glp USING (PATIENT_DK)) AS both_n
""").to_dataframe().iloc[0]
print(f"  liposarcoma patients            {int(res['lipo_n']):>9,}")
print(f"  cancer patients ever on a GLP-1 {int(res['glp_n']):>9,}")
print(f"  BOTH                            {int(res['both_n']):>9,}   <-- decides it")
n = int(res["both_n"])
print("\n  verdict: ", end="")
if n < 30:
    print(f"{n} exposed. NOT ANSWERABLE -- report the count and stop.")
elif n < 150:
    print(f"{n} exposed. Descriptive only, no significance testing.")
else:
    print(f"{n} exposed. Worth a proper design, WITH diabetes adjustment.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"glp1_scope2 | lipo={int(res['lipo_n'])} | glp1={int(res['glp_n'])} "
      f"| both={n} | source_table={tab}")

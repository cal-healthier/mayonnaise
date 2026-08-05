"""
orders_glp1_v2.py -- discover FACT_ORDERS' real columns, then use them.

Last run assumed PATIENT_CLINIC_NUMBER and a date column named ORDER_DTM.
Neither was verified. This asks the schema first and builds the query from what
is actually there.

Same two goals:
  A. honest descriptive answer on liposarcoma + GLP-1 (72 patients, no p-values)
  B. profile the medication record we missed all day
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)

cols = C.query(f"""SELECT column_name, data_type FROM {S}.COLUMNS
  WHERE table_name='FACT_ORDERS' ORDER BY ordinal_position""").to_dataframe()
print("FACT_ORDERS columns:")
print(", ".join(cols["column_name"]))

up = {c.upper(): c for c in cols["column_name"]}
IDC = next((up[k] for k in ("PATIENT_DK", "PATIENT_CLINIC_NUMBER", "PERSON_ID")
            if k in up), None)
DATEC = next((c for c in cols["column_name"]
              if any(k in c.upper() for k in ("ORDER_DTM", "ORDER_DATE", "_DTM"))), None)
DRUGC = next((c for c in cols["column_name"]
              if "MED_GENERIC" in c.upper()), None)
print(f"\n  patient id -> {IDC}\n  date       -> {DATEC}\n  drug name  -> {DRUGC}")
if not (IDC and DRUGC):
    raise SystemExit("cannot proceed without id + drug columns -- tell Claude")

GLP1 = ["SEMAGLUTIDE","OZEMPIC","WEGOVY","RYBELSUS","LIRAGLUTIDE","VICTOZA",
        "SAXENDA","DULAGLUTIDE","TRULICITY","EXENATIDE","BYETTA","BYDUREON",
        "TIRZEPATIDE","MOUNJARO","ZEPBOUND","LIXISENATIDE"]
G = f"UPPER(IFNULL(CAST({DRUGC} AS STRING),''))"
like = " OR ".join(f"{G} LIKE '%{d}%'" for d in GLP1)
HCOL = "HISTOLOGIC_TYPE_ICD_O_3"
dsel = f"MIN(DATE({DATEC}))" if DATEC else "CAST(NULL AS DATE)"

print("\n" + "=" * 84)
print("A. LIPOSARCOMA + GLP-1 -- DESCRIPTIVE ONLY, NO TESTS")
print("=" * 84)
join_on = ("g.pid = CAST(l.PATIENT_DK AS STRING)" if IDC.upper() == "PATIENT_DK"
           else "g.pid = CAST(l.clinic AS STRING)")
lipo_sel = ("CAST(PATIENT_DK AS STRING) AS PATIENT_DK" if IDC.upper() == "PATIENT_DK"
            else "CAST(PATIENT_DK AS STRING) AS PATIENT_DK")
q = C.query(f"""
WITH lipo AS (
  SELECT CAST(PATIENT_DK AS STRING) AS PATIENT_DK,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(VITAL_STATUS AS STRING)) AS vital,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_seen
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'885[0-8]') GROUP BY 1),
g AS (
  SELECT CAST({IDC} AS STRING) AS pid, {dsel} AS first_glp1,
         MAX(CASE WHEN {G} LIKE '%WEIGHT LOSS%' THEN 1 ELSE 0 END) AS wl
  FROM {D}.FACT_ORDERS WHERE {like} GROUP BY 1)
SELECT l.PATIENT_DK, l.dx, l.vital, l.last_seen, g.first_glp1, g.wl
FROM lipo l LEFT JOIN g ON g.pid = l.PATIENT_DK
""").to_dataframe()
for c in ("dx", "last_seen", "first_glp1"):
    q[c] = pd.to_datetime(q[c], errors="coerce")
exp = q[q["first_glp1"].notna()]
print(f"  liposarcoma patients      {len(q):,}")
print(f"  ever prescribed a GLP-1   {len(exp):,}")
if len(exp):
    b = int((exp["first_glp1"] <= exp["dx"]).sum())
    a = int((exp["first_glp1"] > exp["dx"]).sum())
    print(f"    started BEFORE diagnosis  {b:,}")
    print(f"    started AFTER diagnosis   {a:,}")
    print(f"    labelled '(WEIGHT LOSS)'  {int(exp['wl'].sum()):,}  (rest likely diabetes)")
    yr = exp["first_glp1"].dt.year.value_counts().sort_index()
    print("\n  year of first GLP-1 order:")
    for y, n in yr.items():
        print(f"    {int(y)}  {'#'*min(int(n),40)} {int(n)}")
    post = exp[exp["first_glp1"] > exp["dx"]]
    fu = ((post["last_seen"] - post["first_glp1"]).dt.days / 365.25).dropna()
    if len(fu):
        print(f"\n  follow-up AFTER starting the drug (n={len(fu)}):")
        print(f"    median {fu.median():.2f}y | p75 {fu.quantile(.75):.2f}y | "
              f"max {fu.max():.2f}y")
        print(f"    with >=2 years of follow-up: {int((fu>=2).sum())}")
        print(f"    with >=3 years of follow-up: {int((fu>=3).sum())}")
print("\n  NOT TESTED. GLP-1 prescribing only scaled from ~2021, users have")
print("  diabetes/obesity (both affect cancer outcomes), and immortal time")
print("  applies. At this n a significant result would more likely be noise.")

print("\n" + "=" * 84)
print("B. FACT_ORDERS -- the medication record missed all day")
print("=" * 84)
dfilter = f", MIN(EXTRACT(YEAR FROM DATE({DATEC}))) y0, MAX(EXTRACT(YEAR FROM DATE({DATEC}))) y1" if DATEC else ""
t = C.query(f"""SELECT COUNT(*) rows_,
  COUNT(DISTINCT CAST({IDC} AS STRING)) pts{dfilter}
  FROM {D}.FACT_ORDERS""").to_dataframe().iloc[0]
span = f" | {int(t['y0'])}-{int(t['y1'])}" if DATEC else ""
print(f"  {int(t['rows_']):,} orders | {int(t['pts']):,} patients{span}")

CLASSES = {
 "corticosteroids (ICI confounder)": ["PREDNISONE","DEXAMETHASONE","METHYLPREDNISOLONE",
                                      "HYDROCORTISONE","PREDNISOLONE"],
 "diabetes drugs":                   ["METFORMIN","GLIPIZIDE","INSULIN","SITAGLIPTIN",
                                      "EMPAGLIFLOZIN","DAPAGLIFLOZIN"],
 "anticoagulants":                   ["WARFARIN","APIXABAN","RIVAROXABAN","ENOXAPARIN"],
 "statins":                          ["ATORVASTATIN","SIMVASTATIN","ROSUVASTATIN"],
 "beta blockers":                    ["METOPROLOL","ATENOLOL","CARVEDILOL"],
 "opioids":                          ["OXYCODONE","MORPHINE","HYDROMORPHONE","FENTANYL"],
 "bone agents":                      ["ZOLEDRONIC","DENOSUMAB","XGEVA","PROLIA"],
 "GLP-1 agonists":                   GLP1,
}
print(f"\n  {'drug class':<36}{'orders':>13}{'patients':>12}")
for name, drugs in CLASSES.items():
    lk = " OR ".join(f"{G} LIKE '%{d}%'" for d in drugs)
    r = C.query(f"""SELECT COUNT(*) n, COUNT(DISTINCT CAST({IDC} AS STRING)) p
      FROM {D}.FACT_ORDERS WHERE {lk}""").to_dataframe().iloc[0]
    print(f"  {name:<36}{int(r['n']):>13,}{int(r['p']):>12,}")

print("\n  steroids at checkpoint-inhibitor start are a known confounder of")
print("  immunotherapy outcomes -- invisible in every model built today.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"orders_glp1_v2 | id_col={IDC} | lipo={len(q)} | lipo_glp1={len(exp)} "
      f"| orders_rows={int(t['rows_'])} | orders_pts={int(t['pts'])}")

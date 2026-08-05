"""
orders_and_glp1.py -- two things.

A. THE HONEST LIPOSARCOMA ANSWER. 72 exposed patients. Describe the cohort --
   when they were exposed, for what indication, how much follow-up exists --
   and show why no test is warranted. NO p-values.

B. PROFILE FACT_ORDERS, which we have missed all day. We treated
   FACT_TREATMENT_DETAIL (the ONCOLOGY table) as "the drug table". Every
   non-cancer medication has been invisible: steroids, diabetes drugs,
   anticoagulants, cardiac drugs. That matters well beyond this question --
   steroid use at checkpoint-inhibitor start is a known confounder we could not
   see, comorbidity burden had no proxy, and concomitant medications were a
   whole trial-eligibility category we wrote off as unevaluable.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 62)

GLP1 = ["SEMAGLUTIDE","OZEMPIC","WEGOVY","RYBELSUS","LIRAGLUTIDE","VICTOZA",
        "SAXENDA","DULAGLUTIDE","TRULICITY","EXENATIDE","BYETTA","BYDUREON",
        "TIRZEPATIDE","MOUNJARO","ZEPBOUND","LIXISENATIDE"]
G = "UPPER(IFNULL(CAST(MED_GENERIC AS STRING),''))"
like = " OR ".join(f"{G} LIKE '%{d}%'" for d in GLP1)
HCOL = "HISTOLOGIC_TYPE_ICD_O_3"

print("=" * 84)
print("A. LIPOSARCOMA + GLP-1: DESCRIPTIVE ONLY")
print("=" * 84)
q = C.query(f"""
WITH lipo AS (
  SELECT PATIENT_DK, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(VITAL_STATUS AS STRING)) AS vital,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_seen
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE REGEXP_CONTAINS(CAST({HCOL} AS STRING), r'885[0-8]') GROUP BY 1),
g AS (
  SELECT CAST(p.PATIENT_DK AS STRING) AS pdk,
         MIN(DATE(o.ORDER_DTM)) AS first_glp1,
         MAX(CASE WHEN {G} LIKE '%WEIGHT LOSS%' THEN 1 ELSE 0 END) AS wl_label
  FROM {D}.FACT_ORDERS o
  JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
                          = CAST(o.PATIENT_CLINIC_NUMBER AS STRING)
  WHERE ({like}) AND o.ORDER_DTM IS NOT NULL
  GROUP BY 1)
SELECT CAST(l.PATIENT_DK AS STRING) AS pdk, l.dx, l.vital, l.last_seen,
       g.first_glp1, g.wl_label
FROM lipo l LEFT JOIN g ON g.pdk = CAST(l.PATIENT_DK AS STRING)
""").to_dataframe()
for c in ("dx", "last_seen", "first_glp1"):
    q[c] = pd.to_datetime(q[c])
exp = q[q["first_glp1"].notna()]
print(f"  liposarcoma patients            {len(q):,}")
print(f"  ever prescribed a GLP-1         {len(exp):,}")
if len(exp):
    print(f"    started BEFORE diagnosis      {int((exp['first_glp1'] <= exp['dx']).sum()):,}")
    print(f"    started AFTER diagnosis       {int((exp['first_glp1'] > exp['dx']).sum()):,}")
    print(f"    labelled '(WEIGHT LOSS)'      {int(exp['wl_label'].sum()):,}"
          f"   (rest likely diabetes)")
    yr = exp["first_glp1"].dt.year.value_counts().sort_index()
    print("\n  year of first GLP-1 order:")
    for y, n in yr.items():
        print(f"    {int(y)}  {'#'*int(n)} {int(n)}")
    post = exp[exp["first_glp1"] > exp["dx"]]
    if len(post):
        fu = (post["last_seen"] - post["first_glp1"]).dt.days / 365.25
        fu = fu.dropna()
        print(f"\n  follow-up AFTER starting the drug (post-diagnosis starters, "
              f"n={len(fu)}):")
        for lbl, v in [("median", fu.median()), ("p75", fu.quantile(.75)),
                       ("max", fu.max())]:
            print(f"    {lbl:<8}{v:>6.2f} years")
        print(f"    with >=2 years of follow-up: {int((fu >= 2).sum())}")
print("\n  NOT TESTED, and here is why:")
print("    - GLP-1 prescribing only scaled from ~2021, so exposed follow-up is short")
print("    - users have diabetes and/or obesity; both change cancer outcomes")
print("    - immortal time: you must survive to be prescribed anything")
print("    - at this n a 'significant' result is more likely noise than signal")
print("  Honest answer for the patient: too few people like them in this data.")

print("\n" + "=" * 84)
print("B. FACT_ORDERS -- the medication record we have been missing all day")
print("=" * 84)
tot = C.query(f"""SELECT COUNT(*) AS rows_,
  COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS pts,
  MIN(EXTRACT(YEAR FROM DATE(ORDER_DTM))) AS y0,
  MAX(EXTRACT(YEAR FROM DATE(ORDER_DTM))) AS y1
  FROM {D}.FACT_ORDERS""").to_dataframe().iloc[0]
print(f"  {int(tot['rows_']):,} orders | {int(tot['pts']):,} patients | "
      f"{int(tot['y0'])}-{int(tot['y1'])}")

CLASSES = {
 "corticosteroids (ICI confounder)": ["PREDNISONE","DEXAMETHASONE","METHYLPREDNISOLONE",
                                      "HYDROCORTISONE","PREDNISOLONE"],
 "metformin / diabetes":             ["METFORMIN","GLIPIZIDE","INSULIN","SITAGLIPTIN",
                                      "EMPAGLIFLOZIN","DAPAGLIFLOZIN"],
 "anticoagulants":                   ["WARFARIN","APIXABAN","RIVAROXABAN","ENOXAPARIN",
                                      "HEPARIN"],
 "statins":                          ["ATORVASTATIN","SIMVASTATIN","ROSUVASTATIN",
                                      "PRAVASTATIN"],
 "beta blockers":                    ["METOPROLOL","ATENOLOL","CARVEDILOL","PROPRANOLOL"],
 "opioids":                          ["OXYCODONE","MORPHINE","HYDROMORPHONE","FENTANYL"],
 "bone agents (prostate/bone mets)": ["ZOLEDRONIC","DENOSUMAB","XGEVA","PROLIA","PAMIDRONATE"],
 "GLP-1 agonists":                   GLP1,
}
print(f"\n  {'drug class':<36}{'orders':>13}{'patients':>12}")
for name, drugs in CLASSES.items():
    lk = " OR ".join(f"{G} LIKE '%{d}%'" for d in drugs)
    r = C.query(f"""SELECT COUNT(*) AS n,
      COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS p
      FROM {D}.FACT_ORDERS WHERE {lk}""").to_dataframe().iloc[0]
    print(f"  {name:<36}{int(r['n']):>13,}{int(r['p']):>12,}")

print("\n  why this matters beyond today's question:")
print("    - steroids at checkpoint-inhibitor start are a KNOWN confounder we")
print("      could not see; now we can")
print("    - medication counts are a usable proxy for comorbidity burden, which")
print("      we had no way to adjust for in any model today")
print("    - concomitant medications were a whole trial-eligibility criterion")
print("      category written off as unevaluable -- it is evaluable")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"orders_and_glp1 | lipo={len(q)} | lipo_glp1={len(exp)} "
      f"| orders_rows={int(tot['rows_'])} | orders_pts={int(tot['pts'])}")

"""
glp1_logline.py -- the plain descriptive picture, and a logline you can say out loud.

No p-values, no models. Just: who are these patients, when did they start the
drug, and what does the record actually show about what happened afterwards.

Includes the one thing that has to be said alongside any raw comparison:
a patient must SURVIVE long enough to be prescribed a drug, so GLP-1 users will
look better than non-users even if the drug does nothing at all. The script
measures how big that head start is rather than just asserting it.
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

GLP1 = ["SEMAGLUTIDE","OZEMPIC","WEGOVY","RYBELSUS","LIRAGLUTIDE","VICTOZA",
        "SAXENDA","DULAGLUTIDE","TRULICITY","EXENATIDE","BYETTA","BYDUREON",
        "TIRZEPATIDE","MOUNJARO","ZEPBOUND","LIXISENATIDE"]
G = "UPPER(IFNULL(CAST(MED_GENERIC AS STRING),''))"
like = " OR ".join(f"{G} LIKE '%{d}%'" for d in GLP1)

q = C.query(f"""
WITH lipo AS (
  SELECT CAST(PATIENT_DK AS STRING) AS pdk,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(VITAL_STATUS AS STRING)) AS vital,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_seen,
         ANY_VALUE(CAST(HISTOLOGIC_TYPE_ICD_O_3 AS STRING)) AS hist
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE REGEXP_CONTAINS(CAST(HISTOLOGIC_TYPE_ICD_O_3 AS STRING), r'885[0-8]')
  GROUP BY 1),
g AS (
  SELECT CAST(PATIENT_DK AS STRING) AS pdk,
         MIN(DATE(ORDER_APPROVE_DTM)) AS first_glp1
  FROM {D}.FACT_ORDERS WHERE {like} AND ORDER_APPROVE_DTM IS NOT NULL
  GROUP BY 1)
SELECT l.*, g.first_glp1 FROM lipo l LEFT JOIN g USING (pdk)
""").to_dataframe()
for c in ("dx", "last_seen", "first_glp1"):
    q[c] = pd.to_datetime(q[c], errors="coerce")
q["dead"] = q["vital"].astype(str).str.upper().str.contains("DEAD|DECEASED|0 ")
q["fu_yrs"] = (q["last_seen"] - q["dx"]).dt.days / 365.25
q["glp1"] = q["first_glp1"].notna()
E = q[q["glp1"]]; N = q[~q["glp1"]]

print("=" * 82)
print("WHO ARE THEY?")
print("=" * 82)
print(f"  liposarcoma patients in the record   {len(q):,}")
print(f"  ever prescribed a GLP-1              {len(E):,}  ({len(E)/len(q):.1%})")
print(f"  diagnosis years                      {int(q['dx'].dt.year.min())}"
      f"-{int(q['dx'].dt.year.max())}")
after = E[E["first_glp1"] > E["dx"]]
print(f"  started the drug AFTER diagnosis     {len(after):,} of {len(E):,}")
print(f"  first GLP-1 order in 2022 or later   "
      f"{int((E['first_glp1'].dt.year >= 2022).sum()):,} of {len(E):,}")

print("\n" + "=" * 82)
print("WHY A RAW COMPARISON WOULD MISLEAD")
print("=" * 82)
gap = ((after["first_glp1"] - after["dx"]).dt.days / 365.25)
print(f"  time from diagnosis to starting the drug: median {gap.median():.1f} years")
print(f"  -> every one of these {len(after)} patients had to SURVIVE that long")
print(f"     to be prescribed anything. Someone who died in year 1 can never")
print(f"     appear in the GLP-1 group. That head start alone makes users look")
print(f"     better, whatever the drug does.")
alive_at = (N["fu_yrs"] >= gap.median()).mean()
print(f"  for scale: only {alive_at:.0%} of non-users are even followed that long.")

print("\n" + "=" * 82)
print("WHAT THE RECORD ACTUALLY SHOWS  (raw, uncorrected -- read with the above)")
print("=" * 82)
print(f"  {'group':<26}{'n':>7}{'died':>9}{'median follow-up':>20}")
for lbl, d in [("ever on a GLP-1", E), ("never on a GLP-1", N)]:
    print(f"  {lbl:<26}{len(d):>7,}{d['dead'].mean():>9.0%}"
          f"{d['fu_yrs'].median():>18.1f}y")

fu_drug = ((after["last_seen"] - after["first_glp1"]).dt.days / 365.25).dropna()
print(f"\n  follow-up AFTER starting the drug: median "
      f"{fu_drug.median()*12:.0f} months")
print(f"    with >=1 year:  {int((fu_drug >= 1).sum())} of {len(fu_drug)}")
print(f"    with >=3 years: {int((fu_drug >= 3).sum())} of {len(fu_drug)}")

print("\n" + "=" * 82)
print("THE LOGLINE")
print("=" * 82)
print(f"""
  Mayo's records hold {len(q):,} people with liposarcoma. {len(E)} of them were
  ever prescribed a GLP-1 drug -- about {len(E)/len(q)*100:.0f} in 100.
  {int((E['first_glp1'].dt.year >= 2022).sum())} of those {len(E)} started it in 2022 or later, and {len(after)} started
  only after their cancer had already been diagnosed.

  Half of them have been followed for less than {fu_drug.median()*12:.0f} months since starting
  the drug, and just {int((fu_drug >= 3).sum())} have three years or more.

  So the data can tell you WHO is taking it. It cannot yet tell you what
  happens to them, because almost nobody has been on it long enough for an
  outcome to have occurred. That is not a null result -- it is an empty one.

  And a raw comparison would flatter the drug regardless: patients waited a
  median of {gap.median():.1f} years after diagnosis before starting it, so anyone who
  died early could never have been counted as a user.
""")

print("-" * 74)
print("FINAL LINE:")
print(f"glp1_logline | lipo={len(q)} | on_glp1={len(E)} | after_dx={len(after)} "
      f"| med_fu_months={fu_drug.median()*12:.0f} | ge3yr={int((fu_drug>=3).sum())} "
      f"| med_dx_to_drug_yrs={gap.median():.1f}")

"""
exclusion_audit.py -- STUDY 1, step 1: who does oncology leave out?

Take real recruiting trials for the commonest cancers, apply the entry
requirements we CAN check to Mayo's actual patients, and count who is shut out
and by which specific rule.

Three things make this newly possible:
  - the criteria parser already works (50 trials -> 1,055 clauses)
  - FACT_ORDERS (2.0B orders, 3.3M patients) makes concomitant-medication and
    comorbidity rules checkable -- a category we had written off
  - dense labs give organ-function thresholds, the commonest quantitative rule

This step sizes it: for each cancer, how many patients do we have with enough
recent lab data to be assessed against a trial at all? That is the denominator
for everything else, and it decides which cancers are in.

Counts only.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

fm = pd.read_csv("feature_map_final.csv")
CID = ",".join(str(int(x)) for x in fm["concept_id"])

# the labs that carry the commonest quantitative entry rules
ORGAN = {"CBC_Hb": "haemoglobin", "CBC_PLT": "platelets", "CBC_WBC": "white cells",
         "Chem_ALB": "albumin", "Chem_ALP": "alk phos", "Chem_ALT": "ALT"}
oc = fm[fm["feature"].isin(ORGAN)]["concept_id"].astype(int).tolist()
OC = ",".join(str(x) for x in oc)
print(f"organ-function concepts: {len(oc)} of {len(fm)} mapped features")

SITES = {"C50": "breast", "C34": "lung", "C61": "prostate", "C18": "colon",
         "C25": "pancreas", "C56": "ovary", "C64": "kidney", "C67": "bladder",
         "C44": "skin/melanoma", "C22": "liver", "C16": "stomach", "C20": "rectum"}
site_sql = ",".join(f"'{s}'" for s in SITES)

print("\npulling assessable-population counts by cancer ...")
q = C.query(f"""
WITH reg AS (
  SELECT PATIENT_DK,
         SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3) AS site3,
         MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(CAST(VITAL_STATUS AS STRING)) AS vital,
         MAX(DATE(DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_seen,
         ANY_VALUE(CAST(DERIVED_SUMMARY_STAGE_2018 AS STRING)) AS stage
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3) IN ({site_sql})
  GROUP BY 1,2),
br AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
       FROM {D}.DIM_PATIENT),
pe AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS pid
       FROM {D}.person GROUP BY 1),
lab AS (
  SELECT pe.clinic, COUNT(DISTINCT m.measurement_concept_id) AS n_organ,
         MAX(DATE(m.measurement_date)) AS last_lab
  FROM pe JOIN {D}.measurement m ON m.person_id = pe.pid
  WHERE m.measurement_concept_id IN ({OC}) AND m.value_as_number IS NOT NULL
  GROUP BY 1)
SELECT reg.site3,
       COUNT(*) AS patients,
       COUNTIF(lab.clinic IS NOT NULL) AS with_any_organ_lab,
       COUNTIF(lab.n_organ >= 4) AS with_4plus_organ_labs,
       COUNTIF(lab.n_organ >= 4 AND lab.last_lab >= DATE_SUB(reg.last_seen,
               INTERVAL 180 DAY)) AS assessable,
       COUNTIF(reg.stage IS NOT NULL AND CAST(reg.stage AS STRING) NOT IN
               ('9','99','','Unknown')) AS with_stage
FROM reg JOIN br USING (PATIENT_DK) LEFT JOIN lab ON lab.clinic = br.clinic
GROUP BY 1 ORDER BY patients DESC
""").to_dataframe()

print("\n" + "=" * 92)
print("THE ASSESSABLE POPULATION -- who could even be checked against a trial?")
print("=" * 92)
print(f"  {'site':<6}{'cancer':<16}{'patients':>10}{'any labs':>11}"
      f"{'4+ organ labs':>15}{'recent too':>12}{'has stage':>11}")
for _, r in q.iterrows():
    print(f"  {r['site3']:<6}{SITES.get(r['site3'],''):<16}{int(r['patients']):>10,}"
          f"{int(r['with_any_organ_lab']):>11,}{int(r['with_4plus_organ_labs']):>15,}"
          f"{int(r['assessable']):>12,}{int(r['with_stage']):>11,}")
tot = q[["patients", "assessable"]].sum()
print(f"\n  {'TOTAL':<22}{int(tot['patients']):>10,}{'':>11}{'':>15}"
      f"{int(tot['assessable']):>12,}")
print(f"  assessable share: {tot['assessable']/tot['patients']:.0%}")

print("\n" + "=" * 92)
print("MEDICATION-BASED RULES -- newly checkable via FACT_ORDERS")
print("=" * 92)
G = "UPPER(IFNULL(CAST(MED_GENERIC AS STRING),''))"
RULES = {
  "on systemic steroids (common exclusion)": ["PREDNISONE","DEXAMETHASONE",
                                              "METHYLPREDNISOLONE","PREDNISOLONE"],
  "on anticoagulation (common exclusion)":   ["WARFARIN","APIXABAN","RIVAROXABAN"],
  "strong CYP3A inhibitors (interaction)":   ["KETOCONAZOLE","ITRACONAZOLE",
                                              "CLARITHROMYCIN","RITONAVIR"],
  "immunosuppressants (ICI exclusion)":      ["TACROLIMUS","CYCLOSPORINE",
                                              "MYCOPHENOLATE","AZATHIOPRINE"],
  "insulin (diabetes severity)":             ["INSULIN"],
}
print(f"  {'rule':<44}{'patients affected':>20}")
for name, drugs in RULES.items():
    lk = " OR ".join(f"{G} LIKE '%{d}%'" for d in drugs)
    n = list(C.query(f"""SELECT COUNT(DISTINCT PATIENT_DK) n
      FROM {D}.FACT_ORDERS WHERE {lk}"""))[0].n
    print(f"  {name:<44}{int(n):>20,}")
print("\n  these were the 'concomitant medication' criteria we wrote off as")
print("  unevaluable when only the oncology drug table was visible.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"exclusion_audit | cancers={len(q)} | patients={int(tot['patients'])} "
      f"| assessable={int(tot['assessable'])} "
      f"| share={tot['assessable']/tot['patients']:.0%}")

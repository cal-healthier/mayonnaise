"""
ovarian_size.py -- is there enough ovarian cancer to run the platinum study?

Counts only. Cheap. Decides go / no-go before any real extraction.

The study, if it is viable: predict time to CA-125 progression after
first-line platinum chemotherapy. Directly parallel to the prostate/PSA work
(GCIG CA-125 progression criteria play the role PCWG3 played there), and it
sidesteps the broken registry recurrence field entirely.

Ovarian, fallopian tube and primary peritoneal cancers are one disease
clinically -- counted separately here so we can decide whether to pool.

Bonus at the end: quick drug counts for the other two candidate studies.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. CA-125 concept
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
ca125 = cc[cc["name_l"].str.contains("ca 125|ca-125|cancer ag 125|cancer antigen 125", na=False)
           | cc["code"].astype(str).isin(["10334-1", "83088-5", "2006-4"])]
ca125 = ca125.sort_values("n_persons", ascending=False)
print("CA-125 concepts in measurement:")
if len(ca125):
    print(ca125[["cid", "name", "code", "n_persons", "p50", "unit"]].head(6).to_string(index=False))
else:
    print("  NONE FOUND -- study 3 is dead on arrival, tell Claude")
CA_SQL = ",".join(str(int(x)) for x in ca125.head(4)["cid"]) if len(ca125) else "0"

# ---------------------------------------------------------------- 2. the disease
print("\n" + "=" * 72)
print("HOW MUCH OVARIAN / TUBAL / PERITONEAL CANCER IS THERE?")
print("=" * 72)
sites = C.query(f"""
  SELECT SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING), 1, 3) AS site3,
         COUNT(DISTINCT PATIENT_DK) AS n
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C56%'
     OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C57%'
     OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C48%'
  GROUP BY 1 ORDER BY n DESC
""").to_dataframe()
LBL = {"C56": "ovary", "C57": "other female genital (incl. fallopian tube)",
       "C48": "peritoneum / retroperitoneum (primary peritoneal)"}
for _, r in sites.iterrows():
    print(f"  {r['site3']}  {LBL.get(r['site3'], ''):<48}{r['n']:>7,} patients")

# ---------------------------------------------------------------- 3. the drugs
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = " || ' ' || ".join(f"UPPER(CAST({c} AS STRING))" for c in namecols)

FAMILIES = {
    "platinum":      ["CARBOPLATIN", "CISPLATIN"],
    "taxane":        ["PACLITAXEL", "DOCETAXEL"],
    "PARP inhibitor": ["OLAPARIB", "NIRAPARIB", "RUCAPARIB", "LYNPARZA", "ZEJULA", "RUBRACA"],
    "bevacizumab":   ["BEVACIZUMAB", "AVASTIN"],
    "checkpoint inhibitor (study 1)": ["PEMBROLIZUMAB", "NIVOLUMAB", "ATEZOLIZUMAB",
                                       "DURVALUMAB", "IPILIMUMAB", "KEYTRUDA", "OPDIVO",
                                       "TECENTRIQ", "IMFINZI", "YERVOY"],
}
def like(drugs):
    return " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in drugs)

print("\n" + "=" * 72)
print("DRUG EXPOSURE (patients ever treated, ANY cancer)")
print("=" * 72)
for fam, drugs in FAMILIES.items():
    n = list(C.query(f"""
      SELECT COUNT(DISTINCT t.PATIENT_DK) AS n
      FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE ({like(drugs)}) AND t.TREATMENT_DTM IS NOT NULL
    """))[0].n
    print(f"  {fam:<34}{n:>8,} patients")

# ---------------------------------------------------------------- 4. the actual cohort
CTES = f"""
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING), 1, 3)) AS site3
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND (CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C56%'
      OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C57%'
      OR CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C48%')
  GROUP BY 1
),
plat AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date,
         MAX(DATE(t.TREATMENT_DTM)) AS tx_last
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({like(FAMILIES['platinum'])}) AND t.TREATMENT_DTM IS NOT NULL
  GROUP BY 1
),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
           FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, r.site3, pl.tx_date, pl.tx_last
  FROM reg r JOIN plat pl USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK) JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND pl.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY pl.tx_date) = 1
)
"""
print("\n" + "=" * 72)
print("THE COHORT: single-primary, platinum-treated")
print("=" * 72)
co = C.query(f"WITH {CTES} SELECT site3, COUNT(*) AS n FROM cohort GROUP BY 1 ORDER BY n DESC"
             ).to_dataframe()
for _, r in co.iterrows():
    print(f"  {r['site3']}  {LBL.get(r['site3'], ''):<48}{r['n']:>7,}")
total = int(co["n"].sum())
print(f"  {'':<54}{total:>7,}  TOTAL")

# ---------------------------------------------------------------- 5. CA-125 coverage
print("\n" + "=" * 72)
print("CA-125 COVERAGE (the thing that makes or breaks it)")
print("=" * 72)
cov = C.query(f"""
WITH {CTES},
m AS (
  SELECT c.clinic,
         COUNTIF(DATE(x.measurement_date) BETWEEN DATE_SUB(c.tx_date, INTERVAL 90 DAY)
                                              AND c.tx_date) AS n_pre,
         COUNTIF(DATE(x.measurement_date) > c.tx_date) AS n_post,
         MAX(DATE_DIFF(DATE(x.measurement_date), c.tx_date, DAY)) AS last_day
  FROM cohort c
  JOIN {D}.measurement x ON x.person_id = c.person_id
  WHERE x.measurement_concept_id IN ({CA_SQL}) AND x.value_as_number IS NOT NULL
  GROUP BY 1
)
SELECT COUNT(*) AS any_ca125,
       COUNTIF(n_pre >= 1) AS with_baseline,
       COUNTIF(n_pre >= 1 AND n_post >= 3) AS baseline_and_3_post,
       COUNTIF(n_pre >= 1 AND n_post >= 3 AND last_day >= 180) AS modelable,
       APPROX_QUANTILES(n_post, 100)[OFFSET(50)] AS median_post_draws,
       APPROX_QUANTILES(last_day, 100)[OFFSET(50)] AS median_last_day
FROM m
""").to_dataframe().iloc[0]
print(f"  any CA-125 at all:                     {int(cov['any_ca125']):>7,}")
print(f"  with a baseline value:                 {int(cov['with_baseline']):>7,}")
print(f"  baseline + >=3 follow-up:              {int(cov['baseline_and_3_post']):>7,}")
print(f"  ...and >=180 days follow-up:           {int(cov['modelable']):>7,}   <-- MODELABLE n")
print(f"  median CA-125 draws after treatment:   {int(cov['median_post_draws']):>7,}")
print(f"  median follow-up:                      {cov['median_last_day']/365.25:>7.1f} years")

n_model = int(cov["modelable"])
print("\n  verdict: ", end="")
if n_model >= 1500:
    print("GO -- comparable to the prostate study, run it")
elif n_model >= 600:
    print("VIABLE but smaller than prostate; expect wider error bars")
else:
    print("TOO SMALL -- pick study 1 (checkpoint inhibitors) or 2 (neutropenia) instead")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"ovarian_size | cohort={total} | any_ca125={int(cov['any_ca125'])} "
      f"| modelable={n_model} | median_draws={int(cov['median_post_draws'])} "
      f"| median_fu_yrs={cov['median_last_day']/365.25:.1f}")

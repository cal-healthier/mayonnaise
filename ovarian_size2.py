"""
ovarian_size2.py -- fixed.

Bug 1: built the searchable drug text with `a || ' ' || b`. In BigQuery ANY
       NULL in a concatenation makes the whole expression NULL, and
       NULL LIKE '%X%' is never true -- so every row with a missing name
       column silently vanished. Platinum came back 0. Now IFNULL'd.
       (The hormone-drug search got lucky: those rows had all columns filled.)
Bug 2: hardcoded LOINC 83088-5 as CA-125. It is CORTISOL. Removed -- match on
       name, and print what matched so it is auditable.

Prints the actual matched drug rows this time, so a 0 can never be mistaken
for a real answer again.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 220)

def n_of(df_or_val, default=0):
    try:
        return int(df_or_val)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------------- 1. CA-125
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
ca = cc[cc["name_l"].str.contains("ca 125|ca-125|cancer ag 125|cancer antigen 125", na=False)]
ca = ca[~ca["name_l"].str.contains("cortisol", na=False)].sort_values("n_persons", ascending=False)
print("CA-125 concepts (name-matched, cortisol excluded):")
print(ca[["cid", "name", "code", "n_persons", "p50", "unit"]].head(6).to_string(index=False))
CA_SQL = ",".join(str(int(x)) for x in ca.head(4)["cid"])

# ---------------------------------------------------------------- 2. drug search, NULL-safe
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
print(f"\nsearchable columns: {namecols}")

FAMILIES = {
    "platinum":        ["CARBOPLATIN", "CISPLATIN", "OXALIPLATIN"],
    "taxane":          ["PACLITAXEL", "DOCETAXEL", "TAXOL", "TAXOTERE", "ABRAXANE"],
    "PARP inhibitor":  ["OLAPARIB", "NIRAPARIB", "RUCAPARIB", "LYNPARZA", "ZEJULA", "RUBRACA"],
    "bevacizumab":     ["BEVACIZUMAB", "AVASTIN"],
    "checkpoint inhib": ["PEMBROLIZUMAB", "NIVOLUMAB", "ATEZOLIZUMAB", "DURVALUMAB",
                         "IPILIMUMAB", "KEYTRUDA", "OPDIVO", "TECENTRIQ", "IMFINZI", "YERVOY"],
}
def like(drugs):
    return " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in drugs)

print("\n" + "=" * 74)
print("SANITY CHECK -- what actually matches 'platinum'?")
print("=" * 74)
pm = C.query(f"""
  SELECT {', '.join(namecols)} FROM {D}.DIM_MED_NAME
  WHERE {like(FAMILIES['platinum'])} LIMIT 10
""").to_dataframe()
print(f"  matched drug records (sample of {len(pm)}):")
print(pm.to_string(index=False) if len(pm) else "  STILL ZERO -- stop, tell Claude")

print("\n" + "=" * 74)
print("DRUG EXPOSURE (distinct patients ever treated, any cancer)")
print("=" * 74)
counts = {}
for fam, drugs in FAMILIES.items():
    counts[fam] = n_of(list(C.query(f"""
      SELECT COUNT(DISTINCT t.PATIENT_DK) AS n
      FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE ({like(drugs)}) AND t.TREATMENT_DTM IS NOT NULL
    """))[0].n)
    print(f"  {fam:<20}{counts[fam]:>9,} patients")

# ---------------------------------------------------------------- 3. cohort
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
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({like(FAMILIES['platinum'])}) AND t.TREATMENT_DTM IS NOT NULL
  GROUP BY 1
),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
           FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, r.site3, pl.tx_date
  FROM reg r JOIN plat pl USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK) JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND pl.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY pl.tx_date) = 1
)
"""
LBL = {"C56": "ovary", "C57": "other female genital (incl. tube)",
       "C48": "peritoneum (primary peritoneal)"}
print("\n" + "=" * 74)
print("THE COHORT: single-primary ovarian/tubal/peritoneal, platinum-treated")
print("=" * 74)
co = C.query(f"WITH {CTES} SELECT site3, COUNT(*) AS n FROM cohort GROUP BY 1 ORDER BY n DESC"
             ).to_dataframe()
for _, r in co.iterrows():
    print(f"  {r['site3']}  {LBL.get(r['site3'], ''):<38}{r['n']:>7,}")
total = int(co["n"].sum()) if len(co) else 0
print(f"  {'':<44}{total:>7,}  TOTAL")

# ---------------------------------------------------------------- 4. CA-125 coverage
print("\n" + "=" * 74)
print("CA-125 COVERAGE (makes or breaks it)")
print("=" * 74)
cov = C.query(f"""
WITH {CTES},
m AS (
  SELECT c.clinic,
         COUNTIF(DATE(x.measurement_date) BETWEEN DATE_SUB(c.tx_date, INTERVAL 90 DAY)
                                              AND c.tx_date) AS n_pre,
         COUNTIF(DATE(x.measurement_date) > c.tx_date) AS n_post,
         MAX(DATE_DIFF(DATE(x.measurement_date), c.tx_date, DAY)) AS last_day
  FROM cohort c JOIN {D}.measurement x ON x.person_id = c.person_id
  WHERE x.measurement_concept_id IN ({CA_SQL}) AND x.value_as_number IS NOT NULL
  GROUP BY 1
)
SELECT COUNT(*) AS any_ca, COUNTIF(n_pre >= 1) AS with_base,
       COUNTIF(n_pre >= 1 AND n_post >= 3) AS base_3post,
       COUNTIF(n_pre >= 1 AND n_post >= 3 AND last_day >= 180) AS modelable,
       APPROX_QUANTILES(n_post, 100)[OFFSET(50)] AS med_draws,
       APPROX_QUANTILES(last_day, 100)[OFFSET(50)] AS med_last
FROM m
""").to_dataframe().iloc[0]
print(f"  any CA-125:                          {n_of(cov['any_ca']):>7,}")
print(f"  with a baseline value:               {n_of(cov['with_base']):>7,}")
print(f"  baseline + >=3 follow-up:            {n_of(cov['base_3post']):>7,}")
print(f"  ...and >=180 days follow-up:         {n_of(cov['modelable']):>7,}   <-- MODELABLE n")
print(f"  median CA-125 draws after treatment: {n_of(cov['med_draws']):>7,}")
print(f"  median follow-up:                    {n_of(cov['med_last'])/365.25:>7.1f} years")

n_model = n_of(cov["modelable"])
print("\n  verdict: ", end="")
if n_model >= 1500:
    print("GO -- comparable to the prostate study")
elif n_model >= 600:
    print("VIABLE, smaller than prostate; wider error bars")
else:
    print("TOO SMALL -- switch to checkpoint inhibitors or neutropenia")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"ovarian_size2 | platinum_pts={counts['platinum']} | checkpoint={counts['checkpoint inhib']} "
      f"| parp={counts['PARP inhibitor']} | cohort={total} | modelable={n_model} "
      f"| med_draws={n_of(cov['med_draws'])} | med_fu_yrs={n_of(cov['med_last'])/365.25:.1f}")

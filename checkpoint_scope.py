"""
checkpoint_scope.py -- what endpoints are actually derivable for immunotherapy?

Unlike PSA and CA-125 there is NO blood marker for checkpoint inhibitor
response. So before any modelling, find out what we can actually measure.

Four candidate endpoints, all checked here for feasibility:
  A. TIME TO NEXT TREATMENT -- patient switches to a different drug => the
     checkpoint inhibitor failed. Standard real-world proxy. Needs the
     treatment table to show a genuine switch, not concurrent combination
     therapy (pembrolizumab + carboplatin is standard first-line now).
  B. OVERALL SURVIVAL from checkpoint start -- clean, but prognosis not response.
  C. IMMUNE-RELATED THYROID DYSFUNCTION -- thyroiditis is the commonest irAE and
     it is directly visible in TSH. Normal TSH before, abnormal after.
  D. IMMUNE-RELATED HEPATITIS -- ALT rise, same logic.

C and D are the interesting ones: toxicity prediction is clinically actionable
(monitoring intensity) and irAEs are also associated with BETTER response, so
they are a response correlate too.

Counts and coverage only. No model. Decides the design.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 220)

def n_of(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------------- 0. what records death?
print("=" * 76)
print("WHERE IS DEATH / VITAL STATUS RECORDED?")
print("=" * 76)
tabs = C.query(f"""
  SELECT table_name FROM {S}.TABLES
  WHERE LOWER(table_name) LIKE '%death%' OR LOWER(table_name) LIKE '%vital%'
     OR LOWER(table_name) LIKE '%person%' ORDER BY table_name
""").to_dataframe()
print("  candidate tables:", ", ".join(tabs["table_name"].tolist()[:12]) or "none")
dcols = C.query(f"""
  SELECT table_name, column_name FROM {S}.COLUMNS
  WHERE (LOWER(column_name) LIKE '%death%' OR LOWER(column_name) LIKE '%deceased%'
      OR LOWER(column_name) LIKE '%vital_status%')
  ORDER BY table_name LIMIT 25
""").to_dataframe()
for _, r in dcols.iterrows():
    print(f"    {r['table_name']:<42}{r['column_name']}")

# ---------------------------------------------------------------- 1. the drugs
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")

ICI = {
    "pembrolizumab": ["PEMBROLIZUMAB", "KEYTRUDA"],
    "nivolumab":     ["NIVOLUMAB", "OPDIVO"],
    "atezolizumab":  ["ATEZOLIZUMAB", "TECENTRIQ"],
    "durvalumab":    ["DURVALUMAB", "IMFINZI"],
    "ipilimumab":    ["IPILIMUMAB", "YERVOY"],
    "cemiplimab":    ["CEMIPLIMAB", "LIBTAYO"],
    "avelumab":      ["AVELUMAB", "BAVENCIO"],
}
def like(ds):
    return " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in ds)
ALL_ICI = like([d for v in ICI.values() for d in v])

print("\n" + "=" * 76)
print("WHICH CHECKPOINT INHIBITORS, AND HOW MANY PATIENTS?")
print("=" * 76)
for name, ds in ICI.items():
    n = n_of(list(C.query(f"""
      SELECT COUNT(DISTINCT t.PATIENT_DK) AS n
      FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE ({like(ds)}) AND t.TREATMENT_DTM IS NOT NULL"""))[0].n)
    print(f"  {name:<16}{n:>8,}")

# ---------------------------------------------------------------- 2. cohort
CTES = f"""
ici AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS ici_date
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({ALL_ICI}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1),
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3)) AS site3
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C%' GROUP BY 1),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
           FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, r.site3, i.ici_date
  FROM ici i JOIN reg r USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK) JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND i.ici_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY i.ici_date) = 1)
"""
tot = n_of(list(C.query(f"WITH {CTES} SELECT COUNT(*) AS n FROM cohort"))[0].n)
print(f"\ncohort: single-primary, registry-linked, on a checkpoint inhibitor: {tot:,}")

print("\n  top cancer sites (this is the pan-cancer breadth):")
sites = C.query(f"WITH {CTES} SELECT site3, COUNT(*) AS n FROM cohort GROUP BY 1 "
                f"ORDER BY n DESC LIMIT 14").to_dataframe()
SITE = {"C34": "lung", "C43": "melanoma", "C64": "kidney", "C67": "bladder",
        "C50": "breast", "C18": "colon", "C25": "pancreas", "C22": "liver",
        "C32": "larynx", "C09": "tonsil", "C11": "nasopharynx", "C15": "oesophagus",
        "C16": "stomach", "C61": "prostate", "C56": "ovary", "C71": "brain",
        "C77": "lymph node", "C44": "skin", "C10": "oropharynx", "C20": "rectum"}
for _, r in sites.iterrows():
    print(f"    {r['site3']}  {SITE.get(r['site3'],''):<14}{r['n']:>7,}")

# ---------------------------------------------------------------- 3. endpoint A: TTNT
print("\n" + "=" * 76)
print("ENDPOINT A -- TIME TO NEXT TREATMENT (a genuine switch, not a combination)")
print("=" * 76)
ttnt = C.query(f"""
WITH {CTES},
nxt AS (
  SELECT c.clinic,
         MIN(CASE WHEN DATE(t.TREATMENT_DTM) > DATE_ADD(c.ici_date, INTERVAL 28 DAY)
                  THEN DATE(t.TREATMENT_DTM) END) AS next_date,
         MAX(DATE(t.TREATMENT_DTM)) AS last_tx
  FROM cohort c
  JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic
  JOIN {D}.FACT_TREATMENT_DETAIL t ON t.PATIENT_DK = p.PATIENT_DK
  JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE UPPER(CAST(m.MED_THERAPEUTIC_CLASS_DESCRIPTION AS STRING)) = 'ANTINEOPLASTICS'
    AND NOT ({ALL_ICI})
    AND t.TREATMENT_DTM IS NOT NULL
  GROUP BY 1)
SELECT COUNT(*) AS with_any_other_drug,
       COUNTIF(next_date IS NOT NULL) AS switched_after_28d,
       APPROX_QUANTILES(DATE_DIFF(next_date, last_tx, DAY), 100)[OFFSET(50)] AS dummy
FROM nxt
""").to_dataframe().iloc[0]
print(f"  patients with any non-checkpoint antineoplastic:  "
      f"{n_of(ttnt['with_any_other_drug']):>7,}")
print(f"  ...starting >28 days after checkpoint (a switch): "
      f"{n_of(ttnt['switched_after_28d']):>7,}   <-- events for endpoint A")

# ---------------------------------------------------------------- 4. endpoints C/D: irAE
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
def find(pat, exclude=None):
    h = cc[cc["name_l"].str.contains(pat, na=False)]
    if exclude:
        h = h[~h["name_l"].str.contains(exclude, na=False)]
    return h.sort_values("n_persons", ascending=False)
tsh = find(r"thyrotropin|thyroid stimulating hormone|\btsh\b")
alt = find(r"alanine aminotransferase", "ratio")
print("\n" + "=" * 76)
print("ENDPOINTS C/D -- IMMUNE-RELATED TOXICITY, VISIBLE IN LABS")
print("=" * 76)
for lbl, df in [("TSH (thyroiditis)", tsh), ("ALT (hepatitis)", alt)]:
    print(f"  {lbl}:")
    if len(df):
        print(df[["cid", "name", "code", "n_persons", "p50", "unit"]].head(3).to_string(index=False))
    else:
        print("    NONE FOUND")
TSH_SQL = ",".join(str(int(x)) for x in tsh.head(3)["cid"]) if len(tsh) else "0"
ALT_SQL = ",".join(str(int(x)) for x in alt.head(3)["cid"]) if len(alt) else "0"

cov = C.query(f"""
WITH {CTES},
t AS (
  SELECT c.clinic,
    COUNTIF(m.measurement_concept_id IN ({TSH_SQL})
            AND DATE(m.measurement_date) BETWEEN DATE_SUB(c.ici_date, INTERVAL 180 DAY)
                                             AND c.ici_date) AS tsh_pre,
    COUNTIF(m.measurement_concept_id IN ({TSH_SQL})
            AND DATE(m.measurement_date) > c.ici_date) AS tsh_post,
    COUNTIF(m.measurement_concept_id IN ({ALT_SQL})
            AND DATE(m.measurement_date) BETWEEN DATE_SUB(c.ici_date, INTERVAL 180 DAY)
                                             AND c.ici_date) AS alt_pre,
    COUNTIF(m.measurement_concept_id IN ({ALT_SQL})
            AND DATE(m.measurement_date) > c.ici_date) AS alt_post,
    MAX(DATE_DIFF(DATE(m.measurement_date), c.ici_date, DAY)) AS last_day
  FROM cohort c JOIN {D}.measurement m ON m.person_id = c.person_id
  WHERE m.value_as_number IS NOT NULL
  GROUP BY 1)
SELECT COUNT(*) AS any_labs,
       COUNTIF(tsh_pre >= 1 AND tsh_post >= 2) AS thyroid_evaluable,
       COUNTIF(alt_pre >= 1 AND alt_post >= 2) AS liver_evaluable,
       COUNTIF(last_day >= 180) AS fu_180d,
       APPROX_QUANTILES(last_day, 100)[OFFSET(50)] AS med_last
FROM t
""").to_dataframe().iloc[0]
print(f"\n  patients with any labs at all:              {n_of(cov['any_labs']):>7,}")
print(f"  TSH before + >=2 after  (thyroid endpoint): {n_of(cov['thyroid_evaluable']):>7,}")
print(f"  ALT before + >=2 after  (liver endpoint):   {n_of(cov['liver_evaluable']):>7,}")
print(f"  >=180 days of lab follow-up:                {n_of(cov['fu_180d']):>7,}")
print(f"  median lab follow-up:                       "
      f"{n_of(cov['med_last'])/365.25:>7.1f} years")

print("\n" + "-" * 72)
print("FINAL LINE:")
print(f"checkpoint_scope | cohort={tot} | switched={n_of(ttnt['switched_after_28d'])} "
      f"| thyroid={n_of(cov['thyroid_evaluable'])} | liver={n_of(cov['liver_evaluable'])} "
      f"| fu180={n_of(cov['fu_180d'])} | top_site={sites['site3'].iloc[0] if len(sites) else 'NA'}")

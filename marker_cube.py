"""
marker_cube.py -- step 1 of the treatment-response world model.

Claim to be tested: treatment response is ONE phenomenon -- a tumour-burden
marker perturbed by therapy, then escaping -- shared across cancers, markers
and drug classes. PSA under hormone therapy, CA-125 under platinum, CEA under
chemo, AFP under systemic therapy.

If that is true, a model trained on one should transfer to another. Nobody has
shown that. It is the difference between a world model and a pile of
disease-specific predictors.

This sizes the pooled cohort: for every (cancer, marker, treatment) triple, how
many patients have a DENSE trajectory -- baseline plus enough follow-up to see
response and escape. Decides which arms are in.

Counts only. No model.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

cc = pd.read_parquet("measurement_concepts.parquet")
cc["nl"] = cc["name"].astype(str).str.lower()
def concepts(pat, exclude=None, k=4):
    h = cc[cc["nl"].str.contains(pat, na=False)]
    if exclude:
        h = h[~h["nl"].str.contains(exclude, na=False)]
    h = h.sort_values("n_persons", ascending=False).head(k)
    return ",".join(str(int(x)) for x in h["cid"]), h

medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS "
    f"WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
def like(ds):
    return " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in ds)

ARMS = [
  ("prostate",   "C61", r"prostate specific|prostate-specific", None,
   ["LEUPROLIDE","GOSERELIN","DEGARELIX","TRIPTORELIN","HISTRELIN","LUPRON",
    "ELIGARD","ZOLADEX","FIRMAGON"], "PSA / hormone therapy"),
  ("ovary",      "C56|C57|C48", r"ca 125|cancer ag 125|cancer antigen 125", "cortisol",
   ["CARBOPLATIN","CISPLATIN"], "CA-125 / platinum"),
  ("colorectal", "C18|C19|C20", r"carcinoembryonic", None,
   ["FLUOROURACIL","CAPECITABINE","OXALIPLATIN","IRINOTECAN","FOLFOX","FOLFIRI"],
   "CEA / chemotherapy"),
  ("liver",      "C22", r"alpha.?1?.?fetoprotein|alpha fetoprotein", None,
   ["SORAFENIB","LENVATINIB","ATEZOLIZUMAB","BEVACIZUMAB","NEXAVAR","LENVIMA"],
   "AFP / systemic therapy"),
  ("pancreas",   "C25", r"ca 19-9|cancer ag 19-9|carbohydrate antigen 19", None,
   ["GEMCITABINE","FLUOROURACIL","NAB-PACLITAXEL","ABRAXANE","IRINOTECAN",
    "OXALIPLATIN","CAPECITABINE"], "CA 19-9 / chemotherapy"),
  ("breast",     "C50", r"ca 15-3|cancer ag 15-3|ca 27", None,
   ["ANASTROZOLE","LETROZOLE","EXEMESTANE","TAMOXIFEN","FULVESTRANT"],
   "CA 15-3 / endocrine therapy"),
  ("thyroid",    "C73", r"thyroglobulin", "antibod",
   ["LEVOTHYROXINE","LENVATINIB","SORAFENIB"], "thyroglobulin / therapy"),
]

print(f"{'arm':<12}{'marker / treatment':<30}{'on tx':>8}{'w/ marker':>11}"
      f"{'DENSE':>8}{'med draws':>11}{'med f/u':>9}")
print("-" * 92)
rows = []
for arm, sites, pat, exc, drugs, lbl in ARMS:
    cid_sql, hits = concepts(pat, exc)
    if not cid_sql:
        print(f"{arm:<12}{lbl:<30}   no marker concept found")
        continue
    site_sql = " OR ".join(
        f"CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE '{s}%'" for s in sites.split("|"))
    try:
        q = C.query(f"""
        WITH reg AS (
          SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx
          FROM {D}.FACT_CANCER_DATA_REPOSITORY
          WHERE DATE_OF_DIAGNOSIS IS NOT NULL AND ({site_sql}) GROUP BY 1),
        tx AS (
          SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
          FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
          WHERE ({like(drugs)}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1),
        br AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
               FROM {D}.DIM_PATIENT),
        pe AS (SELECT CAST(person_source_value AS STRING) AS clinic,
                      MIN(person_id) AS person_id FROM {D}.person GROUP BY 1),
        co AS (
          SELECT b.clinic, p.person_id, tx.tx_date FROM reg r
          JOIN tx USING (PATIENT_DK) JOIN br b USING (PATIENT_DK)
          JOIN pe p ON p.clinic = b.clinic
          WHERE r.n_prim = 1 AND tx.tx_date >= r.dx
          QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY tx.tx_date) = 1),
        mk AS (
          SELECT co.clinic,
            COUNTIF(DATE(m.measurement_date) BETWEEN DATE_SUB(co.tx_date, INTERVAL 90 DAY)
                                                 AND co.tx_date) AS pre,
            COUNTIF(DATE(m.measurement_date) > co.tx_date) AS post,
            MAX(DATE_DIFF(DATE(m.measurement_date), co.tx_date, DAY)) AS last_day
          FROM co JOIN {D}.measurement m ON m.person_id = co.person_id
          WHERE m.measurement_concept_id IN ({cid_sql}) AND m.value_as_number IS NOT NULL
          GROUP BY 1)
        SELECT (SELECT COUNT(*) FROM co) AS on_tx,
               COUNT(*) AS with_marker,
               COUNTIF(pre >= 1 AND post >= 4 AND last_day >= 180) AS dense,
               APPROX_QUANTILES(post, 100)[OFFSET(50)] AS med_draws,
               APPROX_QUANTILES(last_day, 100)[OFFSET(50)] AS med_fu
        FROM mk""").to_dataframe().iloc[0]
        d = int(q["dense"])
        rows.append({"arm": arm, "label": lbl, "dense": d})
        print(f"{arm:<12}{lbl:<30}{int(q['on_tx']):>8,}{int(q['with_marker']):>11,}"
              f"{d:>8,}{int(q['med_draws']):>11}{int(q['med_fu'])/365.25:>8.1f}y")
    except Exception as e:
        print(f"{arm:<12}{lbl:<30}   ERROR {type(e).__name__}: {str(e)[:60]}")

R = pd.DataFrame(rows)
if len(R):
    tot = int(R["dense"].sum())
    print("-" * 92)
    print(f"{'POOLED':<12}{'all arms':<30}{'':>8}{'':>11}{tot:>8,}")
    ok = R[R["dense"] >= 300]
    print(f"\n  arms with >=300 dense trajectories: {len(ok)} "
          f"({', '.join(ok['arm'])})")
    print(f"  pooled trainable cohort: {int(ok['dense'].sum()):,} patients")
    print("\n  transfer test would be: train on the largest arm, predict the others")
    print("  zero-shot -- different marker, different drug class, different disease.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"marker_cube | arms={len(R)} | pooled_dense={int(R['dense'].sum()) if len(R) else 0} "
      f"| usable_arms={int((R['dense']>=300).sum()) if len(R) else 0}")

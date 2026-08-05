"""
psa_cohort.py -- Oncoformer Fig 6F reproduction, step 1: BUILD AND CHECK THE LABEL.

Prostate cancer patients who started hormone-blocking treatment. PSA is drawn
every few weeks, so whether the treatment worked is visible in the data --
no registry field involved.

  baseline PSA = last PSA in the 90 days before treatment
  nadir PSA    = lowest PSA in the 9 months after treatment
  responder    = nadir <= 50% of baseline   (the standard clinical definition)

NO MODEL FITTED HERE. We look at the label first -- that is the lesson the
recurrence dead end taught us. The number that matters is the responder rate:
if ~95% respond there is nothing to predict and we switch to a continuous
endpoint (how deep the drop, how long it lasts).
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA.COLUMNS"
pd.set_option("display.width", 200)

# ---------------------------------------------------------------- 1. find PSA
cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc.name.astype(str).str.lower()
psa = cc[cc.name_l.str.contains("prostate specific|prostate-specific", na=False)
         | cc.code.astype(str).isin(["2857-1", "83112-3", "35741-8", "19195-7"])]
psa = psa.sort_values("n_persons", ascending=False)
print("PSA concepts found in measurement:")
print(psa[["cid", "name", "code", "n_persons", "p50", "unit"]].head(8).to_string(index=False))
PSA_CIDS = [int(x) for x in psa.head(6).cid.tolist()]
if not PSA_CIDS:
    raise SystemExit("no PSA concept found -- stop, tell Claude")
PSA_SQL = ",".join(str(c) for c in PSA_CIDS)

# ---------------------------------------------------------------- 2. find ADT drugs
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S} WHERE table_name='DIM_MED_NAME'")]
print(f"\nDIM_MED_NAME columns: {medcols}")
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
searchable = " || ' ' || ".join(f"UPPER(CAST({c} AS STRING))" for c in namecols)

ADT = ["LEUPROLIDE", "GOSERELIN", "DEGARELIX", "TRIPTORELIN", "HISTRELIN",
       "LUPRON", "ELIGARD", "ZOLADEX", "FIRMAGON", "TRELSTAR"]
like = " OR ".join(f"{searchable} LIKE '%{d}%'" for d in ADT)
meds = C.query(f"""
    SELECT MED_NAME_DK, {', '.join(namecols)}
    FROM {D}.DIM_MED_NAME WHERE {like}
""").to_dataframe()
print(f"\nhormone-blocking drug records matched: {len(meds)}")
if len(meds) == 0:
    raise SystemExit("no ADT drugs matched -- stop, tell Claude")
print(meds.head(12).to_string(index=False))

# ---------------------------------------------------------------- 3. cohort + PSA
CTES = f"""
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C61%'
  GROUP BY 1
),
adt AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS tx_date
  FROM {D}.FACT_TREATMENT_DETAIL t
  JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL
  GROUP BY 1
),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic
           FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, a.tx_date
  FROM reg r
  JOIN adt a USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK)
  JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND a.tx_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY a.tx_date) = 1
)
"""
n = list(C.query(f"WITH {CTES} SELECT COUNT(*) AS n FROM cohort"))[0].n
print(f"\nprostate patients on hormone-blocking therapy: {n:,}")

print("pulling PSA around treatment start ...")
psa_v = C.query(f"""
WITH {CTES}
SELECT c.clinic, c.dx, c.tx_date, DATE(m.measurement_date) AS d, m.value_as_number AS psa
FROM cohort c
JOIN {D}.measurement m ON m.person_id = c.person_id
WHERE m.measurement_concept_id IN ({PSA_SQL})
  AND m.value_as_number IS NOT NULL AND m.value_as_number >= 0
  AND DATE(m.measurement_date) BETWEEN DATE_SUB(c.tx_date, INTERVAL 90 DAY)
                                   AND DATE_ADD(c.tx_date, INTERVAL 365 DAY)
""").to_dataframe()
psa_v["tx_date"] = pd.to_datetime(psa_v.tx_date)
psa_v["d"] = pd.to_datetime(psa_v.d)
psa_v["day"] = (psa_v.d - psa_v.tx_date).dt.days
psa_v.to_parquet("psa_values.parquet")
print(f"  PSA rows: {len(psa_v):,}   patients with any PSA: {psa_v.clinic.nunique():,}")

# ---------------------------------------------------------------- 4. the label
pre = psa_v[psa_v.day <= 0].sort_values("day").groupby("clinic").tail(1)[["clinic", "psa", "day"]]
pre = pre.rename(columns={"psa": "baseline", "day": "baseline_day"}).set_index("clinic")
post = psa_v[(psa_v.day > 0) & (psa_v.day <= 270)]
npost = post.groupby("clinic").size().rename("n_post")
nadir = post.loc[post.groupby("clinic").psa.idxmin()][["clinic", "psa", "day"]]
nadir = nadir.rename(columns={"psa": "nadir", "day": "days_to_nadir"}).set_index("clinic")

L = pre.join(nadir, how="inner").join(npost, how="inner")
L = L[(L.baseline > 0) & (L.n_post >= 2)]
L["pct_change"] = (L.nadir - L.baseline) / L.baseline * 100
L["responder"] = (L.nadir <= 0.5 * L.baseline).astype(int)
L.to_parquet("psa_label.parquet")

print("\n" + "=" * 72)
print("THE LABEL")
print("=" * 72)
print(f"  patients with baseline PSA + >=2 follow-up PSAs: {len(L):,}")
print(f"  responders (PSA fell by half or more):           {L.responder.sum():,} "
      f"({L.responder.mean():.1%})   <-- THE NUMBER")
print(f"  non-responders:                                  {(1-L.responder).sum():,}")
print(f"\n  median baseline PSA {L.baseline.median():.1f} | median nadir {L.nadir.median():.2f}")
print(f"  median drop {L.pct_change.median():.0f}% | median days to nadir "
      f"{L.days_to_nadir.median():.0f}")
print(f"  median follow-up PSAs per patient {L.n_post.median():.0f}")

print("\n  spread of PSA change (is there anything to predict?):")
for q in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    p{q:<3} {L.pct_change.quantile(q/100):>8.1f}%")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"psa_cohort | on_adt={n} | with_psa={psa_v.clinic.nunique()} | labeled={len(L)} "
      f"| responder_rate={L.responder.mean():.1%} | median_drop={L.pct_change.median():.0f}% "
      f"| iqr={L.pct_change.quantile(.25):.0f}to{L.pct_change.quantile(.75):.0f}")

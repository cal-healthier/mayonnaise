"""
checkpoint_thyroid.py -- who develops immune-related thyroid dysfunction?

Checkpoint inhibitors have no blood marker for response, so we predict the
thing that IS cleanly measurable: immune-related thyroiditis, visible in TSH.
Clinically actionable (it sets monitoring intensity) and, because irAEs are
associated with BETTER response, it doubles as a response correlate.

  eligible : normal TSH (0.4-4.5 mIU/L) in the 180 days before the first
             checkpoint dose -- so we are not picking up pre-existing disease
  event    : TSH > 10 (overt hypothyroidism) or < 0.1 (thyroiditis /
             thyrotoxicosis) at any point after
  censored : last TSH draw

POSITIVE CONTROL, printed before any model: ipilimumab (anti-CTLA-4) causes
substantially more immune toxicity than the PD-1 drugs. If our label does not
reproduce that, the label is wrong and nothing downstream matters. This is the
same trick that caught the broken recurrence field with Oncotype DX.

Overall survival is built too -- death is recorded in four places.
Everything cached to parquet as it goes.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 220)
TSH_LO, TSH_HI = 0.4, 4.5          # normal range, mIU/L
EV_LO, EV_HI = 0.1, 10.0           # event thresholds

medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
ICI_DRUGS = ["PEMBROLIZUMAB", "KEYTRUDA", "NIVOLUMAB", "OPDIVO", "ATEZOLIZUMAB",
             "TECENTRIQ", "DURVALUMAB", "IMFINZI", "IPILIMUMAB", "YERVOY",
             "CEMIPLIMAB", "LIBTAYO", "AVELUMAB", "BAVENCIO"]
ALL_ICI = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in ICI_DRUGS)
CTLA4 = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in ["IPILIMUMAB", "YERVOY"])

cc = pd.read_parquet("measurement_concepts.parquet")
cc["name_l"] = cc["name"].astype(str).str.lower()
tsh_c = cc[cc["name_l"].str.contains(r"thyrotropin|thyroid stimulating hormone", na=False)]
TSH_SQL = ",".join(str(int(x)) for x in
                   tsh_c.sort_values("n_persons", ascending=False).head(3)["cid"])
print(f"TSH concepts: {TSH_SQL}")

CTES = f"""
ici AS (
  SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) AS ici_date,
         MAX(CASE WHEN ({CTLA4}) THEN 1 ELSE 0 END) AS got_ctla4
  FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
  WHERE ({ALL_ICI}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1),
reg AS (
  SELECT PATIENT_DK, COUNT(*) AS n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) AS dx,
         ANY_VALUE(SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3)) AS site3
  FROM {D}.FACT_CANCER_DATA_REPOSITORY
  WHERE DATE_OF_DIAGNOSIS IS NOT NULL
    AND CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C%' GROUP BY 1),
bridge AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
                  PATIENT_DEATH_DATE FROM {D}.DIM_PATIENT),
pers AS (SELECT CAST(person_source_value AS STRING) AS clinic, MIN(person_id) AS person_id
         FROM {D}.person GROUP BY 1),
cohort AS (
  SELECT b.clinic, p.person_id, r.dx, r.site3, i.ici_date, i.got_ctla4,
         DATE(b.PATIENT_DEATH_DATE) AS death_date
  FROM ici i JOIN reg r USING (PATIENT_DK)
  JOIN bridge b USING (PATIENT_DK) JOIN pers p ON p.clinic = b.clinic
  WHERE r.n_prim = 1 AND i.ici_date >= r.dx
  QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY i.ici_date) = 1)
"""

# ---------------------------------------------------------------- 1. TSH history
if os.path.exists("ici_tsh.parquet"):
    tv = pd.read_parquet("ici_tsh.parquet")
    print("loaded cached ici_tsh.parquet")
else:
    print("pulling TSH history ...")
    tv = C.query(f"""
    WITH {CTES}
    SELECT c.clinic, c.site3, c.ici_date, c.got_ctla4, c.death_date,
           DATE(m.measurement_date) AS d, m.value_as_number AS tsh
    FROM cohort c JOIN {D}.measurement m ON m.person_id = c.person_id
    WHERE m.measurement_concept_id IN ({TSH_SQL})
      AND m.value_as_number IS NOT NULL AND m.value_as_number >= 0
    """).to_dataframe()
    for col in ("ici_date", "d", "death_date"):
        tv[col] = pd.to_datetime(tv[col])
    tv["day"] = (tv["d"] - tv["ici_date"]).dt.days
    tv.to_parquet("ici_tsh.parquet")
print(f"  TSH rows {len(tv):,}   patients {tv['clinic'].nunique():,}")

# ---------------------------------------------------------------- 2. the label
base = tv[(tv["day"] <= 0) & (tv["day"] >= -180)].sort_values("day").groupby("clinic").tail(1)
base = base.set_index("clinic")
elig = base[(base["tsh"] >= TSH_LO) & (base["tsh"] <= TSH_HI)]
print(f"\n  with a pre-treatment TSH:      {len(base):,}")
print(f"  ...normal at baseline:         {len(elig):,} "
      f"({len(elig)/max(len(base),1):.0%})  <- eligible")

post = tv[(tv["day"] > 0) & tv["clinic"].isin(elig.index)].sort_values(["clinic", "day"])
post = post.copy()
post["abn"] = (post["tsh"] > EV_HI) | (post["tsh"] < EV_LO)
rows = []
for clinic, g in post.groupby("clinic", sort=False):
    days, abn = g["day"].values, g["abn"].values
    hit = days[abn]
    rows.append({"clinic": clinic, "irae": int(len(hit) > 0),
                 "time": int(hit[0]) if len(hit) else int(days[-1]),
                 "n_post": len(g), "last_day": int(days[-1])})
E = pd.DataFrame(rows).set_index("clinic")
E = E.join(elig[["site3", "got_ctla4", "ici_date", "death_date"]], how="inner")
E = E[E["n_post"] >= 2]
E["base_tsh"] = elig["tsh"].reindex(E.index)
E.to_parquet("ici_thyroid_label.parquet")

print("\n" + "=" * 76)
print("THE LABEL: time to immune-related thyroid dysfunction")
print("=" * 76)
print(f"  eligible with >=2 follow-up TSH:  {len(E):,}")
print(f"  developed thyroid dysfunction:    {int(E['irae'].sum()):,} "
      f"({E['irae'].mean():.1%})")
print(f"  median time to event:             "
      f"{E.loc[E['irae']==1,'time'].median()/30.4:.1f} months")
print(f"  median follow-up (no event):      "
      f"{E.loc[E['irae']==0,'time'].median()/30.4:.1f} months")
print(f"  median TSH draws after start:     {E['n_post'].median():.0f}")

# ---------------------------------------------------------------- 3. POSITIVE CONTROL
print("\n" + "=" * 76)
print("POSITIVE CONTROL -- anti-CTLA-4 (ipilimumab) must show MORE toxicity")
print("=" * 76)
for flag, lbl in [(1, "got ipilimumab (CTLA-4)"), (0, "PD-1/PD-L1 only")]:
    s = E[E["got_ctla4"] == flag]
    if len(s):
        print(f"  {lbl:<26} n={len(s):>6,}   irAE rate {s['irae'].mean():6.1%}")
ipi = E.loc[E["got_ctla4"] == 1, "irae"].mean() if (E["got_ctla4"] == 1).any() else np.nan
pd1 = E.loc[E["got_ctla4"] == 0, "irae"].mean() if (E["got_ctla4"] == 0).any() else np.nan
ratio = ipi / pd1 if pd1 else np.nan
print(f"\n  ratio {ratio:.2f}x   -> ", end="")
if ratio >= 1.15:
    print("LABEL VALIDATED: reproduces the known CTLA-4 toxicity signal.")
elif ratio >= 1.0:
    print("WEAK: directionally right but small. Treat results cautiously.")
else:
    print("LABEL FAILS ITS POSITIVE CONTROL. Do not model this. Tell Claude.")

print("\n  irAE rate by cancer site (should be broadly similar -- it is a drug effect):")
top = E["site3"].value_counts().head(8).index
for s in top:
    sub = E[E["site3"] == s]
    print(f"    {s}  n={len(sub):>5,}  irAE {sub['irae'].mean():6.1%}")

# ---------------------------------------------------------------- 4. survival label
E["died"] = E["death_date"].notna().astype(int)
E["os_days"] = np.where(E["death_date"].notna(),
                        (E["death_date"] - E["ici_date"]).dt.days,
                        E["last_day"])
E = E[E["os_days"] > 0]
print("\n" + "=" * 76)
print("SECONDARY ENDPOINT: overall survival from first checkpoint dose")
print("=" * 76)
print(f"  deaths recorded: {int(E['died'].sum()):,} ({E['died'].mean():.1%})")
print(f"  median OS (died):        {E.loc[E['died']==1,'os_days'].median()/365.25:.2f} years")
print(f"  median follow-up (alive): {E.loc[E['died']==0,'os_days'].median()/365.25:.2f} years")
E.to_parquet("ici_thyroid_label.parquet")

print("\n" + "-" * 72)
print("FINAL LINE:")
print(f"checkpoint_thyroid | eligible={len(E)} | irae={int(E['irae'].sum())} "
      f"({E['irae'].mean():.0%}) | ctla4_ratio={ratio:.2f} "
      f"| deaths={int(E['died'].sum())} | med_mo={E.loc[E['irae']==1,'time'].median()/30.4:.1f}")

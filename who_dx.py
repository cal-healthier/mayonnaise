"""
who_dx.py -- who is actually diagnosed with cancer in 2021+?

"Diagnosed 2021+" = MIN(DATE_OF_DIAGNOSIS) >= 2021, i.e. their FIRST-EVER cancer
is recent. Two questions about who they are:

  1. HOW LONG had they been a Mayo patient before that diagnosis?
     lead = (first cancer dx) - (first note). Large lead = established patient
     (seen for other things, then got cancer). ~0 lead = new arrival.
  2. WHAT cancers are being newly diagnosed 2021+?

Lead time comes from the cached recency table (free). Cancer type needs the
site at the earliest diagnosis -- one cheap registry query.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
CUT = 2021
TYPE = {"C61": "prostate", "C50": "breast", "C34": "lung", "C18": "colon",
        "C25": "pancreas", "C71": "brain", "C43": "melanoma", "C44": "skin",
        "C56": "ovary", "C64": "kidney", "C67": "bladder", "C22": "liver",
        "C16": "stomach", "C53": "cervix", "C54": "uterus/corpus", "C73": "thyroid",
        "C20": "rectum", "C15": "esophagus", "C90": "myeloma", "C91": "leukemia",
        "C92": "leukemia", "C82": "lymphoma", "C83": "lymphoma", "C85": "lymphoma",
        "C77": "lymph node", "C26": "GI other"}

R = pd.read_parquet("recency.parquet")
R["dx"] = pd.to_datetime(R["dx"], errors="coerce")
R["first_note"] = pd.to_datetime(R["first_note"], errors="coerce")

sc = "dx_site.parquet"
if os.path.exists(sc):
    S = pd.read_parquet(sc)
else:
    sql = f"""
    SELECT PATIENT_DK, site3 FROM (
      SELECT PATIENT_DK,
             SUBSTR(CAST(SITE_PRIMARY_ICD_O_3 AS STRING),1,3) AS site3,
             ROW_NUMBER() OVER (PARTITION BY PATIENT_DK
                                ORDER BY DATE(DATE_OF_DIAGNOSIS)) AS rn
      FROM {D}.FACT_CANCER_DATA_REPOSITORY
      WHERE DATE_OF_DIAGNOSIS IS NOT NULL)
    WHERE rn = 1
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"registry scan: {job.total_bytes_processed/1e9:.1f} GB")
    S = C.query(sql).to_dataframe()
    S.to_parquet(sc)

R = R.merge(S, on="PATIENT_DK", how="left")
R["dx_yr"] = R["dx"].dt.year
R["lead_yr"] = (R["dx"] - R["first_note"]).dt.days / 365.25

D21 = R[R["dx_yr"] >= CUT].copy()
n = len(D21)
print("=" * 68)
print(f"WHO IS DIAGNOSED (first cancer) IN {CUT}+ ?   {n:,} patients")
print("=" * 68)

print("\n1. HOW LONG were they a Mayo patient BEFORE that diagnosis?")
print("   (first note -> first cancer diagnosis)")
lead = D21["lead_yr"]
buckets = [(-1, 0.25, "new arrival (<3 mo before dx)"),
           (0.25, 1, "3-12 months before"),
           (1, 3, "1-3 years before"),
           (3, 7, "3-7 years before"),
           (7, 100, "7+ years before")]
for lo, hi, name in buckets:
    m = ((lead >= lo) & (lead < hi)).sum()
    print(f"   {name:<32}{m:>9,}  ({m/n:.0%})")
nap = lead.isna().sum()
if nap:
    print(f"   {'(no note date)':<32}{nap:>9,}  ({nap/n:.0%})")
est = (lead >= 1).sum()
print(f"\n   established >=1yr before dx: {est:,} ({est/n:.0%})   "
      f"median lead {lead[lead>=0].median():.1f} yrs")
print("   => most were ALREADY Mayo patients (primary care, other conditions,")
print("      screening) who then developed their first cancer -- not new arrivals.")

print(f"\n2. WHAT cancers are newly diagnosed {CUT}+ ?")
D21["type"] = D21["site3"].map(lambda s: TYPE.get(s, f"other({s})"))
vc = D21["type"].value_counts()
for t, c in vc.head(14).items():
    print(f"   {t:<16}{c:>8,}  ({c/n:.0%})")
oth = vc[14:].sum()
if oth:
    print(f"   {'... rest':<16}{oth:>8,}  ({oth/n:.0%})")

print(f"""
{'=' * 68}
THE ANSWER
{'=' * 68}
  The {CUT}+ newly-diagnosed are mostly people Mayo was ALREADY caring for:
  {est/n:.0%} had been patients >=1 year before their cancer diagnosis (median
  {lead[lead>=0].median():.1f} yrs). That fits a big referral/tertiary centre -- primary
  care, chronic disease, and screening populations that convert to a cancer
  diagnosis over time -- plus a minority ({(lead<0.25).sum()/n:.0%}) who show up new for it.

  For the product this is the GOOD population: their pre-cancer history is
  already in the record, so the case packet at diagnosis is rich. For a
  "first seen 2021+" filter you lose them, which is why that cut was so
  much harsher ({(R['first_note'].dt.year>=CUT).sum()/len(R):.0%} vs {n/len(R):.0%}).""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"who_dx | dx{CUT}+={n} | established>=1yr={est}({est/n:.0%}) "
      f"| new<3mo={(lead<0.25).sum()}({(lead<0.25).sum()/n:.0%}) "
      f"| top_type={vc.index[0]}({vc.iloc[0]/n:.0%})")

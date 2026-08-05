"""
panel_overlap.py -- the last place usable molecular data could be.

The registry is a dead end for this: MSI recorded for 68 of 6,057 immunotherapy
patients (1.1%), KRAS 0.7%, BRAF 0.4%. Molecular markers added exactly nothing
(0.663 -> 0.663) and the comparison was underpowered anyway.

But FACT_ONBASE_DOCUMENTS_GENOMICS -- OCR'd sequencing panel reports -- had
microsatellite status for 1,882 patients and TMB for 426, i.e. 28x what the
registry holds for our cohort. The molecular data exists; it is in the panel
reports, not the registry.

Question: do those 2,031 patients overlap the cohorts we have already built?
If yes, that is the usable molecular source. If no, genomics is closed here.

Also pulls the actual VALUES for MSI and TMB so we can see whether the OCR
output is clean enough to model on.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 70)

G = "FACT_ONBASE_DOCUMENTS_GENOMICS"
tot = C.query(f"""SELECT COUNT(*) AS rows_,
   COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS pts
   FROM {D}.{G}""").to_dataframe().iloc[0]
print(f"panel reports: {int(tot['rows_']):,} rows over {int(tot['pts']):,} patients")

print("\n" + "=" * 80)
print("OVERLAP WITH THE COHORTS WE HAVE BUILT")
print("=" * 80)
COH = [("psa_progression.parquet", "prostate on hormone therapy"),
       ("ici_thyroid_label.parquet", "immunotherapy"),
       ("ov_label.parquet", "ovarian on platinum")]
for f, lbl in COH:
    try:
        idx = pd.read_parquet(f).index.astype(str)
        ids = "','".join(idx[:25000])
        r = C.query(f"""
          SELECT COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS n,
                 COUNT(DISTINCT CASE WHEN LOWER(CAST(KEY AS STRING))
                       LIKE '%microsatellite%'
                       THEN CAST(PATIENT_CLINIC_NUMBER AS STRING) END) AS msi,
                 COUNT(DISTINCT CASE WHEN LOWER(CAST(KEY AS STRING))
                       LIKE '%mutational burden%'
                       THEN CAST(PATIENT_CLINIC_NUMBER AS STRING) END) AS tmb
          FROM {D}.{G}
          WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')""").to_dataframe().iloc[0]
        print(f"  {lbl:<30}{int(r['n']):>6,} of {len(idx):,} have a panel "
              f"({int(r['n'])/len(idx):5.1%})   MSI {int(r['msi']):>4,}  "
              f"TMB {int(r['tmb']):>4,}")
    except Exception as e:
        print(f"  {lbl:<30}skipped ({type(e).__name__})")

print("\n" + "=" * 80)
print("ARE THE OCR'D VALUES CLEAN ENOUGH TO MODEL ON?")
print("=" * 80)
for key, lbl in [("microsatellite", "MICROSATELLITE STATUS"),
                 ("mutational burden", "TUMOUR MUTATIONAL BURDEN"),
                 ("tumor type", "TUMOUR TYPE")]:
    v = C.query(f"""
      SELECT CAST(VALUE AS STRING) AS val, COUNT(*) AS n,
             COUNT(DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING)) AS pts,
             ROUND(AVG(SAFE_CAST(VALUE_CONFIDENCE AS FLOAT64)), 3) AS conf
      FROM {D}.{G}
      WHERE LOWER(CAST(KEY AS STRING)) LIKE '%{key}%'
      GROUP BY 1 ORDER BY n DESC LIMIT 12""").to_dataframe()
    print(f"\n  {lbl}:")
    print(v.to_string(index=False) if len(v) else "    none")

print("\n  which cancers do the panelled patients have?")
sites = C.query(f"""
  SELECT SUBSTR(CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING),1,3) AS site3,
         COUNT(DISTINCT CAST(p.PATIENT_CLINIC_NUMBER AS STRING)) AS pts
  FROM {D}.{G} g
  JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
                          = CAST(g.PATIENT_CLINIC_NUMBER AS STRING)
  JOIN {D}.FACT_CANCER_DATA_REPOSITORY r ON r.PATIENT_DK = p.PATIENT_DK
  WHERE CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING) LIKE 'C%'
  GROUP BY 1 ORDER BY pts DESC LIMIT 12""").to_dataframe()
SITE = {"C34": "lung", "C44": "skin/melanoma", "C50": "breast", "C18": "colon",
        "C64": "kidney", "C25": "pancreas", "C61": "prostate", "C56": "ovary",
        "C71": "brain", "C22": "liver", "C20": "rectum", "C67": "bladder"}
for _, r in sites.iterrows():
    print(f"    {r['site3']}  {SITE.get(r['site3'],''):<16}{int(r['pts']):>6,}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"panel_overlap | panel_patients={int(tot['pts'])} "
      f"| top_site={sites['site3'].iloc[0] if len(sites) else 'NA'}")

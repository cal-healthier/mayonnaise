"""
explore.py -- tools for poking at the Mayo data yourself.

Run this once, then call the helpers from any cell.

  tables("radiol")            find tables whose name matches
  cols("FACT_RADIOLOGY")      list columns, types, and how full each one is
  peek("DIM_MED_NAME", 5)     sample rows
  values("FACT_CANCER_DATA_REPOSITORY", "DERIVED_SUMMARY_STAGE_2018")
                              value distribution for one column
  find(site="C61", n=5)       find some patients to look at
  patient("<clinic number>")  one person's whole record, chronologically
  q("SELECT ...")             run any SQL, returns a DataFrame

Everything is read-only. Queries are billed by bytes SCANNED, so filter early
and avoid SELECT * on the billion-row tables.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 100)

def q(sql, dry=False):
    """run SQL. dry=True estimates cost without running it."""
    if dry:
        job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
        gb = job.total_bytes_processed / 1e9
        print(f"  would scan {gb:.2f} GB  (~${gb*0.005:.2f})")
        return None
    return C.query(sql).to_dataframe()

def tables(pat=""):
    """find tables by name fragment"""
    df = q(f"""SELECT table_name FROM {S}.TABLES
              WHERE LOWER(table_name) LIKE '%{pat.lower()}%'
              ORDER BY table_name""")
    print(f"{len(df)} tables matching '{pat}'")
    for i in range(0, len(df), 3):
        print("  " + "".join(f"{t:<46}" for t in df["table_name"][i:i+3]))
    return df

def cols(table, fill=True):
    """columns, types, and (optionally) how populated each is"""
    df = q(f"""SELECT column_name, data_type FROM {S}.COLUMNS
              WHERE table_name='{table}' ORDER BY ordinal_position""")
    print(f"{table}: {len(df)} columns")
    if fill and len(df) <= 60:
        sel = ", ".join(f"COUNTIF({c} IS NOT NULL) AS c{i}"
                        for i, c in enumerate(df["column_name"]))
        r = q(f"SELECT COUNT(*) AS n, {sel} FROM {D}.{table}").iloc[0]
        n = int(r["n"])
        df["filled"] = [int(r[f"c{i}"]) / max(n, 1) for i in range(len(df))]
        print(f"  {n:,} rows\n")
        for _, x in df.iterrows():
            bar = "#" * int(x["filled"] * 22)
            print(f"  {x['column_name']:<42}{x['data_type']:<12}"
                  f"{x['filled']:>6.0%} {bar}")
    else:
        for _, x in df.iterrows():
            print(f"  {x['column_name']:<44}{x['data_type']}")
    return df

def peek(table, n=5, where=""):
    """sample rows"""
    w = f"WHERE {where}" if where else ""
    return q(f"SELECT * FROM {D}.{table} {w} LIMIT {n}")

def values(table, col, n=15):
    """what values does this column actually hold? watch for coded blanks"""
    df = q(f"""SELECT CAST({col} AS STRING) AS value, COUNT(*) AS n
              FROM {D}.{table} GROUP BY 1 ORDER BY n DESC LIMIT {n}""")
    tot = df["n"].sum()
    print(f"{table}.{col}  (top {n})")
    for _, r in df.iterrows():
        print(f"  {str(r['value'])[:44]:<46}{int(r['n']):>10,}  "
              f"({r['n']/tot:5.1%})  {'#'*int(r['n']/tot*20)}")
    print("\n  NB registries use CODED blanks -- 9, 99, 999, 88, 'Unknown' are")
    print("  'not recorded', not findings. They will inflate any fill rate.")
    return df

def find(site=None, histology=None, n=5):
    """find some patients to look at. site = ICD-O-3 topography e.g. C61 prostate"""
    conds = ["DATE_OF_DIAGNOSIS IS NOT NULL"]
    if site:
        conds.append(f"CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE '{site}%'")
    if histology:
        conds.append(f"CAST(HISTOLOGIC_TYPE_ICD_O_3 AS STRING) LIKE '{histology}%'")
    return q(f"""
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             MIN(DATE(r.DATE_OF_DIAGNOSIS)) AS dx,
             ANY_VALUE(CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING)) AS site,
             ANY_VALUE(CAST(r.HISTOLOGIC_TYPE_ICD_O_3 AS STRING)) AS histology,
             ANY_VALUE(CAST(r.VITAL_STATUS AS STRING)) AS vital
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r
      JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      WHERE {' AND '.join(conds)}
      GROUP BY 1 LIMIT {n}""")

def patient(clinic, labs=8, notes=10):
    """everything about one person, in order. THE way to build intuition."""
    c = str(clinic)
    print("=" * 78); print(f"PATIENT {c}"); print("=" * 78)

    reg = q(f"""SELECT MIN(DATE(r.DATE_OF_DIAGNOSIS)) AS dx,
       ANY_VALUE(CAST(r.SITE_PRIMARY_ICD_O_3 AS STRING)) AS site,
       ANY_VALUE(CAST(r.HISTOLOGIC_TYPE_ICD_O_3 AS STRING)) AS histology,
       ANY_VALUE(CAST(r.DERIVED_SUMMARY_STAGE_2018 AS STRING)) AS stage,
       ANY_VALUE(CAST(r.VITAL_STATUS AS STRING)) AS vital,
       MAX(DATE(r.DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_contact
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING)='{c}'""")
    print("\nREGISTRY"); print(reg.to_string(index=False))

    tx = q(f"""SELECT DATE(t.TREATMENT_DTM) AS date,
       UPPER(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING)) AS drug
      FROM {D}.DIM_PATIENT p JOIN {D}.FACT_TREATMENT_DETAIL t USING (PATIENT_DK)
      JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING)='{c}'
      GROUP BY 1,2 ORDER BY 1 LIMIT 40""")
    print(f"\nONCOLOGY TREATMENT  ({len(tx)} rows)")
    print(tx.to_string(index=False) if len(tx) else "  none")

    med = q(f"""SELECT UPPER(CAST(o.MED_GENERIC AS STRING)) AS drug,
       COUNT(*) AS orders, MIN(DATE(o.ORDER_APPROVE_DTM)) AS first,
       MAX(DATE(o.ORDER_APPROVE_DTM)) AS last
      FROM {D}.DIM_PATIENT p JOIN {D}.FACT_ORDERS o USING (PATIENT_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING)='{c}' AND o.MED_GENERIC IS NOT NULL
      GROUP BY 1 ORDER BY orders DESC LIMIT 12""")
    print(f"\nALL MEDICATIONS (top 12 of {len(med)})")
    print(med.to_string(index=False) if len(med) else "  none")

    lab = q(f"""SELECT cc.concept_name AS test, COUNT(*) AS n,
       MIN(DATE(m.measurement_date)) AS first, MAX(DATE(m.measurement_date)) AS last,
       ROUND(APPROX_QUANTILES(m.value_as_number,100)[OFFSET(50)],2) AS median
      FROM {D}.person pe JOIN {D}.measurement m ON m.person_id=pe.person_id
      LEFT JOIN {D}.concept cc ON cc.concept_id=m.measurement_concept_id
      WHERE CAST(pe.person_source_value AS STRING)='{c}' AND m.value_as_number IS NOT NULL
      GROUP BY 1 ORDER BY n DESC LIMIT {labs}""")
    print(f"\nMOST-MEASURED LABS")
    print(lab.to_string(index=False) if len(lab) else "  none")

    nt = q(f"""SELECT DATE(n.note_date) AS date, CAST(n.note_title AS STRING) AS title,
       LENGTH(CAST(n.note_text AS STRING)) AS chars
      FROM {D}.person pe JOIN {D}.note n ON n.person_id=pe.person_id
      WHERE CAST(pe.person_source_value AS STRING)='{c}'
      ORDER BY n.note_date DESC LIMIT {notes}""")
    print(f"\nRECENT NOTES (titles only -- use note_text to read one)")
    print(nt.to_string(index=False) if len(nt) else "  none")

    rad = q(f"""SELECT DATE(r.RADIOLOGY_DTM) AS date,
       CAST(r.SERVICE_MODALITY_CODE AS STRING) AS modality, COUNT(*) AS n
      FROM {D}.DIM_PATIENT p JOIN {D}.FACT_RADIOLOGY r USING (PATIENT_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING)='{c}'
      GROUP BY 1,2 ORDER BY 1 DESC LIMIT 10""")
    print(f"\nRECENT IMAGING")
    print(rad.to_string(index=False) if len(rad) else "  none")
    print("\n" + "=" * 78)

print(__doc__)
print("=" * 78)
print("THE JOIN PATH -- memorise this, nothing works without it")
print("=" * 78)
print("""
  FACT_* / DIM_*  (Mayo native)          keyed by PATIENT_DK
        |
        |  DIM_PATIENT.PATIENT_CLINIC_NUMBER   <- the human-readable spine
        |
  person.person_source_value  ->  person_id     (OMOP)
        |
  measurement / note / drug_exposure / observation

  registry, oncology treatment, radiology, pathology  = NATIVE side only
  23.6B lab values, 559M notes                        = OMOP side only
  you almost always need both.
""")
print("try:  find(site='C61')   then   patient('<clinic from that list>')")

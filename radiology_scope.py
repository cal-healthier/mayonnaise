"""
radiology_scope.py -- STUDY 3, step 1: is the radiology text usable?

Goal: turn radiologists' written impressions into a response measure for the
cancers that have no blood marker (lung, melanoma, kidney, bladder).

This step only asks whether the raw material is good enough:
  1. how much radiology text exists, for whom, over what period
  2. how many impressions fall DURING a treatment course in our cohorts
  3. is the language informative? -- measured by how often response vocabulary
     ("interval increase", "new lesion", "no evidence of disease") appears,
     and how templated the text is
  4. how many prostate/ovarian patients have BOTH impressions and marker-based
     progression dates -- that overlap is the validation set, and it is what
     would make this a study rather than a demo

PRIVACY: prints aggregate statistics only -- counts, lengths, keyword rates.
No impression text is printed, so nothing patient-level leaves the enclave.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)

# ---------------------------------------------------------------- 1. what exists
rad = C.query(f"""SELECT table_name FROM {S}.TABLES
  WHERE REGEXP_CONTAINS(LOWER(table_name), r'radiol|imaging|report|note')
  ORDER BY table_name""").to_dataframe()
print("radiology / report / note tables:")
print("  " + ", ".join(rad["table_name"]) if len(rad) else "  none")

TAB = "FACT_RADIOLOGY"
cols = C.query(f"""SELECT column_name, data_type FROM {S}.COLUMNS
  WHERE table_name='{TAB}' ORDER BY ordinal_position""").to_dataframe()
print(f"\n{TAB} columns:")
print("  " + ", ".join(cols["column_name"]))

up = {c.upper(): c for c in cols["column_name"]}
IMP = next((up[k] for k in up if "IMPRESSION" in k), None)
IDC = next((up[k] for k in ("PATIENT_DK", "PATIENT_CLINIC_NUMBER") if k in up), None)
DTC = next((c for c in cols["column_name"] if "DTM" in c.upper()
            or "DATE" in c.upper()), None)
TYPE = next((up[k] for k in up if "EXAM" in k or "MODALITY" in k or "PROCEDURE" in k), None)
print(f"\n  impression -> {IMP}\n  patient id -> {IDC}\n  date       -> {DTC}"
      f"\n  exam type  -> {TYPE}")
if not (IMP and IDC and DTC):
    raise SystemExit("missing a required column -- tell Claude")

t = C.query(f"""SELECT COUNT(*) rows_,
  COUNT(DISTINCT CAST({IDC} AS STRING)) pts,
  COUNTIF({IMP} IS NOT NULL AND LENGTH(CAST({IMP} AS STRING)) > 20) with_text,
  MIN(EXTRACT(YEAR FROM DATE({DTC}))) y0,
  MAX(EXTRACT(YEAR FROM DATE({DTC}))) y1,
  APPROX_QUANTILES(LENGTH(CAST({IMP} AS STRING)), 100)[OFFSET(50)] med_len,
  APPROX_QUANTILES(LENGTH(CAST({IMP} AS STRING)), 100)[OFFSET(90)] p90_len
  FROM {D}.{TAB}""").to_dataframe().iloc[0]
print("\n" + "=" * 84)
print("1. HOW MUCH IS THERE?")
print("=" * 84)
print(f"  reports              {int(t['rows_']):>14,}")
print(f"  patients             {int(t['pts']):>14,}")
print(f"  with usable text     {int(t['with_text']):>14,}  "
      f"({t['with_text']/max(t['rows_'],1):.0%})")
print(f"  years                {int(t['y0'])}-{int(t['y1'])}")
print(f"  impression length    median {int(t['med_len'])} chars, p90 {int(t['p90_len'])}")

# ---------------------------------------------------------------- 2. is it informative?
print("\n" + "=" * 84)
print("2. DOES THE TEXT CONTAIN RESPONSE LANGUAGE?  (aggregate rates only)")
print("=" * 84)
PHRASES = {
 "interval increase / enlarging": ["INTERVAL INCREASE", "ENLARG", "INCREASED IN SIZE"],
 "interval decrease / smaller":   ["INTERVAL DECREASE", "DECREASED IN SIZE", "SMALLER"],
 "new lesion / new metastasis":   ["NEW LESION", "NEW METASTA", "NEW NODULE", "NEW FOCUS"],
 "stable / no change":            ["STABLE", "NO SIGNIFICANT CHANGE", "UNCHANGED"],
 "no evidence of disease":        ["NO EVIDENCE OF DISEASE", "NO EVIDENCE OF RECURRENT",
                                   "NO EVIDENCE OF METASTA"],
 "explicit RECIST wording":       ["RECIST", "PARTIAL RESPONSE", "COMPLETE RESPONSE",
                                   "PROGRESSIVE DISEASE", "STABLE DISEASE"],
 "compares to a prior study":     ["COMPARED TO", "COMPARISON", "PRIOR STUDY",
                                   "SINCE THE PREVIOUS"],
}
U = f"UPPER(CAST({IMP} AS STRING))"
sel = ", ".join(
    "COUNTIF(" + " OR ".join(f"{U} LIKE '%{p}%'" for p in ps) + f") AS `{k}`"
    for k, ps in PHRASES.items())
r = C.query(f"""SELECT COUNT(*) n, {sel} FROM {D}.{TAB}
  WHERE {IMP} IS NOT NULL AND LENGTH(CAST({IMP} AS STRING)) > 20""").to_dataframe().iloc[0]
n = int(r["n"])
print(f"  of {n:,} impressions with text:")
for k in PHRASES:
    v = int(r[k])
    print(f"    {k:<32}{v:>12,}  ({v/n:5.1%})  {'#'*int(v/n*30)}")

dup = C.query(f"""SELECT COUNT(DISTINCT CAST({IMP} AS STRING)) d, COUNT(*) n
  FROM {D}.{TAB} WHERE {IMP} IS NOT NULL
    AND LENGTH(CAST({IMP} AS STRING)) > 20""").to_dataframe().iloc[0]
print(f"\n  distinct impressions: {int(dup['d']):,} of {int(dup['n']):,} "
      f"({int(dup['d'])/int(dup['n']):.0%} unique)")
print("  low uniqueness = templated boilerplate; high = real dictated prose.")

# ---------------------------------------------------------------- 3. the validation set
print("\n" + "=" * 84)
print("3. THE VALIDATION SET -- patients with BOTH impressions and a marker date")
print("=" * 84)
for f, lbl in [("psa_progression.parquet", "prostate (PSA progression date)"),
               ("ov_label.parquet", "ovarian (CA-125 progression date)")]:
    try:
        idx = pd.read_parquet(f)
        idx = idx[idx["prog"] == 1].index.astype(str)
        ids = "','".join(idx[:20000])
        if IDC.upper() == "PATIENT_CLINIC_NUMBER":
            where = f"CAST({IDC} AS STRING) IN ('{ids}')"
        else:
            where = f"""CAST({IDC} AS STRING) IN (
              SELECT CAST(PATIENT_DK AS STRING) FROM {D}.DIM_PATIENT
              WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))"""
        q = C.query(f"""SELECT COUNT(DISTINCT CAST({IDC} AS STRING)) pts,
          COUNT(*) reports FROM {D}.{TAB}
          WHERE {where} AND {IMP} IS NOT NULL
            AND LENGTH(CAST({IMP} AS STRING)) > 20""").to_dataframe().iloc[0]
        print(f"  {lbl:<38}{int(q['pts']):>7,} patients, "
              f"{int(q['reports']):>9,} reports")
    except Exception as e:
        print(f"  {lbl:<38}skipped ({type(e).__name__})")
print("\n  these are men and women where an INDEPENDENT objective progression date")
print("  already exists. Agreement between the extracted date and the marker date")
print("  is the validation almost no note-extraction work ever does.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"radiology_scope | reports={int(t['rows_'])} | patients={int(t['pts'])} "
      f"| with_text={int(t['with_text'])} | unique={int(dup['d'])/int(dup['n']):.0%} "
      f"| years={int(t['y0'])}-{int(t['y1'])}")

"""
genomics_sweep.py -- what molecular / genetic data is actually here?

Working assumption has been "thin": the MCP Omics tables were in the data
dictionary but not in our 189, and only 1% of trial eligibility criteria needed
genomics. But every previous assumption about this dataset has been too
pessimistic, so: look properly.

Four places molecular data could hide:
  1. dedicated tables (genomic / sequencing / variant / specimen / omics)
  2. the CANCER REGISTRY -- NAACCR Site-Specific Data Items carry MSI, EGFR,
     ALK, BRAF, KRAS, PD-L1, HER2, Oncotype across many cancers. We only ever
     looked at Gleason and grade.
  3. OMOP `measurement` -- molecular assays recorded as lab results
  4. FACT_ONBASE_DOCUMENTS_GENOMICS -- 106k key/value rows, never inspected

Mostly INFORMATION_SCHEMA (free) plus a few cheap population counts.
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------- 1. tables
print("=" * 92)
print("1. TABLES THAT LOOK MOLECULAR")
print("=" * 92)
t = C.query(f"""
  SELECT table_name FROM {S}.TABLES
  WHERE REGEXP_CONTAINS(LOWER(table_name),
    r'genom|genet|sequen|variant|mutat|molecul|omic|specimen|biomark|pathol|cyto')
  ORDER BY table_name
""").to_dataframe()
print(f"  {len(t)} found: {', '.join(t['table_name']) if len(t) else 'NONE'}")
alltabs = C.query(f"SELECT COUNT(*) AS n FROM {S}.TABLES").to_dataframe().iloc[0]["n"]
print(f"  (out of {alltabs} tables total)")

# ---------------------------------------------------------------- 2. columns anywhere
print("\n" + "=" * 92)
print("2. COLUMNS NAMED AFTER MOLECULAR MARKERS, ANYWHERE IN THE DATASET")
print("=" * 92)
cols = C.query(f"""
  SELECT table_name, column_name FROM {S}.COLUMNS
  WHERE REGEXP_CONTAINS(UPPER(column_name),
    r'EGFR|\\bALK\\b|ROS1|BRAF|KRAS|NRAS|HER2|ERBB2|PD_?L1|MSI|MMR|TMB|BRCA|'
    r'PIK3CA|IDH|MGMT|\\bEBV\\b|\\bHPV\\b|ONCOTYPE|MAMMAPRINT|RECURRENCE_SCORE|'
    r'ESTROGEN|PROGESTERONE|GENE|MUTAT|VARIANT|SEQUENC|GENOM|MOLECULAR|BIOMARK|'
    r'IMMUNOHISTO|\\bIHC\\b|FISH|\\bNGS\\b|PLOIDY|KI_?67|ALLRED')
  ORDER BY table_name, column_name
""").to_dataframe()
print(f"  {len(cols)} columns across {cols['table_name'].nunique()} tables\n")
for tab, g in cols.groupby("table_name"):
    print(f"  {tab}  ({len(g)})")
    for c in g["column_name"]:
        print(f"      {c}")

# ---------------------------------------------------------------- 3. registry fill rates
reg_cols = cols[cols["table_name"] == "FACT_CANCER_DATA_REPOSITORY"]["column_name"].tolist()
if reg_cols:
    print("\n" + "=" * 92)
    print("3. HOW MANY PATIENTS ACTUALLY HAVE THESE REGISTRY MARKERS?")
    print("=" * 92)
    sel = ", ".join(
        f"COUNTIF({c} IS NOT NULL AND CAST({c} AS STRING) NOT IN "
        f"('','9','99','999','88','888','Unknown','UNKNOWN')) AS `{c}`"
        for c in reg_cols[:45])
    fill = C.query(f"SELECT COUNT(*) AS total, {sel} FROM {D}.FACT_CANCER_DATA_REPOSITORY"
                   ).to_dataframe().iloc[0]
    tot = int(fill["total"])
    print(f"  registry rows: {tot:,}\n")
    r = fill.drop("total").sort_values(ascending=False)
    for k, v in r.items():
        if int(v) > 0:
            print(f"    {k:<46}{int(v):>9,}  ({int(v)/tot:5.1%})")
    empty = [k for k, v in r.items() if int(v) == 0]
    if empty:
        print(f"\n    completely empty ({len(empty)}): {', '.join(empty[:14])}"
              f"{' ...' if len(empty) > 14 else ''}")

# ---------------------------------------------------------------- 4. molecular in labs
print("\n" + "=" * 92)
print("4. MOLECULAR ASSAYS RECORDED AS LAB RESULTS (OMOP measurement)")
print("=" * 92)
cc = pd.read_parquet("measurement_concepts.parquet")
cc["nl"] = cc["name"].astype(str).str.lower()
pat = (r"egfr|\balk\b|ros1|braf|kras|nras|her2|erbb2|pd-?l1|microsatellite|"
       r"mismatch repair|tumor mutation|brca|pik3ca|\bidh\b|mgmt|oncotype|"
       r"mutation|variant|sequencing|genotype|fusion|amplification|ki-?67|"
       r"estrogen receptor|progesterone receptor|immunohisto")
hits = cc[cc["nl"].str.contains(pat, na=False)].sort_values("n_persons", ascending=False)
print(f"  {len(hits)} molecular-looking concepts; top 30 by patient count:\n")
if len(hits):
    print(hits[["cid", "name", "code", "vocab", "n_persons", "n_rows"]].head(30)
          .to_string(index=False))

# ---------------------------------------------------------------- 5. the genomics doc table
print("\n" + "=" * 92)
print("5. FACT_ONBASE_DOCUMENTS_GENOMICS -- what is in it?")
print("=" * 92)
try:
    gc = [r.column_name for r in C.query(
        f"SELECT column_name FROM {S}.COLUMNS "
        f"WHERE table_name='FACT_ONBASE_DOCUMENTS_GENOMICS' ORDER BY ordinal_position")]
    print(f"  columns: {gc}")
    keycol = next((c for c in gc if "KEY" in c.upper() or "NAME" in c.upper()), None)
    n = C.query(f"SELECT COUNT(*) n, COUNT(DISTINCT PATIENT_DK) p "
                f"FROM {D}.FACT_ONBASE_DOCUMENTS_GENOMICS").to_dataframe().iloc[0]
    print(f"  {int(n['n']):,} rows over {int(n['p']):,} patients")
    if keycol:
        top = C.query(f"""SELECT CAST({keycol} AS STRING) AS k, COUNT(*) c,
                          COUNT(DISTINCT PATIENT_DK) p
                          FROM {D}.FACT_ONBASE_DOCUMENTS_GENOMICS
                          GROUP BY 1 ORDER BY c DESC LIMIT 30""").to_dataframe()
        print(f"\n  most common `{keycol}` values:")
        print(top.to_string(index=False))
except Exception as e:
    print(f"  not readable: {type(e).__name__} {str(e)[:120]}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"genomics_sweep | mol_tables={len(t)} | mol_columns={len(cols)} "
       f"| tables_with_cols={cols['table_name'].nunique()} "
       f"| registry_cols={len(reg_cols)} | measurement_concepts={len(hits)}")

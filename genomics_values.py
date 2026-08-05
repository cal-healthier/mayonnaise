"""
genomics_values.py -- are the registry molecular columns real, or full of
"test not done" codes?

Every one of the 36 columns reported 100% non-null, which cannot be true --
almost no cancer patient has BRAF tested. NAACCR site-specific items use CODED
blanks (000 not done, 988 not applicable, 997/998/999 unknown), so a crude
NULL check counts them as data.

Fix: look at the actual VALUE DISTRIBUTION of every column. If one value covers
90%+ of rows, that column is effectively empty and the "100%" was a filler code.

Also checks two things the sweep turned up but did not open:
  - concept 42529070, "Genetic variant details ... Narrative", 73,348 patients
  - DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE (digital pathology slides?)
"""
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA"
pd.set_option("display.width", 240)

cols = [r.column_name for r in C.query(f"""
  SELECT column_name FROM {S}.COLUMNS
  WHERE table_name='FACT_CANCER_DATA_REPOSITORY'
    AND REGEXP_CONTAINS(UPPER(column_name),
      r'EGFR|\\bALK\\b|BRAF|KRAS|NRAS|HER2|ERBB2|PD_?L1|MSI|MICROSAT|MMR|TMB|BRCA|'
      r'ONCOTYPE|MULTIGENE|ESTROGEN|PROGESTERONE|MOLECULAR|CYTOGEN|KI_?67|'
      r'METHYLATION|KIT_GENE|ALLRED')
  ORDER BY column_name""")]
print(f"{len(cols)} molecular registry columns\n")

# one query, all columns, top values each
union = "\nUNION ALL\n".join(
    f"SELECT '{c}' AS col, IFNULL(CAST({c} AS STRING),'<NULL>') AS val, COUNT(*) AS n "
    f"FROM {D}.FACT_CANCER_DATA_REPOSITORY GROUP BY 1,2" for c in cols)
V = C.query(union).to_dataframe()
tot = int(V[V["col"] == cols[0]]["n"].sum())

print("=" * 96)
print(f"VALUE DISTRIBUTIONS  (registry rows = {tot:,})")
print("=" * 96)
summary = []
for c in cols:
    g = V[V["col"] == c].sort_values("n", ascending=False)
    top_share = g["n"].iloc[0] / tot
    informative = 1 - top_share
    summary.append({"col": c, "top_val": g["val"].iloc[0], "top_share": top_share,
                    "informative_rows": int(tot - g["n"].iloc[0]),
                    "n_distinct": len(g)})
    flag = "EMPTY" if top_share > 0.95 else ("thin" if top_share > 0.90 else "USABLE")
    print(f"\n  {c}   [{flag}]  {len(g)} distinct values")
    for _, r in g.head(6).iterrows():
        bar = "#" * max(1, int(r["n"] / tot * 40))
        print(f"      {str(r['val'])[:28]:<30}{int(r['n']):>9,} ({r['n']/tot:5.1%}) {bar}")

Sm = pd.DataFrame(summary).sort_values("informative_rows", ascending=False)
print("\n" + "=" * 96)
print("RANKED BY HOW MANY PATIENTS ACTUALLY HAVE A RESULT")
print("=" * 96)
print(f"  {'column':<40}{'rows w/ a result':>18}{'%':>8}   dominant filler value")
for _, r in Sm.iterrows():
    print(f"  {r['col']:<40}{r['informative_rows']:>18,}{1-r['top_share']:>8.1%}   "
          f"{str(r['top_val'])[:22]}")
usable = Sm[Sm["top_share"] <= 0.95]
print(f"\n  {len(usable)} of {len(cols)} columns have >5% of patients with a real result")

# ---------------------------------------------------------------- narrative reports
print("\n" + "=" * 96)
print("THE 73,348-PATIENT 'GENETIC VARIANT DETAILS - NARRATIVE' CONCEPT")
print("=" * 96)
mcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='measurement'")]
txt = [c for c in mcols if "value" in c.lower() or "source" in c.lower()]
print(f"  measurement value/source columns: {txt}")
sel = ", ".join(f"COUNTIF({c} IS NOT NULL) AS `{c}`" for c in txt)
r = C.query(f"""SELECT COUNT(*) AS rows_, COUNT(DISTINCT person_id) AS people, {sel}
  FROM {D}.measurement WHERE measurement_concept_id = 42529070""").to_dataframe().iloc[0]
print(f"  {int(r['rows_']):,} rows over {int(r['people']):,} patients")
for c in txt:
    print(f"    {c:<28}{int(r[c]):>10,} populated")

# ---------------------------------------------------------------- pathology slides
print("\n" + "=" * 96)
print("DIGITAL PATHOLOGY SLIDES?")
print("=" * 96)
for tab in ("DIM_PATHOLOGY_SLIDE_ID_DICOM_BRIDGE", "FACT_PATHOLOGY",
            "FACT_PATHOLOGY_SPECIMEN_INFORMATION"):
    try:
        cn = [r.column_name for r in C.query(
            f"SELECT column_name FROM {S}.COLUMNS WHERE table_name='{tab}' "
            f"ORDER BY ordinal_position LIMIT 14")]
        n = list(C.query(f"SELECT COUNT(*) AS n FROM {D}.{tab}"))[0].n
        print(f"  {tab}: {int(n):,} rows")
        print(f"      {cn}")
    except Exception as e:
        print(f"  {tab}: {type(e).__name__} {str(e)[:70]}")

print("\n" + "-" * 74)
print("FINAL LINE:")
best = Sm.iloc[0]
print(f"genomics_values | cols={len(cols)} | usable={len(usable)} "
      f"| best={best['col']}({best['informative_rows']}) "
      f"| narrative_pts={int(r['people'])}")

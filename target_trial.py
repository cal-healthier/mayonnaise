"""
target_trial.py -- can we emulate a trial whose answer we already know?

Pivot rationale: every failure today was a label or confounding problem, not
model capacity. Prediction also does not change care -- "he will progress in 8
months" tells no one what to do. Effect estimation does.

The natural target trial is ADT INTENSIFICATION in prostate cancer, because the
answer is published:
    LATITUDE / STAMPEDE  ADT + abiraterone   OS HR ~0.62-0.66
    ENZAMET / ARCHES     ADT + enzalutamide  OS HR ~0.67
    CHAARTED / STAMPEDE  ADT + docetaxel     OS HR ~0.61
If an emulation on Mayo data recovers ~0.62, the METHOD is validated against
ground truth. Then the same machinery answers what trials cannot: does
intensification help patients those trials EXCLUDED (older, comorbid, poor
organ function) -- the majority of real patients.

This sizes it. Counts only, no estimation:
  - how many men got each intensifier, and when
  - the treated/untreated split within the era when intensification was standard
  - how many would have FAILED trial eligibility (the extension cohort)
  - immortal-time check: gap between ADT start and intensifier start
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS "
    f"WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")
def like(ds):
    return " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in ds)

INTENS = {
  "abiraterone (LATITUDE/STAMPEDE, HR~0.62)": ["ABIRATERONE", "ZYTIGA", "YONSA"],
  "enzalutamide (ENZAMET/ARCHES, HR~0.67)":   ["ENZALUTAMIDE", "XTANDI"],
  "docetaxel (CHAARTED/STAMPEDE, HR~0.61)":   ["DOCETAXEL", "TAXOTERE"],
  "apalutamide (TITAN, HR~0.67)":             ["APALUTAMIDE", "ERLEADA"],
  "darolutamide (ARASENS, HR~0.68)":          ["DAROLUTAMIDE", "NUBEQA"],
}

P = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
P["adt_date"] = pd.to_datetime(tx.reindex(P.index))
P["year"] = P["adt_date"].dt.year
print(f"prostate ADT cohort: {len(P):,} men, ADT start {int(P['year'].min())}"
      f"-{int(P['year'].max())}")

ids = "','".join(P.index.astype(str))
print("\n" + "=" * 88)
print("WHO GOT AN INTENSIFIER, AND WHEN RELATIVE TO ADT START?")
print("=" * 88)
got = {}
for lbl, drugs in INTENS.items():
    df = C.query(f"""
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             MIN(DATE(t.TREATMENT_DTM)) AS first_dt
      FROM {D}.DIM_PATIENT p
      JOIN {D}.FACT_TREATMENT_DETAIL t ON t.PATIENT_DK = p.PATIENT_DK
      JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
        AND ({like(drugs)}) AND t.TREATMENT_DTM IS NOT NULL
      GROUP BY 1""").to_dataframe()
    if len(df):
        df["first_dt"] = pd.to_datetime(df["first_dt"])
        s = df.set_index("clinic")["first_dt"]
        gap = (s.reindex(P.index) - P["adt_date"]).dt.days
        got[lbl] = gap
        within = ((gap >= -30) & (gap <= 120)).sum()
        print(f"  {lbl:<44}{len(df):>6,} men   "
              f"started within 4mo of ADT: {within:,}")
    else:
        print(f"  {lbl:<44}{0:>6}")

any_int = pd.concat(got.values(), axis=1).min(axis=1) if got else pd.Series(dtype=float)
P["intens_gap"] = any_int
P["intensified"] = ((P["intens_gap"] >= -30) & (P["intens_gap"] <= 120)).fillna(False)

print("\n" + "=" * 88)
print("IMMORTAL TIME -- when does intensification start?")
print("=" * 88)
g = P["intens_gap"].dropna()
if len(g):
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<3} {g.quantile(q/100):>8.0f} days after ADT start")
    print(f"  -> a 4-month grace window is the standard fix; men who die or")
    print(f"     progress inside it must be handled, not silently dropped.")

print("\n" + "=" * 88)
print("THE EMULATION COHORT, BY ERA")
print("=" * 88)
print(f"  {'ADT era':<14}{'men':>8}{'intensified':>14}{'%':>8}{'progressed':>12}")
for lo, hi, lbl in [(2013, 2016, "2013-2016"), (2017, 2019, "2017-2019"),
                    (2020, 2022, "2020-2022"), (2023, 2026, "2023-2026")]:
    s = P[(P["year"] >= lo) & (P["year"] <= hi)]
    if len(s):
        print(f"  {lbl:<14}{len(s):>8,}{int(s['intensified'].sum()):>14,}"
              f"{s['intensified'].mean():>8.1%}{int(s['prog'].sum()):>12,}")

era = P[(P["year"] >= 2018) & (P["year"] <= 2023)]
n_t = int(era["intensified"].sum()); n_c = len(era) - n_t
print(f"\n  primary emulation window 2018-2023: {len(era):,} men "
      f"({n_t:,} intensified vs {n_c:,} ADT alone)")
print(f"  events: {int(era['prog'].sum()):,}")

print("\n" + "=" * 88)
print("THE EXTENSION COHORT -- men the trials would have EXCLUDED")
print("=" * 88)
labs = pd.read_parquet("psa_labs.parquet").reindex(era.index)
def col(stem, suf="__last"):
    c = [x for x in labs.columns if x.lower().startswith(stem) and x.endswith(suf)]
    return labs[c[0]] if c else pd.Series(np.nan, index=labs.index)
crit = {
  "haemoglobin < 10 g/dL":    col("cbc_hb") < 10,
  "platelets < 100 K/uL":     col("cbc_plt") < 100,
  "albumin < 3.0 g/dL":       col("chem_alb") < 3.0,
  "ALT > 2.5x normal (>70)":  col("chem_alt") > 70,
  "alk phos > 2.5x (>325)":   col("chem_alp") > 325,
}
excl = pd.DataFrame(crit).fillna(False)
any_excl = excl.any(axis=1)
for k, v in crit.items():
    print(f"    {k:<28}{int(pd.Series(v).fillna(False).sum()):>6,}")
print(f"\n  would fail >=1 common trial lab criterion: {int(any_excl.sum()):,} "
      f"({any_excl.mean():.1%} of the window)")
print("  -> these men are absent from LATITUDE/ENZAMET. Whether intensification")
print("     helps them is unanswered and only answerable with data like this.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"target_trial | window_n={len(era)} | intensified={n_t} | control={n_c} "
       f"| events={int(era['prog'].sum())} | trial_ineligible={int(any_excl.sum())} "
       f"| med_gap_days={int(g.median()) if len(g) else -1}")

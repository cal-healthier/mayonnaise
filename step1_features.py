"""
Step 1 - map the 28 Oncoformer input features onto Mayo's OMOP `measurement`.

One cached aggregation pass over measurement, then for each of the 28 features:
  - find every candidate concept by name keyword
  - pick the candidate whose median value is closest to the expected value
    (this auto-corrects unit choices: kg not ounces, x10^12/L not fL, etc.)
  - show all candidates so the choice is auditable

Writes feature_map.csv (the chosen concept per feature).

Usage inside the enclave:
    pull("step1_features.py"); %run step1_features.py
"""
import os
import math
import pandas as pd
from google.cloud import bigquery

PROJECT = "mcp-acc-055-dbg-p-7e23"
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
S = D + ".INFORMATION_SCHEMA.COLUMNS"
CACHE = "measurement_concepts.parquet"

C = bigquery.Client(project=PROJECT)


def Q(sql, label=""):
    try:
        return list(C.query(sql))
    except Exception as e:
        print(f"  [ERR {label}] {str(e)[:160]}")
        return []


# ---------------------------------------------------------------- 0. schema
cols = [r.column_name for r in Q(
    f"SELECT column_name FROM {S} WHERE table_name='measurement' ORDER BY ordinal_position",
    "measurement cols")]
VALCOL = "value_as_number" if "value_as_number" in cols else None
UNITCOL = next((c for c in ("unit_source_value", "unit_concept_id") if c in cols), None)
if VALCOL is None:
    raise SystemExit("no value_as_number column in measurement")


# ---------------------------------------------------------------- 1. one pass (cached)
if os.path.exists(CACHE):
    df = pd.read_parquet(CACHE)
    print(f"loaded cache: {len(df):,} concepts\n")
else:
    print("aggregating measurement (one pass, few $, cached after)\n")
    unit_sel = (f"APPROX_TOP_COUNT(m.{UNITCOL}, 1)[OFFSET(0)].value AS unit"
                if UNITCOL else "CAST(NULL AS STRING) AS unit")
    df = C.query(f"""
        SELECT m.measurement_concept_id AS cid, c.concept_name AS name,
               c.concept_code AS code, c.vocabulary_id AS vocab,
               COUNT(*) AS n_rows, APPROX_COUNT_DISTINCT(m.person_id) AS n_persons,
               APPROX_QUANTILES(m.{VALCOL}, 100)[OFFSET(50)] AS p50,
               {unit_sel}
        FROM {D}.measurement m
        LEFT JOIN {D}.concept c ON c.concept_id = m.measurement_concept_id
        GROUP BY 1,2,3,4 HAVING COUNT(*) > 50000 ORDER BY n_rows DESC
    """).to_dataframe()
    df.to_parquet(CACHE)
    print(f"{len(df):,} concepts >50k rows (cached to {CACHE})\n")

names = df["name"].fillna("")


# ---------------------------------------------------------------- 2. the 28 features
# (label, name keyword regex, expected median in whatever unit Mayo is likely to use)
# 'expect' is only used to pick the right candidate when several match.
TARGETS = [
    ("CBC_WBC",        r"leukocyte|white blood cell|\bwbc\b",                6.6),
    ("CBC_RBC",        r"erythrocyte|red blood cell|\brbc\b",               4.6),
    ("CBC_Hb",         r"hemoglobin",                                      13.5),
    ("CBC_Hct",        r"hematocrit",                                      41.0),
    ("CBC_PLT",        r"platelet",                                       250.0),
    ("CBC_MCV",        r"mean corpuscular volume|\bmcv\b",                 90.0),
    ("CBC_MCH",        r"mean corpuscular hemoglobin|\bmch\b",             30.0),
    ("CBC_MCHC",       r"corpuscular hemoglobin concentration|\bmchc\b",   33.5),
    ("CBC_RDW",        r"distribution width|\brdw\b",                      13.5),
    ("CBC_MPV",        r"platelet.*volume|mean platelet|\bmpv\b",          10.0),
    ("CBC_PDW",        r"platelet distribution|\bpdw\b",                   16.0),
    ("CBC_PCT",        r"plateletcrit",                                     0.25),
    ("CBC_NEUT_perc",  r"neutrophil",                                      60.0),
    ("CBC_LYMPH_perc", r"lymphocyte",                                      30.0),
    ("CBC_MONO_perc",  r"monocyte",                                         8.0),
    ("CBC_EOS_perc",   r"eosinophil",                                       2.5),
    ("CBC_BASO_perc",  r"basophil",                                         0.6),
    ("Chem_ALB",       r"albumin",                                          4.2),
    ("Chem_ALP",       r"alkaline phosphatase",                            75.0),
    ("Chem_ALT",       r"alanine aminotransferase",                        22.0),
    ("sign_SBP",       r"systolic blood pressure",                        128.0),
    ("sign_DBP",       r"diastolic blood pressure",                        74.0),
    ("sign_temp",      r"body temperature",                               37.0),
    ("sign_heartrate", r"heart rate",                                      78.0),
    ("sign_pulse",     r"^pulse|pulse rate",                              78.0),
    ("sign_breath",    r"respiratory rate",                               17.0),
    ("sign_height",    r"body height",                                    170.0),
    ("sign_weight",    r"body weight",                                     75.0),
]


def closeness(p50, expect):
    """log-distance to expected; huge if p50 missing/zero so it loses."""
    if p50 is None or (isinstance(p50, float) and math.isnan(p50)) or p50 <= 0:
        return 999.0
    return abs(math.log(p50 / expect))


found, missing = [], []
print("=" * 100)
print(f"{'feature':<16}{'chosen p50':>11}  {'unit':<14}{'persons':>11}  concept")
print("=" * 100)

for label, pat, expect in TARGETS:
    hits = df[names.str.contains(pat, case=False, regex=True, na=False)].copy()
    if hits.empty:
        print(f"{label:<16}{'-- NO MATCH --':>11}")
        missing.append(label)
        continue
    hits["score"] = hits["p50"].apply(lambda v: closeness(v, expect))
    hits = hits.sort_values(["score", "n_rows"], ascending=[True, False])
    best = hits.iloc[0]
    print(f"{label:<16}{best.p50:>11.2f}  {str(best.unit)[:14]:<14}{best.n_persons:>11,}  "
          f"{str(best.name)[:44]}")
    found.append((label, int(best.cid), best.name, int(best.n_persons),
                  float(best.p50), best.unit))
    for _, h in hits.iloc[1:4].iterrows():
        print(f"{'':<16}{h.p50:>11.2f}  {str(h.unit)[:14]:<14}{h.n_persons:>11,}    alt: {str(h.name)[:40]}")

print("\n" + "=" * 60)
print(f"matched {len(found)} of {len(TARGETS)} features")
if missing:
    print("MISSING:", ", ".join(missing))
print("=" * 60)

pd.DataFrame(found, columns=["feature", "concept_id", "concept_name",
                             "n_persons", "p50", "unit"]).to_csv("feature_map.csv", index=False)
print("wrote feature_map.csv")

# ---- diagnostic: for anything still missing, sweep the raw concept names
if missing:
    print("\n=== keyword sweep for the missing features (real concept names) ===")
    KW = {"CBC_MCHC": "mchc|corpuscular hemoglobin conc", "CBC_RDW": "rdw|distribution width",
          "CBC_MPV": "mpv|platelet.*volume", "CBC_PDW": "pdw|platelet distribution",
          "CBC_PCT": "plateletcrit|\\bpct\\b", "CBC_NEUT_perc": "neutrophil",
          "CBC_LYMPH_perc": "lymphocyte", "CBC_MONO_perc": "monocyte",
          "CBC_EOS_perc": "eosinophil", "CBC_BASO_perc": "basophil",
          "sign_height": "height", "sign_temp": "temperature"}
    for lab in missing:
        kw = KW.get(lab, lab.split("_")[-1])
        sweep = df[names.str.contains(kw, case=False, regex=True, na=False)].head(4)
        print(f"\n  {lab}  (searching: {kw})")
        if sweep.empty:
            print("     (nothing - likely stored under concept_id 0 / source value only)")
        for _, r in sweep.iterrows():
            print(f"     cid={int(r.cid):<9} p50={r.p50:>8.2f} {str(r.unit)[:10]:<10} "
                  f"n={int(r.n_persons):>9,}  {str(r.name)[:46]}")

# ---------------------------------------------------------------- 3. one safe line
# Structural metadata only (how many of 28 std labs mapped) - no patient values.
exp = {t[0]: t[2] for t in TARGETS}
suspicious = [f[0] for f in found if closeness(f[4], exp[f[0]]) > 0.5]
print("\n" + "-" * 70)
print("COPY THE NEXT LINE TO CLAUDE:")
print(f"CLAUDE_STATUS step1 | matched={len(found)}/{len(TARGETS)}"
      f" | missing={','.join(missing) if missing else 'none'}"
      f" | check={','.join(suspicious) if suspicious else 'none'}")

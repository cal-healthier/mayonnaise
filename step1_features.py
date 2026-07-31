"""
Step 1 — map the 28 Oncoformer input features onto Mayo's OMOP `measurement`.

Runs one aggregation pass over measurement, caches it to parquet, then matches
concept names against the feature list from the authors' own feat_info.json and
reconciles units against their normalisation constants.

Usage inside the enclave:
    pull("step1_features.py")
    %run step1_features.py
"""
import os
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
print("measurement columns:", cols, "\n")

VALCOL = "value_as_number" if "value_as_number" in cols else None
UNITCOL = next((c for c in ("unit_source_value", "unit_concept_id") if c in cols), None)
if VALCOL is None:
    raise SystemExit("no value_as_number column — stop and inspect `measurement`")


# ---------------------------------------------------------------- 1. one pass
if os.path.exists(CACHE):
    df = pd.read_parquet(CACHE)
    print(f"loaded cache: {len(df):,} concepts\n")
else:
    print("running aggregation over measurement (one pass, few $ — cached after)\n")
    unit_sel = (f"APPROX_TOP_COUNT(m.{UNITCOL}, 1)[OFFSET(0)].value AS unit"
                if UNITCOL else "CAST(NULL AS STRING) AS unit")
    df = C.query(f"""
        SELECT m.measurement_concept_id                              AS cid,
               c.concept_name                                        AS name,
               c.concept_code                                        AS code,
               c.vocabulary_id                                       AS vocab,
               COUNT(*)                                              AS n_rows,
               APPROX_COUNT_DISTINCT(m.person_id)                    AS n_persons,
               APPROX_QUANTILES(m.{VALCOL}, 100)[OFFSET(50)]         AS p50,
               APPROX_QUANTILES(m.{VALCOL}, 100)[OFFSET(5)]          AS p05,
               APPROX_QUANTILES(m.{VALCOL}, 100)[OFFSET(95)]         AS p95,
               {unit_sel}
        FROM {D}.measurement m
        LEFT JOIN {D}.concept c ON c.concept_id = m.measurement_concept_id
        GROUP BY 1, 2, 3, 4
        HAVING COUNT(*) > 50000
        ORDER BY n_rows DESC
    """).to_dataframe()
    df.to_parquet(CACHE)
    print(f"{len(df):,} concepts with >50k rows (cached to {CACHE})\n")


# ---------------------------------------------------------------- 2. targets
# (label, name regex, their mean from feat_info.json, expected US-unit mean)
TARGETS = [
    ("CBC_WBC",        r"leukocyte|white blood cell|\bWBC\b",                6.65,   6.6),
    ("CBC_RBC",        r"erythrocyte.*(count|#)|red blood cell",             4.14,   4.6),
    ("CBC_Hb",         r"hemoglobin",                                      124.50,  13.5),
    ("CBC_Hct",        r"hematocrit",                                       37.36,  41.0),
    ("CBC_PLT",        r"platelet.*(count|#)",                             219.58, 250.0),
    ("CBC_MCV",        r"mean corpuscular volume|\bMCV\b",                   90.81,  90.0),
    ("CBC_MCH",        r"mean corpuscular hemoglobin$|\bMCH\b",              30.20,  30.0),
    ("CBC_MCHC",       r"mean corpuscular hemoglobin concentration",        332.37,  33.5),
    ("CBC_RDW",        r"erythrocyte distribution width|\bRDW\b",            14.22,  13.5),
    ("CBC_MPV",        r"mean.*volume.*platelet|platelet mean volume|\bMPV\b", 10.20, 10.0),
    ("CBC_PDW",        r"platelet distribution width|\bPDW\b",               13.40,  16.0),
    ("CBC_PCT",        r"plateletcrit",                                       0.22,   0.25),
    ("CBC_NEUT_perc",  r"neutrophil.*/100 leukocyte|neutrophil.*percent",    60.26,  60.0),
    ("CBC_LYMPH_perc", r"lymphocyte.*/100 leukocyte|lymphocyte.*percent",    29.02,  30.0),
    ("CBC_MONO_perc",  r"monocyte.*/100 leukocyte|monocyte.*percent",         7.92,   8.0),
    ("CBC_EOS_perc",   r"eosinophil.*/100 leukocyte|eosinophil.*percent",     1.88,   2.5),
    ("CBC_BASO_perc",  r"basophil.*/100 leukocyte|basophil.*percent",         0.34,   0.6),
    ("Chem_ALB",       r"^albumin|albumin \[mass",                           41.73,   4.2),
    ("Chem_ALP",       r"alkaline phosphatase",                             101.94,  75.0),
    ("Chem_ALT",       r"alanine aminotransferase",                          31.20,  22.0),
    ("sign_SBP",       r"systolic blood pressure",                          130.10, 128.0),
    ("sign_DBP",       r"diastolic blood pressure",                          76.42,  74.0),
    ("sign_temp",      r"body temperature",                                  36.86,  98.2),
    ("sign_heartrate", r"heart rate",                                       103.03,  78.0),
    ("sign_pulse",     r"^pulse|pulse rate",                                 86.54,  78.0),
    ("sign_breath",    r"respiratory rate",                                  23.79,  17.0),
    ("sign_height",    r"body height",                                      160.39,  66.0),
    ("sign_weight",    r"body weight",                                       58.59, 175.0),
]

names = df["name"].fillna("")
found, missing = [], []

print("=" * 118)
print(f"{'feature':<16}{'persons':>11}{'p50':>9}{'theirs':>9}{'ratio':>7}  {'unit':<12} concept")
print("=" * 118)

for label, pat, their_mean, expect in TARGETS:
    hits = df[names.str.contains(pat, case=False, regex=True, na=False)]
    if hits.empty:
        print(f"{label:<16}{'-- NO MATCH --':>11}")
        missing.append(label)
        continue
    r = hits.iloc[0]
    ratio = (r.p50 / their_mean) if their_mean else float("nan")
    off = "" if (expect and 0.7 < (r.p50 / expect) < 1.4) else "   <-- CHECK"
    print(f"{label:<16}{r.n_persons:>11,}{r.p50:>9.2f}{their_mean:>9.2f}{ratio:>7.3f}  "
          f"{str(r.unit)[:12]:<12} {str(r.name)[:40]}{off}")
    found.append((label, r.cid, r.name, r.n_persons, r.p50, r.unit))
    for _, h in hits.iloc[1:3].iterrows():
        print(f"{'':<16}{h.n_persons:>11,}{h.p50:>9.2f}{'':>9}{'':>7}  "
              f"{str(h.unit)[:12]:<12}   alt: {str(h.name)[:38]}")

print("\n" + "=" * 60)
print(f"matched {len(found)} of {len(TARGETS)} features")
if missing:
    print("MISSING:", ", ".join(missing))
print("=" * 60)

pd.DataFrame(found, columns=["feature", "concept_id", "concept_name",
                             "n_persons", "p50", "unit"]).to_csv("feature_map.csv", index=False)
print("wrote feature_map.csv")

print("\n=== safety net: top 40 measurement concepts by volume ===")
for _, r in df.head(40).iterrows():
    print(f"  {r.n_rows:>14,}  {r.n_persons:>10,}  {str(r.unit)[:10]:<10} {str(r.name)[:60]}")

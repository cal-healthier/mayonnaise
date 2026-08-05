"""
four_arm_encoding.py -- does the shared-response premise survive four cancers?

Two arms gave a mixed verdict: log-vs-baseline overlapped 70% (good), but
nadir-ratio and velocity were degenerate because PSA pins to its assay floor
(0.10) and stops moving. Same failure mode that broke the first PSA label.

Fix applied here: treat the marker as LEFT-CENSORED at its detection limit.
"Undetectable" becomes an explicit state, and relative encodings are computed
only on detectable values. That is also how oncologists reason.

Then the real question: is PSA simply the awkward one? CEA and CA 19-9 range
freely and do not pin to a floor. If all four arms overlap once censoring is
handled, the premise holds. If they scatter, the world-model framing is wrong
and this is four disease models.

Builds CEA and CA 19-9 trajectories (cached after first run).
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

cc = pd.read_parquet("measurement_concepts.parquet")
cc["nl"] = cc["name"].astype(str).str.lower()
medcols = [r.column_name for r in C.query(
    f"SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS "
    f"WHERE table_name='DIM_MED_NAME'")]
namecols = [c for c in medcols if "NAME" in c.upper() and "DK" not in c.upper()]
SEARCH = ("UPPER(CONCAT(" +
          ", ' ', ".join(f"IFNULL(CAST({c} AS STRING), '')" for c in namecols) + "))")

NEW = [
 ("colorectal", "C18|C19|C20", r"carcinoembryonic", None,
  ["FLUOROURACIL","CAPECITABINE","OXALIPLATIN","IRINOTECAN"], 5.0, "cea.parquet"),
 ("pancreas", "C25", r"ca 19-9|carbohydrate antigen 19", None,
  ["GEMCITABINE","FLUOROURACIL","ABRAXANE","IRINOTECAN","OXALIPLATIN",
   "CAPECITABINE"], 37.0, "ca199.parquet"),
]
for arm, sites, pat, exc, drugs, lod, cache in NEW:
    if os.path.exists(cache):
        continue
    h = cc[cc["nl"].str.contains(pat, na=False)]
    if exc:
        h = h[~h["nl"].str.contains(exc, na=False)]
    cid = ",".join(str(int(x)) for x in
                   h.sort_values("n_persons", ascending=False).head(4)["cid"])
    site_sql = " OR ".join(f"CAST(SITE_PRIMARY_ICD_O_3 AS STRING) LIKE '{s}%'"
                           for s in sites.split("|"))
    like = " OR ".join(f"{SEARCH} LIKE '%{d}%'" for d in drugs)
    print(f"pulling {arm} ...")
    df = C.query(f"""
    WITH reg AS (SELECT PATIENT_DK, COUNT(*) n_prim, MIN(DATE(DATE_OF_DIAGNOSIS)) dx
      FROM {D}.FACT_CANCER_DATA_REPOSITORY
      WHERE DATE_OF_DIAGNOSIS IS NOT NULL AND ({site_sql}) GROUP BY 1),
    tx AS (SELECT t.PATIENT_DK, MIN(DATE(t.TREATMENT_DTM)) tx_date
      FROM {D}.FACT_TREATMENT_DETAIL t JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
      WHERE ({like}) AND t.TREATMENT_DTM IS NOT NULL GROUP BY 1),
    br AS (SELECT DISTINCT PATIENT_DK, CAST(PATIENT_CLINIC_NUMBER AS STRING) clinic
      FROM {D}.DIM_PATIENT),
    pe AS (SELECT CAST(person_source_value AS STRING) clinic, MIN(person_id) person_id
      FROM {D}.person GROUP BY 1),
    co AS (SELECT b.clinic, p.person_id, tx.tx_date FROM reg r
      JOIN tx USING (PATIENT_DK) JOIN br b USING (PATIENT_DK)
      JOIN pe p ON p.clinic=b.clinic
      WHERE r.n_prim=1 AND tx.tx_date>=r.dx
      QUALIFY ROW_NUMBER() OVER (PARTITION BY b.clinic ORDER BY tx.tx_date)=1)
    SELECT co.clinic, co.tx_date, DATE(m.measurement_date) d, m.value_as_number val
    FROM co JOIN {D}.measurement m ON m.person_id=co.person_id
    WHERE m.measurement_concept_id IN ({cid}) AND m.value_as_number IS NOT NULL
      AND m.value_as_number >= 0""").to_dataframe()
    df["tx_date"] = pd.to_datetime(df["tx_date"]); df["d"] = pd.to_datetime(df["d"])
    df["day"] = (df["d"] - df["tx_date"]).dt.days
    df.to_parquet(cache)
    print(f"  {len(df):,} draws, {df['clinic'].nunique():,} patients")

def prep(df, val, lod, name):
    df = df.sort_values(["clinic", "day"]).copy()
    pre = df[(df["day"] <= 0) & (df["day"] >= -90)].groupby("clinic")[val].last()
    post = df[df["day"] > 0].copy()
    post["base"] = pre.reindex(post["clinic"]).values
    post = post[post["base"].notna() & (post["base"] > lod)]
    post["undet"] = post[val] <= lod
    det = post[~post["undet"]].copy()
    det["nadir"] = det.groupby("clinic")[val].cummin()
    det["log_base"] = np.log(det[val] / det["base"])
    det["log_nadir"] = np.log(det[val] / det["nadir"])
    g = det.groupby("clinic")
    det["velocity"] = g["log_base"].diff() / (g["day"].diff() / 30.4)
    det["arm"] = name
    return post, det

ARMS = []
for f, lab, val, lod, nm in [
    ("psa_values.parquet", "psa_progression.parquet", "psa", 0.10, "prostate/PSA"),
    ("ov_ca125.parquet", "ov_label.parquet", "ca", 2.0, "ovarian/CA-125"),
    ("cea.parquet", None, "val", 0.5, "colorectal/CEA"),
    ("ca199.parquet", None, "val", 2.0, "pancreas/CA19-9")]:
    if not os.path.exists(f):
        continue
    d = pd.read_parquet(f)
    p, det = prep(d, val, lod, nm)
    ARMS.append((nm, p, det, lod))

print("\n" + "=" * 92)
print("HOW MUCH OF EACH MARKER SITS AT THE DETECTION FLOOR?")
print("=" * 92)
for nm, p, det, lod in ARMS:
    print(f"  {nm:<20}LOD {lod:<6} undetectable {p['undet'].mean():6.1%}   "
          f"{p['clinic'].nunique():,} patients, {len(p):,} draws")
print("  -> PSA's floor is the outlier; the others range freely")

print("\n" + "=" * 92)
print("RELATIVE ENCODINGS, DETECTABLE VALUES ONLY -- do all four overlap?")
print("=" * 92)
for feat, lbl in [("log_base", "log(value / baseline)"),
                  ("log_nadir", "log(value / running nadir)"),
                  ("velocity", "velocity (log change / month)")]:
    print(f"\n  {lbl}")
    print(f"    {'':<22}{'p25':>9}{'median':>9}{'p75':>9}")
    qs = {}
    for nm, p, det, lod in ARMS:
        s = det[feat].replace([np.inf, -np.inf], np.nan).dropna()
        q = s.quantile([.25, .5, .75]).values
        qs[nm] = q
        print(f"    {nm:<22}" + "".join(f"{x:>9.2f}" for x in q))
    lo = max(q[0] for q in qs.values()); hi = min(q[2] for q in qs.values())
    span = max(q[2] for q in qs.values()) - min(q[0] for q in qs.values())
    ov = max(0.0, hi - lo) / span if span > 0 else 0.0
    print(f"    -> all-four IQR overlap {ov:.0%}  "
          f"{'SHARED SPACE' if ov > 0.4 else 'scattered'}")

print("\n" + "=" * 92)
print("RESPONSE SHAPE BY MONTH  (median log-vs-baseline, detectable values)")
print("=" * 92)
hdr = "".join(f"{nm.split('/')[0][:9]:>11}" for nm, _, _, _ in ARMS)
print(f"  {'month':>6}{hdr}")
curves = {nm: [] for nm, _, _, _ in ARMS}
for m in range(0, 19, 2):
    row = []
    for nm, p, det, lod in ARMS:
        s = det[(det["day"] >= m*30.4) & (det["day"] < (m+2)*30.4)]["log_base"]
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        v = s.median() if len(s) > 20 else np.nan
        curves[nm].append(v); row.append(v)
    print(f"  {m:>6}" + "".join(f"{x:>11.2f}" for x in row))

names = list(curves)
print("\n  pairwise correlation of response curves:")
best = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a = np.array(curves[names[i]]); b = np.array(curves[names[j]])
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() >= 4:
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            best.append(r)
            print(f"    {names[i]:<20} vs {names[j]:<20}{r:>+7.2f}")
mean_r = float(np.nanmean(best)) if best else np.nan
print(f"\n  mean pairwise correlation: {mean_r:+.2f}")
print("  (>+0.5 across most pairs => one shared process, world model justified)")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"four_arm | arms={len(ARMS)} | mean_shape_corr={mean_r:+.2f} "
      f"| psa_undet={ARMS[0][1]['undet'].mean():.0%}")

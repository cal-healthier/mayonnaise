"""
lead_time.py -- do routine blood tests move BEFORE the tumour marker does?

First principles: our advantage is not scale (3.67M vs our 25k). It is DENSITY
and DURATION -- 30 PSA draws per man over 2.8 years, 26 labs before treatment
even starts. So the study has to be one that cannot be done without dense
serial measurement in parallel.

Lead time is that study. Alkaline phosphatase was the top predictor of PSA
progression on two independently built endpoints -- but we only ever used
BASELINE ALP. The question nobody asked: is ALP already climbing while PSA is
still flat?

Method: align every man who progressed at his progression DATE (t=0), then look
backwards month by month at both signals. Compare against men who did not
progress, aligned at a matched follow-up time, so we are not just seeing
"sick people have abnormal labs".

If a lab starts drifting months before the marker crosses its threshold, that
is lead time -- and it changes when you scan, when you switch, when you look
harder.

Pulls during-treatment labs (we only ever had pre-treatment ones). Cached after.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

E = pd.read_parquet("psa_progression.parquet")
pv = pd.read_parquet("psa_values.parquet")
tx = pv.groupby("clinic")["tx_date"].first()
print(f"prostate cohort {len(E):,}, progressed {int(E['prog'].sum()):,}")

# ---------------------------------------------------------------- labs DURING treatment
if os.path.exists("psa_labs_during.parquet"):
    lab = pd.read_parquet("psa_labs_during.parquet")
    print("loaded cached psa_labs_during.parquet")
else:
    fm = pd.read_csv("feature_map_final.csv")
    CID = ",".join(str(int(x)) for x in fm["concept_id"])
    c2f = dict(zip(fm["concept_id"].astype(int), fm["feature"]))
    c2x = dict(zip(fm["concept_id"].astype(int), fm["to_std_factor"].fillna(1.0)))
    ids = "','".join(E.index.astype(str))
    print("pulling DURING-treatment labs (new extraction) ...")
    lab = C.query(f"""
      WITH ppl AS (
        SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
               MIN(pe.person_id) AS person_id
        FROM {D}.DIM_PATIENT p
        JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)
                            = CAST(p.PATIENT_CLINIC_NUMBER AS STRING)
        WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}') GROUP BY 1)
      SELECT ppl.clinic, DATE(m.measurement_date) AS d,
             m.measurement_concept_id AS cid, m.value_as_number AS value
      FROM ppl JOIN {D}.measurement m ON m.person_id = ppl.person_id
      WHERE m.measurement_concept_id IN ({CID}) AND m.value_as_number IS NOT NULL
    """).to_dataframe()
    lab["feature"] = lab["cid"].map(c2f)
    lab["value"] = lab["value"] * lab["cid"].map(c2x)
    lab["d"] = pd.to_datetime(lab["d"])
    lab = lab.dropna(subset=["feature"])[["clinic", "d", "feature", "value"]]
    lab.to_parquet("psa_labs_during.parquet")
print(f"  lab rows {len(lab):,}, {lab['clinic'].nunique():,} men")

# ---------------------------------------------------------------- align at t=0
E = E.copy()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E["anchor"] = E["tx_date"] + pd.to_timedelta(E["time"], unit="D")   # progression, or censor
prog = E[E["prog"] == 1]
ctrl = E[(E["prog"] == 0) & (E["time"] >= 540)]      # >=18mo followed, still responding
print(f"\naligned: {len(prog):,} progressors at their progression date, "
      f"{len(ctrl):,} controls at 18mo+")

lab = lab.join(E["anchor"], on="clinic").dropna(subset=["anchor"])
lab["mo"] = ((lab["d"] - lab["anchor"]).dt.days / 30.44).round().astype(int)
win = lab[(lab["mo"] >= -12) & (lab["mo"] <= 0)].copy()
win["grp"] = np.where(win["clinic"].isin(prog.index), "prog",
              np.where(win["clinic"].isin(ctrl.index), "ctrl", None))
win = win[win["grp"].notna()]

# per-patient baseline = that man's own median 12-9 months before the anchor
base = (win[win["mo"] <= -9].groupby(["clinic", "feature"])["value"].median()
        .rename("base"))
win = win.join(base, on=["clinic", "feature"])
win = win[win["base"].notna() & (win["base"] != 0)]
win["rel"] = win["value"] / win["base"]        # within-patient change; drift-proof

# ---------------------------------------------------------------- the curve
print("\n" + "=" * 92)
print("MONTHS BEFORE PROGRESSION: within-patient change vs own 9-12mo baseline")
print("=" * 92)
FEATS = ["Chem_ALP", "Chem_ALT", "Chem_ALB", "CBC_Hb", "CBC_Hct",
         "CBC_WBC", "CBC_PLT", "CBC_RBC"]
have = [f for f in FEATS if f in win["feature"].unique()]
print(f"  {'month':>6}" + "".join(f"{f.replace('Chem_','').replace('CBC_',''):>10}"
                                  for f in have) + "      PSA")
psa = pv.join(E["anchor"], on="clinic").dropna(subset=["anchor"])
psa["mo"] = ((psa["d"] - psa["anchor"]).dt.days / 30.44).round().astype(int)
psa = psa[psa["clinic"].isin(prog.index)]
pbase = psa[psa["mo"] <= -9].groupby("clinic")["psa"].median()
psa = psa.join(pbase.rename("pb"), on="clinic")
psa = psa[psa["pb"].notna() & (psa["pb"] > 0)]
psa["rel"] = psa["psa"] / psa["pb"]

rows = []
for m in range(-12, 1):
    line = f"  {m:>6}"
    rec = {"mo": m}
    for f in have:
        s = win[(win["grp"] == "prog") & (win["feature"] == f) & (win["mo"] == m)]["rel"]
        v = s.median() if len(s) >= 25 else np.nan
        rec[f] = v
        line += f"{v:>10.3f}" if not np.isnan(v) else f"{'-':>10}"
    sp = psa[psa["mo"] == m]["rel"]
    pv_ = sp.median() if len(sp) >= 25 else np.nan
    rec["PSA"] = pv_
    line += f"{pv_:>9.2f}" if not np.isnan(pv_) else f"{'-':>9}"
    rows.append(rec); print(line)
R = pd.DataFrame(rows).set_index("mo")

print("\n" + "=" * 92)
print("WHO MOVES FIRST?  (first month where the signal is 10% off its own baseline)")
print("=" * 92)
for c in [f for f in have] + ["PSA"]:
    s = R[c].dropna()
    if s.empty:
        continue
    dev = (s - 1).abs()
    hit = dev[dev >= 0.10]
    first = hit.index.min() if len(hit) else None
    print(f"  {c:<12}{'never' if first is None else f'{first:>3} months before':>22}"
          f"   value at t=0: {s.get(0, np.nan):.3f}")
print("\n  a lab that crosses BEFORE PSA does is lead time -- earlier warning than")
print("  the marker oncologists watch.")

print("\n" + "=" * 92)
print("CONTROL: same window, men still responding (should be flat)")
print("=" * 92)
print(f"  {'month':>6}" + "".join(f"{f.replace('Chem_','').replace('CBC_',''):>10}"
                                  for f in have))
for m in (-12, -9, -6, -3, 0):
    line = f"  {m:>6}"
    for f in have:
        s = win[(win["grp"] == "ctrl") & (win["feature"] == f) & (win["mo"] == m)]["rel"]
        v = s.median() if len(s) >= 25 else np.nan
        line += f"{v:>10.3f}" if not np.isnan(v) else f"{'-':>10}"
    print(line)
print("\n  if controls drift too, we are seeing time-on-treatment, not failure.")

alp = R["Chem_ALP"].dropna() if "Chem_ALP" in R else pd.Series(dtype=float)
alp_lead = None
if len(alp):
    d = (alp - 1).abs(); h = d[d >= 0.10]
    alp_lead = int(h.index.min()) if len(h) else None
p = R["PSA"].dropna(); d = (p - 1).abs(); h = d[d >= 0.10]
psa_lead = int(h.index.min()) if len(h) else None
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"lead_time | progressors={len(prog)} | controls={len(ctrl)} "
      f"| alp_first_move={alp_lead} | psa_first_move={psa_lead} "
      f"| lead_months={(psa_lead - alp_lead) if (alp_lead is not None and psa_lead is not None) else 'NA'}")

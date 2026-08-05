"""
lead_time_other.py -- does the early-warning pattern generalise?

ALP-before-PSA is published, but only in castration-RESISTANT disease (men
already on abiraterone/docetaxel). Ours is first-line hormone therapy, which is
new. The much stronger claim would be GENERALITY: does routine bloodwork warn
before the tumour marker in OTHER cancers too?

Prediction from the mechanism we verified earlier: ALP works in prostate
because prostate goes to BONE. Ovarian spreads peritoneally, so ALP should NOT
lead there -- something else should, or nothing. Colorectal goes to LIVER, so
liver enzymes (ALT, ALP, albumin) are the candidates.

If different labs lead in different cancers, matching each disease's pattern of
spread, that is a mechanistic finding about how treatment failure announces
itself -- and nobody has it.

Same method as prostate: align at the marker's progression date, look back 12
months at within-patient change, with a still-responding control group.
Pulls during-treatment labs for ovarian + colorectal. Cached after.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

fm = pd.read_csv("feature_map_final.csv")
CID = ",".join(str(int(x)) for x in fm["concept_id"])
c2f = dict(zip(fm["concept_id"].astype(int), fm["feature"]))
c2x = dict(zip(fm["concept_id"].astype(int), fm["to_std_factor"].fillna(1.0)))

def pull_labs(clinics, cache):
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    ids = "','".join(pd.Index(clinics).astype(str))
    print(f"  pulling labs -> {cache} ...")
    df = C.query(f"""
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
    df["feature"] = df["cid"].map(c2f)
    df["value"] = df["value"] * df["cid"].map(c2x)
    df["d"] = pd.to_datetime(df["d"])
    df = df.dropna(subset=["feature"])[["clinic", "d", "feature", "value"]]
    df.to_parquet(cache)
    return df

def progression(mk, val, lod, uln):
    """GCIG-style: >=2x running nadir, or >=2x ULN if nadir normalised."""
    mk = mk.sort_values(["clinic", "day"]).copy()
    post = mk[mk["day"] > 0].copy()
    post["nadir"] = post.groupby("clinic")[val].cummin()
    post["thr"] = 2.0 * np.maximum(post["nadir"], uln)
    post["flag"] = post[val] >= post["thr"]
    out = []
    for c, g in post.groupby("clinic", sort=False):
        d, f = g["day"].values, g["flag"].values
        fd = d[f]; ev, t = 0, d[-1]
        if len(fd) >= 2:
            for i, d0 in enumerate(fd[:-1]):
                if (fd[i+1:] >= d0 + 7).any():
                    ev, t = 1, d0; break
        out.append({"clinic": c, "prog": ev, "time": t, "n": len(g)})
    return pd.DataFrame(out).set_index("clinic")

ARMS = [
  ("ovarian / CA-125",   "ov_ca125.parquet",  "ca",  2.0, 35.0, "ov_labs_during.parquet"),
  ("colorectal / CEA",   "cea.parquet",       "val", 0.5,  5.0, "crc_labs_during.parquet"),
]
FEATS = ["Chem_ALP", "Chem_ALT", "Chem_ALB", "CBC_Hct", "CBC_WBC", "CBC_RBC", "CBC_PLT"]

for name, mkfile, val, lod, uln, cache in ARMS:
    print("\n" + "=" * 88)
    print(f"{name}")
    print("=" * 88)
    if not os.path.exists(mkfile):
        print(f"  {mkfile} missing -- run the earlier arm script first"); continue
    mk = pd.read_parquet(mkfile)
    if "tx_date" not in mk.columns:
        print("  no tx_date"); continue
    E = progression(mk, val, lod, uln)
    E = E[E["n"] >= 4]
    tx = mk.groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E["anchor"] = E["tx_date"] + pd.to_timedelta(E["time"], unit="D")
    prog = E[E["prog"] == 1]; ctrl = E[(E["prog"] == 0) & (E["time"] >= 540)]
    print(f"  {len(E):,} patients | progressed {len(prog):,} | "
          f"still responding at 18mo+ {len(ctrl):,}")
    if len(prog) < 100:
        print("  too few progressors for a stable curve"); continue

    lab = pull_labs(E.index, cache)
    lab = lab[lab["feature"].isin(FEATS)].join(E["anchor"], on="clinic")
    lab = lab.dropna(subset=["anchor"])
    lab["mo"] = ((lab["d"] - lab["anchor"]).dt.days / 30.44).round().astype(int)
    w = lab[(lab["mo"] >= -12) & (lab["mo"] <= 0)].copy()
    w["grp"] = np.where(w["clinic"].isin(prog.index), "prog",
               np.where(w["clinic"].isin(ctrl.index), "ctrl", None))
    w = w[w["grp"].notna()]
    base = w[w["mo"] <= -9].groupby(["clinic", "feature"])["value"].median().rename("b")
    w = w.join(base, on=["clinic", "feature"])
    w = w[w["b"].notna() & (w["b"] != 0)]
    w["rel"] = w["value"] / w["b"]

    have = [f for f in FEATS if f in w["feature"].unique()]
    print(f"\n  {'month':>6}" + "".join(
        f"{f.replace('Chem_','').replace('CBC_',''):>9}" for f in have) + "   marker")
    mkp = mk[mk["clinic"].isin(prog.index)].join(E["anchor"], on="clinic")
    mkp = mkp.dropna(subset=["anchor"])
    mkp["mo"] = ((mkp["d"] - mkp["anchor"]).dt.days / 30.44).round().astype(int)
    mb = mkp[mkp["mo"] <= -9].groupby("clinic")[val].median()
    mkp = mkp.join(mb.rename("b"), on="clinic")
    mkp = mkp[mkp["b"].notna() & (mkp["b"] > 0)]
    mkp["rel"] = mkp[val] / mkp["b"]
    curves = {f: {} for f in have}; mcurve = {}
    for m in range(-12, 1):
        line = f"  {m:>6}"
        for f in have:
            s = w[(w["grp"] == "prog") & (w["feature"] == f) & (w["mo"] == m)]["rel"]
            v = s.median() if len(s) >= 20 else np.nan
            curves[f][m] = v
            line += f"{v:>9.3f}" if not np.isnan(v) else f"{'-':>9}"
        s = mkp[mkp["mo"] == m]["rel"]
        mv = s.median() if len(s) >= 20 else np.nan
        mcurve[m] = mv
        line += f"{mv:>9.2f}" if not np.isnan(mv) else f"{'-':>9}"
        print(line)

    print("\n  first month RISING >=10% above own baseline (upward only):")
    def first_up(c):
        for m in range(-12, 1):
            v = c.get(m, np.nan)
            if not np.isnan(v) and v >= 1.10:
                return m
        return None
    res = {f: first_up(curves[f]) for f in have}
    res["MARKER"] = first_up(mcurve)
    for k, v in res.items():
        print(f"    {k:<12}{'never' if v is None else f'{v:>3} months before':>20}")
    lead = [f"{f} by {res['MARKER']-res[f]}mo" for f in have
            if res[f] is not None and res["MARKER"] is not None and res[f] < res["MARKER"]]
    print(f"\n  leads the marker: {', '.join(lead) if lead else 'nothing'}")

    print("\n  control (still responding) at -12/-6/0:")
    for f in have:
        vals = []
        for m in (-12, -6, 0):
            s = w[(w["grp"] == "ctrl") & (w["feature"] == f) & (w["mo"] == m)]["rel"]
            vals.append(s.median() if len(s) >= 20 else np.nan)
        print(f"    {f:<12}" + "".join(f"{v:>9.3f}" if not np.isnan(v) else f"{'-':>9}"
                                       for v in vals))

print("\n" + "-" * 74)
print("FINAL LINE:")
print("lead_time_other | see per-arm 'leads the marker' lines above")

"""
radiology_concord.py -- STUDY 3, THE DECIDING TEST.

Can a model read radiology narratives and identify treatment failure at the
same time an objective blood marker says it happened?

If yes, we can build a response endpoint for the cancers that have no marker
at all -- lung, melanoma, kidney -- which is the gap that blocked every
treatment-response question today.

Design:
  100 men who DID progress (PSA date known, PCWG3 criteria)
   40 men who did NOT  -> the false-positive control, without which
                          "we found progression in 80% of progressors" is
                          meaningless
  up to 8 cancer-relevant narratives each, spanning [-12mo, +6mo] around the
  PSA date (or a matched window for controls)

Then: does the first radiology-flagged progression line up with the PSA date,
and does it stay quiet in men who never progressed?

Caches after the pull AND after extraction, so a crash never costs the spend.
"""
import json, os, time
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)
N_PROG, N_CTRL, PER_MAN, THINK = 100, 40, 8, 2048

E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["anchor"] = E["tx_date"] + pd.to_timedelta(E["time"], unit="D")
prog = E[E["prog"] == 1].head(N_PROG)
ctrl = E[(E["prog"] == 0) & (E["time"] >= 540)].head(N_CTRL)
S = pd.concat([prog.assign(grp="progressed"), ctrl.assign(grp="still responding")])
print(f"{len(prog)} progressors + {len(ctrl)} controls = {len(S)} men")

if os.path.exists("rad_notes.parquet"):
    R = pd.read_parquet("rad_notes.parquet")
    print(f"loaded cached rad_notes.parquet ({len(R):,} reports)")
else:
    rows = ",".join(f"('{c}',DATE '{a.date()}')"
                    for c, a in zip(S.index.astype(str), S["anchor"]))
    print("pulling radiology narratives ...")
    R = C.query(f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, anch DATE>>[{rows}])),
    pk AS (SELECT c.clinic, c.anch, p.PATIENT_DK FROM coh c
           JOIN {D}.DIM_PATIENT p ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic)
    SELECT pk.clinic, pk.anch, DATE(r.RADIOLOGY_DTM) AS dt,
           CAST(r.SERVICE_MODALITY_CODE AS STRING) AS modality,
           SUBSTR(CAST(r.RADIOLOGY_NARRATIVE AS STRING), 1, 3500) AS txt
    FROM pk JOIN {D}.FACT_RADIOLOGY r
      ON CAST(r.PATIENT_DK AS STRING) = CAST(pk.PATIENT_DK AS STRING)
    WHERE r.RADIOLOGY_NARRATIVE IS NOT NULL
      AND LENGTH(CAST(r.RADIOLOGY_NARRATIVE AS STRING)) BETWEEN 200 AND 8000
      AND REGEXP_CONTAINS(UPPER(CAST(r.RADIOLOGY_NARRATIVE AS STRING)),
            r'MASS|LESION|METASTA|TUMOR|NODULE|OSSEOUS|SCLEROTIC|PROSTATE')
      AND DATE(r.RADIOLOGY_DTM) BETWEEN DATE_SUB(pk.anch, INTERVAL 365 DAY)
                                    AND DATE_ADD(pk.anch, INTERVAL 180 DAY)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic
            ORDER BY ABS(DATE_DIFF(DATE(r.RADIOLOGY_DTM), pk.anch, DAY))) <= {PER_MAN}
    """).to_dataframe()
    R["dt"] = pd.to_datetime(R["dt"]); R["anch"] = pd.to_datetime(R["anch"])
    R.to_parquet("rad_notes.parquet")
print(f"  {len(R):,} reports | {R['clinic'].nunique()} men | "
      f"median {R.groupby('clinic').size().median():.0f} each")
print(f"  estimated cost: ~{int(np.ceil(len(R)/5))} API calls with thinking")

SCHEMA = """Return ONLY a JSON array, one object per numbered report:
{"i": <index>,
 "is_cancer_assessment": true|false,
 "direction": "improved"|"stable"|"worse"|"new_lesion"|"not_assessable",
 "confident": true|false,
 "evidence_quote": "<the exact sentence you judged from, or null>"}
"worse" = existing disease has grown or progressed.
"new_lesion" = a new site of disease not previously present.
"not_assessable" = no comparison possible, or the scan is not about the cancer.
Judge ONLY what this report states. Do not infer from clinical context."""

if os.path.exists("rad_extracted.parquet"):
    X = pd.read_parquet("rad_extracted.parquet")
    print("loaded cached rad_extracted.parquet")
else:
    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=THINK),
        response_mime_type="application/json", temperature=0)
    out, T, t0 = [], R["txt"].tolist(), time.time()
    print("extracting (this is the spend) ...")
    for i in range(0, len(T), 5):
        ch = T[i:i + 5]
        listed = "\n\n".join(f"### REPORT {j}\n{t[:3000]}" for j, t in enumerate(ch))
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash",
                contents=f"Read these radiology reports.\n\n{SCHEMA}\n\n{listed}",
                config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            out += [{}] * len(ch)
        if i % 50 == 0:
            print(f"  ...{min(i+5, len(T))}/{len(T)}  ({time.time()-t0:.0f}s)", end="\r")
        time.sleep(0.1)
    X = R.join(pd.DataFrame(out))
    X.to_parquet("rad_extracted.parquet")
    print(f"\n  done in {time.time()-t0:.0f}s")

X["prog_flag"] = X["direction"].isin(["worse", "new_lesion"]) & \
                 X["is_cancer_assessment"].fillna(False).astype(bool)
X["grp"] = X["clinic"].map(S["grp"])
X["mo"] = (X["dt"] - X["anch"]).dt.days / 30.44

print("\n" + "=" * 78)
print("1. WHAT DID IT SEE?")
print("=" * 78)
print(f"  reports judged a cancer assessment: "
      f"{X['is_cancer_assessment'].fillna(False).mean():.0%}")
for k, v in X["direction"].value_counts().items():
    print(f"    {str(k):<18}{int(v):>6}  ({v/len(X):5.0%})")

print("\n" + "=" * 78)
print("2. THE CONTROL -- does it stay quiet in men who never progressed?")
print("=" * 78)
per = X.groupby(["clinic", "grp"])["prog_flag"].max().reset_index()
for g in ("progressed", "still responding"):
    s = per[per["grp"] == g]
    if len(s):
        print(f"  {g:<20}{int(s['prog_flag'].sum()):>4}/{len(s):<4} "
              f"({s['prog_flag'].mean():.0%}) had a radiology progression flagged")
a = per[per["grp"] == "progressed"]["prog_flag"].mean()
b = per[per["grp"] == "still responding"]["prog_flag"].mean()
print(f"\n  separation: {a-b:+.0%}")
print("  a big gap means the reader is detecting real disease progression,")
print("  not just describing sick people.")

print("\n" + "=" * 78)
print("3. TIMING -- when radiology flags it vs when PSA did")
print("=" * 78)
p = X[(X["grp"] == "progressed") & X["prog_flag"]]
first = p.groupby("clinic")["mo"].min()
if len(first):
    print(f"  men with a flagged report: {len(first)} of {len(prog)}")
    for q in (10, 25, 50, 75, 90):
        print(f"    p{q:<3} {first.quantile(q/100):>+7.1f} months "
              f"{'(before PSA)' if first.quantile(q/100) < 0 else '(after PSA)'}")
    print(f"\n  within +/-3 months of the PSA date: "
          f"{(first.abs() <= 3).mean():.0%}")
    print(f"  radiology flagged it FIRST:          {(first < 0).mean():.0%}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"rad_concord | reports={len(X)} | men={X['clinic'].nunique()} "
      f"| prog_flagged={a:.0%} | ctrl_flagged={b:.0%} | sep={a-b:+.0%} "
      f"| within3mo={(first.abs()<=3).mean() if len(first) else float('nan'):.0%}")

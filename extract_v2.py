"""
extract_v2.py -- add polarity. The 60% was a schema failure, not a reading failure.

Case 1/76 diagnosed it: the note said "Patient has a history of seizures and can
not receive enzalutamide (Xtandi)". The model returned drug_named=enzalutamide --
a drug explicitly RULED OUT. My schema asked for "the drug being started" and
gave it no way to say excluded-vs-started, so both look identical in the output.

Meanwhile its REASON extraction was exactly right: comorbidity, specific=True,
quoting the seizure sentence. That is precisely the confounder Tier-2 needs --
treatment selection driven by a comorbidity, in prose, invisible in any
structured field.

So: re-extract with an explicit status per drug, and score only the ones marked
as being started. Same notes, same model, same truth -- only the schema changes,
so the comparison is clean.
"""
import json, os, time
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
X = pd.read_parquet("browse_cache.parquet")
X["dec"] = X["mentions_cancer_treatment_decision"].fillna(False).astype(bool)
D = X[X["dec"]].reset_index(drop=True)
print(f"{len(D):,} flagged decision notes from {D['clinic'].nunique()} men")

SCHEMA = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>,
 "drugs": [{"name": "<generic drug name>",
            "status": "starting"|"continuing"|"considered"|"excluded"|"prior"|"other"}],
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
                 "patient_preference"|"toxicity_concern"|"guideline_standard"|
                 "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false,
 "evidence_quote": "<the exact sentence supporting reason_given, or null>"}

STATUS MEANINGS -- get these right, they matter more than the drug name:
  starting   = this patient is being put on it now
  continuing = already on it, carrying on
  considered = discussed as an option, not chosen
  excluded   = ruled OUT (contraindication, allergy, intolerance, refusal)
  prior      = had it in the past, not now
Example: "history of seizures and cannot receive enzalutamide" -> enzalutamide
is EXCLUDED, not starting. Never list a ruled-out drug as starting."""

def extract(texts, batch=6):
    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json", temperature=0)
    out = []
    for i in range(0, len(texts), batch):
        ch = texts[i:i + batch]
        listed = "\n\n".join(f"### NOTE {j}\n{t[:4000]}" for j, t in enumerate(ch))
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash",
                contents=f"Read these oncology notes.\n\n{SCHEMA}\n\n{listed}", config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            print(f"   batch {i}: {type(e).__name__} {str(e)[:70]}")
            out += [{}] * len(ch)
        print(f"   ...{min(i+batch, len(texts))}/{len(texts)}", end="\r")
        time.sleep(0.15)
    return out

print("re-extracting with polarity ...")
R = pd.DataFrame(extract(D["txt"].tolist()))
V = D.join(R.add_prefix("v2_"))
V.to_parquet("extract_v2.parquet")

def names(cell, statuses):
    if not isinstance(cell, (list, np.ndarray)):
        return set()
    return {str(d.get("name", "")).upper() for d in cell
            if isinstance(d, dict) and str(d.get("status", "")).lower() in statuses}

print("\n" + "=" * 78)
print("WHAT STATUS DOES IT ASSIGN?")
print("=" * 78)
cnt = {}
for cell in V["v2_drugs"]:
    if isinstance(cell, (list, np.ndarray)):
        for d in cell:
            if isinstance(d, dict):
                cnt[str(d.get("status", "?")).lower()] = cnt.get(
                    str(d.get("status", "?")).lower(), 0) + 1
tot = sum(cnt.values()) or 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"    {k:<14}{v:>6,}  ({v/tot:5.0%})  {'#'*int(v/tot*28)}")
print(f"\n  'excluded' and 'considered' are drugs my old schema counted as prescribed.")

def score(getter, label):
    hit = miss = none = 0
    for c, g in V.groupby("clinic"):
        t = str(g["truth"].iloc[0] or "")
        if not t:
            continue
        nm = set()
        for cell in g["v2_drugs"]:
            nm |= getter(cell)
        if not nm:
            none += 1
        elif any(any(k in n or n in k for k in t.split(", ")) for n in nm):
            hit += 1
        else:
            miss += 1
    n = hit + miss + none
    acc = hit / (hit + miss) if (hit + miss) else float("nan")
    print(f"\n  {label}  ({n} men)")
    print(f"    right {hit:>4} ({hit/n:.0%})   wrong {miss:>4} ({miss/n:.0%})   "
          f"none {none:>4} ({none/n:.0%})")
    print(f"    -> when it named one, right {acc:.0%}")
    return acc

print("\n" + "=" * 78)
print("ACCURACY, SCORING ONLY DRUGS IT SAYS ARE BEING STARTED")
print("=" * 78)
a_all = score(lambda c: names(c, {"starting", "continuing", "considered",
                                  "excluded", "prior", "other"}),
              "ANY drug mentioned (equivalent to the old schema)")
a_start = score(lambda c: names(c, {"starting", "continuing"}),
                "only status = starting/continuing")
print(f"\n  old schema (single drug_named field): 60%")
print(f"  polarity-aware, started/continuing:    {a_start:.0%}")
print(f"  change: {a_start-0.60:+.0%}")

print("\n" + "=" * 78)
print("REASON EXTRACTION -- unchanged schema, should be stable")
print("=" * 78)
V["v2_spec"] = V["v2_reason_is_specific"].fillna(False).astype(bool)
per = V.groupby("clinic")["v2_spec"].max()
print(f"  patient-specific reason, per man: {per.mean():.0%}")
for k, v in V["v2_reason_given"].value_counts().head(6).items():
    print(f"    {str(k):<22}{int(v):>5}  ({v/len(V):5.0%})")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_v2 | notes={len(V)} | acc_any={a_all:.0%} | acc_started={a_start:.0%} "
      f"| old=60% | spec_perman={per.mean():.0%}")

"""
extract_ablate.py -- why did the richer schema make it worse?

  old schema, no thinking   60%
  polarity schema, none     42%   <- adding structure LOST 18 points

The reason fields degraded too (none_stated 21% -> 50%), and I never touched
them. So it is not negation -- it is load. I have had thinking_budget=0 on every
extraction call today. Fine for "sort this clause into a bucket"; probably not
fine for "read a messy note, decide each drug's status, and separately identify
the clinical rationale".

2x2 on the SAME notes:
              simple schema      polarity schema
  no think    (have it: 60%)     (have it: 42%)
  thinking    ?                  ?

If thinking rescues the polarity schema, the fix was right and I starved it.
If simple+thinking wins outright, the schema was the mistake and less is more.
Either way we stop guessing.
"""
import json, os, time
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
X = pd.read_parquet("browse_cache.parquet")
X["dec"] = X["mentions_cancer_treatment_decision"].fillna(False).astype(bool)
D = X[X["dec"]].reset_index(drop=True).head(140)
print(f"{len(D)} flagged decision notes, {D['clinic'].nunique()} men")

SIMPLE = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>,
 "drug_started": "<the generic drug this patient is being STARTED on, or null>",
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
                 "patient_preference"|"toxicity_concern"|"guideline_standard"|
                 "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false,
 "evidence_quote": "<exact supporting sentence, or null>"}
drug_started: null if the note only discusses options, continues an existing
drug, or rules one out. A drug that is contraindicated is NOT started."""

POLAR = """Return ONLY a JSON array, one object per numbered note:
{"i": <index>,
 "drugs": [{"name": "<generic name>",
            "status": "starting"|"continuing"|"considered"|"excluded"|"prior"}],
 "reason_given": "disease_extent"|"performance_status"|"comorbidity"|
                 "patient_preference"|"toxicity_concern"|"guideline_standard"|
                 "trial_enrollment"|"none_stated",
 "reason_is_specific": true|false,
 "evidence_quote": "<exact supporting sentence, or null>"}
"excluded" = ruled out (contraindication, allergy, refusal). Never mark a
ruled-out drug as starting."""

def run(schema, think, tag, batch=5):
    from google import genai
    from google.genai import types
    cl = genai.Client(vertexai=True,
                      project=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                             "mcp-acc-055-dbg-p-7e23"),
                      location="global")
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=think),
        response_mime_type="application/json", temperature=0)
    out, T = [], D["txt"].tolist()
    t0 = time.time()
    for i in range(0, len(T), batch):
        ch = T[i:i + batch]
        listed = "\n\n".join(f"### NOTE {j}\n{t[:4000]}" for j, t in enumerate(ch))
        try:
            r = cl.models.generate_content(model="gemini-3.5-flash",
                contents=f"Read these oncology notes.\n\n{schema}\n\n{listed}",
                config=cfg)
            got = {int(d["i"]): d for d in json.loads(r.text) if "i" in d}
            out += [got.get(j, {}) for j in range(len(ch))]
        except Exception as e:
            out += [{}] * len(ch)
        print(f"   {tag} ...{min(i+batch, len(T))}/{len(T)}", end="\r")
        time.sleep(0.1)
    print(f"   {tag} done in {time.time()-t0:.0f}s" + " " * 20)
    return pd.DataFrame(out)

def drugs_from(row, polar):
    if polar:
        c = row.get("drugs")
        if not isinstance(c, (list, np.ndarray)):
            return set()
        return {str(d.get("name", "")).upper() for d in c
                if isinstance(d, dict)
                and str(d.get("status", "")).lower() in ("starting", "continuing")}
    v = row.get("drug_started")
    if v is None or str(v).upper() in ("NONE", "NAN", "NULL", ""):
        return set()
    return {str(v).upper()}

def score(R, polar, label):
    V = D.join(R.add_prefix("r_"))
    R2 = R.copy()
    hit = miss = none = 0
    for c, g in V.groupby("clinic"):
        t = str(g["truth"].iloc[0] or "")
        if not t:
            continue
        nm = set()
        for _, row in g.iterrows():
            nm |= drugs_from({k[2:]: v for k, v in row.items()
                              if k.startswith("r_")}, polar)
        if not nm:
            none += 1
        elif any(any(k in n or n in k for k in t.split(", ")) for n in nm):
            hit += 1
        else:
            miss += 1
    acc = hit / (hit + miss) if (hit + miss) else float("nan")
    spec = R.get("reason_is_specific", pd.Series(dtype=object)).fillna(False).astype(bool)
    nonestated = R.get("reason_given", pd.Series(dtype=object)).fillna(
        "none_stated").eq("none_stated")
    print(f"  {label:<34}drug {acc:>5.0%}   specific {spec.mean():>5.0%}   "
          f"none_stated {nonestated.mean():>5.0%}")
    return acc

print("\nrunning the 2x2 (a few minutes) ...")
r1 = run(SIMPLE, 0, "simple/nothink")
r2 = run(SIMPLE, 4096, "simple/think")
r3 = run(POLAR, 0, "polar/nothink")
r4 = run(POLAR, 4096, "polar/think")

print("\n" + "=" * 78)
print("RESULTS -- same 140 notes, same model, same truth")
print("=" * 78)
a1 = score(r1, False, "simple schema, no thinking")
a2 = score(r2, False, "simple schema, THINKING")
a3 = score(r3, True,  "polarity schema, no thinking")
a4 = score(r4, True,  "polarity schema, THINKING")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
best = max([(a1, "simple/nothink"), (a2, "simple/think"),
            (a3, "polar/nothink"), (a4, "polar/think")],
           key=lambda x: (x[0] if x[0] == x[0] else -1))
print(f"  best: {best[1]} at {best[0]:.0%}")
print(f"  thinking helps simple schema:   {a2-a1:+.0%}")
print(f"  thinking helps polarity schema: {a4-a3:+.0%}")
print(f"  polarity helps (with thinking): {a4-a2:+.0%}")
print("\n  if thinking rescues polarity, I starved it. if simple wins even with")
print("  thinking, the schema itself was the mistake and less is more.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_ablate | simple_nothink={a1:.0%} | simple_think={a2:.0%} "
      f"| polar_nothink={a3:.0%} | polar_think={a4:.0%} | best={best[1]}")

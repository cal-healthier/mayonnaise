"""
miss_audit.py -- the 25 wrong-drug cases are inside real decision notes.
Filtering did not fix accuracy (60% either way), so my explanation was wrong.

Two things here:
  A. SELF-CONSISTENCY, checkable with no ground truth. The model calls 75% of
     flagged notes "patient-specific" while assigning `guideline_standard` to
     18% of them. Those cannot both be true. Same for notes marked specific
     while reason_given = none_stated. Contradiction rate is a free accuracy
     signal.
  B. A BROWSER pointed only at wrong-drug cases INSIDE decision notes, showing
     the model's named drug, the drug actually given, and the evidence quote,
     so you can see in one screen whether it misread or whether my truth is
     wrong.

  miss()          next wrong-drug case
  miss(full=True) whole note
  contra()        next self-contradictory case

>>> STAYS IN THE ENCLAVE. <<<
"""
import re, textwrap, random
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
X = pd.read_parquet("browse_cache.parquet")
X["dec"] = X["mentions_cancer_treatment_decision"].fillna(False).astype(bool)
X["spec"] = X["reason_is_specific"].fillna(False).astype(bool)
X["reason"] = X["reason_given"].fillna("none_stated")
X["named"] = X["drug_named"].notna() & ~X["drug_named"].astype(str).str.upper().isin(
    ["NONE", "NAN", "NULL", ""])
print(f"{len(X):,} notes | {X['clinic'].nunique()} men")

print("\n" + "=" * 76)
print("A. SELF-CONSISTENCY  (no ground truth needed)")
print("=" * 76)
d = X[X["dec"]].copy()
c1 = d["spec"] & d["reason"].eq("guideline_standard")
c2 = d["spec"] & d["reason"].eq("none_stated")
c3 = ~d["spec"] & d["reason"].isin(["patient_preference", "comorbidity",
                                    "toxicity_concern"])
c4 = d["named"] & ~d["dec"]
print(f"  flagged decision notes: {len(d):,}\n")
for lbl, mask in [("marked specific BUT reason = guideline_standard", c1),
                  ("marked specific BUT reason = none_stated", c2),
                  ("marked NOT specific BUT reason is inherently specific", c3)]:
    print(f"  {lbl:<52}{int(mask.sum()):>5}  ({mask.mean():>5.0%})")
contra = c1 | c2 | c3
print(f"\n  any contradiction: {int(contra.sum()):,} of {len(d):,} "
      f"({contra.mean():.0%})")
print("  a low rate means the model is at least internally coherent; a high one")
print("  means the 'specific' flag is decorative and cannot be trusted.")

ev = d["evidence_quote"].astype(str)
noev = d["spec"] & (ev.isin(["nan", "None", ""]) | ev.str.len().lt(15))
print(f"\n  claimed specific but gave no usable evidence quote: "
      f"{int(noev.sum())} ({noev.mean():.0%})")
print("  that one is close to a hallucination check -- a specific reason with")
print("  nothing quoted to support it.")

# ---------------------------------------------------------------- browsers
def _cat(r):
    named = str(r.get("drug_named") or "").upper()
    t = str(r.get("truth") or "")
    if not named or named in ("NONE", "NAN", "NULL"):
        return "none"
    if t and any(k in named or named in k for k in t.split(", ")):
        return "right"
    return "wrong" if t else "unknown"
X["dcat"] = X.apply(_cat, axis=1)
MISS = X[(X["dec"]) & (X["dcat"] == "wrong")].to_dict("records")
CON = d[contra | noev].to_dict("records")
random.shuffle(MISS); random.shuffle(CON)
print(f"\n  queued: {len(MISS)} wrong-drug cases, {len(CON)} contradictory cases")

KEY = re.compile(r"(leuprolide|lupron|eligard|goserelin|zoladex|degarelix|firmagon|"
                 r"bicalutamide|casodex|abiraterone|enzalutamide|triptorelin|"
                 r"adt|androgen|hormone|orchiectomy|radiation|systemic|elected|"
                 r"recommend|proceed|plan)", re.I)

def _render(r, full, tag):
    print("\n" + "=" * 74)
    print(f"{tag}   {r['ttl']}   note {r['dt']}   therapy started {r['tx']}")
    print("=" * 74)
    print("  MODEL SAID")
    print(f"    drug named      {r.get('drug_named')!r}")
    print(f"    ACTUALLY given  {r.get('truth') or '(none recorded)'}")
    print(f"    reason          {r.get('reason_given')}   "
          f"specific={r.get('reason_is_specific')}")
    q = r.get("evidence_quote")
    if q and str(q) != "nan":
        print("    quoted evidence:")
        for l in textwrap.wrap(str(q), 64):
            print(f"      > {l}")
    print("\n  NOTE" + (" (full)" if full else " (drug / decision sentences)"))
    if full:
        body = str(r["txt"])
    else:
        sents = re.split(r"(?<=[.!?])\s+", str(r["txt"]))
        hits = [s for s in sents if KEY.search(s)]
        body = "\n".join(hits[:8]) or "\n".join(sents[:6])
    for para in body.split("\n"):
        for l in textwrap.wrap(para, 70) or [""]:
            print(f"    {l}")

_mi = _ci = 0
def miss(full=False):
    global _mi
    if _mi >= len(MISS):
        print("  end of wrong-drug pool"); return
    r = MISS[_mi]; _mi += 1
    _render(r, full, f"WRONG DRUG {_mi}/{len(MISS)}")
    print("\n  -> miss() next | miss(full=True) | contra()")

def contra(full=False):
    global _ci
    if _ci >= len(CON):
        print("  end of contradiction pool"); return
    r = CON[_ci]; _ci += 1
    _render(r, full, f"CONTRADICTION {_ci}/{len(CON)}")
    print("\n  -> contra() next | contra(full=True) | miss()")

print("\n  call  miss()  in a new cell to start reading.")
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"miss_audit | decision_notes={len(d)} | contradiction={contra.mean():.0%} "
      f"| specific_no_evidence={noev.mean():.0%} | wrong_drug_cases={len(MISS)}")

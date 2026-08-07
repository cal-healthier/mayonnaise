"""
rad_controls.py -- the 10 controls that got flagged. Free: already extracted.

77% of progressors vs 30% of controls is a strong separation, but 30% is not
nothing. Three explanations, and they are very different:
  a) the model misread something -> a real error rate
  b) prior disease described as "new" -> a fixable prompt problem
  c) PSA genuinely missed progression -> those are TRUE positives, and the
     reader is catching what the blood marker cannot

(c) would be the most interesting outcome in the whole study.

Also fixes the timing caveat as far as free analysis allows: my query took the
8 reports CLOSEST to the PSA date, so alignment was partly built in. Here we at
least measure how much of the window the reports actually span.

  ctrl()   read the flagged control cases yourself
No new API calls.
"""
import re, textwrap
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
X = pd.read_parquet("rad_extracted.parquet")
E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
X["prog_flag"] = X["direction"].isin(["worse", "new_lesion"]) & \
                 X["is_cancer_assessment"].fillna(False).astype(bool)
X["mo"] = (X["dt"] - X["anch"]).dt.days / 30.44
X["is_prog"] = X["clinic"].map(E["prog"])
print(f"{len(X):,} reports | {X['clinic'].nunique()} men")

print("\n" + "=" * 76)
print("1. HOW MUCH OF THE WINDOW DO THE REPORTS ACTUALLY COVER?")
print("=" * 76)
sp = X.groupby("clinic")["mo"].agg(["min", "max", "size"])
print(f"  intended window: -12.0 to +6.0 months around the PSA date")
print(f"  actual span per man: median {(sp['max']-sp['min']).median():.1f} months")
print(f"    earliest report  median {sp['min'].median():+.1f} mo")
print(f"    latest report    median {sp['max'].median():+.1f} mo")
frac = ((X["mo"].abs() <= 3).mean())
print(f"  reports falling within +/-3 months of the anchor: {frac:.0%}")
print("\n  if most reports sit near the anchor, the median-zero alignment was")
print("  largely my sampling, not the model's timing. a wide span means less so.")

print("\n" + "=" * 76)
print("2. THE FLAGGED CONTROLS -- who are they?")
print("=" * 76)
ctrl = X[X["is_prog"] == 0]
flag = ctrl.groupby("clinic")["prog_flag"].max()
hit = flag[flag].index
print(f"  controls: {flag.size} men | flagged: {len(hit)} ({flag.mean():.0%})")

sub = E.loc[E.index.isin(hit)].copy()
allc = E.loc[E.index.isin(flag.index)].copy()
print(f"\n  follow-up after treatment (months):")
print(f"    flagged controls    median {sub['time'].median()/30.44:>6.1f}"
      f"   n={len(sub)}")
print(f"    all controls        median {allc['time'].median()/30.44:>6.1f}"
      f"   n={len(allc)}")
print(f"\n  baseline PSA:")
print(f"    flagged controls    median {sub['baseline'].median():>6.1f}")
print(f"    all controls        median {allc['baseline'].median():>6.1f}")
print(f"\n  PSA nadir reached:")
print(f"    flagged controls    median {sub['nadir'].median():>6.2f}")
print(f"    all controls        median {allc['nadir'].median():>6.2f}")
print("\n  if flagged controls have higher baseline/nadir PSA or shorter")
print("  follow-up, they are probably men whose disease WAS progressing and")
print("  whose PSA had not yet crossed the formal threshold -> true positives.")

print("\n" + "=" * 76)
print("3. HOW CONFIDENT WAS IT ON THOSE?")
print("=" * 76)
fc = ctrl[ctrl["prog_flag"]]
fp = X[(X["is_prog"] == 1) & X["prog_flag"]]
for lbl, d in [("flagged controls", fc), ("flagged progressors", fp)]:
    if len(d):
        print(f"  {lbl:<24}confident {d['confident'].fillna(False).mean():>5.0%}"
              f"   new_lesion {(d['direction']=='new_lesion').mean():>5.0%}"
              f"   worse {(d['direction']=='worse').mean():>5.0%}")
print("\n  lower confidence on controls would suggest genuine uncertainty rather")
print("  than confident error.")

CASES = fc.to_dict("records")
_i = 0
def ctrl_case(full=False):
    """read a flagged control report"""
    global _i
    if _i >= len(CASES):
        print("  end of pool"); return
    r = CASES[_i]; _i += 1
    print("\n" + "=" * 74)
    print(f"FLAGGED CONTROL {_i}/{len(CASES)}   {r.get('modality')}   "
          f"{r['dt'].date()}   ({r['mo']:+.1f} mo from censor)")
    print("=" * 74)
    print(f"  direction   {r.get('direction')}   confident={r.get('confident')}")
    q = r.get("evidence_quote")
    if q and str(q) != "nan":
        print("  quoted:")
        for l in textwrap.wrap(str(q), 66):
            print(f"    > {l}")
    print("\n  REPORT" + (" (full)" if full else " (excerpt)"))
    body = str(r["txt"]) if full else str(r["txt"])[:1200]
    for para in body.split("\n"):
        for l in textwrap.wrap(para, 70) or [""]:
            print(f"    {l}")
    print("\n  -> ctrl_case() next | ctrl_case(full=True)")

print(f"\n  {len(CASES)} flagged control reports queued.")
print("  call  ctrl_case()  in a new cell to read them.")
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"rad_controls | ctrl_men={flag.size} | flagged={len(hit)} "
      f"| within3mo_of_anchor={frac:.0%} "
      f"| span_months={(sp['max']-sp['min']).median():.1f} "
      f"| ctrl_conf={fc['confident'].fillna(False).mean():.0%}")

"""
funnel.py -- is 23% of criteria worth anything?

Three-valued logic. A patient is DROPPED only when they demonstrably FAIL a
checkable criterion. Missing data = UNKNOWN = patient stays in (conservative:
they'd go to human review). So this measures the one thing partial matching
CAN do: rule people out.

Inclusion vs exclusion criteria invert -- handled explicitly below, it is the
easiest thing in the world to get backwards.

Step 0 discovers what cohort files are actually on disk before assuming any.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 0. what do we have?
print("=" * 72)
print("FILES ON DISK")
print("=" * 72)
cands = sorted(glob.glob("*.parquet") + glob.glob("*.csv"))
for f in cands:
    try:
        n = len(pd.read_parquet(f)) if f.endswith(".parquet") else sum(1 for _ in open(f)) - 1
        print(f"  {f:<44}{n:>9,} rows")
    except Exception as e:
        print(f"  {f:<44}  (unreadable: {type(e).__name__})")

def load_first(names):
    for n in names:
        if os.path.exists(n):
            try:
                return (pd.read_parquet(n) if n.endswith(".parquet") else pd.read_csv(n)), n
            except Exception:
                pass
    return None, None

cohort, cname = load_first(["cohort.parquet", "cohort_meta.parquet", "meta.parquet",
                            "cohort.csv", "patients.parquet"])
tumor, tname = load_first(["tumor.parquet", "tumor_features.parquet", "T.parquet"])
labs,  lname = load_first(["labs.parquet", "lab_cube.parquet", "features.parquet",
                           "feat.parquet"])
print(f"\n  cohort <- {cname}\n  tumor  <- {tname}\n  labs   <- {lname}")
if cohort is None:
    print("\n  No cohort file found. Re-run the cohort extraction first, then this.")
    print("\n" + "-" * 70)
    print("FINAL LINE:")
    print(f"funnel | status=no_cohort_file | files={len(cands)}")
    raise SystemExit

# ---------------------------------------------------------------- 1. patient attributes
P = cohort.copy()
if tumor is not None:
    P = P.join(tumor, how="left") if tumor.index.name == P.index.name else P.merge(
        tumor, left_index=True, right_index=True, how="left")
if labs is not None:
    P = P.merge(labs, left_index=True, right_index=True, how="left")
P.columns = [str(c).lower() for c in P.columns]
print(f"\npatient table: {len(P):,} rows x {P.shape[1]} cols")

# map rule field -> a column we actually have (labs use their 'last' summary)
def find_col(field):
    f = field.lower()
    if f in P.columns:
        return f
    for suffix in ("__last", "__mean", "__min", "__max"):
        if f + suffix in P.columns:
            return f + suffix
    for c in P.columns:                       # loose contains match, last resort
        if f in c:
            return c
    return None

# ---------------------------------------------------------------- 2. rules
R = pd.read_csv("executable_rules.csv")
R = R[R.solid == True].copy()
R["col"] = R.field.map(find_col)
mapped = R.col.notna()
print(f"executable rules: {len(R)}   mapped to a real column: {mapped.sum()} "
      f"({mapped.mean():.0%})")
print("\n  unmapped fields (rule exists, we have no column):")
for f, n in R.loc[~mapped, "field"].value_counts().head(12).items():
    print(f"    {f:<26}{n:>4}")
R = R[mapped]

NUM = re.compile(r"-?\d+\.?\d*")
def evaluate(rule, series):
    """-> Series of 'pass' / 'fail' / 'unknown' for an INCLUSION reading."""
    op, raw = str(rule.op), str(rule.value)
    known = series.notna()
    res = pd.Series("unknown", index=series.index)
    if op == "exists":
        return pd.Series(np.where(known, "pass", "fail"), index=series.index)
    if op == "absent":
        return pd.Series(np.where(known, "fail", "pass"), index=series.index)
    m = NUM.search(raw)
    if op in (">=", "<=", ">", "<") and m:
        v = float(m.group())
        s = pd.to_numeric(series, errors="coerce")
        ok = {">=": s >= v, "<=": s <= v, ">": s > v, "<": s < v}[op]
        return pd.Series(np.where(s.notna(), np.where(ok, "pass", "fail"), "unknown"),
                         index=series.index)
    if op in ("==", "!=", "in"):
        s = series.astype(str).str.lower()
        vals = [w.strip().lower() for w in re.split(r"[,/|]| or ", raw) if w.strip()]
        hit = s.apply(lambda x: any(v in x or x in v for v in vals if v and v != "nan"))
        if op == "!=":
            hit = ~hit
        return pd.Series(np.where(known, np.where(hit, "pass", "fail"), "unknown"),
                         index=series.index)
    return res

# ---------------------------------------------------------------- 3. run the funnel
rows = []
for nct, grp in R.groupby("nct"):
    alive = pd.Series(True, index=P.index)
    n_applied = n_skipped = 0
    for _, rule in grp.iterrows():
        verdict = evaluate(rule, P[rule.col])
        if (verdict == "unknown").all():
            n_skipped += 1
            continue
        n_applied += 1
        # inclusion: drop those who FAIL.  exclusion: drop those who PASS (i.e. match it)
        drop = (verdict == "fail") if rule.kind == "inclusion" else (verdict == "pass")
        alive &= ~drop
    rows.append({"nct": nct, "rules": len(grp), "applied": n_applied,
                 "skipped": n_skipped, "survivors": int(alive.sum()),
                 "frac": alive.mean()})
F = pd.DataFrame(rows).sort_values("survivors")

print("\n" + "=" * 72)
print(f"FUNNEL: {len(P):,} patients -> candidates per trial")
print("=" * 72)
print(f"  trials evaluated:            {len(F)}")
print(f"  median survivors per trial:  {F.survivors.median():,.0f} "
      f"({F.frac.median():.1%} of cohort)")
print(f"  median rules applied:        {F.applied.median():.0f} of "
      f"{F.rules.median():.0f} extracted")
print(f"  trials narrowing to <5%:     {(F.frac < .05).sum()}/{len(F)}")
print(f"  trials narrowing to <1%:     {(F.frac < .01).sum()}/{len(F)}")
print(f"  trials that drop nobody:     {(F.frac == 1).sum()}/{len(F)}")

print("\n  tightest 8 trials:")
for _, r in F.head(8).iterrows():
    print(f"    {r.nct}  {r.survivors:>7,} survivors ({r.frac:5.1%})  "
          f"{r.applied} rules applied")
print("\n  loosest 5 trials:")
for _, r in F.tail(5).iterrows():
    print(f"    {r.nct}  {r.survivors:>7,} survivors ({r.frac:5.1%})  "
          f"{r.applied} rules applied")

F.to_csv("funnel_results.csv", index=False)
print("\nwrote funnel_results.csv")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"funnel | cohort={len(P)} | trials={len(F)} "
      f"| median_survivors={F.survivors.median():.0f} "
      f"| median_frac={F.frac.median():.1%} | under5pct={(F.frac<.05).sum()} "
      f"| mapped_rules={mapped.sum()}")

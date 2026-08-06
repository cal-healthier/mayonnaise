"""
extract_eval.py -- score the extraction on the notes it says are relevant.

The 51% drug accuracy was my error. Reading the cases showed the "misses" were
vascular clinic notes, dermatology visits, primary care -- notes with no cancer
decision in them. The model correctly flagged them (decision=False), named no
drug, stated no reason. I then scored it down for being right.

It even declined to name TESTOSTERONE CYPIONATE from a med list -- the opposite
of androgen deprivation, and an easy wrong answer.

So the honest evaluation has two parts:
  A. TRIAGE -- how well does the decision flag separate relevant from
     irrelevant notes? (this is the valuable capability at 500M-document scale)
  B. EXTRACTION -- among notes it flagged as decisions, is the named drug right
     and is the reason real?

Reads browse_cache.parquet. No BigQuery, no new extraction.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
X = pd.read_parquet("browse_cache.parquet")
X["dec"] = X["mentions_cancer_treatment_decision"].fillna(False).astype(bool)
X["spec"] = X["reason_is_specific"].fillna(False).astype(bool)
X["hasreason"] = X["reason_given"].fillna("none_stated").ne("none_stated")
X["named"] = X["drug_named"].notna() & ~X["drug_named"].astype(str).str.upper().isin(
    ["NONE", "NAN", "NULL", ""])
print(f"{len(X):,} notes | {X['clinic'].nunique()} men")

print("\n" + "=" * 78)
print("A. TRIAGE -- can it find the notes that matter?")
print("=" * 78)
print(f"  flagged as a cancer treatment decision: {int(X['dec'].sum()):,} of "
      f"{len(X):,}  ({X['dec'].mean():.0%})")
print(f"\n  {'':<22}{'flagged decision':>18}{'not flagged':>14}")
for lbl, col in [("names a drug", "named"), ("gives a reason", "hasreason"),
                 ("reason is specific", "spec"),
                 ("has performance status", None)]:
    if col is None:
        a = X.loc[X["dec"], "performance_status"].notna().mean()
        b = X.loc[~X["dec"], "performance_status"].notna().mean()
    else:
        a, b = X.loc[X["dec"], col].mean(), X.loc[~X["dec"], col].mean()
    print(f"  {lbl:<22}{a:>18.0%}{b:>14.0%}")
print("\n  a big gap between the columns means the flag is separating real")
print("  decision notes from vascular clinic and dermatology visits.")

print("\n  what note types get flagged?")
t = X.groupby("ttl").agg(notes=("dec", "size"), flagged=("dec", "mean")).sort_values(
    "notes", ascending=False)
for k, r in t.head(8).iterrows():
    print(f"    {str(k)[:28]:<30}{int(r['notes']):>7,}{r['flagged']:>8.0%}")

print("\n" + "=" * 78)
print("B. EXTRACTION -- accuracy AMONG the notes it flagged")
print("=" * 78)
def score(df, label):
    hit = miss = nodrug = 0
    for c, g in df.groupby("clinic"):
        t = str(g["truth"].iloc[0] or "")
        if not t:
            continue
        names = {str(d).upper() for d in g.loc[g["named"], "drug_named"]}
        if not names:
            nodrug += 1
        elif any(any(k in n or n in k for k in t.split(", ")) for n in names):
            hit += 1
        else:
            miss += 1
    tot = hit + miss + nodrug
    if not tot:
        print(f"  {label}: no men to score"); return np.nan
    print(f"\n  {label}  ({tot} men)")
    print(f"    named the right drug   {hit:>4} ({hit/tot:.0%})")
    print(f"    named the wrong drug   {miss:>4} ({miss/tot:.0%})")
    print(f"    named no drug          {nodrug:>4} ({nodrug/tot:.0%})")
    if hit + miss:
        print(f"    -> when it named one, right {hit/(hit+miss):.0%} of the time")
        return hit / (hit + miss)
    return np.nan
all_acc = score(X, "ALL notes (my original, flawed test)")
dec_acc = score(X[X["dec"]], "ONLY notes it flagged as decisions")

print("\n" + "=" * 78)
print("C. PER-MAN COVERAGE using flagged notes only")
print("=" * 78)
d = X[X["dec"]]
per = d.groupby("clinic").agg(reason=("hasreason", "max"), spec=("spec", "max"),
                              perf=("performance_status", lambda s: s.notna().any()),
                              n=("dec", "size"))
men_any_dec = d["clinic"].nunique()
print(f"  men with >=1 flagged decision note: {men_any_dec} of "
      f"{X['clinic'].nunique()}  ({men_any_dec/X['clinic'].nunique():.0%})")
print(f"  median flagged notes per such man: {per['n'].median():.0f}")
print(f"\n  among those men:")
print(f"    a reason is stated        {per['reason'].mean():.0%}")
print(f"    reason is patient-specific {per['spec'].mean():.0%}")
print(f"    performance status given  {per['perf'].mean():.0%}")

print("\n" + "=" * 78)
print("D. WHAT REASONS DO THE REAL DECISION NOTES GIVE?")
print("=" * 78)
for k, v in d["reason_given"].value_counts().items():
    print(f"    {str(k):<22}{int(v):>6,}  ({v/len(d):5.0%})  {'#'*int(v/len(d)*26)}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"extract_eval | notes={len(X)} | flagged={X['dec'].mean():.0%} "
      f"| acc_all={all_acc:.0%} | acc_flagged={dec_acc:.0%} "
      f"| men_with_decision={men_any_dec} | spec_perman={per['spec'].mean():.0%}")

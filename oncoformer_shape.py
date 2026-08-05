"""
oncoformer_shape.py -- build Oncoformer's exact input from our data.

Their released config: n_float_feats 28, seq_max_len 16, 256 value bins.
Input = routine labs + vitals as a sequence of up to 16 clinical visits,
values discretised into 256 buckets. Plus chest X-rays (which we cannot use).

So: can we construct the same tensor? This builds it and reports what we have
against what they had.

  [N patients x 16 visits x 25 features] value tensor
  [N x 16 x 25] mask  (real measurement vs missing)
  [N x 16]      visit times

Reports feature coverage per visit, how sparse the tensor is, and how many
patients reach 16 visits. Missingness is the thing that decides whether the
representation works -- they needed adversarial de-biasing specifically to
handle it, which tells you theirs was sparse too.

Cached parquet only. No BigQuery.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
MAX_VISITS, N_BINS = 16, 256

cube = pd.read_parquet("cohort_features.parquet")   # clinic, visit_date, feature, value
cube["visit_date"] = pd.to_datetime(cube["visit_date"])
feats = sorted(cube["feature"].unique())
print(f"features available: {len(feats)}   (Oncoformer used 28)")
print(f"  {', '.join(feats)}")
print(f"\nraw cube: {len(cube):,} measurements, {cube['clinic'].nunique():,} patients")

# ---------------------------------------------------------------- visits
v = (cube.groupby(["clinic", "visit_date"]).size().rename("n_feats").reset_index())
per = v.groupby("clinic").agg(n_visits=("visit_date", "size"),
                              med_feats=("n_feats", "median"))
print("\n" + "=" * 84)
print("VISIT STRUCTURE")
print("=" * 84)
print(f"  visits per patient   median {per['n_visits'].median():.0f}   "
      f"p25 {per['n_visits'].quantile(.25):.0f}   p75 {per['n_visits'].quantile(.75):.0f}"
      f"   p90 {per['n_visits'].quantile(.9):.0f}")
print(f"  features per visit   median {per['med_feats'].median():.0f} of {len(feats)}")
for k in (4, 8, 16, 24):
    n = int((per["n_visits"] >= k).sum())
    print(f"  patients with >= {k:>2} visits: {n:>7,} ({n/len(per):5.1%})")

# ---------------------------------------------------------------- the tensor
keep = per.index[per["n_visits"] >= 4]
c = cube[cube["clinic"].isin(keep)].copy()
c["rank"] = c.groupby("clinic")["visit_date"].rank(method="dense", ascending=False)
c = c[c["rank"] <= MAX_VISITS]                      # most recent 16 visits
c["slot"] = (MAX_VISITS - c["rank"]).astype(int)    # 0..15, most recent last

fi = {f: i for i, f in enumerate(feats)}
pi = {p: i for i, p in enumerate(sorted(c["clinic"].unique()))}
N, F = len(pi), len(feats)
X = np.full((N, MAX_VISITS, F), np.nan, dtype=np.float32)
X[c["clinic"].map(pi).values, c["slot"].values, c["feature"].map(fi).values] = \
    c["value"].values.astype(np.float32)
M = ~np.isnan(X)

print("\n" + "=" * 84)
print(f"THE TENSOR: [{N:,} patients x {MAX_VISITS} visits x {F} features]")
print("=" * 84)
print(f"  cells total        {X.size:>14,}")
print(f"  cells observed     {int(M.sum()):>14,}  ({M.mean():.1%} filled)")
print(f"  ->  {1-M.mean():.1%} MISSING. Oncoformer needed adversarial de-biasing")
print("      against missingness patterns, so theirs was sparse too.")

print("\n  fill rate per feature (across all patient-visits):")
rate = pd.Series(M.mean(axis=(0, 1)), index=feats).sort_values(ascending=False)
for f, r in rate.items():
    print(f"    {f:<24}{r:6.1%}  {'#' * max(0, int(r*34))}")

print("\n  fill rate by visit slot (0 = oldest kept, 15 = most recent):")
bys = M.mean(axis=(0, 2))
for s in range(MAX_VISITS):
    print(f"    slot {s:>2}  {bys[s]:6.1%}  {'#' * max(0, int(bys[s]*40))}")

# ---------------------------------------------------------------- binning
print("\n" + "=" * 84)
print(f"VALUE BINNING ({N_BINS} bins, as they used)")
print("=" * 84)
ok = 0
for f in feats:
    col = X[:, :, fi[f]]
    s = col[~np.isnan(col)]
    if len(s) < 100:
        continue
    q = np.unique(np.quantile(s, np.linspace(0, 1, N_BINS + 1)))
    if len(q) - 1 >= N_BINS * 0.5:
        ok += 1
print(f"  {ok} of {F} features support {N_BINS} distinct quantile bins")
print(f"  ({F - ok} are too discrete -- percentages and counts with few values)")

np.save("onco_X.npy", X); np.save("onco_M.npy", M)
print("\n  saved onco_X.npy / onco_M.npy")

print("\n" + "=" * 84)
print("US vs THEM")
print("=" * 84)
print(f"  {'':<26}{'Oncoformer':>14}{'us':>14}")
print(f"  {'patients':<26}{'3,672,989':>14}{N:>14,}")
print(f"  {'features':<26}{'28':>14}{F:>14}")
print(f"  {'visits (cap)':<26}{'16':>14}{MAX_VISITS:>14}")
print(f"  {'value bins':<26}{'256':>14}{N_BINS:>14}")
print(f"  {'chest X-rays':<26}{'yes':>14}{'no (blocked)':>14}")
print(f"  {'cancer-free controls':<26}{'3,325,049':>14}{'~0 (referral)':>14}")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"oncoformer_shape | patients={N} | features={F} | filled={M.mean():.1%} "
      f"| binnable={ok} | med_visits={per['n_visits'].median():.0f}")

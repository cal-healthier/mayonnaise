"""
relative_encoding.py -- test the premise the whole architecture rests on.

Cross-cancer transfer only works if PSA-under-hormone-therapy and
CA-125-under-platinum occupy the SAME space once encoded relatively.
Absolute values cannot (PSA ~4, CA-125 ~500). The claim is that
  log(value / baseline), log(value / running nadir), and velocity
make them comparable.

If the distributions overlap, one model can learn both and transfer is
plausible. If they are disjoint, transfer is dead and the world-model framing
collapses to two separate disease models -- worth knowing now, not after
building an encoder.

Also checks whether the RESPONSE SHAPE is shared: fall to nadir, plateau,
escape. Same shape in both cancers = same underlying process.

Cached parquet only. No BigQuery, no cost.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240)

def build(marker_file, label_file, val_col, name):
    v = pd.read_parquet(marker_file)
    L = pd.read_parquet(label_file)
    v = v[v["clinic"].isin(L.index)].copy()
    v = v.sort_values(["clinic", "day"])
    post = v[v["day"] > 0].copy()
    base = L["baseline"].reindex(post["clinic"]).values
    post["base"] = base
    post["nadir_run"] = post.groupby("clinic")[val_col].cummin()
    eps = 1e-3
    post["log_vs_base"] = np.log((post[val_col] + eps) / (post["base"] + eps))
    post["log_vs_nadir"] = np.log((post[val_col] + eps) / (post["nadir_run"] + eps))
    g = post.groupby("clinic")
    post["dt"] = g["day"].diff()
    post["dlog"] = g["log_vs_base"].diff()
    post["velocity"] = post["dlog"] / (post["dt"] / 30.4)      # per month
    post["cancer"] = name
    return post

P = build("psa_values.parquet", "psa_progression.parquet", "psa", "prostate/PSA")
O = build("ov_ca125.parquet", "ov_label.parquet", "ca", "ovarian/CA-125")
print(f"prostate {P['clinic'].nunique():,} men, {len(P):,} post-treatment draws")
print(f"ovarian  {O['clinic'].nunique():,} women, {len(O):,} post-treatment draws")

print("\n" + "=" * 88)
print("ABSOLUTE VALUES -- incomparable, as expected")
print("=" * 88)
for d, n in ((P, "prostate/PSA"), (O, "ovarian/CA-125")):
    col = "psa" if "psa" in d.columns else "ca"
    q = d[col].quantile([.1, .5, .9]).values
    print(f"  {n:<18}p10 {q[0]:>10.2f}   median {q[1]:>10.2f}   p90 {q[2]:>10.2f}")

print("\n" + "=" * 88)
print("RELATIVE ENCODINGS -- do they land in the same space?")
print("=" * 88)
for feat, lbl in [("log_vs_base", "log(value / baseline)"),
                  ("log_vs_nadir", "log(value / running nadir)"),
                  ("velocity", "velocity  (log change per month)")]:
    print(f"\n  {lbl}")
    print(f"    {'':<18}{'p10':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p90':>9}")
    stats = {}
    for d, n in ((P, "prostate/PSA"), (O, "ovarian/CA-125")):
        s = d[feat].replace([np.inf, -np.inf], np.nan).dropna()
        q = s.quantile([.1, .25, .5, .75, .9]).values
        stats[n] = q
        print(f"    {n:<18}" + "".join(f"{x:>9.2f}" for x in q))
    a, b = stats["prostate/PSA"], stats["ovarian/CA-125"]
    # overlap of the interquartile ranges
    lo = max(a[1], b[1]); hi = min(a[3], b[3])
    span = max(a[3], b[3]) - min(a[1], b[1])
    ov = max(0.0, hi - lo) / span if span > 0 else 0.0
    print(f"    -> IQR overlap {ov:.0%}  "
          f"{'GOOD - same space' if ov > 0.5 else 'POOR - transfer unlikely'}")

print("\n" + "=" * 88)
print("IS THE RESPONSE SHAPE SHARED?  (median log-vs-baseline by month)")
print("=" * 88)
print(f"  {'month':>6}{'prostate':>12}{'ovarian':>12}   trajectory")
for m in range(0, 25, 2):
    row = []
    for d in (P, O):
        s = d[(d["day"] >= m * 30.4) & (d["day"] < (m + 2) * 30.4)]["log_vs_base"]
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        row.append(s.median() if len(s) > 20 else np.nan)
    bar = ""
    if not np.isnan(row[0]):
        bar = "P" + "-" * max(0, int((row[0] + 4) * 6))
    print(f"  {m:>6}{row[0]:>12.2f}{row[1]:>12.2f}   {bar}")
print("\n  both should fall steeply, bottom out, then climb -- that shared shape IS")
print("  the phenomenon the world model would learn.")

corr = []
for m in range(0, 25, 2):
    r = []
    for d in (P, O):
        s = d[(d["day"] >= m * 30.4) & (d["day"] < (m + 2) * 30.4)]["log_vs_base"]
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        r.append(s.median() if len(s) > 20 else np.nan)
    if not any(np.isnan(r)):
        corr.append(r)
Cm = np.array(corr)
shape_r = np.corrcoef(Cm[:, 0], Cm[:, 1])[0, 1] if len(Cm) > 3 else np.nan
print(f"\n  correlation between the two response curves: {shape_r:+.2f}")
print("  (high positive => same underlying process, one model can learn both)")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"relative_encoding | prostate_draws={len(P)} | ovarian_draws={len(O)} "
      f"| shape_corr={shape_r:+.2f}")

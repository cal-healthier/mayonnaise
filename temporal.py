"""
temporal.py -- a fair forward-in-time test, plus the height fix.

The first attempt gave 0.847, HIGHER than the random split. That is a red flag,
not a success: the test set had only 56 events, and men treated in 2024-25 have
1-2 years of follow-up before the data ends, so the only progressions you
OBSERVE are the fast ones -- high PSA, high ALP, obvious disease. An easier
problem, not a better model.

Fix: sweep several cut years instead of one. If C-index climbs as the test
window gets more truncated, that confirms the artefact and tells us which cut
is trustworthy -- the ones whose test patients have real follow-up.

Exact C-index (test sets are small enough) with a bootstrap interval, so the
forward number finally has an error bar.

Also repairs sign_height: raw median 1.53 (metres) was multiplied by 0.0254,
the inches->metres factor, giving 0.039. Writes a corrected feature map.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

pd.set_option("display.width", 220)

# ---------------------------------------------------------------- height fix
fm = pd.read_csv("feature_map_final.csv")
h = fm["feature"].astype(str).str.lower().str.contains("height")
if h.any():
    print(f"height factor was {fm.loc[h, 'to_std_factor'].iloc[0]} -> setting to 1.0")
    fm.loc[h, "to_std_factor"] = 1.0
    fm.loc[h, "note"] = "factor corrected: source already in metres"
    fm.to_csv("feature_map_final.csv", index=False)
    print("  wrote corrected feature_map_final.csv (affects FUTURE extractions)")

P = pd.read_parquet("psa_progression.parquet")
PL = pd.read_parquet("psa_labs.parquet")
hc = [c for c in PL.columns if "height" in c.lower()]
if hc:
    PL[hc] = PL[hc] / 0.0254          # undo the bad conversion in the cached panel
    print(f"  repaired {len(hc)} cached height columns; median now "
          f"{PL[hc[0]].median():.2f} m")

tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
P["year"] = pd.to_datetime(tx.reindex(P.index)).dt.year
P = P[P["year"].notna()]
DATA_END = int(P["year"].max())
print(f"\ncohort {len(P):,} men, {int(P['prog'].sum()):,} events, "
      f"treatment years {int(P['year'].min())}-{DATA_END}")

# ---------------------------------------------------------------- exact C-index
def cidx(t, e, r):
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(r, float)
    conc = perm = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        k = later.sum()
        if k == 0:
            continue
        conc += np.sum(r[i] > r[later]) + 0.5 * np.sum(r[i] == r[later])
        perm += k
    return conc / perm if perm else np.nan

def boot(t, e, r, n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float); e = np.asarray(e, bool); r = np.asarray(r, float)
    out = []
    for _ in range(n):
        ix = rng.integers(0, len(t), len(t))
        v = cidx(t[ix], e[ix], r[ix])
        if not np.isnan(v):
            out.append(v)
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)

def gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)

B = pd.DataFrame({"baseline": P["baseline"], "log_baseline": np.log1p(P["baseline"])})
X = B.join(PL, how="left").reindex(P.index).select_dtypes(include=[np.number])
X = X.loc[:, X.notna().sum() > len(X) * 0.05]
print(f"features: {X.shape[1]}")

# ---------------------------------------------------------------- random baseline
s = []
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    m = gbm().fit(X.iloc[tr], P["prog"].iloc[tr])
    s.append(cidx(P["time"].iloc[te], P["prog"].iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
rand = float(np.mean(s))
print(f"\nrandom split (the number we have been quoting): {rand:.3f} ± {np.std(s):.3f}")

# ---------------------------------------------------------------- the sweep
print("\n" + "=" * 92)
print("FORWARD IN TIME: train on <= cut, test on the two years after")
print("=" * 92)
print(f"  {'cut':>5}{'train n':>9}{'train ev':>10}{'test n':>8}{'test ev':>9}"
      f"{'test med f/u':>14}{'C-index':>10}   95% CI")
rows = []
for cut in range(2018, DATA_END):
    tr_i = P.index[P["year"] <= cut]
    te_i = P.index[(P["year"] > cut) & (P["year"] <= cut + 2)]
    if len(te_i) < 100 or P.loc[tr_i, "prog"].sum() < 50:
        continue
    m = gbm().fit(X.loc[tr_i], P.loc[tr_i, "prog"])
    p = m.predict_proba(X.loc[te_i])[:, 1]
    t, e = P.loc[te_i, "time"].values, P.loc[te_i, "prog"].values
    c = cidx(t, e, p)
    lo, hi = boot(t, e, p)
    fu = np.median(P.loc[te_i, "last_day"]) / 365.25
    ev = int(e.sum())
    rows.append({"cut": cut, "test_n": len(te_i), "ev": ev, "fu": fu, "c": c})
    print(f"  {cut:>5}{len(tr_i):>9,}{int(P.loc[tr_i,'prog'].sum()):>10,}"
          f"{len(te_i):>8,}{ev:>9,}{fu:>13.1f}y{c:>10.3f}   [{lo:.3f}, {hi:.3f}]")

R = pd.DataFrame(rows)
print("\n  Follow-up shrinks as the cut moves later -- that is the truncation.")
if len(R) >= 2:
    good = R[R["fu"] >= 2.5]
    if len(good):
        best = good.iloc[0]
        print(f"\n  TRUSTWORTHY CUT: {int(best['cut'])} — test patients have "
              f"{best['fu']:.1f}y median follow-up and {int(best['ev'])} events")
        print(f"  Forward-in-time C-index = {best['c']:.3f}  "
              f"(random split {rand:.3f}, change {best['c']-rand:+.3f})")
        hdr = best["c"]
    else:
        hdr = np.nan
        print("\n  No cut leaves >=2.5y follow-up in the test set — cannot validate "
              "forward with this data window.")
    corr = R[["fu", "c"]].corr().iloc[0, 1]
    print(f"\n  correlation between test follow-up and C-index: {corr:+.2f}")
    print("  (strongly negative => shorter follow-up inflates the score, "
          "confirming the 0.847 was an artefact)")
else:
    hdr = np.nan

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"temporal | random={rand:.3f} | trustworthy_cut_c="
      f"{'NA' if np.isnan(hdr) else round(float(hdr),3)} | cuts_tested={len(R)} "
      f"| fu_vs_c_corr={R[['fu','c']].corr().iloc[0,1]:+.2f}" if len(R) >= 2 else
      f"temporal | random={rand:.3f} | insufficient_cuts")

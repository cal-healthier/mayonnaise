"""
temporal2.py -- two things the sweep left unresolved.

The forward-in-time C-index at the 2018 cut was 0.685 vs 0.756 random. But:

  A. WHICH CLAIM ACTUALLY MATTERS? The paper says bloodwork BEATS PSA -- a
     relative claim. If both degrade equally over time it survives at a lower
     absolute number. We only ran the combined model. Run all three forward.

  B. IS THE DROP EVEN TEMPORAL? The 2018 cut trains on 876 men / 296 events;
     the random split trains on ~4,500. Later cuts have more training data AND
     shorter test follow-up -- the two move together. Control: random CV with
     the training fold subsampled to the same size. If that also lands ~0.69,
     the drop is sample size, not time.

Exact C-index, bootstrap intervals. Cached files only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

pd.set_option("display.width", 220)
P = pd.read_parquet("psa_progression.parquet")
PL = pd.read_parquet("psa_labs.parquet")
hc = [c for c in PL.columns if "height" in c.lower()]
if hc and PL[hc[0]].median() < 1.0:
    PL[hc] = PL[hc] / 0.0254
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
P["year"] = pd.to_datetime(tx.reindex(P.index)).dt.year
P = P[P["year"].notna()]

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
    o = []
    for _ in range(n):
        ix = rng.integers(0, len(t), len(t))
        v = cidx(t[ix], e[ix], r[ix])
        if not np.isnan(v):
            o.append(v)
    return (np.percentile(o, 2.5), np.percentile(o, 97.5)) if o else (np.nan, np.nan)

def gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0)

def prep(df):
    X = df.reindex(P.index).select_dtypes(include=[np.number])
    return X.loc[:, X.notna().sum() > len(X) * 0.05]

B = pd.DataFrame({"baseline": P["baseline"], "log_baseline": np.log1p(P["baseline"])})
SETS = {"PSA at treatment start only": prep(B),
        "routine bloodwork only":      prep(PL),
        "PSA + bloodwork":             prep(B.join(PL, how="left"))}

# ============================================================ A. relative claim
print("=" * 90)
print("A. DOES 'BLOODWORK BEATS PSA' SURVIVE FORWARD IN TIME?")
print("=" * 90)
for cut in (2018, 2019):
    tr_i = P.index[P["year"] <= cut]
    te_i = P.index[(P["year"] > cut) & (P["year"] <= cut + 2)]
    t, e = P.loc[te_i, "time"].values, P.loc[te_i, "prog"].values
    fu = np.median(P.loc[te_i, "last_day"]) / 365.25
    print(f"\n  train <= {cut} ({len(tr_i):,} men, "
          f"{int(P.loc[tr_i,'prog'].sum())} events)  ->  test {cut+1}-{cut+2} "
          f"({len(te_i):,} men, {int(e.sum())} events, {fu:.1f}y follow-up)")
    got = {}
    for name, X in SETS.items():
        m = gbm().fit(X.loc[tr_i], P.loc[tr_i, "prog"])
        p = m.predict_proba(X.loc[te_i])[:, 1]
        c = cidx(t, e, p); lo, hi = boot(t, e, p)
        got[name] = c
        print(f"    {name:<32}{c:.3f}  [{lo:.3f}, {hi:.3f}]")
    gap = got["routine bloodwork only"] - got["PSA at treatment start only"]
    print(f"    -> bloodwork vs PSA: {gap:+.3f}   "
          f"(random split was +0.088)  {'HOLDS' if gap > 0.02 else 'DOES NOT HOLD'}")

# ============================================================ B. size control
print("\n" + "=" * 90)
print("B. IS THE DROP TIME, OR JUST A SMALLER TRAINING SET?")
print("=" * 90)
X = SETS["PSA + bloodwork"]
rng = np.random.default_rng(0)
for n_tr in (876, 1651, 3008, None):
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        if n_tr is not None and len(tr) > n_tr:
            tr = rng.choice(tr, n_tr, replace=False)
        m = gbm().fit(X.iloc[tr], P["prog"].iloc[tr])
        s.append(cidx(P["time"].iloc[te], P["prog"].iloc[te],
                      m.predict_proba(X.iloc[te])[:, 1]))
    lbl = f"random split, training set capped at {n_tr:,}" if n_tr else \
          "random split, full training set"
    print(f"  {lbl:<46}{np.mean(s):.3f} ± {np.std(s):.3f}")

print("\n  Compare each capped row to the forward-in-time result trained on the")
print("  same number of men. If they match, the '-0.071 drop' was sample size.")
print("  If forward is clearly lower at equal training size, it is genuine")
print("  temporal degradation and 0.685 is the honest headline.")

print("\n" + "-" * 74)
print("FINAL LINE:")
tr_i = P.index[P["year"] <= 2018]; te_i = P.index[(P["year"] > 2018) & (P["year"] <= 2020)]
t, e = P.loc[te_i, "time"].values, P.loc[te_i, "prog"].values
res = {}
for name, XX in SETS.items():
    m = gbm().fit(XX.loc[tr_i], P.loc[tr_i, "prog"])
    res[name] = cidx(t, e, m.predict_proba(XX.loc[te_i])[:, 1])
print(f"temporal2 | fwd_psa={res['PSA at treatment start only']:.3f} "
      f"| fwd_blood={res['routine bloodwork only']:.3f} "
      f"| fwd_both={res['PSA + bloodwork']:.3f} "
      f"| gap={res['routine bloodwork only']-res['PSA at treatment start only']:+.3f}")

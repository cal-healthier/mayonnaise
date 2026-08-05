"""
landmark_fast.py -- same model, ~50x faster.

The slow version did ~4M np.polyfit calls inside nested Python loops, each
after a pandas boolean filter. Identical maths here, but:
  - per-patient numpy arrays built ONCE instead of pandas slicing per landmark
  - np.searchsorted for window bounds instead of boolean masks
  - closed-form least-squares slope instead of np.polyfit

Setup unchanged: one row per patient-month, features from data up to that month
only, label = progression within the next 6 months. Grouped CV by patient.
Benchmarked against the PCWG3 rule scored at the same landmarks.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split

pd.set_option("display.width", 240)
HORIZON, STEP, MIN_M, MAX_M = 6.0, 1.0, 3.0, 48.0
FEATS = ["Chem_ALP", "Chem_ALT", "Chem_ALB", "CBC_Hct", "CBC_WBC",
         "CBC_RBC", "CBC_PLT", "CBC_MCV"]

E = pd.read_parquet("psa_progression.parquet")
pv = pd.read_parquet("psa_values.parquet")
lab = pd.read_parquet("psa_labs_during.parquet")
tx = pv.groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
lab = lab.join(E["tx_date"], on="clinic").dropna(subset=["tx_date"])
lab["mo"] = (lab["d"] - lab["tx_date"]).dt.days / 30.44
lab = lab[lab["feature"].isin(FEATS)]
pv = pv.copy(); pv["mo"] = pv["day"] / 30.44
print(f"{len(E):,} men | {len(lab):,} lab rows | {len(pv):,} PSA rows")

def arrays(df, key, xcol, ycol):
    """{key: (sorted x array, y array)} -- built once, reused for every landmark."""
    d = df.sort_values([key, xcol])
    out, ks = {}, d[key].values
    x, y = d[xcol].values.astype(np.float64), d[ycol].values.astype(np.float64)
    edges = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        out[ks[a]] = (x[a:b], y[a:b])
    return out

PSA = arrays(pv, "clinic", "mo", "psa")
LAB = {f: arrays(g, "clinic", "mo", "value") for f, g in lab.groupby("feature")}
print("indexed.")

def slope(x, y):
    n = len(x)
    if n < 2:
        return np.nan
    sx = x.sum(); sy = y.sum()
    d = n * (x * x).sum() - sx * sx
    if d == 0:
        return np.nan
    return (n * (x * y).sum() - sx * sy) / d

rows, done = [], 0
for clinic, e in E.iterrows():
    done += 1
    if done % 1000 == 0:
        print(f"  {done:,}/{len(E):,}", end="\r")
    p = PSA.get(clinic)
    if p is None or len(p[0]) < 3:
        continue
    px, py = p
    end_mo = e["time"] / 30.44
    prog = e["prog"] == 1
    labp = {f: LAB[f].get(clinic) for f in FEATS if f in LAB}
    for m in np.arange(MIN_M, min(end_mo, MAX_M) + 1e-9, STEP):
        if prog:
            y = 1 if (end_mo - m) <= HORIZON else 0
        else:
            if (end_mo - m) < HORIZON:
                break
            y = 0
        i = np.searchsorted(px, m, side="right")
        if i < 3:
            continue
        hx, hy = px[:i], py[:i]
        cur = hy[-1]; nad = hy.min(); j = int(hy.argmin())
        r = {"clinic": clinic, "mo": m, "y": y,
             "psa_cur": np.log1p(cur),
             "psa_vs_nadir": np.log((cur + .01) / (nad + .01)),
             "psa_vs_base": np.log((cur + .01) / (hy[0] + .01)),
             "mo_since_nadir": m - hx[j], "mo_on_tx": m, "n_psa": i,
             "_nadir": nad, "_cur": cur}
        for w in (3, 6):
            a = np.searchsorted(hx, m - w, side="left")
            r[f"psa_slope{w}"] = slope(hx[a:], np.log1p(hy[a:]))
        for f, arr in labp.items():
            if arr is None:
                continue
            lx, ly = arr
            k = np.searchsorted(lx, m, side="right")
            if k < 2:
                continue
            hxx, hyy = lx[:k], ly[:k]
            b0 = np.searchsorted(hxx, 3.0, side="right")
            base = np.median(hyy[:b0]) if b0 >= 1 else np.nan
            r[f"{f}_cur"] = hyy[-1]
            if base and not np.isnan(base) and base != 0:
                r[f"{f}_vs_base"] = hyy[-1] / base
            for w in (3, 6):
                a = np.searchsorted(hxx, m - w, side="left")
                r[f"{f}_slope{w}"] = slope(hxx[a:], hyy[a:])
        rows.append(r)

L = pd.DataFrame(rows)
print(f"\nlandmarks {len(L):,} over {L['clinic'].nunique():,} men | "
      f"positive (progression within {HORIZON:.0f}mo) {L['y'].mean():.1%} "
      f"({int(L['y'].sum()):,})")

fc = [c for c in L.columns if c not in ("clinic", "mo", "y", "_nadir", "_cur")]
PSA_F = [c for c in fc if c.startswith(("psa_", "mo_", "n_psa"))]
LAB_F = [c for c in fc if c.startswith(("Chem_", "CBC_"))]
print(f"features: {len(PSA_F)} PSA/time + {len(LAB_F)} lab = {len(fc)}")

def usable(cols):
    """sklearn's binner fails on columns with <2 distinct values (labs drawn
    too rarely leave slope columns almost entirely NaN)."""
    return [c for c in cols if L[c].nunique(dropna=True) >= 2]

def run(cols, name):
    cols = usable(cols)
    X, y, g = L[cols], L["y"], L["clinic"]
    a = []
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<42}AUROC {a.mean():.3f} ± {a.std():.3f}  ({len(cols)} feat)")
    return a.mean()

print("\n" + "=" * 80)
print(f"WILL HE PROGRESS IN THE NEXT {HORIZON:.0f} MONTHS?")
print("=" * 80)
rule = ((L["_cur"] >= 1.25 * L["_nadir"]) & (L["_cur"] >= L["_nadir"] + 2)).astype(int)
r_auc = roc_auc_score(L["y"], rule)
tp = int(((rule == 1) & (L["y"] == 1)).sum()); fn = int(((rule == 0) & (L["y"] == 1)).sum())
fp = int(((rule == 1) & (L["y"] == 0)).sum()); tn = int(((rule == 0) & (L["y"] == 0)).sum())
print(f"  {'PCWG3 rule at the landmark (current standard)':<42}AUROC {r_auc:.3f}")
print(f"      fires at {rule.mean():.1%} of visits | sensitivity {tp/max(tp+fn,1):.1%} "
      f"| specificity {tn/max(tn+fp,1):.1%}")
print(f"      NB a binary rule sits at ONE point on the ROC curve, so AUROC")
print(f"      understates it. PCWG3 defines progression that has ALREADY happened --")
print(f"      it confirms failure rather than warning of it. That is the real point.")
a1 = run(PSA_F, "PSA trajectory only")
a2 = run(LAB_F, "routine labs only")
a3 = run(fc, "PSA + labs")
print(f"\n  labs add {a3-a1:+.3f} over PSA trajectory")

# like-for-like: let the model fire on the same fraction of visits as the rule
Xf = L[usable(fc)]
oof = pd.Series(index=L.index, dtype=float)
for tr, te in GroupKFold(5).split(Xf, L["y"], groups=L["clinic"]):
    mm_ = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0).fit(Xf.iloc[tr], L["y"].iloc[tr])
    oof.iloc[te] = mm_.predict_proba(Xf.iloc[te])[:, 1]
thr = oof.quantile(1 - rule.mean())
alert = (oof >= thr).astype(int)
tp2 = int(((alert == 1) & (L["y"] == 1)).sum()); fn2 = int(((alert == 0) & (L["y"] == 1)).sum())
fp2 = int(((alert == 1) & (L["y"] == 0)).sum()); tn2 = int(((alert == 0) & (L["y"] == 0)).sum())
print("\n" + "=" * 80)
print("LIKE FOR LIKE: model set to fire at the SAME visit rate as the rule")
print("=" * 80)
print(f"  {'':<14}{'fires at':>10}{'sensitivity':>14}{'specificity':>14}")
print(f"  {'PCWG3 rule':<14}{rule.mean():>10.1%}{tp/max(tp+fn,1):>14.1%}{tn/max(tn+fp,1):>14.1%}")
print(f"  {'model':<14}{alert.mean():>10.1%}{tp2/max(tp2+fn2,1):>14.1%}{tn2/max(tn2+fp2,1):>14.1%}")
print(f"\n  at an identical alert burden the model catches "
      f"{tp2/max(tp,1):.1f}x as many men who progress within 6 months")

print("\n" + "=" * 80)
print("WHAT CARRIES IT")
print("=" * 80)
FC = usable(fc)
Xtr, Xte, ytr, yte = train_test_split(L[FC], L["y"], test_size=.25,
                                      stratify=L["y"], random_state=0)
mm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
    l2_regularization=1.0, early_stopping=True, validation_fraction=.15,
    random_state=0).fit(Xtr, ytr)
pi = permutation_importance(mm, Xte, yte, scoring="roc_auc", n_repeats=5,
                            random_state=0, n_jobs=-1)
for k, v in pd.Series(pi.importances_mean, index=FC).nlargest(15).items():
    print(f"    {k:<26}{v:+.4f}")

print("\n" + "=" * 80)
print("OPERATING POINTS: flag N% of visits -> catch what share of men who")
print("progress in the next 6 months?  (out-of-fold, PSA + labs)")
print("=" * 80)
print(f"  {'flag % of visits':>18}{'sensitivity':>14}{'precision':>12}"
      f"{'alerts / 100 visits':>22}")
for pct in (0.5, 1, 2, 5, 10, 20):
    th = oof.quantile(1 - pct/100)
    al = (oof >= th)
    tpx = int((al & (L["y"] == 1)).sum()); fnx = int((~al & (L["y"] == 1)).sum())
    fpx = int((al & (L["y"] == 0)).sum())
    sens = tpx / max(tpx + fnx, 1); prec = tpx / max(tpx + fpx, 1)
    mark = "  <- rule fires here" if abs(pct - rule.mean()*100) < 0.3 else ""
    print(f"  {pct:>17.1f}%{sens:>14.1%}{prec:>12.1%}{pct:>21.1f}{mark}")
print("\n  a man is seen ~4x a year, so flagging 10% of visits is roughly one")
print("  extra look every 2.5 years per patient -- a modest clinical burden.")

# how far ahead does an alert actually come?
print("\n  LEAD TIME of a correct alert (months before progression):")
prg = L[(L["y"] == 1)].copy()
prg["score"] = oof.reindex(prg.index)
th10 = oof.quantile(0.90)
fired = prg[prg["score"] >= th10]
if len(fired):
    lead = (fired.groupby("clinic")["mo"].min())
    endm = (E["time"] / 30.44).reindex(lead.index)
    gap = (endm - lead).dropna()
    for q in (25, 50, 75):
        print(f"    p{q:<3} {gap.quantile(q/100):>5.1f} months of warning")
    print(f"    men ever flagged before progressing: {gap.notna().sum():,}")

L.to_parquet("landmarks.parquet")
print("\nsaved landmarks.parquet")
print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"landmark_fast | rows={len(L)} | pos={L['y'].mean():.1%} | rule={r_auc:.3f} "
      f"| psa={a1:.3f} | labs={a2:.3f} | both={a3:.3f} | gain={a3-a1:+.3f}")

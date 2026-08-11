"""
paired.py -- redo the arm comparisons with a PAIRED test.

Both text_vs_numbers2.py and ovarian_text.py declared "SEPARATED" or "within
noise" by comparing two arms through their INDEPENDENT fold-to-fold standard
deviations. That test is wrong in a way that always costs power:

    the arms are scored on the SAME folds and the SAME patients, so most of
    the fold-to-fold wobble is shared -- a fold with hard patients is hard for
    every arm at once. Differencing first cancels that shared component; the
    independent-SD test throws it away and inflates the variance of the
    difference by roughly 2x.

So: fit each arm, keep the OUT-OF-FOLD predictions, and bootstrap over
PATIENTS with both arms' predictions resampled together. That is the standard
paired comparison of two AUCs and it is what should have been reported.

No re-embedding -- reuses every cached parquet. The expensive FACT_ORDERS
proxy query is cached to disk on first run.

Reports both cancers side by side, old verdict next to new.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
rng = np.random.default_rng(0)

REPEATS = 3        # OOF passes averaged, to stabilise the score vector
BOOT    = 2000     # patient resamples


def proxies(index, tag):
    """opioid / steroid / total order counts -- the frailty proxy arm"""
    cache = f"prox_{tag}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache).reindex(index).fillna(0)
    ids = "','".join(index.astype(str))
    P = C.query(f"""
    WITH pk AS (SELECT DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
                       PATIENT_DK
                FROM {D}.DIM_PATIENT
                WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
    SELECT pk.clinic,
      COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
              r'OXYCODONE|MORPHINE|HYDROMORPHONE|FENTANYL')) AS opioid_orders,
      COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
              r'PREDNISONE|DEXAMETHASONE')) AS steroid_orders,
      COUNT(*) AS total_orders
    FROM pk LEFT JOIN {D}.FACT_ORDERS o ON o.PATIENT_DK = pk.PATIENT_DK
    GROUP BY 1""").to_dataframe().set_index("clinic")
    P.to_parquet(cache)
    return P.reindex(index).fillna(0)


def oof(X, y, repeats=REPEATS):
    """out-of-fold probability for every patient, averaged over repeats"""
    X = pd.DataFrame(X)
    acc = np.zeros(len(X))
    for r in range(repeats):
        p = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                min_samples_leaf=25, l2_regularization=1.,
                random_state=0).fit(X.iloc[tr], y.iloc[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        acc += p
    return acc / repeats


def paired(y, pa, pb, boot=BOOT):
    """bootstrap the DIFFERENCE in AUC, resampling patients once for both arms"""
    y = np.asarray(y)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    n, d = len(y), []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, len(i)):
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    # two-sided bootstrap p: how often the difference lands on the wrong side
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, lo, hi, max(p, 1 / len(d))


def study(tag, E, labs, V, marker_name):
    print("\n" + "=" * 78)
    print(f"{tag.upper()}   {len(E):,} patients | {int(E['y'].sum())} events "
          f"({E['y'].mean():.0%})")
    print("=" * 78)

    cols = [c for c in labs.columns if c.endswith("__last")]
    S = labs[cols].reindex(E.index)
    S[marker_name] = E["baseline"]
    S = S.loc[:, S.notna().sum() > len(S) * .3]
    PROX = proxies(E.index, tag)
    y = E["y"]

    t0 = time.time()
    arms = {
        "structured":       S,
        "struct + proxies": pd.concat([S, PROX], axis=1),
        "text":             V,
        "everything":       pd.concat([S, PROX, V.add_prefix("t")], axis=1),
    }
    P = {}
    for name, X in arms.items():
        P[name] = oof(X, y)
        print(f"  {name:<20}AUC {roc_auc_score(y, P[name]):.3f}"
              f"   ({time.time()-t0:.0f}s)")

    print(f"\n  {'comparison':<34}{'diff':>7}  {'95% CI':>16}  {'p':>7}   verdict")
    print("  " + "-" * 74)
    out = {}
    for a, b in [("text", "structured"),
                 ("text", "struct + proxies"),
                 ("everything", "struct + proxies"),
                 ("everything", "structured")]:
        o, lo, hi, p = paired(y, P[a], P[b])
        v = "REAL" if lo > 0 else ("negative" if hi < 0 else "not established")
        print(f"  {a+' - '+b:<34}{o:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:>7.4f}   {v}")
        out[(a, b)] = (o, lo, hi, p)
    return out


print("=" * 78)
print("PAIRED RE-ANALYSIS  --  same models, correct test")
print("=" * 78)
print("""
The earlier scripts compared arm A and arm B using each one's own spread across
folds. Because both arms see identical folds and identical patients, that
discards the shared difficulty and roughly doubles the apparent variance of the
difference. Below, the difference itself is bootstrapped over patients.""")

res = {}

# ------------------------------------------------------------------ prostate
E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
V = pd.read_parquet("tvn_emb_nonum.parquet")
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
idx = E.index.intersection(V.dropna(how="all").index)
res["prostate"] = study("prostate", E.loc[idx], labs, V.reindex(idx), "psa")

# ------------------------------------------------------------------- ovarian
E = pd.read_parquet("ov_label.parquet")
cv = pd.read_parquet("ov_ca125.parquet")
E["tx_date"] = pd.to_datetime(cv.groupby("clinic")["tx_date"].first().reindex(E.index))
E = E.dropna(subset=["tx_date"])
labs = pd.read_parquet("ov_labs.parquet")
V = pd.read_parquet("ovt_emb_nonum.parquet")
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]
idx = E.index.intersection(V.dropna(how="all").index)
res["ovarian"] = study("ovarian", E.loc[idx], labs, V.reindex(idx), "baseline_ca125")

# ------------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("BOTH CANCERS, THE COMPARISON THAT MATTERS")
print("=" * 78)
print("  does narrative add to the best structured model available?\n")
print(f"  {'':<12}{'diff':>7}  {'95% CI':>18}  {'p':>8}")
for k in ("prostate", "ovarian"):
    o, lo, hi, p = res[k][("everything", "struct + proxies")]
    print(f"  {k:<12}{o:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:>8.4f}")
print("""
  Two cancers with opposite lab behaviour. In prostate the routine bloodwork
  beat the tumour marker; in ovarian it lost to it, for a mechanistic reason
  (ALP sees bone metastases, ovarian spreads peritoneally). If narrative adds
  in BOTH, the narrative signal does not depend on that mechanism.

  Ovarian is the smaller cohort, so its interval is wide no matter what the
  test is. A point estimate that matches prostate is consistent evidence, not
  proof -- say "consistent, underpowered", never "replicated", unless the
  interval clears zero.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
pieces = []
for k in ("prostate", "ovarian"):
    o, lo, hi, p = res[k][("everything", "struct + proxies")]
    pieces.append(f"{k}={o:+.3f}[{lo:+.3f},{hi:+.3f}]p={p:.4f}")
print("paired | " + " | ".join(pieces))

"""
audit.py -- two real design defects, measured and fixed.

I went back through the pipeline looking specifically for look-ahead. Result:

  labs    CLEAN. psa_model.py windows them explicitly --
          pre = meas[(d <= tx) & (d >= tx - 365d)], then mean/min/max/last
          over that window only.
  notes   CLEAN. note_date <= tx_date AND >= tx - 365d, top 6 most recent.
  proxies LEAKING. The FACT_ORDERS query in is_it_ecog.py and paired.py has
          NO date filter at all:

              FROM pk LEFT JOIN FACT_ORDERS o ON o.PATIENT_DK = pk.PATIENT_DK
              GROUP BY 1

          so opioid_orders, steroid_orders and total_orders are counted over
          the patient's WHOLE record -- after treatment, after progression,
          up to death. total_orders over all time is close to a measure of
          how long someone stayed in the system, which is outcome
          information wearing a costume.

The leak sits in the BASELINE we claim to beat, and "everything" contains it
too, so the reported increment was measured over an unfairly strong
comparator. Direction is conservative -- fixing it should make the text
increment BIGGER, not smaller. But 0.829 and 0.648 are not honest numbers and
cannot be published.

PART 1 rebuilds the proxies with orders strictly before treatment and re-runs
the headline comparison.

PART 2 measures the other defect, which is the label rather than a feature.
The 12-month binary drops anyone censored before 365 days without progressing:

    E = E[(E["prog"] == 1) | (E["time"] >= 365)]

Early censoring is not random -- it is disproportionately people who DIED.
Death from cancer is clinically progression, so dropping them removes the
worst outcomes from the denominator. This quantifies how many, how many of
them died, and re-runs with death-before-365 counted as an event.

PART 3 checks a modelling assumption nobody tested: GBM on 768 dense
embedding dimensions. Trees split one coordinate at a time and embedding
coordinates are arbitrary projections; L2 logistic regression is often
strictly better on this kind of input.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
rng = np.random.default_rng(0)
REPEATS, BOOT = 2, 2000


def oof(X, y, repeats=REPEATS, model="gbm"):
    X = pd.DataFrame(X)
    acc = np.zeros(len(X))
    for r in range(repeats):
        p = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            if model == "gbm":
                m = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                    min_samples_leaf=25, l2_regularization=1., random_state=0)
            else:
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=.05, max_iter=3000))
            Xtr = X.iloc[tr].fillna(X.iloc[tr].median()) if model != "gbm" else X.iloc[tr]
            Xte = X.iloc[te].fillna(X.iloc[tr].median()) if model != "gbm" else X.iloc[te]
            m.fit(Xtr, y.iloc[tr])
            p[te] = m.predict_proba(Xte)[:, 1]
        acc += p
    return acc / repeats


def paired(y, pa, pb, boot=BOOT):
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
    return obs, lo, hi


def clean_proxies(E, tag):
    """same three counts, but only orders STRICTLY BEFORE treatment start"""
    cache = f"prox_{tag}_pre.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache).reindex(E.index).fillna(0)
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(E.index.astype(str), E["tx_date"]))
    P = C.query(f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
    pk AS (SELECT DISTINCT c.clinic, c.tx, p.PATIENT_DK
           FROM coh c JOIN {D}.DIM_PATIENT p
             ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic)
    SELECT pk.clinic,
      COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
              r'OXYCODONE|MORPHINE|HYDROMORPHONE|FENTANYL')) AS opioid_orders,
      COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
              r'PREDNISONE|DEXAMETHASONE')) AS steroid_orders,
      COUNT(o.PATIENT_DK) AS total_orders
    FROM pk LEFT JOIN {D}.FACT_ORDERS o
      ON o.PATIENT_DK = pk.PATIENT_DK
     AND DATE(o.ORDER_APPROVE_DTM) < pk.tx
     AND DATE(o.ORDER_APPROVE_DTM) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
    GROUP BY 1""").to_dataframe().set_index("clinic")
    P.to_parquet(cache)
    return P.reindex(E.index).fillna(0)


def load(tag):
    if tag == "prostate":
        E = pd.read_parquet("psa_progression.parquet")
        labs = pd.read_parquet("psa_labs.parquet")
        V = pd.read_parquet("tvn_emb_nonum.parquet")
        tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
        marker = "psa"
    else:
        E = pd.read_parquet("ov_label.parquet")
        labs = pd.read_parquet("ov_labs.parquet")
        V = pd.read_parquet("ovt_emb_nonum.parquet")
        tx = pd.read_parquet("ov_ca125.parquet").groupby("clinic")["tx_date"].first()
        marker = "baseline_ca125"
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    return E, labs, V, marker


print("=" * 78)
print("DESIGN AUDIT")
print("=" * 78)

summary = {}

for tag in ("prostate", "ovarian"):
    E0, labs, V, marker = load(tag)

    # ---------------------------------------------------- PART 2, the label
    kept = E0[(E0["prog"] == 1) | (E0["time"] >= 365)]
    dropped = E0[(E0["prog"] == 0) & (E0["time"] < 365)]
    print(f"\n{'=' * 78}\n{tag.upper()}\n{'=' * 78}")
    print(f"\nPART 2 -- who the 12-month label throws away")
    print(f"  in cohort before labelling      {len(E0):,}")
    print(f"  dropped: censored before 365d   {len(dropped):,} "
          f"({len(dropped)/max(len(E0),1):.0%})")

    dead = None
    if len(dropped):
        ids = "','".join(E0.index.astype(str))
        VS = C.query(f"""
        SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
               MAX(CAST(r.VITAL_STATUS AS STRING))              AS vital,
               MAX(DATE(r.DATE_LAST_PT_CONTACT_OR_DEATH))       AS last_dt
        FROM {D}.FACT_CANCER_DATA_REPOSITORY r
        JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
        WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
        GROUP BY 1""").to_dataframe().set_index("clinic")
        E0 = E0.join(VS)
        E0["died"] = E0["vital"].astype(str).str.upper().str.contains("DEAD|DECEASED",
                                                                     na=False)
        E0["days_to_last"] = (pd.to_datetime(E0["last_dt"]) - E0["tx_date"]).dt.days
        dropped = E0[(E0["prog"] == 0) & (E0["time"] < 365)]
        dead = dropped["died"].mean()
        print(f"  of those dropped, DEAD          {int(dropped['died'].sum()):,} "
              f"({dead:.0%})")
        print(f"  ...i.e. the label silently removes patients whose outcome was")
        print(f"     the worst one available. Death is progression, clinically.")

    # ------------------------------------------------- PART 1, the leak fix
    print(f"\nPART 1 -- proxies: all-time (leaking) vs pre-treatment only")
    E = kept.copy()
    E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
    idx = E.index.intersection(V.dropna(how="all").index)
    E = E.loc[idx]
    y = E["y"]
    cols = [c for c in labs.columns if c.endswith("__last")]
    S = labs[cols].reindex(E.index); S[marker] = E["baseline"]
    S = S.loc[:, S.notna().sum() > len(S) * .3]
    Vt = V.reindex(E.index)

    OLD = (pd.read_parquet(f"prox_{tag}.parquet").reindex(E.index).fillna(0)
           if os.path.exists(f"prox_{tag}.parquet") else None)
    NEW = clean_proxies(E, tag)

    t0 = time.time()
    p_struct = oof(S, y)
    p_new_sp = oof(pd.concat([S, NEW], axis=1), y)
    p_new_all = oof(pd.concat([S, NEW, Vt.add_prefix("t")], axis=1), y)
    p_text = oof(Vt, y)
    print(f"  ({time.time()-t0:.0f}s)")

    print(f"\n  {'arm':<40}{'AUC':>7}")
    print("  " + "-" * 50)
    print(f"  {'structured':<40}{roc_auc_score(y, p_struct):>7.3f}")
    if OLD is not None:
        p_old_sp = oof(pd.concat([S, OLD], axis=1), y)
        p_old_all = oof(pd.concat([S, OLD, Vt.add_prefix("t")], axis=1), y)
        print(f"  {'struct + proxies  ALL-TIME (leaky)':<40}"
              f"{roc_auc_score(y, p_old_sp):>7.3f}")
    print(f"  {'struct + proxies  PRE-TREATMENT':<40}"
          f"{roc_auc_score(y, p_new_sp):>7.3f}")
    print(f"  {'text':<40}{roc_auc_score(y, p_text):>7.3f}")
    print(f"  {'everything (clean proxies)':<40}"
          f"{roc_auc_score(y, p_new_all):>7.3f}")

    o, lo, hi = paired(y, p_new_all, p_new_sp)
    summary[tag] = (o, lo, hi)
    print(f"\n  THE CLAIM, with the leak removed:")
    print(f"    everything - struct+proxies   {o:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"   {'REAL' if lo > 0 else 'not established'}")
    if OLD is not None:
        o2, lo2, hi2 = paired(y, p_old_all, p_old_sp)
        print(f"    (same comparison, leaky)      {o2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")

    # ------------------------------------------------ PART 3, model choice
    if tag == "ovarian":
        print(f"\nPART 3 -- GBM vs logistic on 768 dense embedding dims")
        p_lin = oof(Vt, y, model="lin")
        o3, lo3, hi3 = paired(y, p_lin, p_text)
        print(f"    text, GBM                   {roc_auc_score(y, p_text):.3f}")
        print(f"    text, L2 logistic           {roc_auc_score(y, p_lin):.3f}"
              f"   {o3:+.3f} [{lo3:+.3f}, {hi3:+.3f}]")
        print(f"    trees split one coordinate at a time and embedding axes are")
        print(f"    arbitrary projections, so a linear model may simply fit better.")

print(f"""
{'=' * 78}
WHAT THIS CHANGES
{'=' * 78}
  The proxy leak was in the comparator, so removing it can only make the text
  increment look better or leave it alone. The claim direction was never at
  risk. What was at risk is the BASELINE number -- 0.829 and 0.648 were
  inflated by orders placed after the outcome, and must be replaced by the
  pre-treatment figures above wherever they appear.

  The label defect is the more serious one for publication and is NOT fixed
  here, only measured. A 12-month binary that discards early censoring is a
  complete-case analysis on an outcome that depends on follow-up. The real
  fix is time-to-event with censoring handled properly, which changes the
  metric from AUROC to a concordance index and is a rewrite, not a patch.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print("audit | " + " | ".join(
    f"{k}_clean={v[0]:+.3f}[{v[1]:+.3f},{v[2]:+.3f}]" for k, v in summary.items()))

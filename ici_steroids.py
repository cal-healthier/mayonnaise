"""
ici_steroids.py -- does the immunotherapy result survive steroid adjustment?

Baseline corticosteroid use around checkpoint-inhibitor initiation is one of the
best-documented confounders in immuno-oncology: patients on steroids at start do
markedly worse, and it is usually the REASON for the steroids (brain metastases,
symptomatic disease, poor performance status) rather than the steroids
themselves. It was invisible to us all day because we were querying
FACT_TREATMENT_DETAIL, the ONCOLOGY table. FACT_ORDERS has 1,326,210 patients
with corticosteroid orders.

Earlier result: cancer type 0.610 -> + bloodwork 0.658 (+0.048).
Question: how much of that +0.048 was bloodwork detecting sick patients who were
on steroids?

  1. how many of our 6,057 ICI patients were on steroids at start
  2. their survival vs the rest (the confounder, quantified)
  3. refit with steroids in the model -- does bloodwork still add +0.048?
  4. refit WITHIN non-steroid patients only -- the clean subgroup
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 240)

I = pd.read_parquet("ici_thyroid_label.parquet")
I = I[I["os_days"] > 0]
IL = pd.read_parquet("ici_labs.parquet")
print(f"immunotherapy cohort {len(I):,} | deaths {int(I['died'].sum()):,}")

STER = ["PREDNISONE", "DEXAMETHASONE", "METHYLPREDNISOLONE", "HYDROCORTISONE",
        "PREDNISOLONE"]
G = "UPPER(IFNULL(CAST(MED_GENERIC AS STRING),''))"
like = " OR ".join(f"{G} LIKE '%{d}%'" for d in STER)
ids = "','".join(I.index.astype(str))
print("pulling steroid orders around ICI start ...")
s = C.query(f"""
  WITH pk AS (
    SELECT DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic, PATIENT_DK
    FROM {D}.DIM_PATIENT
    WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
  SELECT pk.clinic, DATE(o.ORDER_APPROVE_DTM) AS d, {G} AS drug
  FROM pk JOIN {D}.FACT_ORDERS o ON o.PATIENT_DK = pk.PATIENT_DK
  WHERE ({like}) AND o.ORDER_APPROVE_DTM IS NOT NULL
""").to_dataframe()
s["d"] = pd.to_datetime(s["d"])
s = s.join(I["ici_date"], on="clinic").dropna(subset=["ici_date"])
s["day"] = (s["d"] - s["ici_date"]).dt.days
print(f"  {len(s):,} steroid orders over {s['clinic'].nunique():,} of "
      f"{len(I):,} patients")

# baseline window: 30 days before to 7 days after the first ICI dose
base = s[(s["day"] >= -30) & (s["day"] <= 7)]
I["steroid_base"] = I.index.isin(base["clinic"].unique()).astype(int)
# high-dose proxy: dexamethasone or methylprednisolone (usually not just a premed)
hi = base[base["drug"].str.contains("DEXAMETH|METHYLPRED", na=False)]
I["steroid_hi"] = I.index.isin(hi["clinic"].unique()).astype(int)

print("\n" + "=" * 80)
print("1. THE CONFOUNDER, QUANTIFIED")
print("=" * 80)
for col, lbl in [("steroid_base", "any steroid at ICI start"),
                 ("steroid_hi", "dexamethasone / methylpred at start")]:
    a = I[I[col] == 1]; b = I[I[col] == 0]
    print(f"\n  {lbl}: {len(a):,} ({len(a)/len(I):.0%})")
    print(f"    on steroids   died {a['died'].mean():6.1%}  "
          f"median OS {a['os_days'].median()/365.25:.2f}y")
    print(f"    not           died {b['died'].mean():6.1%}  "
          f"median OS {b['os_days'].median()/365.25:.2f}y")
print("\n  a large gap = a real confounder that was invisible in every model today.")

# ---------------------------------------------------------------- models
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

def run(X, sub, name):
    X = X.reindex(sub.index).select_dtypes(include=[np.number])
    X = X.loc[:, X.nunique(dropna=True) >= 2]
    y, t = sub["died"], sub["os_days"]
    a = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        a.append(cidx(t.iloc[te], y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<44}C-index {a.mean():.3f} ± {a.std():.3f}  ({X.shape[1]} feat)")
    return a.mean()

site = pd.get_dummies(I["site3"].astype(str), prefix="site").astype(float)
ster = I[["steroid_base", "steroid_hi"]].astype(float)
print("\n" + "=" * 80)
print("2. DOES BLOODWORK STILL ADD ONCE STEROIDS ARE IN THE MODEL?")
print("=" * 80)
c0 = run(site, I, "cancer type only")
c1 = run(site.join(IL, how="left"), I, "cancer type + bloodwork  (was +0.048)")
c2 = run(site.join(ster), I, "cancer type + steroids")
c3 = run(site.join(ster).join(IL, how="left"), I, "cancer type + steroids + bloodwork")
print(f"\n  bloodwork adds over cancer type          {c1-c0:+.3f}")
print(f"  bloodwork adds over cancer type+steroids {c3-c2:+.3f}")
print(f"  steroids alone add over cancer type      {c2-c0:+.3f}")

print("\n" + "=" * 80)
print("3. CLEAN SUBGROUP: patients NOT on steroids at ICI start")
print("=" * 80)
clean = I[I["steroid_base"] == 0]
print(f"  n={len(clean):,}, deaths {int(clean['died'].sum()):,}")
if clean["died"].sum() >= 200:
    d0 = run(site, clean, "cancer type only")
    d1 = run(site.join(IL, how="left"), clean, "cancer type + bloodwork")
    print(f"\n  bloodwork adds within non-steroid patients {d1-d0:+.3f}")
    print("  if this matches the full-cohort gain, the finding is not a steroid")
    print("  artefact. if it collapses, bloodwork was partly detecting steroid use.")
else:
    d0 = d1 = float("nan")
    print("  too few deaths for a stable estimate")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"ici_steroids | on_steroid={int(I['steroid_base'].sum())} "
      f"({I['steroid_base'].mean():.0%}) | base={c0:.3f} | blood={c1:.3f} "
      f"| ster={c2:.3f} | both={c3:.3f} | gain_adj={c3-c2:+.3f} "
      f"| gain_clean={d1-d0:+.3f}")

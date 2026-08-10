"""
is_it_ecog.py -- is the narrative just recovering performance status?

The concept probe pointed hard at functional status and care intensity:
  recent admission +0.343, walks with difficulty +0.336, urgent care +0.320,
  emergency visit +0.268, pain +0.258, missed appointments +0.240
and the strongest single prognostic concepts were walks-with-difficulty
(+0.129) and recent admission (+0.120).

That is a description of ECOG performance status -- one of the strongest
prognostic factors in oncology, and one of the worst captured in structured
data (2017+ only here).

So the mechanistic question: if we GIVE the structured model performance
status, does the text advantage disappear?

  gap closes  -> "narrative recovers performance status, which registries lose"
                 clean, mechanistic, immediately useful
  gap persists -> the text carries something BEYOND performance status.
                 stronger and more surprising.

Also builds proxies for the same construct from structured data (admissions,
opioid orders, unscheduled encounters) -- because if a reviewer says "you could
get that from claims", this answers it.

Local, free.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

E = pd.read_parquet("psa_progression.parquet")
labs = pd.read_parquet("psa_labs.parquet")
V = pd.read_parquet("tvn_emb_nonum.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[((E["prog"] == 1) | (E["time"] >= 365)) & E.index.isin(V.index)]
V = V.reindex(E.index)
print(f"{len(E):,} men | {int(E['y'].sum())} events")

# ---------------------------------------------------------------- find ECOG
print("\nlooking for structured performance status ...")
cc = pd.read_parquet("measurement_concepts.parquet")
cc["nl"] = cc["name"].astype(str).str.lower()
ec = cc[cc["nl"].str.contains("ecog|performance status|karnofsky|zubrod", na=False)]
print(f"  measurement concepts: {len(ec)}")
obs = C.query(f"""SELECT column_name FROM {D}.INFORMATION_SCHEMA.COLUMNS
  WHERE REGEXP_CONTAINS(UPPER(column_name), r'ECOG|PERFORMANCE|KARNOFSKY')
  LIMIT 10""").to_dataframe()
print(f"  columns named for it: {list(obs['column_name']) if len(obs) else 'none'}")

ECOG = None
if len(ec):
    ids = "','".join(E.index.astype(str))
    cids = ",".join(str(int(x)) for x in ec.head(4)["cid"])
    q = C.query(f"""
      SELECT CAST(pe.person_source_value AS STRING) AS clinic,
             APPROX_QUANTILES(m.value_as_number,100)[OFFSET(50)] AS ecog
      FROM {D}.person pe JOIN {D}.measurement m ON m.person_id=pe.person_id
      WHERE CAST(pe.person_source_value AS STRING) IN ('{ids}')
        AND m.measurement_concept_id IN ({cids}) AND m.value_as_number IS NOT NULL
      GROUP BY 1""").to_dataframe()
    if len(q):
        ECOG = q.set_index("clinic")["ecog"]
        print(f"  found ECOG for {len(ECOG):,} of {len(E):,} men "
              f"({len(ECOG)/len(E):.0%})")

# ---------------------------------------------------------------- structured proxies
print("\nbuilding structured proxies for the same construct ...")
ids = "','".join(E.index.astype(str))
P = C.query(f"""
WITH pk AS (SELECT DISTINCT CAST(PATIENT_CLINIC_NUMBER AS STRING) AS clinic, PATIENT_DK
            FROM {D}.DIM_PATIENT WHERE CAST(PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}'))
SELECT pk.clinic,
  COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
          r'OXYCODONE|MORPHINE|HYDROMORPHONE|FENTANYL')) AS opioid_orders,
  COUNTIF(REGEXP_CONTAINS(UPPER(IFNULL(CAST(o.MED_GENERIC AS STRING),'')),
          r'PREDNISONE|DEXAMETHASONE')) AS steroid_orders,
  COUNT(*) AS total_orders
FROM pk LEFT JOIN {D}.FACT_ORDERS o ON o.PATIENT_DK = pk.PATIENT_DK
GROUP BY 1""").to_dataframe().set_index("clinic")
print(f"  order-based proxies for {len(P):,} men")

cols = [c for c in labs.columns if c.endswith("__last")]
S = labs[cols].reindex(E.index); S["psa"] = E["baseline"]
S = S.loc[:, S.notna().sum() > len(S)*.3]
PROX = P.reindex(E.index).fillna(0)

def run(X, name, idx=None):
    idx = E.index if idx is None else idx
    X = pd.DataFrame(X).reindex(idx); y = E["y"].reindex(idx)
    a = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06,
            max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(a)
    print(f"  {name:<48}{a.mean():.3f} ± {a.std():.3f}   (n={len(idx):,})")
    return a.mean()

print("\n" + "=" * 80)
print("CAN STRUCTURED DATA CATCH UP IF WE GIVE IT THE FRAILTY SIGNAL?")
print("=" * 80)
s0 = run(S, "structured: labs + PSA")
s1 = run(pd.concat([S, PROX], axis=1), "structured + opioid/steroid/order counts")
t0 = run(V, "TEXT (numerals removed)")
b0 = run(pd.concat([S, PROX, V.add_prefix("t")], axis=1), "everything")
print(f"\n  proxies add to structured        {s1-s0:+.3f}")
print(f"  text still beats structured+prox {t0-s1:+.3f}")

if ECOG is not None and len(ECOG) >= 400:
    sub = E.index[E.index.isin(ECOG.index)]
    print("\n" + "=" * 80)
    print(f"THE ECOG SUBSET  ({len(sub):,} men with a recorded performance status)")
    print("=" * 80)
    Se = S.reindex(sub).copy(); Se["ecog"] = ECOG.reindex(sub)
    a = run(S, "structured, no ECOG", sub)
    b = run(Se, "structured + ECOG", sub)
    c = run(V, "TEXT", sub)
    print(f"\n  ECOG adds to structured     {b-a:+.3f}")
    print(f"  text still beats struct+ECOG {c-b:+.3f}")
    print("""
  If text no longer beats structured+ECOG, the narrative was recovering
  performance status -- a clean mechanistic story.
  If it still beats it, the text carries something BEYOND performance status,
  which is the stronger claim.""")
else:
    print("\n  not enough structured ECOG to run the subset test "
          f"({0 if ECOG is None else len(ECOG)} men)")
    print("  -> that absence is itself the point: performance status is barely")
    print("     recorded, which is exactly why recovering it from text matters.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"is_it_ecog | struct={s0:.3f} | struct+proxy={s1:.3f} | text={t0:.3f} "
      f"| all={b0:.3f} | ecog_n={0 if ECOG is None else len(ECOG)}")

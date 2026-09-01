"""
who_dx2.py -- is "first note" really first Mayo contact, or ingested outside records?

The established-patient story assumed first_note = first Mayo encounter. But
Mayo imports outside records at referral, dated ORIGINALLY -- so a patient
referred in 2021 can show a 2012 "first note" that is really an outside
provider's record. That would fake a long Mayo tenure.

Test with a Mayo-NATIVE signal: a medication ORDER has a Mayo order-approval
timestamp, which cannot be backloaded from an outside chart the way a scanned
note can. Compare, for the 2021+ newly-diagnosed:

  lead-by-note   dx - first note
  lead-by-order  dx - first Mayo medication order

If order-lead is much SHORTER than note-lead, the early notes are ingested
outside records and these patients did come to Mayo around diagnosis.
"""
import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
CUT = 2021

R = pd.read_parquet("recency.parquet")
R["dx"] = pd.to_datetime(R["dx"], errors="coerce")
R["first_note"] = pd.to_datetime(R["first_note"], errors="coerce")

oc = "orders_first.parquet"
if os.path.exists(oc):
    O = pd.read_parquet(oc)
else:
    sql = f"""
    SELECT PATIENT_DK, MIN(DATE(ORDER_APPROVE_DTM)) AS first_order,
           COUNT(*) AS n_orders
    FROM {D}.FACT_ORDERS
    WHERE ORDER_APPROVE_DTM IS NOT NULL
    GROUP BY 1
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"orders scan: {job.total_bytes_processed/1e9:.0f} GB "
          f"(~${job.total_bytes_processed/1e12*5:.1f})")
    O = C.query(sql).to_dataframe()
    O["first_order"] = pd.to_datetime(O["first_order"].astype("datetime64[ns]"),
                                      errors="coerce")
    O.to_parquet(oc)

R = R.merge(O, on="PATIENT_DK", how="left")
R["note_lead"] = (R["dx"] - R["first_note"]).dt.days / 365.25
R["order_lead"] = (R["dx"] - R["first_order"]).dt.days / 365.25

X = R[R["dx"].dt.year >= CUT].copy()
n = len(X)
has_o = X["first_order"].notna()
print("=" * 70)
print(f"IS THE {CUT}+ 'LONG TENURE' REAL?   {n:,} newly-diagnosed patients")
print("=" * 70)
print(f"  have any Mayo order at all: {has_o.sum():,} ({has_o.mean():.0%})\n")
print(f"  {'signal':<26}{'median lead':>13}{'>=1yr before dx':>18}")
print("  " + "-" * 56)
print(f"  {'by first NOTE':<26}{X['note_lead'].median():>11.1f}y"
      f"{(X['note_lead']>=1).mean():>17.0%}")
Xo = X[has_o]
print(f"  {'by first ORDER (native)':<26}{Xo['order_lead'].median():>11.1f}y"
      f"{(Xo['order_lead']>=1).mean():>17.0%}")

print(f"\n  ORDER lead distribution (the trustworthy one):")
for lo, hi, name in [(-100, 0.25, "orders start AT dx (<3mo)"),
                     (0.25, 1, "3-12 mo before"),
                     (1, 3, "1-3 yr before"),
                     (3, 7, "3-7 yr before"),
                     (7, 100, "7+ yr before")]:
    m = ((Xo["order_lead"] >= lo) & (Xo["order_lead"] < hi)).sum()
    print(f"    {name:<28}{m:>9,}  ({m/len(Xo):.0%})")

# the reconciliation: of note-established, how many are order-established?
ne = X[(X["note_lead"] >= 1) & has_o]
truly = (ne["order_lead"] >= 1).mean()
referral = (ne["order_lead"] < 0.25).mean()
print(f"""
  Of patients who look established by NOTES (>=1yr):
    also established by ORDERS (truly long-tenure): {truly:.0%}
    but orders only start AT diagnosis (referral,   {referral:.0%}
      early notes were ingested outside records)
""")

print("=" * 70)
print("VERDICT")
print("=" * 70)
gap = X["note_lead"].median() - Xo["order_lead"].median()
if gap > 2:
    print(f"  first_note OVERSTATES Mayo tenure by ~{gap:.0f} years vs native orders.")
    print(f"  A big share of 'established' patients actually arrive around")
    print(f"  diagnosis -- your intuition was right: people come TO Mayo for the")
    print(f"  cancer, and their outside records travel with them (ingested, back-")
    print(f"  dated). Use first-ORDER, not first-note, as 'first Mayo contact'.")
else:
    print(f"  order-lead and note-lead agree (~{Xo['order_lead'].median():.0f}y): the long")
    print(f"  tenure is REAL -- Mayo Rochester is these patients' actual local")
    print(f"  provider for years before the cancer. Both signals say established.")
print(f"""
  Either way, 'first seen at Mayo' should be measured by a native signal
  (order/lab), not by first note, since notes carry imported outside history.""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"who_dx2 | note_lead={X['note_lead'].median():.1f}y "
      f"| order_lead={Xo['order_lead'].median():.1f}y "
      f"| note_est%={(X['note_lead']>=1).mean():.0%} "
      f"| order_est%={(Xo['order_lead']>=1).mean():.0%}")

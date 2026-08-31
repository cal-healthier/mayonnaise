"""
nextline_scope.py -- can Mayo support "predict the next line and how long it works"?

The decision nodes already exist in our data, built for another purpose:

  prostate   the PCWG3 progression date IS castration resistance -- the moment
             the second-line choice (abiraterone / enzalutamide / docetaxel /
             ...) is actually made
  ovarian    the GCIG progression date after platinum, where the platinum-free
             interval (over vs under ~6 months) drives rechallenge vs switch

Before designing anything, four feasibility questions, all answerable with
one BigQuery pass and no LLM:

  1. LEAKAGE   Mayo is a referral centre; the registry recurrence field broke
               because events happen elsewhere. What fraction of progressors
               have ANY Mayo medication record after progression? If this is
               low, next-line sequences are too holey and we say so.
  2. CHOICE    Which next-line drugs do we actually observe, at what rate,
               and how long after progression?
  3. DURATION  Rough response-duration proxies: time from first to last order
               of the chosen drug, and time to the NEXT drug class (TTNT).
  4. ERA       Pre-exposure counts -- abiraterone/enzalutamide moved upfront
               ~2017-19, so for some men they are not a "next line" at all.

Drugs are matched in BOTH medication tables (FACT_ORDERS via MED_GENERIC,
FACT_TREATMENT_DETAIL via DIM_MED_NAME) and tagged by source, which also
settles where chemo administrations live.

Aggregate statistics only; nothing patient-level is printed.
"""
import numpy as np
import pandas as pd
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
pd.set_option("display.width", 250)

CLASSES = {
    "prostate": [
        ("abiraterone",  r"ABIRATERONE"),
        ("enzalutamide", r"ENZALUTAMIDE"),
        ("other_arpi",   r"APALUTAMIDE|DAROLUTAMIDE"),
        ("docetaxel",    r"DOCETAXEL"),
        ("cabazitaxel",  r"CABAZITAXEL"),
        ("radioligand",  r"RADIUM|XOFIGO|LUTETIUM|PLUVICTO"),
        ("sipuleucel",   r"SIPULEUCEL|PROVENGE"),
        ("parp",         r"OLAPARIB|RUCAPARIB|TALAZOPARIB|NIRAPARIB"),
        ("pembrolizumab", r"PEMBROLIZUMAB"),
        ("legacy",       r"MITOXANTRONE|KETOCONAZOLE|ESTRAMUSTINE"),
    ],
    "ovarian": [
        ("platinum",     r"CARBOPLATIN|CISPLATIN|OXALIPLATIN"),
        ("taxane",       r"PACLITAXEL|DOCETAXEL"),
        ("gemcitabine",  r"GEMCITABINE"),
        ("doxorubicin",  r"DOXORUBICIN"),
        ("topotecan",    r"TOPOTECAN|ETOPOSIDE"),
        ("bevacizumab",  r"BEVACIZUMAB|AVASTIN"),
        ("parp",         r"OLAPARIB|NIRAPARIB|RUCAPARIB"),
    ],
}


def prog_dates(tag):
    if tag == "prostate":
        E = pd.read_parquet("psa_progression.parquet")
        tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
    else:
        E = pd.read_parquet("ov_label.parquet")
        tx = pd.read_parquet("ov_ca125.parquet").groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    P = E[E["prog"] == 1].copy()
    P["pd"] = P["tx_date"] + pd.to_timedelta(P["time"].astype(int), unit="D")
    return P


def med_events(tag, P):
    case = "\n      ".join(
        f"WHEN REGEXP_CONTAINS(o.med, r'{rx}') THEN '{name}'"
        for name, rx in CLASSES[tag])
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(P.index.astype(str), P["pd"]))
    sql = f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, pdt DATE>>[{rows}])),
    pk AS (SELECT DISTINCT c.clinic, c.pdt, p.PATIENT_DK
           FROM coh c JOIN {D}.DIM_PATIENT p
             ON CAST(p.PATIENT_CLINIC_NUMBER AS STRING) = c.clinic),
    o AS (
      SELECT PATIENT_DK,
             UPPER(IFNULL(CAST(MED_GENERIC AS STRING),'')) AS med,
             DATE(ORDER_APPROVE_DTM) AS d, 'orders' AS src
      FROM {D}.FACT_ORDERS
      UNION ALL
      SELECT t.PATIENT_DK,
             UPPER(CONCAT(IFNULL(CAST(m.MED_NAME_DESCRIPTION AS STRING),''), ' ',
                          IFNULL(CAST(m.MED_GENERIC_NAME_DESCRIPTION AS STRING),''))),
             DATE(t.TREATMENT_DTM), 'onc'
      FROM {D}.FACT_TREATMENT_DETAIL t
      JOIN {D}.DIM_MED_NAME m USING (MED_NAME_DK)
    )
    SELECT pk.clinic,
      CASE {case} ELSE '_other' END AS cls,
      COUNTIF(o.d >  pk.pdt) AS n_after,
      COUNTIF(o.d <= pk.pdt) AS n_before,
      COUNTIF(o.src = 'onc') AS n_onc,
      COUNTIF(o.src = 'orders') AS n_ord,
      MIN(IF(o.d > pk.pdt, o.d, NULL))  AS first_after,
      MAX(IF(o.d > pk.pdt, o.d, NULL))  AS last_after,
      MAX(IF(o.d <= pk.pdt, o.d, NULL)) AS last_before
    FROM pk JOIN o ON o.PATIENT_DK = pk.PATIENT_DK
    WHERE o.d IS NOT NULL
    GROUP BY 1, 2
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"  ({job.total_bytes_processed/1e9:.0f} GB scan)")
    return C.query(sql).to_dataframe()


def med_events_cached(tag, P):
    import os
    cache = f"nl_{tag}.parquet"
    if os.path.exists(cache):
        print("  (cached)")
        return pd.read_parquet(cache)
    G = med_events(tag, P)
    G.to_parquet(cache)
    return G


def death(P):
    ids = "','".join(P.index.astype(str))
    R = C.query(f"""
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic,
             MAX(CAST(r.VITAL_STATUS AS STRING)) AS vital,
             MAX(DATE(r.DATE_LAST_PT_CONTACT_OR_DEATH)) AS last_dt
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r
      JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
      GROUP BY 1""").to_dataframe().set_index("clinic")
    R["dead"] = R["vital"].astype(str).str.upper().str.contains("DEAD|DECEASED", na=False)
    R["last_dt"] = pd.to_datetime(R["last_dt"].astype("datetime64[ns]"))
    return R


for tag in ("prostate", "ovarian"):
    P = prog_dates(tag)
    print("\n" + "=" * 78)
    print(f"{tag.upper()}   {len(P):,} progressors | median progression year "
          f"{int(P['pd'].dt.year.median())} (IQR {int(P['pd'].dt.year.quantile(.25))}-"
          f"{int(P['pd'].dt.year.quantile(.75))})")
    print("=" * 78)
    G = med_events_cached(tag, P)
    G["clinic"] = G["clinic"].astype(str)
    for c in ("first_after", "last_after", "last_before"):
        G[c] = pd.to_datetime(G[c].astype("datetime64[ns]"))
    pdmap = P["pd"]
    pdmap.index = pdmap.index.astype(str)
    G = G.join(pdmap.rename("pdt"), on="clinic")

    # 1. leakage -----------------------------------------------------------
    tot_after = G.groupby("clinic")["n_after"].sum()
    seen = tot_after.reindex(P.index.astype(str)).fillna(0)
    print(f"\n  1. LEAKAGE: any Mayo medication record after progression: "
          f"{(seen > 0).mean():.0%} of progressors "
          f"({int((seen == 0).sum())} have none -- likely treated elsewhere)")

    # 2. choice ------------------------------------------------------------
    R = G[G["cls"] != "_other"].copy()
    R["days_to_start"] = (R["first_after"] - R["pdt"]).dt.days
    R["new_start_1y"] = (R["n_before"] == 0) & (R["days_to_start"] <= 365)
    R["span"] = (R["last_after"] - R["first_after"]).dt.days
    n = len(P)
    print(f"\n  2. NEXT-LINE STARTS (new class, never before progression, within 1y)")
    print(f"  {'class':<15}{'new start':>10}{'pre-exposed':>13}{'median d->start':>17}"
          f"{'median days on':>16}{'onc-table share':>17}")
    print("  " + "-" * 88)
    for name, _ in CLASSES[tag]:
        sub = R[R["cls"] == name]
        ns = sub["new_start_1y"].sum()
        pre = (sub["n_before"] > 0).sum()
        st = sub.loc[sub["new_start_1y"], "days_to_start"].median()
        sp = sub.loc[sub["new_start_1y"], "span"].median()
        onc = sub["n_onc"].sum() / max(sub["n_onc"].sum() + sub["n_ord"].sum(), 1)
        print(f"  {name:<15}{ns:>6,} ({ns/n:>4.0%}){pre:>13,}"
              f"{st if pd.notna(st) else float('nan'):>17.0f}"
              f"{sp if pd.notna(sp) else float('nan'):>16.0f}{onc:>16.0%}")

    starts = (R[R["new_start_1y"]]
              .sort_values("first_after")
              .groupby("clinic")
              .agg(first_cls=("cls", "first"), n_cls=("cls", "nunique"),
                   d1=("first_after", "first"), d2=("first_after", lambda s: s.iloc[1] if len(s) > 1 else pd.NaT)))
    any_start = starts.reindex(P.index.astype(str))
    print(f"\n  any new class within 1y: {any_start['first_cls'].notna().mean():.0%} "
          f"of progressors")

    # 3. duration ----------------------------------------------------------
    ttnt = (pd.to_datetime(starts["d2"]) - pd.to_datetime(starts["d1"])).dt.days.dropna()
    if len(ttnt):
        print(f"  3. TTNT (first new class -> second new class): "
              f"median {ttnt.median():.0f}d over {len(ttnt):,} patients with a 2nd switch")

    # 4. cohort-specific ---------------------------------------------------
    if tag == "ovarian":
        plat = G[(G["cls"] == "platinum")].set_index("clinic")
        pfi = (plat["pdt"] - plat["last_before"]).dt.days.dropna()
        band = pfi < 182
        re_1y = R[(R["cls"] == "platinum")].set_index("clinic")["days_to_start"] <= 365
        parp1 = R[(R["cls"] == "parp") & R["new_start_1y"]].set_index("clinic").index
        print(f"\n  4. PLATINUM-FREE INTERVAL: median {pfi.median():.0f}d "
              f"| resistant (<182d): {band.mean():.0%}")
        for lab, idx in (("resistant <182d", pfi[band].index), ("sensitive >=182d", pfi[~band].index)):
            rech = re_1y.reindex(idx).fillna(False).mean()
            parp = pd.Index(idx).isin(parp1).mean()
            print(f"     {lab:<18} platinum again within 1y: {rech:.0%} | PARP: {parp:.0%}")

    DE = death(P)
    dd = (DE["last_dt"] - pdmap.reindex(DE.index)).dt.days
    dead_after = DE["dead"] & (dd > 0)
    print(f"\n  DEATH FOLLOW-UP: {DE['dead'].mean():.0%} dead; among them median "
          f"{dd[dead_after].median():.0f}d progression -> last contact/death "
          f"(overall-survival endpoint is available)")

print("""
{}
READING IT
{}
  The leakage number is the go/no-go. If most progressors have a visible next
  line at Mayo, the retrospective backbone of forecast-then-observe -- lock a
  forecast at the progression date, score it against what the marker and the
  treatment record then did -- runs on data we already hold. If they vanish
  after progression, the sequences are too holey and the study belongs on a
  network that keeps its patients.

  Pre-exposure is the era trap: a man already on abiraterone at progression
  did not "choose" it as a next line. Any modelling has to condition on what
  the patient has already seen.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print("nextline_scope | see LEAKAGE + any-new-class lines per cohort")

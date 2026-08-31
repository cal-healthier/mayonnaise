"""
scenario_census.py -- does the scenario-export architecture actually work?

The plan: agents research SCENARIOS at home, patients map to scenarios inside
the enclave, and only the scenario list ever leaves. That is cheap and easy to
approve ONLY IF a small grid covers most patients. This measures it from data
already cached -- no Gemini, no BigQuery scan.

Scenario key, from structured data alone (the Gemini extractor would refine,
not replace, this):

  prostate   prior exposure among {ARPI, docetaxel, radioligand, PARP} at the
             progression date + era band
  ovarian    platinum-free interval band + prior bevacizumab/PARP + era band

Prints the census with k<10 scenarios rolled into OTHER -- the printout is
itself the prototype of the export artifact, k-anonymity discipline included.
Saves the patient->scenario map to the enclave for the later join.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)


def dates(tag):
    if tag == "prostate":
        E = pd.read_parquet("psa_progression.parquet")
        tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
    else:
        E = pd.read_parquet("ov_label.parquet")
        tx = pd.read_parquet("ov_ca125.parquet").groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    P = E[E["prog"] == 1].copy()
    P["pdt"] = P["tx_date"] + pd.to_timedelta(P["time"].astype(int), unit="D")
    P.index = P.index.astype(str)
    return P


def era(y):
    return np.where(y <= 2017, "pre-2018", np.where(y <= 2021, "2018-21", "2022+"))


OUT = {}
for tag in ("prostate", "ovarian"):
    P = dates(tag)
    G = pd.read_parquet(f"nl_{tag}.parquet")
    G["clinic"] = G["clinic"].astype(str)
    for c in ("last_before",):
        G[c] = pd.to_datetime(G[c].astype("datetime64[ns]"), errors="coerce")
    pre = (G[G["n_before"] > 0].groupby("clinic")["cls"]
           .apply(set).reindex(P.index))
    pre = pre.apply(lambda s: s if isinstance(s, set) else set())

    if tag == "prostate":
        def key(clinic):
            s = pre.loc[clinic]
            parts = []
            if s & {"abiraterone", "enzalutamide", "other_arpi"}:
                parts.append("post-ARPI")
            if "docetaxel" in s or "cabazitaxel" in s:
                parts.append("post-taxane")
            if "radioligand" in s:
                parts.append("post-RLT")
            if "parp" in s:
                parts.append("post-PARP")
            hist = "+".join(parts) if parts else "ARPI/chemo-naive"
            return f"mCRPC | {hist}"
    else:
        plat = G[G["cls"] == "platinum"].set_index("clinic")["last_before"]
        pfi = (P["pdt"] - plat.reindex(P.index)).dt.days

        def key(clinic):
            s = pre.loc[clinic]
            d = pfi.loc[clinic] if clinic in pfi.index else np.nan
            band = ("plat-resistant" if pd.notna(d) and d < 182
                    else "plat-sensitive" if pd.notna(d) else "plat-status-unknown")
            extras = []
            if "bevacizumab" in s:
                extras.append("prior-bev")
            if "parp" in s:
                extras.append("prior-PARP")
            return f"rec-ovarian | {band}" + ("" if not extras else " | " + "+".join(extras))

    S = pd.DataFrame(index=P.index)
    S["scenario"] = [key(c) for c in P.index]
    S["scenario"] = S["scenario"] + " | " + era(P["pdt"].dt.year.values)
    S.to_parquet(f"scenario_map_{tag}.parquet")
    OUT[tag] = S

    n = len(S)
    vc = S["scenario"].value_counts()
    big = vc[vc >= 10]
    print("\n" + "=" * 78)
    print(f"{tag.upper()}   {n:,} progressors -> {len(vc)} raw scenarios")
    print("=" * 78)
    print(f"  {'scenario (exportable rows, k>=10)':<58}{'n':>7}{'cum%':>7}")
    print("  " + "-" * 72)
    cum = 0
    for sc, c in big.items():
        cum += c
        print(f"  {sc:<58}{c:>7,}{cum/n:>7.0%}")
    small = vc[vc < 10]
    print(f"  {'OTHER (k<10, stays inside)':<58}{small.sum():>7,}{1:>7.0%}"
          f"   [{len(small)} scenarios]")
    print(f"\n  coverage: top 5 = {vc.head(5).sum()/n:.0%} | "
          f"top 10 = {vc.head(10).sum()/n:.0%} | "
          f"exportable rows cover {big.sum()/n:.0%}")

print("""
{}
READING IT
{}
  If a dozen exportable rows cover ~90% of patients, the architecture holds:
  the research agent runs once per row at home, and the only thing that ever
  crosses the boundary is the table above -- generic clinical situations with
  counts, k>=10, no per-patient anything.

  The k<10 tail is the price of the design: those patients get the nearest
  broader scenario's evidence pack, or wait for a Gemini-refined second pass.
  The census says how big that price is.

  These keys are structured-data-only. The in-enclave Gemini pass would add
  what structure cannot see (BRCA status from prose, histology subtype,
  contraindications) -- REFINING rows, not exploding them: each refinement
  must justify itself by splitting a big row, or it stays out of the key.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print("scenario_census | " + " | ".join(
    f"{t}: {OUT[t]['scenario'].nunique()} scenarios, "
    f"top10={OUT[t]['scenario'].value_counts().head(10).sum()/len(OUT[t]):.0%}"
    for t in OUT))

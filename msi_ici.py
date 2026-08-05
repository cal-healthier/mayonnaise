"""
msi_ici.py -- does bloodwork add anything beyond the biomarker that actually
governs immunotherapy prescribing?

MSI-high is the FDA-approved PAN-CANCER indication for pembrolizumab -- the
reason it can be given regardless of tumour site. It is the modern standard of
care for this drug class, and we have 5,091 patients with a real MSI result.

This is the sharpest test of the principle so far:
  prostate  -> bloodwork BEAT PSA        (crude marker, real blind spot)
  ovarian   -> bloodwork LOST to CA-125  (good marker, no blind spot)
  ICI toxicity -> bloodwork did NOTHING  (wrong biology entirely)
  ICI survival vs MSI -> ?               (a genuine molecular marker)

Step 1 is coverage: how many immunotherapy patients have a molecular result at
all. If the overlap is thin the study stops there and says so.
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
print(f"immunotherapy cohort {len(I):,}, deaths {int(I['died'].sum()):,}")

FILLER = ("NOT APPLIC", "UNKNOWN", "NOT DOCUMENTED", "(NONE)", "NOT DONE",
          "NO INFORMATION", "NOT PERFORMED", "RESULTS NOT IN")
MARKERS = ["MICROSATELLITE_INSTABILITY_MSI", "KRAS", "EGFR_MUTATIONAL_ANALYSIS",
           "BRAF_MUTATIONAL_ANALYSIS", "NRAS_MUTATIONAL_ANALYSIS",
           "METHYLATION_OF_MGMT", "HER2_OVERALL_SUMMARY",
           "ESTROGEN_RECEPTOR_SUMMARY", "PROGESTERONE_RECEPTOR_SUMMARY",
           "BRAIN_MOLECULAR_MARKERS", "HIGH_RISK_CYTOGENETICS"]
ids = "','".join(I.index.astype(str))
sel = ", ".join(f"ANY_VALUE(CAST(r.{m} AS STRING)) AS {m}" for m in MARKERS)
print("pulling registry molecular markers for these patients ...")
M = C.query(f"""
  SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, {sel}
  FROM {D}.FACT_CANCER_DATA_REPOSITORY r
  JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
  WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
  GROUP BY 1""").to_dataframe().set_index("clinic")

def clean(s):
    s = s.astype(str)
    bad = s.str.upper().apply(lambda v: any(f in v for f in FILLER)) | s.isin(
        ["None", "nan", "<NA>", ""])
    return s.where(~bad)
for m in MARKERS:
    M[m] = clean(M[m])

print("\n" + "=" * 80)
print("COVERAGE: immunotherapy patients with a real molecular result")
print("=" * 80)
cov = {}
for m in MARKERS:
    s = M[m].reindex(I.index)
    cov[m] = int(s.notna().sum())
    if cov[m]:
        vals = s.value_counts().head(4)
        detail = ", ".join(f"{str(k)[:26]}={v}" for k, v in vals.items())
        print(f"  {m:<34}{cov[m]:>6,} ({cov[m]/len(I):5.1%})   {detail}")
    else:
        print(f"  {m:<34}{0:>6}")

msi = M["MICROSATELLITE_INSTABILITY_MSI"].reindex(I.index)
n_msi = int(msi.notna().sum())
print(f"\n  MSI specifically: {n_msi:,} of {len(I):,} ({n_msi/len(I):.1%})")
if n_msi < 200:
    print("  TOO THIN to model MSI. Reporting coverage only -- this is the answer:")
    print("  the biomarker that governs immunotherapy prescribing is recorded for")
    print("  almost none of the patients who received it.")

# ---------------------------------------------------------------- model if we can
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

def run(X, y, t, name):
    s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0).fit(X.iloc[tr], y.iloc[tr])
        s.append(cidx(t.iloc[te], y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    a = np.array(s)
    print(f"  {name:<44}C-index {a.mean():.3f} ± {a.std():.3f}  ({X.shape[1]} feat)")
    return a.mean()

any_mol = M[MARKERS].reindex(I.index).notna().any(axis=1)
sub = I[any_mol]
print(f"\n  patients with ANY molecular marker: {len(sub):,} "
      f"({len(sub)/len(I):.0%}), deaths {int(sub['died'].sum()):,}")

if len(sub) >= 300 and sub["died"].sum() >= 100:
    print("\n" + "=" * 80)
    print(f"SURVIVAL ON IMMUNOTHERAPY, patients with molecular data (n={len(sub):,})")
    print("=" * 80)
    MOL = pd.get_dummies(M[MARKERS].reindex(sub.index), dummy_na=True).astype(float)
    MOL = MOL.loc[:, MOL.sum() >= 20]
    site = pd.get_dummies(sub["site3"].astype(str), prefix="site").astype(float)
    LB = IL.reindex(sub.index).select_dtypes(include=[np.number])
    LB = LB.loc[:, LB.notna().sum() > len(LB) * 0.05]
    y, t = sub["died"], sub["os_days"]
    run(site, y, t, "cancer type only")
    run(pd.concat([site, MOL], axis=1), y, t, "cancer type + molecular markers")
    run(pd.concat([site, LB], axis=1), y, t, "cancer type + bloodwork")
    run(pd.concat([site, MOL, LB], axis=1), y, t, "cancer type + molecular + bloodwork")
else:
    print("\n  not enough patients with molecular data to model -- coverage IS the result")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"msi_ici | cohort={len(I)} | msi={n_msi} | any_marker={int(any_mol.sum())} "
      f"| best_marker={max(cov, key=cov.get)}({max(cov.values())})")

"""
Step 1b - resolve the 5 problem features from step1 by listing candidate
concept NAMES only (LOINC/OMOP ontology labels - no patient values, no medians).
Reads the cache step1 already built. Nothing hits BigQuery.
"""
import pandas as pd

df = pd.read_parquet("measurement_concepts.parquet")
names = df["name"].fillna("")

PROBLEMS = {
    "sign_height":    r"height|stature",
    "sign_heartrate": r"heart rate|cardiac frequency",
    "CBC_RDW":        r"distribution width|\brdw\b|red cell distribution",
    "CBC_PDW":        r"platelet distribution|\bpdw\b",
    "CBC_PCT":        r"plateletcrit|\bpct\b",
}

print("FINAL BLOCK (concept names only - safe to paste the whole thing):")
print("-" * 70)
for lab, pat in PROBLEMS.items():
    hits = df[names.str.contains(pat, case=False, regex=True, na=False)]
    hits = hits.sort_values("n_rows", ascending=False).head(5)
    if hits.empty:
        print(f"{lab} = NONE")
        continue
    parts = [f"cid{int(r.cid)}[{str(r.unit)[:8]}]{str(r.name)[:40]}"
             for _, r in hits.iterrows()]
    print(f"{lab} =")
    for p in parts:
        print(f"   {p}")
print("-" * 70)
print("END FINAL BLOCK")

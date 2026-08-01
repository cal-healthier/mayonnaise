# run.py -- self-healing chain: ensure the feature map exists, then extract.
# `pull` is defined by the fixed runner cell.
import os

if not os.path.exists("feature_map_final.csv"):
    if not os.path.exists("feature_map.csv"):
        print(">> feature_map.csv missing -- running step1 (uses cache if present) ...\n")
        exec(open(pull("step1_features.py")).read())
    print("\n>> finalizing feature map ...\n")
    exec(open(pull("finalize_features.py")).read())

print("\n>> extracting cohort ...\n")
exec(open(pull("extract_cohort.py")).read())

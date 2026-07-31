"""
Finalize the 28-feature map: apply the manual resolutions from step1b and write
feature_map_final.csv (feature, concept_id, unit, to_si_factor, note).
Reads feature_map.csv that step1 wrote. No BigQuery.
"""
import pandas as pd

fm = pd.read_csv("feature_map.csv").set_index("feature")

# unit conversions to bring Mayo values onto a standard scale
#   (factor to MULTIPLY Mayo value by; None = already fine / handle later)
TO_STD = {
    "sign_height":  (0.0254, "inches -> m"),
    "sign_weight":  (1.0,    "kg (Mayo already metric)"),
    "sign_temp":    (1.0,    "DegC"),
    "CBC_Hb":       (1.0,    "g/dL"),
    "Chem_ALB":     (1.0,    "g/dL"),
}

rows = []
for feat in fm.index:
    cid = int(fm.loc[feat, "concept_id"])
    unit = fm.loc[feat, "unit"]
    fac, note = TO_STD.get(feat, (1.0, ""))
    rows.append((feat, cid, unit, fac, note))

# --- manual fixes from step1b ---
# height: the numeric concept is 4030731 (inches)
if "sign_height" not in fm.index:
    rows.append(("sign_height", 4030731, "In", 0.0254, "added from step1b; inches -> m"))
# heart rate: Mayo stores it under the pulse concept -> alias
if "sign_pulse" in fm.index:
    pulse_cid = int(fm.loc["sign_pulse", "concept_id"])
    rows = [r for r in rows if r[0] != "sign_heartrate"]
    rows.append(("sign_heartrate", pulse_cid, "beat/minute", 1.0, "aliased to pulse (no distinct numeric HR concept)"))
# drop the ones we can't trust / that are absent
DROP = {"CBC_RDW", "CBC_PDW", "CBC_PCT"}
rows = [r for r in rows if r[0] not in DROP]

out = pd.DataFrame(rows, columns=["feature", "concept_id", "unit", "to_std_factor", "note"])
out = out.drop_duplicates("feature").sort_values("feature")
out.to_csv("feature_map_final.csv", index=False)

n = len(out)
dropped = sorted(DROP)
print(f"wrote feature_map_final.csv with {n} usable features")
print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"finalize | usable={n}/28 | dropped={','.join(dropped)} | height=ok(inches) | hr=aliased_to_pulse")

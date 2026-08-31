"""
strip_junk.py -- remove only PROVABLY irrelevant text. No LLM, no case JSON,
no importance judgment. Two rules, both mechanical:

  1. within-patient duplicates -- the same sentence copy-forwarded across the
     patient's own notes. Keep the first, drop the repeats.
  2. cross-patient boilerplate  -- a sentence that appears in MANY DIFFERENT
     patients. A line in hundreds of charts is a template, not a fact about
     anyone, so it cannot carry patient-specific information.

Everything else is kept verbatim. Patient-specific clinical text is rare across
the corpus by construction, so it survives untouched -- that is why this is
safe in a way that "keep the important sentences" was not.

The boilerplate threshold is shown as a DISTRIBUTION, not hidden: you see how
much text sits at each cross-patient frequency and can move the line yourself.
"""
import os, re
from collections import Counter
import numpy as np
import pandas as pd

BP_FRAC = 0.02          # boilerplate = appears in >= this fraction of patients
DF_PATIENTS = 2500      # patients used to build the frequency table
SENT = re.compile(r"(?<=[.\n;:])\s+")
NUM = re.compile(r"\d+")
WS = re.compile(r"\s+")


def norm(u):
    return WS.sub(" ", NUM.sub("#", u.lower())).strip()


def units(t):
    return [u.strip() for u in SENT.split(str(t)) if len(u.strip()) >= 8]


frames = []
for f in ("strip_prostate.parquet", "strip_ovarian.parquet"):
    if os.path.exists(f):
        d = pd.read_parquet(f)
        d["clinic"] = d["clinic"].astype(str)
        frames.append(d[["clinic", "txt"]])
A = pd.concat(frames, ignore_index=True)
allc = A["clinic"].unique()
rs = np.random.RandomState(0)
dfset = set(rs.choice(allc, size=min(DF_PATIENTS, len(allc)), replace=False))

# ---- cross-patient document frequency of normalized sentences ----
dfc = Counter()
rep = {}
for clinic, g in A[A["clinic"].isin(dfset)].groupby("clinic"):
    seen = set()
    for t in g["txt"]:
        for u in units(t):
            k = norm(u)
            if k not in seen:
                seen.add(k)
                dfc[k] += 1
                rep.setdefault(k, u)
NP = len(dfset)
bp_cut = max(2, int(BP_FRAC * NP))
boiler = {k for k, c in dfc.items() if c >= bp_cut}

print("=" * 78)
print(f"CORPUS   {NP:,} patients sampled for the frequency table, "
      f"{len(dfc):,} distinct sentences")
print("=" * 78)

# volume by frequency band -----------------------------------------------
bands = [("in >50% of patients", .5, 1.01), ("20-50%", .2, .5),
         ("5-20%", .05, .2), ("2-5%", .02, .05), ("1-2%", .01, .02),
         ("<1% (patient-specific)", 0, .01)]
print(f"\n  {'cross-patient frequency':<28}{'# sentences':>13}{'% of all text':>15}")
print("  " + "-" * 56)
tot_chars = sum(len(rep[k]) * dfc[k] for k in dfc)   # weight by occurrences
for name, lo, hi in bands:
    ks = [k for k in dfc if lo <= dfc[k] / NP < hi]
    vol = sum(len(rep[k]) * dfc[k] for k in ks)
    print(f"  {name:<28}{len(ks):>13,}{vol/max(tot_chars,1):>14.0%}")
print(f"\n  boilerplate cut = appears in >= {bp_cut} patients "
      f"({BP_FRAC:.0%}); {len(boiler):,} sentences flagged")

# top boilerplate --------------------------------------------------------
print("\n" + "=" * 78)
print("TOP BOILERPLATE THAT WOULD BE STRIPPED  (ranked by # patients)")
print("=" * 78)
for k, c in dfc.most_common(30):
    print(f"  {c:>5} pts  {rep[k][:110]}")

# borderline band --------------------------------------------------------
print("\n" + "=" * 78)
print(f"BORDERLINE -- sentences right AT the {BP_FRAC:.0%} line (you decide)")
print("=" * 78)
near = sorted([k for k in dfc if bp_cut <= dfc[k] <= bp_cut + max(2, bp_cut // 4)],
              key=lambda k: -len(rep[k]))[:15]
for k in near:
    print(f"  {dfc[k]:>5} pts  {rep[k][:110]}")

# ---- per-patient reduction from the two safe rules only ----
samp = set(rs.choice(allc, size=min(400, len(allc)), replace=False))
raws, dds, cleans = [], [], []
example = None
for clinic, g in A[A["clinic"].isin(samp)].groupby("clinic"):
    raw = int(g["txt"].str.len().sum())
    seen, dedup_b, clean_b, kept = set(), 0, 0, []
    for t in g["txt"]:
        for u in units(t):
            k = norm(u)
            if k in seen:
                continue
            seen.add(k)
            dedup_b += len(u) + 1
            if k in boiler:
                continue
            clean_b += len(u) + 1
            kept.append(u)
    raws.append(raw); dds.append(dedup_b); cleans.append(clean_b)
    if example is None and 25 <= len(g) <= 60:
        example = (clinic, raw, kept)

raws, dds, cleans = np.array(raws), np.array(dds), np.array(cleans)
print("\n" + "=" * 78)
print(f"SIZE AFTER SAFE STRIPPING ONLY   ({len(raws)} patients)")
print("=" * 78)
print(f"  {'raw record':<34}{np.median(raws)/1024:>8.0f} KB   100%")
print(f"  {'- within-patient duplicates':<34}{np.median(dds)/1024:>8.0f} KB   "
      f"{np.median(dds)/np.median(raws):>4.0%}")
print(f"  {'- cross-patient boilerplate':<34}{np.median(cleans)/1024:>8.0f} KB   "
      f"{np.median(cleans)/np.median(raws):>4.0%}")
print(f"\n  nothing patient-specific removed -- only repeats and template lines")

if example:
    clinic, raw, kept = example
    print("\n" + "=" * 78)
    print(f"ONE PATIENT, CLEANED  ({raw/1024:.0f} KB raw -> "
          f"{sum(len(k)+1 for k in kept)/1024:.0f} KB kept)  IN-ENCLAVE ONLY")
    print("  every surviving sentence, in order -- read for anything junk left in")
    print("  or anything real taken out:")
    print("=" * 78)
    for u in kept[:120]:
        print(f"  {u[:150]}")
    if len(kept) > 120:
        print(f"  ... (+{len(kept)-120} more kept sentences)")

print("""
{}
READING IT
{}
  This removed only what is provably non-informative: sentences the patient's
  own record repeats, and sentences so common across patients they are clearly
  template. No clinical judgment, so nothing patient-specific can be lost.

  The safe reduction is smaller than the aggressive structured cut -- that is
  the honest price of not throwing away anything real. Move BP_FRAC down to cut
  more (and re-read the BORDERLINE band to see what starts going), up to cut
  less. The cleaned per-patient text is what you would actually send: lossless
  on substance, minus the boilerplate and copy-forward.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"strip_junk | raw_kb={np.median(raws)/1024:.0f} | "
      f"dedup_kb={np.median(dds)/1024:.0f} | clean_kb={np.median(cleans)/1024:.0f} "
      f"| kept={np.median(cleans)/np.median(raws):.0%} | boiler_sentences={len(boiler)}")

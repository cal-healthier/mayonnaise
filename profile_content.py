"""
profile_content.py -- what IS the text, by content type? (diagnostic, not a strip)

Frequency stripping only catches verbatim-repeated template lines (~30%). The
bigger deterministic targets are defined by WHAT A LINE IS, independent of how
often it repeats:

  NUMERIC_TABLE   lab/vitals dumps -- mostly numbers/units/pipes. We already
                  hold every lab value structured in measurement (23.6B rows),
                  so this text is REDUNDANT with data we keep separately.
  MED_LINE        medication-list dosing lines. Redundant with FACT_ORDERS.
  ADMIN_META      facility names, addresses, providers, billing, insurance,
                  author/source routing.
  DATE_TS         bare dates / timestamps.
  HEADER          section labels ("General:", "IMPRESSION:").
  ENCODED         %7C / %0A / pipe-delimited table rows (extraction artifacts).
  NORMAL_EXAM     template exam/ROS phrases ("no acute distress", "alert and
                  oriented", "ready to learn"). JUDGMENT bucket -- disease
                  negatives like "no evidence of metastasis" are kept in PROSE.
  PROSE           everything else -- the real narrative.

For each bucket: share of sentences, share of CHARACTER volume, and samples.
Then projected size if we strip each group. Paste it back and we decide which
buckets to cut and where 10% actually lands.
"""
import os, re
import numpy as np
import pandas as pd

SENT = re.compile(r"(?<=[.\n;:])\s+")
rs = np.random.RandomState(0)


def units(t):
    return [u.strip() for u in SENT.split(str(t)) if len(u.strip()) >= 6]


MED = re.compile(r"(?i)\b(mg|mcg|ml|tablet|capsule|\bcap\b|\bpo\b|\bbid\b|\btid\b|"
                 r"\bqhs\b|\bqam\b|\bqd\b|daily|disp:|take \d)")
ADMIN = re.compile(r"(?i)medical center|hospital|\bclinic\b|\bM\.?D\.?\b|\bR\.?N\.?\b|"
                   r"billing|encounter location|author:|\bsource:|referral source|"
                   r"insurance|\bfax\b|\bphone\b|margin code|provider name|"
                   r"\bsuite\b|\d{3}[.\- ]\d{3}[.\- ]\d{4}")
ADDR = re.compile(r"(?i)\b\d{2,5}\s+\w+.{0,20}\b(street|st|ave|avenue|road|rd|drive|"
                  r"dr|blvd|lane|ln|way|court|ct)\b|\b[A-Z]{2}\s+\d{5}\b")
DATEONLY = re.compile(r"^[\d/:\.\-\s]{6,}$|^\d{1,2}[A-Za-z]{3}\d{2,4}")
NORMAL = re.compile(r"(?i)no acute distress|alert and oriented|normocephalic|"
                    r"within normal limits|unremarkable|systems were negative|"
                    r"were reviewed and updated|noncontributory|well[- ]nourished|"
                    r"no apparent|ready to learn|learning (barriers|preferences)|"
                    r"atraumatic|normal (bowel|breath) sounds|no clubbing|"
                    r"moist mucous|regular rate and rhythm")


def classify(s):
    low = s.lower()
    alpha = sum(c.isalpha() for c in s)
    digit = sum(c.isdigit() for c in s)
    fa = alpha / max(len(s), 1)
    if s.count("%7C") + s.count("%0A") >= 2 or s.count("|") >= 3:
        return "ENCODED"
    if fa < 0.45 and digit >= 3:
        return "NUMERIC_TABLE"
    if MED.search(s) and digit >= 1 and (re.search(r"\(\w", s) or "take" in low
                                         or "disp" in low or "tablet" in low):
        return "MED_LINE"
    if ADMIN.search(s) or ADDR.search(s):
        return "ADMIN_META"
    if DATEONLY.match(s.strip()):
        return "DATE_TS"
    if (len(s) <= 34 and s.rstrip().endswith(":")) or (s.isupper() and len(s) <= 42):
        return "HEADER"
    if NORMAL.search(s):
        return "NORMAL_EXAM"
    return "PROSE"


frames = []
for f in ("strip_prostate.parquet", "strip_ovarian.parquet"):
    if os.path.exists(f):
        d = pd.read_parquet(f); d["clinic"] = d["clinic"].astype(str)
        frames.append(d[["clinic", "txt"]])
A = pd.concat(frames, ignore_index=True)
samp = set(rs.choice(A["clinic"].unique(),
                     size=min(300, A["clinic"].nunique()), replace=False))
A = A[A["clinic"].isin(samp)]

from collections import defaultdict
vol = defaultdict(int); cnt = defaultdict(int); ex = defaultdict(list)
tot_vol = 0
for t in A["txt"]:
    for u in units(t):
        b = classify(u)
        vol[b] += len(u); cnt[b] += 1; tot_vol += len(u)
        if len(ex[b]) < 12 and rs.rand() < 0.05:
            ex[b].append(u)

ORDER = ["NUMERIC_TABLE", "MED_LINE", "ENCODED", "ADMIN_META", "DATE_TS",
         "HEADER", "NORMAL_EXAM", "PROSE"]
print("=" * 80)
print(f"CONTENT PROFILE   {len(A['clinic'].unique())} patients, "
      f"{sum(cnt.values()):,} sentences, {tot_vol/1024:.0f} KB total")
print("=" * 80)
print(f"  {'bucket':<16}{'% sentences':>12}{'% of TEXT':>11}{'redundant?':>22}")
print("  " + "-" * 62)
RED = {"NUMERIC_TABLE": "yes: measurement tbl", "MED_LINE": "yes: FACT_ORDERS",
       "ENCODED": "artifact", "ADMIN_META": "no clinical value",
       "DATE_TS": "bare timestamps", "HEADER": "labels only",
       "NORMAL_EXAM": "JUDGMENT", "PROSE": "-- KEEP --"}
for b in ORDER:
    print(f"  {b:<16}{cnt[b]/max(sum(cnt.values()),1):>11.0%}"
          f"{vol[b]/max(tot_vol,1):>11.0%}{RED[b]:>22}")

for b in ORDER:
    print(f"\n{'='*80}\n{b}  ({vol[b]/max(tot_vol,1):.0%} of text) -- {RED[b]}\n{'='*80}")
    for u in ex[b]:
        print(f"  {u[:150]}")

safe = ["NUMERIC_TABLE", "MED_LINE", "ENCODED", "ADMIN_META", "DATE_TS", "HEADER"]
safe_vol = sum(vol[b] for b in safe)
prose_vol = vol["PROSE"]
norm_vol = vol["NORMAL_EXAM"]
print("\n" + "=" * 80)
print("PROJECTED SIZE (this windowed sample; full-history dedup would cut more)")
print("=" * 80)
print(f"  raw                                     100%")
print(f"  - tables/meds/admin/dates/headers/enc   {1-safe_vol/tot_vol:>4.0%}  "
      f"(all redundant-with-structured or pure metadata)")
print(f"  - also normal-exam boilerplate          {1-(safe_vol+norm_vol)/tot_vol:>4.0%}  "
      f"(judgment call)")
print(f"  = PROSE only                            {prose_vol/tot_vol:>4.0%}")
print(f"""
  Redundant note: labs and med-lines duplicate the structured measurement /
  FACT_ORDERS tables we already hold -- stripping them from text loses nothing
  at the SYSTEM level, since the numbers still travel structured. That is the
  cleanest large cut and it is not frequency-based.

  Paste this back: tell me which buckets to strip and I will (a) build the
  deterministic stripper, (b) confirm it on samples, (c) report the real % --
  and say honestly whether 10% is reachable losslessly or needs the full-record
  dedup on top.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"profile | prose={prose_vol/tot_vol:.0%} | safe_strip_to={1-safe_vol/tot_vol:.0%} "
      f"| +normal={1-(safe_vol+norm_vol)/tot_vol:.0%}")

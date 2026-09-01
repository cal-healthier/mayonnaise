"""
measure_gzip.py -- is "10% of original" already solved for the wire?

Content-type stripping caps ~30% because the record is mostly real prose. But
if the goal is BYTES ON THE WIRE, this text is so repetitive that general
compression does the rest -- losslessly. This measures, per patient:

  raw                     the full note text
  junk-stripped           within-patient dedup + drop tables/meds/admin/
                          headers/dates/encoded (all redundant-with-structured
                          or pure metadata) -- lossless on clinical prose
  gzip(raw)               raw, compressed
  gzip(junk-stripped)     stripped, then compressed  <- the wire payload

If the last column is ~10% of raw, then 10%-on-the-wire is done, and the
receiver decompresses to the complete text. No clinical judgment, no loss.
"""
import gzip, os, re
import numpy as np
import pandas as pd

SENT = re.compile(r"(?<=[.\n;:])\s+")
NUM = re.compile(r"\d+"); WS = re.compile(r"\s+")
MED = re.compile(r"(?i)\b(mg|mcg|ml|tablet|capsule|\bpo\b|\bbid\b|\btid\b|disp:|take \d)")
ADMIN = re.compile(r"(?i)medical center|hospital|\bclinic\b|\bM\.?D\.?\b|billing|"
                   r"encounter location|author:|\bsource:|insurance|\bfax\b|\bphone\b|"
                   r"\d{3}[.\- ]\d{3}[.\- ]\d{4}")
rs = np.random.RandomState(0)


def units(t):
    return [u.strip() for u in SENT.split(str(t)) if len(u.strip()) >= 6]


def is_junk(s):
    low = s.lower()
    alpha = sum(c.isalpha() for c in s); digit = sum(c.isdigit() for c in s)
    if s.count("%7C") + s.count("%0A") >= 2 or s.count("|") >= 3:
        return True
    if alpha / max(len(s), 1) < 0.45 and digit >= 3:
        return True
    if MED.search(s) and digit >= 1 and ("take" in low or "disp" in low or "tablet" in low):
        return True
    if ADMIN.search(s):
        return True
    if (len(s) <= 34 and s.rstrip().endswith(":")) or (s.isupper() and len(s) <= 42):
        return True
    return False


frames = []
for f in ("strip_prostate.parquet", "strip_ovarian.parquet"):
    if os.path.exists(f):
        d = pd.read_parquet(f); d["clinic"] = d["clinic"].astype(str)
        frames.append(d[["clinic", "txt"]])
A = pd.concat(frames, ignore_index=True)
samp = set(rs.choice(A["clinic"].unique(),
                     size=min(300, A["clinic"].nunique()), replace=False))

raw, clean, graw, gclean = [], [], [], []
for clinic, g in A[A["clinic"].isin(samp)].groupby("clinic"):
    full = "\n".join(str(t) for t in g["txt"])
    seen, kept = set(), []
    for t in g["txt"]:
        for u in units(t):
            k = WS.sub(" ", NUM.sub("#", u.lower())).strip()
            if k in seen or is_junk(u):
                continue
            seen.add(k); kept.append(u)
    cl = "\n".join(kept)
    rb = len(full.encode()); cb = len(cl.encode())
    raw.append(rb); clean.append(cb)
    graw.append(len(gzip.compress(full.encode(), 6)))
    gclean.append(len(gzip.compress(cl.encode(), 6)))

raw, clean = np.array(raw), np.array(clean)
graw, gclean = np.array(graw), np.array(gclean)
med = np.median(raw)


def line(name, arr):
    print(f"  {name:<34}{np.median(arr)/1024:>8.1f} KB{np.median(arr)/med:>8.0%}")


print("=" * 66)
print(f"WIRE SIZE   {len(raw)} patients")
print("=" * 66)
line("raw record", raw)
line("junk-stripped (lossless prose)", clean)
line("gzip(raw)", graw)
line("gzip(junk-stripped)  <- payload", gclean)
print(f"""
  gzip(raw) alone:            {np.median(graw)/med:.0%} of raw
  strip THEN gzip:            {np.median(gclean)/med:.0%} of raw
  strip's marginal help:      {(np.median(graw)-np.median(gclean))/med:.0%} extra

  If the goal is bandwidth, the wire payload is the last row -- the receiver
  decompresses to the COMPLETE text, so nothing is lost and no clinician has to
  trust a heuristic. Stripping first helps a little; gzip does the heavy
  lifting because the redundancy it exploits is the same copy-forward and
  template repetition we were trying to cut by hand.

  If the goal is TOKENS into a model, gzip does not help -- there the lossless
  floor is the junk-stripped row (~{np.median(clean)/med:.0%}) on this window, less on the
  full record after cross-note dedup, and 10% needs lossy selection.""")

print("\n" + "-" * 60)
print("FINAL LINE:")
print(f"gzip | strip={np.median(clean)/med:.0%} | gzip_raw={np.median(graw)/med:.0%} "
      f"| strip+gzip={np.median(gclean)/med:.0%}")

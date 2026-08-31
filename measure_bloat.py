"""
measure_bloat.py -- how much of a patient's record is signal vs bloat, and how
small can the clinically-useful case get?

Runs a COMPRESSION FUNNEL on cached note text (strip_*.parquet: per note, with
raw / header-stripped / plumbing-stripped versions and note type). No BigQuery
scan.

  RAW            all note text
  - headers      drop "Result Type / Verified By / *Final* / Entered by"
  - plumbing     drop From/To/Sent/Subject/Reference/phone lines
  - DEDUP        drop lines repeated across the patient's notes (copy-forward
                 is the single biggest source of bloat)
  - signal only  keep lines that carry a clinical fact (dx/stage/path/plan/
                 biomarker/treatment/response)
  = STRUCTURED   regex -> compact case JSON: cancer, stage, histology,
                 biomarkers, treatments, ECOG, markers, active problems

IMPORTANT: absolute sizes here are a LOWER BOUND -- the cached sample caps each
patient at <=96 notes and 4,000 chars/note, while the true record is ~2.4 MB
(median 257 notes). So real dedup savings are LARGER than shown (more notes =
more repetition). The RATIOS transfer, and the structured-JSON size is absolute
(a case summary is a case summary regardless of input size).

One example case JSON is printed -- IN-ENCLAVE ONLY, like inspect_notes. The
FINAL LINE carries sizes only.
"""
import json, os, re
import numpy as np
import pandas as pd

NORM = re.compile(r"\d+")
WS = re.compile(r"\s+")
SIGNAL = re.compile(
    r"(?i)(assessment|impression|\bplan\b|diagnos|\bstage\b|gleason|grade group|"
    r"metasta|biops|patholog|positive for|negative for|mutation|amplif|\bbrca\b|"
    r"\bmsi\b|her2|\btmb\b|ecog|performance status|start(ed|ing)|initiat|"
    r"discontinu|\bheld\b|cycle|progress|recurren|response|remission|resect|"
    r"debulk|residual|ascites|effusion|\brisk\b)")

STAGE = re.compile(r"(?i)\bstage\s+(0|I{1,3}V?|IV|[1-4])[ABC]?\b|"
                   r"\b(?:y?p|c)?T[0-4][a-cx]?\s?N[0-3X]\s?M[01X]\b|gleason\s+[0-9+ =]{1,7}|"
                   r"grade group\s+[1-5]|\bmCRPC\b|castrat|metastatic|biochemical recurrence")
HISTO = re.compile(r"(?i)(adeno)?carcinoma|sarcoma|serous|mucinous|clear cell|"
                   r"squamous|melanoma|glioblastoma|lymphoma")
BIOM = re.compile(r"(?i)\bBRCA[12]?\b|\bMSI\b|microsatellite|\bHER2\b|\bTMB\b|"
                  r"foundation ?one|\bPD-?L1\b|\bKRAS\b|\bEGFR\b|\bALK\b|amplif|"
                  r"[A-Z]{2,5}\s+(mutation|amplification|wild.?type)")
DRUG = re.compile(r"(?i)abiraterone|enzalutamide|apalutamide|darolutamide|docetaxel|"
                  r"cabazitaxel|carboplatin|cisplatin|paclitaxel|gemcitabine|"
                  r"doxorubicin|topotecan|bevacizumab|olaparib|niraparib|rucaparib|"
                  r"pembrolizumab|leuprolide|goserelin|degarelix|radium|lutetium|sipuleucel")
ECOG = re.compile(r"(?i)\bECOG\b(?:\s+PS)?(?:\s+(?:of|is|was))?[\s:=-]*([0-4])\b")
SENT = re.compile(r"(?<=[.\n;:])\s+")


def units(t):
    return [u.strip() for u in SENT.split(str(t)) if len(u.strip()) >= 8]


def kb(b):
    return b / 1024.0


def dedup_bytes(texts):
    """keep the first occurrence of each normalized line across the record"""
    seen, kept = set(), 0
    for t in texts:
        for u in units(t):
            key = WS.sub(" ", NORM.sub("#", u.lower()))
            if key not in seen:
                seen.add(key)
                kept += len(u) + 1
    return kept, seen


def signal_bytes(texts):
    seen, kept, n = set(), 0, 0
    for t in texts:
        for u in units(t):
            if not SIGNAL.search(u):
                continue
            key = WS.sub(" ", NORM.sub("#", u.lower()))
            if key not in seen:
                seen.add(key)
                kept += len(u) + 1
                n += 1
    return kept, n


def build_case(cancer, texts):
    joined = "\n".join(str(t) for t in texts)
    def last(rx):
        m = list(rx.finditer(joined))
        return WS.sub(" ", m[-1].group(0)).strip()[:40] if m else None
    def uniq(rx, k=6):
        return sorted({WS.sub(" ", x.group(0)).strip().upper()[:24]
                       for x in rx.finditer(joined)})[:k]
    ecs = [int(m.group(1)) for m in ECOG.finditer(joined)]
    problems = []
    seen = set()
    for u in units(joined):
        if 12 < len(u) < 200 and SIGNAL.search(u):
            key = WS.sub(" ", NORM.sub("#", u.lower()))
            if key not in seen:
                seen.add(key)
                problems.append(u[:180])
        if len(problems) >= 10:
            break
    case = {
        "cancer": cancer,
        "stage": last(STAGE),
        "histology": last(HISTO),
        "biomarkers": uniq(BIOM),
        "treatments": uniq(DRUG, 8),
        "ecog": ecs[-1] if ecs else None,
        "problems": problems,
    }
    return {k: v for k, v in case.items() if v}


rows = []
example = None
for tag, f in (("prostate", "strip_prostate.parquet"),
               ("ovarian", "strip_ovarian.parquet")):
    if not os.path.exists(f):
        print(f"  {f} missing -- run gpu_strip.py first")
        continue
    N = pd.read_parquet(f)
    N["clinic"] = N["clinic"].astype(str)
    # sample ~200 patients/cohort -- the per-patient regex funnel is pure
    # Python and processing all ~6,500 takes minutes for no gain in the medians
    cl = N["clinic"].unique()
    rs = np.random.RandomState(0)
    keep = set(rs.choice(cl, size=min(200, len(cl)), replace=False))
    N = N[N["clinic"].isin(keep)]
    for clinic, g in N.groupby("clinic"):
        raw = g["txt"].str.len().sum()
        nohdr = g["stripped"].str.len().sum()
        plumb = g["clinical"].str.len().sum()
        dd, _ = dedup_bytes(g["clinical"])
        sig, nsig = signal_bytes(g["clinical"])
        cj = build_case(tag, g["clinical"].tolist())
        struct = len(json.dumps(cj, separators=(",", ":")))
        rows.append((tag, raw, nohdr, plumb, dd, sig, struct, len(g)))
        if example is None and nsig >= 5 and len(g) >= 20:
            example = (clinic, tag, cj, raw, struct)

R = pd.DataFrame(rows, columns=["tag", "raw", "nohdr", "plumb", "dedup",
                                "signal", "struct", "n_notes"])

print("=" * 76)
print(f"COMPRESSION FUNNEL   {len(R):,} patients sampled   (sizes are a floor)")
print("=" * 76)
print(f"  {'stage':<22}{'median':>10}{'% of raw':>10}{'p90':>10}")
print("  " + "-" * 54)
base = R["raw"].median()
for col, name in [("raw", "RAW record"),
                  ("nohdr", "- note headers"),
                  ("plumb", "- message plumbing"),
                  ("dedup", "- DEDUP (copy-forward)"),
                  ("signal", "- signal lines only"),
                  ("struct", "= STRUCTURED case JSON")]:
    med = R[col].median()
    unit = "KB" if med < 1024 * 200 else "MB"
    val = kb(med) if unit == "KB" else med / 1e6
    p90 = kb(R[col].quantile(.9)) if unit == "KB" else R[col].quantile(.9) / 1e6
    print(f"  {name:<22}{val:>8.1f}{unit}{med/base:>9.0%}{p90:>8.1f}{unit}")

print(f"""
  notes/patient in sample: median {R['n_notes'].median():.0f} (real median 257)
  signal lines kept: median {R['signal'].median()/60:.0f} distinct facts/patient
""")

print("=" * 76)
print("WHERE THE BLOAT IS")
print("=" * 76)
r = R.drop(columns=["tag"]).median()
print(f"  headers + plumbing:  {(1-r['plumb']/r['raw']):.0%} of raw is note")
print(f"                       scaffolding (who typed it, when, message routing)")
print(f"  copy-forward:        dedup removes another "
      f"{(r['plumb']-r['dedup'])/r['raw']:.0%} -- the SAME history")
print(f"                       pasted into every progress note")
print(f"  non-clinical prose:  keeping only fact-bearing lines drops another "
      f"{(r['dedup']-r['signal'])/r['raw']:.0%}")
print(f"  => the clinically-useful CORE is ~{r['signal']/r['raw']:.0%} of the "
      f"record as text,")
print(f"     and ~{r['struct']/r['raw']:.1%} once structured into fields.")

if example:
    clinic, tag, cj, eraw, estruct = example
    print("\n" + "=" * 76)
    print(f"EXAMPLE STRUCTURED CASE  (IN-ENCLAVE ONLY -- do not copy out)")
    print(f"  this patient: {eraw/1024:.0f} KB of notes -> {estruct} B of JSON "
          f"({eraw/max(estruct,1):.0f}x smaller)")
    print("=" * 76)
    print(json.dumps(cj, indent=2)[:1400])

print(f"""
{'=' * 76}
READING IT
{'=' * 76}
  A patient's record compresses to KB not because we throw away medicine, but
  because clinical notes are mostly NON-medicine: template scaffolding, message
  routing, and the same history copy-pasted into hundreds of notes. The actual
  clinical facts -- stage, histology, biomarkers, treatments, response, active
  problems -- are a few KB.

  For an API payload you would send the STRUCTURED case (last row): ~{R['struct'].median()/1024:.1f} KB,
  built by the in-enclave Gemini extractor, not the raw record. That is a
  ~{base/R['struct'].median():.0f}x reduction on the cached sample and larger on the true
  record. Bandwidth stops being a constraint; a whole cohort is a few MB.

  Caveat worth stating to anyone clinical: aggressive structuring can DROP
  nuance the free text carried (a hedge, a patient preference, a subtle
  finding). The right payload is the structured core PLUS the handful of
  verbatim signal lines that did not fit a field -- still KB, but lossless on
  the parts that matter.""")

print("\n" + "-" * 72)
print("FINAL LINE:")
print(f"bloat | raw_kb={base/1024:.0f} | signal_kb={R['signal'].median()/1024:.1f} "
      f"| struct_kb={R['struct'].median()/1024:.1f} "
      f"| reduction={base/R['struct'].median():.0f}x")

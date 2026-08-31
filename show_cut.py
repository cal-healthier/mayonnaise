"""
show_cut.py -- eyeball what the compression THROWS AWAY.

No model, no scoring. For a few patients with COMPLETE note text
(inspect_*_0.parquet, uncapped), it sorts every sentence into buckets and
prints them so you can judge whether anything clinical is being lost:

  KEPT      the ~1 KB structured case (fields + the problems that made it in)
  DROPPED-SIGNAL   sentences that DID carry a clinical cue but did NOT make the
                   case -- the danger zone. Each is tagged [covered: ...] if the
                   entity it mentions is already in a case field, else [NEW?].
                   Read the [NEW?] ones: those are the only places real loss
                   could hide.
  CUT-NONSIGNAL    a random sample of the sentences dropped as non-clinical --
                   confirm by eye that it is boilerplate, not medicine
  COPY-FORWARD     the most-repeated sentences and how many times each appeared

Patient text is IN-ENCLAVE ONLY; read on screen, copy out just the FINAL LINE.
"""
import json, os, re
from collections import Counter
import numpy as np
import pandas as pd

NORM = re.compile(r"\d+")
WS = re.compile(r"\s+")
SENT = re.compile(r"(?<=[.\n;:])\s+")
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
                  r"foundation ?one|\bPD-?L1\b|\bKRAS\b|\bEGFR\b|\bALK\b")
DRUG = re.compile(r"(?i)abiraterone|enzalutamide|apalutamide|darolutamide|docetaxel|"
                  r"cabazitaxel|carboplatin|cisplatin|paclitaxel|gemcitabine|"
                  r"doxorubicin|topotecan|bevacizumab|olaparib|niraparib|rucaparib|"
                  r"pembrolizumab|leuprolide|goserelin|degarelix|radium|lutetium")
ECOG = re.compile(r"(?i)\bECOG\b(?:\s+PS)?(?:\s+(?:of|is|was))?[\s:=-]*([0-4])\b")


def units(t):
    return [u.strip() for u in SENT.split(str(t)) if len(u.strip()) >= 8]


def key(u):
    return WS.sub(" ", NORM.sub("#", u.lower()))


def build_case(cancer, texts):
    joined = "\n".join(str(t) for t in texts)
    def last(rx):
        m = list(rx.finditer(joined))
        return WS.sub(" ", m[-1].group(0)).strip()[:40] if m else None
    def uniq(rx, k=8):
        return sorted({WS.sub(" ", x.group(0)).strip().upper()[:24]
                       for x in rx.finditer(joined)})[:k]
    ecs = [int(m.group(1)) for m in ECOG.finditer(joined)]
    problems, seen = [], set()
    for u in units(joined):
        if 12 < len(u) < 200 and SIGNAL.search(u):
            k = key(u)
            if k not in seen:
                seen.add(k); problems.append(u[:180])
        if len(problems) >= 10:
            break
    case = {"cancer": cancer, "stage": last(STAGE), "histology": last(HISTO),
            "biomarkers": uniq(BIOM), "treatments": uniq(DRUG, 8),
            "ecog": ecs[-1] if ecs else None, "problems": problems}
    return {k: v for k, v in case.items() if v}


src = next((f for f in ("inspect_ovarian_0.parquet", "inspect_prostate_0.parquet")
            if os.path.exists(f)), None)
if not src:
    raise SystemExit("need inspect_*_0.parquet (full-text notes) -- run inspect_notes.py")
tag = "ovarian" if "ovarian" in src else "prostate"
Z = pd.read_parquet(src)
Z["clinic"] = Z["clinic"].astype(str)

sizes = Z.groupby("clinic").size().sort_values(ascending=False)
picks = list(sizes.index[:3])                       # the 3 with the most notes
rs = np.random.RandomState(1)

for pi, clinic in enumerate(picks, 1):
    g = Z[Z["clinic"] == clinic]
    texts = g["txt"].tolist()
    case = build_case(tag, texts)
    case_blob = json.dumps(case).lower()
    prob_keys = {key(p) for p in case.get("problems", [])}

    sig, nonsig, seen_s, seen_n = [], [], set(), set()
    cnt = Counter()
    for t in texts:
        for u in units(t):
            cnt[key(u)] += 1
            if SIGNAL.search(u):
                if key(u) not in seen_s:
                    seen_s.add(key(u)); sig.append(u)
            elif len(u) > 15 and key(u) not in seen_n:
                seen_n.add(key(u)); nonsig.append(u)

    dropped_sig = [u for u in sig if key(u) not in prob_keys]

    print("\n" + "=" * 78)
    print(f"PATIENT {pi}/{len(picks)}  ({len(g)} notes, {len(sig)} distinct signal "
          f"sentences, {len(nonsig)} non-signal)")
    print("=" * 78)
    print("KEPT -- the structured case that would be sent (~%d B):"
          % len(json.dumps(case)))
    print("  " + json.dumps(case, indent=2).replace("\n", "\n  ")[:1200])

    print(f"\nDROPPED-SIGNAL -- clinical-cue sentences NOT in the case "
          f"({len(dropped_sig)}); read the [NEW?] ones:")
    shown = 0
    for u in dropped_sig:
        ul = u.lower()
        ents = [w for w in re.findall(r"[a-z]{4,}", ul)
                if w in case_blob and w not in
                ("with", "that", "this", "have", "which", "were", "cancer",
                 "disease", "patient", "prostate", "ovarian")]
        tag_txt = f"[covered: {ents[0]}]" if ents else "[NEW?]"
        print(f"  {tag_txt:<18}{u[:130]}")
        shown += 1
        if shown >= 40:
            print(f"  ... (+{len(dropped_sig)-40} more)")
            break

    print(f"\nCUT-NONSIGNAL -- random sample of the fully-dropped non-clinical "
          f"lines (of {len(nonsig)}):")
    for u in (rs.choice(nonsig, size=min(20, len(nonsig)), replace=False)
              if nonsig else []):
        print(f"  {u[:130]}")

    print("\nCOPY-FORWARD -- most-repeated sentences (count x):")
    inv = {}
    for t in texts:
        for u in units(t):
            inv.setdefault(key(u), u)
    for k, c in cnt.most_common(40):
        if c >= 3 and len(inv.get(k, "")) > 25:
            print(f"  {c:>3}x  {inv[k][:120]}")

print("""
{}
HOW TO JUDGE IT
{}
  The only bucket that can hide real loss is DROPPED-SIGNAL tagged [NEW?].
  Read those lines. If they are restatements ("continue current therapy",
  "will follow", "tolerating well") the cut is safe. If any names a real
  event the case does not carry -- a new lesion, a complication, a changed
  plan -- then the case selection (top-10 problems) is too tight and we widen
  it. That is a threshold you set by reading, not a model.

  CUT-NONSIGNAL should read as pure boilerplate; if medicine is in there, the
  SIGNAL filter is too aggressive.  COPY-FORWARD shows the repetition the size
  numbers claim -- confirm the high-count lines are genuinely the same fact.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"show_cut | src={src} | patients={len(picks)} | inspect DROPPED-SIGNAL [NEW?] by eye")

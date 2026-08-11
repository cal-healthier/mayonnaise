"""
inspect_notes.py -- read the actual input with your own eyes.

Prints EVERY pre-treatment note for 10 patients, with the model's blind spots
marked inline. Nothing is capped and nothing is truncated in what you see --
the point is to show what the pipeline throws away, not to reproduce it.

Three things are marked on every note:

  [NOT SEEN - beyond the 6-note cap]   the query keeps only the 6 most recent
                                       notes. Notes 7+ are printed here so you
                                       can judge whether that cap costs us.
  >>> 256-TOKEN CUT <<<                where max_seq_length=256 stops reading.
                                       Everything after it is invisible to the
                                       model. Computed with the REAL tokenizer,
                                       not estimated from character counts.
  >>> 512-TOKEN CUT <<<                the same for the 512 arm.

It also tests a specific hypothesis. Clinical notes run header -> history ->
exam -> labs -> ASSESSMENT AND PLAN, and the A&P at the end is where the
physician says what they actually think. If the A&P consistently falls after
the 256-token cut, the model has never once read the doctor's opinion, and
that -- not the choice of encoder -- is why the number is stuck at 0.63.
The summary at the bottom counts how often that happens.

The text is shown AS THE MODEL GETS IT (numerals stripped) alongside the
original, because that substitution is easy to forget and changes how notes
read.

PATIENT TEXT STAYS IN THE ENCLAVE. Read it on screen; copy back only the
FINAL LINE, which carries counts and no clinical content.

  SEED = 1, 2, 3 ...  for a different ten patients
  COHORT = "prostate" for the men
"""
import os, re, textwrap
import numpy as np
import pandas as pd
from google.cloud import bigquery

COHORT = "ovarian"
SEED = 0
N_PATIENTS = 10
BALANCED = True     # 5 who progressed, 5 who did not -- see note below

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
LAB, MARK = {"ovarian":  ("ov_label.parquet", "ov_ca125.parquet"),
             "prostate": ("psa_progression.parquet", "psa_values.parquet")}[COHORT]

E = pd.read_parquet(LAB)
tx = pd.read_parquet(MARK).groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]

rs = np.random.RandomState(SEED)
if BALANCED:
    # A 24% event rate means a pure random 10 would usually give 2 progressors
    # and you would learn little about the contrast. Set BALANCED = False for
    # a genuinely random draw.
    pos = E[E["y"] == 1].sample(min(N_PATIENTS // 2, int(E["y"].sum())), random_state=rs)
    neg = E[E["y"] == 0].sample(N_PATIENTS - len(pos), random_state=rs)
    SEL = pd.concat([pos, neg]).sample(frac=1, random_state=rs)
else:
    SEL = E.sample(N_PATIENTS, random_state=rs)

cache = f"inspect_{COHORT}_{SEED}.parquet"
if os.path.exists(cache):
    N = pd.read_parquet(cache)
    print(f"cached: {len(N):,} notes for {N['clinic'].nunique()} patients\n")
else:
    rows = ",".join(f"('{c}',DATE '{d.date()}')"
                    for c, d in zip(SEL.index.astype(str), SEL["tx_date"]))
    # NO SUBSTR and NO note cap -- we want the whole record for these ten
    sql = f"""
    WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, tx DATE>>[{rows}])),
    pk AS (SELECT c.clinic, c.tx, pe.person_id FROM coh c
           JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
    SELECT pk.clinic,
           CAST(n.note_text AS STRING) AS txt,
           CAST(n.note_title AS STRING) AS title,
           DATE_DIFF(pk.tx, DATE(n.note_date), DAY) AS days_before,
           ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) AS rn
    FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
    WHERE n.note_text IS NOT NULL
      AND DATE(n.note_date) <= pk.tx
      AND DATE(n.note_date) >= DATE_SUB(pk.tx, INTERVAL 365 DAY)
    """
    job = C.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True))
    print(f"pulling every note for {len(SEL)} patients "
          f"({job.total_bytes_processed/1e9:.0f} GB) ...")
    N = C.query(sql).to_dataframe()
    N.to_parquet(cache)
    print(f"{len(N):,} notes\n")

# the pipeline's own filters, so we can mark which notes actually made it in
TITLES = ('Progress Notes', 'Consults - Outpatient', 'H&P')
N["passes_filter"] = (N["title"].isin(TITLES)
                      & N["txt"].str.len().between(400, 25000))
N["used"] = N["passes_filter"] & (
    N[N["passes_filter"]].groupby("clinic")["days_before"].rank(method="first") <= 6
).reindex(N.index).fillna(False)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/pubmedbert")
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
AP = re.compile(r"(?im)^\s{0,8}(assessment|impression|a\s*/\s*p|a\s*&\s*p|plan)\b"
                r"|(?i)\bassessment\s+and\s+plan\b")


def cut_char(text, n_tok):
    """character offset where the model stops reading, via the real tokenizer"""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False,
              truncation=False)
    offs = enc["offset_mapping"]
    n_content = n_tok - 2                      # [CLS] and [SEP] take two slots
    if len(offs) <= n_content:
        return None, len(offs)
    return offs[n_content - 1][1], len(offs)


stats = {"notes": 0, "used": 0, "ap_found": 0, "ap_after_256": 0, "ap_after_512": 0,
         "chars_total": 0, "chars_seen_256": 0, "over_cap": 0, "patients_at_cap": 0}

for i, (clinic, row) in enumerate(SEL.iterrows(), 1):
    sub = N[N["clinic"] == str(clinic)].sort_values("days_before")
    used = sub[sub["used"]]
    stats["patients_at_cap"] += int(len(sub[sub["passes_filter"]]) > 6)
    stats["over_cap"] += max(0, len(sub[sub["passes_filter"]]) - 6)

    print("=" * 78)
    out = ("PROGRESSED within 12 months" if row["y"] == 1
           else f"no progression (followed {int(row['time'])} days)")
    print(f"PATIENT {i} of {len(SEL)}   |   {out}")
    print(f"{len(sub)} notes in the year before treatment, "
          f"{int(sub['passes_filter'].sum())} pass the filters, "
          f"{len(used)} reach the model")
    print("=" * 78)

    for _, nrow in sub.iterrows():
        t = nrow["txt"]
        stats["notes"] += 1
        c256, ntok = cut_char(t, 256)
        c512, _ = cut_char(t, 512)
        stats["chars_total"] += len(t)
        stats["chars_seen_256"] += len(t) if c256 is None else c256

        if nrow["used"]:
            stats["used"] += 1
            tagline = f"USED  (note {int(nrow['rn'])})"
        elif not nrow["passes_filter"]:
            why = ("wrong type: " + str(nrow["title"])[:30]
                   if nrow["title"] not in TITLES else
                   f"length {len(t):,} outside 400-25,000")
            tagline = f"NOT SEEN - {why}"
        else:
            tagline = "NOT SEEN - beyond the 6-note cap"

        m = AP.search(t)
        if m:
            stats["ap_found"] += 1
            if c256 is not None and m.start() > c256:
                stats["ap_after_256"] += 1
            if c512 is not None and m.start() > c512:
                stats["ap_after_512"] += 1
            ap = (f"A&P at char {m.start():,}"
                  + ("  [AFTER the 256 cut - never read]"
                     if (c256 is not None and m.start() > c256) else "  [within 256]"))
        else:
            ap = "no assessment/plan header found"

        print(f"\n  {'-' * 74}")
        print(f"  {tagline}   |   {int(nrow['days_before'])} days before treatment")
        print(f"  {nrow['title']}   |   {len(t):,} chars, {ntok:,} tokens   |   {ap}")
        print(f"  {'-' * 74}")

        marked = t
        for pos, lab in sorted([(c512, "512-TOKEN CUT"), (c256, "256-TOKEN CUT")],
                               key=lambda z: -(z[0] or 0)):
            if pos is not None:
                marked = (marked[:pos]
                          + f"\n\n>>>>>>>>>> {lab}: model stops reading here <<<<<<<<<<\n\n"
                          + marked[pos:])
        for line in marked.splitlines():
            print(textwrap.fill(line, 96, initial_indent="    ",
                                subsequent_indent="    ") or "")

        if nrow["used"]:
            head = NUMS.sub(" ", t)[:400]
            print(f"\n    ---- as the model receives it (numerals stripped), first 400 chars:")
            print(textwrap.fill(head, 96, initial_indent="    ", subsequent_indent="    "))

print("\n" + "=" * 78)
print("WHAT THE MODEL ACTUALLY READS")
print("=" * 78)
print(f"  notes in these patients' year         {stats['notes']:,}")
print(f"  reaching the model                    {stats['used']:,} "
      f"({stats['used']/max(stats['notes'],1):.0%})")
print(f"  discarded by the 6-note cap           {stats['over_cap']:,}")
print(f"  patients who hit the cap              {stats['patients_at_cap']}/{len(SEL)}")
print(f"  characters available                  {stats['chars_total']:,}")
print(f"  characters inside the 256-token cut   {stats['chars_seen_256']:,} "
      f"({stats['chars_seen_256']/max(stats['chars_total'],1):.0%})")
print(f"\n  notes with an assessment/plan header  {stats['ap_found']:,}")
print(f"  ...where it falls AFTER the 256 cut   {stats['ap_after_256']:,} "
      f"({stats['ap_after_256']/max(stats['ap_found'],1):.0%})")
print(f"  ...where it falls AFTER the 512 cut   {stats['ap_after_512']:,} "
      f"({stats['ap_after_512']/max(stats['ap_found'],1):.0%})")
print("""
  If that second-to-last number is high, the model has essentially never read
  the physician's assessment -- only the header and the start of the history.
  That would make "extract the A&P section and embed that" a far better move
  than any amount of fine-tuning, and it would explain why four very different
  encoders all landed on the same score: they were all reading boilerplate.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"inspect_notes | cohort={COHORT} | patients={len(SEL)} | notes={stats['notes']} "
      f"| used={stats['used']} | over_cap={stats['over_cap']} "
      f"| chars_read={stats['chars_seen_256']/max(stats['chars_total'],1):.0%} "
      f"| ap_after_256={stats['ap_after_256']}/{stats['ap_found']}")

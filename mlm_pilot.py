"""
mlm_pilot.py -- one experiment that kills or keeps every MLM idea.

All five ideas (language lead-time, anomaly detection, practice drift,
expected-vs-actual treatment) rest on one assumption: that ClinicalBERT's
SURPRISAL measures clinical content rather than formatting. Clinical notes are
full of lab dumps, redaction tokens and boilerplate. If surprisal mostly tracks
"how many numbers are in this text", everything downstream is noise.

Three parts, each able to stop the project:

  A. does surprisal behave at all? hand-written normal vs nonsense sentences
  B. what actually drives it in real notes? correlate against number density,
     redaction tokens, length. if those dominate, it is measuring layout.
  C. the real test -- does surprisal shift BEFORE the PSA marker moves?
     progressors aligned at progression vs controls, same framework as the
     lab lead-time analysis.

Efficiency: true per-token surprisal needs one forward pass per token. Instead
we mask 15% at random (as in BERT training) and average the loss over a few
repeats -- an unbiased estimate for ~4 passes per note rather than ~200.

CPU only, no API calls.
"""
import base64, json, os, re, ssl, time, urllib.request
import numpy as np
import pandas as pd
import torch

DEST, REPO = "models/clinicalbert", "cal-healthier/mayo-weights"
CTX = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")

def api(p):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{p}?_={int(time.time())}",
        headers={"Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(req, timeout=180, context=CTX))
def blob(sha):
    return base64.b64decode(api(f"git/blobs/{sha}")["content"])

if not os.path.exists(os.path.join(DEST, "pytorch_model.bin")):
    os.makedirs(DEST, exist_ok=True)
    print("fetching ClinicalBERT ...")
    for e in api("contents/clinicalbert"):
        if e["type"] == "file":
            open(os.path.join(DEST, e["name"]), "wb").write(blob(e["sha"]))
            print(f"  {e['name']}")
    ch = api("contents/clinicalbert/_chunks")
    man = json.loads(blob(next(c["sha"] for c in ch if c["name"] == "manifest.json")))
    parts = []
    for i in range(man["chunks"]):
        e = next(c for c in ch if c["name"] == f"{man['file']}.{i:03d}")
        parts.append(blob(e["sha"]))
        print(f"  chunk {i+1}/{man['chunks']}  {sum(map(len,parts))/1e6:.0f} MB")
    data = b"".join(parts)
    import hashlib
    assert hashlib.sha256(data).hexdigest() == man["sha256"], "CHECKSUM MISMATCH"
    open(os.path.join(DEST, man["file"]), "wb").write(data)
    print("  checksum ok")

from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained(DEST)
mdl = AutoModelForMaskedLM.from_pretrained(DEST).eval()
print(f"loaded: vocab {tok.vocab_size}, has MLM head = "
      f"{hasattr(mdl, 'cls') or hasattr(mdl, 'lm_head')}")

@torch.no_grad()
def surprisal(texts, repeats=3, maxlen=256, batch=8):
    """mean cross-entropy on 15% randomly masked tokens; lower = more expected"""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tok(chunk, return_tensors="pt", truncation=True,
                  max_length=maxlen, padding=True)
        acc = np.zeros(len(chunk))
        for _ in range(repeats):
            ids = enc["input_ids"].clone()
            special = torch.tensor(
                [[t in tok.all_special_ids for t in row] for row in ids])
            prob = torch.full(ids.shape, 0.15)
            prob[special] = 0.0
            mask = torch.bernoulli(prob).bool() & enc["attention_mask"].bool()
            if mask.sum() == 0:
                continue
            labels = ids.clone(); labels[~mask] = -100
            ids[mask] = tok.mask_token_id
            logits = mdl(input_ids=ids, attention_mask=enc["attention_mask"]).logits
            lp = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                reduction="none", ignore_index=-100).view(labels.shape)
            per = (lp * mask).sum(1) / mask.sum(1).clamp(min=1)
            acc += per.numpy()
        out.extend(acc / repeats)
    return np.array(out)

print("\n" + "=" * 74)
print("A. DOES SURPRISAL BEHAVE?")
print("=" * 74)
NORMAL = ["The patient tolerated the infusion without complication.",
          "PSA has decreased since starting androgen deprivation therapy.",
          "No evidence of metastatic disease on today's imaging.",
          "He was counseled regarding the risks and benefits of treatment.",
          "Repeat laboratory studies are scheduled in three months."]
ODD = ["The patient tolerated the infusion without bicycle.",
       "PSA has decreased since starting the piano lesson.",
       "No evidence of metastatic disease on today's sandwich.",
       "He was counseled regarding the risks and benefits of gravel.",
       "Repeat laboratory studies are scheduled in three llamas."]
n, o = surprisal(NORMAL), surprisal(ODD)
print(f"  {'normal clinical':<22}mean {n.mean():.3f}")
print(f"  {'one word replaced':<22}mean {o.mean():.3f}")
print(f"  separation {o.mean()-n.mean():+.3f}  "
      f"{'WORKS' if o.mean()-n.mean() > 0.3 else 'WEAK -- stop here'}")

print("\n" + "=" * 74)
print("B. WHAT DRIVES IT IN REAL NOTES?  (is it medicine or layout?)")
print("=" * 74)
from google.cloud import bigquery
C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
E = pd.read_parquet("psa_progression.parquet")
tx = pd.read_parquet("psa_values.parquet").groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index)); E = E.dropna(subset=["tx_date"])
E["anchor"] = E["tx_date"] + pd.to_timedelta(E["time"], unit="D")
prog = E[E["prog"] == 1].head(60); ctrl = E[(E["prog"] == 0) & (E["time"] >= 540)].head(30)
S = pd.concat([prog.assign(g="progressed"), ctrl.assign(g="responding")])
rows = ",".join(f"('{c}',DATE '{a.date()}')" for c, a in zip(S.index.astype(str), S["anchor"]))
N = C.query(f"""
WITH coh AS (SELECT * FROM UNNEST(ARRAY<STRUCT<clinic STRING, anch DATE>>[{rows}])),
pk AS (SELECT c.clinic, c.anch, pe.person_id FROM coh c
       JOIN {D}.person pe ON CAST(pe.person_source_value AS STRING)=c.clinic)
SELECT pk.clinic, pk.anch, DATE(n.note_date) AS dt,
       SUBSTR(CAST(n.note_text AS STRING),1,1500) AS txt
FROM pk JOIN {D}.note n ON n.person_id=pk.person_id
WHERE n.note_text IS NOT NULL AND LENGTH(CAST(n.note_text AS STRING)) BETWEEN 400 AND 20000
  AND CAST(n.note_title AS STRING) IN ('Progress Notes','Consults - Outpatient')
  AND DATE(n.note_date) BETWEEN DATE_SUB(pk.anch, INTERVAL 540 DAY) AND pk.anch
QUALIFY ROW_NUMBER() OVER (PARTITION BY pk.clinic ORDER BY n.note_date DESC) <= 14
""").to_dataframe()
N["dt"] = pd.to_datetime(N["dt"]); N["anch"] = pd.to_datetime(N["anch"])
N["mo"] = (N["dt"] - N["anch"]).dt.days / 30.44
N["g"] = N["clinic"].map(S["g"])
print(f"  {len(N):,} notes from {N['clinic'].nunique()} men")
t0 = time.time()
N["surp"] = surprisal(N["txt"].tolist())
print(f"  scored in {time.time()-t0:.0f}s ({len(N)/(time.time()-t0):.1f}/sec)")
N["digits"] = N["txt"].str.count(r"\d") / N["txt"].str.len()
N["redact"] = N["txt"].str.count("RDCT|MCP_IL|XXXX")
N["nlen"] = N["txt"].str.len()
print("\n  correlation of surprisal with layout artefacts:")
for c, lbl in [("digits", "digit density"), ("redact", "redaction tokens"),
               ("nlen", "note length")]:
    r = N[["surp", c]].corr().iloc[0, 1]
    print(f"    {lbl:<22}{r:+.2f}  {'<-- DOMINATED' if abs(r) > 0.5 else ''}")
print("\n  if any is beyond +/-0.5, surprisal is reading layout, not medicine.")

print("\n" + "=" * 74)
print("C. DOES IT SHIFT BEFORE THE MARKER DOES?")
print("=" * 74)
base = N[N["mo"] <= -12].groupby("clinic")["surp"].median().rename("b")
N = N.join(base, on="clinic"); N = N[N["b"].notna()]
N["rel"] = N["surp"] - N["b"]
print(f"  {'months before':>14}{'progressed':>14}{'still responding':>20}")
for lo, hi in [(12, 9), (9, 6), (6, 3), (3, 1), (1, -0.01)]:
    a = N[(N["g"] == "progressed") & (N["mo"] < -hi) & (N["mo"] >= -lo)]["rel"]
    b = N[(N["g"] == "responding") & (N["mo"] < -hi) & (N["mo"] >= -lo)]["rel"]
    if len(a) < 15:
        continue
    print(f"  {f'{lo}-{hi}':>14}{a.median():>+14.3f}"
          f"{(b.median() if len(b) > 10 else np.nan):>+20.3f}")
late = N[(N["g"] == "progressed") & (N["mo"] >= -3)]["rel"]
latec = N[(N["g"] == "responding") & (N["mo"] >= -3)]["rel"]
print(f"\n  last 3 months: progressors {late.median():+.3f} vs "
      f"controls {latec.median():+.3f}   diff {late.median()-latec.median():+.3f}")
print("""
  a rising column in progressors that is flat in controls = the writing changes
  before the blood does. that is the finding. flat in both = the idea is dead
  and we stop here.""")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"mlm_pilot | sanity={o.mean()-n.mean():+.2f} | notes={len(N)} "
      f"| corr_digits={N[['surp','digits']].corr().iloc[0,1]:+.2f} "
      f"| late_prog={late.median():+.3f} | late_ctrl={latec.median():+.3f}")

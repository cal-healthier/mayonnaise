"""
gpu_strip.py -- is the result narrative, or is it the note header?

what_carries_it.py returned an unambiguous and unwelcome answer. Sentence
occlusion -- the one causal pass -- put the note's METADATA HEADER at the top
of the risk list twelve times out of twelve:

    +1.049  Result Type: Oncology Miscellaneous Note  Result Date: 27 June ...
    +0.993  Result Type: Oncology Miscellaneous Note  Result Date: 24 Sept ...

Not one clinical sentence. The vocabulary agrees: high-risk tokens are rn,
mst, sent, verified, md, edt and month names -- header furniture and
timestamps. Low-risk tokens are kg, ml, dl, intake, output, urine, creatinine
-- the nutrition/nursing template.

So the model may be reading WHICH SERVICE WROTE THE NOTE, and inferring
where in the treatment pathway the patient is. Real signal, but it is care
pathway bookkeeping rather than clinical narrative, and it is the kind of
thing structured data should already carry.

This strips the metadata and re-embeds so we can find out. Two levels:

  stripped   the "Result Type / Result Date / Result Status / Performed By /
             Verified By / *Final* / Entered by" preamble removed
  clinical   also removes the message plumbing -- From/To/Cc/Sent/Subject
             lines, Reference IDs, phone numbers -- leaving prose only

It also extracts the document type from each header so strip_test.py can test
the deflationary hypothesis directly: can a handful of note-type COUNTS
reproduce the whole text effect?

Prints before/after on real notes so the stripping can be eyeballed before
anything is believed.
"""
import os, re, subprocess, time
import pandas as pd

COHORTS = ("ovarian", "prostate")
SEQ = (512,)
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
USER, KEY, VP = "calder_healthier", os.path.expanduser("~/.ssh/google_compute_engine"), "~/venv/bin"
BOXES = [("a2-highgpu-1g-01-mcp-acc-055-dbg-p-7e23", "us-central1-f", "A100"),
         ("g2-standard-8-01-mcp-acc-055-dbg-p-7e23", "us-central1-c", "L4")]

# the preamble ends at *Final* / *Preliminary*, or after Verified By:, or
# after the Entered by ... line -- whichever we can find
END = re.compile(r"\*\s*(Final|Preliminary|Prelim)\s*\*", re.I)
ENTERED = re.compile(r"Entered by .{0,80}?\d{2}[-/][A-Za-z0-9]{2,4}[-/]\d{2,4}[^\n]{0,20}", re.I)
VERIFIED = re.compile(r"Verified By:[^\n]{0,80}", re.I)
TYPE = re.compile(r"Result Type:\s*(.{2,45}?)\s+Result (Date|Status)", re.I)
PLUMB = re.compile(
    r"(?im)^\s*(from|to|cc|bcc|sent|subject|reference|result title|performed by"
    r"|verified by|result status|result date|result type|page|mrn|fin|dob|author)\s*:.*$")
PHONE = re.compile(r"\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}")
REFID = re.compile(r"\b[A-Z]{4}-\d{6,}\b")
WS = re.compile(r"[ \t]{2,}")


def note_type(t):
    m = TYPE.search(t[:400])
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip().upper()[:40]
    head = re.sub(r"[^A-Za-z /]", " ", t[:60])
    return re.sub(r"\s+", " ", head).strip().upper()[:40] or "UNKNOWN"


def strip_header(t):
    m = END.search(t[:1200])
    if m:
        return t[m.end():].lstrip(" :-\n")
    m = ENTERED.search(t[:1200]) or VERIFIED.search(t[:1200])
    if m:
        return t[m.end():].lstrip(" :-\n")
    return t


def clinical_only(t):
    t = strip_header(t)
    t = PLUMB.sub(" ", t)
    t = PHONE.sub(" ", t)
    t = REFID.sub(" ", t)
    return WS.sub(" ", t).strip()


def sh(cmd, n=12, quiet=False, t=3600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    out = (r.stdout or r.stderr).strip()
    if not quiet:
        print("\n".join("    " + l for l in out.splitlines()[-n:]) or "    (ok)")
    return r.returncode, out


print("=" * 78)
print("STRIPPING THE METADATA HEADER")
print("=" * 78)

for tag in COHORTS:
    src = f"notes96_{tag}.parquet"
    if not os.path.exists(src):
        print(f"  {src} missing -- run gpu_sweep.py first")
        continue
    N = pd.read_parquet(src)
    N["clinic"] = N["clinic"].astype(str)
    N["ntype"] = N["txt"].map(note_type)
    N["stripped"] = N["txt"].map(strip_header)
    N["clinical"] = N["txt"].map(clinical_only)
    keep = N["clinical"].str.len() >= 120
    print(f"\n  {tag}: {len(N):,} notes | {N['ntype'].nunique()} distinct types | "
          f"{(~keep).mean():.0%} become too short after stripping")
    print(f"  chars: raw {N['txt'].str.len().median():.0f} -> "
          f"stripped {N['stripped'].str.len().median():.0f} -> "
          f"clinical {N['clinical'].str.len().median():.0f} (median)")
    print(f"\n  most common document types:")
    for t, c in N["ntype"].value_counts().head(8).items():
        print(f"    {c:>7,}  {t}")
    N[["clinic", "rn", "days_before", "ntype", "txt", "stripped", "clinical"]] \
        .to_parquet(f"strip_{tag}.parquet")
    pd.DataFrame({
        "clinic": N["clinic"], "rn": N["rn"].astype("int64"),
        "days_before": N["days_before"].astype("int64"),
        "stripped": N["stripped"].astype(str),
        "clinical": N["clinical"].astype(str),
    }).to_parquet(f"notes_gpu_strip_{tag}.parquet")

    if tag == "ovarian":
        print(f"\n  {'=' * 74}\n  EYEBALL THE STRIPPING -- 2 examples\n  {'=' * 74}")
        for i in N.index[:2]:
            print(f"\n  RAW ({N.loc[i,'ntype']}):")
            print("    " + str(N.loc[i, "txt"])[:340].replace("\n", " "))
            print(f"  CLINICAL ONLY:")
            print("    " + str(N.loc[i, "clinical"])[:340].replace("\n", " "))

REMOTE = r'''
import os, re, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")
m = SentenceTransformer("models/pubmedbert", device="cuda")
m.max_seq_length = 512
for tag in %s:
    src = f"notes_gpu_strip_{tag}.parquet"
    if not os.path.exists(src):
        print(f"  {src} missing", flush=True); continue
    N = pd.read_parquet(src)
    for col in ("stripped", "clinical"):
        out = f"strip_emb_{tag}_{col}.parquet"
        if os.path.exists(out):
            print(f"  {tag}/{col}: cached", flush=True); continue
        txt = N[col].str.replace(NUMS, " ", regex=True).tolist()
        t0 = time.time()
        V = m.encode(txt, batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False)
        o = pd.DataFrame(V); o.columns = [str(c) for c in o.columns]
        o["clinic"] = N["clinic"].values; o["rn"] = N["rn"].values
        o.to_parquet(out)
        print(f"  {tag}/{col}: {len(N):,} in {time.time()-t0:.0f}s", flush=True)
print("done")
''' % (repr(list(COHORTS)),)

print("\n" + "=" * 78)
print("RE-EMBEDDING WITHOUT THE HEADER")
print("=" * 78)
ip = None
for name, zone, gpu in BOXES:
    code, out = sh(f"gcloud compute instances start {name} --zone={zone} "
                   f"--project={PROJ} 2>&1", quiet=True, t=420)
    if code == 0:
        _, i = sh(f"gcloud compute instances describe {name} --zone={zone} "
                  f"--project={PROJ} "
                  f"--format='value(networkInterfaces[0].networkIP)' 2>&1", quiet=True)
        ip = i.strip().splitlines()[-1]
        print(f"  {gpu} at {ip}")
        break
ok = False
if ip:
    SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=25 {USER}@{ip}"
    SCP = f"scp -i {KEY} -o StrictHostKeyChecking=no"
    t0 = time.time()
    while time.time() - t0 < 420:
        if sh(f"{SSH} 'echo up' 2>&1", quiet=True, t=40)[0] == 0:
            break
        time.sleep(10)
    for tag in COHORTS:
        f = f"notes_gpu_strip_{tag}.parquet"
        if os.path.exists(f):
            sh(f"{SCP} {f} {USER}@{ip}:~/ 2>&1", 2, quiet=True, t=3600)
    open("_remote_strip.py", "w").write(REMOTE)
    sh(f"{SCP} _remote_strip.py {USER}@{ip}:~/ 2>&1", 2, quiet=True)
    sh(f"{SSH} 'cd ~ && {VP}/python _remote_strip.py' 2>&1", 16, t=10800)
    got = []
    for tag in COHORTS:
        for col in ("stripped", "clinical"):
            f = f"strip_emb_{tag}_{col}.parquet"
            if sh(f"{SCP} {USER}@{ip}:~/{f} . 2>&1", 2, quiet=True, t=3600)[0] == 0 \
                    and os.path.exists(f):
                got.append(f)
    print(f"  retrieved {len(got)}/{len(COHORTS)*2}")
    ok = len(got) > 0
    print(f"  GPU still running at {ip}. stop_all() when done.")

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"gpu_strip | cohorts={','.join(COHORTS)} | ok={ok} | gpu=RUNNING")

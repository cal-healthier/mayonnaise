"""
what_carries_it.py -- open-ended discovery of the signal, no concepts supplied.

what_is_it_reading.py could only ever confirm or deny the 45 concepts I chose
to write down. Everything not on that list was invisible by construction. This
supplies nothing and lets the data name the thing.

Four passes, cheapest first, each answering a different question:

  1. THE RISK DIRECTION       Fit a linear model on note vectors. The weight
                              vector IS what the model thinks risk looks like.
                              Rank every note by its projection and read the
                              extremes. Not a guess -- the actual direction.

  2. DISTINCTIVE VOCABULARY   Which words separate the top decile from the
                              bottom? Log-odds with an informative Dirichlet
                              prior (Monroe et al.), so the list is not just
                              rare words with lucky counts.

  3. UNSUPERVISED CLUSTERS    Cluster the notes with no reference to outcome,
                              THEN check which clusters progress. A cluster
                              that is 3x baseline risk is a phenotype nobody
                              named in advance.

  4. SENTENCE OCCLUSION       The causal one. Delete each sentence, re-embed,
                              see how far the risk score moves. Output is the
                              literal sentences that carry the signal. Small
                              sample because it costs one embedding per
                              sentence.

WHY NOT MECHANISTIC INTERPRETABILITY. Mech interp explains what happens INSIDE
a network -- circuits, superposed features, causal tracing through layers. We
do not actually care why PubMedBERT computes what it computes; we care what in
the NOTES predicts progression. That is a question about the data, and it is
much easier. Sparse autoencoders are the one mech-interp tool that might port
across, but they are built for internal activations where features are
superposed, and a final-layer L2-normalised sentence embedding has already
thrown that structure away -- you would likely recover a rotation, not a
concept dictionary. Worth trying later; not the first move.

PATIENT TEXT STAYS IN THE ENCLAVE. Read on screen, copy back the FINAL LINE.
"""
import os, re, textwrap
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

COHORT = "ovarian"
N_SHOW = 8          # notes printed per extreme
N_OCCLUDE = 40      # notes put through sentence occlusion

lab, mark = {"ovarian":  ("ov_label.parquet", "ov_ca125.parquet"),
             "prostate": ("psa_progression.parquet", "psa_values.parquet")}[COHORT]
E = pd.read_parquet(lab)
tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
E = E.dropna(subset=["tx_date"])
E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
E = E[(E["prog"] == 1) | (E["time"] >= 365)]

EMB = next((f for f in (f"rn96_emb_{COHORT}_512.parquet",
                        f"rn96_emb_{COHORT}_256.parquet",
                        f"rn_emb_{COHORT}_512.parquet",
                        f"rn_emb_{COHORT}_256.parquet") if os.path.exists(f)), None)
TXT = next((f for f in (f"notes96_{COHORT}.parquet",
                        f"notes2_{COHORT}.parquet") if os.path.exists(f)), None)
if not EMB or not TXT:
    raise SystemExit("need per-note embeddings + note text -- run gpu_sweep.py first")

V = pd.read_parquet(EMB)
N = pd.read_parquet(TXT)
V = V[V["clinic"].isin(E.index)].reset_index(drop=True)
dims = [c for c in V.columns if c.isdigit()]
NUMS = re.compile(r"[0-9]+(\.[0-9]+)?")

# align text to vectors on (clinic, rn) -- row order is not guaranteed to match
N["clinic"] = N["clinic"].astype(str)
key = N.set_index([N["clinic"], N["rn"].astype(int)])["txt"]
V["txt"] = key.reindex(pd.MultiIndex.from_arrays(
    [V["clinic"].astype(str), V["rn"].astype(int)])).values
V = V.dropna(subset=["txt"]).reset_index(drop=True)
V["nonum"] = V["txt"].str.replace(NUMS, " ", regex=True)
yv = E["y"].reindex(V["clinic"]).values

print("=" * 78)
print(f"WHAT CARRIES THE SIGNAL   {COHORT}, {len(V):,} notes, "
      f"{V['clinic'].nunique():,} patients   [{EMB}]")
print("=" * 78)

# ---------------------------------------------------------- 1. risk direction
X = V[dims].values
grp = V["clinic"].values
proj = np.zeros(len(X))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yv, grp):
    m = LogisticRegression(C=.05, max_iter=4000).fit(X[tr], yv[tr])
    proj[te] = X[te] @ m.coef_[0]
V["risk"] = proj
print(f"\nnote-level AUC of the risk direction: {roc_auc_score(yv, proj):.3f}")
print("(out-of-fold, patients never split across folds)")

order = V["risk"].values.argsort()
for label, idx in (("HIGHEST RISK", order[::-1][:N_SHOW]), ("LOWEST RISK", order[:N_SHOW])):
    print("\n" + "=" * 78)
    print(f"1. {label} NOTES BY THE LEARNED DIRECTION")
    print("=" * 78)
    for i in idx:
        r = V.iloc[i]
        print(f"\n  --- risk {r['risk']:+.2f} | note {int(r['rn'])} | "
              f"{int(r['days_before'])}d before treatment | "
              f"patient progressed: {'YES' if E.loc[r['clinic'], 'y'] == 1 else 'no'}")
        print(textwrap.fill(str(r["txt"])[:1200], 94,
                            initial_indent="    ", subsequent_indent="    "))

# ------------------------------------------------------ 2. distinctive words
print("\n" + "=" * 78)
print("2. VOCABULARY THAT SEPARATES THE EXTREMES  (log-odds, Dirichlet prior)")
print("=" * 78)
hi = V.iloc[order[::-1][:len(V) // 10]]["nonum"]
lo = V.iloc[order[:len(V) // 10]]["nonum"]
cv = CountVectorizer(min_df=15, max_df=.6, stop_words="english",
                     ngram_range=(1, 2), max_features=40000)
cv.fit(pd.concat([hi, lo]))
a = np.asarray(cv.transform(hi).sum(0)).ravel().astype(float)
b = np.asarray(cv.transform(lo).sum(0)).ravel().astype(float)
prior = (a + b)                       # corpus counts as the informative prior
a0 = prior.sum() * .01
aw = prior / prior.sum() * a0
la = np.log((a + aw) / (a.sum() + a0 - a - aw))
lb = np.log((b + aw) / (b.sum() + a0 - b - aw))
z = (la - lb) / np.sqrt(1 / (a + aw) + 1 / (b + aw))
vocab = np.array(cv.get_feature_names_out())
top, bot = z.argsort()[::-1][:30], z.argsort()[:30]
print(f"\n  {'HIGH-RISK NOTES':<42}{'LOW-RISK NOTES':<36}")
print("  " + "-" * 76)
for i in range(30):
    print(f"  {vocab[top[i]]:<30}{z[top[i]]:>7.1f}    "
          f"{vocab[bot[i]]:<28}{z[bot[i]]:>7.1f}")

# ---------------------------------------------------------- 3. clusters
print("\n" + "=" * 78)
print("3. UNSUPERVISED CLUSTERS, RANKED BY OUTCOME AFTERWARDS")
print("=" * 78)
K = 40
km = KMeans(K, n_init=4, random_state=0).fit(X)
V["cl"] = km.labels_
cs = V.groupby("cl").agg(n=("risk", "size"), rate=("risk", lambda s: yv[s.index].mean()))
cs = cs[cs["n"] >= 40].sort_values("rate")
base = yv.mean()
print(f"  baseline progression rate {base:.1%}\n")
for label, rows in (("RISKIEST", cs.tail(3).iloc[::-1]), ("SAFEST", cs.head(3))):
    for cl, row in rows.iterrows():
        print(f"\n  --- {label} cluster {cl}: {int(row['n'])} notes, "
              f"{row['rate']:.0%} progressed ({row['rate']/base:.1f}x baseline)")
        sub = V[V["cl"] == cl]
        d = ((sub[dims].values - km.cluster_centers_[cl]) ** 2).sum(1)
        for i in d.argsort()[:2]:
            print(textwrap.fill(str(sub.iloc[i]["txt"])[:600], 92,
                                initial_indent="      ", subsequent_indent="      "))
            print()

# ------------------------------------------------------- 4. occlusion
print("\n" + "=" * 78)
print("4. SENTENCE OCCLUSION -- delete a sentence, see if the risk moves")
print("=" * 78)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count())
    st = SentenceTransformer("models/pubmedbert")
    st.max_seq_length = 512
    w = LogisticRegression(C=.05, max_iter=4000).fit(X, yv).coef_[0]
    picks = list(order[::-1][:N_OCCLUDE // 2]) + list(order[:N_OCCLUDE // 2])
    rows = []
    for i in picks:
        t = str(V.iloc[i]["txt"])[:2000]
        sents = [s.strip() for s in re.split(r"(?<=[.\n])\s+", t) if len(s.strip()) > 25]
        if len(sents) < 3:
            continue
        variants = [" ".join(sents)] + [" ".join(sents[:j] + sents[j + 1:])
                                        for j in range(len(sents))]
        variants = [NUMS.sub(" ", v) for v in variants]
        ev = st.encode(variants, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False)
        sc = ev @ w
        for j, s in enumerate(sents):
            rows.append({"drop": sc[0] - sc[j + 1], "sent": s})
    R = pd.DataFrame(rows).sort_values("drop")
    print(f"\n  {len(R):,} sentences from {len(picks)} notes\n")
    print("  SENTENCES THAT RAISE RISK MOST  (removing them drops the score)")
    for _, r in R.tail(12).iloc[::-1].iterrows():
        print(f"    {r['drop']:+.3f}  {r['sent'][:110]}")
    print("\n  SENTENCES THAT LOWER RISK MOST")
    for _, r in R.head(12).iterrows():
        print(f"    {r['drop']:+.3f}  {r['sent'][:110]}")
except Exception as e:
    print(f"  skipped: {type(e).__name__}: {str(e)[:160]}")

print("""
{}
HOW TO READ THIS
{}
  Passes 1-3 are correlational: they show what co-occurs with progression.
  Pass 4 is the causal one -- it perturbs the input and measures the effect,
  so a sentence at the top of that list genuinely moves the model.

  Watch for the boring explanations before the interesting ones. If the
  high-risk vocabulary is dominated by care-setting words (inpatient, consult,
  transfer) the model has found WHERE the note was written, not how sick the
  person is. If it is dominated by disease words the notes may simply be
  recording stage, which the registry already gives us for free.

  Anything that survives both filters -- not a setting artefact, not
  restageable from structured data -- is the actual finding, and it becomes
  the next probe.""".format("=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print(f"what_carries_it | {COHORT} | notes={len(V)} | "
      f"note_auc={roc_auc_score(yv, proj):.3f} | clusters={len(cs)} | "
      f"riskiest_cluster_x={cs['rate'].iloc[-1]/base:.1f}")

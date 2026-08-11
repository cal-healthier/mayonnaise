"""
strip_test.py -- the deflationary tests, both of them.

The stripping output changed what needs testing. Two things it revealed:

  Only SOME notes carry a "Result Type: ... *Final*" preamble. Median length
  barely moved (3,487 -> 3,461 chars) because most notes start straight into
  "REFERRAL SOURCE:" or "SUBJECTIVE". So the strip removes the header exactly
  where there is one -- which is fine, those are the notes occlusion flagged
  -- but "document type" cannot be recovered by that regex alone. 5,717 and
  20,984 distinct values is a fallback grabbing the first sixty characters,
  not a taxonomy.

  Worse, look at what a note actually opens with:

      REASON FOR VISIT: Stage 1a right Fallopian tube carcinoma, High-grade
      serous

That is STAGE AND HISTOLOGY, written in prose. And our structured comparator
is labs plus a tumour marker plus order counts -- it has NO STAGE, NO GRADE,
NO HISTOLOGY in it at all. The registry has all three and we never put them
in. So "text adds +0.057 over structured" has been measured against a model
missing the single most important prognostic variable in oncology. A reviewer
would find that immediately, and they would be right to.

So there are two deflationary hypotheses now, and the second is the dangerous
one:

  A. the model reads DOCUMENT TYPE and infers the care pathway
  B. the model reads STAGE AND HISTOLOGY, which the registry already has

Arms, in order of what they rule out:

  structured + proxies              what we have been reporting against
  + REGISTRY stage/grade/histology  the comparator we should have used
  + header vocabulary               top tokens from the first 100 chars of
                                    each note. Pure template identity, no
                                    clinical content. Tests hypothesis A.
  + text, raw                       as reported
  + text, clinical prose only       header and plumbing removed
  + everything                      does prose survive all of it?

The last line is the only number that matters now.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from google.cloud import bigquery

C = bigquery.Client(project="mcp-acc-055-dbg-p-7e23")
D = "`mcp-ss-data-p-5o6i`.vw_accelerate2605_core_v1"
rng = np.random.default_rng(0)
REPEATS, BOOT, K = 2, 2000, 12
COH = {
    "ovarian":  ("ov_label.parquet", "ov_ca125.parquet", "ov_labs.parquet",
                 "baseline_ca125", "prox_ovarian"),
    "prostate": ("psa_progression.parquet", "psa_values.parquet", "psa_labs.parquet",
                 "psa", "prox_prostate"),
}


def oof(X, y, repeats=REPEATS):
    X = pd.DataFrame(X).reindex(y.index)
    acc = np.zeros(len(X))
    for r in range(repeats):
        p = np.zeros(len(X))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=.06, max_leaf_nodes=31,
                min_samples_leaf=25, l2_regularization=1.,
                random_state=0).fit(X.iloc[tr], y.iloc[tr])
            p[te] = m.predict_proba(X.iloc[te])[:, 1]
        acc += p
    return acc / repeats


def paired(y, pa, pb, boot=BOOT):
    y = np.asarray(y)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    n, d = len(y), []
    for _ in range(boot):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, len(i)):
            continue
        d.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return (obs, *np.percentile(d, [2.5, 97.5]))


def registry(index, tag):
    """stage, grade, histology, laterality, nodes, and time from diagnosis.

    Discovers the column names rather than assuming them -- the registry has
    hundreds and we have guessed wrong before (HIGH_RISK_HISTOLOGIC_FEATURES
    is a staging descriptor, blank for 313,130 of 313,442 rows)."""
    cache = f"registry_{tag}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache).reindex(index)
    cols = [r["column_name"] for r in C.query(
        f"SELECT column_name FROM `mcp-ss-data-p-5o6i`."
        f"vw_accelerate2605_core_v1.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE table_name='FACT_CANCER_DATA_REPOSITORY'").result()]
    want = [c for c in cols if any(k in c.upper() for k in
            ("DERIVED_SUMMARY_STAGE", "HISTOLOGIC_TYPE", "GRADE",
             "REGIONAL_NODES_POSITIVE", "REGIONAL_NODES_EXAMINED",
             "LATERALITY", "AGE_AT_DIAGNOSIS", "TUMOR_SIZE"))]
    want = want[:14]
    sel = ", ".join(f"MAX(CAST(r.{c} AS STRING)) AS {c}" for c in want)
    ids = "','".join(index.astype(str))
    R = C.query(f"""
      SELECT CAST(p.PATIENT_CLINIC_NUMBER AS STRING) AS clinic, {sel},
             MIN(DATE(r.DATE_OF_DIAGNOSIS)) AS dx_date
      FROM {D}.FACT_CANCER_DATA_REPOSITORY r
      JOIN {D}.DIM_PATIENT p USING (PATIENT_DK)
      WHERE CAST(p.PATIENT_CLINIC_NUMBER AS STRING) IN ('{ids}')
      GROUP BY 1""").to_dataframe().set_index("clinic")
    print(f"    registry columns used: {', '.join(want)}")
    out = pd.DataFrame(index=R.index)
    for c in want:
        out[c] = pd.factorize(R[c].astype(str))[0]
    out["dx_date"] = pd.to_datetime(R["dx_date"]).astype("int64") // 86400_000_000_000
    out.to_parquet(cache)
    return out.reindex(index)


for tag, (lab, mark, labs_f, marker, proxf) in COH.items():
    S_txt = f"strip_{tag}.parquet"
    if not os.path.exists(S_txt):
        print(f"{S_txt} missing -- run gpu_strip.py first")
        continue
    E = pd.read_parquet(lab)
    tx = pd.read_parquet(mark).groupby("clinic")["tx_date"].first()
    E["tx_date"] = pd.to_datetime(tx.reindex(E.index))
    E = E.dropna(subset=["tx_date"])
    E["y"] = ((E["prog"] == 1) & (E["time"] <= 365)).astype(int)
    E = E[(E["prog"] == 1) | (E["time"] >= 365)]
    y = E["y"]
    labs = pd.read_parquet(labs_f)
    cols = [c for c in labs.columns if c.endswith("__last")]
    S = labs[cols].reindex(E.index); S[marker] = E["baseline"]
    S = S.loc[:, S.notna().sum() > len(S) * .3]
    PROX = None
    for f in (f"{proxf}_pre.parquet", f"{proxf}.parquet"):
        if os.path.exists(f):
            PROX = pd.read_parquet(f).reindex(E.index).fillna(0)
            break
    base = pd.concat([S, PROX], axis=1) if PROX is not None else S

    print("\n" + "=" * 78)
    print(f"{tag.upper()}   {len(E):,} patients, {int(y.sum())} events")
    print("=" * 78)
    REG = registry(E.index, tag)
    REG["days_dx_to_tx"] = (E["tx_date"].astype("int64") // 86400_000_000_000
                            - REG["dx_date"])

    T = pd.read_parquet(S_txt)
    T["clinic"] = T["clinic"].astype(str)
    T = T[T["clinic"].isin(E.index) & (T["rn"] <= K)]

    # ---- hypothesis A: template identity only, from the first 100 chars
    head = T.assign(h=T["txt"].str.slice(0, 100)).groupby("clinic")["h"] \
            .apply(lambda s: " ".join(s))
    cvh = CountVectorizer(max_features=150, min_df=10, token_pattern=r"[A-Za-z]{3,}")
    HV = pd.DataFrame(cvh.fit_transform(head).toarray(),
                      index=head.index,
                      columns=["h_" + w for w in cvh.get_feature_names_out()]
                      ).reindex(E.index).fillna(0)

    def emb(fname):
        if not os.path.exists(fname):
            return None
        V = pd.read_parquet(fname)
        V["clinic"] = V["clinic"].astype(str)
        V = V[V["clinic"].isin(E.index) & (V["rn"] <= K)]
        dims = [c for c in V.columns if c.isdigit()]
        return V.groupby("clinic")[dims].mean().reindex(E.index).add_prefix("t")

    RAW = emb(f"rn96_emb_{tag}_512.parquet")
    CLIN = emb(f"strip_emb_{tag}_clinical.parquet")

    arms = {"structured + proxies": base,
            "+ REGISTRY stage/grade/hist": pd.concat([base, REG], axis=1),
            "+ header vocabulary only": pd.concat([base, REG, HV], axis=1)}
    if RAW is not None:
        arms["+ text, raw (as reported)"] = pd.concat([base, RAW], axis=1)
    if CLIN is not None:
        arms["+ text, clinical prose"] = pd.concat([base, REG, CLIN], axis=1)
        arms["+ registry + header + prose"] = pd.concat([base, REG, HV, CLIN], axis=1)

    P = {}
    for nm, X in arms.items():
        P[nm] = oof(X, y)
        print(f"  {nm:<34}{X.shape[1]:>6} feats   {roc_auc_score(y, P[nm]):.3f}")

    print(f"\n  {'comparison':<52}{'diff':>8}  {'95% CI':>18}")
    print("  " + "-" * 78)

    def show(a, b, note=""):
        if a in P and b in P:
            o, lo, hi = paired(y, P[a], P[b])
            print(f"  {a[:24]+' vs '+b[:22]:<52}{o:+8.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  REAL' if lo > 0 else ''}  {note}")

    show("+ REGISTRY stage/grade/hist", "structured + proxies",
         "<- what we omitted")
    show("+ text, raw (as reported)", "structured + proxies",
         "<- the reported gain")
    show("+ header vocabulary only", "+ REGISTRY stage/grade/hist",
         "<- hypothesis A")
    show("+ text, clinical prose", "+ REGISTRY stage/grade/hist",
         "<- prose over a FAIR comparator")
    show("+ registry + header + prose", "+ header vocabulary only",
         "<- THE DECIDING TEST")

print("""
{}
READING IT
{}
  The reported +0.057 was measured against a structured model with no stage,
  no grade and no histology. If the registry arm alone closes most of that
  gap, the headline number was never the narrative -- it was the comparator
  being weak. That is the finding a reviewer reaches first, and it is better
  to find it here.

  The last line is the claim. Prose, added to a model that already has the
  registry AND the note-template vocabulary, either still adds something or
  it does not. If it does, everything deflationary has now been controlled
  for and the result is real. If it does not, we withdraw it.""".format(
    "=" * 78, "=" * 78))

print("\n" + "-" * 74)
print("FINAL LINE:")
print("strip_test | see the deciding test line per cohort")

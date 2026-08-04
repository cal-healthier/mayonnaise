"""
inspect_other.py -- 55% of criteria clauses fell into 'other'. Look at them
before trusting any feasibility number. Reads the CSV, no network, no BQ.
"""
import pandas as pd

cl = pd.read_csv("trial_criteria_classified.csv")
oth = cl[cl.category == "other"]
print(f"unclassified: {len(oth)} of {len(cl)} clauses ({len(oth)/len(cl):.0%})\n")

print("=" * 78)
print("40 RANDOM UNCLASSIFIED CLAUSES (what is my regex missing?)")
print("=" * 78)
for i, c in enumerate(oth.clause.sample(40, random_state=1), 1):
    print(f"{i:>3}. {c[:150]}")

print("\n" + "=" * 78)
print("MOST COMMON WORDS in unclassified clauses (hints at missing categories)")
print("=" * 78)
STOP = set("""the a an and or of to in for with at least be been is are was were will must
have has had not no any all other than from on by as that this these those patient patients
subject subjects who which their his her its if per within during prior more may can""".split())
words = (oth.clause.str.lower().str.replace(r"[^a-z ]", " ", regex=True)
         .str.split().explode())
top = words[~words.isin(STOP) & (words.str.len() > 3)].value_counts().head(40)
print(", ".join(f"{w}({n})" for w, n in top.items()))

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"inspect_other | unclassified={len(oth)} | share={len(oth)/len(cl):.0%} "
      f"| top_words={','.join(top.head(8).index)}")

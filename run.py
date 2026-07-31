# run.py — the current step. This is the only thing the fixed cell runs;
# it delegates to whatever script we're working on right now.
# `pull` is already defined by the fixed runner cell.
exec(open(pull("step1b_diag.py")).read())

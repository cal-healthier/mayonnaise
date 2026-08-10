"""
token_check.py -- what have we actually spent, and what is left?

Three levels, cheapest first:
  1. PER CALL -- every response carries usage_metadata (prompt / output /
     THINKING tokens, billed separately). Always available.
  2. CUMULATIVE -- Cloud Monitoring publishes Vertex token metrics, if the
     account can read them.
  3. QUOTA LIMITS -- gcloud service quota listing; likely permission-gated the
     same way compute quota was.

Plus a wrapper so future spend is measured rather than guessed from wall-clock
time, which is what I gave you today and it was the wrong proxy.
"""
import os, subprocess
import pandas as pd

PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
def sh(cmd, n=12):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout or r.stderr).strip().splitlines()
        return "\n".join("    " + l for l in out[:n]) or "    (no output)"
    except Exception as e:
        return f"    {type(e).__name__}: {str(e)[:90]}"

print("=" * 76)
print("1. PER-CALL USAGE  (always available)")
print("=" * 76)
from google import genai
from google.genai import types
cl = genai.Client(vertexai=True, project=PROJ, location="global")
rows = []
for label, budget in [("no thinking", 0), ("thinking 2048", 2048)]:
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=budget), temperature=0)
    r = cl.models.generate_content(
        model="gemini-3.5-flash",
        contents="Name three common blood tests. One line.", config=cfg)
    u = r.usage_metadata
    th = getattr(u, "thoughts_token_count", None) or 0
    rows.append({"mode": label, "prompt": u.prompt_token_count,
                 "output": u.candidates_token_count, "thinking": th,
                 "total": u.total_token_count})
    print(f"  {label:<16}prompt {u.prompt_token_count:>5}  "
          f"output {u.candidates_token_count:>5}  thinking {th:>6}  "
          f"total {u.total_token_count:>6}")
if len(rows) == 2 and rows[0]["total"]:
    print(f"\n  thinking multiplier on this trivial prompt: "
          f"{rows[1]['total']/rows[0]['total']:.1f}x")
print("  the thinking column is where today's spend went.")

print("\n" + "=" * 76)
print("2. CUMULATIVE USAGE  (Cloud Monitoring -- may be gated)")
print("=" * 76)
for m in ["aiplatform.googleapis.com/publisher/online_serving/token_count",
          "aiplatform.googleapis.com/publisher/online_serving/model_invocation_count"]:
    print(f"\n  {m.split('/')[-1]}:")
    print(sh(f"gcloud monitoring time-series list --project={PROJ} "
             f"--filter='metric.type=\"{m}\"' --format=json 2>&1 | head -8"))

print("\n" + "=" * 76)
print("3. QUOTA LIMITS  (likely gated, as compute quota was)")
print("=" * 76)
print(sh(f"gcloud alpha services quota list --service=aiplatform.googleapis.com "
         f"--consumer=projects/{PROJ} --format='table(metric,unit)' 2>&1 | head -14"))
print("\n  also try the console: Vertex AI -> Quotas, if you have UI access.")

print("\n" + "=" * 76)
print("4. MEASURE IT FROM NOW ON")
print("=" * 76)
print("""  Paste this once, then use ask() everywhere instead of raw calls:

    USAGE = []
    def ask(prompt, budget=0, model="gemini-3.5-flash", js=True):
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=budget),
            response_mime_type="application/json" if js else None,
            temperature=0)
        r = cl.models.generate_content(model=model, contents=prompt, config=cfg)
        u = r.usage_metadata
        USAGE.append({"prompt": u.prompt_token_count,
                      "output": u.candidates_token_count,
                      "thinking": getattr(u, "thoughts_token_count", 0) or 0,
                      "total": u.total_token_count})
        return r

    import pandas as pd; pd.DataFrame(USAGE).sum()   # running total, any time
""")

print("-" * 74)
print("FINAL LINE:")
print(f"token_check | per_call=ok | nothink_total={rows[0]['total']} "
      f"| think_total={rows[1]['total']} "
      f"| ratio={rows[1]['total']/max(rows[0]['total'],1):.1f}x")

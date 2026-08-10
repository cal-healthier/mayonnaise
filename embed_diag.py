"""
embed_diag.py -- gemini-embedding-2 is allowlisted. Why is it refusing?

The probe showed:
  older models      403 Forbidden        -> genuinely not enabled
  gemini-embedding-2 400 FAILED_PRECONDITION, "Organizat..."
  and the model LIST contains publishers/google/models/gemini-embedding-2

FAILED_PRECONDITION is not a permission denial -- it means a prerequisite is
unmet. The message begins "Organizat", so almost certainly an org policy. That
is something Mayo can switch on, unlike an allowlist omission.

This prints the FULL error (my last script truncated at 70 chars and cut off
the actionable half), and tries a few call shapes in case it is an API-surface
problem rather than a policy one.
"""
import os, json, traceback
PROJ = os.environ.get("GOOGLE_CLOUD_PROJECT", "mcp-acc-055-dbg-p-7e23")
TXT = ["Interval increase in the dominant hepatic lesion."]

from google import genai
from google.genai import types
cl = genai.Client(vertexai=True, project=PROJ, location="global")

print("=" * 74)
print("1. THE FULL ERROR  (this is the bit that matters)")
print("=" * 74)
try:
    r = cl.models.embed_content(model="gemini-embedding-2", contents=TXT)
    print(f"  WORKED. dim={len(r.embeddings[0].values)}")
except Exception as e:
    print(f"  {type(e).__name__}")
    msg = str(e)
    for i in range(0, len(msg), 100):
        print(f"    {msg[i:i+100]}")

print("\n" + "=" * 74)
print("2. OTHER CALL SHAPES  (in case it is the API surface, not policy)")
print("=" * 74)
shapes = [
  ("with output_dimensionality",
   lambda: cl.models.embed_content(model="gemini-embedding-2", contents=TXT,
        config=types.EmbedContentConfig(output_dimensionality=768))),
  ("with task_type RETRIEVAL_DOCUMENT",
   lambda: cl.models.embed_content(model="gemini-embedding-2", contents=TXT,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))),
  ("fully-qualified publisher path",
   lambda: cl.models.embed_content(
        model="publishers/google/models/gemini-embedding-2", contents=TXT)),
  ("plain string not list",
   lambda: cl.models.embed_content(model="gemini-embedding-2", contents=TXT[0])),
]
for label, fn in shapes:
    try:
        r = fn()
        print(f"  {label:<38}OK  dim={len(r.embeddings[0].values)}")
    except Exception as e:
        print(f"  {label:<38}{type(e).__name__}: {str(e).split(chr(10))[0][:80]}")

print("\n" + "=" * 74)
print("3. IS IT REGION-SPECIFIC?")
print("=" * 74)
for loc in ("global", "us-central1", "us-east4"):
    try:
        c2 = genai.Client(vertexai=True, project=PROJ, location=loc)
        r = c2.models.embed_content(model="gemini-embedding-2", contents=TXT)
        print(f"  {loc:<14}OK  dim={len(r.embeddings[0].values)}")
    except Exception as e:
        print(f"  {loc:<14}{type(e).__name__}: {str(e).split(chr(10))[0][:70]}")

print("\n" + "=" * 74)
print("4. WHAT TO ASK MAYO FOR")
print("=" * 74)
print("""  Paste the full error above into the request. The likely fixes, in order:

   - an Organization Policy constraint on Vertex AI generative models needs
     the model added (constraints/vertexai.allowedModels or similar)
   - the model's terms may need accepting once at the org level
   - if it is 'Organization ... has not enabled', that is a one-switch fix

  Worth bundling with the existing allowlist request for
  publishers/google/models/gemini-3.1-pro-preview -- same category of ask,
  same admin, one ticket.""")

print("\n" + "-" * 74)
print("FINAL LINE:")
print("embed_diag | see full error text above -- that determines the ask")

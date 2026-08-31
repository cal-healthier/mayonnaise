"""
check_whitelist.py (v3) -- content check, the only test a MITM proxy can't fake.

v1 and v2 both failed for the same reason: this environment does TLS
INTERCEPTION. Every host -- including known-blocked download.pytorch.org --
handed back a certificate issued by "Cloud Services", not the real origin CA.
So the proxy terminates TLS, inspects, and re-originates; neither "got an HTTP
code" (v1) nor "got an origin cert" (v2) means we reached the real host.

What a re-encrypting proxy CANNOT fake is upstream CONTENT. An allowed host is
forwarded and the proxy hands back the ORIGIN's real response body/headers. A
blocked host gets the proxy's own canned denial -- there is no upstream content
behind it. So compare bodies:

  known-allowed  api.github.com/zen  -> a real one-line sentence from GitHub
  the question   api.healthier.inc   -> your API's real output, or a proxy
                                         block page

The script prints the raw first bytes + telltale headers for each so you can
judge with your own eyes. It does NOT auto-verdict the healthier rows, because
auto-verdict is exactly what went wrong twice. It flags obvious proxy
signatures if present. Empty GET, no data, no auth.
"""
import shutil, subprocess

# (host, path, what a REAL forwarded response looks like)
TESTS = [
    ("api.github.com", "/zen",       "one plain-English sentence (GitHub zen)"),
    ("api.github.com", "/",          "real JSON with current_user_url etc."),
    ("download.pytorch.org", "/",    "KNOWN BLOCKED -> this is what a denial looks like"),
    ("api.healthier.inc", "/",       "*** your API root ***"),
    ("api.healthier.inc", "/health", "*** your health endpoint, if any ***"),
    ("healthier.inc", "/",           "apex"),
]

PROXY_MARKERS = ("access denied", "access to the requested", "blocked", "forbidden by",
                 "squid", "zscaler", "forcepoint", "bluecoat", "proxy",
                 "not permitted", "policy", "x-squid-error", "x-cache", "via:")

if not shutil.which("curl"):
    print("curl not found -- cannot run the content check on this box")
    raise SystemExit

print("=" * 74)
print("CONTENT CHECK  (a re-encrypting proxy can fake the cert, not the body)")
print("=" * 74)

for host, path, expect in TESTS:
    url = f"https://{host}{path}"
    # -i include headers; -s silent; -m timeout; cap the body ourselves
    p = subprocess.run(["curl", "-sSi", "-m", "12", url],
                       capture_output=True, text=True)
    raw = (p.stdout or "")
    err = (p.stderr or "").strip()
    head, _, body = raw.partition("\r\n\r\n")
    if not body:
        head, _, body = raw.partition("\n\n")
    status = head.split("\n", 1)[0].strip() if head else "(no status line)"
    server = next((l.strip() for l in head.splitlines() if l.lower().startswith("server:")), "")
    via    = next((l.strip() for l in head.splitlines() if l.lower().startswith("via:")), "")
    xcache = next((l.strip() for l in head.splitlines()
                   if l.lower().startswith(("x-cache", "x-squid", "x-bluecoat"))), "")
    body1  = " ".join(body.split())[:160]
    low    = (head + " " + body).lower()
    flags  = [m for m in PROXY_MARKERS if m in low]

    print(f"\n  {url}")
    print(f"    expect: {expect}")
    print(f"    status: {status or err[:70]}")
    if server: print(f"    {server}")
    if via:    print(f"    {via}")
    if xcache: print(f"    {xcache}")
    print(f"    body[:160]: {body1 or '(empty)'}")
    if flags:
        print(f"    >> proxy-denial signatures present: {flags}")

print("""
{}
HOW TO READ IT -- with your eyes, not my classifier
{}
  Compare the two api.github.com rows against the healthier rows.

  github /zen returns a real sentence and github / returns real JSON: that is
  what a FORWARDED (allowed) request looks like on this proxy -- genuine origin
  content coming back.

  download.pytorch.org is the known-blocked reference: whatever its body and
  headers look like IS the proxy's denial fingerprint on this environment.

  Then judge healthier:
    * body looks like YOUR API (your JSON, your error shape, even a 404 you
      recognise) -> WHITELISTED, request is being forwarded.
    * body matches the pytorch denial / shows proxy signatures -> NOT live;
      the proxy is answering for it. Approval granted != rule deployed. Ask
      Mayo to confirm the allowlist entry shipped and the exact hostname.

  You know your own API's real output; that recognition is the reliable test
  the two automated probes could not make.""".format("=" * 74, "=" * 74))

print("\n" + "-" * 70)
print("FINAL LINE:")
print("check_whitelist_v3 | compare github (allowed) vs pytorch (blocked) vs healthier by BODY")

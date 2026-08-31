"""
check_whitelist.py -- has the approved api.healthier.inc route gone live?

Mayo approved whitelisting api.healthier.inc. This confirms whether the change
has actually been applied to the environment's egress proxy. It is a PURE
REACHABILITY PROBE and nothing else:

  * no payload, no auth, no patient data -- an empty GET to "/"
  * only the APPROVED host plus a few KNOWN controls, so the result is
    interpretable. This is not a scan for holes; it checks one sanctioned
    change landed.

Verdicts:
  REACHED         origin server answered (any HTTP code) -> host is whitelisted
  BLOCKED(proxy)  the egress proxy refused before the origin -> not yet applied
  DNS             the name did not resolve
  BLOCKED/timeout no route / timed out -> not applied (or origin down)

Controls: pypi.org + api.github.com should be REACHED; download.pytorch.org
should be BLOCKED. If the controls don't come out that way, the probe itself
is misreading and the healthier rows can't be trusted.
"""
import os, shutil, subprocess

HOSTS = [
    ("pypi.org",             "control -> expect REACHED"),
    ("api.github.com",       "control -> expect REACHED"),
    ("download.pytorch.org", "control -> expect BLOCKED"),
    ("api.healthier.inc",    "*** THE APPROVED HOST ***"),
    ("healthier.inc",        "    (apex, in case they whitelisted this form)"),
    ("www.healthier.inc",    "    (www, same reason)"),
]

print("=" * 74)
print("EGRESS / PROXY CONFIG  (what the environment is set to route through)")
print("=" * 74)
prox = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
if prox:
    for k, v in sorted(prox.items()):
        print(f"  {k} = {v}")
else:
    print("  (no *_proxy environment variables set -- egress may be transparent)")
for f in ("~/.pip/pip.conf", "~/.config/pip/pip.conf", "/etc/pip.conf"):
    p = os.path.expanduser(f)
    if os.path.exists(p):
        body = open(p).read().strip().replace("\n", "\n     ")
        print(f"  {f}:\n     {body}")

have_curl = shutil.which("curl")
print(f"\n  probe tool: {'curl' if have_curl else 'urllib fallback'}")


def probe_curl(host):
    p = subprocess.run(
        ["curl", "-sS", "-m", "12", "-o", "/dev/null",
         "-w", "HTTP=%{http_code} IP=%{remote_ip}", f"https://{host}/"],
        capture_output=True, text=True)
    out, err, code = p.stdout.strip(), p.stderr.strip(), p.returncode
    http = out.split("HTTP=")[1].split()[0] if "HTTP=" in out else "000"
    el = err.lower()
    if code == 6 or "could not resolve" in el or "resolve host" in el:
        v = "DNS"
    elif ("from proxy after connect" in el
          or ("received http code" in el and "proxy" in el)
          or code == 407):
        v = "BLOCKED(proxy)"
    elif http != "000":
        v = "REACHED"
    elif code in (7, 28, 35, 52, 56):
        v = "BLOCKED/timeout"
    else:
        v = f"?(exit {code})"
    detail = f"HTTP {http}" if http != "000" else (err[:70] or f"exit {code}")
    return v, detail


def probe_urllib(host):
    import urllib.request, urllib.error
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(f"https://{host}/",
                                   headers={"User-Agent": "reachability-check"}),
            timeout=12)
        return "REACHED", f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return "REACHED", f"HTTP {e.code} from origin"
    except Exception as e:
        s = str(e).lower()
        if "name or service" in s or "resolve" in s:
            return "DNS", str(e)[:70]
        return "BLOCKED/timeout", f"{type(e).__name__}: {str(e)[:60]}"


probe = probe_curl if have_curl else probe_urllib

print("\n" + "=" * 74)
print("REACHABILITY  (empty GET, no data sent)")
print("=" * 74)
results = {}
for host, note in HOSTS:
    v, detail = probe(host)
    results[host] = v
    print(f"  {v:<16}{host:<22}{detail}")
    print(f"  {'':<16}{note}")

ctrl_ok = (results.get("pypi.org") == "REACHED"
           and results.get("api.github.com") == "REACHED"
           and "BLOCKED" in results.get("download.pytorch.org", ""))
hh = [results.get(h, "?") for h in
      ("api.healthier.inc", "healthier.inc", "www.healthier.inc")]
live = any(x == "REACHED" for x in hh)

print("\n" + "=" * 74)
print("READING IT")
print("=" * 74)
if not ctrl_ok:
    print("  CONTROLS DID NOT BEHAVE -- probe is misreading this environment;")
    print("  do not trust the healthier rows. (Are we on the bastion, not the")
    print("  GPU box? Is curl behaving? Re-run and check the control rows.)")
elif live:
    print("  api.healthier.inc (or a variant) is REACHABLE -- the whitelist is")
    print("  LIVE. The origin answered; nothing patient-related was sent. Next:")
    print("  a single authenticated health-check call to confirm the API responds")
    print("  as expected, still with no patient data.")
else:
    print("  Controls behave, but NONE of the healthier hosts are reachable --")
    print("  the approval has NOT been applied to the proxy yet. This is a config")
    print("  ticket on Mayo's side, not something to work around. Ask them to")
    print("  confirm the rule is deployed and which exact hostname they added.")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"check_whitelist | controls_ok={ctrl_ok} | "
      f"api={results.get('api.healthier.inc')} "
      f"apex={results.get('healthier.inc')} www={results.get('www.healthier.inc')} "
      f"| live={live}")

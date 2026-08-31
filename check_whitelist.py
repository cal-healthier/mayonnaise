"""
check_whitelist.py (v2) -- distinguish a PROXY BLOCK from an ORIGIN response.

v1 was wrong in the worst direction: it called everything REACHED, including
download.pytorch.org, which we KNOW is blocked. The reason: this egress proxy
answers 403 ITSELF for hosts it refuses, instead of failing the connection. So
"got an HTTP code" does NOT mean "reached origin" here.

Two reliable tells instead:

  1. TLS to the origin. If we can complete a TLS handshake and read back a
     certificate whose names cover the host, we genuinely reached that host's
     server. A proxy that blocks on CONNECT never lets origin TLS happen.
  2. Proxy fingerprints on the HTTP response: Via / X-Cache / X-Squid-Error /
     a proxy Server header, or a body that names the proxy. Those mark a
     response MANUFACTURED BY THE PROXY, not the origin.

Verdict prefers the TLS signal; the HTTP headers corroborate.
Still an empty request -- no data, no auth.
"""
import os, shutil, socket, ssl, subprocess

HOSTS = [
    ("pypi.org",             "control -> origin cert expected"),
    ("api.github.com",       "control -> origin cert expected"),
    ("download.pytorch.org", "control -> should be BLOCKED (no origin cert)"),
    ("api.healthier.inc",    "*** THE APPROVED HOST ***"),
    ("healthier.inc",        "    (apex)"),
    ("www.healthier.inc",    "    (www)"),
]

print("=" * 74)
print("EGRESS / PROXY CONFIG")
print("=" * 74)
prox = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
for k, v in sorted(prox.items()):
    print(f"  {k} = {v}")
if not prox:
    print("  (no *_proxy env vars)")


def origin_tls(host, port=443, timeout=10):
    """Return the origin cert's names, or None if we never reached the origin.

    Uses the same proxy path everything else uses if https_proxy is set
    (CONNECT tunnel), else a direct socket."""
    hp = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    raw = socket.create_connection(_hostport(hp) if hp else (host, port),
                                   timeout=timeout)
    try:
        if hp:
            raw.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
                        .encode())
            resp = b""
            while b"\r\n\r\n" not in resp:
                b = raw.recv(1024)
                if not b:
                    break
                resp += b
            line = resp.split(b"\r\n", 1)[0].decode(errors="replace")
            if " 200 " not in line:
                return None, f"proxy refused CONNECT: {line[:60]}"
        ctx = ssl.create_default_context()
        ss = ctx.wrap_socket(raw, server_hostname=host)
        cert = ss.getpeercert()
        names = [v for t in cert.get("subject", ()) for k, v in t if k == "commonName"]
        names += [v for k, v in cert.get("subjectAltName", ())]
        iss = dict(x for t in cert.get("issuer", ()) for x in t).get("organizationName", "?")
        ss.close()
        return names, iss
    finally:
        try:
            raw.close()
        except Exception:
            pass


def _hostport(url):
    u = url.split("://")[-1].rstrip("/")
    h, _, p = u.partition(":")
    return (h, int(p) if p else 3128)


def http_fingerprint(host):
    if not shutil.which("curl"):
        return "", ""
    p = subprocess.run(["curl", "-sSI", "-m", "12", f"https://{host}/"],
                       capture_output=True, text=True)
    hdr = (p.stdout + p.stderr)
    marks = [m for m in ("via:", "x-cache", "x-squid", "squid", "forcepoint",
                         "zscaler", "bluecoat", "x-bluecoat")
             if m in hdr.lower()]
    server = ""
    for ln in hdr.splitlines():
        if ln.lower().startswith("server:"):
            server = ln.strip()
    return ",".join(marks), server


print("\n" + "=" * 74)
print("ORIGIN REACHABILITY  (TLS handshake to the host's own server)")
print("=" * 74)
results = {}
for host, note in HOSTS:
    try:
        names, iss = origin_tls(host)
    except Exception as e:
        names, iss = None, f"{type(e).__name__}: {str(e)[:50]}"
    reached = bool(names) and any(host.split(".", 1)[-1] in n or host in n for n in (names or []))
    marks, server = http_fingerprint(host)
    if reached:
        v = "REACHED"
        detail = f"origin cert issuer={iss}"
    else:
        v = "BLOCKED"
        detail = iss if isinstance(iss, str) else "no origin cert"
    results[host] = v
    print(f"  {v:<9}{host:<22}{detail}")
    if marks:
        print(f"  {'':<9}proxy fingerprint on HTTP: {marks}  {server}")
    print(f"  {'':<9}{note}")

ctrl_ok = (results.get("pypi.org") == "REACHED"
           and results.get("api.github.com") == "REACHED"
           and results.get("download.pytorch.org") == "BLOCKED")
live = results.get("api.healthier.inc") == "REACHED"

print("\n" + "=" * 74)
print("READING IT")
print("=" * 74)
if not ctrl_ok:
    print("  Controls still off -- trust the raw cert lines above over the verdict.")
    print(f"  (pypi={results.get('pypi.org')}, github={results.get('api.github.com')},")
    print(f"   pytorch={results.get('download.pytorch.org')} -- pytorch MUST be BLOCKED)")
elif live:
    print("  api.healthier.inc: real origin TLS reached -> WHITELIST IS LIVE.")
    print("  Next: one authenticated health call, still no patient data.")
else:
    print("  Controls correct; healthier hosts show NO origin cert -> the route is")
    print("  NOT live yet. Approval granted != rule deployed. Ask Mayo to confirm")
    print("  the allowlist entry shipped and the exact hostname. Do not work around.")

print("\n" + "-" * 70)
print("FINAL LINE:")
print(f"check_whitelist_v2 | controls_ok={ctrl_ok} | "
      f"api={results.get('api.healthier.inc')} apex={results.get('healthier.inc')} "
      f"www={results.get('www.healthier.inc')} | live={live}")

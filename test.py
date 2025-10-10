#!/usr/bin/env python3
"""
proxy_test_all.py
- Finds a Webshare proxies file (several candidate locations)
- Parses CSV or colon-separated formats
- Tests each proxy against a set of nba_api endpoints
- Writes raw passing proxies to /mnt/data/webshare_bypassing_proxies.txt
"""

from __future__ import annotations
import os, sys, csv, time, random, itertools
from datetime import datetime
from pathlib import Path
from typing import List

# nba_api endpoints (must be installed in your env)
from nba_api.stats.endpoints import (
    LeagueGameLog,
    LeagueDashTeamStats,
    LeagueDashPlayerStats,
    TeamEstimatedMetrics,
)

# =======================
# Config (tweak here)
# =======================
MODE = "POOL"                 # "DIRECT" or "POOL"
TIMEOUT = 16.0                # seconds; raise if network is flaky
SUCCESS_PAUSE = (0.25, 0.60)
FAILURE_COOLDOWN_SEC = 8
MAX_PROXIES_TO_USE = None     # None => test all; or set to e.g. 50 for quick vetting

# Candidate paths to search (will pick first that exists)
CANDIDATE_PATHS = [
    os.getenv("PROXIES_TXT", ""),                               # env override
    "/mnt/data/Webshare 500 proxies.txt",                       # hosted environment
    str(Path.home() / "Desktop" / "PickPocketProject" / "proxies.txt"),  # your screenshot path
    str(Path.cwd() / "proxies.txt"),                            # project root
    str(Path(__file__).resolve().parent / "proxies.txt"),       # same dir as script
]

# Fallback gateway (only used if no proxies parsed)
_IPROYAL_USER = "BtXwlxxHimVuYof5"
_IPROYAL_PASS = "1962e9MI4JBrucQ0"
_IPROYAL_HOST = "geo.iproyal.com"
_IPROYAL_PORT = 12321
_GATEWAY = f"http://{_IPROYAL_USER}:{_IPROYAL_PASS}@{_IPROYAL_HOST}:{_IPROYAL_PORT}"

OUT_PASS_PATH = "/mnt/data/webshare_bypassing_proxies.txt"

# =======================
# Helpers: file discovery + parsing
# =======================
def find_existing_path(candidates: List[str]) -> str | None:
    checked = []
    for p in candidates:
        if not p:
            continue
        checked.append(p)
        if Path(p).exists():
            return p
    # nothing found
    print("Checked candidate paths (none exist):")
    for p in checked:
        print("  ", p)
    return None

def parse_colon_line(raw: str) -> str | None:
    """Parse ip:port:username:password or ip:port or user:pass@ip:port or http://..."""
    s = raw.strip().strip(",;")
    if not s or s.startswith("#"):
        return None
    if "://" in s:
        return s
    if "@" in s and ":" in s:
        return "http://" + s
    parts = [p.strip() for p in s.split(":")]
    if len(parts) >= 4:
        host, port, user, pwd = parts[0], parts[1], parts[2], parts[3]
        if host and port.isdigit() and user and pwd:
            return f"http://{user}:{pwd}@{host}:{port}"
    if len(parts) == 2 and parts[1].isdigit():
        host, port = parts
        return f"http://{host}:{port}"
    return None

def load_proxies_from_txt(path: str) -> List[str]:
    out = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                url = parse_colon_line(ln)
                if url and url not in seen:
                    seen.add(url)
                    out.append(url)
    except Exception as e:
        print("Error reading txt file:", e)
    return out

def load_proxies_from_csv(path: str) -> List[str]:
    out = []
    seen = set()
    try:
        with open(path, newline='', encoding='utf-8', errors='ignore') as fh:
            rdr = csv.DictReader(fh)
            # try common header names
            for r in rdr:
                host = (r.get('Proxy Address') or r.get('Proxy') or r.get('Host') or r.get('ip') or r.get('address'))
                port = (r.get('Port') or r.get('port'))
                user = (r.get('Username') or r.get('username') or r.get('User'))
                pwd  = (r.get('Password') or r.get('password') or r.get('Pass'))
                if host and port and user and pwd:
                    url = f"http://{user}:{pwd}@{host}:{port}"
                    if url not in seen:
                        seen.add(url)
                        out.append(url)
    except Exception:
        # not a CSV or parse failed
        out = []
    return out

def load_proxies(path: str) -> List[str]:
    # show sample head
    print(f"Reading proxy file: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            sample = [next(fh).rstrip("\n") for _ in range(10)]
    except StopIteration:
        sample = []
    except Exception as e:
        print("Could not read file:", e)
        return []

    print("Sample lines (up to 10):")
    for i, s in enumerate(sample, start=1):
        print(f"{i:03d} | {s}")
    # Try CSV first (header detection)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline().strip()
    except Exception:
        first = ""
    header_indicators = ["proxy address", "port", "username", "password", "proxy"]
    if first and any(h in first.lower() for h in header_indicators):
        print("Detected CSV-like header -> parsing as CSV.")
        parsed = load_proxies_from_csv(path)
        print(f"Parsed {len(parsed)} proxies from CSV.")
        return parsed
    # fallback to colon-based parsing
    parsed = load_proxies_from_txt(path)
    print(f"Parsed {len(parsed)} proxies from text.")
    return parsed

# =======================
# Build NBA_PROXY_POOL
# =======================
path = find_existing_path(CANDIDATE_PATHS)
if path:
    parsed = load_proxies(path)
else:
    parsed = []

if MAX_PROXIES_TO_USE is not None:
    parsed = parsed[:MAX_PROXIES_TO_USE]

if not parsed:
    print("No proxies parsed from file; falling back to gateway.")
    NBA_PROXY_POOL = [_GATEWAY]
else:
    NBA_PROXY_POOL = parsed

print(f"Final NBA_PROXY_POOL length: {len(NBA_PROXY_POOL)}")

# If in DIRECT mode override
if MODE == "DIRECT":
    work_plan = [None]
else:
    work_plan = NBA_PROXY_POOL[:]

# =======================
# Test harness
# =======================
def _set_proxy_env(proxy: str | None):
    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

def _mask(p: str | None) -> str:
    if not p:
        return "(DIRECT)"
    try:
        scheme, rest = p.split("://", 1)
        if "@" in rest:
            _, host = rest.split("@", 1)
            return f"{scheme}://***:***@{host}"
    except Exception:
        pass
    return p

def _variants(timeout: float = TIMEOUT):
    return [
        ("LeagueGameLog:P", lambda: LeagueGameLog(season="2024-25", player_or_team_abbreviation="P", timeout=timeout).get_data_frames()[0]),
        ("LeagueGameLog:T", lambda: LeagueGameLog(season="2024-25", player_or_team_abbreviation="T", timeout=timeout).get_data_frames()[0]),
        ("LeagueDashTeamStats", lambda: LeagueDashTeamStats(season="2024-25", timeout=timeout).get_data_frames()[0]),
        ("LeagueDashPlayerStats", lambda: LeagueDashPlayerStats(season="2024-25", timeout=timeout).get_data_frames()[0]),
        ("TeamEstimatedMetrics", lambda: TeamEstimatedMetrics(season="2024-25", timeout=timeout).get_data_frames()[0]),
    ]

VARIANTS = itertools.cycle(_variants())
results = []
good_raw = []
BASE_HTTP, BASE_HTTPS = os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY")

print(f"Starting proxy test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — MODE={MODE} — proxies={len(work_plan)}")

try:
    for i, chosen in enumerate(work_plan, start=1):
        name, fn = next(VARIANTS)
        _set_proxy_env(chosen)
        proxy_label = _mask(chosen)

        t0 = time.perf_counter()
        try:
            df = fn()
            dt = time.perf_counter() - t0
            rows = len(df) if df is not None else 0
            print(f"[{i:03d}] ✅ {name:<22} via {proxy_label:35s}  rows={rows:<6}  {dt:.2f}s")
            results.append((i, name, proxy_label, "OK", rows, round(dt, 2), ""))
            if chosen not in good_raw:
                good_raw.append(chosen)
            time.sleep(random.uniform(*SUCCESS_PAUSE))
        except Exception as e:
            dt = time.perf_counter() - t0
            msg = str(e)
            status = (
                "timeout" if "Read timed out" in msg or "timed out" in msg.lower() else
                ("rate_limited" if "429" in msg else
                 ("blocked" if "403" in msg or "forbidden" in msg.lower() else "error"))
            )
            print(f"[{i:03d}] ❌ {name:<22} via {proxy_label:35s}  {status} — {dt:.2f}s  {msg}")
            results.append((i, name, proxy_label, status, 0, round(dt, 2), msg))
            print(f"       …waiting {FAILURE_COOLDOWN_SEC}s before next proxy")
            time.sleep(FAILURE_COOLDOWN_SEC)

finally:
    # restore env
    if BASE_HTTP is None:
        os.environ.pop("HTTP_PROXY", None)
    else:
        os.environ["HTTP_PROXY"] = BASE_HTTP
    if BASE_HTTPS is None:
        os.environ.pop("HTTPS_PROXY", None)
    else:
        os.environ["HTTPS_PROXY"] = BASE_HTTPS

# =======================
# Summary + save passing proxies
# =======================
ok = sum(1 for r in results if r[3] == "OK")
fail = len(results) - ok
timeouts = [r[0] for r in results if r[3] == "timeout"]

print(f"\n=== Summary: {ok} OK / {fail} FAIL (timeouts at calls: {timeouts if timeouts else '—'}) ===")
for i, name, proxy, status, rows, elapsed, err in results:
    print(f"{i:03d}  {status:9s}  {name:<22}  via {proxy:35s}  rows={rows:<6}  {elapsed:>5.2f}s")

if good_raw:
    try:
        with open(OUT_PASS_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(good_raw))
        print(f"\nSaved {len(good_raw)} passing proxies → {OUT_PASS_PATH}")
        print('To use only these next time (bash):')
        print('export NBA_PROXY_POOL="' + ",".join(good_raw) + '"')
    except Exception as e:
        print("Failed to write passing proxies:", e)
else:
    print("\nNo passing proxies were found (good_raw empty).")

# final exit code: 0 if any OK else 1
sys.exit(0 if ok > 0 else 1)

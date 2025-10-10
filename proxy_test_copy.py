import os, time, random, itertools
from datetime import datetime
from nba_api.stats.endpoints import (
    LeagueGameLog,
    LeagueDashTeamStats,
    LeagueDashPlayerStats,
    TeamEstimatedMetrics,
)

# =======================
# Config
# =======================
NUM_CALLS = 15                 # how many calls in one burst
MODE = "POOL"                # "DIRECT" or "POOL"
TIMEOUT = 12.0
SUCCESS_PAUSE = (0.25, 0.60)   # small random pause after success (keeps cadence organic)
FAILURE_COOLDOWN_SEC = 10     # wait after any failure (so we can see behavior clearly)
RETRY_ON_FAIL = False          # set True if you want to auto-retry with a new proxy (keeps simple = False)





# =======================
# Helpers
# =======================
def _normalize_proxy(p):
    if not p: return None
    t = p.strip().lower()
    return None if t in {"direct", "local", "none"} else p

def _mask(p):
    if not p: return "(DIRECT)"
    try:
        scheme, rest = p.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            return f"{scheme}://***:***@{host}"
    except ValueError:
        pass
    return p

def _set_proxy_env(proxy):
    # proxy=None => go direct
    if proxy:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
    else:
        os.environ.pop("HTTP_PROXY",  None)
        os.environ.pop("HTTPS_PROXY", None)

def _variants(timeout=TIMEOUT):
    # rotate “safe” endpoints; tweak as you like
    return [
        ("LeagueGameLog:P", lambda: LeagueGameLog(season="2024-25", player_or_team_abbreviation="P", timeout=timeout).get_data_frames()[0]),
        ("LeagueGameLog:T", lambda: LeagueGameLog(season="2024-25", player_or_team_abbreviation="T", timeout=timeout).get_data_frames()[0]),
        ("LeagueDashTeamStats", lambda: LeagueDashTeamStats(season="2024-25", timeout=timeout).get_data_frames()[0]),
        ("LeagueDashPlayerStats", lambda: LeagueDashPlayerStats(season="2024-25", timeout=timeout).get_data_frames()[0]),
        ("TeamEstimatedMetrics", lambda: TeamEstimatedMetrics(season="2024-25", timeout=timeout).get_data_frames()[0]),
    ]

# read proxy pool once (used if MODE="POOL")
POOL_RAW = [p.strip() for p in os.getenv("NBA_PROXY_POOL", "").split(",") if p.strip()]
if MODE == "POOL" and not POOL_RAW:
    raise RuntimeError("MODE=POOL but NBA_PROXY_POOL is empty. Set it or switch MODE to 'DIRECT'.")

# cycle endpoint variants
VARIANTS = itertools.cycle(_variants())

# save baseline env to restore at the end
BASE_HTTP, BASE_HTTPS = os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY")

results = []  # (i, name, proxy_mask, status, rows, elapsed, err)
print(f"Starting burst test at {datetime.now().strftime('%H:%M:%S')}  —  MODE={MODE}  —  NUM_CALLS={NUM_CALLS}\n")

try:
    for i in range(1, NUM_CALLS + 1):
        name, fn = next(VARIANTS)

        # choose proxy for this attempt
        if MODE == "DIRECT":
            chosen = None
        else:
            # random choice from pool (supports 'DIRECT' token too)
            chosen = _normalize_proxy(random.choice(POOL_RAW))

        _set_proxy_env(chosen)
        proxy_label = _mask(chosen)

        t0 = time.perf_counter()
        try:
            df = fn()
            dt = time.perf_counter() - t0
            rows = len(df)
            print(f"[{i:02d}] ✅ {name:<22} via {proxy_label:35s}  rows={rows:<6}  {dt:.2f}s")
            results.append((i, name, proxy_label, "OK", rows, round(dt,2), ""))

            # small random pause after success
            time.sleep(random.uniform(*SUCCESS_PAUSE))

        except Exception as e:
            dt = time.perf_counter() - t0
            msg = str(e)
            status = "timeout" if "Read timed out" in msg else ("rate_limited" if "429" in msg else ("blocked" if "403" in msg or "Forbidden" in msg else "error"))
            print(f"[{i:02d}] ❌ {name:<22} via {proxy_label:35s}  {status} — {dt:.2f}s  {msg}")
            results.append((i, name, proxy_label, status, 0, round(dt,2), msg))

            # cooldown so we can observe if the next call still struggles
            print(f"       …waiting {FAILURE_COOLDOWN_SEC}s before next call")
            time.sleep(FAILURE_COOLDOWN_SEC)

            if RETRY_ON_FAIL and MODE == "POOL":
                # optional: immediate retry with different proxy (kept OFF by default for clean timing study)
                alt_choices = [p for p in POOL_RAW if _normalize_proxy(p) != chosen] or POOL_RAW
                alt = _normalize_proxy(random.choice(alt_choices))
                _set_proxy_env(alt)
                proxy_label_alt = _mask(alt)
                t1 = time.perf_counter()
                try:
                    df2 = fn()
                    dt2 = time.perf_counter() - t1
                    r2 = len(df2)
                    print(f"       ↻ ✅ RETRY {name:<16} via {proxy_label_alt:35s} rows={r2:<6} {dt2:.2f}s")
                    results.append((i, name+" (retry)", proxy_label_alt, "OK", r2, round(dt2,2), ""))
                    time.sleep(random.uniform(*SUCCESS_PAUSE))
                except Exception as e2:
                    dt2 = time.perf_counter() - t1
                    msg2 = str(e2)
                    status2 = "timeout" if "Read timed out" in msg2 else ("rate_limited" if "429" in msg2 else ("blocked" if "403" in msg2 or "Forbidden" in msg2 else "error"))
                    print(f"       ↻ ❌ RETRY {name:<16} via {proxy_label_alt:35s} {status2} — {dt2:.2f}s  {msg2}")
                    results.append((i, name+" (retry)", proxy_label_alt, status2, 0, round(dt2,2), msg2))

finally:
    # restore env
    if BASE_HTTP is None:  os.environ.pop("HTTP_PROXY", None)
    else:                  os.environ["HTTP_PROXY"]  = BASE_HTTP
    if BASE_HTTPS is None: os.environ.pop("HTTPS_PROXY", None)
    else:                  os.environ["HTTPS_PROXY"] = BASE_HTTPS

# =======================
# Summary
# =======================
ok = sum(1 for r in results if r[3] == "OK")
fail = len(results) - ok
timeouts = [r[0] for r in results if r[3] == "timeout"]
print(f"\n=== Summary: {ok} OK / {fail} FAIL (timeouts at calls: {timeouts if timeouts else '—'}) ===")
for i, name, proxy, status, rows, elapsed, err in results:
    print(f"{i:02d}  {status:9s}  {name:<22}  via {proxy:35s}  rows={rows:<6}  {elapsed:>5.2f}s")

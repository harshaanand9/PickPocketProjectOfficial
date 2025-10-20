import pandas as pd
import json 
import requests
import inspect
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
from cache_manager import stats_cache  # Import the cache
from typing import Dict

from proxy_coord import acquire as proxy_acquire, release as proxy_release, apply_proxy as proxy_apply
from proxy_coord import rotate_on_failure as proxy_rotate_on_failure, current_proxy as proxy_current




# Set pandas options to display all rows, columns, and full column content
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents

from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelog  # checked the contents; put on discord
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog, LeagueGameLog
from nba_api.stats.endpoints.teamgamelog import TeamGameLog
from nba_api.stats.endpoints import boxscoreadvancedv2, LeagueHustleStatsTeam, LeagueDashTeamShotLocations  # checked the contents; put on discord
from nba_api.stats.endpoints import cumestatsteam  # checked the contents; put on discord
from nba_api.stats.endpoints import boxscoresummaryv2  # checked the contents; put on discord
from nba_api.stats.endpoints import boxscoretraditionalv2  # checked the contents
from nba_api.stats.endpoints import boxscorefourfactorsv2  # checked the contents
from nba_api.stats.endpoints import boxscoremiscv2  # checked the contents
from nba_api.stats.endpoints import boxscorematchupsv3  # checked the contents
from nba_api.stats.endpoints import boxscorematchupsv3  # checked the contents (duplicate import)
from nba_api.stats.endpoints import boxscorescoringv2  # checked the contents
from nba_api.stats.endpoints import boxscorehustlev2  # checked the contents; put on discord
from nba_api.stats.endpoints import boxscoreplayertrackv3  # checked the contents; put on discord
from nba_api.stats.endpoints import boxscoreusagev3  # checked the contents; put on discord
from nba_api.stats.endpoints import gamerotation  # checked the contents
from nba_api.stats.endpoints import leaguedashplayerstats  # checked the contents; put on discord
from nba_api.stats.endpoints import teamplayeronoffdetails
from nba_api.stats.endpoints import teamplayeronoffsummary
from nba_api.stats.static import players
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.endpoints import leaguedashteamshotlocations, LeagueDashTeamClutch
from nba_api.stats.endpoints import leaguehustlestatsteam
from nba_api.stats.endpoints import leagueplayerondetails
from nba_api.stats.endpoints import playerdashptshotdefend
from nba_api.stats.endpoints import leaguedashlineups
from nba_api.stats.endpoints import teamvsplayer
from nba_api.stats.endpoints import playerdashptshots
from nba_api.stats.endpoints import teamdashptshots, commonteamroster
import random
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectTimeout, HTTPError, ConnectionError

# stats_getter.py
import pandas as pd
from datetime import datetime

# --- Retry helper for nba_api calls -----------------------------------------
import os, time, random
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError, HTTPError




# ---- at top (imports) ----
import os, time, random
import pandas as pd
from cache_manager import stats_cache

# retries: handle both requests.* and urllib3.* timeouts
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError, HTTPError
try:
    from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeout
except Exception:
    Urllib3ReadTimeout = tuple()

import os, time, random
import pandas as pd
from cache_manager import stats_cache

# ---- pacing / retry config ----
import os, time, random, threading
from collections import defaultdict

# --- Minimal proxy rotation helpers (used only on failures) ---
import re as _re

# stats_getter.py  (minimal add)
import os

# --- Proxy pool setup: split one big pool into A/B ---------------------------
import os

# 1) Put your full pool here (or set env NBA_PROXY_POOL_ALL as a CSV)
#    Example shows the style of your screenshot; paste the whole list.

class FeatureTimeoutError(RuntimeError):
    """Raised when a stats call timed out after all retries."""
    pass


POOL_ALL = [
    "DIRECT",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10011",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10022",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10023",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10036",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10037",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10038",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10460",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10464",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10466",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10467",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10469",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10483",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10501",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10502",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10505",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10507",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10511",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10516",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10522",
    "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10524",
]

# If you prefer env, set NBA_PROXY_POOL_ALL="DIRECT,http://...,http://..." and uncomment:
# if os.getenv("NBA_PROXY_POOL_ALL"):
#     POOL_ALL = [p.strip() for p in os.getenv("NBA_PROXY_POOL_ALL").split(",") if p.strip()]

# Dedupe, preserve order
_seen = set()
ALL_DEDUP = []
for p in POOL_ALL:
    if p not in _seen:
        _seen.add(p)
        ALL_DEDUP.append(p)

# ---- replace the CORE_A / CORE_B / POOL_A / POOL_B block with this ----
# Determine desired number of worker partitions (default 2 for backwards-compat)
WORKER_COUNT = int(os.getenv("NBA_WORKER_COUNT", "5"))

# Make a list of CORE proxies (non-DIRECT) as you already do
CORE = [p for p in ALL_DEDUP if p.upper() != "DIRECT"]
HAS_DIRECT = any(p.upper() == "DIRECT" for p in ALL_DEDUP)

# Partition CORE into WORKER_COUNT slices using round-robin slicing
# CORE_PARTS[i] = CORE[i::WORKER_COUNT]
CORE_PARTS = [CORE[i::WORKER_COUNT] for i in range(WORKER_COUNT)]

# Build per-partition pools and optionally include DIRECT if present
PARTITION_POOLS = []
for part in CORE_PARTS:
    pool = (["DIRECT"] if HAS_DIRECT else []) + part
    PARTITION_POOLS.append(pool)

# Apply for this worker if NBA_PROXY_POOL not already provided
WORKER = os.environ.get("NBA_WORKER", "").strip().upper() or "X"
if "NBA_PROXY_POOL" not in os.environ and WORKER:
    # support labels like "A", "B", "C", ... (map A->0, B->1, ...)
    if len(WORKER) == 1 and 'A' <= WORKER <= 'Z':
        idx = ord(WORKER) - ord('A')
    else:
        # fallback: if numeric (1..N) or other, try int-1, else 0
        try:
            idx = max(0, int(WORKER) - 1)
        except Exception:
            idx = 0
    if idx < len(PARTITION_POOLS):
        chosen = PARTITION_POOLS[idx]
    else:
        # out-of-range worker label → round-robin map into available partitions
        chosen = PARTITION_POOLS[idx % len(PARTITION_POOLS)]
    os.environ["NBA_PROXY_POOL"] = ",".join(chosen)
    print(f"[proxy] Worker {WORKER} pool size: {len(chosen)} (DIRECT in pool: {HAS_DIRECT})")

def _apply_proxy(p: str | None):
    proxy_apply("DIRECT" if (not p or p == "DIRECT") else p)

def _current_proxy_marker() -> str:
    return proxy_current()

def _rotate_proxy(prev_marker: str):
    # delegate to coordinator to avoid collisions across processes
    proxy_rotate_on_failure(WORKER)



# --- Team name canonicalization ---------------------------------------------

# Expand this map as needed. Right side MUST match the spelling used by your data.
# stats_getter.py  (where your alias/canon map lives)

# --- Team name canonicalization ---------------------------------------------

def _normalize_key(s: str) -> str:
    s = str(s).strip().lower()
    # remove common punctuation/periods so "L.A." == "LA"
    for ch in [".", ","]:
        s = s.replace(ch, "")
    # collapse inner spaces
    s = " ".join(s.split())
    return s

# Map *keys must be lowercase* of _normalize_key; values are your canonical names
_TEAM_ALIASES: dict[str, str] = {
    # Clippers
    "la clippers": "Los Angeles Clippers",
    "l a clippers": "Los Angeles Clippers",
    "los angeles clippers": "Los Angeles Clippers",
    "clippers": "Los Angeles Clippers",

    # Lakers
    "la lakers": "Los Angeles Lakers",
    "l a lakers": "Los Angeles Lakers",
    "los angeles lakers": "Los Angeles Lakers",
    "lakers": "Los Angeles Lakers",

    # >>> Historical renames (primary ask)
    "charlotte bobcats": "Charlotte Hornets",
    "bobcats": "Charlotte Hornets",  # forgiving short form

    # (optional, but often helpful in older data)
    # "new orleans hornets": "New Orleans Pelicans",
    # "new jersey nets": "Brooklyn Nets",
    # "seattle supersonics": "Oklahoma City Thunder",
}

def canon_team_for_season(name: str, season: str) -> str:
    """
    Season-aware canonicalization for timeline/log lookups.
    Keeps legacy labels in seasons where the NBA data used them.
    """
    import stats_getter as sg
    n = sg.canon_team(str(name).strip())
    y = int(str(season)[:4])

    # Charlotte: Bobcats through 2013-14
    if n == "Charlotte Hornets" and y <= 2013:
        return "Charlotte Bobcats"

    # New Orleans: Pelicans from 2013-14; before that Hornets
    if n == "New Orleans Pelicans" and y <= 2012:
        return "New Orleans Hornets"

    # Nets: Brooklyn from 2012-13; before that New Jersey
    if n == "Brooklyn Nets" and y <= 2011:
        return "New Jersey Nets"

    return n


def canon_team(name: str) -> str:
    """
    Normalize a team display name to your canonical string used across the project.
    - Tolerates case, punctuation ('L.A.' vs 'LA'), extra spaces.
    - Maps historical names like 'Charlotte Bobcats' -> 'Charlotte Hornets'.
    """
    raw = str(name)
    key = _normalize_key(raw)
    return _TEAM_ALIASES.get(key, raw.strip())



def canon_df_team_names(df, col):
    """In-place normalize a dataframe column (or columns) of team names."""
    out = df.copy()
    if isinstance(col, (list, tuple)):
        for c in col:
            out = canon_df_team_names(out, c)
        return out
    if col is None or col not in out.columns:
        return out
    out[col] = out[col].astype(str).map(canon_team)  # or your mapping
    return out

# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

NBA_RETRIES          = int(os.getenv("NBA_RETRIES", "2"))
NBA_TIMEOUT          = _env_float("NBA_TIMEOUT", 12.0)      # per-attempt cap
NBA_STEADY_SLEEP     = _env_float("NBA_STEADY_SLEEP", 0.30) # default post-success sleep
NBA_JITTER           = _env_float("NBA_JITTER", 0.15)       # absolute jitter window (seconds)
NBA_GLOBAL_COOLDOWN  = _env_float("NBA_GLOBAL_COOLDOWN", 2.0)
NBA_BACKOFF_CAP      = _env_float("NBA_BACKOFF_CAP", 6.0)   # cap for exponential backoff

# Per-endpoint steady sleeps (can be tuned via env later if you like)
STEADY_SLEEP_BY_ENDPOINT = defaultdict(lambda: NBA_STEADY_SLEEP, {
    "HustleStatsBoxScore":           0.65,
    "LeagueDashTeamShotLocations":   0.60,
    "BoxScoreAdvancedV2":            0.40,
    "LeagueDashTeamStats":           0.40,
    "LeagueGameLog":                 0.50,   # ← add this
    "LeagueDashPlayerStats":         0.45,   # optional, often large
})

# Simple gate to guarantee a minimum spacing between calls to the same endpoint
class PaceGate:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._next_start = 0.0
        self._lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            if now < self._next_start:
                time.sleep(self._next_start - now)
                now = time.time()
            # reserve next slot
            self._next_start = now + self.min_interval

# Build one gate per endpoint
PACE_GATES = defaultdict(lambda: PaceGate(0.0))
for _ep, _sleep in dict(STEADY_SLEEP_BY_ENDPOINT).items():
    PACE_GATES[_ep] = PaceGate(_sleep)

# Global cool-down that applies after a failure (prevents cascaded retries across features)
_GLOBAL_COOLDOWN_UNTIL = 0.0

def _maybe_global_cooldown():
    delay = _GLOBAL_COOLDOWN_UNTIL - time.time()
    if delay > 0:
        time.sleep(delay)

def _trip_global_cooldown(seconds: float):
    global _GLOBAL_COOLDOWN_UNTIL
    _GLOBAL_COOLDOWN_UNTIL = max(_GLOBAL_COOLDOWN_UNTIL, time.time() + float(seconds))

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    # broaden as needed
    retry_keys = (
        "timeout", "timed out", "429", "too many requests", "503", "502",
        "bad gateway", "unavailable", "max retries", "read timed out", "connection",
    )
    return any(k in msg for k in retry_keys)

def _sleep_with_jitter(base: float):
    if base <= 0 and NBA_JITTER <= 0:
        return
    jitter = random.uniform(-NBA_JITTER/2.0, NBA_JITTER/2.0)
    time.sleep(max(0.0, base + jitter))


# Retryable exceptions (requests + urllib3)
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError, HTTPError
try:
    from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeout
except Exception:
    Urllib3ReadTimeout = tuple()

def _retry_nba(fetch_fn, *, endpoint: str, timeout: float | None = None, retries: int | None = None):
    """
    Run `fetch_fn(timeout)` with retries, per-endpoint pacing, a short
    global cool-down on failure, and steady post-success sleep + jitter.

    - `fetch_fn` MUST accept a single float `timeout` argument.
    - `endpoint` is a short name used for pacing (e.g., "HustleStatsBoxScore").
    """
    to = float(timeout if timeout is not None else NBA_TIMEOUT)
    max_tries = int(retries if retries is not None else NBA_RETRIES) + 1

    last_err = None
    for attempt in range(max_tries):
        # global cool-down (if a previous call failed recently)
        _maybe_global_cooldown()

        # per-endpoint pacing gate (avoids bursts)
        PACE_GATES[endpoint].wait()

        try:
            result = fetch_fn(to)  # fetch_fn must pass this timeout through to nba_api
            # success: post-success steady sleep with jitter (keeps you under rate limits)
            _sleep_with_jitter(STEADY_SLEEP_BY_ENDPOINT[endpoint])
            return result

        except Exception as e:
            last_err = e
            if not _is_retryable(e) or attempt >= max_tries - 1:
                break

            # NEW: on retryable failure, rotate to a different proxy than the one just used
            try:
                _rotate_proxy(_current_proxy_marker())
            except Exception:
                print("Could not rotate proxy succesfully")
                pass

            if NBA_GLOBAL_COOLDOWN > 0:
                _trip_global_cooldown(NBA_GLOBAL_COOLDOWN)

            back = min(NBA_BACKOFF_CAP, 1.0 * (2 ** attempt))
            _sleep_with_jitter(back)
    
    if _is_retryable(last_err) and "timeout" in str(last_err).lower():
        raise FeatureTimeoutError(f"{endpoint} timed out after {max_tries} attempts: {last_err}") from last_err
    
    raise last_err




# --- stats_getter.py ---

# bump this whenever you change the mapping logic
_CANON_LG_VERSION = 2

_FORCE_TEAM_MAP = {
    # hard overrides that should *always* apply in logs
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",   # optional
    "New Jersey Nets": "Brooklyn Nets",              # optional
}

def _canonize_league_game_log_df(df):
    """
    Normalize team names in LeagueGameLog so downstream joins never see aliases.
    Applies hard overrides first, then your general canon_team mapping.
    Works for both player ('P') and team ('T') modes.
    """
    import pandas as pd
    from stats_getter import canon_team

    if df is None or df.empty:
        out = df
    else:
        out = df.copy()
        cols = [c for c in ("TEAM_NAME","HOME_TEAM_NAME","VISITOR_TEAM_NAME") if c in out.columns]

        for c in cols:
            # 1) clean strings
            out[c] = out[c].astype(str).str.strip()
            # 2) force historical renames (Bobcats->Hornets, etc.)
            out[c] = out[c].replace(_FORCE_TEAM_MAP)
            # 3) run your general canonicalizer as a final sweep
            out[c] = out[c].map(canon_team)

        # (optional) normalize MATCHUP text too, if you rely on it elsewhere
        # if "MATCHUP" in out.columns:
        #     out["MATCHUP"] = out["MATCHUP"].astype(str).str.replace("CHA", "CHA", regex=False)

    # tag with a version so we can “heal” cached frames later
    if hasattr(out, "attrs"):
        out.attrs["__canonized_lg_ver__"] = _CANON_LG_VERSION
    return out



def get_league_game_log(season: str, *, player_or_team_abbreviation: str | None = "T", **kwargs) -> pd.DataFrame:
    mode = (player_or_team_abbreviation or "T").upper()
    if mode not in ("P", "T"):
        mode = "T"

    def _fetch(season: str, player_or_team_abbreviation: str, **kw):
        # kw can include date_from_nullable/date_to_nullable, etc.
        def _call(timeout: float = 10.0):
            from nba_api.stats.endpoints import LeagueGameLog
            return LeagueGameLog(
                season=season,
                player_or_team_abbreviation=player_or_team_abbreviation,
                timeout=timeout,
                **kw,
            ).get_data_frames()[0]
        # run through your retry/pacing wrapper
        return _retry_nba(lambda t: _call(timeout=t), endpoint="LeagueGameLog", timeout=25.0)

    df = stats_cache.get_or_fetch(
        "LeagueGameLog",
        _fetch,  # <-- accepts **kwargs
        season=season,
        player_or_team_abbreviation=mode,
        **kwargs,
    )

    if not hasattr(df, "attrs") or df.attrs.get("__canonized_lg_ver__") != _CANON_LG_VERSION:
        df = _canonize_league_game_log_df(df)
    return df


# --- stats_getter.py ---

def getLeagueDashTeamClutch(
    season: str,
    date_from: str,
    date_to: str,
    measure_type: str = "Advanced",
    per_mode: str = "PerGame",
) -> pd.DataFrame:
    """
    League-wide clutch stats for a given window (one call).
    Cached, retried, and returns all teams.
    """

    def fetch(season, date_from, date_to, measure_type, per_mode):
        def _call(timeout=10):
            from nba_api.stats.endpoints import LeagueDashTeamClutch
            df = LeagueDashTeamClutch(
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                season=season,
                measure_type_detailed_defense=measure_type,
                per_mode_detailed=per_mode,
                timeout=timeout,
            ).get_data_frames()[0]
            return df

        return _retry_nba(_call, endpoint="LeagueDashTeamClutch", timeout=12.0)

    return stats_cache.get_or_fetch(
        "LeagueDashTeamClutch_LEAGUE",
        fetch,
        season=season,
        date_from=date_from,
        date_to=date_to,
        measure_type=measure_type,
        per_mode=per_mode,
    )



import pandas as pd

def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def resolve_season_for_game_by_logs(date_str: str, home_team: str, away_team: str) -> str:
    """
    Resolve the TRUE NBA season for a given (date, home, away) by consulting
    LeagueGameLog caches. Handles odd calendars (lockouts, bubble).
    """
    day = pd.to_datetime(date_str).normalize()
    y = day.year
    # conservative candidate window (covers lockouts & bubble spillovers)
    candidates = [_season_label(y - 2), _season_label(y - 1),
                  _season_label(y),     _season_label(y + 1)]

    a = get_team_id(home_team)
    b = get_team_id(away_team)

    for season in candidates:
        log = get_league_game_log(season).copy()
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"]).dt.normalize()
        df = log[log["GAME_DATE"] == day]
        if df.empty:
            continue
        gids = set(df.loc[df["TEAM_ID"] == a, "GAME_ID"].astype(str)) & \
               set(df.loc[df["TEAM_ID"] == b, "GAME_ID"].astype(str))
        if gids:
            return season

    # Fallback: if any season has games on that date, return that season
    for season in candidates:
        log = get_league_game_log(season).copy()
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"]).dt.normalize()
        if not log[log["GAME_DATE"] == day].empty:
            return season

    raise ValueError(f"Could not resolve season for {home_team} vs {away_team} on {date_str}")



def get_game_id(home_team: str, season: str, date_str: str) -> str:
    """
    Look up the GAME_ID (10-char string) for the game where `home_team` was at home
    on `date_str` (MM/DD/YYYY) in `season`, using ONLY the cached LeagueGameLog(season).
    """
    def fetch(home_team: str, season: str, date_str: str) -> str:
        # Resolve team id (your get_team_id should already be cached)
        team_id = get_team_id(home_team)
        if team_id is None:
            raise ValueError(f"Unknown team: {home_team}")

        # Get the cached season log (one row per TEAM per GAME)
        df = get_league_game_log(season)

        # Canonicalize dates: compare normalized timestamps
        target = pd.to_datetime(date_str, errors="raise").normalize()
        game_dates = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()

        # Home row is the one with ' vs. ' in MATCHUP
        is_home = df["MATCHUP"].str.contains(r"\svs\.?\s", na=False)
        mask = (df["TEAM_ID"] == team_id) & is_home & (game_dates == target)

        rows = df.loc[mask, ["GAME_ID"]]
        if rows.empty:
            # Helpful debug: show any rows for this team on that date
            nearby = df.loc[(df["TEAM_ID"] == team_id) & (game_dates == target),
                            ["GAME_DATE", "MATCHUP", "GAME_ID"]]
            msg = f"No home game for {home_team} on {target.strftime('%m/%d/%Y')} in {season}."
            if not nearby.empty:
                msg += f"\nFound rows for that date:\n{nearby.to_string(index=False)}"
            raise ValueError(msg)

        # Ensure the 10-char format
        return str(rows.iloc[0]["GAME_ID"]).zfill(10)

    return stats_cache.get_or_fetch(
        "GameID_FromGameLog",
        fetch,
        home_team=home_team, season=season, date_str=date_str
    )



# stats_getter.py
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats as lds
from cache_manager import stats_cache  # make sure this import exists

def getLeagueDashTeamStats(team_name, season,
                           date_from=None, date_to=None,
                           measure_type="Base", per_mode="PerGame",
                           season_type_all_star="Regular Season",
                           timeout=30):
    from stats_getter import get_team_id as _get_team_id
    team_id = _get_team_id(team_name) if team_name else None

    def fetch(team_id, season, date_from, date_to, measure_type, per_mode, season_type_all_star, timeout):
        def _call(to=timeout):
            from nba_api.stats.endpoints import leaguedashteamstats as lds
            try:
                resp = lds.LeagueDashTeamStats(
                    league_id_nullable="00",
                    team_id_nullable=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed_defense=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            except TypeError:
                resp = lds.LeagueDashTeamStats(
                    league_id_nullable="00",
                    team_id_nullable=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            df = resp.get_data_frames()[0]
            if "GAME_DATE" in df.columns:
                import pandas as pd
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
            return df

        # use the same endpoint name you tune in your sleep map
        return _retry_nba(_call, endpoint="LeagueDashTeamStats", timeout=timeout)

    return stats_cache.get_or_fetch(
        "LeagueDashTeamStats:00",
        fetch,
        team_id=team_id, season=season,
        date_from=date_from, date_to=date_to,
        measure_type=measure_type, per_mode=per_mode,
        season_type_all_star=season_type_all_star, timeout=timeout,
        skip_if=lambda x: x is None or (hasattr(x, "empty") and x.empty),
    )


def get_player_id(player_name: str):
    """
    Cached lookup -> returns an int player_id or None.
    Uses nba_api.stats.static.players to avoid network-heavy endpoints.
    """
    norm_name = str(player_name).strip()

    def fetch(player_name):
        from nba_api.stats.static import players as players_static
        hits = players_static.find_players_by_full_name(player_name)
        if not hits:
            return None
        # Prefer exact full_name (case-insensitive), else first hit
        for h in hits:
            if h.get("full_name", "").strip().lower() == player_name.strip().lower():
                return int(h["id"])
        return int(hits[0]["id"])

    return stats_cache.get_or_fetch(
        "get_player_id",
        fetch,
        player_name=norm_name
    )


def get_team_id(team_name):
    """
    Returns the Team ID of the specified NBA team.
    
    Parameters:
        team_name (str): The desired team's name.
        
    Returns:
        int: The team's ID.
    """

    team_name = canon_team(team_name)
    
    def fetch(team_name):  # ✅ Accept the team_name parameter
        """Fetch team data from NBA API"""
        nba_teams = teams.get_teams()
        for team in nba_teams:
            if team['full_name'] == team_name:
                return team['id']
        return None  # Return None if team not found
    
    # Use cache - team data rarely changes so we can cache by team_name
    return stats_cache.get_or_fetch(
        'get_team_id',
        fetch,
        team_name=team_name
    )

def getPlayerID(player_name, season):
    """
    Returns the player ID of a desired NBA player during a specefied season

    Parameters:
        player_name (str): The desired NBA player
        season (str): The desired NBA season, formatted like 2019-20
    """
    stats_df = leaguedashplayerstats.LeagueDashPlayerStats(season=season).get_data_frames()[0]
    filtered = stats_df[stats_df['PLAYER_NAME'].str.lower() == player_name.lower()]
    if filtered.empty:
        raise ValueError(f"Player '{player_name}' not found for season '{season}'")
    return int(filtered.iloc[0]['PLAYER_ID'])


def get_team_abrev(team_name):
    """
    Returns the abbreviation of the desired NBA team.

    Parameters:
        team_name (str): The desired team's name.

    Returns:
        str: team_abrev
    """
    nba_teams = teams.get_teams()
    for team in nba_teams:
        if team['full_name'] == team_name:
            return team['abbreviation']
            break

def get_injured_players(team_name, season, date):
    """
    Returns the list of injured players for an NBA game

    params:
    team_name (str): The desired team's name.
    season (str): The desired season (formatted like "2019-20").
    date (str): The desired date (formatted like "2019/10/22").

    returns:
    inactive_players (list): list of injured players
    """
    inactive_players = []
    game_id = get_game_id(team_name, season, date)
    summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
    inactive_section = summary.inactive_players.get_data_frame()
    for i, player in inactive_section.iterrows():
        first_name = player['FIRST_NAME']
        last_name = player['LAST_NAME']
        inactive_players.append(first_name + " " + last_name)
            
    return inactive_players

def get_desired_game_roster(team_name, date, season):
    """
    Returns the list of players for a desired NBA team on a desired date
    
    params:
    team_name (str): The desired team's name.
    season (str): The desired season (formatted like "2019-20").
    date (str): The desired date (formatted like "10/22/2019").
    
    returns:
    roster (list): list of NBA players on that team
    """
    try:
        # Convert date from "MM/DD/YYYY" to "MMM DD, YYYY" format
        date_obj = datetime.strptime(date, "%m/%d/%Y")
        formatted_date = date_obj.strftime("%b %d, %Y").upper()
        
        # Get the game ID for that team and date
        game_id = get_game_id(team_name, season, formatted_date)
        
        if game_id is None:
            print(f"No game found for {team_name} on {date}")
            return []
        
        # Get the box score data
        roster = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        data_frames = roster.get_data_frames()
        df = data_frames[0]  # Get the first DataFrame
        
        # Determine team abbreviation from team name
        team_abbrev = get_team_abrev(team_name)
        
        # Filter for the specified team's players
        team_players = df[df['TEAM_ABBREVIATION'] == team_abbrev]
        
        # Return list of player names
        return team_players['PLAYER_NAME'].tolist()
        
    except ValueError as e:
        print(f"Error parsing date '{date}'. Please use MM/DD/YYYY format: {e}")
        return []
    except Exception as e:
        print(f"Error getting roster for {team_name} on {date}: {e}")
        return []

def getRoster(team_name: str, season: str, measure_type: str = "Base", per_mode: str = "PerGame"):
    """
    Return a clean list of player names for team_name in season.
    Defaults to Base/PerGame to align with other prev-season features for cache reuse.
    """
    try:
        df = getLeagueDashPlayerStats(
            team_name=team_name,      # your wrapper should pass team_id to the API
            season=season,
            measure_type=measure_type,
            per_mode=per_mode,
        )
        if df is None or df.empty or "PLAYER_NAME" not in df.columns:
            return []

        # Extra safety: lock to this team and real player rows
        from stats_getter import get_team_id
        tid = get_team_id(team_name)
        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"] == tid]

        df = df[df["PLAYER_NAME"].notna()]

        # Optional: keep only players who logged minutes/games
        if "GP" in df.columns:
            df = df[df["GP"] > 0]

        # Unique, trimmed names
        return (
            df["PLAYER_NAME"]
            .astype(str).str.strip()
            .dropna().unique().tolist()
        )
    except Exception as e:
        print(f"Error getting roster for {team_name} in {season}: {e}")
        return []
    

def getLeagueDashPlayerStats(
    team_name: str | None,
    season: str,
    *,
    date_from: str | None = None,     # "MM/DD/YYYY" or None
    date_to: str | None = None,       # "MM/DD/YYYY" or None
    measure_type: str = "Base",
    per_mode: str = "PerGame",
    season_type_all_star: str = "Regular Season",
    timeout: float = 20.0,
) -> pd.DataFrame:
    """
    Cached + retried LeagueDashPlayerStats, optionally bounded to a date window.
    If team_name is None/"" → league-wide; otherwise filters to that team only.

    Parameters mirror nba_api with safer names:
      - date_from/date_to → passed to date_from_nullable/date_to_nullable
      - measure_type → 'Base', 'Defense', 'Advanced', ...
      - per_mode → 'PerGame', 'Per36', ...

    Returns a DataFrame (empty on failure/unresolved team).
    """
    from stats_getter import get_team_id as _get_team_id

    # -------- League-wide path --------
    if not team_name:
        def fetch_league(season, date_from, date_to, measure_type, per_mode, season_type_all_star, timeout):
            def _call(to: float = 10.0):
                from nba_api.stats.endpoints import leaguedashplayerstats as ldps
                # Some nba_api versions renamed args; keep a guarded fallback like TeamStats
                try:
                    resp = ldps.LeagueDashPlayerStats(
                        season=season,
                        season_type_all_star=season_type_all_star,
                        date_from_nullable=date_from,
                        date_to_nullable=date_to,
                        measure_type_detailed_defense=measure_type,
                        per_mode_detailed=per_mode,
                        timeout=to,
                    )
                except TypeError:
                    resp = ldps.LeagueDashPlayerStats(
                        season=season,
                        season_type_all_star=season_type_all_star,
                        date_from_nullable=date_from,
                        date_to_nullable=date_to,
                        measure_type_detailed=measure_type,
                        per_mode_detailed=per_mode,
                        timeout=to,
                    )
                return resp.get_data_frames()[0]
            return _retry_nba(_call, endpoint="LeagueDashPlayerStats", timeout=timeout)

        return stats_cache.get_or_fetch(
            "LeagueDashPlayerStats_All",
            fetch_league,
            season=season,
            date_from=date_from, date_to=date_to,
            measure_type=measure_type, per_mode=per_mode,
            season_type_all_star=season_type_all_star,
            timeout=timeout,
        )

    # -------- Team-specific path --------
    team_id = _get_team_id(team_name)
    if not team_id:
        return pd.DataFrame()

    def fetch(team_id, season, date_from, date_to, measure_type, per_mode, season_type_all_star, timeout):
        def _call(to: float = 8.0):
            from nba_api.stats.endpoints import leaguedashplayerstats as ldps
            try:
                resp = ldps.LeagueDashPlayerStats(
                    team_id_nullable=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed_defense=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            except TypeError:
                resp = ldps.LeagueDashPlayerStats(
                    team_id_nullable=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            return resp.get_data_frames()[0]

        return _retry_nba(_call, endpoint="LeagueDashPlayerStats", timeout=timeout)

    return stats_cache.get_or_fetch(
        "LeagueDashPlayerStats",
        fetch,
        team_id=team_id,
        season=season,
        date_from=date_from, date_to=date_to,
        measure_type=measure_type, per_mode=per_mode,
        season_type_all_star=season_type_all_star,
        timeout=timeout,
    )



# --- stats_getter.py ---

# --------- 3) PERSONAL FOULS DRAWN per game (single team) ----------



def get_team_shot_locations(team_name, season, date_from, date_to,
                            measure_type="Base", per_mode_detailed="PerGame"):
    """Cached + retried LeagueDashTeamShotLocations"""

    def fetch(team_name, season, date_from, date_to, measure_type, per_mode_detailed):
        team_id = get_team_id(team_name)
        if not team_id:
            return pd.DataFrame()

        def _call(timeout=25):
            from nba_api.stats.endpoints import LeagueDashTeamShotLocations
            resp = LeagueDashTeamShotLocations(
                season=season,
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                team_id_nullable=team_id,
                measure_type_simple=measure_type,
                per_mode_detailed=per_mode_detailed,
                timeout=timeout
            )
            dfs = resp.get_data_frames()
            return dfs[0] if dfs else pd.DataFrame()

        return _retry_nba(_call, endpoint="LeagueDashTeamShotLocations", timeout=25)

    return stats_cache.get_or_fetch(
        'LeagueDashTeamShotLocations',
        fetch,
        team_name=team_name,
        season=season,
        date_from=date_from,
        date_to=date_to,
        measure_type=measure_type,
        per_mode_detailed=per_mode_detailed
    )


def getLeagueHustleTeamStats(team_id, season, date_from=None, date_to=None, per_mode="PerGame"):
    """Cached + retried LeagueHustleStatsTeam"""

    def fetch(team_id, season, date_from, date_to, per_mode):
        if not team_id:
            return pd.DataFrame()

        def _call(timeout=30):
            from nba_api.stats.endpoints import LeagueHustleStatsTeam
            resp = LeagueHustleStatsTeam(
                team_id_nullable=team_id,
                season=season,
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                per_mode_time=per_mode,
                timeout=timeout
            )
            return resp.get_data_frames()[0]

        return _retry_nba(_call, endpoint="LeagueHustleStatsTeam", timeout=30.0)

    return stats_cache.get_or_fetch(
        'LeagueHustleStatsTeam',
        fetch,
        team_id=team_id,
        season=season,
        date_from=date_from,
        date_to=date_to,
        per_mode=per_mode
    )



def get_team_pt_shots(team_name, season, date_from, date_to, per_mode="PerGame"):
    """Cached + retried TeamDashPtShots"""

    def fetch(team_name, season, date_from, date_to, per_mode):
        team_id = get_team_id(team_name)
        if not team_id:
            return pd.DataFrame()

        def _call(timeout=25):
            from nba_api.stats.endpoints import teamdashptshots
            resp = teamdashptshots.TeamDashPtShots(
                team_id=team_id,
                season=season,
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                per_mode_simple=per_mode,
                timeout=timeout
            )
            # this endpoint returns a list of frames
            dfs = resp.get_data_frames()
            return dfs if dfs else pd.DataFrame()

        return _retry_nba(_call, endpoint="TeamDashPtShots", timeout=25)

    return stats_cache.get_or_fetch(
        'TeamDashPtShots',
        fetch,
        team_name=team_name,
        season=season,
        date_from=date_from,
        date_to=date_to,
        per_mode=per_mode
    )

def getPlayerGameLogs(
    season: str,
    player_id: int,
    per_mode_simple: str = "PerGame",
    measure_type: str = "Advanced",
) -> pd.DataFrame:
    """
    Cached PlayerGameLogs for a single player & season.
    Params mirror your desired call:
      - per_mode_simple="PerGame"
      - measure_type="Advanced"
    Returns first dataframe (or empty on failure).
    """
    if not player_id:
        return pd.DataFrame()

    def fetch(season, player_id, per_mode_simple, measure_type):
        def _call(timeout=13):
            from nba_api.stats.endpoints import PlayerGameLogs
            resp = PlayerGameLogs(
                season_nullable=season,
                per_mode_simple_nullable=per_mode_simple,
                measure_type_player_game_logs_nullable=measure_type,
                player_id_nullable=player_id,
                timeout=timeout,
            )
            dfs = resp.get_data_frames()
            return dfs[0] if dfs else pd.DataFrame()
        return _retry_nba(_call, endpoint="PlayerGameLogs", timeout=20.0)

    return stats_cache.get_or_fetch(
        "PlayerGameLogs",
        fetch,
        season=season,
        player_id=player_id,
        per_mode_simple=per_mode_simple,
        measure_type=measure_type,
    )

from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=512)
def team_regular_season_range_by_id(team_id: int, season: str):
    """
    Return (date_from, date_to) as MM/DD/YYYY for the given team_id in `season`.
    Uses LeagueGameLog; filters to Regular Season if the column exists.
    """
    df = get_league_game_log(season).copy()
    if df is None or df.empty:
        return None, None

    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df[df["TEAM_ID"] == int(team_id)]
    if df.empty:
        return None, None

    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"].str.contains("Regular", case=False, na=False)]
        if df.empty:
            return None, None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    lo, hi = df["GAME_DATE"].min(), df["GAME_DATE"].max()
    if pd.isna(lo) or pd.isna(hi):
        return None, None
    return lo.strftime("%m/%d/%Y"), hi.strftime("%m/%d/%Y")


# --- Preseason league-wide Base/PerGame (cached) ---
def getPreseasonLeagueDashPlayerStats(season: str, measure_type: str = "Base", per_mode: str = "PerGame"):
    """
    League-wide preseason stats (Base/PerGame) for `season`.
    One call, cached. Used to grab rookie PPG quickly.
    """
    def fetch_league(season, measure_type, per_mode):
        def _call(timeout=15):
            from nba_api.stats.endpoints import leaguedashplayerstats
            resp = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Pre Season",
                measure_type_detailed_defense=measure_type,
                per_mode_detailed=per_mode,
                timeout=timeout,
            )
            dfs = resp.get_data_frames()
            return dfs[0] if dfs else pd.DataFrame()
        return _retry_nba(_call, endpoint="LeagueDashPlayerStats")   # uses your existing retry wrapper

    return stats_cache.get_or_fetch(
        "LeagueDashPlayerStats_Preseason_All",
        fetch_league,
        season=season, measure_type=measure_type, per_mode=per_mode
    )

# --- stats_getter.py additions ---
import os
import pandas as pd
from nba_api.stats.endpoints import HustleStatsBoxScore

def hustle_supported(season: str) -> bool:
    # Hustle box score availability starts 2016-17
    return int(season.split("-")[0]) >= 2016


# --- stats_getter.py additions ---
def build_hustle_team_game_table(season: str) -> pd.DataFrame:
    """
    For a season, return one row per (team, game) with date and hustle counts.
    Uses LeagueGameLog(season) (cached) to get game_ids & dates, then
    calls HustleStatsBoxScore(game_id) ONCE per game_id (cached).
    """
    if not hustle_supported(season):
        return pd.DataFrame(columns=[
            "season","GAME_ID","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE",
            "DEFLECTIONS","SCREEN_ASSISTS"
        ])

    # league log for unique games + canonical date
    log = get_league_game_log(season).copy()
    log["GAME_DATE_TS"] = pd.to_datetime(log["GAME_DATE"], errors="coerce").dt.normalize()
    unique_games = log.drop_duplicates("GAME_ID")[["GAME_ID","GAME_DATE_TS"]]

    rows = []
    for gid, gdate in unique_games.itertuples(index=False, name=None):
        team_rows = get_hustle_team_game_rows(gid)
        if not team_rows.empty:
            tr = team_rows.copy()
            tr["GAME_DATE_TS"] = gdate
            rows.append(tr)

    if not rows:
        return pd.DataFrame(columns=[
            "season","GAME_ID","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE",
            "DEFLECTIONS","SCREEN_ASSISTS"
        ])

    hustle = pd.concat(rows, ignore_index=True)

    # attach canonical GAME_DATE only (no IS_HOME)
    hustle = hustle.merge(
        log[["GAME_ID","GAME_DATE_TS"]],
        on="GAME_ID",
        how="left",
        validate="many_to_one",
    )

    hustle["season"] = season
    hustle = hustle.rename(columns={"GAME_DATE_TS": "GAME_DATE"})

    for c in ["DEFLECTIONS","SCREEN_ASSISTS"]:
        if c in hustle.columns:
            hustle[c] = pd.to_numeric(hustle[c], errors="coerce")

    return hustle[[
        "season","GAME_ID","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE",
        "DEFLECTIONS","SCREEN_ASSISTS"
    ]]



def get_hustle_table_cached(season: str, cache_path: str = ".cache/hustle_{season}.parquet") -> pd.DataFrame:
    os.makedirs(".cache", exist_ok=True)
    path = cache_path.format(season=season)
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = build_hustle_team_game_table(season)
    df.to_parquet(path, index=False)
    return df

import pandas as pd
from cache_manager import stats_cache
from nba_api.stats.endpoints import HustleStatsBoxScore
from stats_getter import get_league_game_log, get_team_id  # or adjust import path

def lookup_game_id_by_teams_date(season: str, date_str: str, home_team: str, away_team: str) -> str:
    """Return GAME_ID (zero-padded str) for the given matchup/date (cached via LeagueGameLog)."""
    log = get_league_game_log(season).copy()
    log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"]).dt.normalize()
    day = pd.to_datetime(date_str).normalize()
    a = get_team_id(home_team)
    b = get_team_id(away_team)
    df = log[log["GAME_DATE"] == day]
    gids = set(df.loc[df["TEAM_ID"] == a, "GAME_ID"].astype(str)) & \
           set(df.loc[df["TEAM_ID"] == b, "GAME_ID"].astype(str))
    if not gids:
        raise ValueError(f"No GAME_ID on {date_str} for {home_team} vs {away_team} in {season}")
    return next(iter(gids)).zfill(10)

import time, random
import pandas as pd
from requests.exceptions import ReadTimeout as RequestsReadTimeout
from nba_api.stats.endpoints import HustleStatsBoxScore
from cache_manager import stats_cache

import time, random
import pandas as pd
from requests.exceptions import ReadTimeout as RequestsReadTimeout
from nba_api.stats.endpoints import HustleStatsBoxScore
# from cache_manager import stats_cache  # assuming you already have this

from nba_api.stats.endpoints import HustleStatsBoxScore
import pandas as pd
from typing import List
from cache_manager import stats_cache  # adjust import to your project

def get_hustle_team_game_rows(game_id: str) -> pd.DataFrame:
    """
    TEAM-level hustle rows for a single GAME_ID, with caching + _retry_nba.
    Returns a 2-row DataFrame with columns:
      GAME_ID, TEAM_ID, TEAM_ABBREVIATION, DEFLECTIONS, SCREEN_ASSISTS
    """
    game_id = str(game_id).zfill(10)
    keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "DEFLECTIONS", "SCREEN_ASSISTS"]

    def _call(timeout: float = 12.0) -> pd.DataFrame:
        frames = HustleStatsBoxScore(game_id=game_id, timeout=timeout).get_data_frames()
        # pick the TEAM-level table (has TEAM_ID; *no* PLAYER_ID)
        team_df = next((df.copy() for df in frames if "TEAM_ID" in df.columns and "PLAYER_ID" not in df.columns), None)
        if team_df is None or team_df.empty:
            raise RuntimeError("HustleStatsBoxScore returned no TEAM table")

        if "GAME_ID" not in team_df.columns:
            team_df["GAME_ID"] = game_id

        for c in keep:
            if c not in team_df.columns:
                team_df[c] = pd.NA

        out = team_df.loc[:, [c for c in keep if c in team_df.columns]].copy()

        if out.shape[0] != 2:
            raise RuntimeError(f"Hustle TEAM rows != 2 for GAME_ID {game_id}")

        for c in ("DEFLECTIONS", "SCREEN_ASSISTS"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        return out

    def _skip_bad_hustle(df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return True
        cols = set(map(str, df.columns))
        # Sometimes the API returns only a status frame – don't cache that.
        if cols == {"GAME_ID", "HUSTLE_STATUS"}:
            return True
        need = {"GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"}
        measures = {"DEFLECTIONS", "SCREEN_ASSISTS"}
        if not need.issubset(cols):
            return True
        if measures.isdisjoint(cols):
            return True
        if df.shape[0] != 2:
            return True
        return False

    def _fetch(**_):
        # >>> THIS was missing the endpoint= argument <<<
        return _retry_nba(
            lambda t: _call(timeout=t),
            endpoint="HustleStatsBoxScore",
            timeout=NBA_TIMEOUT,
        )

    return stats_cache.get_or_fetch(
        "HustleStatsBoxScore",
        _fetch,
        skip_if=_skip_bad_hustle,
        game_id=game_id,
    )

def first_n_game_ids_for_team(season: str, team_name: str, n: int = 3) -> pd.DataFrame:
    """Return the first N GAME_IDs (and dates) for team_name in season."""
    tid = get_team_id(team_name)
    log = get_league_game_log(season).copy()
    log["GAME_DATE_TS"] = pd.to_datetime(log["GAME_DATE"], errors="coerce").dt.normalize()
    team_log = log[log["TEAM_ID"] == tid].sort_values("GAME_DATE_TS", kind="mergesort")
    out = team_log[["GAME_ID", "GAME_DATE_TS"]].drop_duplicates("GAME_ID").head(n)
    return out.reset_index(drop=True)

def build_hustle_subset(season: str, game_ids: list[str]) -> pd.DataFrame:
    """
    Build the same row format as your full builder, but only for game_ids.
    Returns columns: season, GAME_ID, TEAM_ID, TEAM_ABBREVIATION, GAME_DATE, IS_HOME, DEFLECTIONS, SCREEN_ASSISTS
    """
    # prepare home marker + canonical GAME_DATE from league log (once)
    log = get_league_game_log(season).copy()
    log["GAME_DATE_TS"] = pd.to_datetime(log["GAME_DATE"], errors="coerce").dt.normalize()
    log["_IS_HOME"] = log["MATCHUP"].str.contains(r"\svs\s", na=False)  # "vs" = home
    join_cols = ["GAME_ID", "TEAM_ID", "GAME_DATE_TS", "_IS_HOME"]

    rows = []
    for gid in game_ids:
        team_rows = get_hustle_team_game_rows(gid)  # returns two team rows
        if team_rows is None or team_rows.empty:
            continue
        team_rows = team_rows.copy()
        # Keep consistent set
        keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "DEFLECTIONS", "SCREEN_ASSISTS"]
        for c in keep:
            if c not in team_rows.columns:
                team_rows[c] = pd.NA
        team_rows = team_rows[keep]
        rows.append(team_rows)

    if not rows:
        return pd.DataFrame(columns=[
            "season","GAME_ID","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE",
            "IS_HOME","DEFLECTIONS","SCREEN_ASSISTS"
        ])

    hustle = pd.concat(rows, ignore_index=True)

    # attach GAME_DATE + IS_HOME from the log
    hustle = hustle.merge(log[join_cols], on=["GAME_ID", "TEAM_ID"], how="left")
    hustle = hustle.rename(columns={"GAME_DATE_TS": "GAME_DATE", "_IS_HOME": "IS_HOME"})
    hustle["season"] = season

    # numeric clean
    for c in ["DEFLECTIONS","SCREEN_ASSISTS"]:
        if c in hustle.columns:
            hustle[c] = pd.to_numeric(hustle[c], errors="coerce")

    return hustle[
        ["season","GAME_ID","TEAM_ID","TEAM_ABBREVIATION","GAME_DATE",
         "IS_HOME","DEFLECTIONS","SCREEN_ASSISTS"]
    ]


from datetime import datetime, date
from functools import lru_cache
import pandas as pd

from stats_getter import get_league_game_log  # your cached season log fetcher

def _season_key(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"

def _to_date(d) -> date:
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    s = str(d).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"season_for_date_smart: unsupported date format: {d!r}")

@lru_cache(maxsize=256)
def _season_range(season_key: str) -> tuple[date, date] | tuple[None, None]:
    """
    Return (start_date, end_date) from the LeagueGameLog for this season,
    normalized to dates. If unavailable/empty, returns (None, None).
    """
    try:
        log = get_league_game_log(season_key)
    except Exception:
        return (None, None)
    if log is None or len(log) == 0 or "GAME_DATE" not in log:
        return (None, None)
    gd = pd.to_datetime(log["GAME_DATE"], errors="coerce").dt.date.dropna()
    if gd.empty:
        return (None, None)
    return (gd.min(), gd.max())

def _fast_fallback(dt: date) -> str:
    """
    Calendar fallback with explicit exceptions for known anomalies:
    - 2019-20 ran until Oct 2020 (covered by logs, but keep normal rule here)
    - 2020-21 started Dec 22, 2020 -> dates in [2020-07-01, 2020-12-21] should
      NOT roll to 2020-21 by a July-1 rule; keep them on 2019-20.
    """
    # special late-start window: keep mapping to 2019-20
    if date(2020, 7, 1) <= dt <= date(2020, 12, 21):
        return "2019-20"

    # normal July-1 rollover
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return _season_key(start_year)

@lru_cache(maxsize=4096)
def season_for_date_smart(d) -> str:
    """
    Map a date to an NBA season key 'YYYY-YY'.

    Strategy:
    1) Load adjacent-season logs (seasons starting in dt.year-1 and dt.year).
       If dt ∈ [min(GAME_DATE), max(GAME_DATE)] for one of them, return that season.
    2) Otherwise, fall back to a fast calendar rule with a 2020-21 late-start fix.
    """
    dt = _to_date(d)

    # Candidate seasons by start year around the date
    candidates = (_season_key(dt.year - 1), _season_key(dt.year))

    chosen = None
    for sk in candidates:
        s, e = _season_range(sk)
        if s is not None and e is not None and s <= dt <= e:
            chosen = sk
            break

    if chosen:
        return chosen

    # Not inside either season’s actual game window (offseason / gap days) -> fallback
    return _fast_fallback(dt)


# --- League-wide Team Shot Locations (one call per day) ----------------------
def get_league_shot_locations(
    season: str,
    date_from: str | None,
    date_to: str | None,
    *,
    measure_type: str = "Base",          # "Base" or "Opponent"
    per_mode_detailed: str = "PerGame",
    season_type_all_star: str = "Regular Season",
    timeout: float = 25.0,
) -> pd.DataFrame:
    """
    League-wide version of Team Shot Locations (one row per team).
    Cached by (season, date_from, date_to, measure_type, per_mode_detailed, season_type_all_star).
    """
    def _call(timeout: float = 25.0) -> pd.DataFrame:
        resp = LeagueDashTeamShotLocations(
            season=season,
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            measure_type_simple=measure_type,   # "Base" or "Opponent"
            per_mode_detailed=per_mode_detailed,
            season_type_all_star=season_type_all_star,
            timeout=timeout,
        )
        dfs = resp.get_data_frames()
        return dfs[0] if dfs else pd.DataFrame()

    return stats_cache.get_or_fetch(
        "LeagueDashTeamShotLocations:LEAGUE",
        lambda **_: _retry_nba(lambda t: _call(timeout=t),
                               endpoint="LeagueDashTeamShotLocations",
                               timeout=30.0),
        season=season,
        date_from=date_from,
        date_to=date_to,
        measure_type=measure_type,
        per_mode_detailed=per_mode_detailed,
        season_type_all_star=season_type_all_star,
    )


def _team_row_from_league_shotloc(df: pd.DataFrame, team_name: str) -> pd.Series:
    """
    Robustly select a single team row from a league-wide shot-locations frame.
    Works whether the ID/Name columns are plain or MultiIndex level-0 == ''.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # Find TEAM_ID / TEAM_NAME columns in either flat or MultiIndex layout.
    def _find(colname):
        # exact match
        if colname in df.columns:
            return colname
        # MultiIndex: look for ('', 'TEAM_NAME') style
        for c in df.columns:
            try:
                if isinstance(c, tuple) and len(c) >= 2 and c[-1] == colname:
                    return c
            except Exception:
                pass
        return None

    name_col = _find("TEAM_NAME")
    if name_col is None:
        return pd.Series(dtype="float64")

    # Normalize naming just in case (LA vs Los Angeles)
    work = df.copy()
    work[name_col] = work[name_col].astype(str)
    work[name_col] = work[name_col].map(canon_team)

    row = work.loc[work[name_col] == canon_team(team_name)]
    return row.iloc[0] if not row.empty else pd.Series(dtype="float64")


def _corner3_fga_from_row(row: pd.Series) -> float:
    """
    Sum Left + Right Corner-3 FGA from a shot-locations row.
    Handles both ('Left Corner 3','FGA') and ('Left Corner 3','OPP_FGA').
    """
    def get2(zone: str) -> float:
        # try OPP_FGA first (opponent tables) then FGA (base tables)
        for stat in ("OPP_FGA", "FGA"):
            key = (zone, stat)
            if key in row.index:
                v = pd.to_numeric(row[key], errors="coerce")
                return float(v) if pd.notna(v) else 0.0
        return 0.0

    return get2("Left Corner 3") + get2("Right Corner 3")

# --- TeamPlayerOnOffDetails (cached, retryable) --------------------------------
from nba_api.stats.endpoints import teamplayeronoffdetails as _tpo

def getTeamPlayerOnOffDetails(team_name: str,
                              season: str,
                              date_from: str | None = None,
                              date_to: str | None = None,
                              measure_type: str = "Advanced",
                              per_mode: str = "PerGame",
                              season_type_all_star: str = "Regular Season",
                              timeout: float = 30.0) -> dict:
    """
    Cached/paced wrapper around nba_api TeamPlayerOnOffDetails.

    Returns a dict with keys:
        {"on": <DataFrame>, "off": <DataFrame>}
    where "on" is the team's metrics when each player is ON the court,
    and "off" is when that player is OFF the court.

    Notes
    - `measure_type` accepts "Base", "Advanced", "Four Factors", "Misc",
      "Scoring", "Opponent", "Usage".
    - `per_mode` usually "PerGame".
    - Respects your global pacing/retry and proxy rotation.
    """
    team_id = get_team_id(team_name)

    def fetch(team_id, season, date_from, date_to,
              measure_type, per_mode, season_type_all_star, timeout):
        def _call(to=timeout):
            # nba_api has had parameter name drift across versions; handle both.
            try:
                resp = _tpo.TeamPlayerOnOffDetails(
                    team_id=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            except TypeError:
                resp = _tpo.TeamPlayerOnOffDetails(
                    team_id=team_id,
                    season=season,
                    season_type_all_star=season_type_all_star,
                    date_from_nullable=date_from,
                    date_to_nullable=date_to,
                    measure_type_detailed_defense=measure_type,
                    per_mode_detailed=per_mode,
                    timeout=to,
                )
            return resp

        obj = _retry_nba(_call, endpoint="TeamPlayerOnOffDetails", timeout=timeout)

        # Prefer accessors if present; else fall back to frames list.
        try:
            on_df = obj.on_court.get_data_frame()
        except Exception:
            frames = obj.get_data_frames()
            on_df = frames[1] if frames else None

        try:
            off_df = obj.off_court.get_data_frame()
        except Exception:
            frames = obj.get_data_frames()
            off_df = frames[2] if len(frames) > 2 else None

        # Canonicalize team names if present (keeps downstream merges happy)
        on_df = canon_df_team_names(on_df, "TEAM_NAME") if on_df is not None else None
        off_df = canon_df_team_names(off_df, "TEAM_NAME") if off_df is not None else None

        return {"on": on_df, "off": off_df}

    return stats_cache.get_or_fetch(
        "TeamPlayerOnOffDetails",
        fetch,
        team_id=team_id,
        season=season,
        date_from=date_from,
        date_to=date_to,
        measure_type=measure_type,
        per_mode=per_mode,
        season_type_all_star=season_type_all_star,
        timeout=timeout,
        skip_if=lambda v: not isinstance(v, dict) or v.get("on") is None,
    )









    
    
    

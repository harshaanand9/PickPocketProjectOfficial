from __future__ import annotations
import glob
import os
import re
import advanced_ledger
from hustle_ledger import append_hustle_game
import FourFactors_ledger
import hustle_ledger
import misc_ledger
from misc_ledger import append_misc_game
from FourFactors_ledger import append_FourFactors_game  # or append_fourfactors_game
from elo_team_ledger import get_ELO_home as _elo_home, get_ELO_away as _elo_away
 # 14435 total data points
 
 

import shotloc_ptshot
import os, importlib, stats_getter

os.environ["NBA_PROXY_POOL"] = (
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10011," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10022," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10034," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10036," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10037," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10038," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10460," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10464," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10466," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10467," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10476," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10483," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10501," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10502," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10505," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10507," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10511," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10516," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10524," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10534," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10356," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10357," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10358," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10549," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10522," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10555," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10556," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10560," # WORKS
   "http://spbi6ee2j0:jD7Pk~5fMlcV4ty2bt@dc.decodo.com:10573," # WORKS

    
)

import os
if os.getenv("NBA_RESET_CACHE", "0") == "1":
    from cache_manager import stats_cache
    stats_cache.clear_all(all_workers=True)
    try:
        import advanced_ledger
        advanced_ledger.clear_ledger()
    except Exception:
        pass


# ---------------------------
# FAST PROFILES: pacing & sleeps
# ---------------------------
import os, importlib
import stats_getter
import shotloc_ptshot


# === Choose your profile ===
#   "A" â†’ Ambitious but safe (start here)
#   "B" â†’ Very aggressive (best with 2 processes on distinct proxies)
PROFILE = os.getenv("NBA_PACING_PROFILE", "A").strip().upper()
# training_set_loader_copy.py
# add near other imports
from proxy_coord import acquire as proxy_acquire, release as proxy_release, apply_proxy as proxy_apply, current_proxy as proxy_current

WORKER = os.environ.get("NBA_WORKER", "").strip().upper() or "X"


import os, pandas as pd
from datetime import datetime


def _maybe_rotate_after_quota(worker: str, games_on_this_proxy: int) -> tuple[bool, int]:
    """
    Returns (rotated, new_counter). Rotate after 3 games for proxies, 12 for DIRECT.
    """
    cur = proxy_current()
    quota = 12 if cur == "DIRECT" else 3
    if games_on_this_proxy >= quota:
        # release current and acquire another
        proxy_release(worker, cur)
        nxt = proxy_acquire(worker, exclude=cur)
        if nxt:
            proxy_apply(nxt)
            print(f"[proxy] ({worker}) post-quota rotate â†’ {nxt} (quota={quota})")
        else:
            # if nothing free, keep current; don't reset counter so weâ€™ll try again next game
            print(f"[proxy] ({worker}) rotate wanted but none free; staying on {cur}")
            return False, games_on_this_proxy
        return True, 0
    return False, games_on_this_proxy

def _ensure_proxy(worker: str, *, exclude: str | None = None):
    """Ensure we have a proxy reserved and applied for this worker."""
    cur = proxy_current()
    if cur and (exclude is None or cur != exclude):
        # already set; ensure it's reserved in state (idempotent acquire)
        # try to acquire same proxy; if in use by other worker, get a new one
        got = proxy_acquire(worker, exclude=None)  # may return a different one if cur is taken
        if got and got != cur:
            proxy_apply(got)
            print(f"[proxy] ({worker}) corrected reservation â†’ {got}")
        return
    nxt = proxy_acquire(worker, exclude=exclude)
    if nxt:
        proxy_apply(nxt)
        print(f"[proxy] ({worker}) initial â†’ {nxt}")


# Profile definitions
_profiles = {
    "A": {
        # env-level knobs (defaults for everything not overridden below)
        "NBA_STEADY_SLEEP": 0.15,   # keep default for non-overridden endpoints
        "NBA_JITTER":       0.08,   # Â±0.04 s
        "NBA_GLOBAL_COOLDOWN": 1.5,

        # per-endpoint steady sleeps (seconds)
        "endpoint_sleeps": {
            "LeagueGameLog":               0.35,
            "LeagueDashTeamStats":         0.25,
            "LeagueDashPlayerStats":       0.30,
            "TeamDashPtShots":             0.48,
            "LeagueDashTeamShotLocations": 0.48,
            "LeagueHustleStatsTeam":       0.55,
        },
    },
    "B": {
        "NBA_STEADY_SLEEP": 0.12,   # riskier default for everything else
        "NBA_JITTER":       0.10,   # Â±0.05 s
        "NBA_GLOBAL_COOLDOWN": 1.2,

        "endpoint_sleeps": {
            "LeagueGameLog":               0.32,
            "LeagueDashTeamStats":         0.22,
            "LeagueDashPlayerStats":       0.24,
            "TeamDashPtShots":             0.40,
            "LeagueDashTeamShotLocations": 0.40,
            "LeagueHustleStatsTeam":       0.46,
        },
    },
}

_cfg = _profiles.get(PROFILE, _profiles["A"])

# 1) Apply env knobs (stats_getter reads these at import-time)
os.environ.update({
    "NBA_STEADY_SLEEP":      f'{_cfg["NBA_STEADY_SLEEP"]}',
    "NBA_JITTER":            f'{_cfg["NBA_JITTER"]}',
    "NBA_GLOBAL_COOLDOWN":   f'{_cfg["NBA_GLOBAL_COOLDOWN"]}',
    # keep your existing NBA_RETRIES / NBA_TIMEOUT as-is
})

# 2) Reload modules that capture env at import
importlib.reload(stats_getter)
importlib.reload(shotloc_ptshot)

# 3) Per-endpoint overrides
from stats_getter import STEADY_SLEEP_BY_ENDPOINT, PACE_GATES, PaceGate

# Update the per-endpoint steady sleeps per the selected profile
STEADY_SLEEP_BY_ENDPOINT.update(_cfg["endpoint_sleeps"])

# 4) Rebuild gates so new sleeps take effect immediately
for ep, s in dict(STEADY_SLEEP_BY_ENDPOINT).items():
    PACE_GATES[ep] = PaceGate(s)

print(f"[PACE] Applied profile {PROFILE}: "
      f"default={_cfg['NBA_STEADY_SLEEP']}s, jitter={_cfg['NBA_JITTER']}s, "
      f"cooldown={_cfg['NBA_GLOBAL_COOLDOWN']}s; "
      f"overrides={_cfg['endpoint_sleeps']}")




import os


import math
import os
import numpy as np
from advanced_first_game import poss_first_game_mixed
from advanced_ledger import _save_ledger, get_advanced_team_game_rows
import features_loader_copy
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import LeagueDashTeamShotLocations, teamgamelog, LeagueDashPlayerStats, playergamelogs, PlayerGameLogs, HustleStatsBoxScore, BoxScoreAdvancedV2, LeagueGameLog,  LeagueDashTeamStats, LeagueDashTeamClutch, LeagueHustleStatsTeam, TeamGameLog, TeamGameLogs
from nba_api.stats.static import teams
import importlib
from stats_getter import build_hustle_subset, first_n_game_ids_for_team, getLeagueDashPlayerStats, get_player_id, getRoster
importlib.reload(features_loader_copy)
importlib.reload(hustle_ledger)
from features_loader_copy import *
from typing import Dict, List
import cache_manager
from cache_manager import stats_cache
import time
import random

# In training_set_loader.ipynb

from importlib import reload
import pandas as pd
import numpy as np

# ---- first_game_cache.py (inline in your file is fine) ----
import os
import pandas as pd

def _stable_sort_games(df_games: pd.DataFrame) -> pd.DataFrame:
    """Deterministic sort so 'idx' means the same thing on every run."""
    g = df_games.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    # tie-breaks to keep ordering stable if there are same-day duplicates
    return g.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)



# (season, team) -> pd.Timestamp (earliest date)
_FIRST_GAME_CACHE = {}
_TRAINING_SIG = None  # (mtime_ns, size)

from functools import lru_cache
from stats_getter import get_league_game_log, getLeagueHustleTeamStats
# you already import `teams` earlier: from nba_api.stats.static import teams

# --- helpers used by the functions below ---
from datetime import datetime
import pandas as pd
import math, time


import stats_getter

from advanced_ledger import append_adv_game


import json, os, time
from pathlib import Path

RUN_TAG = os.environ.get("NBA_RUN_TAG", "default")
_STATE_DIR = Path("runs") / RUN_TAG
_STATE_DIR.mkdir(parents=True, exist_ok=True)




# ----- idempotent ledger appends (shared helper) -----
def _is_empty(obj) -> bool:
    try:
        # pandas-like
        if hasattr(obj, "empty"):
            return bool(obj.empty)
        # sequence / mapping
        return len(obj) == 0
    except Exception:
        return obj is None

def _append_game_idempotent(mod_name: str,
                            get_rows_fn: str,
                            append_fn: str,
                            season: str,
                            gid: str,
                            quiet: bool = False):
    """
    Generic guard: if GAME_ID already exists in the ledger, skip append.
    Works whether get_* returns a DataFrame or a list/tuple.
    """
    mod = __import__(mod_name, fromlist=["*"])
    get_rows = getattr(mod, get_rows_fn)
    do_append = getattr(mod, append_fn)

    existing = get_rows(season, gid)
    if existing is None or _is_empty(existing):
        do_append(season, gid)
        if not quiet:
            print(f"[{mod_name}] appended GAME_ID={gid}")
    else:
        if not quiet:
            print(f"[{mod_name}] already has GAME_ID={gid}, skipping")

def append_adv_game_idempotent(season: str, gid: str, quiet: bool = False):
    _append_game_idempotent(
        mod_name="advanced_ledger",
        get_rows_fn="get_ledger_rows_for_game",
        append_fn="append_adv_game",
        season=season, gid=gid, quiet=quiet
    )
def append_misc_game_idempotent(season: str, gid: str, quiet: bool = False):
    _append_game_idempotent(
        mod_name="misc_ledger",
        get_rows_fn="get_ledger_rows_for_game",
        append_fn="append_misc_game",
        season=season, gid=gid, quiet=quiet
    )
def append_fourfactors_game_idempotent(season: str, gid: str, quiet: bool = False):
    _append_game_idempotent(
        mod_name="FourFactors_ledger",
        get_rows_fn="get_ledger_rows_for_game",
        append_fn="append_FourFactors_game",
        season=season, gid=gid, quiet=quiet
    )
def append_hustle_game_idempotent(season: str, gid: str, quiet: bool = False):
    _append_game_idempotent(
        mod_name="hustle_ledger",
        get_rows_fn="get_ledger_rows_for_game",
        append_fn="append_hustle_game",
        season=season, gid=gid, quiet=quiet
    )



def _seasons_from_env(seasons: list[str]) -> list[str]:
    """
    Returns the list of seasons THIS worker should process.
    Default: split seasons across workers (contiguous chunks), not games.
    Opt out by NBA_SPLIT_SEASONS=0 to let every worker see all seasons.
    """
    import os, math

    # if you want all workers to see all seasons (then games split elsewhere)
    if os.getenv("NBA_SPLIT_SEASONS", "1") == "0":
        return seasons

    worker = os.getenv("NBA_WORKER", "").strip().upper()
    worker_count = int(os.getenv("NBA_WORKER_COUNT", "1"))

    # Map worker label â†’ index (Aâ†’0, Bâ†’1, â€¦ or "1"â†’0, "2"â†’1)
    if len(worker) == 1 and "A" <= worker <= "Z":
        idx = ord(worker) - ord("A")
    else:
        try:
            idx = max(0, int(worker) - 1)
        except Exception:
            idx = 0
    idx = idx % max(worker_count, 1)

    # Strategy: contiguous chunks (keeps prior-season dependencies local)
    n = len(seasons)
    if worker_count <= 1 or n == 0:
        return seasons

    # size each chunk as ceil(n / worker_count)
    chunk = math.ceil(n / worker_count)
    start = idx * chunk
    end = min(start + chunk, n)
    chosen = seasons[start:end]
    return chosen or []



def get_season_games(season):
    """Get all games for a season from your existing data"""
    # Read from your training_set.csv or use the league game log
    df = pd.read_csv('training_set_complete.csv')
    df_season = df[df['season'] == season].copy()
    
    # If no data in training_set, get from API
    if df_season.empty:
        df_league = get_league_game_log(season)
        # Process to get unique games (home/away pairs)
        games = []
        seen_games = set()
        
        for _, row in df_league.iterrows():
            game_id = row['GAME_ID']
            if game_id not in seen_games:
                seen_games.add(game_id)
                # Determine home/away from MATCHUP
                if '@' in row['MATCHUP']:
                    # This team is away
                    continue
                else:
                    # This team is home
                    home_team = row['TEAM_NAME']
                    # Find corresponding away team row
                    away_row = df_league[
                        (df_league['GAME_ID'] == game_id) & 
                        (df_league['TEAM_ID'] != row['TEAM_ID'])
                    ].iloc[0]
                    away_team = away_row['TEAM_NAME']
                    
                    games.append({
                        'date': row['GAME_DATE'],
                        'home_team': home_team,
                        'away_team': away_team,
                        'season': season
                    })
        
        df_season = pd.DataFrame(games)
    
    return df_season

def get_previous_season(season):
    """Get the previous season string from current season"""
    try:
        start_year = int(season.split('-')[0])
        prev_start = start_year - 1
        prev_end = start_year
        return f"{prev_start}-{str(prev_end)[2:]}"
    except:
        return None

def _resolve_game_id(season: str, home_team: str, away_team: str, date_str: str) -> str | None:
    """
    Robustly find GAME_ID for (season, home_team vs away_team) on date_str (MM/DD/YYYY).
    Prefers the 'home' row (MATCHUP contains 'vs'). Returns a 10-char string or None.
    """
    lg = stats_getter.get_league_game_log(season).copy()
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"]).dt.normalize()
    day = pd.to_datetime(date_str).normalize()

    day_rows = lg[lg["GAME_DATE"] == day]
    if day_rows.empty:
        return None

    # Prefer explicit home row
    home_row = day_rows[
        (day_rows["TEAM_NAME"] == home_team) &
        (day_rows["MATCHUP"].astype(str).str.contains("vs", case=False, regex=False))
    ]
    if not home_row.empty:
        return str(home_row.iloc[0]["GAME_ID"]).zfill(10)

    # Fallback: group with exactly these two teams
    cand = day_rows[day_rows["TEAM_NAME"].isin([home_team, away_team])]
    grp = cand.groupby("GAME_ID").filter(lambda g: set(g["TEAM_NAME"]) == {home_team, away_team})
    if not grp.empty:
        return str(grp.iloc[0]["GAME_ID"]).zfill(10)

    return None



def is_first_game_of_season(team_name: str, season: str, date_str: str) -> bool:
    """
    Robust check using NBA API: return True if `date_str` is the team's first game of `season`.
    Uses get_league_game_log(season) instead of training_set.csv cache.

    Parameters:
        team_name : str   # e.g. "Boston Celtics"
        season    : str   # e.g. "2019-20"
        date_str  : str   # e.g. "10/25/2019"

    Returns:
        bool — True if this date is the first game for that team in that season.
    """
    import pandas as pd
    from datetime import datetime
    import stats_getter as sg
    try:
        lg = sg.get_league_game_log(season).copy()
    except Exception as e:
        print(f"[is_first_game_of_season] league log fetch failed for {season}: {e}")
        return False

    # Normalize and filter
    lg["TEAM_NAME"] = lg["TEAM_NAME"].astype(str)
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"], errors="coerce").dt.normalize()

    sub = lg[lg["TEAM_NAME"] == team_name]
    if sub.empty:
        print(f"[is_first_game_of_season] no games found for {team_name} in {season}")
        return False

    first_game_date = sub["GAME_DATE"].min()
    target_date = pd.to_datetime(date_str, errors="coerce").normalize()

    if pd.isna(first_game_date) or pd.isna(target_date):
        return False

    return target_date == first_game_date


def build_dataset_for_window(
    start_date="10/26/2015",
    end_date="11/26/2015",
    seasons=("2015-16",),
    output_file="nba_features_2015-10-26_to_2016-11-26.csv",
    save_every=10,
) -> pd.DataFrame:
    """
    Build a features dataset only for games between start_date and end_date (inclusive),
    over the provided seasons. Appends the current game to the advanced ledger
    **after** computing features (so priors evolve naturally).
    """
    try:
        globals()['get_adaptive_delay'] = lambda operation_type='default', attempt_number=1: 0.0
    except Exception:
        pass

    rows = []
    total_seen = 0
    start = pd.to_datetime(start_date, format="%m/%d/%Y")
    end   = pd.to_datetime(end_date,   format="%m/%d/%Y")

    for season in seasons:
        df_games = get_season_games(season).copy()
        if df_games.empty:
            print(f"[{season}] No games found in training_set.csv or via fallback.")
            continue

        df_games['date'] = pd.to_datetime(df_games['date'], format="%m/%d/%Y", errors='coerce')
        df_games = df_games[(df_games['date'] >= start) & (df_games['date'] <= end)]
        df_games = df_games.sort_values('date').reset_index(drop=True)
        if df_games.empty:
            print(f"[{season}] No games in window {start_date}â€“{end_date}.")
            continue

        print(f"Processing {len(df_games)} games for {season} within {start_date}â€“{end_date}...")
        for _, g in df_games.iterrows():
            dstr = g['date'].strftime("%m/%d/%Y"); home = g['home_team']; away = g['away_team']
            try:
                feats = calculate_game_features(home_team=home, away_team=away, date=dstr, season=season)
                feats['date'] = dstr; feats['home_team'] = home; feats['away_team'] = away; feats['season'] = season
                for col in ["home_money_line","away_money_line","home_spread","away_spread","home_score","away_score"]:
                    if col in g.index: feats[col] = g[col]
                rows.append(feats)
                total_seen += 1
                print(f"Game between {home} and {away} on {dstr} was recorded")

                if total_seen % int(save_every) == 0:
                    pd.DataFrame(rows).to_csv(output_file, index=False)
                    print(f"Saved progress after {total_seen} games -> {output_file}")

            except Exception as e:
                print(f"âš ï¸  Skipped {home} vs {away} on {dstr} ({season}): {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(output_file, index=False)
        print(f"âœ… Done. Wrote {len(df_out)} games to {output_file}")
    else:
        print("No rows written (empty selection or all failed).")
    return df_out



def initialize_features_dataframe():
    """Initialize an empty DataFrame whose columns align 1:1 with calculate_game_features()."""
    import pandas as pd

    # --- game/meta columns (unchanged) ---
    base_cols = [
        "date", "home_team", "away_team", "season",
        "home_money_line", "away_money_line",
        "home_spread", "away_spread",
        "home_score", "away_score",
    ]

    # --- feature columns (split home/away; include first-game fallbacks) ---
    feat = [
        # General availability / fatigue
        "home_b2b", "away_b2b",

        # ELO RATINGS
        "elo_home", "elo_away",

        # Hustle: allow for first-game 'stocks_*' and later 'stocks_deflections_*'
        "stocks_home", "stocks_away",
        "stocks_deflections_home", "stocks_deflections_away",
        "screen_assists_home", "screen_assists_away",

        # General defense
        "pf_home", "pf_away",

        # General offense
        "pace_home", "pace_away",

        # Clutch
        "clutch_netrtg_home", "clutch_netrtg_away",

        # Free throws
        "fta_rate_relative_home", "fta_rate_relative_away",
        "pfd_home", "pfd_away",

        # Rebounding (split)
        "home_dreb_pct", "home_oreb_pct",
        "away_dreb_pct", "away_oreb_pct",

        # Second-chance (per 100)
        "second_chance_pts_per_100_home", "second_chance_pts_per_100_away",

        # Playmaking
        "home_ast_ratio", "away_ast_ratio",
        "home_TOV_PCT", "away_TOV_PCT",
        "home_OPP_TOV_PCT", "away_OPP_TOV_PCT",
        "pct_ast_fgm_home", "pct_ast_fgm_away",

        # --- On-date player aggregates (new: 6) ---
        "home_player_agg_ppg", "away_player_agg_ppg",
        "home_player_agg_asstov", "away_player_agg_asstov",
        "home_player_agg_drebstock", "away_player_agg_drebstock",

        # --- Shooting profiles (defender distance) ---
        # Contested 3PT
        "home_contested3p_rate", "away_contested3p_rate",
        "home_contested3p_eff",  "away_contested3p_eff",
        "home_opp_contested3p_rate", "away_opp_contested3p_rate",
        "home_opp_contested3p_eff",  "away_opp_contested3p_eff",

        # Open 3PT
        "home_open3p_rate", "away_open3p_rate",
        "home_open3p_eff",  "away_open3p_eff",
        "home_opp_open3p_rate", "away_opp_open3p_rate",
        "home_opp_open3p_eff",  "away_opp_open3p_eff",

        # Contested 2PT
        "home_contested2p_rate", "away_contested2p_rate",
        "home_contested2p_eff",  "away_contested2p_eff",
        "home_opp_contested2p_rate", "away_opp_contested2p_rate",
        "home_opp_contested2p_eff",  "away_opp_contested2p_eff",

        # Open 2PT
        "home_open2p_rate", "away_open2p_rate",
        "home_open2p_eff",  "away_open2p_eff",
        "home_opp_open2p_rate", "away_opp_open2p_rate",
        "home_opp_open2p_eff",  "away_opp_open2p_eff",

        # --- Shot-location buckets ---
        # Corner 3
        "home_corner3p_rate", "away_corner3p_rate",
        "home_corner3p_eff",  "away_corner3p_eff",
        "home_opp_corner3p_rate", "away_opp_corner3p_rate",
        "home_opp_corner3p_eff",  "away_opp_corner3p_eff",

        # Above-the-break 3
        "home_abovebreak3p_rate", "away_abovebreak3p_rate",
        "home_abovebreak3p_eff",  "away_abovebreak3p_eff",
        "home_opp_abovebreak3p_rate", "away_opp_abovebreak3p_rate",
        "home_opp_abovebreak3p_eff",  "away_opp_abovebreak3p_eff",

        # Paint shots
        "home_paintshot_rate", "away_paintshot_rate",
        "home_paintshot_eff",  "away_paintshot_eff",
        "home_opp_paintshot_rate", "away_opp_paintshot_rate",
        "home_opp_paintshot_eff",  "away_opp_paintshot_eff",

        # Midrange shots
        "home_midrangeshot_rate", "away_midrangeshot_rate",
        "home_midrangeshot_eff",  "away_midrangeshot_eff",
        "home_opp_midrangeshot_rate", "away_opp_midrangeshot_rate",
        "home_opp_midrangeshot_eff",  "away_opp_midrangeshot_eff",

        # Misc scoring
        "home_pts_off_tov_rate", "away_pts_off_tov_rate",
        "home_pts_fastbreak_pg", "away_pts_fastbreak_pg",

        # Last 5 games (split)
        "recent_netrtg_home", "recent_netrtg_away",
        "recent_oreb_pct_home", "recent_oreb_pct_away",
        "recent_efg_pct_home",  "recent_efg_pct_away",
        "recent_tov_pct_home",  "recent_tov_pct_away",
        "recent_ft_rate_home",  "recent_ft_rate_away",

        # --- Last-year player aggregates (new: 6; 2nd-half window) ---
        "home_player_agg_ppg_last_year", "away_player_agg_ppg_last_year",
        "home_player_agg_asstov_last_year", "away_player_agg_asstov_last_year",
        "home_player_agg_stocksdreb_last_year", "away_player_agg_stocksdreb_last_year",

        # Prior-season anchors & counters
        "home_last_season_netrtg", "away_last_season_netrtg",
        "home_last_season_oreb_pct", "away_last_season_oreb_pct",
        "home_last_season_efg_pct",  "away_last_season_efg_pct",
        "home_last_season_tov_pct",  "away_last_season_tov_pct",
        "home_last_season_ft_rate",  "away_last_season_ft_rate",
        "home_netrtg_diff_prev_season", "away_netrtg_diff_prev_season",
        "home_game_number", "away_game_number",

    ]

        # --- Player slot features (ON−OFF deltas + POSS/G) ---
    _slot_metrics = ["off_rating", "def_rating", "reb_pct", "tm_tov_pct", "efg_pct"]  # (on−off)
    for _s in range(1, 10):  # slots 1..9
        # home
        for _m in _slot_metrics:
            feat.append(f"home_p{_s}_{_m}_onoff")
        feat.append(f"home_p{_s}_poss_pg")  # ON-court possessions per game (no delta)

        # away
        for _m in _slot_metrics:
            feat.append(f"away_p{_s}_{_m}_onoff")
        feat.append(f"away_p{_s}_poss_pg")


    cols = base_cols + feat
    return pd.DataFrame(columns=cols)



def get_team_season_for_stats(is_first_game, current_season):
    """Returns the appropriate season to use for stats based on game status"""
    if is_first_game:
        return get_previous_season(current_season)
    else:
        return current_season

def load_all_features(seasons=None, output_file='nba_features_2015_2025.csv'):
    """
    Load all 52 features for NBA games from 2015-16 through 2024-25
    """
    if seasons is None:
        seasons = ['2013-14','2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', 
                   '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
        
    seasons = _seasons_from_env(seasons)
    # Initialize results DataFrame
    all_features_df = initialize_features_dataframe()
    
    # Process each season
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Processing season: {season}")
        print(f"{'='*50}")
        
        try:
            season_features = process_season(season)
            all_features_df = pd.concat([all_features_df, season_features], 
                                       ignore_index=True)
            
            # Save intermediate results after each season
            all_features_df.to_csv(f'{output_file}.temp', index=False)
            print(f"âœ… Completed {season} - {len(season_features)} games processed")
            
        except Exception as e:
            import traceback

            # 1) Print the full stack trace (most helpful)
            print(f"âŒ Error processing {season}: {e}")
            traceback.print_exc()

            # 2) Also print the *origin* of the exception (deepest frame)
            tb = e.__traceback__
            while tb.tb_next:          # walk to the last frame
                tb = tb.tb_next
            frame = tb.tb_frame
            func  = frame.f_code.co_name
            file  = frame.f_code.co_filename
            line  = tb.tb_lineno
            print(f"â†³ Origin: {func} at {file}:{line}")

            continue
    
    # Save final results
    all_features_df.to_csv(output_file, index=False)
    print(f"\nâœ… All features saved to {output_file}")
    print(f"Total games processed: {len(all_features_df)}")
    
    return all_features_df

def process_season(season, mutate_ledgers: bool = True):
    import os
    import pandas as pd

    # --- timelines & schedule ---
    timeline_curr = get_roster_timeline(season)
    prev_season   = get_previous_season(season)
    timeline_prev = get_roster_timeline(prev_season)

    df_games = get_season_games(season)
    df_games = _stable_sort_games(df_games)  # deterministic order: ["date","home_team","away_team"]

    # --- league log (for GAME_ID resolution) ---
    from stats_getter import get_team_id, get_league_game_log
    league_log = get_league_game_log(season).copy()
    league_log["_DATE"] = pd.to_datetime(league_log["GAME_DATE"], errors="coerce").dt.normalize()

    def _resolve_game_id(date_str, home_name, away_name):
        hid, aid = get_team_id(home_name), get_team_id(away_name)
        if hid is None or aid is None:
            return None
        d = pd.to_datetime(date_str, errors="coerce").normalize()
        rows = league_log[(league_log["_DATE"] == d) & (league_log["TEAM_ID"].isin([hid, aid]))]
        gids = rows["GAME_ID"].dropna().unique().tolist()
        if len(gids) == 1:
            return str(gids[0])
        for gid in gids:
            sub = league_log[league_log["GAME_ID"] == gid]
            if set(sub["TEAM_ID"].unique()).issuperset({hid, aid}):
                return str(gid)
        return None

    # --- proxy boot ---
    _ensure_proxy(WORKER)
    games_on_this_proxy = 0

    # --- canonical season CSV path (single source of truth) ---
    csv_path = _season_csv_path(season)

    # warm already-written rows (return value only)
    season_features = []
    if os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            if not prev.empty:
                season_features.extend(prev.to_dict("records"))
                print(f"[resume] warmed {len(prev)} rows from {csv_path}")
        except Exception:
            pass

    # --- compute resume index from the CSV ONLY ---
    start_idx = 0
    if os.path.exists(csv_path):
        try:
            last = pd.read_csv(csv_path).tail(1)
            if not last.empty:
                last_date = pd.to_datetime(last.iloc[0]["date"], errors="coerce").normalize()
                last_home = str(last.iloc[0]["home_team"])
                last_away = str(last.iloc[0]["away_team"])

                kdf = df_games.copy()
                kdf["date"] = pd.to_datetime(kdf["date"], errors="coerce").dt.normalize()
                hit = kdf[
                    (kdf["date"] == last_date) &
                    (kdf["home_team"] == last_home) &
                    (kdf["away_team"] == last_away)
                ]
                if not hit.empty:
                    start_idx = int(hit.index[-1]) + 1
        except Exception:
            start_idx = 0

    # --- main loop ---
    for idx in range(start_idx, len(df_games)):
        game = df_games.iloc[idx]
        date_str = game['date'] if isinstance(game['date'], str) else game['date'].strftime("%m/%d/%Y")
        home = game['home_team']
        away = game['away_team']

        rotated, games_on_this_proxy = _maybe_rotate_after_quota(WORKER, games_on_this_proxy)
        if rotated:
            pass

        try:
            feats = calculate_game_features(
                home_team=home,
                away_team=away,
                date=date_str,
                season=season
            )
            games_on_this_proxy += 1

            # --- single, canonical append (writes metadata + features, correct header/order)
            append_feature_row_csv(
                season=season,
                date_str=date_str,
                home_team=home,
                away_team=away,
                features=feats,
            )

            # keep in-memory copy (optional)
            row_for_return = {'date': date_str, 'home_team': home, 'away_team': away, 'season': season}
            row_for_return.update(feats)
            season_features.append(row_for_return)

            # idempotent ledger appends (safe if re-run)
            if mutate_ledgers:
                gid = _resolve_game_id(date_str, home, away)
                if gid:
                    # always safe for these three
                    append_adv_game_idempotent(season, gid, quiet=False)
                    append_misc_game_idempotent(season, gid, quiet=False)
                    append_fourfactors_game_idempotent(season, gid, quiet=False)

                    # --- skip hustle for 2016-17 or earlier ---
                    from stats_getter import season_for_date_smart
                    sk = season_for_date_smart(date_str)          # e.g., "2015-16"
                    start_year = int(str(sk)[:4])                 # 2015

                    if start_year >= 2017:
                        append_hustle_game_idempotent(season, gid, quiet=False)
                    else:
                        print(f"[hustle_ledger] skipping {away} @ {home} on {date_str} "
                            f"(season {sk} ≤ 2016-17; hustle not available)")

            print(f"[{season}] {idx+1}/{len(df_games)} {away} @ {home} on {date_str} ✓")

        except KeyboardInterrupt:
            print("\n[graceful-exit] saved; stopping...")
            raise

        except Exception as e:
            print(f"⚠️  Skipped {away} @ {home} on {date_str} ({season}): {e}")

    return pd.DataFrame(season_features)



def calculate_game_features(home_team, away_team, date, season):
    """
    Returns a dict whose keys align 1:1 with initialize_features_dataframe().
    - Warms advanced ledger up to (date) for stability.
    - Uses cached first-game check to swap prior-season where needed.
    - Keeps hustle/first-game mixed fallbacks.
    """
    import numpy as np, time
    from datetime import datetime

    from stats_getter import canon_team

    


    home_team = canon_team(home_team)
    away_team = canon_team(away_team)

    print(home_team, away_team)



    # ---------- setup ----------
    features = {}
    feature_times = {}

    # normalize date to "MM/DD/YYYY"
    date_str = date if isinstance(date, str) else date.strftime("%m/%d/%Y")

        # --- print league-wide game number like "game 3/1230" ---
    try:
        # League log â†’ de-dup to 1 row per GAME_ID, sort by date then GAME_ID
        lg = get_league_game_log(season).copy()
        lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"]).dt.normalize()
        games = (
            lg[["GAME_ID", "GAME_DATE"]]
            .drop_duplicates()
            .sort_values(["GAME_DATE", "GAME_ID"])
            .reset_index(drop=True)
        )
        total_games = len(games)

        # Find this matchup's GAME_ID and its 1-based position
        gid = _resolve_game_id(season, home_team, away_team, date_str)
        if gid:
            mask = games["GAME_ID"].astype(str) == str(gid)
            idx_list = games.index[mask].tolist()
            if idx_list:
                print(f"ðŸ“… game {idx_list[0] + 1}/{total_games}")
            else:
                print(f"ðŸ“… game ?/{total_games} (unresolved GAME_ID position)")
        else:
            print(f"ðŸ“… game ?/{total_games} (couldn't resolve GAME_ID)")
    except Exception as e:
        print(f"(game index print skipped: {e})")



    # Per-team season selection (prior season on first game)
    home_is_first = is_first_game_of_season(home_team, season, date_str)
    away_is_first = is_first_game_of_season(away_team, season, date_str)
    home_season   = get_previous_season(season) if home_is_first else season
    away_season   = get_previous_season(season) if away_is_first else season
    first_game_any = home_is_first or away_is_first
    
    def try_feature(name, func, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            features[name] = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {name}: {e}")
            features[name] = np.nan
        finally:
            feature_times[name] = time.perf_counter() - t0
            print(f"â±  {name:35s} {feature_times[name]:7.4f}s")

    # ---------- General ----------
    try_feature('home_b2b', getB2B, home_team, home_season, date_str) # 100%
    try_feature('away_b2b', getB2B, away_team, away_season, date_str) # 100%

    # ---------- Elo ratings (pre-game as of the game date) ----------
    # Uses team parquets; functions should return the PRE-game Elo (previous game's ELO_POST)
    try_feature('elo_home', _elo_home, home_team, away_team, date_str)
    try_feature('elo_away', _elo_away, home_team, away_team, date_str)

    # ---------- Hustle (stocks/deflections + screens) ----------
    import re, math

    def _season_start_year(s: str) -> int:
        m = re.match(r"^(\d{4})-\d{2}$", str(s))
        if not m:
            raise ValueError(f"Bad season format: {s}")
        return int(m.group(1))

    def _plus12_nan_safe(x):
        try:
            return (x + 12.0) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else np.nan
        except Exception:
            return np.nan

    start_year = _season_start_year(season)
    use_fallback_defl = (start_year <= 2016) or first_game_any   # <= 2016-17 OR first game of any season

    if use_fallback_defl:
        # compute stocks; set stocks_deflections = stocks + 12
        try_feature("stocks_home", getStocks, home_team, home_season, date_str)
        try_feature("stocks_away", getStocks, away_team, away_season, date_str)

        features["stocks_deflections_home"] = _plus12_nan_safe(features.get("stocks_home"))
        features["stocks_deflections_away"] = _plus12_nan_safe(features.get("stocks_away"))

        # screen assists policy: first game -> NaN, else compute
        if first_game_any:
            features["screen_assists_home"] = np.nan
            features["screen_assists_away"] = np.nan
            feature_times["screen_assists_home"] = 0.0
            feature_times["screen_assists_away"] = 0.0
        else:
            try_feature("screen_assists_home", getScreenAssists, home_team, home_season, date_str)
            try_feature("screen_assists_away", getScreenAssists, away_team, away_season, date_str)
    else:
        # 2017-18+ and not first game -> call real deflections endpoint
        try_feature("stocks_deflections_home", getStocksDeflections, home_team, home_season, date_str)
        try_feature("stocks_deflections_away", getStocksDeflections, away_team, away_season, date_str)
        try_feature("screen_assists_home", getScreenAssists, home_team, home_season, date_str)
        try_feature("screen_assists_away", getScreenAssists, away_team, away_season, date_str)

    # Always include PF diff (general defense)
# ---------- General defense ----------
    try_feature("pf_home", getPF, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("pf_away", getPF, away_team, away_season, date_str)
    

   
    try_feature("pace_home", getPace, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("pace_away", getPace, away_team, away_season, date_str)

    # ---------- CLUTCH NETRTG (per team) ----------
    try_feature("clutch_netrtg_home", getClutchNetRtg, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("clutch_netrtg_away", getClutchNetRtg, away_team, away_season, date_str)

    # ---------- Free throws ----------
    # FTA-rate-relative is team-view (needs opponent context for "allowed" side)
    try_feature("fta_rate_relative_home", getFTARateRelative, home_team, away_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("fta_rate_relative_away", getFTARateRelative, away_team, home_team, away_season, date_str)

    # PFD per game (single team)
    try_feature("pfd_home", getPFD, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("pfd_away", getPFD, away_team, away_season, date_str)



   # ---------- Rebounding (split) ----------

    # Home team rebound %s
    try_feature("home_dreb_pct", getDrebPct, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("home_oreb_pct", getOrebPct, home_team, home_season, date_str)  # FIRST GAME FALLBACK WORKS

    # Away team rebound %s
    try_feature("away_dreb_pct", getDrebPct, away_team, away_season, date_str)
    try_feature("away_oreb_pct", getOrebPct, away_team, away_season, date_str)


    # ---------- Second-chance points per 100 ----------
    try_feature("second_chance_pts_per_100_home", misc_ledger.get_second_chance_pts_per_100, # FIRST GAME FALLBACK WORKS
                home_team, home_season, date_str)
    try_feature("second_chance_pts_per_100_away", misc_ledger.get_second_chance_pts_per_100,
                away_team, away_season, date_str)
    
    # ---------- Playmaking ----------
    # ---------- Assist ratio (per team) ----------
    try_feature("home_ast_ratio", getAstRatio, home_team, season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("away_ast_ratio", getAstRatio, away_team, season, date_str)


    # Turnovers â€“ split into team TOV% and opponent TOV% allowed
    try_feature("home_TOV_PCT", getTurnoverPct, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("away_TOV_PCT", getTurnoverPct, away_team, away_season, date_str)

    try_feature("home_OPP_TOV_PCT", getOppTurnoverPctAllowed, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("away_OPP_TOV_PCT", getOppTurnoverPctAllowed, away_team, away_season, date_str)

    # % of FGM that are assisted
    try_feature("pct_ast_fgm_home", getPctAstFGM, home_team, home_season, date_str) # FIRST GAME FALLBACK WORKS
    try_feature("pct_ast_fgm_away", getPctAstFGM, away_team, away_season, date_str)


    # ---------- 3PT shooting (directional: homeâ†’away & awayâ†’home) ----------
    # PT SHOTS buckets (defender distance)

    # Contested 3P Features:
    try_feature('home_contested3p_rate', shotloc_ptshot.contested_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_contested3p_rate', shotloc_ptshot.contested_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_contested3p_eff',  shotloc_ptshot.contested_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_contested3p_eff',  shotloc_ptshot.contested_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_contested3p_rate', shotloc_ptshot.opp_contested_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_contested3p_rate', shotloc_ptshot.opp_contested_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_contested3p_eff',  shotloc_ptshot.opp_contested_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_contested3p_eff',  shotloc_ptshot.opp_contested_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # OPEN 3P Features:
    try_feature('home_open3p_rate', shotloc_ptshot.open_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_open3p_rate', shotloc_ptshot.open_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_open3p_eff',  shotloc_ptshot.open_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_open3p_eff',  shotloc_ptshot.open_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_open3p_rate', shotloc_ptshot.opp_open_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_open3p_rate', shotloc_ptshot.opp_open_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_open3p_eff',  shotloc_ptshot.opp_open_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_open3p_eff',  shotloc_ptshot.opp_open_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # Contested 2P Features:
    try_feature('home_contested2p_rate', shotloc_ptshot.contested_2pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_contested2p_rate', shotloc_ptshot.contested_2pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_contested2p_eff',  shotloc_ptshot.contested_2pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_contested2p_eff',  shotloc_ptshot.contested_2pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_contested2p_rate', shotloc_ptshot.opp_contested_2pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_contested2p_rate', shotloc_ptshot.opp_contested_2pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_contested2p_eff',  shotloc_ptshot.opp_contested_2pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_contested2p_eff',  shotloc_ptshot.opp_contested_2pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # Open 2P Features:
    try_feature('home_open2p_rate', shotloc_ptshot.open_2pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_open2p_rate', shotloc_ptshot.open_2pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_open2p_eff',  shotloc_ptshot.open_2pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_open2p_eff',  shotloc_ptshot.open_2pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_open2p_rate', shotloc_ptshot.opp_open_2pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_open2p_rate', shotloc_ptshot.opp_open_2pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_open2p_eff',  shotloc_ptshot.opp_open_2pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_open2p_eff',  shotloc_ptshot.opp_open_2pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # SHOT-LOCATION buckets (zones)

    # Corner 3P Features:
    try_feature('home_corner3p_rate', shotloc_ptshot.corner_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_corner3p_rate', shotloc_ptshot.corner_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)  

    try_feature('home_corner3p_eff',  shotloc_ptshot.corner_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_corner3p_eff',  shotloc_ptshot.corner_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_corner3p_rate', shotloc_ptshot.opp_corner_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_corner3p_rate', shotloc_ptshot.opp_corner_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)

    try_feature('home_opp_corner3p_eff',  shotloc_ptshot.opp_corner_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)    
    try_feature('away_opp_corner3p_eff',  shotloc_ptshot.opp_corner_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # Above-the-break 3P Features:
    try_feature('home_abovebreak3p_rate', shotloc_ptshot.above_break_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_abovebreak3p_rate', shotloc_ptshot.above_break_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)   
    try_feature('home_abovebreak3p_eff',  shotloc_ptshot.above_break_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_abovebreak3p_eff',  shotloc_ptshot.above_break_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_abovebreak3p_rate', shotloc_ptshot.opp_above_break_3pt_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_abovebreak3p_rate', shotloc_ptshot.opp_above_break_3pt_rate_away, home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_abovebreak3p_eff',  shotloc_ptshot.opp_above_break_3pt_eff_home,  home_team, away_team, home_season, away_season, date_str)    
    try_feature('away_opp_abovebreak3p_eff',  shotloc_ptshot.opp_above_break_3pt_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # Paint Shot Features:
    try_feature('home_paintshot_rate', shotloc_ptshot.paint_shot_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_paintshot_rate', shotloc_ptshot.paint_shot_rate_away, home_team, away_team, home_season, away_season, date_str)
    try_feature('home_paintshot_eff',  shotloc_ptshot.paint_shot_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_paintshot_eff',  shotloc_ptshot.paint_shot_eff_away,  home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_paintshot_rate', shotloc_ptshot.opp_paint_shot_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_paintshot_rate', shotloc_ptshot.opp_paint_shot_rate_away, home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_paintshot_eff',  shotloc_ptshot.opp_paint_shot_eff_home,  home_team, away_team, home_season, away_season, date_str)    
    try_feature('away_opp_paintshot_eff',  shotloc_ptshot.opp_paint_shot_eff_away,  home_team, away_team, home_season, away_season, date_str) 

    # Mid Range Shot Features:
    try_feature('home_midrangeshot_rate', shotloc_ptshot.midrange_shot_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_midrangeshot_rate', shotloc_ptshot.midrange_shot_rate_away, home_team, away_team, home_season, away_season, date_str)
    try_feature('home_midrangeshot_eff',  shotloc_ptshot.midrange_shot_eff_home,  home_team, away_team, home_season, away_season, date_str)
    try_feature('away_midrangeshot_eff',  shotloc_ptshot.midrange_shot_eff_away,  home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_midrangeshot_rate', shotloc_ptshot.opp_midrange_shot_rate_home, home_team, away_team, home_season, away_season, date_str)
    try_feature('away_opp_midrangeshot_rate', shotloc_ptshot.opp_midrange_shot_rate_away, home_team, away_team, home_season, away_season, date_str)
    try_feature('home_opp_midrangeshot_eff',  shotloc_ptshot.opp_midrange_shot_eff_home,  home_team, away_team, home_season, away_season, date_str)    
    try_feature('away_opp_midrangeshot_eff',  shotloc_ptshot.opp_midrange_shot_eff_away,  home_team, away_team, home_season, away_season, date_str)

    # --------- Misc scoring (split) ----------
    try_feature('home_pts_off_tov_rate', misc_ledger.get_pts_off_tov_rate_home,
                home_team, away_team, home_season, away_season, date_str)
    try_feature('away_pts_off_tov_rate', misc_ledger.get_pts_off_tov_rate_away,
                home_team, away_team, home_season, away_season, date_str)

    try_feature('home_pts_fastbreak_pg', misc_ledger.get_pts_fb_pg_home,
                home_team, away_team, home_season, away_season, date_str)
    try_feature('away_pts_fastbreak_pg', misc_ledger.get_pts_fb_pg_away,
                home_team, away_team, home_season, away_season, date_str)


   # -------- Last 5 games (split) --------
    try_feature('recent_netrtg_home', get_recent_netrtg_home,
                home_team, away_team, season, season, date_str)
    try_feature('recent_netrtg_away', get_recent_netrtg_away,
                home_team, away_team, season, season, date_str)

    try_feature('recent_oreb_pct_home', get_recent_oreb_pct_home,
                home_team, away_team, season, season, date_str)
    try_feature('recent_oreb_pct_away', get_recent_oreb_pct_away,
                home_team, away_team, season, season, date_str)

    try_feature('recent_efg_pct_home', get_recent_efg_pct_home,
                home_team, away_team, season, season, date_str)
    try_feature('recent_efg_pct_away', get_recent_efg_pct_away,
                home_team, away_team, season, season, date_str)

    try_feature('recent_tov_pct_home', get_recent_tov_pct_home,
                home_team, away_team, season, season, date_str)
    try_feature('recent_tov_pct_away', get_recent_tov_pct_away,
                home_team, away_team, season, season, date_str)

    try_feature('recent_ft_rate_home', get_recent_ft_rate_home,
                home_team, away_team, season, season, date_str)
    try_feature('recent_ft_rate_away', get_recent_ft_rate_away,
                home_team, away_team, season, season, date_str)

    # ---------- Prior-season anchors ----------
    try_feature('home_last_season_netrtg', get_last_season_NETRTG, home_team, date_str)
    try_feature('away_last_season_netrtg', get_last_season_NETRTG, away_team, date_str)
    try_feature('home_last_season_oreb_pct', get_last_season_OREB_PCT, home_team, date_str)
    try_feature('away_last_season_oreb_pct', get_last_season_OREB_PCT, away_team, date_str)
    try_feature('home_last_season_efg_pct', FourFactors_ledger.get_last_season_EFG_PCT, home_team, date_str)
    try_feature('away_last_season_efg_pct', FourFactors_ledger.get_last_season_EFG_PCT, away_team, date_str)
    try_feature('home_last_season_tov_pct', FourFactors_ledger.get_last_season_TMV_TOV_PCT, home_team, date_str)
    try_feature('away_last_season_tov_pct', FourFactors_ledger.get_last_season_TMV_TOV_PCT, away_team, date_str)
    try_feature('home_last_season_ft_rate', get_last_season_FT_RATE, home_team, date_str)
    try_feature('away_last_season_ft_rate', get_last_season_FT_RATE, away_team, date_str)

    try_feature('home_netrtg_diff_prev_season', get_netrtg_diff_prev_season, home_team, season, date_str)
    try_feature('away_netrtg_diff_prev_season', get_netrtg_diff_prev_season, away_team, season, date_str)
    try_feature('home_game_number', get_game_number, home_team, season, date_str)
    try_feature('away_game_number', get_game_number, away_team, season, date_str)

    # ---------- Player-aggregated roster strength (12) ----------
    # Uses ONLY:
    #   getTeam_PlayerAggregated_{PPG, ASTTOV, DREBSTOCK}_on_date
    #   getTeam_PlayerAggregated_{PPG_last_year, AstTov_last_year, StocksDREB_last_year}

    # Build current-season timeline once (cached).
    timeline_curr = build_team_roster_timeline(season)

    # 1) On-date aggregates (6)
    try_feature('home_player_agg_ppg',
        getTeam_PlayerAggregated_PPG_on_date,
        home_team, date_str, season, timeline_curr
    )
    try_feature('away_player_agg_ppg',
        getTeam_PlayerAggregated_PPG_on_date,
        away_team, date_str, season, timeline_curr
    )

    try_feature('home_player_agg_asstov',
        getTeam_PlayerAggregated_ASTTOV_on_date,
        home_team, date_str, season, timeline_curr
    )

    try_feature('away_player_agg_asstov',
        getTeam_PlayerAggregated_ASTTOV_on_date,
    away_team, date_str, season, timeline_curr
    )

    try_feature('home_player_agg_drebstock',
        getTeam_PlayerAggregated_DREBSTOCK_on_date,
        home_team, date_str, season, timeline_curr
    )
    try_feature('away_player_agg_drebstock',
        getTeam_PlayerAggregated_DREBSTOCK_on_date,
        away_team, date_str, season, timeline_curr
    )

    # 2) Last-year aggregates (6)
    prev_season = get_previous_season(season)
    if prev_season:
        timeline_prev = build_team_roster_timeline(prev_season)

        try_feature('home_player_agg_ppg_last_year',
            getTeam_PlayerAggregated_PPG_last_year,
            home_team, prev_season, timeline_prev
        )
        try_feature('away_player_agg_ppg_last_year',
            getTeam_PlayerAggregated_PPG_last_year,
            away_team, prev_season, timeline_prev
        )

        try_feature('home_player_agg_asstov_last_year',
            getTeam_PlayerAggregated_AstTov_last_year,
            home_team, prev_season, timeline_prev
        )
        try_feature('away_player_agg_asstov_last_year',
            getTeam_PlayerAggregated_AstTov_last_year,
            away_team, prev_season, timeline_prev
        )

        try_feature('home_player_agg_stocksdreb_last_year',
            getTeam_PlayerAggregated_StocksDREB_last_year,
            home_team, prev_season, timeline_prev
        )
        try_feature('away_player_agg_stocksdreb_last_year',
            getTeam_PlayerAggregated_StocksDREB_last_year,
            away_team, prev_season, timeline_prev
        )
    else:
        # If there's no previous season (edge case), fill the *_last_year features with NaN
        for k in [
            'home_player_agg_ppg_last_year','away_player_agg_ppg_last_year',
            'home_player_agg_asttov_last_year','away_player_agg_asttov_last_year',
            'home_player_agg_stocksdreb_last_year','away_player_agg_stocksdreb_last_year',
        ]:
            features[k] = np.nan
            feature_times[k] = 0.0
            
    # ---- Player slot team metrics (slots 1..9) ----
    # Assumes functions like: getPlayer{slot}_Team_OFF_RATING, DEF_RATING, REB_PCT, TM_TOV_PCT, EFG_PCT, POSS

    _SLOT_ONOFF = [
        ("OFF_RATING", "off_rating"),
        ("DEF_RATING", "def_rating"),
        ("REB_PCT",    "reb_pct"),
        ("TM_TOV_PCT", "tm_tov_pct"),
        ("EFG_PCT",    "efg_pct"),
    ]

    for slot in range(1, 10):  # 1..9
        # on-off deltas
        for code_name, suffix in _SLOT_ONOFF:
            func_name = f"getPlayer{slot}_Team_{code_name}"
            func = globals().get(func_name)
            if not func:
                continue
            try_feature(f"home_p{slot}_{suffix}_onoff", func, home_team, season, date_str)
            try_feature(f"away_p{slot}_{suffix}_onoff", func, away_team, season, date_str)

        # possessions per game
        poss_func = globals().get(f"getPlayer{slot}_Team_POSS")
        if poss_func:
            try_feature(f"home_p{slot}_poss_pg", poss_func, home_team, season, date_str)
            try_feature(f"away_p{slot}_poss_pg", poss_func, away_team, season, date_str)


    game_id = stats_getter.get_game_id(home_team, season, date_str)
    if game_id is None:
        print(f"Could not resolve GAME_ID for {home_team} vs {away_team} on {date_str} ({season})")
        return None
    
     # (Optional) print a short timing summary
    print(f"\nâœ… Finished features for {home_team} vs {away_team} on {date_str} ({season})")
    slow = sorted(feature_times.items(), key=lambda x: -x[1])[:5]
    print(f"â±  Top slow features -> {slow}\n", flush=True)

    # ---------- per-game FULL feature summary (toggle via env) ----------
    import os, math
    if os.getenv("SHOW_FEATURE_SUMMARY", "1") == "1":
        # Match your print_full_features_* style
        precision = int(os.getenv("FEATURE_PRINT_PRECISION", "6"))
        sort_keys = os.getenv("FEATURE_PRINT_SORT_KEYS", "1") == "1"

        def _fmt(v):
            try:
                if v is None:
                    return "None"
                # numpy / float handling with NaN guard
                f = float(v)
                return "nan" if math.isnan(f) else f"{f:.{precision}g}"
            except Exception:
                return str(v)

        print(f"ðŸ“Š feature summary â€” {away_team} @ {home_team} on {date_str} ({season})")
        keys = sorted(features.keys()) if sort_keys else list(features.keys())
        for k in keys:
            print(f"{k}: {_fmt(features.get(k))}")
        print("â€” end of feature summary â€”\n")

        # ========== PRIORS SNAPSHOT (all features your functions use) ==========
        try:
            dt = datetime.strptime(date_str, "%m/%d/%Y")
            home_id = stats_getter.get_team_id(home_team)
            away_id = stats_getter.get_team_id(away_team)
            if home_id is None or away_id is None:
                print("âš ï¸  Could not resolve TEAM_ID(s) for priors")
            else:
                # -----------------------
                # helpers
                # -----------------------
                def _fmt(x):
                    try:
                        import math
                        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.6f}"
                    except Exception:
                        return "nan"              
                # -----------------------
                # ADVANCED priors
                # -----------------------
                adv_away = {
                    "NET_RATING": advanced_ledger.get_prior_net_rating(away_season, away_id, dt),
                    "EFG_PCT":    advanced_ledger.get_prior_efg_pct(away_season, away_id, dt),
                    "TM_TOV_PCT": advanced_ledger.get_prior_tm_tov_pct(away_season, away_id, dt),
                    "OREB_PCT":   advanced_ledger.get_prior_oreb_pct(away_season, away_id, dt),
                    "DREB_PCT":   advanced_ledger.get_prior_dreb_pct(away_season, away_id, dt),
                    "PACE":       advanced_ledger.get_prior_pace(away_season, away_id, dt),
                }
                adv_home = {
                    "NET_RATING": advanced_ledger.get_prior_net_rating(home_season, home_id, dt),
                    "EFG_PCT":    advanced_ledger.get_prior_efg_pct(home_season, home_id, dt),
                    "TM_TOV_PCT": advanced_ledger.get_prior_tm_tov_pct(home_season, home_id, dt),
                    "OREB_PCT":   advanced_ledger.get_prior_oreb_pct(home_season, home_id, dt),
                    "DREB_PCT":   advanced_ledger.get_prior_dreb_pct(home_season, home_id, dt),
                    "PACE":       advanced_ledger.get_prior_pace(home_season, home_id, dt),
                }
                print("ðŸ“Š Advanced priors (through previous day):")
                print(f"  {away_team:<25} NETRTG {_fmt(adv_away['NET_RATING'])}  "
                    f"EFG% {_fmt(adv_away['EFG_PCT'])}  TOV% {_fmt(adv_away['TM_TOV_PCT'])}  "
                    f"OREB% {_fmt(adv_away['OREB_PCT'])}  DREB% {_fmt(adv_away['DREB_PCT'])}  "
                    f"PACE {_fmt(adv_away['PACE'])}")
                print(f"  {home_team:<25} NETRTG {_fmt(adv_home['NET_RATING'])}  "
                    f"EFG% {_fmt(adv_home['EFG_PCT'])}  TOV% {_fmt(adv_home['TM_TOV_PCT'])}  "
                    f"OREB% {_fmt(adv_home['OREB_PCT'])}  DREB% {_fmt(adv_home['DREB_PCT'])}  "
                    f"PACE {_fmt(adv_home['PACE'])}")

                # -----------------------
                # Four Factors priors
                # -----------------------
                ff_away = {
                    "EFG_PCT":   FourFactors_ledger.get_prior_efg_pct(away_season, away_id, dt),
                    "FTA_RATE":  FourFactors_ledger.get_prior_fta_rate(away_season, away_id, dt),
                    "TM_TOV_PCT":FourFactors_ledger.get_prior_tm_tov_pct(away_season, away_id, dt),
                    "OREB_PCT":  FourFactors_ledger.get_prior_oreb_pct(away_season, away_id, dt),
                }
                ff_home = {
                    "EFG_PCT":   FourFactors_ledger.get_prior_efg_pct(home_season, home_id, dt),
                    "FTA_RATE":  FourFactors_ledger.get_prior_fta_rate(home_season, home_id, dt),
                    "TM_TOV_PCT":FourFactors_ledger.get_prior_tm_tov_pct(home_season, home_id, dt),
                    "OREB_PCT":  FourFactors_ledger.get_prior_oreb_pct(home_season, home_id, dt),
                }
                print("ðŸ§® Four Factors priors (through previous day):")
                print(f"  {away_team:<25} EFG% {_fmt(ff_away['EFG_PCT'])}  FTA_RATE {_fmt(ff_away['FTA_RATE'])}  "
                    f"TOV% {_fmt(ff_away['TM_TOV_PCT'])}  OREB% {_fmt(ff_away['OREB_PCT'])}")
                print(f"  {home_team:<25} EFG% {_fmt(ff_home['EFG_PCT'])}  FTA_RATE {_fmt(ff_home['FTA_RATE'])}  "
                    f"TOV% {_fmt(ff_home['TM_TOV_PCT'])}  OREB% {_fmt(ff_home['OREB_PCT'])}")

                # -----------------------
                # Misc priors
                # -----------------------
                misc_away = {
                    "PTS_OFF_TOV":   misc_ledger.get_prior_pts_off_tov(away_season, away_id, dt),
                    "PTS_FB":        misc_ledger.get_prior_pts_fb(away_season, away_id, dt),
                    "PTS_2ND_CHANCE":misc_ledger.get_prior_pts_2nd_chance(away_season, away_id, dt),
                }
                misc_home = {
                    "PTS_OFF_TOV":   misc_ledger.get_prior_pts_off_tov(home_season, home_id, dt),
                    "PTS_FB":        misc_ledger.get_prior_pts_fb(home_season, home_id, dt),
                    "PTS_2ND_CHANCE":misc_ledger.get_prior_pts_2nd_chance(home_season, home_id, dt),
                }
                print("ðŸ§± Misc priors (per-game, through previous day):")
                print(f"  {away_team:<25} PTS_OFF_TOV {_fmt(misc_away['PTS_OFF_TOV'])}  "
                    f"PTS_FB {_fmt(misc_away['PTS_FB'])}  "
                    f"PTS_2ND_CHANCE {_fmt(misc_away['PTS_2ND_CHANCE'])}")
                print(f"  {home_team:<25} PTS_OFF_TOV {_fmt(misc_home['PTS_OFF_TOV'])}  "
                    f"PTS_FB {_fmt(misc_home['PTS_FB'])}  "
                    f"PTS_2ND_CHANCE {_fmt(misc_home['PTS_2ND_CHANCE'])}")

                # -----------------------
                # Hustle priors (DEFLECTIONS/SCREEN_ASSISTS from ledger; STOCKS from league log)
                # -----------------------
                hustle_away = {
                    "DEFLECTIONS":    hustle_ledger.get_prior_deflections_pg(away_season, away_id, dt),
                    "SCREEN_ASSISTS": hustle_ledger.get_prior_screen_assists_pg(away_season, away_id, dt),
                    "STOCKS":         features_loader_copy._stocks_prior_pg(away_season, away_id, dt),
                }
                hustle_home = {
                    "DEFLECTIONS":    hustle_ledger.get_prior_deflections_pg(home_season, home_id, dt),
                    "SCREEN_ASSISTS": hustle_ledger.get_prior_screen_assists_pg(home_season, home_id, dt),
                    "STOCKS":         features_loader_copy._stocks_prior_pg(home_season, home_id, dt),
                }
                print("ðŸ§¹ Hustle priors (per-game, through previous day):")
                print(f"  {away_team:<25} DEFLECTIONS {_fmt(hustle_away['DEFLECTIONS'])}  "
                    f"SCREEN_AST {_fmt(hustle_away['SCREEN_ASSISTS'])}  "
                    f"STOCKS {_fmt(hustle_away['STOCKS'])}")
                print(f"  {home_team:<25} DEFLECTIONS {_fmt(hustle_home['DEFLECTIONS'])}  "
                    f"SCREEN_AST {_fmt(hustle_home['SCREEN_ASSISTS'])}  "
                    f"STOCKS {_fmt(hustle_home['STOCKS'])}")
                
                # =-----------------------
        except Exception as e:
            print(f"(Priors print skipped: {e})")
        # =====================================================================
    return features


def get_team_games_in_season(team_name, season):
    """Get all games for a specific team in a season"""
    try:
        # Try to get from training_set.csv first
        df = pd.read_csv('training_set_complete.csv')
        df_season = df[df['season'] == season]
        
        # Filter games where this team played (either home or away)
        team_games = df_season[
            (df_season['home_team'] == team_name) | 
            (df_season['away_team'] == team_name)
        ].copy()
        
        return team_games
        
    except Exception as e:
        print(f"Error getting team games: {e}")
        return pd.DataFrame()

# Note: Roster change functions are imported from features_loader_copy.py


# Note: Roster change functions are imported from features_loader_copy.py


def load_features_with_recovery(start_season=None):
    """Load features with ability to resume from a specific season"""
    seasons = ['2013-14','2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', 
               '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
    
    if start_season:
        start_idx = seasons.index(start_season)
        seasons = seasons[start_idx:]
    
    # Check if temp file exists
    temp_file = 'nba_features_2015_2025.csv.temp'
    if os.path.exists(temp_file):
        print(f"Found existing temp file. Loading previous progress...")
        existing_df = pd.read_csv(temp_file)
        processed_seasons = existing_df['season'].unique()
        seasons = [s for s in seasons if s not in processed_seasons]
        print(f"Resuming from seasons: {seasons}")
    
    return load_all_features(seasons)

def get_adaptive_delay(operation_type='default', attempt_number=1):
    """
    Calculate delay based on operation type and retry attempts
    
    operation_type: 'team_change', 'endpoint_change', 'same_endpoint', 'minor'
    attempt_number: Number of attempts (increases delay exponentially on retries)
    """
    base_delays = {
        'team_change': 3.0,      # Between different teams
        'endpoint_change': 2.5,   # Between different stat types
        'same_endpoint': 1.0,     # Same endpoint, different parameters
        'minor': 0.5             # Minor operations
    }
    
    base = base_delays.get(operation_type, 1.0)
    
    # Time of day adjustment
    hour = datetime.now().hour
    if 9 <= hour <= 17:
        base *= 1.2
    
    # Exponential backoff for retries
    if attempt_number > 1:
        base *= (1.5 ** (attempt_number - 1))
    
    # Add randomness
    final_delay = base + random.uniform(0, base * 0.5)
    
    return final_delay

def fix_dates_in_training_set(csv_file='training_set.csv', out_file=None):
    df = pd.read_csv(csv_file)
    # Convert 'date' to datetime, then to MM/DD/YYYY
    df['date'] = pd.to_datetime(df['date']).dt.strftime("%m/%d/%Y")
    # Save back to file
    if out_file is None:
        out_file = csv_file  # overwrite
    df.to_csv(out_file, index=False)
    print(f"Dates fixed and saved to {out_file}")

def load_features_for_window(season: str,
                             start_date: str,
                             end_date: str,
                             output_file: str = "nba_features_subset.csv",
                             disable_sleep: bool = True) -> pd.DataFrame:
    """
    Load features only for games in [start_date, end_date] (MM/DD/YYYY) for a single season,
    writing incrementally to disk. Builds the advanced ledger **as games pass**:
    after computing features for a game, we append that game's 2-row TEAM table.
    """
    if disable_sleep:
        globals()['get_adaptive_delay'] = lambda *a, **k: 0.0

    df_games = get_season_games(season).copy()
    if df_games.empty:
        print(f"No games found for season {season}.")
        return pd.DataFrame()

    df_games['date'] = pd.to_datetime(df_games['date'], format="%m/%d/%Y", errors='coerce')
    start = pd.to_datetime(start_date, format="%m/%d/%Y")
    end   = pd.to_datetime(end_date,   format="%m/%d/%Y")
    df_games = df_games[(df_games['date'] >= start) & (df_games['date'] <= end)]
    df_games = df_games.sort_values('date').reset_index(drop=True)

    if df_games.empty:
        print(f"No games in window {start_date}â€“{end_date} for {season}.")
        return pd.DataFrame()

    rows = []
    for i, game in df_games.iterrows():
        dstr = game['date'].strftime("%m/%d/%Y")
        home = game['home_team']; away = game['away_team']
        try:
            feats = calculate_game_features(
                home_team=home, away_team=away, date=dstr, season=season
            )
            # Base info
            feats['date'] = dstr
            feats['home_team'] = home
            feats['away_team'] = away
            feats['season'] = season
            for col in ['home_money_line','away_money_line','home_spread','away_spread','home_score','away_score']:
                if col in game.index:
                    feats[col] = game[col]
            rows.append(feats)

            # occasional save
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(output_file, index=False)
                print(f"Saved progress: {i+1}/{len(df_games)} games -> {output_file}")

        except Exception as e:
            print(f"âš ï¸  Error: {home} vs {away} on {dstr}: {e}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)
    print(f"âœ… Wrote {len(df_out)} games to {output_file}")
    return df_out

def clear_feature_csvs(
    search_dirs: List[str] = ("out_features", "."),   # where to look (relative to repo root)
    mode: str = "delete",                             # "delete" | "truncate" (truncate applies only to CSVs)
    dry_run: bool = True,                             # set False to actually modify files
    include_numbers: bool = True                      # also delete matching .numbers files
) -> List[str]:
    """
    Remove or truncate all season feature files:
      - CSVs:  features_YYYY-YY.csv  (in search_dirs)
      - Numbers: features_YYYY-YY.numbers (if include_numbers=True)

    Returns list of affected file paths.
    """
    # match hyphen OR en-dash between years
    season_rx = re.compile(r"^features_\d{4}[-–]\d{2}\.(csv|numbers)$", re.IGNORECASE)

    def _collect(dirpath: str) -> List[str]:
        hits = []
        # CSVs
        hits += [p for p in glob.glob(os.path.join(dirpath, "features_*.[cC][sS][vV]")) 
                 if season_rx.match(os.path.basename(p))]
        if include_numbers:
            # Numbers docs
            hits += [p for p in glob.glob(os.path.join(dirpath, "features_*.numbers")) 
                     if season_rx.match(os.path.basename(p))]
        return hits

    candidates: List[str] = []
    for d in search_dirs:
        candidates.extend(_collect(d))

        # also catch nested Numbers files under these dirs (common if saved in project root)
        if include_numbers:
            candidates += [
                p for p in glob.glob(os.path.join(d, "**", "features_*.numbers"), recursive=True)
                if season_rx.match(os.path.basename(p))
            ]

    affected: List[str] = []
    for p in sorted(set(candidates)):
        if dry_run:
            print(f"[dry-run] would {mode if p.lower().endswith('.csv') else 'delete'} {p}")
            affected.append(p)
            continue

        try:
            if p.lower().endswith(".csv"):
                if mode == "truncate":
                    with open(p, "w", encoding="utf-8") as f:
                        f.write("")
                    print(f"[truncated] {p}")
                else:
                    os.remove(p)
                    print(f"[deleted] {p}")
            else:
                # .numbers: always delete
                os.remove(p)
                print(f"[deleted] {p}")
            affected.append(p)
        except Exception as e:
            print(f"[error] {p}: {e}")

    if not affected and dry_run:
        print("[info] no matching feature files found")
    return affected


# --- CSV writing helpers: add metadata columns up front ----------------------
import os, io, pandas as pd
from pathlib import Path

import os

SEASON_CSV_DIR = os.environ.get("NBA_SEASON_CSV_DIR",
                                os.path.join(os.getcwd(), "out_features"))

def _season_csv_path(season: str) -> str:
    os.makedirs(SEASON_CSV_DIR, exist_ok=True)
    return os.path.join(SEASON_CSV_DIR, f"features_{season}.csv")


def _read_existing_columns(csv_path: Path) -> list[str] | None:
    """Return list of columns if file exists and is non-empty; else None."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    # Read the header only (fast)
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    if not header:
        return None
    return [c.strip() for c in header.split(",")]

def _ensure_season_csv_header(csv_path: Path, desired_cols: list[str]) -> None:
    """
    If file doesn't exist, create it with `desired_cols`.
    If it exists with different columns, rewrite it (one-time migration) so that
    'date','home_team','away_team' are present and first; other columns are preserved.
    """
    existing = _read_existing_columns(csv_path)
    if existing is None:
        # brand-new file
        pd.DataFrame(columns=desired_cols).to_csv(csv_path, index=False)
        return

    # If metadata already first and all cols match, nothing to do
    meta = ["date", "home_team", "away_team", "season"]
    if existing[:3] == meta and set(existing) == set(desired_cols):
        return

    # One-time migration: read whole file, add missing metadata cols (NaN), reorder, rewrite
    df_old = pd.read_csv(csv_path)
    for m in meta:
        if m not in df_old.columns:
            df_old[m] = pd.NA
    # Ensure any new feature columns also get added (rare)
    for c in desired_cols:
        if c not in df_old.columns:
            df_old[c] = pd.NA

    # Reorder: metadata first, then the rest following `desired_cols` order
    rest = [c for c in desired_cols if c not in meta]
    ordered = meta + rest
    df_old = df_old[ordered]
    tmp = csv_path.with_suffix(".csv.tmp")
    df_old.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)

def append_feature_row_csv(season: str, date_str: str, home_team: str, away_team: str, features: dict) -> None:
    """
    Append one row to features_{season}.csv with 'date','home_team','away_team','season' first.
    The rest of the columns follow the order of initialize_features_dataframe().
    """
    import pandas as pd

    csv_path = _season_csv_path(season)

    # --- canonical column order: metadata + your feature columns ---
    meta = ["date", "home_team", "away_team", "season"]   # <— include season

    try:
        cols_features = list(initialize_features_dataframe().columns)
    except Exception:
        cols_features = list(features.keys())

    desired_cols = meta + [c for c in cols_features if c not in meta]

    # ensure header exists (and in the right order)
    _ensure_season_csv_header(csv_path, desired_cols)

    # build one row in the exact order
    row = {"date": date_str, "home_team": home_team, "away_team": away_team, "season": season}
    row.update(features)

   # -- coerce blanks to NaN/NA so they don't write as empty cells --
    import math
    import pandas as pd

    def _nanify(v):
      if v is None:
          return pd.NA
      if isinstance(v, float) and math.isnan(v):
          return pd.NA
      if isinstance(v, str):
          s = v.strip()
          if s == "" or s.lower() in {"nan", "none", "null"}:
              return pd.NA
      return v

    data = {c: _nanify(row.get(c, pd.NA)) for c in desired_cols}
    df_row = pd.DataFrame([data], columns=desired_cols)

    # append without header (header already ensured above)
    df_row.to_csv(csv_path, mode="a", header=False, index=False, na_rep="nan")




# ---- advanced_first_game_mixed.py -------------------------------------------
from __future__ import annotations
import math
from functools import lru_cache
import pandas as pd
import numpy as np

from stats_getter import (
    get_team_id,
    resolve_season_for_game_by_logs,
    getLeagueDashTeamStats,  # your cached wrapper around LeagueDashTeamStats
)
from advanced_ledger import (
    ensure_advanced_for_matchup,
    get_prior_pace,
    get_prior_net_rating,
    get_prior_dreb_pct,
    get_prior_oreb_pct,
    get_prior_poss,
    get_prior_tm_tov_pct,
    get_prior_efg_pct,
)

# ---------- shared helpers (mirrors your hustle helpers) ---------------------

def _normalize_day(x):
    return pd.to_datetime(x, errors="raise").normalize()

# --- advanced first-game fallback helpers ---
from functools import lru_cache
import numpy as np
import pandas as pd
from stats_getter import get_team_id, getLeagueDashTeamStats
from advanced_ledger import (
    get_prior_pace, get_prior_net_rating, get_prior_dreb_pct, get_prior_oreb_pct,
    get_prior_poss, get_prior_tm_tov_pct, get_prior_efg_pct
)

def _prev_label(season: str) -> str:
    y0 = int(season.split("-")[0])        # e.g., 2016
    y1 = str(y0 - 1)                      # 2015
    return f"{y1}-{y0 % 100:02d}"         # 2015-16

@lru_cache(maxsize=1024)
def _prev_season_adv_pg(team_name: str, season: str, col: str) -> float:
    """
    Previous-season Advanced per-game value for `team_name`.
    Uses your wrapper: getLeagueDashTeamStats(team_name, season, ...).
    """
    prev = _prev_label(season)
    if not prev:
        return np.nan
    df = getLeagueDashTeamStats(team_name, prev, measure_type="Advanced", per_mode="PerGame")
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col].iloc[0], errors="coerce"))

def _prior_or_prev(team_name: str, season: str, date_str: str,
                   col: str, prior_getter) -> float:
    tid = get_team_id(team_name)
    val = prior_getter(season, tid, pd.to_datetime(date_str))

    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (_prev_season_poss_pg(team_name, season)
                if col == "POSS"
                else _prev_season_adv_value(team_name, season, col))
    return float(val)


def _mixed_prior(
    home_team: str, away_team: str,
    home_season: str, away_season: str, date_str: str,
    *,
    prior_getter_home, prior_getter_away,   # (season, team_id, date) -> float
    prev_col: str,                          # column name in LeagueDashTeamStats for fallback
) -> float:
    """
    Ensure ledger rows for this specific matchup exist, read each side's 'prior to this game'
    value from the advanced ledger, and if a side has NaN (first game), fall back to
    previous-season per-game ADV value. For POSS we compute POSS/G as POSS / GP.

    Returns: home_value - away_value
    """
    day = _normalize_day(date_str)

    # Resolve the true season for this calendar date & matchup (handles cross-season openers)
    season_hint = resolve_season_for_game_by_logs(str(day.date()), home_team, away_team)

    # Build only this game into the advanced ledger (no sweeping the whole season)
    ensure_advanced_for_matchup(season_hint, day)

    # Read 'prior to game' from your rolling ledger
    hid = get_team_id(home_team)
    aid = get_team_id(away_team)
    a = prior_getter_home(home_season, hid, day)
    b = prior_getter_away(away_season, aid, day)

    # --- First-game fallback (only for the side that is NaN) -----------------
    # Use your cached helpers to avoid double calls:
    #   - POSS: use _prev_season_poss_pg(team, season)  -> POSS / GP
    #   - other metrics: _prev_season_adv_value(team, season, prev_col)
    if a is None or (isinstance(a, float) and math.isnan(a)):
        a = (
            _prev_season_poss_pg(home_team, home_season)
            if prev_col == "POSS"
            else _prev_season_adv_value(home_team, home_season, prev_col)
        )
    if b is None or (isinstance(b, float) and math.isnan(b)):
        b = (
            _prev_season_poss_pg(away_team, away_season)
            if prev_col == "POSS"
            else _prev_season_adv_value(away_team, away_season, prev_col)
        )

    # If either side is still NaN, bubble up a NaN so callers can decide how to handle it.
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return float("nan")

    return float(a) - float(b)





@lru_cache(maxsize=256)
def _prev_season_row(team_name: str, season: str):
    prev = _prev_label(season)
    if not prev:
        return None
    df = getLeagueDashTeamStats(team_name, prev, measure_type="Advanced", per_mode="PerGame")
    if df is None or df.empty:
        return None
    # One row per team
    return df.iloc[0]

def _prev_season_poss_pg(team_name: str, season: str) -> float:
    row = _prev_season_row(team_name, season)
    if row is None:
        return float("nan")
    poss = pd.to_numeric(row.get("POSS"), errors="coerce")
    gp   = pd.to_numeric(row.get("GP"),   errors="coerce")
    return float(np.divide(poss, gp))  # robust to 0 or NaN

def _prev_season_adv_value(team_name: str, season: str, col: str) -> float:
    """Generic fetch for non-POSS metrics (NET_RATING, EFG_PCT, TM_TOV_PCT, etc.)."""
    row = _prev_season_row(team_name, season)
    if row is None or col not in row:
        return float("nan")
    v = pd.to_numeric(row[col], errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


# --------------------- metric-specific mixed functions -----------------------

def pace_first_game_mixed(home_team: str, away_team: str,
                          home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_pace,
        prior_getter_away=get_prior_pace,
        prev_col="PACE",   # LeagueDashTeamStats Advanced column
    )

def net_rating_first_game_mixed(home_team: str, away_team: str,
                                home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_net_rating,
        prior_getter_away=get_prior_net_rating,
        prev_col="NET_RATING",
    )

def dreb_pct_first_game_mixed(home_team: str, away_team: str,
                              home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_dreb_pct,
        prior_getter_away=get_prior_dreb_pct,
        prev_col="DREB_PCT",
    )

def oreb_pct_first_game_mixed(home_team: str, away_team: str,
                              home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_oreb_pct,
        prior_getter_away=get_prior_oreb_pct,
        prev_col="OREB_PCT",
    )

def poss_first_game_mixed(home_team: str, away_team: str,
                          home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_poss,
        prior_getter_away=get_prior_poss,
        prev_col="POSS",
    )


def tm_tov_pct_first_game_mixed(home_team: str, away_team: str,
                                home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_tm_tov_pct,
        prior_getter_away=get_prior_tm_tov_pct,
        prev_col="TM_TOV_PCT",
    )

def efg_pct_first_game_mixed(home_team: str, away_team: str,
                             home_season: str, away_season: str, date: str) -> float:
    return _mixed_prior(
        home_team, away_team, home_season, away_season, date,
        prior_getter_home=get_prior_efg_pct,
        prior_getter_away=get_prior_efg_pct,
        prev_col="EFG_PCT",
    )



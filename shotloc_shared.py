# shotloc_shared.py
from __future__ import annotations
import math
from functools import lru_cache
from datetime import datetime
import pandas as pd

import stats_getter
from stats_getter import get_team_id
from advanced_ledger import get_prior_poss
from cache_manager import stats_cache

# ---- minimal, robust MultiIndex getter used by shotloc wrappers ----
def _mi_get(df: pd.DataFrame, lvl1: str, lvl2: str, default=0.0):
    """
    Safely pull a value from a (possibly) MultiIndex columns DataFrame that came
    from LeagueDashTeamShotLocations. Falls back to `default` if missing.
    """
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # most common case from get_league_shot_locations(...)
            return pd.to_numeric(df.loc[:, (lvl1, lvl2)].iloc[0], errors="coerce")
        # occasional flat-column fallbacks (defensive)
        col = f"{lvl1}_{lvl2}"
        if col in df.columns:
            return pd.to_numeric(df[col].iloc[0], errors="coerce")
        if lvl2 in df.columns:
            return pd.to_numeric(df[lvl2].iloc[0], errors="coerce")
    except Exception:
        pass
    return default

# ---- possessions/game (ledger-first, API-fallback) with no dependency on features_loader_copy ----
@lru_cache(maxsize=16384)
def _ledger_poss_pg(team_id: int, season: str, date_str: str) -> float:
    day = pd.to_datetime(date_str).normalize()
    return float(get_prior_poss(season, team_id, day))

def poss_pg_from_ledger(team_name: str, season: str, date_str: str) -> float:
    """
    Preferred path: read possessions/game from our ledger up to date_str.
    Fallback: if ledger value is missing/NaN/non-positive, use Advanced totals from
    LeagueDashTeamStats for the SAME `season` and normalize by GP.
    """
    tid = get_team_id(team_name)
    if tid is None:
        return float("nan")

    def _fallback_poss_pg_via_advanced(team_id, season):
        # mirror your existing advanced fallback: Advanced totals ÷ GP
        try:
            df = stats_getter.getLeagueDashTeamStats(
                team_name=team_name, season=season,
                date_from=None, date_to=None,
                measure_type="Advanced", per_mode="PerGame"
            )
            if df is None or df.empty:
                return float("nan")
            # Team possessions per game is usually in "POSS" or derivable; be defensive:
            if "POSS" in df.columns:
                v = pd.to_numeric(df["POSS"].iloc[0], errors="coerce")
                return float(v) if pd.notna(v) and v > 0 else float("nan")
            return float("nan")
        except Exception:
            return float("nan")

    def _fetch(team_id, season, date_str):
        v = _ledger_poss_pg(team_id, season, date_str)
        if isinstance(v, float) and not math.isnan(v) and v > 0:
            return v
        return _fallback_poss_pg_via_advanced(team_id, season)

    return stats_cache.get_or_fetch(
        "poss_pg_any_source_v2", _fetch, team_id=tid, season=season, date_str=date_str
    )

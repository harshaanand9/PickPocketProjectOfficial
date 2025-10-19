# hustle_ledger.py
# Parquet-backed TEAM hustle ledger with on-the-fly priors (up to day BEFORE).
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence
import math
import pandas as pd

from cache_manager import stats_cache
from stats_getter import get_league_game_log, _retry_nba
import stats_getter  # e.g., for get_team_id if needed later
from ledger_io import load_df, save_df, path_for, debug_where
_KIND = "hustle"

# -------------------- config --------------------

# NOTE: STOCKS removed from the hustle ledger — no longer pulled from HustleStatsBoxScore.
HUSTLE_METRICS: Sequence[str] = (
    "DEFLECTIONS",
    "SCREEN_ASSISTS",
)

def get_ledger(
    season: str,
    *,
    team_id: int | None = None,
    game_id: str | None = None,
    after: str | None = None,   # "MM/DD/YYYY" inclusive
    before: str | None = None,  # "MM/DD/YYYY" inclusive
    head: int | None = 20,      # how many rows to print; None = print all
    sort: bool = True,
    print_summary: bool = True,
):
    """
    Load and (optionally) filter this module's ledger parquet for `season`,
    print a compact summary + preview, and return the (possibly filtered) DataFrame.

    Works identically across advanced_ledger.py, misc_ledger.py,
    FourFactors_ledger.py, and hustle_ledger.py (paste into each).
    """
    import pandas as pd

    df = _load_ledger(season)
    if df is None or df.empty:
        print(f"[{__name__}] ledger is empty for season {season}")
        return pd.DataFrame()

    # --- normalize common columns (robust across ledgers) ---
    if "GAME_ID" in df.columns:
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    if "TEAM_ID" in df.columns:
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()

    out = df

    # --- filters ---
    if team_id is not None and "TEAM_ID" in out.columns:
        out = out[out["TEAM_ID"] == int(team_id)]
    if game_id is not None and "GAME_ID" in out.columns:
        gid = str(game_id).zfill(10)
        out = out[out["GAME_ID"] == gid]
    if ("GAME_DATE" in out.columns) and (after or before):
        if after:
            ad = pd.to_datetime(after, errors="coerce").normalize()
            out = out[out["GAME_DATE"] >= ad]
        if before:
            bd = pd.to_datetime(before, errors="coerce").normalize()
            out = out[out["GAME_DATE"] <= bd]

    # --- ordering ---
    if sort:
        cols = [c for c in ["GAME_DATE", "GAME_ID", "TEAM_ID"] if c in out.columns]
        if cols:
            out = out.sort_values(cols, kind="mergesort").reset_index(drop=True)

    # --- summary ---
    if print_summary:
        parts = [f"season={season}", f"rows={len(out)}"]
        if "GAME_ID" in out.columns:
            parts.append(f"games={out['GAME_ID'].nunique()}")
        if "TEAM_ID" in out.columns:
            parts.append(f"teams={out['TEAM_ID'].nunique()}")
        if "GAME_DATE" in out.columns and not out.empty:
            parts.append(f"date_range=[{out['GAME_DATE'].min().date()} → {out['GAME_DATE'].max().date()}]")
        print(f"[{__name__}] " + " | ".join(parts))

    # --- preview print ---
    if head is not None:
        # Avoid super-wide dumps: show a small, informative subset if available
        preview_cols = [c for c in ["GAME_DATE", "GAME_ID", "TEAM_ID"] if c in out.columns]
        # add a few metric-ish columns if present
        preview_cols += [c for c in out.columns if c not in preview_cols][:7]
        print(out[preview_cols].head(head).to_string(index=False))

    return out




def _norm_day(x):
    v = pd.to_datetime(x, errors="coerce")
    if isinstance(v, pd.Series):        return v.dt.normalize()
    if isinstance(v, pd.DatetimeIndex): return v.normalize()
    return pd.Timestamp(v).normalize()

# -------------------- load/save --------------------

# advanced_ledger.py
def get_ledger_rows_for_game(season: str, game_id: str):
    """Return ledger rows for this GAME_ID (2 rows if present), else empty DataFrame."""
    import pandas as pd
    df = _load_ledger(season)
    if df is None or df.empty:
        return pd.DataFrame()
    gid = str(game_id).zfill(10)
    out = df[df["GAME_ID"].astype(str) == gid].copy()
    return out

_MEMO: dict[str, pd.DataFrame] = {}

def _load_ledger(season: str) -> pd.DataFrame | None:
    return load_df(_KIND, season)

def _save_ledger(season: str, df: pd.DataFrame) -> None:
    save_df(_KIND, season, df)

# -------------------- Hustle endpoint fetch (DEFLECTIONS, SCREEN_ASSISTS only) --------------------

def _skip_bad_hustle(df: pd.DataFrame) -> bool:
    return not isinstance(df, pd.DataFrame) or df.empty or "TEAM_ID" not in df.columns

def _fetch_hustle_team_table(game_id: str) -> pd.DataFrame:
    """Return 2 TEAM rows with DEFLECTIONS and SCREEN_ASSISTS (no STOCKS)."""
    import pandas as pd
    gid = str(game_id).zfill(10)

    def _process(frames) -> pd.DataFrame:
        team = next((f for f in frames
                     if isinstance(f, pd.DataFrame)
                     and "TEAM_ID" in f.columns
                     and "PLAYER_ID" not in f.columns), None)
        if team is None or team.empty:
            return pd.DataFrame()

        cols = team.columns
        out = team.loc[:, [c for c in ["GAME_ID","TEAM_ID","TEAM_ABBREVIATION",
                                       "DEFLECTIONS","SCREEN_ASSISTS"]
                           if c in cols]].copy()
        if "GAME_ID" not in out.columns:
            out["GAME_ID"] = gid

        for c in ["DEFLECTIONS","SCREEN_ASSISTS"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        keep = ["GAME_ID","TEAM_ID","TEAM_ABBREVIATION","DEFLECTIONS","SCREEN_ASSISTS"]
        return out[[c for c in keep if c in out.columns]]

    # Accept kwargs forwarded by stats_cache.get_or_fetch (e.g., game_id)
    def _fetch(*, game_id: str, timeout: float = 10.0, **_):
        from nba_api.stats.endpoints import HustleStatsBoxScore
        g_local = str(game_id).zfill(10)
        def _once(t):
            resp = HustleStatsBoxScore(game_id=g_local, timeout=t)
            return _process(resp.get_data_frames())
        return _retry_nba(lambda t: _once(t), endpoint="HustleStatsBoxScore", timeout=timeout)

    return stats_cache.get_or_fetch(
        "HustleStatsBoxScore_Team",
        _fetch,
        game_id=gid,                 # used for both cache key + fetch kwargs
        skip_if=_skip_bad_hustle,
    )


# -------------------- appender --------------------

def append_hustle_game(season: str, game_id: str) -> None:
    """Append TEAM hustle rows for GAME_ID. Call AFTER computing that game's features."""
    gl = get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce").dt.normalize()
    g = gl[gl["GAME_ID"].astype(str) == str(game_id)]
    if g.empty:
        return
    gdate = g["GAME_DATE"].iloc[0]

    rows = _fetch_hustle_team_table(game_id)
    if _skip_bad_hustle(rows):
        return

    rows = rows.copy()
    rows["SEASON"] = season
    rows["GAME_DATE"] = gdate

    led = _load_ledger(season)
    led = pd.concat([led, rows], ignore_index=True)
    led.drop_duplicates(subset=["TEAM_ID","GAME_ID"], keep="last", inplace=True)
    led.sort_values(["TEAM_ID","GAME_DATE","GAME_ID"], inplace=True)
    _save_ledger(season, led)

# -------------------- display helpers (like show_team_advanced_ledger) --------------------

def show_team_hustle_ledger(
    season: str,
    team: int | str,
    last_n: int = 12,
    include_priors: bool = True,
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Return a tidy slice of the hustle ledger for a team, sorted by date,
    optionally with on-the-fly prior (shifted expanding mean) columns.

    - `team` can be TEAM_ID (int) or team name (str).
    - `last_n=None` shows all rows.
    """
    import pandas as pd
    import numpy as np

    # resolve team_id from either id or name
    if isinstance(team, (int, np.integer)):
        team_id = int(team)
        team_name = None
    else:
        # use your canonicalizer + id lookup from stats_getter
        tname = stats_getter.canon_team(str(team))
        team_id = int(stats_getter.get_team_id(tname))
        team_name = tname

    df = _load_ledger(season).copy()
    if df.empty:
        print(f"[hustle_ledger] empty ledger for {season}")
        return df

    # normalize and filter slice
    df["GAME_DATE"] = _norm_day(df["GAME_DATE"])
    s = df[df["TEAM_ID"].astype(int) == team_id].copy()
    if s.empty:
        label = team_name or team_id
        print(f"[hustle_ledger] no rows for {label} in {season}")
        return s

    s.sort_values(["GAME_DATE", "GAME_ID"], inplace=True, ignore_index=True)

    # optional: compute priors on the fly (strictly before each game)
    if include_priors:
        for m in HUSTLE_METRICS:
            if m in s.columns:
                vals = pd.to_numeric(s[m], errors="coerce")
                s[f"{m}_prior"] = vals.shift(1).expanding().mean()

    # view + rounding
    if isinstance(last_n, int) and last_n > 0:
        s = s.tail(last_n).copy()

    num_cols = [c for c in s.columns if s[c].dtype.kind in "fc"]
    if round_digits is not None and num_cols:
        s[num_cols] = s[num_cols].round(round_digits)

    cols = ["SEASON", "GAME_DATE", "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]
    for m in HUSTLE_METRICS:
        if m in s.columns: cols.append(m)
        if include_priors and f"{m}_prior" in s.columns: cols.append(f"{m}_prior")

    print(f"[hustle_ledger] {season} — team_id={team_id} — showing {len(s)} rows")
    return s[cols]


# -------------------- prior getters (mean of games strictly before day) --------------------

def _prior_mean(season: str, team_id: int, game_date, metric: str) -> float:
    day = _norm_day(game_date)
    df = _load_ledger(season)
    if df.empty or metric not in df.columns:
        return math.nan
    pre = df[(df["TEAM_ID"].astype(int) == int(team_id)) & (df["GAME_DATE"] < day)]
    if pre.empty:
        return math.nan
    vals = pd.to_numeric(pre[metric], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else math.nan

def get_prior_deflections_pg(season: str, team_id: int, game_date) -> float:
    return _prior_mean(season, team_id, game_date, "DEFLECTIONS")

def get_prior_screen_assists_pg(season: str, team_id: int, game_date) -> float:
    return _prior_mean(season, team_id, game_date, "SCREEN_ASSISTS")

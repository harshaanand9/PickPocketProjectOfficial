# four_factors_ledger.py
# Incremental “ledger” for BoxScoreFourFactorsV2 (TEAM table) with prior (to-date) averages.
# Mirrors the design & API shape of advanced_ledger.py.

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence
import os
import math
import numpy as np
import pandas as pd
from ledger_io import load_df, save_df, path_for, debug_where
_KIND = "four_factors"

from cache_manager import stats_cache
from stats_getter import (
    NBA_TIMEOUT,
    get_league_game_log,
    _retry_nba,                        # your existing retry wrapper
    getLeagueDashTeamStats,            # your cached wrapper around LeagueDashTeamStats
    team_regular_season_range_by_id,   # (team_id, season) -> (date_from, date_to)
)

# TEAM-side Four Factors from BoxScoreFourFactorsV2
FF_METRICS: Sequence[str] = ("EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "OREB_PCT")


def _normalize_ff_cols(df: pd.DataFrame, cols=("OREB_PCT",)) -> pd.DataFrame:
    """
    Ensure four-factors percent columns are fractions in [0,1].
    If values look like 0–100, scale them; then clamp to [0,1].
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].notna().any():
            q90 = out[c].quantile(0.90)
            if pd.notna(q90) and q90 > 1.5:  # likely 0–100
                out[c] = out[c] / 100.0
        out[c] = out[c].clip(0.0, 1.0)
    return out


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


def get_ledger_rows_for_game(season: str, game_id: str):
    """Return ledger rows for this GAME_ID (2 rows if present), else empty DataFrame."""
    import pandas as pd
    df = _load_ledger(season)
    if df is None or df.empty:
        return pd.DataFrame()
    gid = str(game_id).zfill(10)
    out = df[df["GAME_ID"].astype(str) == gid].copy()
    return out


def _to_day(x):
    dt = pd.to_datetime(x, errors="coerce")
    if isinstance(dt, pd.Series):
        return dt.dt.normalize()
    if isinstance(dt, pd.DatetimeIndex):
        return dt.normalize()
    # scalar Timestamp/NaT
    return pd.NaT if pd.isna(dt) else dt.normalize()


# -------------------- in-process memo --------------------

_LEDGER_MEMO: dict[str, pd.DataFrame] = {}

def _load_ledger(season: str) -> pd.DataFrame | None:
    return load_df(_KIND, season)


def _save_ledger(season: str, df: pd.DataFrame) -> None:
    save_df(_KIND, season, df)

# -------------------- fetchers --------------------

def _team_second_half_window_by_id(team_id: int, season: str) -> tuple[str | None, str | None]:
    """
    Second-half regular-season window for a team in `season`, based on LeagueGameLog.
    """
    import pandas as pd
    log = get_league_game_log(season)
    if log is None or log.empty or "TEAM_ID" not in log.columns or "GAME_DATE" not in log.columns:
        return (None, None)

    df = log.copy()
    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"].str.contains("Regular", case=False, na=False)]
    df = df[df["TEAM_ID"].astype(int) == int(team_id)].copy()
    if df.empty:
        return (None, None)

    df["GAME_DATE"] = _to_day(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)
    split_idx = n // 2
    start = df.iloc[split_idx]["GAME_DATE"]; end = df.iloc[-1]["GAME_DATE"]
    return (start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y"))

@lru_cache(maxsize=8192)
def _prev_season_ff_avg_window(team_id: int, prev_season: str, metric: str,
                               date_from: str | None, date_to: str | None) -> float:
    """
    Previous-season average of a four-factors metric over a DATE WINDOW,
    computed from the ledger **only if** the ledger has that team’s full RS.
    Returns NaN if not available so the caller can fall back to one API call.
    """
    if not _ledger_has_full_prev_team_season(team_id, prev_season):
        return math.nan
    if not date_from or not date_to:
        return math.nan

    df = _load_ledger(prev_season)
    if df is None or df.empty or metric not in df.columns:
        return math.nan

    s = df[df["TEAM_ID"].astype(int) == int(team_id)].copy()
    s["GAME_DATE"] = _to_day(s["GAME_DATE"])
    s = s[(s["GAME_DATE"] >= _to_day(date_from)) & (s["GAME_DATE"] <= _to_day(date_to))]
    vals = pd.to_numeric(s[metric], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else math.nan



def _select_team_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    # Prefer table with TEAM_ID and no PLAYER_ID (two rows expected)
    for f in frames:
        if isinstance(f, pd.DataFrame) and "TEAM_ID" in f.columns and "PLAYER_ID" not in f.columns:
            return f.copy()
    # Fallback: aggregate player table if needed
    for f in frames:
        if "TEAM_ID" in f.columns and "PLAYER_ID" in f.columns:
            g = f.copy()
            keep = ["TEAM_ID", "GAME_ID", "TEAM_ABBREVIATION", *[c for c in FF_METRICS if c in g.columns]]
            g = g.loc[:, [c for c in keep if c in g.columns]]
            agg = {c: "mean" for c in FF_METRICS if c in g.columns}
            agg["GAME_ID"] = "first"
            agg["TEAM_ABBREVIATION"] = "first"
            return g.groupby("TEAM_ID", dropna=False).agg(agg).reset_index()
    return pd.DataFrame()

def get_fourfactors_team_game_rows(game_id: str, timeout: float = 10.0) -> pd.DataFrame:
    gid = str(game_id).zfill(10)

    def _fetch_once(t: float) -> pd.DataFrame:
        from nba_api.stats.endpoints import BoxScoreFourFactorsV2
        resp = BoxScoreFourFactorsV2(game_id=gid, timeout=t)
        team = _select_team_table(resp.get_data_frames())
        if team is None or team.empty:
            return pd.DataFrame()
        keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", *[c for c in FF_METRICS if c in team.columns]]
        out = team.loc[:, [c for c in keep if c in team.columns]].copy()
        out["GAME_ID"] = gid
        for col in FF_METRICS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    # fetch only on cache miss; accept **kwargs so game_id can be forwarded
    def _fetch_cached(**_):
        return _retry_nba(_fetch_once, endpoint="BoxScoreFourFactorsV2", timeout=timeout)

    df = stats_cache.get_or_fetch(
        "BoxScoreFourFactorsV2_Team",
        _fetch_cached,
        game_id=gid,
        # don't cache empties
        skip_if=lambda x: not isinstance(x, pd.DataFrame) or x.empty,
    )
    df = _normalize_ff_cols(df, cols=("OREB_PCT",))
    return df


# -------------------- recompute priors --------------------

def _recompute_team(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    s = df[df["TEAM_ID"] == int(team_id)].sort_values(["GAME_DATE", "GAME_ID"]).copy()
    for col in FF_METRICS:
        vals = pd.to_numeric(s[col], errors="coerce")
        s[f"{col}_prior"] = vals.groupby(s["TEAM_ID"]).apply(
            lambda x: x.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True)
    return s

def append_FourFactors_game(season: str, game_id: str) -> None:
    gl = get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gdate = gl.loc[gl["GAME_ID"].astype(str) == str(game_id), "GAME_DATE"]
    if gdate.empty:
        raise ValueError(f"GAME_ID {game_id} not found in LeagueGameLog for {season}")
    gdate = gdate.iloc[0]

    team_rows = get_fourfactors_team_game_rows(str(game_id))
    if team_rows is None or team_rows.empty:
        return

    team_rows = team_rows.copy()
    team_rows["SEASON"] = season
    team_rows["GAME_DATE"] = pd.to_datetime(gdate)

    # --- NEW: fallback for OREB_PCT using box-score totals ---
    need_oreb_fix = ("OREB_PCT" not in team_rows.columns) or team_rows["OREB_PCT"].isna().any()
    if need_oreb_fix:
        gl_pair = gl.loc[gl["GAME_ID"].astype(str) == str(game_id)]
        if len(gl_pair) == 2:
            team_rows = _fill_oreb_pct_from_league_log(team_rows, gl_pair)
    
    led = _load_ledger(season)
    team_rows = _normalize_ff_cols(team_rows, cols=("OREB_PCT",))
    led = pd.concat([led, team_rows], ignore_index=True) if led is not None else team_rows
    led.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)
    led.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], inplace=True)

    _save_ledger(season, led)




# -------------------- previous-season helpers --------------------

def _ledger_has_full_prev_team_season(team_id: int, prev_season: str) -> bool:
    df = _load_ledger(prev_season)
    if df is None or df.empty:
        return False
    log = get_league_game_log(prev_season).copy()
    if log.empty:
        return False
    d0, d1 = team_regular_season_range_by_id(team_id, prev_season)
    log["GAME_DATE"] = _to_day(log["GAME_DATE"])
    mask = (log["TEAM_ID"].astype(int) == int(team_id)) & (
        (log["GAME_DATE"] >= _to_day(d0)) & (log["GAME_DATE"] <= _to_day(d1))
    )
    expected = set(log.loc[mask, "GAME_ID"].astype(str))
    have = set(df.loc[df["TEAM_ID"].astype(int) == int(team_id), "GAME_ID"].astype(str))
    return expected.issubset(have)

@lru_cache(maxsize=8192)
def _prev_season_ff_avg(team_id: int, prev_season: str, metric: str) -> float:
    """
    Previous-season average of a four-factors metric, computed from the ledger
    **only if** the ledger contains the team’s full regular season.
    Otherwise returns NaN so the caller can fall back to one API call.
    """
    if not _ledger_has_full_prev_team_season(team_id, prev_season):
        return math.nan

    df = _load_ledger(prev_season)
    if df is None or df.empty or metric not in df.columns:
        return math.nan

    d0, d1 = team_regular_season_range_by_id(team_id, prev_season)
    s = df[df["TEAM_ID"].astype(int) == int(team_id)].copy()
    s["GAME_DATE"] = _to_day(s["GAME_DATE"])
    s = s[(s["GAME_DATE"] >= _to_day(d0)) & (s["GAME_DATE"] <= _to_day(d1))]
    vals = pd.to_numeric(s[metric], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else math.nan

@lru_cache(maxsize=1024)
def _prev_season_ff_pg(team_id: int, season: str, metric: str) -> float:
    """
    Previous-season per-game metric via a single LeagueDashTeamStats call
    (measure_type='Four Factors', per_mode='PerGame').
    """
    # We need a TEAM display name to pass to your getLeagueDashTeamStats; use the game log.
    log = get_league_game_log(season)
    trow = log.loc[log["TEAM_ID"].astype(int) == int(team_id)]
    team_name = trow["TEAM_NAME"].iloc[0] if not trow.empty and "TEAM_NAME" in trow.columns else ""
    if not team_name:
        return float("nan")

    df = getLeagueDashTeamStats(team_name, season, measure_type="Four Factors", per_mode="PerGame")
    if df is None or df.empty or metric not in df.columns:
        return float("nan")

    # Prefer the row with the largest GP if multiple
    sel = df.copy()
    if "TEAM_NAME" in sel.columns:
        sel = sel[sel["TEAM_NAME"].notna()]
    if "GP" in sel.columns and len(sel) > 1:
        sel = sel.sort_values("GP", ascending=False)
    return float(pd.to_numeric(sel.iloc[0][metric], errors="coerce"))

def _fill_oreb_pct_from_league_log(team_rows: pd.DataFrame, gl_pair: pd.DataFrame) -> pd.DataFrame:
    """
    If OREB_PCT is missing/NaN in the 2-row TEAM frame for a GAME_ID,
    compute it from the LeagueGameLog box score as a FRACTION:
        OREB_PCT = ORB / (ORB + opp_DREB)
    Only fills missing entries; does not overwrite valid values.
    """
    import numpy as np
    out = team_rows.copy()

    if "OREB_PCT" not in out.columns:
        out["OREB_PCT"] = np.nan

    keep = ["TEAM_ID", "OREB", "DREB"]
    if not all(c in gl_pair.columns for c in keep) or len(gl_pair) != 2 or len(out) != 2:
        return out

    g = gl_pair.loc[:, keep].copy()
    for c in keep:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(subset=["TEAM_ID"]).set_index("TEAM_ID")

    for i, row in out.iterrows():
        tid = row.get("TEAM_ID")
        if tid not in g.index:
            continue
        opp_ids = [x for x in g.index if x != tid]
        if not opp_ids:
            continue
        opp = opp_ids[0]

        if pd.isna(out.at[i, "OREB_PCT"]):
            t_orb = g.at[tid, "OREB"]
            o_drb = g.at[opp, "DREB"]
            den = (t_orb + o_drb)
            out.at[i, "OREB_PCT"] = (float(t_orb) / float(den)) if den else np.nan

    return _normalize_ff_cols(out, cols=("OREB_PCT",))



# -------------------- prior getters (per-team at date) --------------------

def get_prior_metric(season: str, team_id: int, game_date, metric: str) -> float:
    """
    Running average of `metric` for this team using ONLY rows strictly before `game_date`
    from the Four Factors ledger. If no prior rows exist (first game), fall back to the
    previous-season LeagueDashTeamStats Four Factors per-game value.
    """
    import pandas as pd
    import numpy as np

    day = _to_day(game_date)
    df = _load_ledger(season)

    # Fast exit if ledger empty
    if df is None or df.empty:
        return float(_prev_season_ff_pg(team_id, season, metric))

    # Normalize types once; _load_ledger already normalizes GAME_DATE
    tid = pd.to_numeric(df.get("TEAM_ID"), errors="coerce")
    dcol = "GAME_DATE" if "GAME_DATE" in df.columns else None
    if dcol is None:
        return float(_prev_season_ff_pg(team_id, season, metric))

    # Rows for this team strictly before the game date
    pre = df[(tid == int(team_id)) & (df[dcol] < day)].copy()
    if pre.empty:
        # First game (or data not loaded yet) → previous season
        return float(_prev_season_ff_pg(team_id, season, metric))

    col = str(metric)
    if col not in pre.columns:
        # Ledger doesn’t carry this metric column
        return float(_prev_season_ff_pg(team_id, season, metric))

    vals = pd.to_numeric(pre[col], errors="coerce").dropna()
    if len(vals) == 0:
        return float(_prev_season_ff_pg(team_id, season, metric))

    return float(vals.mean())


# Convenience wrappers
def get_prior_efg_pct(season: str, team_id: int, game_date) -> float:    return get_prior_metric(season, team_id, game_date, "EFG_PCT")
def get_prior_fta_rate(season: str, team_id: int, game_date) -> float:   return get_prior_metric(season, team_id, game_date, "FTA_RATE")
def get_prior_tm_tov_pct(season: str, team_id: int, game_date) -> float: return get_prior_metric(season, team_id, game_date, "TM_TOV_PCT")
def get_prior_oreb_pct(season: str, team_id: int, game_date) -> float:   return get_prior_metric(season, team_id, game_date, "OREB_PCT")

# -------------------- build helper --------------------


# helpers_four_factors.py (or tuck these where you keep last-season helpers)

import math
import pandas as pd


from stats_getter import (
    season_for_date_smart,
    get_team_id,                        # whatever helper you use to map name->TEAM_ID
    getLeagueDashTeamStats,
    team_regular_season_range_by_id,
)

# remove: from features_loader_copy import prev_season

def _prev_season(season: str) -> str:
    y = int(season[:4])
    return f"{y-1}-{str(y-2000).zfill(2)}"

def get_last_season_ff_metric(team_name: str, date_str: str, metric: str) -> float:
    """
    Previous-season SECOND-HALF ONLY average of a Four Factors metric for `team_name`.
    Try ledger (bounded to 2nd-half window) → single perGame endpoint fallback (bounded).
    """
    current_season = season_for_date_smart(date_str)
    prev_season = _prev_season(current_season)

    tid = get_team_id(team_name)
    d_from, d_to = _team_second_half_window_by_id(int(tid), prev_season) if tid is not None else (None, None)


    # 1) Ledger (if full RS present)
    if tid is not None:
        v = _prev_season_ff_avg_window(int(tid), prev_season, metric, d_from, d_to)
        if not (isinstance(v, float) and math.isnan(v)):
            return float(v)

    # 2) One call fallback (bounded to the same window)
    df = getLeagueDashTeamStats(
        team_name, prev_season,
        date_from=d_from, date_to=d_to,
        measure_type="Four Factors", per_mode="PerGame",
    )

    if df is None or df.empty or metric not in df.columns:
        return math.nan

    sel = df.copy()
    if "TEAM_NAME" in sel.columns:
        sel = sel[sel["TEAM_NAME"].notna()]
    if "GP" in sel.columns and len(sel) > 1:
        sel = sel.sort_values("GP", ascending=False)
    return float(pd.to_numeric(sel.iloc[0][metric], errors="coerce"))


# Concrete wrappers matching your original function names
def get_last_season_EFG_PCT(team_name: str, date_str: str) -> float:
    return get_last_season_ff_metric(team_name, date_str, "EFG_PCT")

def get_last_season_TMV_TOV_PCT(team_name: str, date_str: str) -> float:
    # Keep your original name (with 'TMV') but route to the correct metric.
    return get_last_season_ff_metric(team_name, date_str, "TM_TOV_PCT")



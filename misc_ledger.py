# misc_ledger.py
# Ledger/parquet for BoxScoreMiscV2 (TEAM table) with running "prior" averages.

from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Optional, Sequence

import math
import numpy as np
import pandas as pd

from cache_manager import stats_cache
import stats_getter
from stats_getter import (
    get_league_game_log,
    _retry_nba,
)

from ledger_io import load_df, save_df, path_for, debug_where
_KIND = "misc"


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

# -------------------- config & helpers --------------------

_LEDGER_TMP_ROOT = os.environ.get("NBA_LEDGER_TMP_ROOT", "ledgers_tmp")
_WORKER = os.environ.get("NBA_WORKER", "").strip().upper()
if _WORKER:
    _LEDGER_TMP_ROOT = f"{_LEDGER_TMP_ROOT}/process_{_WORKER}"
    
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

MISC_METRICS: Sequence[str] = (
    "PTS_OFF_TOV",      # points off turnovers (team)
    "PTS_FB",           # fast break points
    "PTS_2ND_CHANCE",   # 2nd chance points

)

def _path(season: str) -> Path:
    p = Path(_LEDGER_TMP_ROOT) / f"misc_{season}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _normalize_day(x):
    v = pd.to_datetime(x, errors="coerce")
    if isinstance(v, pd.Series):        return v.dt.normalize()
    if isinstance(v, pd.DatetimeIndex): return v.normalize()
    return pd.Timestamp(v).normalize()

def _prev_label(season: str) -> Optional[str]:
    try:
        y0 = int(season.split("-")[0])
        return f"{y0-1}-{str(y0)[2:]}"
    except Exception:
        return None

# Resolve a usable team name for LeagueDashTeamStats fallback
@lru_cache(maxsize=512)
def _team_name_from_id(team_id: int, season: str) -> Optional[str]:
    try:
        for s in (season, _prev_label(season)):
            if not s:
                continue
            log = get_league_game_log(s)
            if "TEAM_ID" not in log.columns:
                continue
            cols = [c for c in ("TEAM_NAME", "TEAM_ABBREVIATION") if c in log.columns]
            if not cols:
                continue
            df = log[log["TEAM_ID"].astype(int) == int(team_id)]
            if not df.empty:
                return str(df[cols[0]].iloc[0])
    except Exception:
        pass
    return None

# -------------------- load/save ledger --------------------

_MEMO: dict[str, pd.DataFrame] = {}

def _load_ledger(season: str) -> pd.DataFrame | None:
    return load_df(_KIND, season)

def _save_ledger(season: str, df: pd.DataFrame) -> None:
    save_df(_KIND, season, df)

# -------------------- game fetch (TEAM misc table) --------------------

def _fetch_misc_team_table(game_id: str, timeout: float = 10.0) -> pd.DataFrame:
    """
    Return the TEAM table (2 rows) from BoxScoreMiscV2 for this game_id,
    keeping only the misc metrics + ids. Cached via stats_cache.
    """
    game_id = str(game_id).zfill(10)

    def _process_frames(frames) -> pd.DataFrame:
        team_df = None
        for f in frames:
            if (
                isinstance(f, pd.DataFrame)
                and "TEAM_ID" in f.columns
                and "PLAYER_ID" not in f.columns
                and f.shape[0] == 2
            ):
                team_df = f
                break
        if team_df is None:
            return pd.DataFrame()  # don't cache errors

        keep = ["GAME_ID","TEAM_ID","TEAM_ABBREVIATION", *[c for c in MISC_METRICS if c in team_df.columns]]
        out = team_df.loc[:, [c for c in keep if c in team_df.columns]].copy()
        if "GAME_ID" not in out.columns:
            out["GAME_ID"] = game_id
        for col in MISC_METRICS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    # Fetcher for cache MISS; must accept **kwargs since get_or_fetch forwards them.
    def _fetch_cached(**_):
        from nba_api.stats.endpoints import BoxScoreMiscV2
        def _fetch_once(t: float):
            resp = BoxScoreMiscV2(game_id=game_id, timeout=t)
            return _process_frames(resp.get_data_frames())
        try:
            return _retry_nba(lambda t: _fetch_once(t), endpoint="BoxScoreMiscV2", timeout=timeout)
        except NameError:
            return _fetch_once(timeout)

    return stats_cache.get_or_fetch(
        "BoxScoreMiscV2_Team",
        _fetch_cached,
        game_id=game_id,
        # Optional: avoid caching empties
        # skip_if=lambda x: not isinstance(x, pd.DataFrame) or x.empty,
    )



# -------------------- recompute "prior" (shifted expanding mean) --------------------

def _recompute_team(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    s = df[df["TEAM_ID"] == int(team_id)].sort_values(["GAME_DATE","GAME_ID"]).copy()
    for col in MISC_METRICS:
        vals = pd.to_numeric(s[col], errors="coerce")
        s[f"{col}_prior"] = vals.groupby(s["TEAM_ID"]).apply(
            lambda x: x.shift(1).expanding().mean()
        ).reset_index(level=0, drop=True)
    return s

def append_misc_game(season: str, game_id: str) -> None:
    """
    Append the TEAM BoxScoreMiscV2 rows for this GAME_ID into the misc ledger
    and persist to parquet. Mirrors advanced_ledger.append_adv_game.
    """
    gl = get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gdate = gl.loc[gl["GAME_ID"].astype(str) == str(game_id), "GAME_DATE"]
    if gdate.empty:
        raise ValueError(f"GAME_ID {game_id} not found in LeagueGameLog for {season}")
    gdate = gdate.iloc[0]

    team_rows = _fetch_misc_team_table(str(game_id))
    if team_rows is None or team_rows.empty:
        # still write placeholder rows so priors line up by date if you want; else just return
        return

    team_rows = team_rows.copy()
    team_rows["SEASON"] = season
    team_rows["GAME_DATE"] = pd.to_datetime(gdate)

    led = _load_ledger(season)
    led = pd.concat([led, team_rows], ignore_index=True) if led is not None else team_rows
    led.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)
    led.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], inplace=True)
    _save_ledger(season, led)




# -------------------- previous-season fallback --------------------

@lru_cache(maxsize=1024)
def _prev_season_misc_pg(team_id: int, season: str, metric: str) -> float:
    """Previous-season per-game value from LeagueDashTeamStats(Misc, PerGame)."""
    team_name = _team_name_from_id(team_id, season)
    if not team_name:
        return float("nan")

    df = stats_getter.getLeagueDashTeamStats(
        team_name, season, measure_type="Misc", per_mode="PerGame"
    )
    if df is None or df.empty or metric not in df.columns:
        return float("nan")

    sel = df.copy()
    if "TEAM_NAME" in sel.columns:
        sel = sel[sel["TEAM_NAME"].notna()]
        m1 = sel["TEAM_NAME"] == team_name if "TEAM_NAME" in sel.columns else None
        if m1 is not None and m1.any():
            sel = sel[m1]
    if "GP" in sel.columns and len(sel) > 1:
        sel = sel.sort_values("GP", ascending=False)

    val = pd.to_numeric(sel.iloc[0][metric], errors="coerce")
    return float(val) if pd.notna(val) else float("nan")

# -------------------- prior getters --------------------

def get_prior_misc(season: str, team_id: int, game_date, metric: str) -> float:
    """
    Running average of `metric` for this team using ONLY rows strictly before `game_date`
    from the Misc ledger. If no prior rows exist (first game), fall back to the
    previous-season LeagueDashTeamStats Misc per-game value.
    """
    import pandas as pd

    day = _normalize_day(game_date)
    df = _load_ledger(season)

    # Fast exit if ledger empty
    if df is None or df.empty:
        return float(_prev_season_misc_pg(team_id, season, metric))

    # Normalize types once; _load_ledger already normalizes GAME_DATE
    tid = pd.to_numeric(df.get("TEAM_ID"), errors="coerce")
    dcol = "GAME_DATE" if "GAME_DATE" in df.columns else None
    if dcol is None:
        return float(_prev_season_misc_pg(team_id, season, metric))

    # Rows for this team strictly before the game date
    pre = df[(tid == int(team_id)) & (df[dcol] < day)].copy()
    if pre.empty:
        # First game (or data not loaded yet) → previous season
        return float(_prev_season_misc_pg(team_id, season, metric))

    col = str(metric)
    if col not in pre.columns:
        # Ledger doesn’t carry this metric column
        return float(_prev_season_misc_pg(team_id, season, metric))

    vals = pd.to_numeric(pre[col], errors="coerce").dropna()
    if len(vals) == 0:
        return float(_prev_season_misc_pg(team_id, season, metric))

    return float(vals.mean())


# Friendly wrappers
def get_prior_pts_off_tov(season: str, team_id: int, game_date) -> float:
    return get_prior_misc(season, team_id, game_date, "PTS_OFF_TOV")
def get_prior_pts_fb(season: str, team_id: int, game_date) -> float:
    return get_prior_misc(season, team_id, game_date, "PTS_FB")
def get_prior_pts_2nd_chance(season: str, team_id: int, game_date) -> float:
    return get_prior_misc(season, team_id, game_date, "PTS_2ND_CHANCE")




# features_misc_ledger.py (or inline in your existing features module)

import math
from datetime import datetime
import pandas as pd

import stats_getter
import advanced_ledger as adv
import misc_ledger as misc



# ----- helpers that compute BOTH teams at once -----

def _pts_off_tov_rates_pair(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str):
    """Return (A_rate, B_rate) where rate = PTS_OFF_TOV_pg / poss_pg. math.nan on invalid."""
    import numpy as np, math
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan, math.nan

    a_pts = misc.get_prior_pts_off_tov(team_a_season, a_id, d)
    b_pts = misc.get_prior_pts_off_tov(team_b_season, b_id, d)
    a_poss_pg = adv.get_prior_poss(team_a_season, a_id, d)
    b_poss_pg = adv.get_prior_poss(team_b_season, b_id, d)

    if any(map(lambda x: (x is None) or (isinstance(x, float) and (math.isnan(x) or x <= 0)),
               [a_pts, b_pts, a_poss_pg, b_poss_pg])):
        return math.nan, math.nan

    a_rate = float(a_pts) / float(a_poss_pg)
    b_rate = float(b_pts) / float(b_poss_pg)
    return a_rate, b_rate


def _pts_fb_pair(team_a_name: str, team_b_name: str,
                 team_a_season: str, team_b_season: str,
                 date_str: str):
    """Return (A_FB_pg, B_FB_pg). math.nan on invalid."""
    import math
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan, math.nan

    a_fb = misc.get_prior_pts_fb(team_a_season, a_id, d)
    b_fb = misc.get_prior_pts_fb(team_b_season, b_id, d)
    if any(map(lambda x: (x is None) or (isinstance(x, float) and math.isnan(x)), [a_fb, b_fb])):
        return math.nan, math.nan

    return float(a_fb), float(b_fb)

# ----- NEW: per-team features (split, not ratios) -----

def get_pts_off_tov_rate_home(team_a_name: str, team_b_name: str,
                              team_a_season: str, team_b_season: str,
                              date_str: str) -> float:
    a_rate, _ = _pts_off_tov_rates_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return a_rate

def get_pts_off_tov_rate_away(team_a_name: str, team_b_name: str,
                              team_a_season: str, team_b_season: str,
                              date_str: str) -> float:
    _, b_rate = _pts_off_tov_rates_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return b_rate


def get_pts_fb_pg_home(team_a_name: str, team_b_name: str,
                       team_a_season: str, team_b_season: str,
                       date_str: str) -> float:
    a_fb, _ = _pts_fb_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return a_fb

def get_pts_fb_pg_away(team_a_name: str, team_b_name: str,
                       team_a_season: str, team_b_season: str,
                       date_str: str) -> float:
    _, b_fb = _pts_fb_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return b_fb




def get_second_chance_pts_per_100(team_name: str,
                                  season: str,
                                  date_str: str) -> float:
    """
    Team's 2nd-chance points per 100 possessions using ledger priors.
    """
    d = datetime.strptime(date_str, "%m/%d/%Y")
    tid = stats_getter.get_team_id(team_name)
    if tid is None:
        return math.nan

    pts = misc.get_prior_pts_2nd_chance(season, tid, d)
    poss_pg = adv.get_prior_poss(season, tid, d)
    if any(map(lambda x: x is None or (isinstance(x, float) and (math.isnan(x) or x <= 0)), [pts, poss_pg])):
        return math.nan
    return float(pts) / float(poss_pg) * 100.0



if __name__ == "__main__":
    print(f"[{__file__}] writing ledgers to: {_LEDGER_TMP_ROOT}")
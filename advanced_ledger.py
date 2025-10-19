# advanced_ledger.py
# Incremental “ledger” for BoxScoreAdvancedV2 (team table) with prior (to-date) averages.
# Mirrors the design and API shape of hustle_ledger.py.

from __future__ import annotations
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Optional, Sequence, Iterable

import pandas as pd
from cache_manager import stats_cache
from nba_api.stats.endpoints import boxscoreadvancedv2
from ledger_io import load_df, save_df, path_for, debug_where
_KIND = "advanced"

import os
from pathlib import Path
from ledger_io import ROOT


from stats_getter import (
    NBA_TIMEOUT,
    get_league_game_log,
    resolve_season_for_game_by_logs,   # same functions used in hustle_ledger
    lookup_game_id_by_teams_date,      # if not available, swap to get_game_id(home_team,...)
    _retry_nba,
    team_regular_season_range_by_id, 
)
import stats_getter

# at top of file if it's not already imported:
from stats_getter import get_league_game_log

# ...

def _normalize_pct_cols(df: pd.DataFrame, cols=("OREB_PCT","DREB_PCT","REB_PCT")) -> pd.DataFrame:
    """
    Ensure all pct columns are fractions in [0,1].
    If values look like 0–100, scale them down. Then clamp to [0,1].
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns: 
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

        # Detect 0–100 scale (use 90th percentile to be robust to a few NaNs)
        q90 = out[c].quantile(0.90) if out[c].notna().any() else float("nan")
        if pd.notna(q90) and q90 > 1.5:          # likely 0–100
            out[c] = out[c] / 100.0

        # Clamp to [0,1]
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    return out


def _fill_reb_pcts_for_game(df_adv: pd.DataFrame, season: str, game_id: str, league_log: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    If OREB_PCT/DREB_PCT/REB_PCT are NaN in the 2-row advanced team dataframe
    for this game, fill them using LeagueGameLog (OREB/DREB/REB) via the existing
    _fill_reb_pcts_from_league_log helper.

    Returns a new df_adv with any filled values (or the original df if nothing to fill).
    """
    needs_cols = [c for c in ("OREB_PCT", "DREB_PCT", "REB_PCT") if c in df_adv.columns]
    if not needs_cols:
        return df_adv
    if not df_adv[needs_cols].isna().any().any():
        return df_adv

    # must be exactly the two team rows for this game
    if "TEAM_ID" not in df_adv.columns or len(df_adv) != 2:
        return df_adv

    # cache or pull the whole season log once
    if league_log is None:
        league_log = get_league_game_log(season)

    # prepare the two-team pair for this game
    gid = str(game_id).zfill(10)
    gl_pair = league_log.loc[league_log["GAME_ID"].astype(str).str.zfill(10) == gid, ["TEAM_ID", "OREB", "DREB", "REB"]]
    if gl_pair.shape[0] != 2 or gl_pair.isna().any().any():
        return df_adv

    # use your existing backfill logic
    try:
        return _fill_reb_pcts_from_league_log(df_adv, gl_pair)
    except Exception:
        # never break the pipeline on fill; return original
        return df_adv



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


def _prev_label(season: str) -> str | None:
    try:
        y0 = int(season.split("-")[0])
        return f"{y0-1}-{str(y0)[2:]}"   # '2019-20' -> '2018-19'
    except Exception:
        return None

# -------------------- configuration --------------------

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

ADV_METRICS: Sequence[str] = (
    "PACE",
    "NET_RATING",
    "DREB_PCT",
    "OREB_PCT",
    "POSS",
    "TM_TOV_PCT",
    "EFG_PCT",
)

def _normalize_day(x):
    v = pd.to_datetime(x, errors="coerce")
    if isinstance(v, pd.Series):        return v.dt.normalize()
    if isinstance(v, pd.DatetimeIndex): return v.normalize()
    return pd.Timestamp(v).normalize()


def where():
    debug_where(_KIND)

from functools import lru_cache
import numpy as np
import pandas as pd
from stats_getter import getLeagueDashTeamStats, get_league_game_log



@lru_cache(maxsize=512)
def _team_name_from_id(team_id: int, season: str) -> str | None:
    """
    Resolve a display name (TEAM_NAME or TEAM_ABBREVIATION) that we can pass
    to getLeagueDashTeamStats(...). Tries this season, then prev season.
    """
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


@lru_cache(maxsize=1024)
def _prev_season_adv_pg(team_id: int, season: str, metric: str) -> float:
    """
    Previous-season Advanced *per-game* value for team/metric.
    If metric == 'POSS', return POSS / GP explicitly (not PACE).
    Returns float or NaN.
    """

    team_name = _team_name_from_id(team_id, season)
    if not team_name:
        return float("nan")

    # Pull previous-season advanced, per-game table for this team.

    df = getLeagueDashTeamStats(
        team_name, season, measure_type="Advanced", per_mode="PerGame"
    )
    if df is None or df.empty:
        return float("nan")

    # Prefer the regular-season aggregate row:
    #  - some responses include a short early window row with TEAM_NAME=None
    #  - if multiple rows remain, pick the one with largest GP
    sel = df.copy()
    if "TEAM_NAME" in sel.columns:
        sel = sel[sel["TEAM_NAME"].notna()]
        # (Optional) sanity filter for the exact team, if present
        m1 = (sel["TEAM_NAME"] == team_name) if "TEAM_NAME" in sel.columns else None
        if m1 is not None and m1.any():
            sel = sel[m1]
    if "GP" in sel.columns and len(sel) > 1:
        sel = sel.sort_values("GP", ascending=False)
    row = sel.iloc[0]

    if metric == "POSS":
        poss = pd.to_numeric(row.get("POSS", np.nan), errors="coerce")
        gp   = pd.to_numeric(row.get("GP",   np.nan), errors="coerce")
        if pd.notna(poss) and pd.notna(gp) and gp != 0:
            return float(poss / gp)
        # Fallback (very rare): approximate using PACE if POSS/GP unavailable
        pace = pd.to_numeric(row.get("PACE", np.nan), errors="coerce")
        return float(pace) if pd.notna(pace) else float("nan")

    val = pd.to_numeric(row.get(metric, np.nan), errors="coerce")
    return float(val) if pd.notna(val) else float("nan")



# -------------------- in-process memo --------------------

_LEDGER_MEMO: dict[str, pd.DataFrame] = {}

def _load_ledger(season: str) -> pd.DataFrame | None:
    return load_df(_KIND, season)

def _save_ledger(season: str, df: pd.DataFrame) -> None:
    save_df(_KIND, season, df)
# -------------------- payload fetcher --------------------

def _skip_empty(df: Optional[pd.DataFrame]) -> bool:
    if df is None: return True
    if not isinstance(df, pd.DataFrame): return True
    if df.empty: return True
    if "TEAM_ID" not in df.columns: return True
    return False

def _select_team_table(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # Prefer a frame with TEAM_ID and no PLAYER_ID
    candidates = []
    for f in dfs:
        cols = set(map(str, f.columns))
        if "TEAM_ID" in cols and "PLAYER_ID" not in cols:
            candidates.append(f)
    if len(candidates) == 1:
        return candidates[0].copy()
    if len(candidates) > 1:
        # pick the one that already has some of our metrics
        for f in candidates:
            if any(c in f.columns for c in ADV_METRICS):
                return f.copy()
        return candidates[0].copy()
    # Fallback: if only a player table exists, aggregate mean (minutes-weighted if possible)
    for f in dfs:
        cols = set(map(str, f.columns))
        if "TEAM_ID" in cols and "PLAYER_ID" in cols:
            df = f.copy()
            # to numeric for metrics
            for col in ADV_METRICS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "MIN" in df.columns:
                def _parse_min(v):
                    if pd.isna(v): return pd.NA
                    s = str(v)
                    if ":" in s:
                        mm, ss = s.split(":")
                        try: return float(mm) + float(ss)/60.0
                        except Exception: return pd.NA
                    try: return float(s)
                    except Exception: return pd.NA
                w = df["MIN"].map(_parse_min).fillna(0.0)
                def wmean(x, w_):
                    x = pd.to_numeric(x, errors="coerce")
                    denom = w_.sum()
                    return (x*w_).sum()/denom if denom else pd.NA
                agg = {col: (lambda x, _c=col: wmean(x, w.loc[x.index])) for col in ADV_METRICS if col in df.columns}
            else:
                agg = {col: "mean" for col in ADV_METRICS if col in df.columns}
            agg["TEAM_ABBREVIATION"] = "first"
            agg["GAME_ID"] = "first"
            return df.groupby("TEAM_ID", dropna=False).agg(agg).reset_index()
    return pd.DataFrame()

def get_advanced_team_game_rows(game_id: str, timeout: float = 10.0) -> pd.DataFrame:
    """
    Always return the 2-row TEAM advanced table for this GAME_ID, with numeric metrics.
    """
    import pandas as pd
    game_id = str(game_id).zfill(10)

    def _process_frames(frames) -> pd.DataFrame:
        # pick TEAM table: exactly 2 rows, has TEAM_ID, and no PLAYER_ID column
        team_df = None
        for df in frames:
            if not isinstance(df, pd.DataFrame):
                continue
            if df.shape[0] == 2 and "TEAM_ID" in df.columns and "PLAYER_ID" not in df.columns:
                team_df = df
                break
        if team_df is None:
            raise RuntimeError(f"TEAM table not found for GAME_ID {game_id}")

        keep = ["GAME_ID","TEAM_ID","TEAM_ABBREVIATION", *[c for c in ADV_METRICS if c in team_df.columns]]
        out = team_df.loc[:, [c for c in keep if c in team_df.columns]].copy()
        if "GAME_ID" not in out.columns:
            out["GAME_ID"] = game_id

        # numeric coercion
        for col in ADV_METRICS:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = _normalize_pct_cols(out)

        return out

    # single call with retry, then process frames
    def _fetch_once(t: float):
        from nba_api.stats.endpoints import BoxScoreAdvancedV2
        resp = BoxScoreAdvancedV2(game_id=game_id, timeout=t)
        return _process_frames(resp.get_data_frames())

    # your retry wrapper if you have one
    try:
        df = _retry_nba(lambda t: _fetch_once(t), endpoint="BoxScoreAdvancedV2", timeout=timeout)
    except NameError:
        df = _fetch_once(timeout)

    # optional cache wrapper: but make sure it returns the processed TEAM df
    try:
        from cache_manager import stats_cache
        df = stats_cache.get_or_fetch(
            "BoxScoreAdvancedV2_Team",
            lambda: df,   # already processed TEAM df
            game_id=game_id,
        )
    except Exception:
        pass

    if df.shape[0] != 2:
        raise RuntimeError(f"TEAM rows != 2 for GAME_ID {game_id}")
    return df



# (Empty returns are NOT cached per stats_cache rules)  # :contentReference[oaicite:4]{index=4}

# -------------------- recompute (priors) --------------------

def _recompute_team(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Recompute prior (to-date) averages ONLY for this team, sorted by date."""
    s = df[df["TEAM_ID"] == team_id].sort_values(["GAME_DATE","GAME_ID"]).copy()

    # For rate-like metrics we just use shifted expanding mean.
    for col in ADV_METRICS:
        vals = pd.to_numeric(s[col], errors="coerce")
        s[f"{col}_prior"] = vals.groupby(s["TEAM_ID"]).apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    return s

# -------------------- ensure (bulk up to date) --------------------

def ensure_advanced_for_matchup(season: str, cutoff_date):
    """
    Append all missing advanced rows up to cutoff_date and recompute priors
    only for teams we touched. Mirrors ensure_hustle_up_to.
    """
    ledger = _load_ledger(season)
    log = get_league_game_log(season).copy()
    log["GAME_DATE"] = _normalize_day(log["GAME_DATE"])
    cutoff = _normalize_day(cutoff_date)

    want = set(log.loc[log["GAME_DATE"] <= cutoff, "GAME_ID"].astype(str))
    have = set(ledger["GAME_ID"].astype(str)) if not ledger.empty else set()
    missing = sorted(want - have)
    if not missing:
        return

    new_rows: list[dict] = []
    touched: set[int] = set()

    for gid in missing:
        gdate = log.loc[log["GAME_ID"].astype(str) == str(gid), "GAME_DATE"].iloc[0]
        team_df = get_advanced_team_game_rows(str(gid))  # cached one hop

        # Ensure ADV columns exist (so later fills can target them)
        if team_df is not None and not team_df.empty:
            for m in ADV_METRICS:
                if m not in team_df.columns:
                    team_df[m] = pd.NA

            # --- POSS fallback (per-team) if missing/NaN ---
            if team_df["POSS"].isna().any():
                team_df["POSS"] = pd.to_numeric(team_df["POSS"], errors="coerce")
                for idx, r in team_df.iterrows():
                    if pd.isna(r["POSS"]):
                        try:
                            v = _fallback_poss_pg_via_advanced(int(r["TEAM_ID"]), season)
                        except Exception:
                            v = float("nan")
                        team_df.loc[idx, "POSS"] = v

            # --- Rebounding % fallback from game log (needs the two rows) ---
            gl_pair = log.loc[log["GAME_ID"].astype(str) == str(gid)].copy()
            if len(gl_pair) == 2 and all(c in gl_pair.columns for c in ["TEAM_ID", "OREB", "DREB", "REB"]):
                team_df = _fill_reb_pcts_from_league_log(team_df, gl_pair)
                team_df = _normalize_pct_cols(team_df)


        if team_df is None or team_df.empty:
            # Endpoint blipped: still emit 2 rows so priors align to date.
            # (We can only fill reb% if game-log present; other ADV fields remain NaN.)
            pair = log.loc[log["GAME_ID"].astype(str) == str(gid), ["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()
            for r in pair.itertuples(index=False):
                row = {
                    "SEASON": season,
                    "GAME_ID": str(gid).zfill(10),
                    "GAME_DATE": gdate,
                    "TEAM_ID": int(r.TEAM_ID),
                    "TEAM_ABBREVIATION": getattr(r, "TEAM_ABBREVIATION", ""),
                }
                row.update({m: pd.NA for m in ADV_METRICS})
                new_rows.append(row)
                touched.add(int(r.TEAM_ID))
        else:
            for r in team_df.itertuples(index=False):
                row = {
                    "SEASON": season,
                    "GAME_ID": str(getattr(r, "GAME_ID", gid)).zfill(10),
                    "GAME_DATE": gdate,
                    "TEAM_ID": int(r.TEAM_ID),
                    "TEAM_ABBREVIATION": getattr(r, "TEAM_ABBREVIATION", ""),
                }
                for m in ADV_METRICS:
                    row[m] = pd.to_numeric(getattr(r, m, pd.NA), errors="coerce")
                new_rows.append(row)
                touched.add(int(r.TEAM_ID))

    if not new_rows:
        return

    ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # Recompute priors only for touched teams
    out_chunks = []
    for tid in touched:
        out_chunks.append(_recompute_team(ledger, tid))
    if touched:
        keep = ~ledger["TEAM_ID"].isin(list(touched))
        ledger = pd.concat([ledger.loc[keep], *out_chunks], ignore_index=True)
        ledger = ledger.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    _save_ledger(season, ledger)
    # ensure subsequent prior calls see fresh rows
    try:
        _team_adv_slice.cache_clear()
    except Exception:
        pass

def ensure_possessions(season: str, cutoff_date: str | None = None, *, verbose: bool = True) -> None:
    """
    Backfill missing POSS in this season's advanced ledger using ONLY
    BoxScoreAdvancedV2 TEAM table. Prints progress per game_id.
    """
    import numpy as np
    import pandas as pd

    led = _load_ledger(season)
    if led is None or led.empty:
        if verbose: print(f"[POSS] season={season} ledger empty; nothing to do")
        return

    # normalize core ids
    led["GAME_ID"] = led["GAME_ID"].astype(str).str.zfill(10)
    led["TEAM_ID"] = pd.to_numeric(led["TEAM_ID"], errors="coerce")

    # ensure POSS column exists (numeric)
    if "POSS" not in led.columns:
        led["POSS"] = np.nan
    led["POSS"] = pd.to_numeric(led["POSS"], errors="coerce")

    # optional cutoff filter (only affects what we attempt to fill)
    work_mask = pd.Series(True, index=led.index)
    if cutoff_date is not None and "GAME_DATE" in led.columns:
        cd = pd.to_datetime(cutoff_date, errors="coerce").normalize()
        work_mask &= (pd.to_datetime(led["GAME_DATE"], errors="coerce").dt.normalize() <= cd)

    # which pairs need filling?
    need_pairs = led.loc[work_mask & led["POSS"].isna(), ["GAME_ID", "TEAM_ID"]].drop_duplicates()
    if need_pairs.empty:
        if verbose: print(f"[POSS] season={season} nothing missing; done")
        return

    total_games_touched = 0
    total_rows_filled = 0
    touched_teams: set[int] = set()

    for gid in need_pairs["GAME_ID"].unique():
        # rows still missing for this game
        m = (led["GAME_ID"] == gid) & led["POSS"].isna()
        if not m.any():
            continue

        try:
            team_df = get_advanced_team_game_rows(gid)  # 2-row TEAM table, has POSS
        except Exception as e:
            if verbose: print(f"[POSS] game_id={gid} skipped (endpoint error: {e})")
            continue

        if "POSS" not in team_df.columns:
            if verbose: print(f"[POSS] game_id={gid} returned without POSS; skipped")
            continue

        # map by TEAM_ID → POSS
        poss_map = (
            team_df[["TEAM_ID", "POSS"]]
            .dropna(subset=["TEAM_ID"])
            .astype({"TEAM_ID": "int64"})
            .set_index("TEAM_ID")["POSS"]
        )

        before_missing = m.sum()
        for i in led.index[m]:
            tid = int(led.at[i, "TEAM_ID"])
            if tid in poss_map.index and pd.notna(poss_map.loc[tid]):
                led.at[i, "POSS"] = float(poss_map.loc[tid])
                touched_teams.add(tid)

        after_missing = ((led["GAME_ID"] == gid) & led["POSS"].isna()).sum()
        filled_here = before_missing - after_missing
        if filled_here > 0:
            total_games_touched += 1
            total_rows_filled += filled_here
            if verbose:
                print(f"[POSS] filled game_id={gid} rows={filled_here}")
        else:
            if verbose:
                print(f"[POSS] game_id={gid} no rows filled (already complete or API had NaNs)")

    if total_rows_filled == 0:
        if verbose: print(f"[POSS] season={season} completed (no changes written)")
        return

    # recompute priors only for teams we touched
    chunks = []
    for tid in touched_teams:
        chunks.append(_recompute_team(led, int(tid)))
    keep = ~led["TEAM_ID"].isin(list(touched_teams))
    led = pd.concat([led.loc[keep], *chunks], ignore_index=True)
    led = led.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    _save_ledger(season, led)
    try:
        _team_adv_slice.cache_clear()
    except Exception:
        pass

    if verbose:
        print(f"[POSS] season={season} done | games_touched={total_games_touched} rows_filled={total_rows_filled}")


# -------------------- ensure (single game / matchup) --------------------

def ensure_advanced_for_game(season: str, game_id: str | int):
    """
    Append JUST this game's advanced rows (if missing) and recompute priors for
    the two teams in that game. Mirrors ensure_hustle_for_game.
    """
    ledger = _load_ledger(season)
    gid = str(game_id).zfill(10)
    if not ledger.empty and (ledger["GAME_ID"].astype(str) == gid).any():
        return

    log = get_league_game_log(season).copy()
    grow = log.loc[log["GAME_ID"].astype(str) == gid]
    if grow.empty:
        raise ValueError(f"GAME_ID {gid} not found in LeagueGameLog for {season}")
    gdate = _normalize_day(grow["GAME_DATE"].iloc[0])

    team_df = get_advanced_team_game_rows(gid)

    new_rows: list[dict] = []
    touched: set[int] = set()

    if team_df is not None and not team_df.empty:
        # Ensure ADV columns exist
        for m in ADV_METRICS:
            if m not in team_df.columns:
                team_df[m] = pd.NA

        # POSS fallback
        if team_df["POSS"].isna().any():
            team_df["POSS"] = pd.to_numeric(team_df["POSS"], errors="coerce")
            for idx, r in team_df.iterrows():
                if pd.isna(r["POSS"]):
                    try:
                        v = _fallback_poss_pg_via_advanced(int(r["TEAM_ID"]), season)
                    except Exception:
                        v = float("nan")
                    team_df.loc[idx, "POSS"] = v

        # Rebounding % fallback using the two game-log rows
        gl_pair = log.loc[log["GAME_ID"].astype(str) == gid]
        if len(gl_pair) == 2 and all(c in gl_pair.columns for c in ["TEAM_ID", "OREB", "DREB", "REB"]):
            team_df = _fill_reb_pcts_from_league_log(team_df, gl_pair)
            team_rows = _normalize_pct_cols(team_rows)


        for r in team_df.itertuples(index=False):
            base = {
                "SEASON": season,
                "GAME_ID": gid,
                "GAME_DATE": gdate,
                "TEAM_ID": int(r.TEAM_ID),
                "TEAM_ABBREVIATION": getattr(r, "TEAM_ABBREVIATION", ""),
            }
            for m in ADV_METRICS:
                base[m] = pd.to_numeric(getattr(r, m, pd.NA), errors="coerce")
            new_rows.append(base)
            touched.add(int(r.TEAM_ID))
    else:
        # Endpoint blip: emit two aligned rows; try to at least fill reb% from game log.
        for r in grow[["TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates().itertuples(index=False):
            base = {
                "SEASON": season,
                "GAME_ID": gid,
                "GAME_DATE": gdate,
                "TEAM_ID": int(r.TEAM_ID),
                "TEAM_ABBREVIATION": getattr(r, "TEAM_ABBREVIATION", ""),
            }
            base.update({m: pd.NA for m in ADV_METRICS})
            new_rows.append(base)
            touched.add(int(r.TEAM_ID))

        # If we have both rows, compute reb% on those new rows
        gl_pair = log.loc[log["GAME_ID"].astype(str) == gid]
        if len(gl_pair) == 2 and all(c in gl_pair.columns for c in ["TEAM_ID", "OREB", "DREB", "REB"]):
            tmp = pd.DataFrame(new_rows)
            tmp = _fill_reb_pcts_from_league_log(tmp, gl_pair)
            new_rows = tmp.to_dict(orient="records")

    if not new_rows:
        return

    ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # Recompute priors for just these two teams
    chunks = []
    for tid in touched:
        chunks.append(_recompute_team(ledger, tid))
    keep = ~ledger["TEAM_ID"].isin(list(touched))
    ledger = pd.concat([ledger.loc[keep], *chunks], ignore_index=True)
    ledger = ledger.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    _save_ledger(season, ledger)
    # Bust the slice cache so prior getters read fresh rows
    try:
        _team_adv_slice.cache_clear()
    except Exception:
        pass




# -------------------- prior getters (per-team at date) --------------------

# advanced_ledger.py  (replace the whole function)

def get_prior_metric(season: str, team_id: int, game_date, metric: str) -> float:
    """
    Running average of `metric` for this team using ONLY rows strictly before `game_date`
    from the advanced ledger. If no prior rows exist (first game), fall back to the
    previous-season LeagueDashTeamStats Advanced per-game value.

    Returns float (NaN if both ledger and prev-season lookup fail).
    """
    import pandas as pd
    import numpy as np
    day = _normalize_day(game_date)
    df = _load_ledger(season)

    # Fast exit if ledger empty
    if df is None or df.empty:
        return float(_prev_season_adv_pg(team_id, season, metric))


    # Normalize types once; _load_ledger already normalizes GAME_DATE
    tid = pd.to_numeric(df.get("TEAM_ID"), errors="coerce")
    dcol = "GAME_DATE" if "GAME_DATE" in df.columns else None
    if dcol is None:
        return float(_prev_season_adv_pg(team_id, season, metric))

    # Rows for this team strictly before the game date
    pre = df[(tid == int(team_id)) & (df[dcol] < day)].copy()
    if pre.empty:
        # First game (or data not loaded yet) → previous season
        return float(_prev_season_adv_pg(team_id, season, metric))

    col = str(metric)
    if col not in pre.columns:
        # If the ledger doesn’t carry this metric column, nothing to average
        return float(_prev_season_adv_pg(team_id, season, metric))

    vals = pd.to_numeric(pre[col], errors="coerce").dropna()
    if len(vals) == 0:
        return float(_prev_season_adv_pg(team_id, season, metric))

    return float(vals.mean())



# Convenience wrappers if you want parity with hustle_ledger’s accessors:
def get_prior_pace(season: str, team_id: int, game_date) -> float:        return get_prior_metric(season, team_id, game_date, "PACE")
def get_prior_net_rating(season: str, team_id: int, game_date) -> float:  return get_prior_metric(season, team_id, game_date, "NET_RATING")
def get_prior_dreb_pct(season: str, team_id: int, game_date) -> float:    return get_prior_metric(season, team_id, game_date, "DREB_PCT")
def get_prior_oreb_pct(season: str, team_id: int, game_date) -> float:    return get_prior_metric(season, team_id, game_date, "OREB_PCT")
def get_prior_poss(season: str, team_id: int, game_date) -> float:        return get_prior_metric(season, team_id, game_date, "POSS")
def get_prior_tm_tov_pct(season: str, team_id: int, game_date) -> float:  return get_prior_metric(season, team_id, game_date, "TM_TOV_PCT")
def get_prior_efg_pct(season: str, team_id: int, game_date) -> float:     return get_prior_metric(season, team_id, game_date, "EFG_PCT")

# --- add or rename this in advanced_ledger.py ---

# --- add or rename this in advanced_ledger.py ---


@lru_cache(maxsize=8192)
def _team_adv_slice(season: str, team_id: int) -> pd.DataFrame:
    df = _load_ledger(season)  # <- DO NOT call build_advanced_team_ledger() here
    if df is None or df.empty:
        return pd.DataFrame()
    s = df[df["TEAM_ID"] == int(team_id)].copy()
    if s.empty:
        return s
    s["GAME_DATE"] = pd.to_datetime(s["GAME_DATE"])
    s.sort_values(["GAME_DATE", "GAME_ID"], inplace=True)
    s.reset_index(drop=True, inplace=True)
    return s

def _team_prev_season_game_ids(team_id: int, prev_season: str) -> set[str]:
    """Expected GAME_IDs for this team's regular season in prev_season."""
    log = get_league_game_log(prev_season).copy()
    if log.empty:
        return set()
    d0, d1 = team_regular_season_range_by_id(team_id, prev_season)
    log["GAME_DATE"] = _to_day(log["GAME_DATE"])
    m = (log["TEAM_ID"].astype(int) == int(team_id)) & (
        (log["GAME_DATE"] >= _to_day(d0)) & (log["GAME_DATE"] <= _to_day(d1))
    )
    return set(log.loc[m, "GAME_ID"].astype(str))

def _fill_reb_pcts_from_league_log(team_rows, gl_pair):
    """
    Fill missing OREB_PCT / DREB_PCT / REB_PCT in a 2-row TEAM frame
    using league game log box-score totals.
    - team_rows: DataFrame with two rows (TEAM view for the game),
                 must include 'TEAM_ID' and possibly the *_PCT cols.
    - gl_pair:   DataFrame with exactly the two rows for this GAME_ID
                 from get_league_game_log(season), must include
                 ['TEAM_ID','OREB','DREB','REB'].
    Returns a new DataFrame with NaNs filled where possible.
    """
    import numpy as np
    import pandas as pd

    # Ensure numeric
    for c in ["TEAM_ID", "OREB", "DREB", "REB"]:
        if c in gl_pair.columns:
            gl_pair[c] = pd.to_numeric(gl_pair[c], errors="coerce")

    # Build quick lookups
    gl_pair = gl_pair[["TEAM_ID", "OREB", "DREB", "REB"]].dropna(subset=["TEAM_ID"])
    gl_pair = gl_pair.set_index("TEAM_ID")

    out = team_rows.copy()
    if "REB_PCT" not in out.columns:
        out["REB_PCT"] = np.nan  # create if missing

    def _pct(num, den):
        den = float(den)
        return (float(num) / den) if den and den == den else np.nan  # FRACTION (0–1), no *100


    for i, row in out.iterrows():
        tid = row.get("TEAM_ID")
        if tid not in gl_pair.index:
            continue
        opp_tid = [x for x in gl_pair.index if x != tid]
        if not opp_tid:
            continue
        opp_tid = opp_tid[0]

        t_orb = gl_pair.at[tid, "OREB"]
        t_drb = gl_pair.at[tid, "DREB"]
        t_trb = gl_pair.at[tid, "REB"]
        o_orb = gl_pair.at[opp_tid, "OREB"]
        o_drb = gl_pair.at[opp_tid, "DREB"]
        o_trb = gl_pair.at[opp_tid, "REB"]

        # Only fill if missing (NaN)
        if ("OREB_PCT" in out.columns) and pd.isna(out.at[i, "OREB_PCT"]):
            out.at[i, "OREB_PCT"] = _pct(t_orb, t_orb + o_drb)
        if ("DREB_PCT" in out.columns) and pd.isna(out.at[i, "DREB_PCT"]):
            out.at[i, "DREB_PCT"] = _pct(t_drb, t_drb + o_orb)
        if pd.isna(out.at[i, "REB_PCT"]):
            out.at[i, "REB_PCT"] = _pct(t_trb, t_trb + o_trb)

    return _normalize_pct_cols(out)


def append_adv_game(season: str, game_id: str) -> None:
    import pandas as pd
    gl = stats_getter.get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gdate = gl.loc[gl["GAME_ID"].astype(str) == str(game_id), "GAME_DATE"].iloc[0]

    team_rows = get_advanced_team_game_rows(game_id)
    team_rows = team_rows.copy()
    team_rows["SEASON"] = season
    team_rows["GAME_DATE"] = pd.to_datetime(gdate)

    # ------- POSS fallback (if missing or NaN) -------
    if "POSS" not in team_rows.columns or team_rows["POSS"].isna().any():
        import pandas as pd
        # ensure column exists for assignment
        if "POSS" not in team_rows.columns:
            team_rows["POSS"] = float("nan")
        team_rows["POSS"] = pd.to_numeric(team_rows["POSS"], errors="coerce")
        for idx, r in team_rows.iterrows():
            if pd.isna(r["POSS"]):
                try:
                    v = _fallback_poss_pg_via_advanced(int(r["TEAM_ID"]), season)
                except Exception:
                    v = float("nan")
                team_rows.loc[idx, "POSS"] = v


    # ------- NEW: fallback fill for rebounding % -------
    need_fallback = False
    for col in ["OREB_PCT", "DREB_PCT", "REB_PCT"]:
        if col not in team_rows.columns:
            need_fallback = True
            break
        if team_rows[col].isna().any():
            need_fallback = True
            break

    if need_fallback:
        gl_pair = gl.loc[gl["GAME_ID"].astype(str) == str(game_id)]
        # ensure we have exactly two rows and required columns
        if len(gl_pair) == 2 and all(c in gl_pair.columns for c in ["TEAM_ID","OREB","DREB","REB"]):
            team_rows = _fill_reb_pcts_from_league_log(team_rows, gl_pair)
        # else: silently skip; we’ll just save whatever we have

    led = _load_ledger(season)
    led = pd.concat([led, team_rows], ignore_index=True) if led is not None else team_rows
    led.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)
    led.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], inplace=True)
    _save_ledger(season, led)
    try:
        _team_adv_slice.cache_clear()
    except Exception:
        pass


def _ledger_has_full_prev_team_season(team_id: int, prev_season: str) -> bool:
    """True iff the parquet already has a row for every regular-season game for this team."""
    df = _load_ledger(prev_season)
    if df is None or df.empty:
        return False
    expected = _team_prev_season_game_ids(team_id, prev_season)
    if not expected:
        return False
    have = set(
        df.loc[df["TEAM_ID"].astype(int) == int(team_id), "GAME_ID"].astype(str)
    )
    return expected.issubset(have)

def _to_day(x):
    return _normalize_day(x)

def _ensure_team_priors(season: str, team_id: int, cutoff_date: datetime, N: int = 5) -> None:
    gl = stats_getter.get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    team_gl = gl[(gl["TEAM_ID"] == int(team_id)) & (gl["GAME_DATE"] < cutoff_date)] \
              .sort_values("GAME_DATE")
    if team_gl.empty:
        return

    want_ids = team_gl["GAME_ID"].astype(str).tail(N).tolist()
    led = _load_ledger(season)
    have = set() if led is None or led.empty else set(led[led["TEAM_ID"] == int(team_id)]["GAME_ID"].astype(str))
    missing = [gid for gid in want_ids if gid not in have]
    if not missing:
        return

    new_chunks = []
    for gid in missing:
        team_rows = get_advanced_team_game_rows(gid)   # must be the TEAM table (2 rows)
        if team_rows.empty:
            continue
        gdate = team_gl.loc[team_gl["GAME_ID"].astype(str) == str(gid), "GAME_DATE"].iloc[0]
        team_rows = team_rows.copy()
        team_rows["SEASON"] = season
        team_rows["GAME_DATE"] = pd.to_datetime(gdate)
        new_chunks.append(team_rows)

    if new_chunks:
        led = pd.concat([led, *new_chunks], ignore_index=True) if led is not None else pd.concat(new_chunks, ignore_index=True)
        led.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="last", inplace=True)
        led.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"], inplace=True)
        _save_ledger(season, led)
        try:
            _team_adv_slice.cache_clear()   # bust stale slice cache
        except Exception:
            pass

@lru_cache(maxsize=8192)
def _prev_season_adv_avg(team_id: int, prev_season: str, metric: str) -> float:
    """
    Possessions-weighted previous-season average of an advanced metric,
    computed from the ledger **only if** the entire previous season is present.
    Otherwise returns NaN so the caller can fall back to one LeagueDashTeamStats call.
    """
    # Only compute from ledger if we know the season is fully materialized
    if not _ledger_has_full_prev_team_season(team_id, prev_season):
        return math.nan

    df = _load_ledger(prev_season)
    if df is None or df.empty or metric not in df.columns:
        return math.nan

    # restrict to the team's regular-season window (your helper already exists)
    d0, d1 = team_regular_season_range_by_id(team_id, prev_season)
    s = df[df["TEAM_ID"].astype(int) == int(team_id)].copy()
    s["GAME_DATE"] = _to_day(s["GAME_DATE"])
    s = s[(s["GAME_DATE"] >= _to_day(d0)) & (s["GAME_DATE"] <= _to_day(d1))]
    if s.empty:
        return math.nan

    vals = pd.to_numeric(s[metric], errors="coerce").dropna()
    if vals.empty:
        return math.nan

    # possessions-weighted mean when POSS is available and we're not literally asking for POSS itself
    if ("POSS" in s.columns) and (metric != "POSS"):
        w = pd.to_numeric(s.loc[vals.index, "POSS"], errors="coerce").fillna(0.0)
        ws = w.sum()
        if ws > 0:
            return float((vals * w).sum() / ws)

    # otherwise, simple mean (including when metric == "POSS")
    return float(vals.mean())

# --- helper: API fallback when ledger has no possessions ---
def _fallback_poss_pg_via_advanced(team_id: int, season: str) -> float:
    """
    Return possessions per game for (team_id, season) using LeagueDashTeamStats (Advanced).
    NBA API reports TOTAL possessions even if perGame is requested, so we divide by GP.
    """
    try:
        # Local import to avoid import cost unless needed
        from nba_api.stats.endpoints import leaguedashteamstats, leaguegamelog

        # Pull Advanced totals for the whole season
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_def="Advanced",
            per_mode_detailed="Totals",           # be explicit: totals, we'll normalize
            # team_id filter isn't supported on all deployments; we filter below anyway
        )
        df = resp.get_data_frames()[0]
        if df is None or df.empty:
            return float("nan")

        # Isolate the row for this team
        if "TEAM_ID" in df.columns:
            row = df[df["TEAM_ID"] == int(team_id)]
            if row.empty:  # very defensive
                return float("nan")
            row = row.iloc[0]
        else:
            # Unexpected shape – bail safely
            return float("nan")

        poss_total = float(row.get("POSS", float("nan")))
        gp = row.get("GP", row.get("G", float("nan")))
        gp = float(gp) if gp is not None else float("nan")

        # If GP is missing for any reason, count from game logs
        if not gp or math.isnan(gp) or gp <= 0:
            try:
                gl = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star="Regular Season",
                    team_id_nullable=str(team_id),
                ).get_data_frames()[0]
                gp = float(len(gl)) if gl is not None else float("nan")
            except Exception:
                gp = float("nan")

        if math.isnan(poss_total) or math.isnan(gp) or gp <= 0:
            return float("nan")

        return poss_total / gp

    except Exception:
        # Any API/network hiccup => return NaN, letting upstream handle it gracefully
        return float("nan")

@lru_cache(maxsize=65536)
def _lastN_adv_mean(season: str, team_id: int, date_key: str, metric: str, N: int) -> float:
    d = datetime.strptime(date_key, "%Y-%m-%d")

    # fast early NaN: if prior count < N, skip all I/O
    try:
        gl = stats_getter.get_league_game_log(season).copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        prior_cnt = gl[(gl["TEAM_ID"] == int(team_id)) & (gl["GAME_DATE"] < d)].shape[0]
        if prior_cnt < N:
            return math.nan
    except Exception:
        # if log fails, we’ll fall back to the ledger-only path
        pass

    # fetch only the last N prior rows if we don't have them
    _ensure_team_priors(season, team_id, d, N)

    s = _team_adv_slice(season, int(team_id))
    if s.empty or metric not in s.columns:
        return math.nan

    vals = pd.to_numeric(s.loc[s["GAME_DATE"] < d, metric], errors="coerce").dropna().tail(N)
    if len(vals) < N:
        return math.nan
    return float(vals.mean())






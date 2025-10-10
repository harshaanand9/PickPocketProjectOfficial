import math
import numpy as np
from advanced_first_game import _prior_or_prev
from advanced_ledger import _fallback_poss_pg_via_advanced, _lastN_adv_mean, get_prior_dreb_pct, get_prior_efg_pct, get_prior_metric, get_prior_net_rating, get_prior_oreb_pct, get_prior_pace, get_prior_poss, get_prior_tm_tov_pct
from shotloc_ptshot import get_ptshots_efficiency, get_ptshots_rate, get_shotloc_efficiency, get_shotloc_rate
import stats_getter
from stats_getter import canon_df_team_names, get_league_game_log, get_team_id, getLeagueDashTeamStats
import pandas as pd
from datetime import datetime, timedelta
import importlib


importlib.reload(stats_getter)
from typing import Dict
import cache_manager
from cache_manager import stats_cache

import time
import random
from stats_getter import resolve_season_for_game_by_logs  # add this import

# --- fast possessions-per-game from advanced_ledger ---
from functools import lru_cache
import pandas as pd
from stats_getter import get_team_id
from advanced_ledger import get_prior_poss  # this exists already

from cache_manager import stats_cache

from datetime import datetime
from functools import lru_cache
import math
import pandas as pd
import stats_getter

# ---------- fast prior-games precheck ----------

from functools import lru_cache
import math
from datetime import datetime
import pandas as pd

from stats_getter import (
    get_league_game_log,
    team_regular_season_range_by_id,
    getLeagueDashTeamStats,
    season_for_date_smart,
    get_team_id,
)

def get_roster_timeline(season): 
    return stats_cache.get_or_fetch("roster_timeline", build_team_roster_timeline, season=season)

def _stocks_prior_pg(season: str, team_id: int, cutoff_day: pd.Timestamp) -> float:
    lg = stats_getter.get_league_game_log(season)
    if lg is None or lg.empty:
        return math.nan
    df = lg.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
    df = df[(df["TEAM_ID"].astype(int) == int(team_id)) & (df["GAME_DATE"] < cutoff_day)]
    if df.empty:
        return math.nan
    stl = pd.to_numeric(df.get("STL"), errors="coerce")
    blk = pd.to_numeric(df.get("BLK"), errors="coerce")
    vals = (stl.add(blk, fill_value=0)).dropna()
    return float(vals.mean()) if not vals.empty else math.nan

# if not already imported in this file:
from advanced_ledger import _load_ledger  # uses the in-memory memo + parquet cache

@lru_cache(maxsize=65536)
def _prior_games_count(season: str, team_id: int, date_key: str) -> int:
    """
    Count prior games for team_id in `season` strictly before `date_key` (YYYY-MM-DD).
    Cached so repeated calls are O(1).
    """
    d = datetime.strptime(date_key, "%Y-%m-%d")
    gl = stats_getter.get_league_game_log(season).copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    return int(((gl["TEAM_ID"] == int(team_id)) & (gl["GAME_DATE"] < d)).sum())

def _has_at_least_n_priors(season: str, team_id: int, d: datetime, n: int = 5) -> bool:
    return _prior_games_count(season, int(team_id), d.strftime("%Y-%m-%d")) >= n


from functools import lru_cache
import math, pandas as pd
from datetime import datetime
import stats_getter

from functools import lru_cache
import math, pandas as pd
from datetime import datetime
import stats_getter
import math

def poss_pg_from_ledger(team_name: str, season: str, date_str: str) -> float:
    """
    Preferred path: read possessions/game from our ledger up to date_str.
    Fallback: if ledger value is missing/NaN/non-positive, use Advanced totals from
    LeagueDashTeamStats for the SAME `season` and normalize by GP.
    Note: your callers already pass previous-season `season` for 1st-game cases.
    """
    tid = get_team_id(team_name)
    if tid is None:
        print(f"⚠️ Warning: Unknown team name '{team_name}'")
        return float("nan")

    def _fetch(team_id, season, date_str):
        # 1) try ledger
        val = _ledger_poss_pg(team_id, season, date_str)  # this already hits lru_cache
        if val is not None:
            try:
                v = float(val)
                if not math.isnan(v) and v > 0:
                    return v
            except Exception:
                pass

        # 2) fallback to API (Advanced totals ÷ GP)
        v_api = _fallback_poss_pg_via_advanced(team_id, season)
        return v_api

    # Cache the combined strategy so all dependent functions benefit automatically
    return stats_cache.get_or_fetch(
        "poss_pg_any_source", _fetch, team_id=tid, season=season, date_str=date_str
    )


@lru_cache(maxsize=16384)
def _ledger_poss_pg(team_id: int, season: str, date_str: str) -> float:
    """
    Running possessions per game up to (but not incl.) date_str,
    with opener fallback handled inside advanced_ledger.
    """
    day = pd.to_datetime(date_str).normalize()
    return float(get_prior_poss(season, team_id, day))

from datetime import datetime
import math


import numpy as np
import pandas as pd
from stats_getter import get_league_game_log, get_team_id

def getB2B(team_name: str, season: str, date_str: str) -> float:
    """
    Return 1.0 if this game is the second night of a back-to-back for `team_name`,
    else 0.0. Returns NaN if the (team, date) game can't be located.
    The season used comes from season_for_date_smart(date_str); the `season`
    arg is ignored to avoid mismatches.
    """
    import math
    import numpy as np
    import pandas as pd
    import stats_getter as sg  # your module

    # --- parse date for comparisons ---
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return math.nan
    d = d.normalize()

    # --- get season FROM THE STRING (expects MM/DD/YYYY) ---
    season_key = season_for_date_smart(date_str)  # <-- pass string, not Timestamp

    # --- resolve team id ---
    tid = sg.get_team_id(team_name)
    if tid is None:
        return math.nan

    # --- pull cached league game log for that season ---
    gl = sg.get_league_game_log(season_key).copy()
    if gl is None or gl.empty:
        return math.nan

    # normalize types
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce").dt.normalize()
    gl["TEAM_ID"] = pd.to_numeric(gl["TEAM_ID"], errors="coerce")

    team_log = (
        gl[gl["TEAM_ID"] == int(tid)]
        .sort_values(["GAME_DATE", "GAME_ID"], kind="mergesort")
        .reset_index(drop=True)
    )
    if team_log.empty:
        return math.nan

    # find this game
    matches = team_log.index[team_log["GAME_DATE"] == d].tolist()
    if not matches:
        return math.nan
    i = matches[0]
    if i == 0:
        return 0.0  # first game of season -> not B2B

    prev_date = team_log.iloc[i - 1]["GAME_DATE"]
    return float((d - prev_date).days == 1)




def getPF(team_name: str, season: str, date_str: str) -> float:
    """
    Average personal fouls per game for a single team up to cutoff date.
    """
    team_id = get_team_id(team_name)
    if team_id is None:
        raise ValueError(f"Unknown team: {team_name}")

    target_date = datetime.strptime(date_str, "%m/%d/%Y")

    # ---- helper from your get_pf_diff ----
    def calculate_team_pf_average(team_id, team_season, team_name):
        df_team_season = get_league_game_log(team_season)
        df_team_season["GAME_DATE"] = pd.to_datetime(df_team_season["GAME_DATE"])
        df_filtered = df_team_season[df_team_season["GAME_DATE"] < target_date]
        df_team_games = df_filtered[df_filtered["TEAM_ID"] == team_id]

        if df_team_games.empty:
            print(f"⚠️ Warning: No games for {team_name} in {team_season} before {date_str}")
            return 0.0

        return df_team_games["PF"].mean()
    # ---------------------------------------------------------

    return calculate_team_pf_average(team_id, season, team_name)

def getPace(team_name: str, season: str, date_str: str) -> float:
    """
    Returns PACE for one team using ledger prior values
    (avg through the day BEFORE date_str).
    """
    d = datetime.strptime(date_str, "%m/%d/%Y")
    tid = get_team_id(team_name)
    if tid is None:
        return np.nan

    val = float(get_prior_pace(season, tid, d))
    if math.isnan(val):
        return np.nan
    return val


# ---------- 1) CLUTCH NET RATING (single team) ----------
def getClutchNetRtg(team_name: str, team_season: str, date_str: str) -> float:
    """
    Returns the team's clutch NET_RATING up to the day BEFORE `date_str`,
    using the league-wide cached endpoint.
    """

    from datetime import datetime, timedelta
    import math, numpy as np, pandas as pd
    import stats_getter as sg

    # --- Prepare dates ---
    target = datetime.strptime(date_str, "%m/%d/%Y")
    cutoff = (target - timedelta(days=1)).strftime("%m/%d/%Y")
    start_yr = int(team_season.split("-")[0])
    date_from = f"10/01/{start_yr}"

    # --- Fetch league-wide clutch stats ---
    df = sg.getLeagueDashTeamClutch(
        season=team_season,
        date_from=date_from,
        date_to=cutoff,
        measure_type="Advanced",
        per_mode="PerGame",
    )

    if df is None or df.empty:
        return math.nan

    # --- Canonicalize names and extract team row ---
    df = df.copy()
    if "TEAM_NAME" in df.columns:
        df["TEAM_NAME"] = df["TEAM_NAME"].map(sg.canon_team)

    team_name = sg.canon_team(team_name)
    row = df.loc[df["TEAM_NAME"] == team_name]

    if row.empty or "NET_RATING" not in row.columns:
        return math.nan

    try:
        return float(row["NET_RATING"].iloc[0])
    except Exception:
        return math.nan

# ---------- 2) FTA-RATE RELATIVE (single team) ----------
def getFTARateRelative(team_name: str, opp_name: str, team_season: str, date_str: str) -> float:
    """
    Team's FTA_RATE * Opponent-FTA-Allowed_RATE, using only games BEFORE `date_str`.
    This is exactly the `_feature(...)` helper inside your ratio function.
    """
    from datetime import datetime
    import math
    import pandas as pd
    import stats_getter as sg

    target_dt = datetime.strptime(date_str, "%m/%d/%Y")

    # ---- helper lifted unchanged from your ratio function ----
    def _feature(team_name: str, opp_name: str, team_season: str) -> float:
        df = sg.get_league_game_log(team_season).copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df_before = df[df["GAME_DATE"] < target_dt]
        if df_before.empty:
            return math.nan

        team_id = sg.get_team_id(team_name)
        if team_id is None:
            return math.nan

        # this team
        df_team = df_before[df_before["TEAM_ID"] == team_id]
        if df_team.empty:
            return math.nan
        fga_team = df_team["FGA"].sum()
        if fga_team == 0:
            return math.nan
        fta_rate_team = df_team["FTA"].sum() / fga_team

        # opponents in those same games
        opp_rows = df_before[df_before["GAME_ID"].isin(df_team["GAME_ID"]) & (df_before["TEAM_ID"] != team_id)]
        if opp_rows.empty:
            return math.nan
        fga_opp = opp_rows["FGA"].sum()
        if fga_opp == 0:
            return math.nan
        fta_rate_opp_allowed = opp_rows["FTA"].sum() / fga_opp

        return fta_rate_team * fta_rate_opp_allowed
    # ---------------------------------------------------------

    return _feature(team_name, opp_name, team_season)




# ---------- 3) PERSONAL FOULS DRAWN per game (single team) ----------
def getPFD(team_name: str, team_season: str, date_str: str) -> float:
    """
    Team's prior per-game PFD up to the day BEFORE `date_str`.
    Single league-wide call; extract our team's row. No first-game branching.
    """
    from datetime import datetime, timedelta
    import math, pandas as pd
    import stats_getter as sg

    # --- Prepare dates (same style as getClutchNetRtg) ---
    target = datetime.strptime(date_str, "%m/%d/%Y")
    cutoff = (target - timedelta(days=1)).strftime("%m/%d/%Y")
    start_yr = int(team_season.split("-")[0])
    date_from = f"10/01/{start_yr}"

    # --- League-wide TeamStats (Base / PerGame) ---
    df = sg.getLeagueDashTeamStats(
        team_name=None,          # league-wide one call (team_id_nullable=None)
        season=team_season,
        date_from=date_from,
        date_to=cutoff,
        measure_type="Base",
        per_mode="PerGame",
    )

    if df is None or df.empty or "PFD" not in df.columns:
        return math.nan

    # --- Select our team row (ID preferred; fallback to canonical name) ---
    tid = sg.get_team_id(team_name)
    row = pd.DataFrame()
    if "TEAM_ID" in df.columns and tid is not None:
        row = df.loc[df["TEAM_ID"] == int(tid)]
    if row.empty and "TEAM_NAME" in df.columns:
        work = df.copy()
        work["TEAM_NAME"] = work["TEAM_NAME"].astype(str).map(sg.canon_team)
        row = work.loc[work["TEAM_NAME"] == sg.canon_team(team_name)]

    if row.empty:
        return math.nan

    try:
        return float(pd.to_numeric(row["PFD"].iloc[0], errors="coerce"))
    except Exception:
        return math.nan


from datetime import datetime, timedelta
import math
import pandas as pd
import stats_getter
from stats_getter import get_team_id
from advanced_ledger import get_prior_dreb_pct  # ledger prior accessor
from datetime import datetime
import math
import numpy as np

from stats_getter import get_team_id
from advanced_ledger import (
    get_prior_oreb_pct,
    get_prior_dreb_pct,
)

def _as_frac(x: float) -> float:
    """Normalize 0–100 style percentages to 0–1 fractions; pass NaN through."""
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if math.isnan(x):
        return x
    if x > 1.0:
        x = x / 100.0
    # clamp for safety
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return x


def getOrebPct(team_name: str, season: str, date_str: str) -> float:
    """
    Returns a team's prior Offensive Rebound Percentage up to the day BEFORE `date_str`.
    """
    d = datetime.strptime(date_str, "%m/%d/%Y")
    tid = get_team_id(team_name)
    if tid is None:
        return np.nan

    val = get_prior_oreb_pct(season, tid, d)
    return float(val) if val is not None and not math.isnan(val) else np.nan





def getDrebPct(team_name: str, season: str, date_str: str) -> float:
    """
    Returns a team's prior Defensive Rebound Percentage up to the day BEFORE `date_str`.
    """
    d = datetime.strptime(date_str, "%m/%d/%Y")
    tid = get_team_id(team_name)
    if tid is None:
        return np.nan

    val = get_prior_dreb_pct(season, tid, d)
    return float(val) if val is not None and not math.isnan(val) else np.nan




from datetime import datetime, timedelta
import math
import pandas as pd
import stats_getter

# assumes you already added this helper earlier in the file:
#   from . import poss_pg_from_ledger
# or if it's in the same file, just make sure it's defined above this function.

def getAstRatio(team_name: str, team_season: str, date_str: str) -> float:
    """
    Team assist ratio up to the day BEFORE `date_str`:
        AST_Ratio = (team assists per game) / (team possessions per game)

    First-game fallback:
        If the team has no games before `date_str` in `team_season`,
        use the team's previous season *second-half*:
            AST_Ratio = mean(AST over window) / poss_pg_from_ledger(team, prev_season, window_end)
    """
    from datetime import datetime, timedelta
    import math, numpy as np, pandas as pd
    import stats_getter as sg
    from advanced_ledger import get_prior_poss as poss_pg_from_ledger

    def _to_date(s):
        return pd.to_datetime(s, errors="coerce").normalize()

    # --- current-season cutoff (day BEFORE date_str) ---
    target    = datetime.strptime(date_str, "%m/%d/%Y")
    cutoff_dt = target - timedelta(days=1)

    # --- league log up to cutoff ---
    try:
        gla = sg.get_league_game_log(team_season).copy()
        gla["GAME_DATE"] = pd.to_datetime(gla["GAME_DATE"], errors="coerce")
        gla = gla[gla["GAME_DATE"] <= cutoff_dt]
    except Exception:
        return np.nan

    tid = sg.get_team_id(team_name)
    if tid is None:
        return np.nan

    a_rows = gla[gla["TEAM_ID"] == tid]

    # ---------- FIRST-GAME FALLBACK ----------
    if a_rows.empty:
        try:
            # previous season string
            try:
                prev_season = sg.get_previous_season(team_season)
            except Exception:
                start_y = int(team_season[:4]) - 1
                prev_season = f"{start_y}-{str((start_y+1)%100).zfill(2)}"

            # window: previous season's second half
            win_from, win_to = _team_second_half_window_by_id(tid, prev_season)
            if not win_from or not win_to:
                return np.nan

            prev_log = sg.get_league_game_log(prev_season).copy()
            prev_log["GAME_DATE"] = pd.to_datetime(prev_log["GAME_DATE"], errors="coerce").dt.normalize()
            df = prev_log[
                (prev_log["TEAM_ID"].astype(int) == int(tid)) &
                (prev_log["GAME_DATE"] >= _to_date(win_from)) &
                (prev_log["GAME_DATE"] <= _to_date(win_to))
            ].copy()
            if df.empty or "AST" not in df.columns:
                return np.nan

            ast_pg = float(df["AST"].mean())

            # possessions per game from your ledger prior (prev season, window end)
            team_id = sg.get_team_id(team_name)
            poss_pg = get_prior_poss(prev_season, team_id, win_to)
            if poss_pg is None or (isinstance(poss_pg, float) and (math.isnan(poss_pg) or poss_pg <= 0)):
                return np.nan

            return ast_pg / float(poss_pg)

        except Exception:
            return np.nan

    # ---------- NORMAL PATH ----------
    if "AST" not in a_rows.columns:
        return np.nan

    ast_pg_curr = float(a_rows["AST"].mean())
    team_id = sg.get_team_id(team_name)
    # here you have to go one day back from date_strprev_date_str:
    prev_date_str = (datetime.strptime(date_str, "%m/%d/%Y") - timedelta(days=1)).strftime("%m/%d/%Y")
    poss_pg_curr = get_prior_poss(team_season, team_id, prev_date_str)
    if poss_pg_curr is None or (isinstance(poss_pg_curr, float) and (math.isnan(poss_pg_curr) or poss_pg_curr <= 0)):
        return np.nan

    return ast_pg_curr / float(poss_pg_curr)




# --- Turnover % (team) ---
def getTurnoverPct(team_name: str, season: str, date_str: str) -> float:
    """
    Team TOV% up to the day BEFORE `date_str` from advanced_ledger.
    """
    from datetime import datetime
    import math, numpy as np
    d = datetime.strptime(date_str, "%m/%d/%Y")
    tid = stats_getter.get_team_id(team_name)
    if tid is None:
        return np.nan
    try:
        val = float(get_prior_metric(season, tid, d, "TM_TOV_PCT"))
        return val if not math.isnan(val) else np.nan
    except Exception:
        return np.nan

# --- Opponents' Turnover % allowed vs this team ---
def getOppTurnoverPctAllowed(team_name: str, season: str, date_str: str) -> float:
    """
    Team_OPP_TOV_PCT: average opponent TOV% when facing `team_name`,
    computed as (opp TOV per game / team possessions per game) * 100
    using games strictly BEFORE `date_str`. Mirrors the logic embedded in
    your old get_turnover_ratio_relative.
    """
    from datetime import datetime, timedelta
    import math, numpy as np, pandas as pd

    target_dt = datetime.strptime(date_str, "%m/%d/%Y")
    cutoff_dt = target_dt - timedelta(days=1)

    tid = stats_getter.get_team_id(team_name)
    if tid is None:
        return np.nan

    # league game log to cutoff
    try:
        gl = stats_getter.get_league_game_log(season).copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        glb = gl[gl["GAME_DATE"] <= cutoff_dt]
    except Exception:
        return np.nan
    if glb.empty:
        return np.nan

    # rows for this team, then pull opponent rows in those same games
    team_rows = glb[glb["TEAM_ID"] == tid]
    if team_rows.empty:
        return np.nan
    opp_rows = glb[glb["GAME_ID"].isin(team_rows["GAME_ID"]) & (glb["TEAM_ID"] != tid)]
    if opp_rows.empty:
        return np.nan

    # opponent TOV per game
    opp_tov_pg = float(opp_rows["TOV"].mean()) if "TOV" in opp_rows else np.nan
    if math.isnan(opp_tov_pg):
        return np.nan

    # possessions per game (use your ledger prior, same as before)
    poss_pg = adv.get_prior_poss(season, tid, cutoff_dt)
    if poss_pg is None or (isinstance(poss_pg, float) and (math.isnan(poss_pg) or poss_pg <= 0)):
        return np.nan

    return (opp_tov_pg / float(poss_pg)) * 100.0



# --- % of FGM that are assisted (single team) ---
def getPctAstFGM(team_name: str, season: str, date_str: str) -> float:
    """
    Team's PCT_AST_FGM up to the day BEFORE `date_str` (Scoring → PerGame),
    using a single league-wide call and extracting the team's row.
    """
    from datetime import datetime, timedelta
    import math, numpy as np, pandas as pd
    import stats_getter as sg

    # --- window [10/01/<start_yr>, cutoff) to mirror your other features ---
    target   = datetime.strptime(date_str, "%m/%d/%Y")
    cutoff   = (target - timedelta(days=1)).strftime("%m/%d/%Y")
    start_yr = int(season.split("-")[0])
    date_from = f"10/01/{start_yr}"

    # --- league-wide frame (one call; cached) ---
    df = sg.getLeagueDashTeamStats(
        team_name=None,         # league-wide (team_id_nullable=None)
        season=season,
        date_from=date_from,
        date_to=cutoff,
        measure_type="Scoring",
        per_mode="PerGame",
    )

    if df is None or df.empty or "PCT_AST_FGM" not in df.columns:
        return np.nan

    # --- pick team row: prefer TEAM_ID, fallback to canonical TEAM_NAME ---
    tid = sg.get_team_id(team_name)
    row = pd.DataFrame()
    if "TEAM_ID" in df.columns and tid is not None:
        row = df.loc[df["TEAM_ID"] == int(tid)]

    if row.empty and "TEAM_NAME" in df.columns:
        work = df.copy()
        work["TEAM_NAME"] = work["TEAM_NAME"].astype(str).map(sg.canon_team)
        row = work.loc[work["TEAM_NAME"] == sg.canon_team(team_name)]

    if row.empty:
        return np.nan

    try:
        return float(pd.to_numeric(row["PCT_AST_FGM"].iloc[0], errors="coerce"))
    except Exception:
        return np.nan



# ---- small helpers ----
def _mi_get(df, cat, stat, default=0.0):
    """Read a value from a MultiIndex column (cat, stat)."""
    try:
        val = df[(cat, stat)].iloc[0]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def _mi_get_any(df, cat, stat_candidates, default=0.0):
    """Try multiple possible stat names (e.g., 'FGA' vs 'OPP_FGA')."""
    for s in stat_candidates:
        if (cat, s) in getattr(df, "columns", []):
            return _mi_get(df, cat, s, default)
    return default



# ---------- helpers ----------

def _season_key(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _season_start_date(season: str):
    """Actual first game date for `season` from LeagueGameLog (cached)."""
    try:
        df = stats_getter.get_league_game_log(season)
        if df is None or df.empty:
            return None
        return pd.to_datetime(df["GAME_DATE"]).min()
    except Exception:
        return None

def season_for_date_smart(date_str: str) -> str:
    """
    Map a calendar date (MM/DD/YYYY) to the correct NBA season key (YYYY-YY),
    using real season start dates when available, else a month heuristic.
    """
    dt = datetime.strptime(date_str, "%m/%d/%Y")
    years = [dt.year - 1, dt.year, dt.year + 1]
    candidates = [_season_key(y) for y in years]

    starts = []
    for s in candidates:
        sd = _season_start_date(s)
        if sd is not None and pd.notna(sd):
            starts.append((s, sd))
    if starts:
        starts.sort(key=lambda x: x[1])  # by start date
        for i, (s, s_start) in enumerate(starts):
            next_start = starts[i + 1][1] if i + 1 < len(starts) else None
            if dt >= s_start and (next_start is None or dt < next_start):
                return s
        # if date precedes earliest known start, attribute to previous season
        earliest_season, earliest_start = starts[0]
        if dt < earliest_start:
            y = int(earliest_season[:4]) - 1
            return _season_key(y)
    # fallback heuristic for normal years
    start_year = dt.year if dt.month >= 8 else dt.year - 1
    return _season_key(start_year)

def _get_team_id(team_name: str):
    """Try namespaced getter first; fall back to global if present."""
    try:
        return stats_getter.get_team_id(team_name)
    except AttributeError:
        return get_team_id(team_name)

def get_team_last_5_games_date(team_name: str, target_date: str, season: str):
    """
    Start date for the team's last-5-games window BEFORE target_date (MM/DD/YYYY).
    Returns NaN if:
      - target_date maps to a different NBA season than `season` (first-game scenario),
      - fewer than 5 games before target_date in that season,
      - team not found or API issues.
    """
    # If you're using last season's data for a new-season game, return NaN by design.
    if season_for_date_smart(target_date) != season:
        return math.nan

    try:
        df_all = stats_getter.get_league_game_log(season)
    except Exception:
        return math.nan

    team_id = _get_team_id(team_name)
    if not team_id:
        return math.nan

    df = df_all[df_all["TEAM_ID"] == team_id].copy()
    if df.empty:
        return math.nan

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    target_dt = datetime.strptime(target_date, "%m/%d/%Y")
    games_before = df[df["GAME_DATE"] < target_dt].sort_values("GAME_DATE", ascending=False)

    if len(games_before) < 5:
        return math.nan

    fifth_date = games_before.iloc[4]["GAME_DATE"]
    return fifth_date.strftime("%m/%d/%Y")


# ---------- main feature ----------

from datetime import datetime
import math
import stats_getter

from datetime import datetime
import math
import stats_getter

def get_recent_netrtg_home(team_a_name: str, team_b_name: str,
                           team_a_season: str, team_b_season: str,
                           date_str: str, N: int = 5) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, N) or
        not _has_at_least_n_priors(team_b_season, b_id, d, N)):
        return math.nan
    a_lastN = _lastN_adv_mean(team_a_season, a_id, d.strftime("%Y-%m-%d"), "NET_RATING", N)
    return a_lastN

def get_recent_netrtg_away(team_a_name: str, team_b_name: str,
                           team_a_season: str, team_b_season: str,
                           date_str: str, N: int = 5) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, N) or
        not _has_at_least_n_priors(team_b_season, b_id, d, N)):
        return math.nan
    b_lastN = _lastN_adv_mean(team_b_season, b_id, d.strftime("%Y-%m-%d"), "NET_RATING", N)
    return b_lastN



def get_recent_oreb_pct_home(team_a_name: str, team_b_name: str,
                             team_a_season: str, team_b_season: str,
                             date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_a_season, a_id, d.strftime("%Y-%m-%d"), "OREB_PCT", 5)

def get_recent_oreb_pct_away(team_a_name: str, team_b_name: str,
                             team_a_season: str, team_b_season: str,
                             date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_b_season, b_id, d.strftime("%Y-%m-%d"), "OREB_PCT", 5)



def get_recent_efg_pct_home(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_a_season, a_id, d.strftime("%Y-%m-%d"), "EFG_PCT", 5)

def get_recent_efg_pct_away(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_b_season, b_id, d.strftime("%Y-%m-%d"), "EFG_PCT", 5)



def get_recent_tov_pct_home(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_a_season, a_id, d.strftime("%Y-%m-%d"), "TM_TOV_PCT", 5)

def get_recent_tov_pct_away(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    d = datetime.strptime(date_str, "%m/%d/%Y")
    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan
    if (not _has_at_least_n_priors(team_a_season, a_id, d, 5) or
        not _has_at_least_n_priors(team_b_season, b_id, d, 5)):
        return math.nan
    return _lastN_adv_mean(team_b_season, b_id, d.strftime("%Y-%m-%d"), "TM_TOV_PCT", 5)



def _recent_ft_rate_pair(team_a_name: str, team_b_name: str,
                         team_a_season: str, team_b_season: str,
                         date_str: str):
    import math
    from datetime import datetime
    import pandas as pd

    # quick guard via your current helpers
    a_last5 = get_team_last_5_games_date(team_a_name, date_str, team_a_season)
    b_last5 = get_team_last_5_games_date(team_b_name, date_str, team_b_season)
    if pd.isna(a_last5) or pd.isna(b_last5):
        return math.nan, math.nan

    target_dt = datetime.strptime(date_str, "%m/%d/%Y")

    df_a = stats_getter.get_league_game_log(team_a_season).copy()
    df_b = df_a if team_b_season == team_a_season else stats_getter.get_league_game_log(team_b_season).copy()

    if "GAME_DATE" not in df_a.columns or "GAME_DATE" not in df_b.columns:
        return math.nan, math.nan
    df_a["GAME_DATE"] = pd.to_datetime(df_a["GAME_DATE"])
    df_b["GAME_DATE"] = pd.to_datetime(df_b["GAME_DATE"])

    df_a_before = df_a[df_a["GAME_DATE"] < target_dt]
    df_b_before = df_b[df_b["GAME_DATE"] < target_dt]
    if df_a_before.empty or df_b_before.empty:
        return math.nan, math.nan

    a_id = stats_getter.get_team_id(team_a_name)
    b_id = stats_getter.get_team_id(team_b_name)
    if a_id is None or b_id is None:
        return math.nan, math.nan

    needed = {"TEAM_ID", "GAME_DATE", "FTA", "FGA"}
    if (not needed.issubset(df_a_before.columns)) or (not needed.issubset(df_b_before.columns)):
        return math.nan, math.nan

    a_rows = (df_a_before[df_a_before["TEAM_ID"] == a_id]
              .sort_values("GAME_DATE", ascending=False)
              .head(5))
    b_rows = (df_b_before[df_b_before["TEAM_ID"] == b_id]
              .sort_values("GAME_DATE", ascending=False)
              .head(5))
    if len(a_rows) < 5 or len(b_rows) < 5:
        return math.nan, math.nan

    a_fga_sum = a_rows["FGA"].sum()
    b_fga_sum = b_rows["FGA"].sum()
    if a_fga_sum == 0 or b_fga_sum == 0:
        return math.nan, math.nan

    a_ft_rate_last5 = a_rows["FTA"].sum() / a_fga_sum
    b_ft_rate_last5 = b_rows["FTA"].sum() / b_fga_sum
    if pd.isna(a_ft_rate_last5) or pd.isna(b_ft_rate_last5):
        return math.nan, math.nan

    return float(a_ft_rate_last5), float(b_ft_rate_last5)


def get_recent_ft_rate_home(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    a, _ = _recent_ft_rate_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return a

def get_recent_ft_rate_away(team_a_name: str, team_b_name: str,
                            team_a_season: str, team_b_season: str,
                            date_str: str) -> float:
    _, b = _recent_ft_rate_pair(team_a_name, team_b_name, team_a_season, team_b_season, date_str)
    return b



def _get_team_id_safe(team_name: str):
    try:
        return stats_getter.get_team_id(team_name)
    except AttributeError:
        return get_team_id(team_name)
    
def _team_regular_season_range(team_name: str, season: str):
    """
    Return (date_from, date_to) in MM/DD/YYYY for the team's REGULAR SEASON games in `season`.
    Returns (None, None) if not found.
    """
    try:
        # Your wrapper should already default to Regular Season (nba_api does).
        df = stats_getter.get_league_game_log(season)
    except Exception:
        return None, None

    tid = _get_team_id_safe(team_name)
    if not tid or df is None or df.empty:
        return None, None

    df = df[df["TEAM_ID"] == tid].copy()
    if df.empty:
        return None, None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    date_from = df["GAME_DATE"].min().strftime("%m/%d/%Y")
    date_to   = df["GAME_DATE"].max().strftime("%m/%d/%Y")
    return date_from, date_to
    
from datetime import datetime
import math
import pandas as pd
import stats_getter

from datetime import datetime
import math
import pandas as pd
import stats_getter

def _prev_season(season_key: str) -> str:
    start = int(season_key[:4]) - 1
    return f"{start}-{str(start+1)[2:]}"   # e.g. '2019-20' -> '2018-19'

def _ensure_scalar_date(x):
    import pandas as pd, numpy as np
    # Accepts string / Timestamp / numpy datetime64 / python datetime.
    # Rejects Series/Index/ndarray of multiple dates.
    from datetime import datetime
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)):
        # If someone passed a single-element vector, unwrap; else error (it shouldn't happen)
        if len(x) == 1:
            x = x[0]
        else:
            raise TypeError(f"date_str must be scalar; got {type(x)} with len={len(x)}")
    return x


def _last_season_metric(team_name: str, date_str: str, metric: str) -> float:
    """
    Previous season, SECOND-HALF ONLY average for the given `metric`
    (e.g., 'NET_RATING', 'OREB_PCT'), ledger-first with a single-endpoint fallback.
    """
    import math
    import pandas as pd
    try:
        import stats_getter as sg
    except Exception:
        sg = __import__("stats_getter")

    date_str = _ensure_scalar_date(date_str)
    prev_season = _prev_season(season_for_date_smart(date_str))
    tid = sg.get_team_id(team_name)
    if tid is None:
        return math.nan

    # compute the team-specific 2nd-half window for the PREVIOUS season
    d_from, d_to = _team_second_half_window_by_id(int(tid), prev_season)

    # 1) Try ADV ledger if present: filter by date range and average the metric
    try:
        from advanced_ledger import _load_ledger as _load_adv_ledger  # your helper
    except Exception:
        _load_adv_ledger = None

    if _load_adv_ledger is not None and d_from and d_to:
        led = _load_adv_ledger(prev_season)
        if led is not None and not led.empty and metric in led.columns:
            s = led[led["TEAM_ID"].astype(int) == int(tid)].copy()
            s["GAME_DATE"] = pd.to_datetime(s["GAME_DATE"], errors="coerce").dt.normalize()
            s = s[(s["GAME_DATE"] >= pd.to_datetime(d_from)) & (s["GAME_DATE"] <= pd.to_datetime(d_to))]
            vals = pd.to_numeric(s[metric], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.mean())

    # 2) Fallback: one LeagueDashTeamStats call, bounded to the 2nd-half window
    df = sg.getLeagueDashTeamStats(
        team_name,
        prev_season,
        date_from=d_from,
        date_to=d_to,
        measure_type="Advanced",
        per_mode="PerGame",
    )
    if df is None or df.empty:
        return math.nan

    sel = df.copy()
    if "TEAM_NAME" in sel.columns:
        sel = sel[sel["TEAM_NAME"].notna()]
    if "GP" in sel.columns and len(sel) > 1:
        sel = sel.sort_values("GP", ascending=False)
    row = sel.iloc[0]

    if metric == "POSS":
        poss = pd.to_numeric(row.get("POSS"), errors="coerce")
        gp   = pd.to_numeric(row.get("GP"),   errors="coerce")
        if pd.notna(poss) and pd.notna(gp) and gp != 0:
            return float(poss / gp)

    return float(pd.to_numeric(row.get(metric), errors="coerce"))


# Thin wrappers you already have:
def get_last_season_NETRTG(team_name: str, date_str: str) -> float:
    date_str = _ensure_scalar_date(date_str)
    return _last_season_metric(team_name, date_str, "NET_RATING")

def get_last_season_OREB_PCT(team_name: str, date_str: str) -> float:
    return _last_season_metric(team_name, date_str, "OREB_PCT")



def get_last_season_FT_RATE(
    team_name: str,
    date_str: str,
    stats_getter=None,
) -> float:
    """
    Team FTA/FGA for the SECOND HALF of the season immediately preceding `date_str`.
    """
    import math
    import pandas as pd

    sg = stats_getter or __import__("stats_getter")
    prev_season = _prev_season(season_for_date_smart(date_str))

    team_id = sg.get_team_id(team_name)
    if team_id is None:
        return math.nan

    # second-half window (team-specific)
    date_from, date_to = _team_second_half_window_by_id(int(team_id), prev_season)
    if not date_from or not date_to:
        return math.nan

    # League gamelog (cached)
    try:
        df = sg.get_league_game_log(prev_season)
    except Exception:
        return math.nan
    if df is None or df.empty:
        return math.nan

    needed = {"TEAM_ID", "GAME_DATE", "FTA", "FGA"}
    if not needed.issubset(set(df.columns)):
        return math.nan

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    start = pd.to_datetime(date_from)
    end   = pd.to_datetime(date_to)

    mask = (df["TEAM_ID"].astype(int) == int(team_id)) & df["GAME_DATE"].between(start, end, inclusive="both")
    if "SEASON_TYPE" in df.columns:
        mask &= df["SEASON_TYPE"].str.contains("Regular", case=False, na=False)

    rows = df.loc[mask, ["FTA", "FGA"]]
    if rows.empty:
        return math.nan

    fta_tot = pd.to_numeric(rows["FTA"], errors="coerce").sum()
    fga_tot = pd.to_numeric(rows["FGA"], errors="coerce").sum()
    if pd.isna(fta_tot) or pd.isna(fga_tot) or fga_tot == 0:
        return math.nan

    return float(fta_tot / fga_tot)


def roster_on_date(team_name: str, game_date, timeline: pd.DataFrame) -> set[str]:
    """
    Return the set of PLAYER_NAME on `team_name`'s roster ON `game_date`,
    using a prebuilt team/player timeline that has first/last appearance
    windows per (TEAM_NAME, PLAYER_NAME).

    Parameters
    ----------
    team_name : str
        Team label (any common variant; will be canonicalized).
    game_date : str | datetime-like
        Date to test membership on (inclusive). If str, expects "MM/DD/YYYY".
    timeline : pd.DataFrame
        DataFrame with columns:
          - TEAM_NAME (canonicalized, e.g., "LA Clippers")
          - PLAYER_NAME (string)
          - first_game_date (Timestamp)
          - last_game_date  (Timestamp)

    Returns
    -------
    set[str]
        Players on the roster for that team on that date.
    """
    import pandas as pd
    try:
        import stats_getter as sg
    except Exception:
        sg = __import__("stats_getter")  # lazy import fallback

    # Normalize inputs
    team_canon = sg.canon_team(team_name)
   
    
    dt = pd.to_datetime(game_date) if not isinstance(game_date, pd.Timestamp) else game_date

    # Defensive column checks / coercions
    df = timeline
    if "TEAM_NAME" not in df.columns or "PLAYER_NAME" not in df.columns:
        return set()
    # Coerce date columns if needed
    if "first_game_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["first_game_date"]):
        df = df.copy()
        df["first_game_date"] = pd.to_datetime(df["first_game_date"], errors="coerce")
    if "last_game_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["last_game_date"]):
        if df is timeline:
            df = df.copy()
        df["last_game_date"] = pd.to_datetime(df["last_game_date"], errors="coerce")

    # Filter: team match (normalize just in case), and date inside [first, last]
    team_col_norm = df["TEAM_NAME"].astype(str).map(sg.canon_team)
    team_canon    = sg.canon_team(team_name)
    mask_team     = team_col_norm == team_canon


    # Some players might have missing starts/ends early in season; treat NaN conservatively.
    start_ok = (df["first_game_date"].isna()) | (df["first_game_date"] <= dt)
    end_ok   = (df["last_game_date"].isna())  | (df["last_game_date"] >= dt)

    roster = (
        df.loc[mask_team & start_ok & end_ok, "PLAYER_NAME"]
          .dropna()
          .astype(str)
          .str.strip()
    )

    return set(roster.tolist())



def get_netrtg_diff_prev_season(team_name: str, season: str, date_str: str) -> float:
    """
    Difference between the team's current-season NET_RATING prior (avg through the day
    BEFORE `date_str`, from the advanced ledger only) and the team's previous-season
    NET_RATING (full regular season).

    Returns NaN if `date_str` belongs to a different season than `season` (first-game
    scenario) or if any lookup fails.
    """
    import math
    from datetime import datetime
    import pandas as pd

    # First-game guard: if the date's season doesn't match `season`, return NaN
    season_from_date = season_for_date_smart(date_str)
    if season_from_date != season:
        return math.nan

    # Parse date and resolve team id
    d = datetime.strptime(date_str, "%m/%d/%Y")
    team_id = stats_getter.get_team_id(team_name)
    if team_id is None:
        return math.nan

    # Current-season prior NET_RATING from the advanced ledger (no bulk ensure)
    try:
        from advanced_ledger import get_prior_net_rating
        curr_netrtg = float(get_prior_net_rating(season, team_id, d))
    except Exception:
        # Conservative fallback to generic accessor if names differ in your file
        from advanced_ledger import get_prior_metric
        curr_netrtg = float(get_prior_metric(season, team_id, d, "NET_RATING"))

    if math.isnan(curr_netrtg):
        return math.nan

    # Previous-season NET_RATING (your existing helper; ledger-first with endpoint fallback)
    prev_netrtg = get_last_season_NETRTG(team_name, date_str)
    if math.isnan(prev_netrtg):
        return math.nan

    return curr_netrtg - float(prev_netrtg)



def get_game_number(team_name: str, season: str, date_str: str) -> int:
    """
    Returns the game number for a team on a given date within the **correct** NBA season.
    If the caller passes last season for an opening-night game, we infer the season
    from `date_str` so the first game is correctly returned as 1.
    Returns 0 if the team did not play on `date_str`.
    """
    # Use the actual season that this date belongs to (handles opening night cases)
    season_to_use = season_for_date_smart(date_str)

    # Team ID
    try:
        team_id = stats_getter.get_team_id(team_name)
    except AttributeError:
        team_id = get_team_id(team_name)
    if not team_id:
        return 0

    # League game log for the inferred season
    df_league = stats_getter.get_league_game_log(season_to_use)
    if df_league is None or df_league.empty:
        return 0

    # Filter to this team's games, normalize dates
    df_team = df_league[df_league["TEAM_ID"] == team_id].copy()
    if df_team.empty:
        return 0

    df_team["GAME_DATE"] = pd.to_datetime(df_team["GAME_DATE"]).dt.normalize()
    df_team = df_team.sort_values("GAME_DATE").reset_index(drop=True)

    target_date = pd.to_datetime(date_str).normalize()

    # Find the row for the target date
    mask = (df_team["GAME_DATE"] == target_date)
    if not mask.any():
        return 0  # no game on that date

    # 0-based index -> 1-based game number
    game_index = mask.idxmax()  # first True index
    return int(game_index) + 1


def getTeam_PlayerAggregated_StocksDREB_last_year(
    team_name: str,
    prev_season: str,
    timeline_prev: pd.DataFrame,
    gp_cut: int = 13,
    min_cut_per_game: float = 12.0,
    name_aliases: dict[str, str] | None = None,
    measure_type: str = "Base",
    per_mode: str = "PerGame",
) -> float:
    import re, unicodedata
    import pandas as pd
    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)


    prev_set_raw = timeline_prev.loc[
        timeline_prev["TEAM_NAME"].astype(str).str.strip() == team_name, "PLAYER_NAME"
    ].astype(str).str.strip()
    prev_set = set(prev_set_raw)

    

    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    aliases = {"robert williams": "robert williams iii", "enes freedom": "enes kanter",
               "juancho hernangomez": "juancho hernangomez"}
    if name_aliases: aliases.update({_norm(k): v for k, v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    prev_setN = {_apply_alias(x) for x in prev_set}

    tid = sg.get_team_id(team_name)
    d_from, d_to = _team_second_half_window_by_id(int(tid), prev_season) if tid else (None, None)

    df_team_prev = sg.getLeagueDashPlayerStats(
        team_name, prev_season,
        measure_type=measure_type, per_mode=per_mode,
        date_from=d_from, date_to=d_to
    )


    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["PLAYER_NAME","GP","MIN","STL","BLK","DREB","_N","_STOCKS_DREB"])
        out = df.copy()
        out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
        out["_N"] = out["PLAYER_NAME"].apply(_apply_alias)
        for c in ("GP","MIN","STL","BLK","DREB"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["_STOCKS_DREB"] = out.get("STL",0.0) + out.get("BLK",0.0) + out.get("DREB",0.0)
        return out

    t = _prep(df_team_prev)
    t = t[t["_N"].isin(prev_setN)]
    if "GP" in t.columns:  t = t[t["GP"]  >= gp_cut]
    if "MIN" in t.columns: t = t[t["MIN"] >= min_cut_per_game]

    return float(pd.to_numeric(t["_STOCKS_DREB"], errors="coerce").sum())



def getTeam_PlayerAggregated_AstTov_last_year(
    team_name: str,
    prev_season: str,
    timeline_prev: pd.DataFrame,
    gp_cut: int = 13,
    min_cut_per_game: float = 12.0,
    name_aliases: dict[str, str] | None = None,
    measure_type: str = "Base",
    per_mode: str = "PerGame",
) -> float:
    import re, unicodedata
    import pandas as pd
    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    prev_set_raw = timeline_prev.loc[
        timeline_prev["TEAM_NAME"].astype(str).str.strip() == team_name, "PLAYER_NAME"
    ].astype(str).str.strip()
    prev_set = set(prev_set_raw)

    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    aliases = {"robert williams": "robert williams iii", "enes freedom": "enes kanter",
               "juancho hernangomez": "juancho hernangomez"}
    if name_aliases: aliases.update({_norm(k): v for k, v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    prev_setN = {_apply_alias(x) for x in prev_set}

    tid = sg.get_team_id(team_name)
    d_from, d_to = _team_second_half_window_by_id(int(tid), prev_season) if tid else (None, None)

    df_team_prev = sg.getLeagueDashPlayerStats(
        team_name, prev_season,
        measure_type=measure_type, per_mode=per_mode,
        date_from=d_from, date_to=d_to
    )

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["PLAYER_NAME","GP","MIN","AST","TOV","_N","_AST_TOV"])
        out = df.copy()
        out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
        out["_N"] = out["PLAYER_NAME"].apply(_apply_alias)
        for c in ("GP","MIN","AST","TOV"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["_AST_TOV"] = out.get("AST",0.0) - out.get("TOV",0.0)
        return out

    t = _prep(df_team_prev)
    t = t[t["_N"].isin(prev_setN)]
    if "GP" in t.columns:  t = t[t["GP"]  >= gp_cut]
    if "MIN" in t.columns: t = t[t["MIN"] >= min_cut_per_game]

    return float(pd.to_numeric(t["_AST_TOV"], errors="coerce").sum())






def getTeamTotalMinutesLost_on_date(
    team_name: str,
    game_date: str,
    curr_season: str,
    prev_season: str,
    timeline_curr: pd.DataFrame,
    timeline_prev: pd.DataFrame,
    gp_cut: int = 30,
    min_cut: int = 14,
    per_mode: str = "PerGame",
    name_aliases: dict[str, str] | None = None,
) -> float:
    """
    Net TOTAL minutes lost as of `game_date`:
        net = minutes_lost_by_departures − minutes_added_by_arrivals

    - Rosters are derived from timelines (ON `game_date` for current season; full prev-season membership).
    - Lost side: prev-season TEAM Base, gates GP>=gp_cut & MIN>=min_cut.
      PerGame => sum(MIN * GP); otherwise => sum(MIN).
    - Added side:
        * VETS: prev-season LEAGUE Base with same gates + same minutes rule.
        * ROOKIES: current-season PRESEASON league Base (single pull), NO gates; same minutes rule.
    """
    import math, re, unicodedata
    import pandas as pd

    sg = stats_getter or __import__("stats_getter")

    # --- roster snapshots from timelines ---
    curr_set = roster_on_date(team_name, game_date, timeline_curr)
    prev_set = set(
        timeline_prev.loc[
            timeline_prev["TEAM_NAME"].astype(str).str.strip() == team_name, "PLAYER_NAME"
        ].astype(str).str.strip()
    )

    added = curr_set - prev_set
    lost  = prev_set - curr_set


    # --- name normalization + aliases (same as other helpers) ---
    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    aliases = {
        "robert williams": "robert williams iii",
        "juancho hernangomez": "juancho hernangomez",
        "enes freedom": "enes kanter",
    }
    if name_aliases:
        aliases.update({_norm(k): v for k, v in name_aliases.items()})

    def _apply_alias(n: str) -> str:
        nn = _norm(n)
        return _norm(aliases.get(nn, n))

    # --- cached, one-time per season pulls ---
    df_team_prev = sg.getLeagueDashPlayerStats(
        team_name, prev_season, measure_type="Base", per_mode=per_mode
    )
    df_league_prev = sg.getLeagueDashPlayerStats(
        "", prev_season, measure_type="Base", per_mode=per_mode
    )

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["PLAYER_NAME", "GP", "MIN", "_N"])
        out = df.copy()
        out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
        out["_N"] = out["PLAYER_NAME"].apply(_apply_alias)
        for c in ("GP", "MIN"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    team_prev   = _prep(df_team_prev)
    league_prev = _prep(df_league_prev)

    addedN = {_apply_alias(x) for x in added}
    lostN  = {_apply_alias(x) for x in lost}

    # --- minutes helper with optional GP/MIN gates ---
    def _minutes_sum(df: pd.DataFrame, names: set[str], apply_gates: bool) -> float:
        if not names or df is None or df.empty:
            return 0.0
        sub = df[df["_N"].isin(names)]
        if sub.empty:
            return 0.0
        if apply_gates:
            sub = sub[(sub["GP"] >= gp_cut) & (sub["MIN"] >= min_cut)]
            if sub.empty:
                return 0.0
        if isinstance(per_mode, str) and per_mode.lower() == "pergame":
            vals = sub["MIN"] * sub["GP"]
        else:
            vals = sub["MIN"]
        return float(vals.sum())

    # --- LOST: prev-season TEAM with gates ---
    minutes_lost = _minutes_sum(team_prev, lostN, apply_gates=True)

    # --- ADDED: split vets vs rookies using last year's league table ---
    prev_namesN = set(league_prev["_N"].unique())
    added_vetsN    = addedN & prev_namesN
    added_rookiesN = addedN - prev_namesN

    # VETS: prev-season LEAGUE with gates
    minutes_added_vets = _minutes_sum(league_prev, added_vetsN, apply_gates=True)

    # ROOKIES: current-season PRESEASON league, NO gates
    if added_rookiesN:
        df_pre = sg.getPreseasonLeagueDashPlayerStats(
            season=curr_season, measure_type="Base", per_mode=per_mode
        )
        pre = _prep(df_pre)
        minutes_added_rookies = _minutes_sum(pre, added_rookiesN, apply_gates=False)
    else:
        minutes_added_rookies = 0.0

    return minutes_lost - (minutes_added_vets + minutes_added_rookies)


def getStocks(team_name: str, season: str, cutoff_date: str) -> float:
    dte_cutoff = datetime.strptime(cutoff_date, "%m/%d/%Y")
    df_lg = stats_getter.get_league_game_log(season)
    df_lg["GAME_DATE"] = pd.to_datetime(df_lg["GAME_DATE"])

    team_id = stats_getter.get_team_id(team_name)
    if team_id is None:
        raise ValueError(f"Unknown team: {team_name}")

    # Filter games before the cutoff
    df_team = df_lg[
        (df_lg["TEAM_ID"] == team_id) &
        (df_lg["GAME_DATE"] < dte_cutoff)
    ]
    if df_team.empty:
        raise ValueError(f"{team_name} has no games before {cutoff_date} in {season}")

    stl_pg = df_team["STL"].mean()
    blk_pg = df_team["BLK"].mean()

    return stl_pg + blk_pg

import math
import pandas as pd
from stats_getter import get_team_id, get_league_game_log
from hustle_ledger import (
    get_prior_deflections_pg,
    get_prior_screen_assists_pg
)


from stats_getter import get_team_id
import math
import pandas as pd

# features_hustle.py (or wherever those functions live)

import math
from datetime import datetime
import pandas as pd

import stats_getter
import advanced_ledger as adv          # for possessions if you need per-100 or per-possession later
import hustle_ledger as hust           # NEW FILE above

def getStocksDeflections(team_name: str, season: str, date: str | pd.Timestamp):
    """
    Prior per-game (STOCKS + DEFLECTIONS) for a single team.
    Reuses the exact helper logic from stocksDeflectionDiff.
    """
    day = pd.to_datetime(date, errors="raise").normalize()

    team_id = stats_getter.get_team_id(team_name)
    if team_id is None:
        return math.nan

    # ---- helper copied from your stocksDeflectionDiff with no behavior changes ----
    def _stocks_prior_pg(season: str, team_id: int, cutoff_day: pd.Timestamp) -> float:
        lg = stats_getter.get_league_game_log(season)
        if lg is None or lg.empty:
            return math.nan
        df = lg.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
        dft = df[(df["TEAM_ID"].astype(int) == int(team_id)) & (df["GAME_DATE"] < cutoff_day)]
        if dft.empty:
            return math.nan
        stl = pd.to_numeric(dft.get("STL"), errors="coerce")
        blk = pd.to_numeric(dft.get("BLK"), errors="coerce")
        vals = (stl.add(blk, fill_value=0)).dropna()
        return float(vals.mean()) if not vals.empty else math.nan
    # -----------------------------------------------------------------------------

    stk = _stocks_prior_pg(season, team_id, day)
    dfl = hust.get_prior_deflections_pg(season, team_id, day)

    if any((x is None) or (isinstance(x, float) and math.isnan(x)) for x in [stk, dfl]):
        return math.nan
    return float(stk) + float(dfl)




def getScreenAssists(team_name: str, season: str, date: str | pd.Timestamp):
    """
    Prior per-game SCREEN_ASSISTS for a single team using hustle_ledger priors.
    Mirrors your original screenAssistDiff internals, but returns one team value.
    """
    day = pd.to_datetime(date, errors="raise").normalize()

    team_id = stats_getter.get_team_id(team_name)
    if team_id is None:
        return math.nan

    val = hust.get_prior_screen_assists_pg(season, team_id, day)
    if (val is None) or (isinstance(val, float) and math.isnan(val)):
        return math.nan
    return float(val)



import math
import pandas as pd

from stats_getter import (
    getLeagueHustleTeamStats,
)

# ---- last-season per-game values via your cached helper ----

def _prev_season_deflections_pg(team_name: str, last_season: str) -> float:
    tid = get_team_id(team_name)
    df = getLeagueHustleTeamStats(tid, last_season, per_mode="PerGame")
    if df.empty or "DEFLECTIONS" not in df.columns:
        return float("nan")
    v = pd.to_numeric(df["DEFLECTIONS"].iloc[0], errors="coerce")
    return float(v) if pd.notna(v) else float("nan")

def _prev_season_screen_ast_pg(team_name: str, last_season: str) -> float:
    tid = get_team_id(team_name)
    df = getLeagueHustleTeamStats(tid, last_season, per_mode="PerGame")
    if df.empty or "SCREEN_ASSISTS" not in df.columns:
        return float("nan")
    v = pd.to_numeric(df["SCREEN_ASSISTS"].iloc[0], errors="coerce")
    return float(v) if pd.notna(v) else float("nan")

# If you already have _prev_season_stocks_pg(team_name, current_season) that returns the prev-season (STL+BLK) pg,
# we'll reuse it below.

# ---- Mixed first-game fallback versions ----
# Use these *only when the normal feature came back NaN*.

# --- prev-season helpers used by *_first_game_mixed ---

from functools import lru_cache
import pandas as pd
import numpy as np
from stats_getter import get_league_game_log, getLeagueHustleTeamStats, get_team_id

def _prev_label(season: str) -> str | None:
    try:
        y0 = int(season.split("-")[0])
        return f"{y0-1}-{str(y0)[2:]}"   # '2019-20' -> '2018-19'
    except Exception:
        return None
    

@lru_cache(maxsize=64)
def _prev_season_stocks_pg(team_name: str, season: str) -> float:
    prev = _prev_label(season)
    if not prev:
        return np.nan
    df = get_league_game_log(prev)
    if df is None or df.empty:
        return np.nan
    t = df[df["TEAM_NAME"] == team_name]
    if t.empty:
        return np.nan
    stl = pd.to_numeric(t["STL"], errors="coerce")
    blk = pd.to_numeric(t["BLK"], errors="coerce")
    return float((stl + blk).mean())

@lru_cache(maxsize=64)
def _prev_season_deflections_pg(team_name: str, season: str) -> float:
    prev = _prev_label(season)
    if not prev:
        return np.nan
    tid = get_team_id(team_name)
    if not tid:
        return np.nan
    df = getLeagueHustleTeamStats(tid, prev, per_mode="PerGame")
    if df is None or df.empty or "DEFLECTIONS" not in df.columns:
        return np.nan
    return float(pd.to_numeric(df["DEFLECTIONS"], errors="coerce").iloc[0])

@lru_cache(maxsize=64)
def _prev_season_screen_ast_pg(team_name: str, season: str) -> float:
    prev = _prev_label(season)
    if not prev or prev == "2014-15":  # not available pre-2016-17
        return np.nan
    tid = get_team_id(team_name)
    if not tid:
        return np.nan
    df = getLeagueHustleTeamStats(tid, prev, per_mode="PerGame")
    if df is None or df.empty or "SCREEN_ASSISTS" not in df.columns:
        return np.nan
    return float(pd.to_numeric(df["SCREEN_ASSISTS"], errors="coerce").iloc[0])

# ---- DIFFs (home − away) ----
def adv_diff_first_game_mixed(home_team: str, away_team: str,
                              home_season: str, away_season: str,
                              date_str: str, col: str, prior_getter) -> float:
    h = _prior_or_prev(home_team, home_season, date_str, col, prior_getter)
    a = _prior_or_prev(away_team, away_season, date_str, col, prior_getter)
    if any(np.isnan([h, a])): return np.nan
    return h - a

# ---- RATIOS (home / away) ----
def adv_ratio_first_game_mixed(home_team: str, away_team: str,
                               home_season: str, away_season: str,
                               date_str: str, col: str, prior_getter) -> float:
    h = _prior_or_prev(home_team, home_season, date_str, col, prior_getter)
    a = _prior_or_prev(away_team, away_season, date_str, col, prior_getter)
    
    if any(np.isnan([h, a])) or a == 0: return np.nan
    return h / a

# Convenience wrappers for the specific stats you asked for
def pace_ratio_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_ratio_first_game_mixed(home, away, h_season, a_season, dstr,
                                      "PACE", get_prior_pace)

def net_rating_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "NET_RATING", get_prior_net_rating)

def dreb_pct_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "DREB_PCT", get_prior_dreb_pct)

def oreb_pct_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "OREB_PCT", get_prior_oreb_pct)

def poss_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "POSS", get_prior_poss)

def tm_tov_pct_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "TM_TOV_PCT", get_prior_tm_tov_pct)

def efg_pct_diff_first_game_mixed(home, away, h_season, a_season, dstr):
    return adv_diff_first_game_mixed(home, away, h_season, a_season, dstr,
                                     "EFG_PCT", get_prior_efg_pct)

# advanced_accessors.py  (or wherever you keep feature helpers)

import pandas as pd
from stats_getter import get_team_id
from advanced_ledger import (
    get_prior_net_rating, get_prior_poss, get_prior_tm_tov_pct, get_prior_efg_pct
)

def _to_day(x):
    import pandas as pd, numpy as np
    v = pd.to_datetime(x, errors="coerce")
    if isinstance(v, (pd.Series, pd.Index, pd.DatetimeIndex, np.ndarray)):
        if isinstance(v, pd.DatetimeIndex):
            return v.normalize()
        if not isinstance(v, pd.Series):
            v = pd.Series(v)
        return v.dt.normalize()
    return pd.NaT if pd.isna(v) else pd.Timestamp(v).normalize()



def get_net_rating_diff(home_team, away_team, home_season, away_season, date_str) -> float:
    d = _to_day(date_str)
    hid, aid = get_team_id(home_team), get_team_id(away_team)
    return get_prior_net_rating(home_season, hid, d) - get_prior_net_rating(away_season, aid, d)

def get_poss_diff(home_team, away_team, home_season, away_season, date_str) -> float:
    d = _to_day(date_str)
    hid, aid = get_team_id(home_team), get_team_id(away_team)
    return get_prior_poss(home_season, hid, d) - get_prior_poss(away_season, aid, d)

def get_turnover_pct_relative(home_team, away_team, home_season, away_season, date_str) -> float:
    d = _to_day(date_str)
    hid, aid = get_team_id(home_team), get_team_id(away_team)
    return get_prior_tm_tov_pct(home_season, hid, d) - get_prior_tm_tov_pct(away_season, aid, d)

def get_prior_efg_pct_relative(home_team, away_team, home_season, away_season, date_str) -> float:
    d = _to_day(date_str)
    hid, aid = get_team_id(home_team), get_team_id(away_team)
    return get_prior_efg_pct(home_season, hid, d) - get_prior_efg_pct(away_season, aid, d)

# roster_timeline.py
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog as _LeagueGameLog

def build_team_roster_timeline(season: str) -> pd.DataFrame:
    """
    Build a league-wide roster timeline for `season` using PLAYER game logs.

    For each (TEAM_NAME, PLAYER_NAME), compute:
      - first_game_date: the player's first appearance date for that team
      - last_game_date:  the player's last appearance date for that team

    Returns
    -------
    pd.DataFrame with columns:
      ["TEAM_NAME", "PLAYER_NAME", "first_game_date", "last_game_date"]
      TEAM_NAME is canonicalized (e.g., "LA Clippers"), dates are pandas Timestamps.
    """
    import pandas as pd

    # Pull player game logs for the season (your helper; P = player logs)
    logs = get_league_game_log(season=season, player_or_team_abbreviation="P")

    # Keep only what we need
    df = logs.loc[:, ["TEAM_NAME", "PLAYER_NAME", "GAME_DATE"]].copy()

    # Clean strings
    df["TEAM_NAME"] = df["TEAM_NAME"].astype(str).str.strip()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()

    # Normalize team names so "Los Angeles Clippers" == "LA Clippers"
    # (expects canon_df_team_names() to be defined in this module)
    canon_df_team_names(df, "TEAM_NAME")

    # Coerce dates
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Aggregate to first/last appearance per team-player
    out = (
        df.groupby(["TEAM_NAME", "PLAYER_NAME"], as_index=False)["GAME_DATE"]
          .agg(first_game_date="min", last_game_date="max")
          .sort_values(["TEAM_NAME", "PLAYER_NAME"])
          .reset_index(drop=True)
    )

    return out


def getTeam_PlayerAggregated_PPG_last_year(
    team_name: str,
    prev_season: str,
    timeline_prev: pd.DataFrame,
    gp_cut: int = 13,
    min_cut_per_game: float = 12.0,
    name_aliases: dict[str, str] | None = None,
    measure_type: str = "Base",
    per_mode: str = "PerGame",
) -> float:
    import re, unicodedata
    import pandas as pd
    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)



    # build roster set from last season's timeline (same as before)
    prev_set_raw = timeline_prev.loc[
        timeline_prev["TEAM_NAME"].astype(str).str.strip() == team_name, "PLAYER_NAME"
    ].astype(str).str.strip()
    prev_set = set(prev_set_raw)

    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    aliases = {"robert williams": "robert williams iii", "enes freedom": "enes kanter",
               "juancho hernangomez": "juancho hernangomez"}
    if name_aliases: aliases.update({_norm(k): v for k, v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    prev_setN = {_apply_alias(x) for x in prev_set}

    # --- limit to the TEAM's second half of prev season
    tid = sg.get_team_id(team_name)
    d_from, d_to = _team_second_half_window_by_id(int(tid), prev_season) if tid else (None, None)

    df_team_prev = sg.getLeagueDashPlayerStats(
        team_name, prev_season,
        measure_type=measure_type, per_mode=per_mode,
        date_from=d_from, date_to=d_to
    )

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["PLAYER_NAME", "GP", "MIN", "PTS", "_N"])
        out = df.copy()
        out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
        out["_N"] = out["PLAYER_NAME"].apply(_apply_alias)
        for c in ("GP", "MIN", "PTS"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    team_prev = _prep(df_team_prev)
    team_prev = team_prev[team_prev["_N"].isin(prev_setN)]

    # gates
    if "GP" in team_prev.columns:
        team_prev = team_prev[team_prev["GP"] >= gp_cut]
    if "MIN" in team_prev.columns:
        team_prev = team_prev[team_prev["MIN"] >= min_cut_per_game]

    return float(pd.to_numeric(team_prev.get("PTS", 0.0), errors="coerce").sum())






import math




# -----------------------------------
# Public feature getters (8 total)
# Names match your feature list
# -----------------------------------

def contested_3pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_rate("contested3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def contested_3pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_efficiency("contested3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def open_3pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_rate("open3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def open_3pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_efficiency("open3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def corner_3pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
    *,
    season_type_all_star: str = "Regular Season",
) -> float:
    return get_shotloc_rate("corner3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str,
                                  season_type_all_star=season_type_all_star)

def corner_3pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_shotloc_efficiency("corner3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def above_break_3pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
    *,
    season_type_all_star: str = "Regular Season",
) -> float:
    return get_shotloc_rate("above_the_break3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str,
                                  season_type_all_star=season_type_all_star)

def above_break_3pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_shotloc_efficiency("above_thebreak3p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def paint_shot_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
    *,
    season_type_all_star: str = "Regular Season",
) -> float:
    return get_shotloc_rate("paintshots",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str,
                                  season_type_all_star=season_type_all_star)

def paint_shot_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_shotloc_efficiency("paintshots",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)


def midrange_shot_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
    *,
    season_type_all_star: str = "Regular Season",
) -> float:
    return get_shotloc_rate("midrangeshots",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str,
                                  season_type_all_star=season_type_all_star)

def midrange_shot_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_shotloc_efficiency("midrangeshots",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def contested_2pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_rate("contested2p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def contested_2pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_efficiency("contested2p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def open_2pt_rate(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_rate("open2p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

def open_2pt_efficiency(
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
) -> float:
    return get_ptshots_efficiency("open2p",
                                  team_a_name, team_b_name,
                                  team_a_season, team_b_season,
                                  date_str)

# ---- small helpers ----
def _mi_get(df, cat, stat, default=0.0):
    """Read a value from a MultiIndex column (cat, stat)."""
    try:
        val = df[(cat, stat)].iloc[0]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except Exception:
        return default

def _mi_get_any(df, cat, stat_candidates, default=0.0):
    """Try multiple possible stat names (e.g., 'FGA' vs 'OPP_FGA')."""
    for s in stat_candidates:
        if (cat, s) in getattr(df, "columns", []):
            return _mi_get(df, cat, s, default)
    return default


import pandas as pd, numpy as np
from datetime import datetime

# AGGREGATED PLAYER STATS ON ROSTER AS OF DATE

# === helper used by all three fallbacks (kept local for minimal diff) ===
def _season_string_prev(season: str) -> str:
    # "2022-23" -> "2021-22"
    y1 = int(season.split("-")[0]) - 1
    y2 = (y1 + 1) % 100
    return f"{y1}-{y2:02d}"

def _season_string_prev2(season: str) -> str:
    return _season_string_prev(_season_string_prev(season))

def _prep_dash(df, apply_alias):
    import pandas as pd
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "PLAYER_NAME","GP","MIN","PTS","AST","TOV","DREB","STL","BLK","_N"
        ])
    out = df.copy()
    out["PLAYER_NAME"] = out["PLAYER_NAME"].astype(str).str.strip()
    out["_N"] = out["PLAYER_NAME"].apply(apply_alias)
    for c in ("GP","MIN","PTS","AST","TOV","DREB","STL","BLK"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def _player_prior_value(playerN, df_prev, df_prev2, metric_cols, gp_cut, min_cut_pg):
    """
    metric_cols: list of column names to sum (per-game columns). e.g.
      ["PTS"] or ["AST","-TOV"] or ["DREB","STL","BLK"] (use "-COL" to subtract)
    Returns a single float for this player following the fallback rule.
    """
    import numpy as np
    def _row_val(row):
        total = 0.0
        for col in metric_cols:
            if col.startswith("-"):
                total -= float(row[col[1:]])
            else:
                total += float(row[col])
        return total

    r1 = df_prev.loc[df_prev["_N"] == playerN]
    if not r1.empty:
        gp1 = float(r1["GP"].iloc[0])
        min1 = float(r1["MIN"].iloc[0])
        if (gp1 >= gp_cut) and (min1 >= min_cut_pg):
            return _row_val(r1.iloc[0])
        # else: only try prev2 if MIN passes but GP fails
        if (min1 >= min_cut_pg) and (gp1 < gp_cut) and (df_prev2 is not None and not df_prev2.empty):
            r2 = df_prev2.loc[df_prev2["_N"] == playerN]
            if not r2.empty:
                gp2 = float(r2["GP"].iloc[0])
                min2 = float(r2["MIN"].iloc[0])
                if (gp2 >= gp_cut) and (min2 >= min_cut_pg):
                    return _row_val(r2.iloc[0])
    return 0.0

# First game fallback for this obtains the players playing the first game using roster_on_date(),
# and then obtains their prior-season per-game stats from the previous season's second half.
def getTeam_PlayerAggregated_PPG_on_date(
    team_name: str,
    game_date: str,                 # "MM/DD/YYYY"
    curr_season: str,               # "2022-23"
    timeline_curr: pd.DataFrame,
    name_aliases: dict[str, str] | None = None,
    # dynamic threshold controls:
    dynamic_after_team_games: int = 12,
    early_gp_cut: int = 1,
    late_gp_cut: int = 3,
    late_min_minutes_total: float = 50.0,
    # fallback gates:
    fallback_gp_cut: int = 30,
    fallback_min_cut_pg: float = 12.0,
) -> float:
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)


    # --- active roster on date ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)


    # --- aliasing ---
    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",s)
        s = re.sub(r"\s+"," ",s).strip()
        return s
    aliases = {
        "robert williams":"robert williams iii",
        "juancho hernangomez":"juancho hernangomez",
        "enes freedom":"enes kanter",
    }
    if name_aliases:
        aliases.update({_norm(k): v for k,v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- player logs current season up to day BEFORE game ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        team_games_to_date = 0
    else:
        df = logs.copy()
        df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
        df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
        df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
        cutoff = datetime.strptime(game_date, "%m/%d/%Y")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
        team_games_to_date = int(df["GAME_ID"].nunique()) if not df.empty else 0

        
     # ---- FIRST GAME FALLBACK ----
    if team_games_to_date == 0:
        GP_GATE = float(fallback_gp_cut) if fallback_gp_cut is not None else 10.0
        MIN_GATE = float(fallback_min_cut_pg) if fallback_min_cut_pg is not None else 12.0

        # lock to 10 GP as you requested
        GP_GATE = 10.0

        prev1 = _season_string_prev(curr_season)

        # --- build 2nd-half window for prev1 ---
        try:
            _tid = sg.get_team_id(team_name)
            d_from, d_to = _second_half_window_by_id(_tid, prev1)
        except Exception:
            d_from = pd.Timestamp(f"02/01/{int(prev1[:4]) + 1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev1[:4]) + 1}")

        def _fmt_mdY(x):
            return pd.to_datetime(x, errors="coerce").strftime("%m/%d/%Y")
        d_from_s, d_to_s = _fmt_mdY(d_from), _fmt_mdY(d_to)

        # --- prev1 stats (per-game) ---
        df_prev1 = sg.getLeagueDashPlayerStats(
            "", prev1, measure_type="Base", per_mode="PerGame",
            date_from=d_from_s, date_to=d_to_s
        )

        df_prev1 = sg.getLeagueDashPlayerStats("", prev1, measure_type="Base", per_mode="PerGame",date_from=d_from_s, date_to=d_to_s)
        if df_prev1 is None:
            df_prev1 = pd.DataFrame()
        else:
            df_prev1 = df_prev1.copy()


        if not df_prev1.empty:
            df_prev1 = df_prev1[df_prev1["PLAYER_NAME"].notna()].copy()
            df_prev1["PLAYER_NAME"] = df_prev1["PLAYER_NAME"].astype(str).str.strip()
            df_prev1["_N"] = df_prev1["PLAYER_NAME"].apply(_apply_alias)
            for c in ("GP","MIN","PTS"):
                if c in df_prev1.columns:
                    df_prev1[c] = pd.to_numeric(df_prev1[c], errors="coerce")

        # keep only today's roster
        prev1_roster = df_prev1[df_prev1.get("_N", "").isin(activeN)][["_N","GP","MIN","PTS"]] if not df_prev1.empty else pd.DataFrame(columns=["_N","GP","MIN","PTS"])

        # *** STRICT prev1 rule: require GP >= 10; do NOT accept MPG-only ***
        priors_prev1_ok = prev1_roster.loc[(prev1_roster["GP"].fillna(0) >= GP_GATE)].copy()

        # Who still needs help from prev2? (either missing in prev1 OR had GP < 10 there)
        names_prev1_ok = set(priors_prev1_ok["_N"]) if not priors_prev1_ok.empty else set()
        names_needing_prev2 = (set(activeN) - names_prev1_ok)

        # --- prev2 window (two seasons ago) ---
        prev2 = _season_string_prev(prev1)
        try:
            _tid2 = sg.get_team_id(team_name)
            d2_from, d2_to = _second_half_window_by_id(_tid2, prev2)
        except Exception:
            d2_from = pd.Timestamp(f"02/01/{int(prev2[:4]) + 1}")
            d2_to   = pd.Timestamp(f"07/15/{int(prev2[:4]) + 1}")
        d2_from_s, d2_to_s = _fmt_mdY(d2_from), _fmt_mdY(d2_to)

        df_prev2 = sg.getLeagueDashPlayerStats(
            "", prev2, measure_type="Base", per_mode="PerGame",
            date_from=d2_from_s, date_to=d2_to_s
        )
        df_prev1 = sg.getLeagueDashPlayerStats("", prev1, measure_type="Base", per_mode="PerGame",date_from=d_from_s, date_to=d_to_s)
        if df_prev1 is None:
            df_prev1 = pd.DataFrame()
        else:
            df_prev1 = df_prev1.copy()

        if not df_prev2.empty:
            df_prev2 = df_prev2[df_prev2["PLAYER_NAME"].notna()].copy()
            df_prev2["PLAYER_NAME"] = df_prev2["PLAYER_NAME"].astype(str).str.strip()
            df_prev2["_N"] = df_prev2["PLAYER_NAME"].apply(_apply_alias)
            for c in ("GP","MIN","PTS"):
                if c in df_prev2.columns:
                    df_prev2[c] = pd.to_numeric(df_prev2[c], errors="coerce")
            # restrict to *only* those who still need prev2
            df_prev2 = df_prev2[df_prev2["_N"].isin(names_needing_prev2)]

        # *** prev2 rule: accept if GP >= 10 OR MPG >= 12 ***
        if not df_prev2.empty:
            prev2_ok = df_prev2.loc[
                (df_prev2["GP"].fillna(0) >= GP_GATE) | (df_prev2["MIN"].fillna(0) >= MIN_GATE),
                ["_N","GP","MIN","PTS"]
            ].copy()
        else:
            prev2_ok = pd.DataFrame(columns=["_N","GP","MIN","PTS"])

        # Combine: prev1_ok plus prev2_ok (add missing or overwrite if somehow duplicated)
        if priors_prev1_ok.empty:
            priors = prev2_ok.copy()
        elif prev2_ok.empty:
            priors = priors_prev1_ok.copy()
        else:
            # if any dupes, prefer prev1_ok (shouldn't usually happen due to names_needing_prev2)
            priors = pd.concat([priors_prev1_ok, prev2_ok[~prev2_ok["_N"].isin(names_prev1_ok)]], ignore_index=True)

        # Sum per-game PTS
        if priors is None or priors.empty:
            return 0.0
        
        return float(np.nansum(priors["PTS"].to_numpy()))
    
    # ---- Regular path (unchanged) ----
    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all='ignore'):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0)/60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan

    g = df.groupby("_N", as_index=False).agg(
        GP=("GAME_ID","count"), PTS=("PTS","sum"), MIN_TOT=("__MIN_F","sum")
    )
    g["PPG"] = g["PTS"] / g["GP"].replace(0, np.nan)

    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut
    qual_gp = g["GP"] >= gp_cut
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    g = g[(g["_N"].isin(activeN)) & (qualifier)]
    if g.empty:
        return 0.0
    return float(np.nansum(g["PPG"].to_numpy()))

def getTeam_PlayerAggregated_ASTTOV_on_date(
    team_name: str,
    game_date: str,
    curr_season: str,
    timeline_curr: pd.DataFrame,
    name_aliases: dict[str, str] | None = None,
    dynamic_after_team_games: int = 12,
    early_gp_cut: int = 1,
    late_gp_cut: int = 3,
    late_min_minutes_total: float = 36.0,
    # fallback gates:
    fallback_gp_cut: int = 30,
    fallback_min_cut_pg: float = 12.0,
) -> float:
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    # --- active roster on date ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)

    # --- aliasing (same map you use elsewhere) ---
    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",s)
        s = re.sub(r"\s+"," ",s).strip()
        return s
    aliases = {
        "robert williams":"robert williams iii",
        "juancho hernangomez":"juancho hernangomez",
        "enes freedom":"enes kanter",
    }
    if name_aliases:
        aliases.update({_norm(k): v for k,v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- player logs current season up to day BEFORE game ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        team_games_to_date = 0
    else:
        df = logs.copy()
        df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
        df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
        df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
        cutoff = datetime.strptime(game_date, "%m/%d/%Y")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
        team_games_to_date = int(df["GAME_ID"].nunique()) if not df.empty else 0

    # ---- FIRST GAME FALLBACK (mirror PPG logic) ----
    if team_games_to_date == 0:
        # Gates: strict 10 GP for prev1; prev2 allows (GP>=10 OR MPG>=12)
        GP_GATE = 10.0
        MIN_GATE = float(fallback_min_cut_pg) if fallback_min_cut_pg is not None else 12.0

        prev1 = _season_string_prev(curr_season)

        # --- build 2nd-half window for prev1 ---
        try:
            _tid = sg.get_team_id(team_name)
            d_from, d_to = _second_half_window_by_id(_tid, prev1)
        except Exception:
            d_from = pd.Timestamp(f"02/01/{int(prev1[:4]) + 1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev1[:4]) + 1}")

        def _fmt_mdY(x):
            return pd.to_datetime(x, errors="coerce").strftime("%m/%d/%Y")
        d_from_s, d_to_s = _fmt_mdY(d_from), _fmt_mdY(d_to)

        # --- prev1 stats (PerGame) ---
        df_prev1 = sg.getLeagueDashPlayerStats(
            "", prev1, measure_type="Base", per_mode="PerGame",
            date_from=d_from_s, date_to=d_to_s
        )
        if df_prev1 is None:
            df_prev1 = pd.DataFrame()
        else:
            df_prev1 = df_prev1.copy()

        if not df_prev1.empty:
            df_prev1 = df_prev1[df_prev1["PLAYER_NAME"].notna()].copy()
            df_prev1["PLAYER_NAME"] = df_prev1["PLAYER_NAME"].astype(str).str.strip()
            df_prev1["_N"] = df_prev1["PLAYER_NAME"].apply(_apply_alias)
            # numeric cast
            for c in ("GP","MIN","AST","TOV"):
                if c in df_prev1.columns:
                    df_prev1[c] = pd.to_numeric(df_prev1[c], errors="coerce")

        # keep only today's roster, relevant cols
        prev1_roster = (
            df_prev1[df_prev1.get("_N", "").isin(activeN)][["_N","GP","MIN","AST","TOV"]]
            if not df_prev1.empty else
            pd.DataFrame(columns=["_N","GP","MIN","AST","TOV"])
        )

        # STRICT prev1 rule: require GP >= 10 (no MPG-only)
        priors_prev1_ok = prev1_roster.loc[(prev1_roster["GP"].fillna(0) >= GP_GATE)].copy()

        # who still needs prev2? (not in prev1_ok)
        names_prev1_ok = set(priors_prev1_ok["_N"]) if not priors_prev1_ok.empty else set()
        names_needing_prev2 = (set(activeN) - names_prev1_ok)

        # --- prev2 window (two seasons ago) ---
        prev2 = _season_string_prev(prev1)
        try:
            _tid2 = sg.get_team_id(team_name)
            d2_from, d2_to = _second_half_window_by_id(_tid2, prev2)
        except Exception:
            d2_from = pd.Timestamp(f"02/01/{int(prev2[:4]) + 1}")
            d2_to   = pd.Timestamp(f"07/15/{int(prev2[:4]) + 1}")
        d2_from_s, d2_to_s = _fmt_mdY(d2_from), _fmt_mdY(d2_to)

        df_prev2 = sg.getLeagueDashPlayerStats(
            "", prev2, measure_type="Base", per_mode="PerGame",
            date_from=d2_from_s, date_to=d2_to_s
        )
        if df_prev2 is None:
            df_prev2 = pd.DataFrame()
        else:
            df_prev2 = df_prev2.copy()

        if not df_prev2.empty:
            df_prev2 = df_prev2[df_prev2["PLAYER_NAME"].notna()].copy()
            df_prev2["PLAYER_NAME"] = df_prev2["PLAYER_NAME"].astype(str).str.strip()
            df_prev2["_N"] = df_prev2["PLAYER_NAME"].apply(_apply_alias)
            for c in ("GP","MIN","AST","TOV"):
                if c in df_prev2.columns:
                    df_prev2[c] = pd.to_numeric(df_prev2[c], errors="coerce")
            # restrict to names that still need prev2
            df_prev2 = df_prev2[df_prev2["_N"].isin(names_needing_prev2)]

        # prev2 rule: accept if GP >= 10 OR MPG >= 12
        if not df_prev2.empty:
            prev2_ok = df_prev2.loc[
                (df_prev2["GP"].fillna(0) >= GP_GATE) | (df_prev2["MIN"].fillna(0) >= MIN_GATE),
                ["_N","GP","MIN","AST","TOV"]
            ].copy()
        else:
            prev2_ok = pd.DataFrame(columns=["_N","GP","MIN","AST","TOV"])

        # Combine: prev1_ok plus prev2_ok (add missing)
        if priors_prev1_ok.empty:
            priors = prev2_ok.copy()
        elif prev2_ok.empty:
            priors = priors_prev1_ok.copy()
        else:
            priors = pd.concat(
                [priors_prev1_ok, prev2_ok[~prev2_ok["_N"].isin(names_prev1_ok)]],
                ignore_index=True
            )

        if priors is None or priors.empty:
            return 0.0
        

        # Sum per-game (AST - TOV) across qualified roster
        return float(np.nansum((priors["AST"] - priors["TOV"]).to_numpy()))

    # ---- Regular path (unchanged) ----
    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    team_games_to_date = int(df["GAME_ID"].nunique())
    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all="ignore"):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0)/60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan

    g = df.groupby("_N", as_index=False).agg(
        GP=("GAME_ID","count"),
        AST_SUM=("AST","sum"),
        TOV_SUM=("TOV","sum"),
        MIN_TOT=("__MIN_F","sum"),
    )
    gp = g["GP"].replace(0, np.nan)
    g["ASTmTOV_PG"] = (g["AST_SUM"]/gp) - (g["TOV_SUM"]/gp)

    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut
    qual_gp = g["GP"] >= gp_cut
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    g = g[(g["_N"].isin(activeN)) & qualifier]
    if g.empty:
        return 0.0

    return float(np.nansum(g["ASTmTOV_PG"].to_numpy()))


def getTeam_PlayerAggregated_DREBSTOCK_on_date(
    team_name: str,
    game_date: str,
    curr_season: str,
    timeline_curr: pd.DataFrame,
    name_aliases: dict[str, str] | None = None,
    min_games: int = 1,  # legacy, ignored
    dynamic_after_team_games: int = 12,
    early_gp_cut: int = 1,
    late_gp_cut: int = 4,
    late_min_minutes_total: float = 120.0,
    # fallback gates:
    fallback_gp_cut: int = 30,
    fallback_min_cut_pg: float = 12.0,
) -> float:
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    
    # --- active roster ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)

    def _norm(n: str) -> str:
        s = unicodedata.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
        s = s.lower().replace(".", "").replace("'", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b","",s)
        s = re.sub(r"\s+"," ",s).strip()
        return s
    aliases = {
        "robert williams":"robert williams iii",
        "juancho hernangomez":"juancho hernangomez",
        "enes freedom":"enes kanter",
    }
    if name_aliases:
        aliases.update({_norm(k): v for k,v in name_aliases.items()})
    def _apply_alias(n: str) -> str:
        nn = _norm(n); return _norm(aliases.get(nn, n))
    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- logs current season ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        team_games_to_date = 0
    else:
        df = logs.copy()
        df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
        df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
        df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
        cutoff = datetime.strptime(game_date, "%m/%d/%Y")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
        team_games_to_date = int(df["GAME_ID"].nunique()) if not df.empty else 0

    # ---- FIRST GAME FALLBACK ----
    if team_games_to_date == 0:
        GP_GATE = 10.0  # strict
        MIN_GATE = float(fallback_min_cut_pg) if fallback_min_cut_pg is not None else 12.0

        prev1 = _season_string_prev(curr_season)
        # prev1 2nd half window
        try:
            _tid = sg.get_team_id(team_name)
            d_from, d_to = _second_half_window_by_id(_tid, prev1)
        except Exception:
            d_from = pd.Timestamp(f"02/01/{int(prev1[:4]) + 1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev1[:4]) + 1}")

        def _fmt_mdY(x):
            return pd.to_datetime(x, errors="coerce").strftime("%m/%d/%Y")
        d_from_s, d_to_s = _fmt_mdY(d_from), _fmt_mdY(d_to)

        df_prev1 = sg.getLeagueDashPlayerStats(
            "", prev1, measure_type="Base", per_mode="PerGame",
            date_from=d_from_s, date_to=d_to_s
        )
        if df_prev1 is None:
            df_prev1 = pd.DataFrame()
        else:
            df_prev1 = df_prev1.copy()
        if not df_prev1.empty:
            df_prev1 = df_prev1[df_prev1["PLAYER_NAME"].notna()].copy()
            df_prev1["PLAYER_NAME"] = df_prev1["PLAYER_NAME"].astype(str).str.strip()
            df_prev1["_N"] = df_prev1["PLAYER_NAME"].apply(_apply_alias)
            for c in ("GP","MIN","DREB","STL","BLK"):
                if c in df_prev1.columns:
                    df_prev1[c] = pd.to_numeric(df_prev1[c], errors="coerce")

        prev1_roster = (
            df_prev1[df_prev1.get("_N", "").isin(activeN)][["_N","GP","MIN","DREB","STL","BLK"]]
            if not df_prev1.empty else
            pd.DataFrame(columns=["_N","GP","MIN","DREB","STL","BLK"])
        )
        priors_prev1_ok = prev1_roster.loc[(prev1_roster["GP"].fillna(0) >= GP_GATE)].copy()

        names_prev1_ok = set(priors_prev1_ok["_N"]) if not priors_prev1_ok.empty else set()
        names_needing_prev2 = (set(activeN) - names_prev1_ok)

        # prev2 2nd half window
        prev2 = _season_string_prev(prev1)
        try:
            _tid2 = sg.get_team_id(team_name)
            d2_from, d2_to = _second_half_window_by_id(_tid2, prev2)
        except Exception:
            d2_from = pd.Timestamp(f"02/01/{int(prev2[:4]) + 1}")
            d2_to   = pd.Timestamp(f"07/15/{int(prev2[:4]) + 1}")
        d2_from_s, d2_to_s = _fmt_mdY(d2_from), _fmt_mdY(d2_to)

        df_prev2 = sg.getLeagueDashPlayerStats(
            "", prev2, measure_type="Base", per_mode="PerGame",
            date_from=d2_from_s, date_to=d2_to_s
        )
        if df_prev2 is None:
            df_prev2 = pd.DataFrame()
        else:
            df_prev2 = df_prev2.copy()
        if not df_prev2.empty:
            df_prev2 = df_prev2[df_prev2["PLAYER_NAME"].notna()].copy()
            df_prev2["PLAYER_NAME"] = df_prev2["PLAYER_NAME"].astype(str).str.strip()
            df_prev2["_N"] = df_prev2["PLAYER_NAME"].apply(_apply_alias)
            for c in ("GP","MIN","DREB","STL","BLK"):
                if c in df_prev2.columns:
                    df_prev2[c] = pd.to_numeric(df_prev2[c], errors="coerce")
            df_prev2 = df_prev2[df_prev2["_N"].isin(names_needing_prev2)]

        if not df_prev2.empty:
            prev2_ok = df_prev2.loc[
                (df_prev2["GP"].fillna(0) >= GP_GATE) | (df_prev2["MIN"].fillna(0) >= MIN_GATE),
                ["_N","GP","MIN","DREB","STL","BLK"]
            ].copy()
        else:
            prev2_ok = pd.DataFrame(columns=["_N","GP","MIN","DREB","STL","BLK"])

        if priors_prev1_ok.empty:
            priors = prev2_ok.copy()
        elif prev2_ok.empty:
            priors = priors_prev1_ok.copy()
        else:
            priors = pd.concat(
                [priors_prev1_ok, prev2_ok[~prev2_ok["_N"].isin(names_prev1_ok)]],
                ignore_index=True
            )

        if priors is None or priors.empty:
            return 0.0


        # Sum per-game (DREB+STL+BLK)
        return float(np.nansum((priors["DREB"] + priors["STL"] + priors["BLK"]).to_numpy()))

    # ---- Regular path ----
    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    team_games_to_date = int(df["GAME_ID"].nunique())

    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all="ignore"):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0)/60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan

    g = df.groupby("_N", as_index=False).agg(
        GP=("GAME_ID","count"),
        DREB_SUM=("DREB","sum"),
        STL_SUM=("STL","sum"),
        BLK_SUM=("BLK","sum"),
        MIN_TOT=("__MIN_F","sum"),
    )
    gp = g["GP"].replace(0, np.nan)
    g["DREB_STOCK_PG"] = (g["DREB_SUM"] + g["STL_SUM"] + g["BLK_SUM"]) / gp

    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut
    qual_gp = g["GP"] >= gp_cut
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    g = g[(g["_N"].isin(activeN)) & qualifier]
    if g.empty:
        return 0.0

    return float(np.nansum(g["DREB_STOCK_PG"].to_numpy()))



def _team_second_half_window_by_id(team_id: int, season: str) -> tuple[str | None, str | None]:
    """
    For a given team (TEAM_ID) and NBA season key (e.g., '2023-24'),
    return (date_from, date_to) that spans ONLY the team's second half
    of the regular season, using the cached LeagueGameLog.
    """
    import pandas as pd
    try:
        import stats_getter as sg
    except Exception:
        sg = __import__("stats_getter")

    log = sg.get_league_game_log(season)
    if log is None or log.empty or "TEAM_ID" not in log.columns or "GAME_DATE" not in log.columns:
        return (None, None)

    # keep only this team's regular-season rows
    df = log.copy()
    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"].str.contains("Regular", case=False, na=False)]

    df = df[df["TEAM_ID"].astype(int) == int(team_id)].copy()
    if df.empty:
        return (None, None)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.normalize()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # 2nd half starts at the ceil(N/2)-th index (0-based)
    n = len(df)
    split_idx = n // 2
    start = df.iloc[split_idx]["GAME_DATE"]
    end   = df.iloc[-1]["GAME_DATE"]

    return (start.strftime("%m/%d/%Y"), end.strftime("%m/%d/%Y"))

# --- Player-slot ON/OFF (ON-court) metrics via TeamPlayerOnOffDetails ----------
# Dependencies used here:
#   - stats_getter.season_for_date_smart
#   - stats_getter.get_league_game_log
#   - stats_getter.get_team_id
#   - stats_getter.getTeamPlayerOnOffDetails
#   - get_roster_timeline(season) and roster_on_date(timeline, date_str)
# These exist elsewhere in your repo per prior helpers.

import math
import pandas as pd
from datetime import datetime, timedelta
import stats_getter as sg
from cache_manager import stats_cache

# Resolve the day BEFORE a game (cutoff for "to date" stats)
def _cutoff(date_str: str) -> str:
    d = datetime.strptime(date_str, "%m/%d/%Y") - timedelta(days=1)
    return d.strftime("%m/%d/%Y")

def _second_half_window_by_id(team_id: int, season: str) -> tuple[str | None, str | None]:
    """Compute (date_from, date_to) covering the 2nd half of the team's regular-season games."""
    try:
        gl = sg.get_league_game_log(season).copy()
    except Exception:
        return None, None
    if gl is None or gl.empty:
        return None, None
    gl = gl[pd.to_numeric(gl["TEAM_ID"], errors="coerce") == int(team_id)]
    if gl.empty:
        return None, None
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce").dt.normalize()
    gl = gl.sort_values(["GAME_DATE", "GAME_ID"], kind="mergesort").reset_index(drop=True)
    n = len(gl)
    split = max(0, n // 2)
    if n == 0:
        return None, None
    date_from = gl.iloc[split]["GAME_DATE"].strftime("%m/%d/%Y")
    date_to   = gl.iloc[-1]["GAME_DATE"].strftime("%m/%d/%Y")
    return date_from, date_to

def _minutes_per_game_to_date(player_id: int, team_id: int, season: str, date_str: str) -> float:
    """Mean MIN for this player for this TEAM up to date_str (inclusive)."""
    try:
        lg = sg.get_league_game_log(season, player_or_team_abbreviation="P").copy()
    except Exception:
        return 0.0
    if lg is None or lg.empty:
        return 0.0
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"], errors="coerce").dt.normalize()
    cutoff = pd.to_datetime(date_str, errors="coerce").normalize()
    # Filter both by player and team_id (handles mid-season trades)
    sub = lg[(pd.to_numeric(lg["PLAYER_ID"], errors="coerce") == int(player_id)) &
             (pd.to_numeric(lg["TEAM_ID"],   errors="coerce") == int(team_id)) &
             (lg["GAME_DATE"] <= cutoff)]
    if sub.empty or "MIN" not in sub:
        return 0.0
    # nba_api generally returns MIN as numeric for v2 logs; guard otherwise
    mins = pd.to_numeric(sub["MIN"], errors="coerce").dropna()
    return float(mins.mean()) if not mins.empty else 0.0

def _top_roster_by_mpg(team_name: str, season: str, date_str: str, top_n: int = 9,
                       fallback_second_half_if_empty: bool = False):
    """
    Return a list of (player_id, player_name, mpg) for players on the roster ON date_str,
    ranked by MIN/G computed up to the day BEFORE date_str.

    First-game fallback now uses EACH PLAYER'S MPG from the **second half of the PREVIOUS season**
    (league-wide, regardless of team).
    """
    import pandas as pd

    tid = sg.get_team_id(team_name)
    timeline = get_roster_timeline(season)
    roster = roster_on_date(team_name, date_str, timeline)
    if not roster:
        return []

    def _prev_season(season_s: str) -> str:
        start = int(season_s[:4])
        end_2 = int(season_s[-2:])
        return f"{start - 1}-{end_2 - 1:02d}"

    # detect first game for this team
    is_first_game = False
    try:
        if sg.season_for_date_smart(date_str) == season:
            gl = sg.get_league_game_log(season).copy()
            gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
            mask_team = (pd.to_numeric(gl["TEAM_ID"], errors="coerce") == int(tid))
            is_first_game = (gl.loc[mask_team & (gl["GAME_DATE"] < pd.to_datetime(date_str))].shape[0] == 0)
    except Exception:
        is_first_game = True

    items = []

    if is_first_game:
        # ---------- FALLBACK: use 2nd half of previous season for each player ----------
        prev_season = _prev_season(season)

        # Get the (date_from, date_to) window for the team's 2H last season
        try:
            d_from, d_to = _second_half_window_by_id(int(tid), prev_season)
        except Exception:
            d_from, d_to = None, None

        # If helper couldn't compute a split, use a safe default 2H window
        if not d_from or not d_to:
            # Feb 1 -> mid-July of the prev season’s end-year
            d_from = pd.Timestamp(f"02/01/{int(prev_season[:4]) + 1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev_season[:4]) + 1}")

        # Load previous season player logs once
        lgp = sg.get_league_game_log(prev_season, player_or_team_abbreviation="P").copy()
        lgp["GAME_DATE"] = pd.to_datetime(lgp["GAME_DATE"], errors="coerce").dt.normalize()
        lgp["PLAYER_ID"] = pd.to_numeric(lgp["PLAYER_ID"], errors="coerce")
        lgp["MIN"] = pd.to_numeric(lgp["MIN"], errors="coerce")

        # Compute MPG for each rostered player in that 2H window (any team)
        d_from_ts = pd.to_datetime(d_from)
        d_to_ts   = pd.to_datetime(d_to)

        for name in roster:
            pid = sg.get_player_id(name)
            if not pid:
                continue
            pid = int(pid)

            sub = lgp[(lgp["PLAYER_ID"] == pid) &
                      (lgp["GAME_DATE"] >= d_from_ts) &
                      (lgp["GAME_DATE"] <= d_to_ts)]

            # Strict: if no minutes in 2H last season, treat as 0.0 MPG (rookie/injury, etc.)
            mpg = float(sub["MIN"].mean()) if not sub.empty else 0.0
            items.append((pid, name, mpg))
        # ---------- end fallback ----------

    else:
        # current-season to date (day BEFORE date_str)
        cutoff = _cutoff(date_str)
        for name in roster:
            pid = sg.get_player_id(name)
            if not pid:
                continue
            mpg = _minutes_per_game_to_date(int(pid), int(tid), season, cutoff)
            items.append((int(pid), name, float(mpg)))

    items.sort(key=lambda t: t[2], reverse=True)
    return items[:top_n]




_METRIC_COL_SAFE = {"OFF_RATING","DEF_RATING","NET_RATING","POSS","REB_PCT","TM_TOV_PCT","EFG_PCT"}

# REPLACE the old `_team_player_on_metric` with this delta version
def _team_player_onoff_delta_metric(team_name: str, season: str, date_str: str,
                                    player_id: int, metric: str) -> float:
    """
    Return (ON - OFF) for `metric` from TeamPlayerOnOffDetails for the focal `player_id`.

    Normal path: current team in `season`, window 10/01/{start_year} -> (date_str - 1 day).
    First-game fallback: second half of the *previous* season on the one team the player
    played the majority of 2H games for (tie-break: latest game date).
    NEW: Normal-path fallback — if normal path returns NaN (e.g., metric is None due
    to tiny sample), do the same previous-season-2H majority-team fallback.
    """
    import math
    import pandas as pd
    from datetime import datetime, timedelta
    import stats_getter as sg

    tid = sg.get_team_id(team_name)
    start_year = int(season[:4])

    # cutoff is the day BEFORE the game date
    cutoff_dt = datetime.strptime(date_str, "%m/%d/%Y") - timedelta(days=1)
    cutoff = cutoff_dt.strftime("%m/%d/%Y")
    date_from_default = f"10/01/{start_year}"

    # --- helper: extract ON-OFF delta for VS_PLAYER_ID = player_id ---
    def _extract_delta(details) -> float:
        if details is None:
            return math.nan
        on = details.get("on")
        off = details.get("off")
        if on is None or off is None or on.empty or off.empty:
            return math.nan
        try:
            row_on  = on[pd.to_numeric(on["VS_PLAYER_ID"],  errors="coerce") == int(player_id)]
            row_off = off[pd.to_numeric(off["VS_PLAYER_ID"], errors="coerce") == int(player_id)]
        except Exception:
            return math.nan
        if row_on.empty or row_off.empty:
            return math.nan

        # metric column lookup (robust to case)
        mcols = list(row_on.columns)
        m = metric if metric in mcols else (metric.upper() if metric.upper() in mcols else None)
        if m is None:
            return math.nan

        val_on  = pd.to_numeric(row_on[m],  errors="coerce")
        val_off = pd.to_numeric(row_off[m], errors="coerce")
        if val_on.empty or val_off.empty or pd.isna(val_on.iloc[0]) or pd.isna(val_off.iloc[0]):
            return math.nan
        return float(val_on.iloc[0] - val_off.iloc[0])

    # --- helper: robust team_id -> team_name mapping ---
    def _team_name_from_id(tid_int: int, lgp_any: pd.DataFrame | None = None) -> str | None:
        for fn in ("get_team_name_by_id", "team_id_to_name", "get_team_name", "team_name_from_id"):
            if hasattr(sg, fn):
                try:
                    return getattr(sg, fn)(int(tid_int))
                except Exception:
                    pass
        if lgp_any is not None:
            sub_any = lgp_any[pd.to_numeric(lgp_any.get("TEAM_ID"), errors="coerce") == int(tid_int)]
            if "TEAM_NAME" in sub_any.columns and not sub_any["TEAM_NAME"].isna().all():
                return str(sub_any["TEAM_NAME"].dropna().iloc[-1])
            if "TEAM_ABBREVIATION" in sub_any.columns and not sub_any["TEAM_ABBREVIATION"].isna().all():
                abbr = str(sub_any["TEAM_ABBREVIATION"].dropna().iloc[-1])
                if hasattr(sg, "canon_team"):
                    try:
                        return sg.canon_team(abbr)
                    except Exception:
                        return abbr
                return abbr
        return None

    # --- shared fallback routine: previous season 2H on player's majority team ---
    def _prev_season_2h_majority_team_delta() -> float:
        prev_season = f"{start_year-1}-{str(start_year)[-2:]}"
        # get 2H window
        try:
            d_from, d_to = _second_half_window_by_id(int(tid), prev_season)
        except Exception:
            d_from, d_to = None, None
        if not d_from or not d_to:
            # conservative default window if helper unavailable
            d_from = pd.Timestamp(f"02/01/{int(prev_season[:4])+1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev_season[:4])+1}")

        d_from_ts = pd.to_datetime(d_from)
        d_to_ts   = pd.to_datetime(d_to)

        # player game logs (player mode)
        try:
            lgp_prev = sg.get_league_game_log(prev_season, player_or_team_abbreviation="P").copy()
            lgp_prev["GAME_DATE"] = pd.to_datetime(lgp_prev["GAME_DATE"], errors="coerce").dt.normalize()
            lgp_prev["PLAYER_ID"] = pd.to_numeric(lgp_prev["PLAYER_ID"], errors="coerce")
            lgp_prev["TEAM_ID"]   = pd.to_numeric(lgp_prev.get("TEAM_ID"), errors="coerce")
        except Exception:
            return math.nan

        sub_pl = lgp_prev[(lgp_prev["PLAYER_ID"] == int(player_id)) &
                          (lgp_prev["GAME_DATE"] >= d_from_ts) &
                          (lgp_prev["GAME_DATE"] <= d_to_ts)]
        if sub_pl.empty or "TEAM_ID" not in sub_pl.columns:
            return math.nan

        counts = (sub_pl.groupby("TEAM_ID")
                          .agg(games=("GAME_ID", "nunique"),
                               last_date=("GAME_DATE", "max"))
                          .reset_index()
                          .sort_values(["games", "last_date"], ascending=[False, False]))

        cand_tid = int(counts.iloc[0]["TEAM_ID"])
        cand_name = _team_name_from_id(cand_tid, lgp_any=lgp_prev)
        if not cand_name:
            return math.nan

        try:
            details = sg.getTeamPlayerOnOffDetails(
                team_name=cand_name,
                season=prev_season,
                date_from=pd.to_datetime(d_from_ts).strftime("%m/%d/%Y"),
                date_to=pd.to_datetime(d_to_ts).strftime("%m/%d/%Y"),
                measure_type="Advanced",
                per_mode="PerGame",
                season_type_all_star="Regular Season",
            )
        except Exception:
            return math.nan

        return _extract_delta(details)

    # --- first-game detection for this team in this season ---
    use_prev_half = False
    try:
        gl = sg.get_league_game_log(season).copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
        use_prev_half = int(((pd.to_numeric(gl["TEAM_ID"], errors="coerce") == int(tid)) &
                             (gl["GAME_DATE"] < pd.to_datetime(date_str))).sum()) == 0
    except Exception:
        use_prev_half = False  # default to normal path if unsure

    # ---------- main paths ----------
    if use_prev_half:
        # first-game fallback (existing behavior)
        return _prev_season_2h_majority_team_delta()

    # Normal path
    try:
        details = sg.getTeamPlayerOnOffDetails(
            team_name=team_name,
            season=season,
            date_from=date_from_default,
            date_to=cutoff,
            measure_type="Advanced",
            per_mode="PerGame",
            season_type_all_star="Regular Season",
        )
    except Exception:
        details = None

    delta = _extract_delta(details)

    # NEW: if metric is NaN/None from normal path, fallback to prev season 2H majority team
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        delta = _prev_season_2h_majority_team_delta()

    return delta




def getPlayerTeamMetric(team_name: str, season: str, date_str: str,
                        slot: int, metric: str) -> float:
    """
    Choose the `slot`-th player by MIN/G (up to the day BEFORE `date_str`) and
    return (ON - OFF) for the team’s advanced metric when that player is on/off.
    """
    import math
    slot = int(slot)
    if slot < 1 or slot > 9:
        return math.nan

    ranked = _top_roster_by_mpg(
        team_name, season, date_str, top_n=max(9, slot),
        fallback_second_half_if_empty=True
    )

    if not ranked or len(ranked) < slot:
        return math.nan


    player_id, _, _ = ranked[slot - 1]
    return _team_player_onoff_delta_metric(team_name, season, date_str, player_id, metric)

# --- Thin convenience wrappers the model can reference directly ---------------
# Example usage inside your feature builder:
#   getPlayer1_Team_OFF_RATING(team, season, date)  → float

def _mk(slot: int, metric: str):
    def fn(team_name: str, season: str, date_str: str) -> float:
        return getPlayerTeamMetric(team_name, season, date_str, slot, metric)
    fn.__name__ = f"getPlayer{slot}_Team_{metric}"
    return fn

# Create 1..9 × desired metrics
for _slot in range(1, 10):
    globals()[f"getPlayer{_slot}_Team_OFF_RATING"] = _mk(_slot, "OFF_RATING")
    globals()[f"getPlayer{_slot}_Team_DEF_RATING"] = _mk(_slot, "DEF_RATING")
    globals()[f"getPlayer{_slot}_Team_REB_PCT"]    = _mk(_slot, "REB_PCT")
    globals()[f"getPlayer{_slot}_Team_TM_TOV_PCT"] = _mk(_slot, "TM_TOV_PCT")
    globals()[f"getPlayer{_slot}_Team_EFG_PCT"]    = _mk(_slot, "EFG_PCT")


def _team_player_poss_pg(team_name: str, season: str, date_str: str, player_id: int) -> float:
    """
    Return ON-court possessions per game for this player up to the day BEFORE `date_str`.

    Normal path: use current team in `season` from 10/01 to (date_str - 1 day).
    First-game fallback: use the player's team from the SECOND HALF of the PREVIOUS season.
      - Determine the team by counting the player's games in that 2H window (majority wins; tie → most recent game).
      - Query TeamPlayerOnOffDetails for that team over that window.
    Returns NaN if no qualifying data is found.
    """
    import math
    import pandas as pd
    from datetime import datetime, timedelta
    import stats_getter as sg

    tid = sg.get_team_id(team_name)
    start_year = int(season[:4])

    # cutoff is the day BEFORE the game date
    cutoff_dt = datetime.strptime(date_str, "%m/%d/%Y") - timedelta(days=1)
    cutoff = cutoff_dt.strftime("%m/%d/%Y")
    date_from_default = f"10/01/{start_year}"

    # --- helper: robust team_id -> team_name mapping ---
    def _team_name_from_id(tid_int: int, lgp_any: pd.DataFrame | None = None) -> str | None:
        for fn in ("get_team_name_by_id", "team_id_to_name", "get_team_name", "team_name_from_id"):
            if hasattr(sg, fn):
                try:
                    return getattr(sg, fn)(int(tid_int))
                except Exception:
                    pass
        if lgp_any is not None:
            sub_any = lgp_any[pd.to_numeric(lgp_any.get("TEAM_ID"), errors="coerce") == int(tid_int)]
            if "TEAM_NAME" in sub_any.columns and not sub_any["TEAM_NAME"].isna().all():
                return str(sub_any["TEAM_NAME"].dropna().iloc[-1])
            if "TEAM_ABBREVIATION" in sub_any.columns and not sub_any["TEAM_ABBREVIATION"].isna().all():
                abbr = str(sub_any["TEAM_ABBREVIATION"].dropna().iloc[-1])
                if hasattr(sg, "canon_team"):
                    try:
                        return sg.canon_team(abbr)
                    except Exception:
                        return abbr
                return abbr
        return None

    # --- detect if this is the first game for the team in `season` ---
    use_prev_half = False
    try:
        gl = sg.get_league_game_log(season).copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
        use_prev_half = int(((pd.to_numeric(gl["TEAM_ID"], errors="coerce") == int(tid)) &
                             (gl["GAME_DATE"] < pd.to_datetime(date_str))).sum()) == 0
    except Exception:
        # If we can't confirm, stick with normal path (safer to avoid wrong-season lookups)
        use_prev_half = False

    if not use_prev_half:
        # ---- normal path: current season, current team ----
        try:
            details = sg.getTeamPlayerOnOffDetails(
                team_name=team_name,
                season=season,
                date_from=date_from_default,
                date_to=cutoff,
                measure_type="Advanced",
                per_mode="PerGame",
                season_type_all_star="Regular Season",
            )
        except Exception:
            return math.nan
    else:
        # ---- first-game fallback: second half of previous season, team player actually played for ----
        prev_season = f"{start_year-1}-{str(start_year)[-2:]}"

        # 2H window: use helper; if it fails, use Feb 1 → Jul 15
        try:
            d_from, d_to = _second_half_window_by_id(int(tid), prev_season)
        except Exception:
            d_from, d_to = None, None
        if not d_from or not d_to:
            d_from = pd.Timestamp(f"02/01/{int(prev_season[:4]) + 1}")
            d_to   = pd.Timestamp(f"07/15/{int(prev_season[:4]) + 1}")

        d_from_ts = pd.to_datetime(d_from)
        d_to_ts   = pd.to_datetime(d_to)

        # Player's games in that 2H window (league-wide logs at player granularity)
        lgp_prev = sg.get_league_game_log(prev_season, player_or_team_abbreviation="P").copy()
        lgp_prev["GAME_DATE"] = pd.to_datetime(lgp_prev["GAME_DATE"], errors="coerce").dt.normalize()
        lgp_prev["PLAYER_ID"] = pd.to_numeric(lgp_prev["PLAYER_ID"], errors="coerce")
        lgp_prev["TEAM_ID"]   = pd.to_numeric(lgp_prev.get("TEAM_ID"), errors="coerce")

        sub_pl = lgp_prev[(lgp_prev["PLAYER_ID"] == int(player_id)) &
                          (lgp_prev["GAME_DATE"] >= d_from_ts) &
                          (lgp_prev["GAME_DATE"] <= d_to_ts)]

        if sub_pl.empty or "TEAM_ID" not in sub_pl.columns:
            return math.nan  # by your rule: no prior 2H minutes → undefined

        # Choose the ONE team with most games; break ties by most recent game date
        counts = (sub_pl.groupby("TEAM_ID")
                          .agg(games=("GAME_ID", "nunique"),
                               last_date=("GAME_DATE", "max"))
                          .reset_index()
                          .sort_values(["games", "last_date"], ascending=[False, False]))

        cand_tid = int(counts.iloc[0]["TEAM_ID"])
        cand_name = _team_name_from_id(cand_tid, lgp_any=lgp_prev)
        if not cand_name:
            return math.nan

        try:
            details = sg.getTeamPlayerOnOffDetails(
                team_name=cand_name,
                season=prev_season,
                date_from=d_from_ts.strftime("%m/%d/%Y"),
                date_to=d_to_ts.strftime("%m/%d/%Y"),
                measure_type="Advanced",
                per_mode="PerGame",
                season_type_all_star="Regular Season",
            )
        except Exception:
            return math.nan

    # ---- extract POSS/GP for the focal player from the "on" table ----
    on = details.get("on")
    if on is None or on.empty:
        return math.nan

    try:
        row_on = on[pd.to_numeric(on["VS_PLAYER_ID"], errors="coerce") == int(player_id)]
    except Exception:
        return math.nan
    if row_on.empty:
        return math.nan

    poss = pd.to_numeric(row_on.get("POSS"), errors="coerce")
    gp   = pd.to_numeric(row_on.get("GP"),   errors="coerce")

    if poss is None or gp is None or poss.empty or gp.empty or pd.isna(poss.iloc[0]) or pd.isna(gp.iloc[0]) or gp.iloc[0] == 0:
        return math.nan

    # Many endpoints report per-game already, but compute robustly
    return float(poss.iloc[0] / gp.iloc[0])

    
def getPlayerTeamPOSS(team_name: str, season: str, date_str: str, slot: int) -> float:
    """
    Choose the `slot`-th player (1..9) by MIN/G (to the day BEFORE `date_str`),
    and return that player's ON-court POSS/G windowed as described above.
    """
    import math
    slot = int(slot)
    if slot < 1 or slot > 9:
        return math.nan

    ranked = _top_roster_by_mpg(
        team_name, season, date_str, top_n=max(9, slot),
        fallback_second_half_if_empty=True
    )

    if not ranked or len(ranked) < slot:
        return math.nan

    player_id, _, _ = ranked[slot - 1]
    return _team_player_poss_pg(team_name, season, date_str, player_id)


def getPlayer1_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 1)
def getPlayer2_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 2)
def getPlayer3_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 3)
def getPlayer4_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 4)
def getPlayer5_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 5)
def getPlayer6_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 6)
def getPlayer7_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 7)
def getPlayer8_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 8)
def getPlayer9_Team_POSS(team_name: str, season: str, date_str: str) -> float: return getPlayerTeamPOSS(team_name, season, date_str, 9)










        

#stats_cache.clear_cache()

#result = get_hardcoded_2014_2015_averages()
#print(result)






# First, let's see what shot categories are available

# Then filter more carefully






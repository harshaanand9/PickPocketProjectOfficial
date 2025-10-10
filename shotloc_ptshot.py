# ================================
# Generic "Shot Locations" wrappers
# ================================
_EPS_R = 5e-5   # per-possession floor for rate terms
_EPS_P = 1e-3   # floor for proportions -> clip to [0.001, 0.999]
# shotloc_ptshot.py


from shotloc_shared import _mi_get, poss_pg_from_ledger

from stats_getter import _team_row_from_league_shotloc, get_league_shot_locations, get_team_pt_shots, team_regular_season_range_by_id




_SHOTLOC_KIND_TO_CATS = {
    "corner3p":           ["Left Corner 3", "Right Corner 3"],
    "above_the_break3p":  ["Above the Break 3"],
    "paintshots":         ["Restricted Area", "In The Paint (Non-RA)"],
    "midrangeshots":      ["Mid-Range"],
}

from datetime import datetime, timedelta
from stats_getter import get_team_id

DATE_FMT = "%m/%d/%Y"

import math

# ---- numeric stability (not "gating", just tiny floors/clips) ----


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))

def _clip01(p: float, eps: float = _EPS_P) -> float:
    if p < eps: 
        return eps
    if p > 1.0 - eps: 
        return 1.0 - eps
    return p


def _window_regular_before_game(team_name: str, season: str, date_str: str):
    """
    Returns (date_from, date_to) as strings:
      date_from = first RS game (clamped to Oct 1)
      date_to   = min(day BEFORE `date_str`, last RS game)
    Uses your existing team_regular_season_range_by_id helper.
    """
    # Oct 1 baseline
    start_yr = int(season.split("-")[0])
    oct1 = datetime.strptime(f"10/01/{start_yr}", DATE_FMT)

    # team RS span
    tid = get_team_id(team_name)
    lo_str, hi_str = team_regular_season_range_by_id(tid, season)  # your helper
    lo = datetime.strptime(lo_str, DATE_FMT) if lo_str else oct1
    hi = datetime.strptime(hi_str, DATE_FMT) if hi_str else datetime.strptime(f"06/30/{start_yr+1}", DATE_FMT)

    # matchup cutoff = day-before game
    target = datetime.strptime(date_str, DATE_FMT)
    cutoff = target - timedelta(days=1)

    # clamp
    date_from_dt = max(lo, oct1)
    date_to_dt   = min(cutoff, hi)
    if date_to_dt < date_from_dt:
        date_to_dt = date_from_dt  # empty window protection

    return date_from_dt.strftime(DATE_FMT), date_to_dt.strftime(DATE_FMT)


def _shotloc_totals_any(row, cats, fga_label: str, fgm_label: str):
    """
    Sum FGA/FGM for the given shot categories from a single-team row of the
    LEAGUE shot-locations table. Works for both Base ('FGA','FGM') and
    Opponent ('OPP_FGA','OPP_FGM') schemas.

    We reuse _mi_get so it handles the MultiIndex columns cleanly.
    """
    import math

    df1 = row.to_frame().T if hasattr(row, "to_frame") else row  # ensure DataFrame
    fga = 0.0
    fgm = 0.0
    for c in cats:
        fga += float(_mi_get(df1, c, fga_label, 0.0))
        fgm += float(_mi_get(df1, c, fgm_label, 0.0))
    # guard against NaNs sneaking in
    if isinstance(fga, float) and math.isnan(fga): fga = 0.0
    if isinstance(fgm, float) and math.isnan(fgm): fgm = 0.0
    return fga, fgm


def get_shotloc_rate(
    kind: str,
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str,
    *, season_type_all_star: str = "Regular Season"
) -> float:
    """
    Log *product* matchup using LEAGUE shot-locations
    (A: Base; B: Opponent).
    Returns: log( (XFGA_A / POSS_A + eps) * (XFGA_allowed_by_B / DEF_POSS_B + eps) )
    """

    cats = _SHOTLOC_KIND_TO_CATS.get(kind)
    if not cats:
        return 0.0

    a_poss = poss_pg_from_ledger(team_a_name, team_a_season, date_str)
    b_oppp = poss_pg_from_ledger(team_b_name, team_b_season, date_str)
    if a_poss <= 0 or b_oppp <= 0:
        return 0.0

    # A: Base
    dfa, cutA = _window_regular_before_game(team_a_name, team_a_season, date_str)
    dfa_all = get_league_shot_locations(team_a_season, dfa, cutA,
                                        measure_type="Base",
                                        season_type_all_star=season_type_all_star)
    row_a = _team_row_from_league_shotloc(dfa_all, team_a_name)

    # B: Opponent
    dfb, cutB = _window_regular_before_game(team_b_name, team_b_season, date_str)
    dfb_opp = get_league_shot_locations(team_b_season, dfb, cutB,
                                        measure_type="Opponent",
                                        season_type_all_star=season_type_all_star)
    row_bopp = _team_row_from_league_shotloc(dfb_opp, team_b_name)

    if row_a is None or row_a.empty or row_bopp is None or row_bopp.empty:
        return 0.0

    a_fga, _   = _shotloc_totals_any(row_a, cats, "FGA")        # Base
    bopp_fga, _= _shotloc_totals_any(row_bopp, cats, "OPP_FGA") # Opponent

    a_rate    = (a_fga / a_poss)    if a_poss  > 0 else 0.0
    bopp_rate = (bopp_fga / b_oppp) if b_oppp > 0 else 0.0

    # NEW: log-product
    return math.log((a_rate + _EPS_R) * (bopp_rate + _EPS_R))



def get_shotloc_efficiency(kind: str,
                           team_a_name: str, team_b_name: str,
                           team_a_season: str, team_b_season: str,
                           date_str: str,
                           *, season_type_all_star: str = "Regular Season") -> float:
    """
    Logit efficiency matchup using LEAGUE shot-locations (A: Base; B: Opponent).
    Returns: logit(p_A) + logit(p_Aopp), with minimal clipping to (eps, 1-eps).
    p = XFGM / XFGA
    """
    cats = _SHOTLOC_KIND_TO_CATS.get(kind)
    if not cats:
        return 0.0

    # A: Base
    dfa, cutA = _window_regular_before_game(team_a_name, team_a_season, date_str)
    dfa_all = get_league_shot_locations(team_a_season, dfa, cutA,
                                        measure_type="Base",
                                        season_type_all_star=season_type_all_star)
    row_a = _team_row_from_league_shotloc(dfa_all, team_a_name)

    # B: Opponent
    dfb, cutB = _window_regular_before_game(team_b_name, team_b_season, date_str)
    dfb_opp = get_league_shot_locations(team_b_season, dfb, cutB,
                                        measure_type="Opponent",
                                        season_type_all_star=season_type_all_star)
    row_bopp = _team_row_from_league_shotloc(dfb_opp, team_b_name)

    if row_a is None or row_a.empty or row_bopp is None or row_bopp.empty:
        return 0.0

    a_fga, a_fgm       = _shotloc_totals_any(row_a,    cats, "FGA",     "FGM")        # Base
    bopp_fga, bopp_fgm = _shotloc_totals_any(row_bopp, cats, "OPP_FGA", "OPP_FGM")    # Opponent

    pA    = (a_fgm    / a_fga)    if a_fga    > 0 else 0.0
    pAopp = (bopp_fgm / bopp_fga) if bopp_fga > 0 else 0.0

    pA    = _clip01(pA)
    pAopp = _clip01(pAopp)
    return _logit(pA) + _logit(pAopp)




# ==========================
# Generic "PT Shots" wrappers
# ==========================

# ---- PT-SHOTS bucket definitions (unchanged idea) ----
_PTSHOTS_DEF = {
    "contested3p": {"ranges": ["0-2 Feet - Very Tight", "2-4 Feet - Tight"], "is3": True},
    "open3p":      {"ranges": ["4-6 Feet - Open", "6+ Feet - Wide Open"],     "is3": True},
    "contested2p": {"ranges": ["0-2 Feet - Very Tight", "2-4 Feet - Tight"], "is3": False},
    "open2p":      {"ranges": ["4-6 Feet - Open", "6+ Feet - Wide Open"],     "is3": False},
}


def _ptshots_pick_tables(dfs):
    """
    Your environment:
      dfs[3] -> team own closest-defender table
      dfs[4] -> opponents vs team closest-defender table
    """
    own = dfs[3] if dfs and len(dfs) > 3 else None
    opp = dfs[4] if dfs and len(dfs) > 4 else None
    return own, opp

def _ptshots_totals(df, ranges, is3: bool):
    """
    Sum attempts/makes for the requested defender-distance ranges.
    We use FG3A/FG3M for 3s and FG2A/FG2M for 2s (weighted by attempts).
    """
    if df is None or getattr(df, "empty", True):
        return 0.0, 0.0
    sub = df[df["CLOSE_DEF_DIST_RANGE"].isin(ranges)]
    if sub.empty:
        return 0.0, 0.0
    if is3:
        fga = float(sub["FG3A"].sum()); fgm = float(sub["FG3M"].sum())
    else:
        fga = float(sub["FG2A"].sum()); fgm = float(sub["FG2M"].sum())
    return fga, fgm

# === PT-SHOTS fetch: hard rule for season "2012-13" ==========================
def _ptshots_fetch_with_fallback(team_name: str, season: str, date_str: str):
    """
    If season == "2012-13": return TEAM PT-shots for the full 2013-14 regular season
    (10/01/2013 → 06/30/2014) using per_mode="PerGame".

    Else: return the usual pre-game (day-before) current-season window.
    """
    def _empty(df):
        return (df is None) or getattr(df, "empty", True)

    # Force 2013-14 full-season averages whenever loader passes "2012-13"
    if season == "2012-13":
        fetch_season = "2013-14"
        lo_str = "10/01/2013"
        hi_str = "06/30/2014"
        dfs_full = get_team_pt_shots(team_name, fetch_season, lo_str, hi_str, per_mode="PerGame")
        own_full, opp_full = _ptshots_pick_tables(dfs_full)
        # Even if empty, return as-is; downstream code already handles 0s safely
        return own_full, opp_full

    # Normal behavior for all other seasons
    dfrom, dto = _window_regular_before_game(team_name, season, date_str)
    dfs = get_team_pt_shots(team_name, season, dfrom, dto, per_mode="PerGame")
    own, opp = _ptshots_pick_tables(dfs)
    return own, opp




# ==========================
# (REPLACE) Generic PT Shots wrappers to use fallback
# ==========================

def get_ptshots_rate(
    kind: str,
    team_a_name: str, team_b_name: str,
    team_a_season: str, team_b_season: str,
    date_str: str
) -> float:
    """
    Log *product* matchup using TEAM PT Shots
    (A: team shooting rate, B: opponent-allowed rate).
    Returns: log( (XFGA_A / POSS_A + eps) * (XFGA_allowed_by_B / DEF_POSS_B + eps) )
    """
    spec = _PTSHOTS_DEF.get(kind)
    if not spec:
        return 0.0

    a_poss = poss_pg_from_ledger(team_a_name, team_a_season, date_str)
    b_oppp = poss_pg_from_ledger(team_b_name, team_b_season, date_str)
    if a_poss <= 0 or b_oppp <= 0:
        return 0.0

    # A own (with 2013-14 first-game fallback)
    df_a_own, _ = _ptshots_fetch_with_fallback(team_a_name, team_a_season, date_str)
    a_fga, _ = _ptshots_totals(df_a_own, spec["ranges"], spec["is3"])

    # B opponent-allowed (with 2013-14 first-game fallback)
    _, df_b_opp = _ptshots_fetch_with_fallback(team_b_name, team_b_season, date_str)
    bopp_fga, _ = _ptshots_totals(df_b_opp, spec["ranges"], spec["is3"])

    a_rate    = (a_fga / a_poss)   if a_poss  > 0 else 0.0
    bopp_rate = (bopp_fga / b_oppp) if b_oppp > 0 else 0.0
    return math.log((a_rate + _EPS_R) * (bopp_rate + _EPS_R))


def get_ptshots_efficiency(kind: str,
                           team_a_name: str, team_b_name: str,
                           team_a_season: str, team_b_season: str,
                           date_str: str) -> float:
    """
    Logit efficiency matchup using TEAM PT Shots (attempt-weighted).
    Returns: logit(p_A) + logit(p_Aopp) with minimal clipping; p = XFGM / XFGA.
    """
    spec = _PTSHOTS_DEF.get(kind)
    if not spec:
        return 0.0

    # A own (with 2013-14 first-game fallback)
    df_a_own, _ = _ptshots_fetch_with_fallback(team_a_name, team_a_season, date_str)
    a_fga, a_fgm = _ptshots_totals(df_a_own, spec["ranges"], spec["is3"])

    # B opponent-allowed (with 2013-14 first-game fallback)
    _, df_b_opp = _ptshots_fetch_with_fallback(team_b_name, team_b_season, date_str)
    bopp_fga, bopp_fgm = _ptshots_totals(df_b_opp, spec["ranges"], spec["is3"])

    pA    = (a_fgm    / a_fga)    if a_fga    > 0 else 0.0
    pAopp = (bopp_fgm / bopp_fga) if bopp_fga > 0 else 0.0
    pA    = _clip01(pA)
    pAopp = _clip01(pAopp)
    return _logit(pA) + _logit(pAopp)


# === NEW: component helpers using the same fallback ==========================

def _ptshots_rate_components(kind, team_a_name, team_b_name, team_a_season, team_b_season, date_str):
    # returns (a_rate, bopp_rate) without log or product
    spec = _PTSHOTS_DEF.get(kind)
    if not spec:
        return 0.0, 0.0

    a_poss = poss_pg_from_ledger(team_a_name, team_a_season, date_str)
    b_oppp = poss_pg_from_ledger(team_b_name, team_b_season, date_str)
    if a_poss <= 0 or b_oppp <= 0:
        return 0.0, 0.0

    df_a_own, _ = _ptshots_fetch_with_fallback(team_a_name, team_a_season, date_str)
    a_fga, _ = _ptshots_totals(df_a_own, spec["ranges"], spec["is3"])

    _, df_b_opp = _ptshots_fetch_with_fallback(team_b_name, team_b_season, date_str)
    bopp_fga, _ = _ptshots_totals(df_b_opp, spec["ranges"], spec["is3"])

    a_rate    = (a_fga / a_poss)   if a_poss  > 0 else 0.0
    bopp_rate = (bopp_fga / b_oppp) if b_oppp > 0 else 0.0
    return a_rate, bopp_rate


def _ptshots_eff_components(kind, team_a_name, team_b_name, team_a_season, team_b_season, date_str):
    # returns (pA, pAopp) as raw proportions (no logit)
    spec = _PTSHOTS_DEF.get(kind)
    if not spec:
        return 0.0, 0.0

    df_a_own, _ = _ptshots_fetch_with_fallback(team_a_name, team_a_season, date_str)
    a_fga, a_fgm = _ptshots_totals(df_a_own, spec["ranges"], spec["is3"])

    _, df_b_opp = _ptshots_fetch_with_fallback(team_b_name, team_b_season, date_str)
    bopp_fga, bopp_fgm = _ptshots_totals(df_b_opp, spec["ranges"], spec["is3"])

    pA    = (a_fgm    / a_fga)    if a_fga    > 0 else 0.0
    pAopp = (bopp_fgm / bopp_fga) if bopp_fga > 0 else 0.0
    return pA, pAopp



def _shotloc_rate_components(kind, team_a_name, team_b_name, team_a_season, team_b_season, date_str, *, season_type_all_star="Regular Season"):
    # returns (a_rate, bopp_rate) without log or product
    cats = _SHOTLOC_KIND_TO_CATS.get(kind)
    if not cats:
        return 0.0, 0.0

    a_poss = poss_pg_from_ledger(team_a_name, team_a_season, date_str)
    b_oppp = poss_pg_from_ledger(team_b_name, team_b_season, date_str)
    if a_poss <= 0 or b_oppp <= 0:
        return 0.0, 0.0

    dfa, cutA = _window_regular_before_game(team_a_name, team_a_season, date_str)
    dfa_all = get_league_shot_locations(team_a_season, dfa, cutA, measure_type="Base", season_type_all_star=season_type_all_star)
    row_a = _team_row_from_league_shotloc(dfa_all, team_a_name)

    dfb, cutB = _window_regular_before_game(team_b_name, team_b_season, date_str)
    dfb_opp = get_league_shot_locations(team_b_season, dfb, cutB, measure_type="Opponent", season_type_all_star=season_type_all_star)
    row_bopp = _team_row_from_league_shotloc(dfb_opp, team_b_name)

    if row_a is None or row_a.empty or row_bopp is None or row_bopp.empty:
        return 0.0, 0.0

    a_fga, _    = _shotloc_totals_any(row_a, cats, "FGA", "FGM")
    bopp_fga, _ = _shotloc_totals_any(row_bopp, cats, "OPP_FGA", "OPP_FGM")

    a_rate    = (a_fga / a_poss)   if a_poss  > 0 else 0.0
    bopp_rate = (bopp_fga / b_oppp) if b_oppp > 0 else 0.0
    return a_rate, bopp_rate


def _shotloc_eff_components(kind, team_a_name, team_b_name, team_a_season, team_b_season, date_str, *, season_type_all_star="Regular Season"):
    # returns (pA, pAopp) as raw proportions (no logit)
    cats = _SHOTLOC_KIND_TO_CATS.get(kind)
    if not cats:
        return 0.0, 0.0

    dfa, cutA = _window_regular_before_game(team_a_name, team_a_season, date_str)
    dfa_all = get_league_shot_locations(team_a_season, dfa, cutA, measure_type="Base", season_type_all_star=season_type_all_star)
    row_a = _team_row_from_league_shotloc(dfa_all, team_a_name)

    dfb, cutB = _window_regular_before_game(team_b_name, team_b_season, date_str)
    dfb_opp = get_league_shot_locations(team_b_season, dfb, cutB, measure_type="Opponent", season_type_all_star=season_type_all_star)
    row_bopp = _team_row_from_league_shotloc(dfb_opp, team_b_name)

    if row_a is None or row_a.empty or row_bopp is None or row_bopp.empty:
        return 0.0, 0.0

    a_fga, a_fgm       = _shotloc_totals_any(row_a,    cats, "FGA",     "FGM")
    bopp_fga, bopp_fgm = _shotloc_totals_any(row_bopp, cats, "OPP_FGA", "OPP_FGM")

    pA    = (a_fgm    / a_fga)    if a_fga    > 0 else 0.0
    pAopp = (bopp_fgm / bopp_fga) if bopp_fga > 0 else 0.0
    return pA, pAopp

# === NEW: public wrappers to use in try_feature ==============================

# ---- PT SHOTS (defender distance) ----
def contested_3pt_rate_home(home, away, home_season, away_season, date_str): 
    return _ptshots_rate_components("contested3p", home, away, home_season, away_season, date_str)[0]

def contested_3pt_rate_away(home, away, home_season, away_season, date_str): 
    return _ptshots_rate_components("contested3p", away, home, away_season, home_season, date_str)[0]

def contested_3pt_eff_home(home, away, home_season, away_season, date_str): 
    return _ptshots_eff_components("contested3p", home, away, home_season, away_season, date_str)[0]

def contested_3pt_eff_away(home, away, home_season, away_season, date_str): 
    return _ptshots_eff_components("contested3p", away, home, away_season, home_season, date_str)[0]

def opp_contested_3pt_rate_home(home, away, home_season, away_season, date_str): 
    return _ptshots_rate_components("contested3p", home, away, home_season, away_season, date_str)[1]

def opp_contested_3pt_rate_away(home, away, home_season, away_season, date_str): 
    return _ptshots_rate_components("contested3p", away, home, away_season, home_season, date_str)[1]

def opp_contested_3pt_eff_home(home, away, home_season, away_season, date_str): 
    return _ptshots_eff_components("contested3p", home, away, home_season, away_season, date_str)[1]

def opp_contested_3pt_eff_away(home, away, home_season, away_season, date_str): 
    return _ptshots_eff_components("contested3p", away, home, away_season, home_season, date_str)[1]


# Mirror the eight wrappers above for: open3p, contested2p, open2p
def open_3pt_rate_home(h, a, hs, as_, d):  return _ptshots_rate_components("open3p",     h, a, hs, as_, d)[0]
def open_3pt_rate_away(h, a, hs, as_, d):  return _ptshots_rate_components("open3p",     a, h, as_, hs, d)[0]
def open_3pt_eff_home(h, a, hs, as_, d):   return _ptshots_eff_components("open3p",      h, a, hs, as_, d)[0]
def open_3pt_eff_away(h, a, hs, as_, d):   return _ptshots_eff_components("open3p",      a, h, as_, hs, d)[0]
def opp_open_3pt_rate_home(h, a, hs, as_, d): return _ptshots_rate_components("open3p",  h, a, hs, as_, d)[1]
def opp_open_3pt_rate_away(h, a, hs, as_, d): return _ptshots_rate_components("open3p",  a, h, as_, hs, d)[1]
def opp_open_3pt_eff_home(h, a, hs, as_, d):  return _ptshots_eff_components("open3p",   h, a, hs, as_, d)[1]
def opp_open_3pt_eff_away(h, a, hs, as_, d):  return _ptshots_eff_components("open3p",   a, h, as_, hs, d)[1]

def contested_2pt_rate_home(h, a, hs, as_, d):  return _ptshots_rate_components("contested2p", h, a, hs, as_, d)[0]
def contested_2pt_rate_away(h, a, hs, as_, d):  return _ptshots_rate_components("contested2p", a, h, as_, hs, d)[0]
def contested_2pt_eff_home(h, a, hs, as_, d):   return _ptshots_eff_components("contested2p",  h, a, hs, as_, d)[0]
def contested_2pt_eff_away(h, a, hs, as_, d):   return _ptshots_eff_components("contested2p",  a, h, as_, hs, d)[0]
def opp_contested_2pt_rate_home(h, a, hs, as_, d): return _ptshots_rate_components("contested2p", h, a, hs, as_, d)[1]
def opp_contested_2pt_rate_away(h, a, hs, as_, d): return _ptshots_rate_components("contested2p", a, h, as_, hs, d)[1]
def opp_contested_2pt_eff_home(h, a, hs, as_, d):  return _ptshots_eff_components("contested2p",  h, a, hs, as_, d)[1]
def opp_contested_2pt_eff_away(h, a, hs, as_, d):  return _ptshots_eff_components("contested2p",  a, h, as_, hs, d)[1]

def open_2pt_rate_home(h, a, hs, as_, d):  return _ptshots_rate_components("open2p", h, a, hs, as_, d)[0]
def open_2pt_rate_away(h, a, hs, as_, d):  return _ptshots_rate_components("open2p", a, h, as_, hs, d)[0]
def open_2pt_eff_home(h, a, hs, as_, d):   return _ptshots_eff_components("open2p",  h, a, hs, as_, d)[0]
def open_2pt_eff_away(h, a, hs, as_, d):   return _ptshots_eff_components("open2p",  a, h, as_, hs, d)[0]
def opp_open_2pt_rate_home(h, a, hs, as_, d): return _ptshots_rate_components("open2p", h, a, hs, as_, d)[1]
def opp_open_2pt_rate_away(h, a, hs, as_, d): return _ptshots_rate_components("open2p", a, h, as_, hs, d)[1]
def opp_open_2pt_eff_home(h, a, hs, as_, d):  return _ptshots_eff_components("open2p",  h, a, hs, as_, d)[1]
def opp_open_2pt_eff_away(h, a, hs, as_, d):  return _ptshots_eff_components("open2p",  a, h, as_, hs, d)[1]


# ---- SHOT-LOCATION buckets (zones) ----
def corner_3pt_rate_home(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_rate_components("corner3p", h, a, hs, as_, d, season_type_all_star=season_type_all_star)[0]

def corner_3pt_rate_away(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_rate_components("corner3p", a, h, as_, hs, d, season_type_all_star=season_type_all_star)[0]

def corner_3pt_eff_home(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_eff_components("corner3p", h, a, hs, as_, d, season_type_all_star=season_type_all_star)[0]

def corner_3pt_eff_away(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_eff_components("corner3p", a, h, as_, hs, d, season_type_all_star=season_type_all_star)[0]

def opp_corner_3pt_rate_home(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_rate_components("corner3p", h, a, hs, as_, d, season_type_all_star=season_type_all_star)[1]

def opp_corner_3pt_rate_away(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_rate_components("corner3p", a, h, as_, hs, d, season_type_all_star=season_type_all_star)[1]

def opp_corner_3pt_eff_home(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_eff_components("corner3p", h, a, hs, as_, d, season_type_all_star=season_type_all_star)[1]

def opp_corner_3pt_eff_away(h, a, hs, as_, d, *, season_type_all_star="Regular Season"):
    return _shotloc_eff_components("corner3p", a, h, as_, hs, d, season_type_all_star=season_type_all_star)[1]

# Repeat the same 8 wrappers for: above_the_break3p, paintshots, midrangeshots
def above_break_3pt_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("above_the_break3p", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def above_break_3pt_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("above_the_break3p", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def above_break_3pt_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("above_the_break3p", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def above_break_3pt_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("above_the_break3p", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def opp_above_break_3pt_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("above_the_break3p", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_above_break_3pt_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("above_the_break3p", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]
def opp_above_break_3pt_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("above_the_break3p", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_above_break_3pt_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("above_the_break3p", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]

def paint_shot_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("paintshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def paint_shot_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("paintshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def paint_shot_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("paintshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def paint_shot_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("paintshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def opp_paint_shot_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("paintshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_paint_shot_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("paintshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]
def opp_paint_shot_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("paintshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_paint_shot_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("paintshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]

def midrange_shot_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("midrangeshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def midrange_shot_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("midrangeshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def midrange_shot_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("midrangeshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[0]
def midrange_shot_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("midrangeshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[0]
def opp_midrange_shot_rate_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("midrangeshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_midrange_shot_rate_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_rate_components("midrangeshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]
def opp_midrange_shot_eff_home(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("midrangeshots", h,a,hs,as_,d, season_type_all_star=season_type_all_star)[1]
def opp_midrange_shot_eff_away(h,a,hs,as_,d,*,season_type_all_star="Regular Season"):
    return _shotloc_eff_components("midrangeshots", a,h,as_,hs,d, season_type_all_star=season_type_all_star)[1]



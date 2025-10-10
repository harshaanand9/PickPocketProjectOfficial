from __future__ import annotations
import os
import math
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

# Your repo helper (same contract as in elo_builder)
import stats_getter as sg

# --- Canonicalization for historical rebrands (names in leaguegamelog) ---
_NAME_CANON = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}

def _name_aliases(modern_name: str) -> list[str]:
    # Accept legacy names around rebrands
    alias = {
        "Charlotte Hornets": ["Charlotte Hornets", "Charlotte Bobcats"],
        "New Orleans Pelicans": ["New Orleans Pelicans", "New Orleans Hornets"],
        "Brooklyn Nets": ["Brooklyn Nets", "New Jersey Nets"],
        # add any others you care about here
    }
    return alias.get(modern_name, [modern_name])

def canon_team_name(name: str) -> str:
    return _NAME_CANON.get(name, name)

# --- Elo helpers (same math / spirit as your elo_builder) ---
def _expected_from_delta(delta: float) -> float:
    # logistic on 400-Elo scale
    return 1.0 / (1.0 + math.pow(10.0, -delta / 400.0))

def _k_mov_sensitive(mov: float, winner_delta: float) -> float:
    # Figure-11 style MOV-sensitive K
    return 20.0 * math.pow(mov + 3.0, 0.8) / (7.5 + 0.006 * abs(winner_delta))

# ---- Seeds loader (robust to a few common header variants) ----
def load_seeds_2015_16(path_csv: str) -> Dict[str, float]:
    """
    Return {canonical_team_name: elo_seed} for the start of 2015-16.
    Accepts columns like: TEAM_NAME / Team, TEAM_ID (optional), ELO / ELO_SEED / seed / rating.
    """
    df = pd.read_csv(path_csv)
    cols = {c.lower(): c for c in df.columns}

    # team-name column
    name_col = None
    for cand in ("team_name", "team", "name"):
        if cand in cols:
            name_col = cols[cand]
            break
    if not name_col:
        raise ValueError("Could not find a team-name column in seeds CSV.")

    # rating column
    rating_col = None
    for cand in ("elo", "elo_seed", "seed", "rating", "init", "initial", "elo_seed_2015_16"):
        if cand in cols:
            rating_col = cols[cand]
            break
    if not rating_col:
        raise ValueError("Could not find an Elo rating column in seeds CSV.")

    seeds: Dict[str, float] = {}
    for _, r in df.iterrows():
        tname = canon_team_name(str(r[name_col]))
        seeds[tname] = float(r[rating_col])
    if len(seeds) != 30:
        # not fatal, but warn loudly
        print(f"[ELO] WARNING: seeds count = {len(seeds)} (expected 30)")
    return seeds

# ---- Core builder: writes 30 Parquets, one per TEAM_ID ----
def build_elo_team_parquets(
    *,
    output_dir: str,
    seeds_csv_2015_16: str,
    seasons_full: Iterable[str] = (
        # if you need earlier, append ("2011-12", ... "2014-15")
        "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ),
    init_rating: float = 1505.0,
    hca_points: float = 100.0,
) -> None:
    """
    One pass over league logs; persist 30 team-scope Parquets under `output_dir`.

    - Starts the 2015-16 season from your provided seeds CSV.
    - For later seasons, ratings carry over as 0.75 * R + 0.25 * init_rating (same policy used in your builder).
    - Each Parquet contains all rows for that team across all seasons with the columns listed in the docstring.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ratings keyed by canonical *team name* (stable across rebrands), but we also track TEAM_IDs
    ratings: Dict[str, float] = defaultdict(lambda: init_rating)
    # team_id map (name -> latest TEAM_ID encountered in logs)
    name_to_id: Dict[str, int] = {}

    # Where we accumulate per-team rows before writing out in one shot
    per_team_rows: Dict[int, list] = defaultdict(list)

    # GAME_IDX per (season, team_id)
    game_idx: Dict[Tuple[str, int], int] = defaultdict(int)

    # --- Seed 2015-16 from CSV ---
    seeds = load_seeds_2015_16(seeds_csv_2015_16)

    prev_season: Optional[str] = None
    for season in seasons_full:
        # If we are at the beginning of a post-seed season, apply 0.75 carryover
        if prev_season is not None:
            for tname in list(ratings.keys()):
                ratings[tname] = 0.75 * ratings[tname] + 0.25 * init_rating

        # If this is exactly 2015-16, override with seeds
        if season == "2015-16":
            for tname, r0 in seeds.items():
                ratings[tname] = float(r0)
            print(f"[ELO] Loaded {len(seeds)} seeds for 2015-16 from {seeds_csv_2015_16}")

        print(f"[ELO] === Begin season {season} ===")
        log = sg.get_league_game_log(season).copy()
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
        log["TEAM_NAME_CANON"] = log["TEAM_NAME"].map(canon_team_name)
        log = log.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

        n_games = log["GAME_ID"].nunique()
        processed = 0

        # Process each NBA game (two team rows per GAME_ID)
        for gid, pair in log.groupby("GAME_ID", sort=False):
            if len(pair) != 2:
                continue

            # Identify home/away
            idx_home = pair["MATCHUP"].str.contains("vs.").idxmax()
            idx_away = (set(pair.index) - {idx_home}).pop()
            row_home = pair.loc[idx_home]
            row_away = pair.loc[idx_away]

            # Canonical names and numeric IDs
            home_name = row_home["TEAM_NAME_CANON"]
            away_name = row_away["TEAM_NAME_CANON"]
            home_id = int(row_home["TEAM_ID"])
            away_id = int(row_away["TEAM_ID"])
            name_to_id[home_name] = home_id
            name_to_id[away_name] = away_id

            # Pre-game ratings
            R_home = float(ratings[home_name])
            R_away = float(ratings[away_name])

            # Expectation (home gets HCA in the logistic only)
            delta_home = (R_home - R_away + hca_points)
            E_home = _expected_from_delta(delta_home)
            E_away = 1.0 - E_home

            # Actual outcomes
            S_home = 1.0 if row_home["WL"] == "W" else 0.0
            S_away = 1.0 - S_home

            # MOV (absolute, winner perspective)
            if "PLUS_MINUS" in pair.columns and not pd.isna(row_home["PLUS_MINUS"]):
                mov = abs(float(row_home["PLUS_MINUS"]))
            else:
                mov = abs(float(row_home["PTS"]) - float(row_away["PTS"]))

            # K based on winner perspective delta
            winner_is_home = S_home == 1.0
            if winner_is_home:
                winner_delta = (R_home - R_away + hca_points)
            else:
                winner_delta = (R_away - R_home - hca_points)
            k = _k_mov_sensitive(mov, winner_delta)

            # Post updates
            R_home_post = R_home + k * (S_home - E_home)
            R_away_post = R_away + k * (S_away - E_away)

            gdate = pd.to_datetime(row_home["GAME_DATE"])

            # Per-season running index
            game_idx[(season, home_id)] += 1
            game_idx[(season, away_id)] += 1

            # Append two team-game rows to their respective buckets
            per_team_rows[home_id].append({
                "SEASON": season,
                "GAME_DATE": gdate,
                "GAME_ID": gid,
                "TEAM_ID": home_id,
                "OPP_TEAM_ID": away_id,
                "ELO_PRE": R_home,
                "ELO_POST": R_home_post,
                "E_EXPECTED": E_home,
                "MOV": mov,
                "HCA_USED": hca_points,
                "K_USED": k,
                "GAME_IDX": game_idx[(season, home_id)],
            })
            per_team_rows[away_id].append({
                "SEASON": season,
                "GAME_DATE": gdate,
                "GAME_ID": gid,
                "TEAM_ID": away_id,
                "OPP_TEAM_ID": home_id,
                "ELO_PRE": R_away,
                "ELO_POST": R_away_post,
                "E_EXPECTED": E_away,
                "MOV": mov,
                "HCA_USED": 0.0,
                "K_USED": k,
                "GAME_IDX": game_idx[(season, away_id)],
            })

            # Commit new ratings
            ratings[home_name] = R_home_post
            ratings[away_name] = R_away_post

            processed += 1
            if processed % 200 == 0:
                print(f"[ELO] {season}: processed ~{processed} games of {n_games}")

        prev_season = season
        print(f"[ELO] === End season {season} | games={processed} ===")


    # ---- Write 30 Parquets (one per TEAM_ID) ----
    for team_id, rows in per_team_rows.items():
        df = pd.DataFrame(rows).sort_values(["SEASON", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        # Partition by team id (file-per-team)
        out_path = os.path.join(output_dir, f"elo_team_{team_id}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[ELO] wrote {out_path}  ({len(df)} rows)")

# ---- Tiny O(1) accessor with an in-process LRU on each file ----
@lru_cache(maxsize=64)
def _load_team_parquet(output_dir: str, team_id: int) -> pd.DataFrame:
    path = os.path.join(output_dir, f"elo_team_{team_id}.parquet")
    return pd.read_parquet(path)

def get_elo_for_game(output_dir: str, team_id: int, game_id: str) -> Tuple[float, float]:
    """
    O(1) (hash lookup) on the in-memory dataframe row after a single parquet load per team_id.
    Returns (ELO_PRE, ELO_POST) for (team_id, game_id).
    """
    df = _load_team_parquet(output_dir, team_id)
    row = df.loc[df["GAME_ID"] == game_id]
    if row.empty:
        raise KeyError(f"No Elo row for TEAM_ID={team_id} GAME_ID={game_id}")
    r = row.iloc[0]
    return float(r["ELO_PRE"]), float(r["ELO_POST"])



import os, glob
from functools import lru_cache
from datetime import datetime
import pandas as pd

from stats_getter import canon_team, get_league_game_log

import os, glob
from functools import lru_cache
from datetime import datetime
import pandas as pd

from stats_getter import canon_team, get_league_game_log

ELO_DIR_DEFAULT = "data/elo_ledgers_by_team"

@lru_cache(maxsize=64)
def _load_team_parquet(output_dir: str, team_id: int) -> pd.DataFrame:
    path = os.path.join(output_dir, f"elo_team_{team_id}.parquet")
    df = pd.read_parquet(path)
    return df.sort_values(["SEASON","GAME_DATE","GAME_ID"]).reset_index(drop=True)

def _season_from_date(d: datetime) -> str:
    # NBA season label by calendar date: Aug–Dec belong to season Y-(Y+1); Jan–Jul to (Y-1)-Y
    y = d.year
    if d.month >= 8:
        return f"{y}-{str((y+1)%100).zfill(2)}"
    else:
        return f"{y-1}-{str(y%100).zfill(2)}"

def _resolve_game_id_for_date(home_team: str, away_team: str, d: datetime) -> str | None:
    season = _season_from_date(d)
    lg = get_league_game_log(season).copy()
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"]).dt.normalize()
    dn = pd.to_datetime(d).normalize()

    # Use stable numeric IDs for matching
    hid = _team_id_from_log(home_team, season)
    aid = _team_id_from_log(away_team, season)

    # Find the HOME row for that date (MATCHUP has "vs.")
    mask_home = (
        (lg["GAME_DATE"] == dn) &
        (lg["TEAM_ID"] == hid) &
        (lg["MATCHUP"].str.contains(r"vs\.", regex=True))
    )
    if mask_home.any():
        return str(lg.loc[mask_home, "GAME_ID"].iloc[0])

    # (Rare) fallback: derive from AWAY row if present
    mask_away = (
        (lg["GAME_DATE"] == dn) &
        (lg["TEAM_ID"] == aid) &
        (lg["MATCHUP"].str.contains(r"@", regex=True))
    )
    if mask_away.any():
        return str(lg.loc[mask_away, "GAME_ID"].iloc[0])

    return None


def _team_id_from_log(team_name: str, season: str) -> int:
    lg = get_league_game_log(season).copy()

    # Primary: use the stable numeric TEAM_ID for the (modern/canonical) name
    tid = sg.get_team_id(team_name)
    if (lg["TEAM_ID"] == tid).any():
        return int(tid)

    # Fallback: match legacy/modern strings if TEAM_ID path didn’t hit
    names = _name_aliases(team_name)
    sel = lg[lg["TEAM_NAME"].isin(names)]
    if not sel.empty:
        return int(sel["TEAM_ID"].iloc[0])

    # Helpful error
    raise ValueError(
        f"Could not resolve TEAM_ID for {team_name} in {season}. "
        f"Tried TEAM_ID={tid} and names={names}. "
        f"Seen names sample: {sorted(lg['TEAM_NAME'].unique())[:10]}"
    )


def _elo_as_of_date(output_dir: str, team_id: int, d: datetime) -> float:
    """
    Elo 'as of the start of date d' (i.e., after all games strictly before d).
    If team's first game is on/after d, returns that first row's ELO_PRE (seed/carry).
    """
    df = _load_team_parquet(output_dir, team_id)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()

    prior = df[df["GAME_DATE"] < d.normalize()]
    if not prior.empty:
        # last completed game before d → take its post-game rating
        return float(prior.iloc[-1]["ELO_POST"])

    # No prior game: use the first available row's ELO_PRE (seed/carryover)
    first = df.iloc[0]
    return float(first["ELO_PRE"])

def get_ELO_home(home_team: str, away_team: str, date, output_dir: str = ELO_DIR_DEFAULT) -> float:
    """
    Pre-game Elo for the home team on `date`:
      - If the teams play that day, returns that row's ELO_PRE.
      - Otherwise, returns Elo as of the start of `date` (all games strictly before date).
    """
    d = date if isinstance(date, pd.Timestamp) else (
        pd.to_datetime(date) if isinstance(date, str) else pd.to_datetime(date)
    )
    home_team = canon_team(home_team)
    away_team = canon_team(away_team)

    season = _season_from_date(d)
    gid = _resolve_game_id_for_date(home_team, away_team, d)

    team_id = _team_id_from_log(home_team, season)
    df = _load_team_parquet(output_dir, team_id)

    if gid is not None:
        row = df.loc[df["GAME_ID"].astype(str) == str(gid)]
        if not row.empty:
            return float(row.iloc[0]["ELO_PRE"])

    # Fallback: Elo 'as of' the start of the date
    return _elo_as_of_date(output_dir, team_id, d)

def get_ELO_away(home_team: str, away_team: str, date, output_dir: str = ELO_DIR_DEFAULT) -> float:
    """
    Pre-game Elo for the away team on `date` (same semantics as get_ELO_home).
    """
    d = date if isinstance(date, pd.Timestamp) else (
        pd.to_datetime(date) if isinstance(date, str) else pd.to_datetime(date)
    )
    home_team = canon_team(home_team)
    away_team = canon_team(away_team)

    season = _season_from_date(d)
    gid = _resolve_game_id_for_date(home_team, away_team, d)

    team_id = _team_id_from_log(away_team, season)
    df = _load_team_parquet(output_dir, team_id)

    if gid is not None:
        row = df.loc[df["GAME_ID"].astype(str) == str(gid)]
        if not row.empty:
            return float(row.iloc[0]["ELO_PRE"])

    return _elo_as_of_date(output_dir, team_id, d)


# ---- Seeds loader for starting 2013-14 from 2012-13 FINAL Elo ----
def load_final_elo_2012_13(path_csv: str) -> Dict[str, float]:
    """
    Return {canonical_team_name: elo_final} to seed the START of 2013-14.
    Accepts columns like: TEAM_NAME / Team / Name and ELO / rating / final_elo.
    Canonicalizes names to be stable across rebrands (e.g., Bobcats→Hornets).
    """
    import pandas as pd
    df = pd.read_csv(path_csv)
    cols = {c.lower(): c for c in df.columns}

    # team-name column
    name_col = None
    for cand in ("team_name", "team", "name"):
        if cand in cols:
            name_col = cols[cand]
            break
    if not name_col:
        raise ValueError("Could not find a team-name column in 2012-13 final Elo CSV.")

    # rating column
    rating_col = None
    for cand in ("elo", "final_elo", "rating", "final", "elo_final"):
        if cand in cols:
            rating_col = cols[cand]
            break
    if not rating_col:
        raise ValueError("Could not find an Elo rating column in 2012-13 final Elo CSV.")

    seeds: Dict[str, float] = {}
    for _, r in df.iterrows():
        tname = canon_team_name(str(r[name_col]))  # map historical names
        seeds[tname] = float(r[rating_col])

    if len(seeds) != 30:
        print(f"[ELO] WARNING: 2012-13 final seeds count = {len(seeds)} (expected 30)")
    return seeds


def build_elo_team_parquets_from_2013(
    *,
    output_dir: str,
    seeds_csv_final_2012_13: str,
    seasons_full: Iterable[str] = (
        "2013-14", "2014-15",
        "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ),
    init_rating: float = 1505.0,
    hca_points: float = 100.0,
) -> None:
    """
    Build/write 30 per-team Parquets from 2013-14→present using the *final* 2012-13
    Elo ratings as the season-opening seeds for 2013-14.

    Notes:
    - EXACTLY at 2013-14: overwrite all team ratings with the CSV finals.
    - For every *subsequent* offseason: apply 0.75*R + 0.25*init_rating.
    - Output schema/behavior matches your existing builder so downstream code keeps working.
    """
    import os
    import pandas as pd
    from collections import defaultdict
    from typing import Dict, Tuple, Optional

    os.makedirs(output_dir, exist_ok=True)

    ratings: Dict[str, float] = defaultdict(lambda: init_rating)  # keyed by canonical team name
    name_to_id: Dict[str, int] = {}
    per_team_rows: Dict[int, list] = defaultdict(list)
    game_idx: Dict[Tuple[str, int], int] = defaultdict(int)

    # Load finals from 2012-13 → seeds for the start of 2013-14
    seeds_2013 = load_final_elo_2012_13(seeds_csv_final_2012_13)

    prev_season: Optional[str] = None
    for season in seasons_full:
        # Offseason regression for every season AFTER the first seeded season
        if prev_season is not None:
            for tname in list(ratings.keys()):
                ratings[tname] = 0.75 * ratings[tname] + 0.25 * init_rating

        # Seed the first season (2013-14) from final_elo_2012_13.csv
        if season == "2013-14":
            for tname, r0 in seeds_2013.items():
                ratings[tname] = float(r0)
            print(f"[ELO] Loaded {len(seeds_2013)} seeds for 2013-14 from {seeds_csv_final_2012_13}")

        print(f"[ELO] === Begin season {season} ===")
        log = sg.get_league_game_log(season).copy()
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
        log["TEAM_NAME_CANON"] = log["TEAM_NAME"].map(canon_team_name)
        log = log.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)

        n_games = log["GAME_ID"].nunique()
        processed = 0

        for gid, pair in log.groupby("GAME_ID", sort=False):
            if len(pair) != 2:
                continue

            idx_home = pair["MATCHUP"].str.contains("vs.").idxmax()
            idx_away = (set(pair.index) - {idx_home}).pop()
            row_home = pair.loc[idx_home]
            row_away = pair.loc[idx_away]

            home_name = row_home["TEAM_NAME_CANON"]
            away_name = row_away["TEAM_NAME_CANON"]
            home_id = int(row_home["TEAM_ID"])
            away_id = int(row_away["TEAM_ID"])
            name_to_id[home_name] = home_id
            name_to_id[away_name] = away_id

            R_home = float(ratings[home_name])
            R_away = float(ratings[away_name])

            delta_home = (R_home - R_away + hca_points)
            E_home = _expected_from_delta(delta_home)
            E_away = 1.0 - E_home

            S_home = 1.0 if row_home["WL"] == "W" else 0.0
            S_away = 1.0 - S_home

            if "PLUS_MINUS" in pair.columns and not pd.isna(row_home["PLUS_MINUS"]):
                mov = abs(float(row_home["PLUS_MINUS"]))
            else:
                mov = abs(float(row_home["PTS"]) - float(row_away["PTS"]))

            winner_is_home = S_home == 1.0
            if winner_is_home:
                winner_delta = (R_home - R_away + hca_points)
            else:
                winner_delta = (R_away - R_home - hca_points)
            k = _k_mov_sensitive(mov, winner_delta)

            R_home_post = R_home + k * (S_home - E_home)
            R_away_post = R_away + k * (S_away - E_away)

            gdate = pd.to_datetime(row_home["GAME_DATE"])

            game_idx[(season, home_id)] += 1
            game_idx[(season, away_id)] += 1

            per_team_rows[home_id].append({
                "SEASON": season,
                "GAME_DATE": gdate,
                "GAME_ID": gid,
                "TEAM_ID": home_id,
                "OPP_TEAM_ID": away_id,
                "ELO_PRE": R_home,
                "ELO_POST": R_home_post,
                "E_EXPECTED": E_home,
                "MOV": mov,
                "HCA_USED": hca_points,
                "K_USED": k,
                "GAME_IDX": game_idx[(season, home_id)],
            })
            per_team_rows[away_id].append({
                "SEASON": season,
                "GAME_DATE": gdate,
                "GAME_ID": gid,
                "TEAM_ID": away_id,
                "OPP_TEAM_ID": home_id,
                "ELO_PRE": R_away,
                "ELO_POST": R_away_post,
                "E_EXPECTED": E_away,
                "MOV": mov,
                "HCA_USED": 0.0,
                "K_USED": k,
                "GAME_IDX": game_idx[(season, away_id)],
            })

            ratings[home_name] = R_home_post
            ratings[away_name] = R_away_post

            processed += 1
            if processed % 200 == 0:
                print(f"[ELO] {season}: processed ~{processed} games of {n_games}")

        prev_season = season
        print(f"[ELO] === End season {season} | games={processed} ===")

    # Write 30 Parquets (one per TEAM_ID)
    for team_id, rows in per_team_rows.items():
        df = pd.DataFrame(rows).sort_values(["SEASON", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        out_path = os.path.join(output_dir, f"elo_team_{team_id}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[ELO] wrote {out_path}  ({len(df)} rows)")


# save as tools/write_final_elo_2024_25_from_parquets.py (or anywhere in your repo)
def write_final_elo_csv_from_parquets(
    parquet_dir: str = "data/elo_ledgers_by_team",
    season: str = "2024-25",
    out_csv: str = "final_elo_2024_25_from_parquets.csv",
):
    """
    Read each team parquet (elo_team_<TEAM_ID>.parquet), grab the final ELO for `season`,
    and write a CSV with TEAM_ID, TEAM_NAME, ELO_FINAL sorted by ELO_FINAL desc.
    """
    import os, glob
    import pandas as pd

    # --- collect all team parquets ---
    paths = sorted(glob.glob(os.path.join(parquet_dir, "elo_team_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    # --- build TEAM_ID -> TEAM_NAME (canonical) mapping from league log ---
    try:
        import stats_getter as sg
    except Exception:
        sg = __import__("stats_getter")  # fallback import style

    log = sg.get_league_game_log(season).copy()
    id_to_name = (
        log[["TEAM_ID", "TEAM_NAME"]]
        .drop_duplicates()
        .assign(TEAM_NAME=lambda d: d["TEAM_NAME"].map(sg.canon_team))
    )

    # --- gather final ELO per team for the target season ---
    rows = []
    for p in paths:
        try:
            df = pd.read_parquet(
                p,
                columns=["SEASON", "GAME_DATE", "GAME_ID", "TEAM_ID", "ELO_POST"],
            )
        except Exception:
            df = pd.read_parquet(p)  # older pyarrow might not support columns=

        df = df[df["SEASON"] == season]
        if df.empty:
            # No rows for this team in the target season (shouldn't happen, but guard anyway)
            continue

        # Take last game chronologically (tie-break by GAME_ID just in case)
        df = df.sort_values(["GAME_DATE", "GAME_ID"])
        last = df.iloc[-1]

        rows.append(
            {
                "TEAM_ID": int(last["TEAM_ID"]),
                "ELO_FINAL": float(last["ELO_POST"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No final ELOs collected for season {season} from {parquet_dir}")

    # attach names
    out = out.merge(id_to_name, on="TEAM_ID", how="left")
    # if any names are missing, try to fill with a stable fallback
    if out["TEAM_NAME"].isna().any():
        # fallback: leave TEAM_NAME as "<Unknown TEAM_ID>"
        out["TEAM_NAME"] = out.apply(
            lambda r: r["TEAM_NAME"] if pd.notna(r["TEAM_NAME"]) else f"TEAM_{int(r['TEAM_ID'])}",
            axis=1,
        )

    # reorder & sort
    out = out[["TEAM_ID", "TEAM_NAME", "ELO_FINAL"]].sort_values("ELO_FINAL", ascending=False)

    # basic sanity checks
    missing = set(id_to_name["TEAM_ID"]) - set(out["TEAM_ID"])
    if missing:
        print(f"[WARN] Missing {len(missing)} team(s) with no rows for {season}: {sorted(missing)}")

    print(f"[INFO] Writing {len(out)} rows → {out_csv}")
    out.to_csv(out_csv, index=False)

    # pretty print top/bottom few in console
    with pd.option_context("display.max_rows", None):
        print(out.head(5))
        print("…")
        print(out.tail(5))


if __name__ == "__main__":
    # adjust paths if needed
    write_final_elo_csv_from_parquets(
        parquet_dir="data/elo_ledgers_by_team",
        season="2024-25",
        out_csv="final_elo_2024_25_from_parquets.csv",
    )


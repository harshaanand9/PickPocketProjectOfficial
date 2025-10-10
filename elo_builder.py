# elo_builder.py
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Tuple, Dict, Optional

import math
import pandas as pd

# Your stats getter (assumed available in repo)
import stats_getter as sg


# --- Canonicalization for historical rebrands (team *names* in leaguegamelog) ---
_NAME_CANON = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
}
def canon_team_name(name: str) -> str:
    return _NAME_CANON.get(name, name)


def _is_home(matchup: str) -> bool:
    """NBA 'MATCHUP' string has 'vs.' for home, '@' for away."""
    # Examples: "NYK vs. BOS" -> home; "NYK @ BOS" -> away
    return "vs." in matchup


def _expected_from_delta(delta: float) -> float:
    """
    Logistic expectation using 400 Elo scale.

    We define delta = (team_rating - opp_rating + HCA_if_home),
    so positive delta => favored team => E close to 1.
    """
    return 1.0 / (1.0 + math.pow(10.0, -delta / 400.0))


def _k_mov_sensitive(mov: float, winner_delta: float) -> float:
    """
    MOV-sensitive K, using your Figure 11:
        k = 20 * ( (MOV_winner + 3)^0.8 ) / ( 7.5 + 0.006 * |elo_difference_winner| )
    """
    return 20.0 * math.pow(mov + 3.0, 0.8) / (7.5 + 0.006 * abs(winner_delta))


def build_elo_seed_ledger(
    seasons: Iterable[str] = ("2010-11","2010-11","2011-12", "2012-13", "2013-14", "2014-15"),
    *,
    init_rating: float = 1505.0,
    hca_points: float = 100.0,  # “538-like” home-court, used only in expectation / delta
    apply_carryover: bool = True,
    csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Build a per-team-game Elo ledger for the seasons provided (default 2011-12..2014-15).

    Returns:
        ledger_df: tidy DataFrame with one row per *team-game* containing:
            [season, game_date, game_id, team_name, opp_name, is_home,
             elo_pre, elo_post, e_expected, s_actual, mov, k_used, hca_used, team_id, opp_team_id]
        seeds_2015_16: dict {canonical_team_name: rating_after_carryover_from_2014_15}
    If parquet_path is provided, writes the ledger to that path as Parquet.
    """
    # Current ratings (by canonical team name)
    ratings: Dict[str, float] = defaultdict(lambda: init_rating)

    rows = []

    prev_season: Optional[str] = None

    for season in seasons:
        # Apply season carryover from prior season (once, before first game of this season)
        if apply_carryover and prev_season is not None:
            for t in list(ratings.keys()):
                ratings[t] = 0.75 * ratings[t] + 0.25 * init_rating
            print(f"[ELO] Carryover applied for start of {season} (0.75*R + 0.25*{init_rating})")

        print(f"[ELO] === Begin season {season} ===")

        # 2) After fetching log
        log = sg.get_league_game_log(season).copy()
        # Normalize types
        log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"])
        # Canonicalize the *names* up front
        log["TEAM_NAME_CANON"] = log["TEAM_NAME"].map(canon_team_name)
        # Stable chronological order: date then GAME_ID (string but sortable)
        log = log.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)
        n_rows = len(log)
        n_games = log["GAME_ID"].nunique()
        print(f"[ELO] {season}: fetched league log -> {n_rows} team-rows ({n_games} games)")
        processed = 0
        # Process games by GAME_ID (each appears twice: once per team)
        for gid, pair in log.groupby("GAME_ID", sort=False):
            # Guard: skip malformed entries
            if len(pair) != 2:
                continue

            # Extract home/away rows using MATCHUP
            # There are always exactly 2 teams; figure out which is home.
            idx_home = pair["MATCHUP"].str.contains("vs.").idxmax()
            idx_away = (set(pair.index) - {idx_home}).pop()

            row_home = pair.loc[idx_home]
            row_away = pair.loc[idx_away]

            # Canonical names
            home = row_home["TEAM_NAME_CANON"]
            away = row_away["TEAM_NAME_CANON"]

            # Pre-game ratings
            R_home = ratings[home]
            R_away = ratings[away]

            # Expectation with HCA applied only to the home team inside the logistic
            delta_home = (R_home - R_away + hca_points)
            E_home = _expected_from_delta(delta_home)
            E_away = 1.0 - E_home  # symmetric

            # Outcomes
            S_home = 1.0 if row_home["WL"] == "W" else 0.0
            S_away = 1.0 - S_home

            # MOV (winner perspective, positive number)
            # Prefer PLUS_MINUS if available; otherwise compute from PTS.
            if "PLUS_MINUS" in pair.columns and not pd.isna(row_home["PLUS_MINUS"]):
                mov = abs(float(row_home["PLUS_MINUS"]))
            else:
                mov = abs(float(row_home["PTS"]) - float(row_away["PTS"]))

            # Winner’s delta for K
            winner_is_home = S_home == 1.0
            if winner_is_home:
                winner_delta = (R_home - R_away + hca_points)
            else:
                winner_delta = (R_away - R_home - hca_points)
            k = _k_mov_sensitive(mov, winner_delta)

            # Post updates
            R_home_post = R_home + k * (S_home - E_home)
            R_away_post = R_away + k * (S_away - E_away)

            # Persist ledger rows (team-game granularity)
            gdate = pd.to_datetime(row_home["GAME_DATE"])
            rows.append(
                {
                    "season": season,
                    "game_date": gdate,
                    "game_id": gid,
                    "team_name": home,
                    "opp_name": away,
                    "is_home": True,
                    "elo_pre": R_home,
                    "elo_post": R_home_post,
                    "e_expected": E_home,
                    "s_actual": S_home,
                    "mov": mov,
                    "k_used": k,
                    "hca_used": hca_points,
                    "team_id": int(row_home["TEAM_ID"]),
                    "opp_team_id": int(row_away["TEAM_ID"]),
                }
            )
            rows.append(
                {
                    "season": season,
                    "game_date": gdate,
                    "game_id": gid,
                    "team_name": away,
                    "opp_name": home,
                    "is_home": False,
                    "elo_pre": R_away,
                    "elo_post": R_away_post,
                    "e_expected": E_away,
                    "s_actual": S_away,
                    "mov": mov,
                    "k_used": k,
                    "hca_used": 0.0,  # away gets 0 in expectation
                    "team_id": int(row_away["TEAM_ID"]),
                    "opp_team_id": int(row_home["TEAM_ID"]),
                }
            )

            # Commit new ratings
            ratings[home] = R_home_post
            ratings[away] = R_away_post

            processed += 1

            if processed % 100 == 0:
                print(f"[ELO] {season}: processed {n_games}/82 games ...")

        rvals = pd.Series(ratings.values())
        print(f"[ELO] === End season {season} === "
            f"processed={processed} games | rating mean={rvals.mean():.2f}, sd={rvals.std():.2f}, "
            f"min={rvals.min():.1f}, max={rvals.max():.1f}")
        
        prev_season = season

    # After 2014-15 finishes, apply carryover once more if you want to seed 2015-16
    seeds_2015_16 = {}
    for t, r in ratings.items():
        seeds_2015_16[t] = 0.75 * r + 0.25 * init_rating if apply_carryover else r

    ledger_df = pd.DataFrame(rows).sort_values(["season", "game_date", "game_id", "team_name"]).reset_index(drop=True)

    if csv_path:
        ledger_df.to_csv(csv_path, index=False)   # <— CSV instead of Parquet

    return ledger_df, seeds_2015_16

import pandas as pd

import pandas as pd

import pandas as pd

def write_final_elo_csv(
    ledger_df: pd.DataFrame,
    season: str = "2012-13",
    out_path: str = "final_elo_2012_13.csv",
    elo_col: str | None = None,
) -> pd.DataFrame:
    """
    Write the final Elo for each (modern) NBA team at the end of `season`.
    - Normalizes legacy team names to modern ones before grouping.
    - Validates you have exactly 30 unique teams after mapping.
    Returns the DataFrame written (sorted by final_elo desc).
    """

    if ledger_df.empty:
        raise ValueError("ledger_df is empty.")

    df = ledger_df[ledger_df["season"] == season].copy()
    if df.empty:
        raise ValueError(f"No rows found for season {season} in the ledger.")

    # --- which column holds the post-game Elo? ---
    if elo_col is None:
        for cand in ("elo_post", "elo_after", "elo"):
            if cand in df.columns:
                elo_col = cand
                break
        else:
            raise KeyError(
                "Could not find an Elo column. "
                f"Available columns: {list(df.columns)}"
            )

    # --- normalize dates for reliable ordering ---
    if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # --- map legacy → modern names BEFORE grouping ---
    name_map = {
        "Charlotte Bobcats": "Charlotte Hornets",
        "New Orleans Hornets": "New Orleans Pelicans",
        "New Jersey Nets": "Brooklyn Nets",
        "LA Clippers": "Los Angeles Clippers",
    }
    df["team_name_norm"] = df["team_name"].map(name_map).fillna(df["team_name"])

    # --- order so tail(1) is the final game per (modern) team ---
    sort_cols = ["team_name_norm", "game_date"]
    if "game_id" in df.columns:
        sort_cols.append("game_id")
    df = df.sort_values(sort_cols)

    final = (
        df.groupby("team_name_norm", as_index=False)
          .tail(1)[["team_name_norm", elo_col]]
          .rename(columns={"team_name_norm": "team_name", elo_col: "final_elo"})
          .sort_values("final_elo", ascending=False)
          .reset_index(drop=True)
    )

    # --- sanity check: ensure 30 teams after mapping ---
    expected_teams = {
        "Atlanta Hawks","Boston Celtics","Brooklyn Nets","Charlotte Hornets",
        "Chicago Bulls","Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets",
        "Detroit Pistons","Golden State Warriors","Houston Rockets","Indiana Pacers",
        "Los Angeles Clippers","Los Angeles Lakers","Memphis Grizzlies","Miami Heat",
        "Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans","New York Knicks",
        "Oklahoma City Thunder","Orlando Magic","Philadelphia 76ers","Phoenix Suns",
        "Portland Trail Blazers","Sacramento Kings","San Antonio Spurs","Toronto Raptors",
        "Utah Jazz","Washington Wizards",
    }
    got = set(final["team_name"])
    missing = sorted(expected_teams - got)
    extras = sorted(got - expected_teams)
    if missing or extras or len(final) != 30:
        print("[ELO FINAL WARNING]")
        if missing:
            print("  Missing:", missing)
        if extras:
            print("  Unexpected:", extras)
        print(f"  Row count: {len(final)} (expected 30)")

    final.to_csv(out_path, index=False)
    return final




if __name__ == "__main__":
    seasons = ("2009-10", "2010-11", "2011-12", "2012-13")
    ledger, seeds = build_elo_seed_ledger(
        seasons=seasons,
        csv_path="elo_seed_2009_2013.csv",   # <— was parquet_path=...
    )

    print("Finished Elo build from 2009–10 through 2012–13")
    print("Sample rows:\n", ledger.head())
    print("\nSeeds for 2013–14 (first 30 teams):")
    for team, rating in list(seeds.items())[:30]:
        print(f"{team}: {rating:.2f}")
    
    ledger_df, _ = build_elo_seed_ledger(seasons=("2009-10","2010-11","2011-12","2012-13"))
    final_elo = write_final_elo_csv(ledger_df, season="2012-13", out_path="final_elo_2012_13.csv")


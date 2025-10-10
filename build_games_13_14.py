# scripts/build_games_13_15.py
# Create a game-level CSV for 2013-14 and 2014-15 (Regular Season),
# with betting fields set to NaN. Sorted by date like training_set.csv.

import numpy as np
import pandas as pd
import stats_getter as sg

OUTFILE = "games_13-15.csv"
SEASONS = ["2013-14", "2014-15"]  # extend if you like

def build_games_for_season(season: str) -> pd.DataFrame:
    print(f"[build] Loading LeagueGameLog for {season} …")
    df = sg.get_league_game_log(season).copy()
    if df is None or df.empty:
        raise RuntimeError(f"LeagueGameLog returned empty for {season}")

    # Keep regular season only (matches your training_set style)
    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"].astype(str).str.contains("Regular", case=False, na=False)].copy()

    # Normalize date + sort stable
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"], kind="mergesort").reset_index(drop=True)

    # Home/away flags from MATCHUP: 'vs.' -> home, '@' -> away
    mu = df["MATCHUP"].astype(str)
    homes = df[mu.str.contains(r"vs\.", regex=True)].copy()
    aways = df[mu.str.contains("@")].copy()

    # Minimal columns, rename to target schema
    homes = homes[["GAME_ID", "GAME_DATE", "TEAM_NAME", "PTS"]].rename(
        columns={"TEAM_NAME": "home_team", "PTS": "home_score"}
    )
    aways = aways[["GAME_ID", "TEAM_NAME", "PTS"]].rename(
        columns={"TEAM_NAME": "away_team", "PTS": "away_score"}
    )

    # Merge (one home + one away per GAME_ID)
    games = pd.merge(homes, aways, on="GAME_ID", how="inner", validate="one_to_one")

    # Sanity: ensure we got every game id present in the log
    expected = df["GAME_ID"].nunique()
    got = games["GAME_ID"].nunique()
    if got != expected:
        missing = sorted(set(df["GAME_ID"].unique()) - set(games["GAME_ID"].unique()))
        raise AssertionError(
            f"[{season}] Mismatch: expected {expected} games, built {got}. "
            f"Missing GAME_IDs sample: {missing[:10]}"
        )

    # Build training_set-like columns
    games["date"] = games["GAME_DATE"].dt.strftime("%m/%d/%Y")
    games["season"] = season

    # Betting fields -> NaN
    games["home_money_line"] = np.nan
    games["away_money_line"] = np.nan
    games["home_spread"] = np.nan
    games["away_spread"] = np.nan

    out_cols = [
        "date",
        "home_team", "away_team",
        "home_money_line", "home_spread",
        "away_money_line", "away_spread",
        "home_score", "away_score",
        "season",
    ]
    return games[out_cols].sort_values("date", kind="mergesort").reset_index(drop=True)

def main():
    frames = [build_games_for_season(s) for s in SEASONS]
    all_games = pd.concat(frames, ignore_index=True)
    # Global sort by date across both seasons (stable)
    all_games = all_games.sort_values("date", kind="mergesort").reset_index(drop=True)
    print(f"[build] Total games: {len(all_games)} "
          f"({SEASONS[0]}: {len(frames[0])}, {SEASONS[1]}: {len(frames[1])})")
    all_games.to_csv(OUTFILE, index=False)
    print(f"[build] Wrote {OUTFILE} ✓")

if __name__ == "__main__":
    main()

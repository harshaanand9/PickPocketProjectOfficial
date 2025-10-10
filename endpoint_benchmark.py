
#!/usr/bin/env python3
"""
NBA Stats API Endpoint Benchmark Harness
----------------------------------------

Purpose
- Measure "computational cost" per endpoint call so you can multiply by your call counts
  to estimate total runtime/CPU/network costs across ~15,000 games and hundreds of features.

What it measures per call
- wall_time_sec: end-to-end elapsed seconds (includes network + server latency + JSON->DataFrame parsing)
- cpu_time_sec: process CPU time (parsing/JSON handling overhead on your machine)
- result_bytes: size of raw JSON text in bytes (proxy for network volume)
- df_shape: (rows, cols) if endpoint returns a DataFrame (best-effort)
- status: "ok" or "error"
- error: error string if any
- retries: how many retries were attempted
- http_status: best-effort extraction if available

Run examples
- python3 endpoint_benchmark.py --season 2024-25 --prev-season 2023-24 --samples 3
- python3 endpoint_benchmark.py --season 2019-20 --prev-season 2018-19 --samples 1 --timeout 15
- Configure proxies via env (see "Proxies" section below).

Notes
- Uses representative/sample calls for each of your listed endpoints.
- For BoxScore* endpoints that require a GAME_ID, we automatically fetch one GAME_ID from
  LeagueGameLog for the indicated season(s). You can pin a specific GAME_ID via CLI.
- For "second half of previous season" windows, we default to Feb 10 → Apr 30 of the end-year.
  Override with --second-half-start and --second-half-end if you want exact dates (MM/DD/YYYY).

Dependencies
- pip install nba_api pandas psutil (psutil optional; included here for future extension)
"""

import argparse, json, os, sys, time, traceback
from datetime import datetime
from typing import Any, Dict, Tuple

import pandas as pd

# nba_api endpoints
from nba_api.stats.endpoints import (
    LeagueGameLog,
    LeagueDashTeamStats,
    LeagueDashPlayerStats,
    LeagueDashTeamPtShot,
    LeagueDashTeamShotLocations,
    LeagueDashTeamClutch,
    TeamPlayerOnOffDetails,
    BoxScoreAdvancedV2,
    BoxScoreMiscV2,
    HustleStatsBoxScore,
)

# --------------- Helpers ---------------

def mmddyyyy(date_str: str) -> str:
    """Validate/normalize MM/DD/YYYY."""
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%m/%d/%Y")
    except ValueError:
        raise SystemExit(f"Bad date format (expected MM/DD/YYYY): {date_str!r}")

def season_start_end(season: str) -> Tuple[str, str]:
    """
    Rough regular-season bounds for a given 'YYYY-YY' season, used for sampling.
    Start: Oct 01 of start year, End: Apr 30 of end year.
    """
    start_year = int(season[:4])
    end_year = start_year + 1
    return f"10/01/{start_year}", f"04/30/{end_year}"

def second_half_bounds(prev_season: str, default_start="02/10", default_end="04/30") -> Tuple[str, str]:
    """
    "Second half of previous season" defaults.
    Defaults to Feb 10 -> Apr 30 of (prev_season end year).
    Override via CLI for exactness.
    """
    start_year = int(prev_season[:4])
    end_year = start_year + 1
    start = f"{default_start}/{end_year}"
    end = f"{default_end}/{end_year}"
    # validate
    return mmddyyyy(start), mmddyyyy(end)

def pick_sample_game_id(season: str, who="T", timeout=12.0) -> str:
    """
    Grab one GAME_ID from LeagueGameLog for this season.
    who: "T" (teams) or "P" (players). We'll dedupe and take the first.
    """
    lg = LeagueGameLog(season=season, season_type_all_star="Regular Season",
                       player_or_team_abbreviation=who, timeout=timeout)
    df = lg.get_data_frames()[0]
    df = df.drop_duplicates(subset=["GAME_ID"]).sort_values(["GAME_DATE"], ascending=True)
    return str(df.iloc[0]["GAME_ID"])

def size_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj))
    except Exception:
        try:
            return len(str(obj).encode("utf-8", errors="ignore"))
        except Exception:
            return -1

def time_call(label: str, func, *args, **kwargs) -> Dict[str, Any]:
    """
    Execute `func(*args, **kwargs)` measuring wall/CPU time and extracting useful metadata.
    Returns a dict record.
    """
    record = dict(endpoint=label, wall_time_sec=None, cpu_time_sec=None, result_bytes=None,
                  df_rows=None, df_cols=None, status="ok", error="", retries=0, http_status=None)
    start_wall = time.perf_counter()
    start_cpu  = time.process_time()
    try:
        obj = func(*args, **kwargs)  # nba_api call returns an endpoint object
        # Attempt to capture HTTP status if exposed via response attribute (not guaranteed)
        try:
            record["http_status"] = getattr(obj, "status_code", None)
        except Exception:
            pass

        # Convert to JSON and DataFrame (best effort)
        try:
            raw = obj.get_json()
            record["result_bytes"] = len(raw.encode("utf-8", errors="ignore"))
        except Exception:
            # Some endpoints only support get_data_frames()
            raw = None
            record["result_bytes"] = None

        try:
            dfs = obj.get_data_frames()
            if len(dfs) > 0:
                r, c = dfs[0].shape
                record["df_rows"], record["df_cols"] = int(r), int(c)
        except Exception:
            pass

    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["result_bytes"] = None
    finally:
        record["wall_time_sec"] = round(time.perf_counter() - start_wall, 6)
        record["cpu_time_sec"]  = round(time.process_time() - start_cpu, 6)
    return record

# --------------- Benchmark set-up ---------------

def build_bench_plan(args) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict of endpoint-name -> config with callable and parameters.
    Uses your specific list with sane defaults for parameters and windows.
    """
    season = args.season
    prev_season = args.prev_season
    t_start, t_end = season_start_end(season)
    p2_start, p2_end = second_half_bounds(prev_season,
                                          default_start=args.second_half_start,
                                          default_end=args.second_half_end)

    # Sample GAME_IDs for BoxScore* and HustleStatsBoxScore
    sample_game_id_currT = args.sample_game_id or pick_sample_game_id(season, "T", timeout=args.timeout)
    sample_game_id_prevT = pick_sample_game_id(prev_season, "T", timeout=args.timeout)

    # A representative team_id for endpoints that require one.
    # We derive it by scanning a LeagueDashTeamStats response.
    team_id = None
    try:
        lds = LeagueDashTeamStats(season=season, timeout=args.timeout)
        df = lds.get_data_frames()[0]
        team_id = int(df.iloc[0]["TEAM_ID"])
    except Exception:
        # Fallback: common team id
        team_id = 1610612747  # LAL as a stable default
    # ------------------ Plan ------------------
    plan = {
        # LeagueGameLog (Team vs Player modes)
        "LeagueGameLog_Current_T": dict(call=lambda: LeagueGameLog(
            season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="T",
            timeout=args.timeout)),
        "LeagueGameLog_Prev_T": dict(call=lambda: LeagueGameLog(
            season=prev_season, season_type_all_star="Regular Season", player_or_team_abbreviation="T",
            timeout=args.timeout)),
        "LeagueGameLog_Current_P": dict(call=lambda: LeagueGameLog(
            season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="P",
            timeout=args.timeout)),
        "LeagueGameLog_Prev_P": dict(call=lambda: LeagueGameLog(
            season=prev_season, season_type_all_star="Regular Season", player_or_team_abbreviation="P",
            timeout=args.timeout)),

        # BoxScore* & Hustle by GAME_ID
        "BoxScoreAdvancedV2": dict(call=lambda: BoxScoreAdvancedV2(game_id=sample_game_id_currT, timeout=args.timeout)),
        "BoxScoreMiscV2":     dict(call=lambda: BoxScoreMiscV2(game_id=sample_game_id_currT, timeout=args.timeout)),
        "HustleStatsBoxScore":dict(call=lambda: HustleStatsBoxScore(game_id=sample_game_id_currT, timeout=args.timeout)),

        # LeagueDashTeamClutch (Advanced), team_id=None
        "LeagueDashTeamClutch_Advanced": dict(call=lambda: LeagueDashTeamClutch(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="PerGame", league_id_nullable="00",
            timeout=args.timeout)),

        # LeagueDashTeamStats (Base / Scoring / Advanced / Four Factors)
        "LeagueDashTeamStats_Base": dict(call=lambda: LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Base", league_id_nullable = "00", timeout=args.timeout)),
        "LeagueDashTeamStats_Scoring": dict(call=lambda: LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Scoring", league_id_nullable = "00", timeout=args.timeout)),
        "LeagueDashTeamStats_Advanced_2HPrev": dict(call=lambda: LeagueDashTeamStats(
            season=prev_season, measure_type_detailed_defense="Advanced", league_id_nullable = "00", date_from_nullable=p2_start, date_to_nullable=p2_end,
            timeout=args.timeout)),
        "LeagueDashTeamStats_FourFactors_2HPrev": dict(call=lambda: LeagueDashTeamStats(
            season=prev_season, measure_type_detailed_defense="Four Factors", league_id_nullable = "00", date_from_nullable=p2_start, date_to_nullable=p2_end,
            timeout=args.timeout)),

        # LeagueDashPTShots (requires team_id)
        "LeagueDashPTShots_Team": dict(call=lambda: LeagueDashTeamPtShot(
            season=season, team_id_nullable=team_id, league_id = "00", timeout=args.timeout)),

        # LeagueDashTeamShotLocations (team_id=None)
        "LeagueDashTeamShotLocations_AllTeams": dict(call=lambda: LeagueDashTeamShotLocations(
            season=season, league_id_nullable="00", timeout=args.timeout)),

        # LeagueDashPlayerStats (Base) for 2H of previous season
        "LeagueDashPlayerStats_Base_2HPrev": dict(call=lambda: LeagueDashPlayerStats(
            season=prev_season, measure_type_detailed_defense="Base",
            date_from_nullable=p2_start, date_to_nullable=p2_end, timeout=args.timeout)),

        # TeamPlayerOnOffDetails (Advanced) for 2H prev + generic team
        "TeamPlayerOnOffDetails_Advanced_2HPrev": dict(call=lambda: TeamPlayerOnOffDetails(
            season=prev_season, team_id=team_id, measure_type_detailed_defense="Advanced",
            date_from_nullable=p2_start, date_to_nullable=p2_end, timeout=args.timeout)),
        "TeamPlayerOnOffDetails_Advanced_Team": dict(call=lambda: TeamPlayerOnOffDetails(
            season=season, team_id=team_id, measure_type_detailed_defense="Advanced",
            timeout=args.timeout)),
    }
    return plan

# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark selected NBA Stats API endpoints.")
    ap.add_argument("--season", required=True, help="Target season, e.g., 2024-25")
    ap.add_argument("--prev-season", required=True, help="Previous season, e.g., 2023-24")
    ap.add_argument("--samples", type=int, default=3, help="How many repeated calls per endpoint (median reported)")
    ap.add_argument("--timeout", type=float, default=12.0, help="Per-request timeout (seconds)")
    ap.add_argument("--second-half-start", default="02/10", help="MM/DD cutoff for 2H prev-season window (default 02/10)")
    ap.add_argument("--second-half-end",   default="04/30", help="MM/DD cutoff for 2H prev-season window (default 04/30)")
    ap.add_argument("--sample-game-id", default=None, help="Optional GAME_ID to use for BoxScore/Hustle endpoints")
    ap.add_argument("--out", default="benchmark_results.csv", help="CSV output path")
    args = ap.parse_args()

    plan = build_bench_plan(args)

    # Warm up: cheap ping (helps primer DNS/TLS, etc.)
    try:
        _ = LeagueGameLog(season=args.season, season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=args.timeout)
    except Exception:
        pass

    records = []
    for name, cfg in plan.items():
        print(f"==> {name}")
        times = []
        # Run N samples; capture median wall/cpu
        for i in range(args.samples):
            rec = time_call(name, cfg["call"])
            rec["sample_idx"] = i
            records.append(rec)
            times.append(rec["wall_time_sec"])
            if rec["status"] != "ok":
                print(f"   sample {i}: ERROR {rec['error']}  (wall {rec['wall_time_sec']}s)")
            else:
                print(f"   sample {i}: wall={rec['wall_time_sec']} cpu={rec['cpu_time_sec']} bytes={rec['result_bytes']} rows={rec['df_rows']} cols={rec['df_cols']}")

    # Aggregate per endpoint (median of samples for stable estimate)
    df = pd.DataFrame.from_records(records)
    agg = (df.groupby("endpoint")
             .agg(wall_time_sec_median=("wall_time_sec", "median"),
                  cpu_time_sec_median=("cpu_time_sec", "median"),
                  result_bytes_median=("result_bytes", "median"),
                  df_rows_median=("df_rows", "median"),
                  df_cols_median=("df_cols", "median"),
                  err_rate=("status", lambda s: (s != "ok").mean()))
             .reset_index()
          )
    # Add your multiplier (call counts) as a reference column; you can edit these numbers or multiply externally.
    multipliers = {
        "LeagueGameLog_Current_T": 1,
        "LeagueGameLog_Prev_T": 1,
        "LeagueGameLog_Current_P": 1,  # you listed 1x P (current)
        "LeagueGameLog_Prev_P": 1,     # and 1x P (prev)
        "HustleStatsBoxScore": 1230,
        "BoxScoreAdvancedV2": 1230,
        "LeagueDashTeamClutch_Advanced": 160,
        "LeagueDashTeamStats_Base": 160,
        "BoxScoreMiscV2": 1230,
        "LeagueDashTeamStats_Scoring": 160,
        "LeagueDashTeamPtShot_Team": 2460,
        "LeagueDashTeamShotLocations_AllTeams": 160,
        "LeagueDashTeamStats_Advanced_2HPrev": 30,
        "LeagueDashTeamStats_FourFactors_2HPrev": 30,
        "LeagueGameLog_Current_T_DUP": 0,  # placeholder if needed
        "LeagueDashPlayerStats_Base_2HPrev": 30,
        "TeamPlayerOnOffDetails_Advanced_2HPrev": 30,
        "TeamPlayerOnOffDetails_Advanced_Team": 2460,
    }
    agg["multiplier_hint"] = agg["endpoint"].map(multipliers).fillna(0).astype(int)

    # Compute projected_total_wall_time = median_wall * multiplier_hint (your editable estimate column)
    agg["projected_total_wall_time_sec"] = agg["wall_time_sec_median"] * agg["multiplier_hint"]

    # Save both raw samples and aggregated
    out = args.out
    agg_out = out.replace(".csv", "_agg.csv")
    df.to_csv(out, index=False)
    agg.to_csv(agg_out, index=False)

    print(f"\nSaved raw samples to: {out}")
    print(f"Saved aggregates to:  {agg_out}")
    print("\nColumns in *_agg.csv:")
    print(agg.columns.tolist())

    # Pretty print aggregated table
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\n=== Aggregated (median per endpoint) ===")
        print(agg.sort_values("projected_total_wall_time_sec", ascending=False))

if __name__ == "__main__":
    main()

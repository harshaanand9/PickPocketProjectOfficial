def debug_calc_adv_only(home_team, away_team, date, season,
                        mutate_ledgers: bool = True,
                        precision: int = 6, quiet: bool = False):
    """
    Replica-style debug helper for calculate_game_features:
      - Calls the SAME feature functions you use in calculate_game_features:
          get_pace_diff, get_oreb_pct_relative, get_dreb_pct_relative
      - Prints those 3 feature values
      - Prints each team's running average (prior) up to the *day before* the game
      - Mutates ledgers the same way your pipeline does: append current game AFTER printing
      - No bulky ensure_* sweeps
    """
    import math, time
    from datetime import datetime
    import pandas as pd

    # ---- match calculate_game_features date handling ----
    date_str = date if isinstance(date, str) else date.strftime("%m/%d/%Y")
    dt = datetime.strptime(date_str, "%m/%d/%Y")

    # ---- feature calls (identical call shapes) ----
    t0 = time.perf_counter()
    pace_diff = get_pace_diff(home_team, away_team, season, season, date_str)
    t_pace = time.perf_counter() - t0

    t0 = time.perf_counter()
    oreb_rel = get_oreb_pct_relative(home_team, away_team, season, season, date_str)
    t_oreb = time.perf_counter() - t0

    t0 = time.perf_counter()
    dreb_rel = get_dreb_pct_relative(home_team, away_team, season, season, date_str)
    t_dreb = time.perf_counter() - t0

    # ---- priors (running avgs up to day-before) ----
    from stats_getter import get_team_id
    from advanced_ledger import get_prior_pace, get_prior_oreb_pct, get_prior_dreb_pct

    hid = get_team_id(home_team)
    aid = get_team_id(away_team)

    def _fmt(x):
        try:
            xf = float(x)
            if math.isnan(xf): return "nan"
            return f"{xf:.{precision}g}"
        except Exception:
            return str(x)

    h_pace = get_prior_pace(season, hid, dt) if hid is not None else float("nan")
    a_pace = get_prior_pace(season, aid, dt) if aid is not None else float("nan")
    h_oreb = get_prior_oreb_pct(season, hid, dt) if hid is not None else float("nan")
    a_oreb = get_prior_oreb_pct(season, aid, dt) if aid is not None else float("nan")
    h_dreb = get_prior_dreb_pct(season, hid, dt) if hid is not None else float("nan")
    a_dreb = get_prior_dreb_pct(season, aid, dt) if aid is not None else float("nan")

    # ---- print exactly what you need (and nothing else) ----
    if not quiet:
        print(f"{season}  {date_str}:  {away_team} @ {home_team}")
        print("-" * 75)
        print(f"get_pace_diff:         {_fmt(pace_diff)}   (time {t_pace:.3f}s)")
        print(f"get_oreb_pct_relative: {_fmt(oreb_rel)}    (time {t_oreb:.3f}s)")
        print(f"get_dreb_pct_relative: {_fmt(dreb_rel)}    (time {t_dreb:.3f}s)")
        print("Team priors (running avgs up to day-before):")
        print(f"  HOME {home_team:>18} | PACE={_fmt(h_pace)}  OREB_PCT={_fmt(h_oreb)}  DREB_PCT={_fmt(h_dreb)}")
        print(f"  AWAY {away_team:>18} | PACE={_fmt(a_pace)}  OREB_PCT={_fmt(a_oreb)}  DREB_PCT={_fmt(a_dreb)}")

    # ---- mutate ledgers AFTER computing features (so priors were truly 'up to day-before') ----
    if mutate_ledgers:
        # resolve game_id from league log (same approach you used elsewhere)
        from stats_getter import get_league_game_log
        from advanced_ledger import append_adv_game

        league_log = get_league_game_log(season).copy()
        league_log["_DATE"] = pd.to_datetime(league_log["GAME_DATE"], errors="coerce").dt.normalize()
        dnorm = pd.to_datetime(date_str, errors="coerce").normalize()

        # find the GAME_ID that has both teams on that date
        gids = league_log.loc[
            (league_log["_DATE"] == dnorm) &
            (league_log["TEAM_ID"].isin([hid, aid]))
        ]["GAME_ID"].dropna().unique().tolist()

        gid = None
        if len(gids) == 1:
            gid = str(gids[0])
        else:
            # if multiple rows, pick the one whose entries include both teams
            for g in gids:
                sub = league_log[league_log["GAME_ID"] == g]
                if set(sub["TEAM_ID"].unique()).issuperset({hid, aid}):
                    gid = str(g); break

        if gid:
            append_adv_game(season, gid)
            if not quiet:
                print(f"[ledger] appended GAME_ID={gid}")
        else:
            if not quiet:
                print("[warn] Could not resolve GAME_ID; ledger not updated for this game.")

    # return values so you can assert in tests if you want
    return {
        "pace_diff": pace_diff,
        "oreb_pct_relative": oreb_rel,
        "dreb_pct_relative": dreb_rel,
        "home_pace_prior": h_pace, "home_oreb_pct_prior": h_oreb, "home_dreb_pct_prior": h_dreb,
        "away_pace_prior": a_pace, "away_oreb_pct_prior": a_oreb, "away_dreb_pct_prior": a_dreb,
    }


def debug_adv_only_first_n(n: int, season: str = "2024-25", quiet: bool = False):
    import pandas as pd

    sched = get_season_games(season).copy()
    sched["date"] = pd.to_datetime(sched["date"], errors="coerce")
    sched = sched.sort_values("date").head(int(n)).reset_index(drop=True)

    out = []
    for _, g in sched.iterrows():
        out.append(
            debug_calc_adv_only(
                g["home_team"], g["away_team"], g["date"], season,
                mutate_ledgers=True, quiet=quiet
            )
        )
    return out

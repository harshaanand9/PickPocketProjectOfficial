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
    late_min_minutes_total: float = 36.0,  # e.g., 36.0 (apply only after threshold)
) -> float:
    """
    Sum over active roster (as of `game_date`) of each player's to-date PER-GAME PPG
    (from season start up to the day BEFORE `game_date`), with a dynamic GP threshold:

      - If team_games_to_date < dynamic_after_team_games: require GP >= early_gp_cut (default 1).
      - Else (later in season): require GP >= late_gp_cut (default 3).
        If late_min_minutes_total is set, a player also qualifies if MIN_total >= that value.

    Rationale: keep early-season signal, curb single-game noise later; allow heavy-minutes players
    with slightly < late_gp_cut GP to still count.
    """
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    # --- active roster on date ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)

    # --- aliasing (same style as your roster-change funcs) ---
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
        aliases.update({ _norm(k): v for k, v in name_aliases.items() })

    def _apply_alias(n: str) -> str:
        nn = _norm(n)
        return _norm(aliases.get(nn, n))

    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- pull player logs for current season ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        return 0.0

    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)

    # cutoff: strictly before this game's date
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    # compute team games to date (unique game IDs)
    team_games_to_date = int(df["GAME_ID"].nunique())
    if team_games_to_date == 0:
        return 0.0

    # parse minutes to numeric minutes
    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            # handle "MM:SS"
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all='ignore'):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0) / 60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan  # no minutes field; minutes filter will simply be ignored

    # per-player aggregates
    g = df.groupby("_N", as_index=False).agg(
        GP   = ("GAME_ID", "count"),
        PTS  = ("PTS", "sum"),
        MIN_TOT = ("__MIN_F", "sum"),
    )
    g["PPG"] = g["PTS"] / g["GP"].replace(0, np.nan)

    # decide dynamic GP cut
    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut

    # base qualifier: GP >= gp_cut
    qual_gp = g["GP"] >= gp_cut

    # late-season extra qualifier: minutes total
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    # restrict to active roster & qualifiers
    g = g[(g["_N"].isin(activeN)) & (qualifier)]
    if g.empty:
        return 0.0
    
    print(g)

    return float(np.nansum(g["PPG"].to_numpy()))



def getTeam_PlayerAggregated_ASTTOV_on_date(
    team_name: str,
    game_date: str,                 # "MM/DD/YYYY"
    curr_season: str,               # "2022-23"
    timeline_curr: pd.DataFrame,
    name_aliases: dict[str, str] | None = None,
    # new dynamic threshold controls (match PPG helper):
    dynamic_after_team_games: int = 12,
    early_gp_cut: int = 1,
    late_gp_cut: int = 3,
    late_min_minutes_total: float = 36.0,  # also qualifies late-season if minutes total >= this
) -> float:
    """
    Sum over active roster (as of `game_date`) of each player's to-date PER-GAME (AST - TOV),
    from season start up to the day BEFORE `game_date`, with a dynamic GP/Minutes gate:

      - If team_games_to_date < dynamic_after_team_games: require GP >= early_gp_cut.
      - Else: require GP >= late_gp_cut, OR MIN_total >= late_min_minutes_total (if set).

    Notes:
      * `min_games` and `per_game` are ignored now (always per-game) to mirror PPG helper.
    """
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    # --- active roster on date ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)

    # --- aliasing (same style as PPG helper) ---
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
        aliases.update({ _norm(k): v for k, v in name_aliases.items() })

    def _apply_alias(n: str) -> str:
        nn = _norm(n)
        return _norm(aliases.get(nn, n))

    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- player logs current season ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        return 0.0

    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)

    # cutoff strictly before game_date
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    # team games to date (unique GAME_ID)
    team_games_to_date = int(df["GAME_ID"].nunique())
    if team_games_to_date == 0:
        return 0.0

    # parse minutes to numeric total minutes (for late-season alternative gate)
    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all="ignore"):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0) / 60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan

    # per-player aggregates
    g = df.groupby("_N", as_index=False).agg(
        GP      = ("GAME_ID", "count"),
        AST_SUM = ("AST", "sum"),
        TOV_SUM = ("TOV", "sum"),
        MIN_TOT = ("__MIN_F", "sum"),
    )
    gp = g["GP"].replace(0, np.nan)
    g["ASTmTOV_PG"] = (g["AST_SUM"] / gp) - (g["TOV_SUM"] / gp)

    # dynamic qualifier
    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut
    qual_gp = g["GP"] >= gp_cut
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    # restrict to active roster & qualifiers
    g = g[(g["_N"].isin(activeN)) & qualifier]
    if g.empty:
        return 0.0

    # optional: inspect who passed the gate
    # print(g[["_N","GP","MIN_TOT","ASTmTOV_PG"]])
    print(g)

    return float(np.nansum(g["ASTmTOV_PG"].to_numpy()))




def getTeam_PlayerAggregated_DREBSTOCK_on_date(
    team_name: str,
    game_date: str,                 # "MM/DD/YYYY"
    curr_season: str,               # "2022-23"
    timeline_curr: pd.DataFrame,
    name_aliases: dict[str, str] | None = None,
    # legacy arg (ignored; kept for compatibility):
    min_games: int = 1,
    # new dynamic threshold controls (match PPG helper):
    dynamic_after_team_games: int = 12,
    early_gp_cut: int = 1,
    late_gp_cut: int = 3,
    late_min_minutes_total: float = 36.0,
) -> float:
    """
    Sum over active roster (as of `game_date`) of each player's to-date PER-GAME
    (DREB + STL + BLK), from season start up to the day BEFORE `game_date`,
    with a dynamic GP/Minutes gate identical to the PPG helper.
    """
    import pandas as pd, numpy as np, re, unicodedata
    from datetime import datetime

    sg = stats_getter or __import__("stats_getter")
    team_name = sg.canon_team(team_name)

    # --- active roster on date ---
    active_set_raw = roster_on_date(team_name, game_date, timeline_curr)

    # --- aliasing (same pattern) ---
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
        aliases.update({ _norm(k): v for k, v in name_aliases.items() })

    def _apply_alias(n: str) -> str:
        nn = _norm(n)
        return _norm(aliases.get(nn, n))

    activeN = {_apply_alias(x) for x in active_set_raw}

    # --- player logs current season ---
    logs = sg.get_league_game_log(season=curr_season, player_or_team_abbreviation="P")
    if logs is None or logs.empty:
        return 0.0

    df = logs.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str).str.strip()
    df["TEAM_NAME"]   = df["TEAM_NAME"].astype(str).str.strip()
    df["_N"]          = df["PLAYER_NAME"].apply(_apply_alias)

    # cutoff strictly before game_date
    cutoff = datetime.strptime(game_date, "%m/%d/%Y")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df[(df["TEAM_NAME"] == team_name) & (df["GAME_DATE"] < cutoff)]
    if df.empty:
        return 0.0

    # team games to date (unique GAME_ID)
    team_games_to_date = int(df["GAME_ID"].nunique())
    if team_games_to_date == 0:
        return 0.0

    # parse minutes field for minutes-based late-season qualifier
    if "MIN" in df.columns:
        if df["MIN"].dtype == object:
            mmss = df["MIN"].str.split(":", n=1, expand=True)
            with np.errstate(all="ignore"):
                df["__MIN_F"] = pd.to_numeric(mmss[0], errors="coerce") + pd.to_numeric(mmss[1], errors="coerce").fillna(0) / 60.0
        else:
            df["__MIN_F"] = pd.to_numeric(df["MIN"], errors="coerce")
    else:
        df["__MIN_F"] = np.nan

    # per-player aggregates
    g = df.groupby("_N", as_index=False).agg(
        GP      = ("GAME_ID", "count"),
        DREB_SUM= ("DREB", "sum"),
        STL_SUM = ("STL",  "sum"),
        BLK_SUM = ("BLK",  "sum"),
        MIN_TOT = ("__MIN_F", "sum"),
    )
    gp = g["GP"].replace(0, np.nan)
    g["DREB_STOCK_PG"] = (g["DREB_SUM"] + g["STL_SUM"] + g["BLK_SUM"]) / gp

    # dynamic qualifier
    gp_cut = early_gp_cut if team_games_to_date < dynamic_after_team_games else late_gp_cut
    qual_gp = g["GP"] >= gp_cut
    if (team_games_to_date >= dynamic_after_team_games) and (late_min_minutes_total is not None):
        qual_min = g["MIN_TOT"].fillna(0) >= float(late_min_minutes_total)
        qualifier = qual_gp | qual_min
    else:
        qualifier = qual_gp

    # restrict to active roster & qualifiers
    g = g[(g["_N"].isin(activeN)) & qualifier]
    if g.empty:
        return 0.0

    # optional: debug peek
    # print(g[["_N","GP","MIN_TOT","DREB_STOCK_PG"]])
    print(g)

    return float(np.nansum(g["DREB_STOCK_PG"].to_numpy()))
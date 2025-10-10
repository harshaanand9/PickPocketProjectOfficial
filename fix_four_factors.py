#!/usr/bin/env python3
import os, re
import pandas as pd
from glob import glob

LEDGER_DIR = "/Users/harsha/Desktop/PickPocketProjectOfficial/ledgers/four_factors_ledger"

def _normalize_ff_cols(df: pd.DataFrame, cols=("OREB_PCT",)) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].notna().any():
            q90 = out[c].quantile(0.90)
            if pd.notna(q90) and q90 > 1.5:
                out[c] = out[c] / 100.0
        out[c] = out[c].clip(0.0, 1.0)
    return out

def get_league_game_log(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog
    d = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    d["GAME_ID"] = d["GAME_ID"].astype(str).str.zfill(10)
    d["TEAM_ID"] = pd.to_numeric(d["TEAM_ID"], errors="coerce")
    return d

def _fill_oreb_from_gl(pair_df: pd.DataFrame, gl_pair: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    req = {"TEAM_ID","OREB","DREB"}
    if not req.issubset(gl_pair.columns) or len(gl_pair) != 2 or len(out) != 2:
        return out
    g = gl_pair[["TEAM_ID","OREB","DREB"]].copy()
    g[["TEAM_ID","OREB","DREB"]] = g[["TEAM_ID","OREB","DREB"]].apply(pd.to_numeric, errors="coerce")
    g = g.dropna(subset=["TEAM_ID"]).set_index("TEAM_ID")
    for i, r in out.iterrows():
        tid = int(r["TEAM_ID"])
        if tid not in g.index: 
            continue
        opp = [x for x in g.index if x != tid][0]
        t_orb, o_drb = g.at[tid,"OREB"], g.at[opp,"DREB"]
        den = (t_orb + o_drb)
        if pd.isna(out.at[i,"OREB_PCT"]):
            out.at[i,"OREB_PCT"] = (float(t_orb)/float(den)) if den else float("nan")
    return _normalize_ff_cols(out, ("OREB_PCT",))

def _season_from_name(fn: str) -> str|None:
    m = re.search(r'(\d{4}-\d{2})', fn)
    return m.group(1) if m else None

def main():
    paths = sorted(glob(os.path.join(LEDGER_DIR, "*.parquet")))
    if not paths:
        print(f"No parquets in {LEDGER_DIR}"); return
    cache = {}
    for p in paths:
        fn = os.path.basename(p)
        season = _season_from_name(fn)
        print(f"\n=== {season or '???'} — repairing {fn} ===")
        df = pd.read_parquet(p)
        if df.empty:
            print("  empty; skip"); continue
        # normalize any 0–100
        df = _normalize_ff_cols(df, ("OREB_PCT",))
        # backfill NaNs via league game log
        if "OREB_PCT" in df.columns and df["OREB_PCT"].isna().any() and season:
            if season not in cache:
                cache[season] = get_league_game_log(season)
            gl = cache[season]
            for gid in df.loc[df["OREB_PCT"].isna(), "GAME_ID"].astype(str).unique():
                idx = df.index[df["GAME_ID"].astype(str) == gid]
                if len(idx) != 2: 
                    continue
                pair = df.loc[idx]
                gl_pair = gl[gl["GAME_ID"] == gid]
                if len(gl_pair) != 2: 
                    continue
                fixed = _fill_oreb_from_gl(pair, gl_pair)
                df.loc[idx, ["OREB_PCT"]] = fixed[["OREB_PCT"]].values
        # final clamp
        df = _normalize_ff_cols(df, ("OREB_PCT",))
        df.to_parquet(p, index=False)
        print(f"  wrote {p}")

if __name__ == "__main__":
    main()

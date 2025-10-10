#!/usr/bin/env python3
import os, sys, re
import pandas as pd

# === Configure your ledgers directory ===
LEDGER_DIR = "/Users/harsha/Desktop/PickPocketProjectOfficial/ledgers/advanced_ledger"

# --- helpers (mirror the module logic) ---
def _normalize_pct_cols(df, cols=("OREB_PCT","DREB_PCT","REB_PCT")):
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

def _fill_reb_from_gl(team_rows: pd.DataFrame, gl_pair: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    out = team_rows.copy()
    req = {"TEAM_ID","OREB","DREB","REB"}
    if not req.issubset(set(gl_pair.columns)) or len(gl_pair) != 2:
        return out

    g = gl_pair[["TEAM_ID","OREB","DREB","REB"]].copy()
    g[["TEAM_ID","OREB","DREB","REB"]] = g[["TEAM_ID","OREB","DREB","REB"]].apply(pd.to_numeric, errors="coerce")
    g = g.dropna(subset=["TEAM_ID"]).set_index("TEAM_ID")

    for i, r in out.iterrows():
        tid = int(r["TEAM_ID"])
        if tid not in g.index: 
            continue
        opp = [x for x in g.index if x != tid]
        if not opp: 
            continue
        opp = opp[0]
        toreb, tdreb, treb = g.at[tid,"OREB"], g.at[tid,"DREB"], g.at[tid,"REB"]
        ooreb, odreb, oreb = g.at[opp,"OREB"], g.at[opp,"DREB"], g.at[opp,"REB"]
        if "OREB_PCT" in out.columns and pd.isna(out.at[i,"OREB_PCT"]):
            out.at[i,"OREB_PCT"] = (toreb / (toreb + odreb)) if (toreb + odreb) else np.nan
        if "DREB_PCT" in out.columns and pd.isna(out.at[i,"DREB_PCT"]):
            out.at[i,"DREB_PCT"] = (tdreb / (tdreb + ooreb)) if (tdreb + ooreb) else np.nan
        if "REB_PCT" in out.columns and pd.isna(out.at[i,"REB_PCT"]):
            out.at[i,"REB_PCT"] = (treb / (treb + oreb)) if (treb + oreb) else np.nan

    return _normalize_pct_cols(out)

# --- load league log once per season (fewer API calls) ---
def get_league_game_log(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog
    df = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df

def infer_season_from_filename(fn: str) -> str | None:
    # supports "2013-14.parquet" and "advanced_2013-14.parquet"
    m = re.search(r'(\d{4}-\d{2})', fn)
    return m.group(1) if m else None

def main():
    from glob import glob
    paths = sorted(glob(os.path.join(LEDGER_DIR, "*.parquet")))
    if not paths:
        print(f"No parquet files in {LEDGER_DIR}")
        return

    for p in paths:
        season = infer_season_from_filename(os.path.basename(p))
        if not season:
            print(f"Skip {p} (cannot infer season)")
            continue

        print(f"\n=== {season} — repairing {os.path.basename(p)} ===")
        df = pd.read_parquet(p)
        if df.empty:
            print("  empty; skipping")
            continue

        # normalize existing %
        df = _normalize_pct_cols(df)

        # fill NaNs from game log, game by game (only where needed)
        need = df[["GAME_ID","TEAM_ID"]].copy()
        for col in ["OREB_PCT","DREB_PCT","REB_PCT"]:
            if col in df.columns:
                need[col] = pd.to_numeric(df[col], errors="coerce")
        mask = (
            (need.get("OREB_PCT").isna() if "OREB_PCT" in need else False) |
            (need.get("DREB_PCT").isna() if "DREB_PCT" in need else False) |
            (need.get("REB_PCT").isna() if "REB_PCT" in need else False)
        )

        if mask.any():
            gl = get_league_game_log(season)
            for gid in df.loc[mask, "GAME_ID"].astype(str).unique():
                pair = df[df["GAME_ID"].astype(str) == gid].copy()
                gl_pair = gl[gl["GAME_ID"] == str(gid)]
                if len(pair) == 2 and len(gl_pair) == 2:
                    repaired = _fill_reb_from_gl(pair, gl_pair)
                    df.loc[pair.index, ["OREB_PCT","DREB_PCT","REB_PCT"]] = repaired[["OREB_PCT","DREB_PCT","REB_PCT"]].values

        # final normalize & clamp again
        df = _normalize_pct_cols(df)

        # write back
        df.to_parquet(p, index=False)
        print(f"  wrote {p}")

if __name__ == "__main__":
    main()

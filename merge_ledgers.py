# merge_ledgers.py
import os, sys
from pathlib import Path
import pandas as pd

TMP_ROOT = Path(os.environ.get("NBA_LEDGER_TMP_ROOT", "ledgers_tmp"))
OUT_ROOT = Path(os.environ.get("NBA_LEDGER_OUT_ROOT", "ledgers"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "advanced":     dict(fname="advanced_{season}.parquet",     pk=["TEAM_ID","GAME_ID"]),
    "fourfactors":  dict(fname="fourfactors_{season}.parquet",  pk=["TEAM_ID","GAME_ID"]),
    "hustle":       dict(fname="hustle_{season}.parquet",       pk=["TEAM_ID","GAME_ID"]),
    "misc":         dict(fname="misc_{season}.parquet",         pk=["TEAM_ID","GAME_ID"]),
}

def _read_if(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def merge_family_season(fam: str, season: str):
    spec = FAMILIES[fam]
    patt = spec["fname"].format(season=season)
    a = _read_if(TMP_ROOT / "process_A" / patt)
    b = _read_if(TMP_ROOT / "process_B" / patt)
    if a.empty and b.empty:
        return False

    df = pd.concat([a, b], ignore_index=True)
    # normalize GAME_ID, GAME_DATE if present
    if "GAME_ID" in df.columns:
        df["GAME_ID"] = df["GAME_ID"].astype(str)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # drop dupes by PK, keep last
    df.drop_duplicates(subset=spec["pk"], keep="last", inplace=True)

    # stable sort if columns exist
    order = [c for c in ["TEAM_ID","GAME_DATE","GAME_ID"] if c in df.columns]
    if order:
        df.sort_values(order, inplace=True)

    outp = OUT_ROOT / patt
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, outp)
    print(f"[MERGE] {fam} {season} → {outp} ({len(df)} rows)")
    return True

def seasons_to_merge():
    # discover seasons present in either process folder
    seen = set()
    for proc in ("process_A","process_B"):
        base = TMP_ROOT / proc
        if not base.exists(): 
            continue
        for p in base.glob("*.parquet"):
            # filenames like advanced_2020-21.parquet
            name = p.stem
            if "_" in name:
                _, season = name.split("_", 1)
                seen.add(season)
    return sorted(seen)

def main():
    seasons = sys.argv[1:] or seasons_to_merge()
    if not seasons:
        print("No temp ledgers found.")
        return
    for season in seasons:
        for fam in FAMILIES:
            try:
                merge_family_season(fam, season)
            except Exception as e:
                print(f"[MERGE][WARN] {fam} {season}: {e}")

if __name__ == "__main__":
    main()

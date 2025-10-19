# ledger_io.py
from __future__ import annotations
import os, re, glob
from pathlib import Path
import pandas as pd


DATA_ROOT = Path(os.environ.get("PICKPOCKET_DATA", Path.cwd() / "data"))
ROOT = DATA_ROOT

def _season_fname(season: str) -> str:
    """
    '2013-14' → '201314.parquet'
    also accepts '201314' or '201314.parquet'
    """
    s = str(season).strip().replace("–", "-").replace("—", "-")
    if "-" in s and len(s) >= 7:
        return f"{s[:4]}{s[-2:]}.parquet"
    t = re.sub(r"\D", "", s)            # keep digits only
    if len(t) == 6:
        return f"{t}.parquet"
    raise ValueError(f"Bad season label: {season}")

FOLDERS = {
    "advanced":      DATA_ROOT / "advanced_ledger",
    "four_factors":  DATA_ROOT / "four_factors_ledger",
    "misc":          DATA_ROOT / "misc_ledger",
    "hustle":        DATA_ROOT / "hustle_ledger",
}
for p in FOLDERS.values(): p.mkdir(parents=True, exist_ok=True)


# replace path_for with migration-aware version
def path_for(kind: str, season: str) -> Path:
    d = DATA_ROOT / f"{kind}_ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d / _season_fname(season)


def season_norm(s: str) -> str:
    import re
    m = re.match(r"^(\d{4})[-_](\d{2})$", s)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d{4})[-_](\d{4})$", s)
    if m: return f"{m.group(1)}-{m.group(2)[2:]}"
    return s

def save_df(kind: str, season: str, df: pd.DataFrame) -> Path:
    out = path_for(kind, season)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out

def load_df(kind: str, season: str) -> pd.DataFrame | None:
    p = path_for(kind, season)
    if not p.exists(): return None
    df = pd.read_parquet(p)
    for c in ("GAME_DATE","game_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    return df

def list_files(kind: str) -> list[str]:
    return sorted(glob.glob(str(FOLDERS[kind] / "*.parquet")))

def debug_where(kind: str):
    print(f"[ledger_io] {kind} dir = {FOLDERS[kind]}")
    print(f"[ledger_io] present = {list_files(kind)}")

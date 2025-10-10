# ledger_io.py
from __future__ import annotations
import os, re, glob
from pathlib import Path
import pandas as pd

def _project_root() -> Path:
    return Path(os.environ.get(
        "PICKPOCKET_DATA",
        Path(__file__).resolve().parents[1] / "data"   # <repo>/data by default
    )).resolve()

ROOT = _project_root()
FOLDERS = {
    "advanced":      ROOT / "advanced_ledger",
    "four_factors":  ROOT / "four_factors_ledger",
    "misc":          ROOT / "misc_ledger",
    "hustle":        ROOT / "hustle_ledger",
}
for p in FOLDERS.values(): p.mkdir(parents=True, exist_ok=True)

def season_norm(s: str) -> str:
    import re
    m = re.match(r"^(\d{4})[-_](\d{2})$", s)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d{4})[-_](\d{4})$", s)
    if m: return f"{m.group(1)}-{m.group(2)[2:]}"
    return s

def path_for(kind: str, season: str) -> Path:
    kind = kind.lower()
    return FOLDERS[kind] / f"{season_norm(season)}.parquet"

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

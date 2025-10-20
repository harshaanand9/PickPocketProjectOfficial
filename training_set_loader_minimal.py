# training_set_loader_minimal.py
from __future__ import annotations

# --- Core stdlib ---
import os
import math
import importlib
from pathlib import Path
from typing import List, Dict

# --- Third party ---
import pandas as pd
import numpy as np

# --- Project modules (do not remove) ---
import stats_getter as sg
from training_set_loader_copy import calculate_game_features as calc_feats
import advanced_ledger
import FourFactors_ledger
import misc_ledger
import hustle_ledger
from elo_team_ledger import _elo_home, _elo_away  # imported so features stay available

# --- Proxy coordinator (critical) ---
from proxy_coord import (
    acquire as proxy_acquire,
    release as proxy_release,
    apply_proxy as proxy_apply,
    current_proxy as proxy_current,
)
import training_set_loader_copy

# ======================================================================================
# Paths and outputs
# ======================================================================================

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = (REPO_ROOT / "out_features").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _season_csv_path(season: str) -> Path:
    # out_features/features_2013-14.csv, etc.
    return OUT_DIR / f"features_{season}.csv"

# ======================================================================================
# Proxy pacing profile (kept intact)
# ======================================================================================

PROFILE = os.getenv("NBA_PACING_PROFILE", "A").strip().upper()

_profiles = {
    "A": {
        "NBA_STEADY_SLEEP": 0.15,
        "NBA_JITTER": 0.08,
        "NBA_GLOBAL_COOLDOWN": 1.5,
        "endpoint_sleeps": {
            "LeagueGameLog":               0.35,
            "LeagueDashTeamStats":         0.25,
            "LeagueDashPlayerStats":       0.30,
            "TeamDashPtShots":             0.48,
            "LeagueDashTeamShotLocations": 0.48,
            "LeagueHustleStatsTeam":       0.55,
        },
    },
    "B": {
        "NBA_STEADY_SLEEP": 0.12,
        "NBA_JITTER": 0.10,
        "NBA_GLOBAL_COOLDOWN": 1.2,
        "endpoint_sleeps": {
            "LeagueGameLog":               0.32,
            "LeagueDashTeamStats":         0.22,
            "LeagueDashPlayerStats":       0.24,
            "TeamDashPtShots":             0.40,
            "LeagueDashTeamShotLocations": 0.40,
            "LeagueHustleStatsTeam":       0.46,
        },
    },
}
_cfg = _profiles.get(PROFILE, _profiles["A"])

# Apply env knobs so stats_getter picks them up
os.environ.update({
    "NBA_STEADY_SLEEP":    f'{_cfg["NBA_STEADY_SLEEP"]}',
    "NBA_JITTER":          f'{_cfg["NBA_JITTER"]}',
    "NBA_GLOBAL_COOLDOWN": f'{_cfg["NBA_GLOBAL_COOLDOWN"]}',
})

# Reload modules that snapshot env at import
importlib.reload(sg)


# Per-endpoint overrides
from stats_getter import STEADY_SLEEP_BY_ENDPOINT, PACE_GATES, PaceGate
STEADY_SLEEP_BY_ENDPOINT.update(_cfg["endpoint_sleeps"])
for ep, s in dict(STEADY_SLEEP_BY_ENDPOINT).items():
    PACE_GATES[ep] = PaceGate(s)

print(
    f"[PACE] profile={PROFILE} "
    f"default={_cfg['NBA_STEADY_SLEEP']}s jitter={_cfg['NBA_JITTER']}s "
    f"cooldown={_cfg['NBA_GLOBAL_COOLDOWN']}s "
    f"overrides={_cfg['endpoint_sleeps']}"
)

# ======================================================================================
# Proxy management (critical; unchanged behavior)
# ======================================================================================

WORKER = os.environ.get("NBA_WORKER", "").strip().upper() or "X"

def _ensure_proxy(worker: str, *, exclude: str | None = None) -> None:
    cur = proxy_current()
    if cur and (exclude is None or cur != exclude):
        got = proxy_acquire(worker, exclude=None)  # idempotent reservation
        if got and got != cur:
            proxy_apply(got)
            print(f"[proxy] ({worker}) corrected reservation → {got}")
        return
    nxt = proxy_acquire(worker, exclude=exclude)
    if nxt:
        proxy_apply(nxt)
        print(f"[proxy] ({worker}) initial → {nxt}")

def _maybe_rotate_after_quota(worker: str, games_on_this_proxy: int) -> tuple[bool, int]:
    """
    Rotate after 3 games on a proxy, 12 on DIRECT.
    Returns (rotated, new_counter).
    """
    cur = proxy_current()
    quota = 12 if cur == "DIRECT" else 3
    if games_on_this_proxy >= quota:
        proxy_release(worker, cur)
        nxt = proxy_acquire(worker, exclude=cur)
        if nxt:
            proxy_apply(nxt)
            print(f"[proxy] ({worker}) post-quota rotate → {nxt} (quota={quota})")
            return True, 0
        print(f"[proxy] ({worker}) rotate wanted but none free; staying on {cur}")
        return False, games_on_this_proxy
    return False, games_on_this_proxy

# ======================================================================================
# Helpers to split seasons across workers and to resolve game IDs
# ======================================================================================

def _seasons_from_env(seasons: List[str]) -> List[str]:
    """
    Split seasons across workers in contiguous chunks if NBA_SPLIT_SEASONS=1 (default).
    """
    if os.getenv("NBA_SPLIT_SEASONS", "1") == "0":
        return seasons

    worker = os.getenv("NBA_WORKER", "").strip().upper()
    worker_count = int(os.getenv("NBA_WORKER_COUNT", "1"))

    if len(worker) == 1 and "A" <= worker <= "Z":
        idx = ord(worker) - ord("A")
    else:
        try:
            idx = max(0, int(worker) - 1)
        except Exception:
            idx = 0
    idx = idx % max(worker_count, 1)

    n = len(seasons)
    if worker_count <= 1 or n == 0:
        return seasons

    chunk = math.ceil(n / worker_count)
    start = idx * chunk
    end = min(start + chunk, n)
    chosen = seasons[start:end]
    return chosen or []

def _resolve_game_id_via_log(season: str, home_name: str, away_name: str, date_str: str) -> str | None:
    """
    Resolve GAME_ID for the exact (home vs away) on date, using a cached season log.
    """
    lg = sg.get_league_game_log(season).copy()
    lg["_DATE"] = pd.to_datetime(lg["GAME_DATE"], errors="coerce").dt.normalize()
    d = pd.to_datetime(date_str, errors="coerce").normalize()
    hid, aid = sg.get_team_id(home_name), sg.get_team_id(away_name)
    rows = lg[(lg["_DATE"] == d) & (lg["TEAM_ID"].isin([hid, aid]))]
    gids = rows["GAME_ID"].dropna().unique().tolist()
    if len(gids) == 1:
        return str(gids[0])
    for gid in gids:
        sub = lg[lg["GAME_ID"] == gid]
        if set(sub["TEAM_ID"].unique()).issuperset({hid, aid}):
            return str(gid)
    return None

# ======================================================================================
# Idempotent ledger appends with prints
# ======================================================================================

def _is_empty(obj) -> bool:
    try:
        if hasattr(obj, "empty"):
            return bool(obj.empty)
        return len(obj) == 0
    except Exception:
        return obj is None

def _append_game_idempotent(mod_name: str, get_rows_fn: str, append_fn: str, season: str, gid: str):
    mod = __import__(mod_name, fromlist=["*"])
    get_rows = getattr(mod, get_rows_fn)
    do_append = getattr(mod, append_fn)
    existing = get_rows(season, gid)
    if existing is None or _is_empty(existing):
        do_append(season, gid)
        print(f"[{mod_name}] appended GAME_ID={gid}")
    else:
        print(f"[{mod_name}] already has GAME_ID={gid}, skipping")

def append_adv_game_idempotent(season: str, gid: str):
    _append_game_idempotent("advanced_ledger", "get_ledger_rows_for_game", "append_adv_game", season, gid)

def append_misc_game_idempotent(season: str, gid: str):
    _append_game_idempotent("misc_ledger", "get_ledger_rows_for_game", "append_misc_game", season, gid)

def append_fourfactors_game_idempotent(season: str, gid: str):
    _append_game_idempotent("FourFactors_ledger", "get_ledger_rows_for_game", "append_FourFactors_game", season, gid)

def append_hustle_game_idempotent(season: str, gid: str, date_str: str):
    # Hustle available >= 2017-18; skip earlier
    start_year = int(str(sg.season_for_date_smart(date_str))[:4])
    if start_year >= 2017:
        _append_game_idempotent("hustle_ledger", "get_ledger_rows_for_game", "append_hustle_game", season, gid)
    else:
        print(f"[hustle_ledger] skipping (≤ 2016-17) GAME_ID={gid}")

# ======================================================================================
# CSV append helper
# ======================================================================================

def append_feature_row_csv(season: str, row: Dict) -> None:
    csv_path = _season_csv_path(season)
    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)

# ======================================================================================
# Season game source: training_set (preferred) or league log
# ======================================================================================

def get_season_games(season: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: date, home_team, away_team, season.
    Prefers training_set_complete.csv in repo root. Falls back to league log.
    """
    # Preferred CSV if present
    tsc = REPO_ROOT / "training_set_complete.csv"
    if tsc.exists():
        df = pd.read_csv(tsc)
        df = df[df["season"] == season].copy()
        if not df.empty:
            return df[["date", "home_team", "away_team", "season"]].reset_index(drop=True)

    # Fallback via league log
    lg = sg.get_league_game_log(season).copy()
    lg["GAME_DATE"] = pd.to_datetime(lg["GAME_DATE"]).dt.normalize()
    games = []
    for gid, pair in lg.groupby("GAME_ID", sort=False):
        if len(pair) != 2:
            continue
        idx_home = pair["MATCHUP"].str.contains("vs.").idxmax()
        idx_away = (set(pair.index) - {idx_home}).pop()
        row_home, row_away = pair.loc[idx_home], pair.loc[idx_away]
        games.append({
            "date": row_home["GAME_DATE"].strftime("%m/%d/%Y"),
            "home_team": str(row_home["TEAM_NAME"]),
            "away_team": str(row_away["TEAM_NAME"]),
            "season": season,
        })
    return pd.DataFrame(games)

def _stable_sort_games(df_games: pd.DataFrame) -> pd.DataFrame:
    g = df_games.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    return g.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

# ======================================================================================
# Main APIs
# ======================================================================================

def process_season(season: str, mutate_ledgers: bool = True) -> pd.DataFrame:
    """
    Compute features for all games in a season.
    Writes incrementally to out_features/features_<season>.csv.
    Prints per-game progress and ledger status.
    """
    # Proxy boot
    _ensure_proxy(WORKER)
    games_on_this_proxy = 0

    # Warm resume state from season CSV if exists
    csv_path = _season_csv_path(season)
    df_games = _stable_sort_games(get_season_games(season))
    season_rows: list[dict] = []

    start_idx = 0
    if csv_path.exists():
        try:
            prev = pd.read_csv(csv_path)
            if not prev.empty:
                season_rows.extend(prev.to_dict("records"))
                last = prev.tail(1).iloc[0]
                last_date = pd.to_datetime(last["date"], errors="coerce").normalize()
                mask = pd.to_datetime(df_games["date"], errors="coerce").dt.normalize().eq(last_date)
                mask &= (df_games["home_team"] == str(last["home_team"])) & (df_games["away_team"] == str(last["away_team"]))
                hits = df_games[mask]
                if not hits.empty:
                    start_idx = int(hits.index[-1]) + 1
                print(f"[resume] warmed {len(prev)} rows from {csv_path}")
        except Exception:
            start_idx = 0

    # Cached season log for GAME_ID resolution
    league_log = sg.get_league_game_log(season).copy()
    league_log["_DATE"] = pd.to_datetime(league_log["GAME_DATE"], errors="coerce").dt.normalize()

    total = len(df_games)
    print("\n" + "="*54)
    print(f"Processing season: {season} | total games: {total}")
    print("="*54)

    for idx in range(start_idx, total):
        game = df_games.iloc[idx]
        date_str = game["date"] if isinstance(game["date"], str) else game["date"].strftime("%m/%d/%Y")
        home = sg.canon_team(str(game["home_team"]))
        away = sg.canon_team(str(game["away_team"]))

        # Rotate proxy if needed
        rotated, games_on_this_proxy = _maybe_rotate_after_quota(WORKER, games_on_this_proxy)
        if rotated:
            pass

        try:
            # Compute features using your existing implementation
            feats = training_set_loader_copy.calculate_game_features(
                home_team=home, away_team=away, date=date_str, season=season
            )
            games_on_this_proxy += 1

            # Write one row to season CSV
            row = {"date": date_str, "home_team": home, "away_team": away, "season": season}
            if isinstance(feats, dict):
                row.update(feats)
            append_feature_row_csv(season, row)

            # Append ledgers idempotently and print status
            gid = _resolve_game_id_via_log(season, home, away, date_str)
            if gid:
                append_adv_game_idempotent(season, gid)
                append_misc_game_idempotent(season, gid)
                append_fourfactors_game_idempotent(season, gid)
                append_hustle_game_idempotent(season, gid, date_str)
            else:
                print(f"[ledger] GAME_ID unresolved for {away} @ {home} on {date_str}")

            print(f"[{season}] {idx+1}/{total} {away} @ {home} on {date_str} ✓")

        except KeyboardInterrupt:
            print("\n[graceful-exit] stopping…")
            raise
        except sg.FeatureTimeoutError as e:
            print(f"[FATAL] Feature timeout at {away} @ {home} on {date_str}: {e}")
            raise SystemExit(2)
        except Exception as e:
            print(f"⚠️  Skipped {away} @ {home} on {date_str} ({season}): {e}")


    # Return the season DataFrame loaded from CSV to keep columns consistent
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

def load_all_features(seasons: List[str] | None = None) -> pd.DataFrame:
    """
    Entry point used by your worker commands.
    Splits seasons across workers and writes per-season CSVs.
    Returns concatenated DataFrame across this worker's seasons.
    """
    if seasons is None:
        seasons = [
            "2013-14","2014-15","2015-16","2016-17","2017-18","2018-19",
            "2019-20","2020-21","2021-22","2022-23","2023-24","2024-25",
        ]

    seasons = _seasons_from_env(seasons)
    all_parts = []
    for s in seasons:
        try:
            part = process_season(s, mutate_ledgers=True)
            all_parts.append(part)
            print(f"✓ Completed {s} → {len(part) if not part.empty else 0} rows")
        except Exception as e:
            print(f"✗ Error processing {s}: {e}")
    if all_parts:
        cat = pd.concat(all_parts, ignore_index=True)
        print(f"\n✓ Done. Wrote per-season CSVs under {OUT_DIR}")
        print(f"Total rows (this worker): {len(cat)}")
        return cat
    print("\nNo rows written.")
    return pd.DataFrame()

# Optional: allow quick local run
if __name__ == "__main__":
    load_all_features()

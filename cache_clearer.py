# put this in a small utils file (e.g., maintenance.py) or at the bottom of advanced_ledger.py

from pathlib import Path

def clear_all_ledgers(parquet_root: str = ".cache", verbose: bool = True) -> int:
    """
    Delete all ledger/parquet files so priors/ledgers rebuild from scratch.
    - Targets *.parquet under `parquet_root` (default: .cache).
    - Also tries to clear in-memory LRU slices if advanced_ledger is imported.
    Returns: number of files deleted.
    """
    root = Path(parquet_root)
    if not root.exists():
        if verbose:
            print(f"[clear_all_ledgers] No such directory: {root!s}")
        return 0

    # You can narrow/extend these patterns if you keep other ledgers next to advanced_*.parquet
    patterns = ["advanced_*.parquet", "hustle_*.parquet", "fourfactors_*.parquet", "*.parquet"]

    seen = set()
    files = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p not in seen:
                files.append(p)
                seen.add(p)

    deleted = 0
    for p in files:
        try:
            p.unlink()
            deleted += 1
            if verbose:
                print(f"[clear_all_ledgers] deleted {p}")
        except Exception as e:
            if verbose:
                print(f"[clear_all_ledgers] could not delete {p}: {e}")

    # Try to drop any in-memory LRU slices so new reads don’t surface stale frames
    try:
        import advanced_ledger as adv
        for fn_name in ("_team_adv_slice",):
            fn = getattr(adv, fn_name, None)
            if fn and hasattr(fn, "cache_clear"):
                fn.cache_clear()
                if verbose:
                    print(f"[clear_all_ledgers] cleared LRU for advanced_ledger.{fn_name}")
    except Exception:
        pass

    # Optionally prune empty folders under .cache
    for d in sorted(root.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except Exception:
                pass

    if verbose:
        print(f"[clear_all_ledgers] total files deleted: {deleted}")
    return deleted


def clear_everything(also_endpoint_cache: bool = True, verbose: bool = True) -> None:
    """
    Convenience wrapper:
      1) Clears all ledgers/parquets (via clear_all_ledgers).
      2) Optionally clears endpoint caches (RAM + disk) using StatsCache.
    """
    clear_all_ledgers(verbose=verbose)

    if also_endpoint_cache:
        try:
            from cache_manager import stats_cache
            stats_cache.clear_memory()     # RAM LRU
            stats_cache.clear_disk()       # disk files under 'cache/' by endpoint subfolders
            if verbose:
                print("[clear_everything] cleared stats_cache RAM + disk")
        except Exception as e:
            if verbose:
                print(f"[clear_everything] could not clear endpoint cache: {e}")

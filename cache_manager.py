from collections import OrderedDict
import time, json, hashlib, os, pickle, gzip, builtins
from pathlib import Path
import threading

import pandas as pd

class StatsCache:
    """
    RAM + optional disk-backed cache for endpoint results.

    - RAM: OrderedDict LRU with optional TTL
    - Disk: write-through .pkl.gz files keyed by endpoint + hashed params
    """

    def __init__(self, *, maxsize=5000, ttl=None, verbose=False,
                 enable_disk=True, disk_dir="cache"):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl  # seconds or None
        self.verbose = verbose

        # disk settings
        self.enable_disk = enable_disk
        # Allow shared or per-worker disk cache
        _root = os.environ.get("NBA_CACHE_ROOT", disk_dir)            # e.g. "cache"
        _worker = os.environ.get("NBA_WORKER", "").strip().upper()    # "A" | "B" | "C" | ...
        _per_worker = os.environ.get("NBA_CACHE_PER_WORKER", "1")     # "1" (default) = per-worker, "0" = shared
        if _worker and _per_worker != "0":
            _root = f"{_root}/process_{_worker}"                      # cache/process_A, etc.
        self.disk_dir = Path(_root)
        if self.enable_disk:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
        # simple lock so RAM ops are safe when you thread the loader
        self._lock = threading.Lock()

    def get_cache_key(self, endpoint_name: str, **kwargs) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        h = hashlib.md5(payload.encode()).hexdigest()
        return f"{endpoint_name}:{h}"

    # ---------- time/key helpers ----------
    def _now(self):
        return time.time()

    def _key_parts(self, endpoint_name, kwargs):
        payload = json.dumps(kwargs, sort_keys=True, default=str).encode()
        h = hashlib.md5(payload).hexdigest()
        return h, f"{endpoint_name}:{h}"

    def _disk_path(self, endpoint_name, kwargs):
        h, _ = self._key_parts(endpoint_name, kwargs)
        # one subfolder per endpoint â†’ keeps directories tidy
        return self.disk_dir / endpoint_name / f"{h}.pkl.gz"

    # ---------- disk I/O ----------
    def _disk_read(self, endpoint_name, kwargs):
        if not self.enable_disk:
            return None
        p = self._disk_path(endpoint_name, kwargs)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rb") as f:
                value, exp = pickle.load(f)  # tuple
            # respect TTL if you configured one
            if self.ttl and exp and self._now() > exp:
                try: p.unlink()
                except Exception: pass
                return None
            return value, exp
        except Exception:
            # corrupted/old file â†’ treat as miss
            try: p.unlink()
            except Exception: pass
            return None

    def _disk_write(self, endpoint_name, kwargs, value, exp):
        if not self.enable_disk:
            return
        p = self._disk_path(endpoint_name, kwargs)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            with gzip.open(tmp, "wb") as f:
                pickle.dump((value, exp), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)  # atomic
        except Exception:
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass

    # --------- emptiness helpers ---------
    def _is_empty_df_like(self, value):
        """Return True if value is an empty pandas DataFrame (or a sequence of only-empty DFs)."""
        try:
            import pandas as pd  # local import to avoid hard dep at import time
        except Exception:
            pd = None

        # Single DF
        if pd is not None and isinstance(value, pd.DataFrame):
            return value.empty

        # List/tuple of DFs (nba_api sometimes returns lists)
        if isinstance(value, (list, tuple)) and value:
            if pd is not None and all(isinstance(v, pd.DataFrame) for v in value):
                return all(v.empty for v in value)

        # Treat None as empty (donâ€™t cache)
        if value is None:
            return True

        return False

    def _should_skip_cache(self, value, custom_skip_if):
        """Decide whether to skip caching this value."""
        if callable(custom_skip_if):
            try:
                return bool(custom_skip_if(value))
            except Exception:
                # if user-supplied predicate blows up, fall back to default
                pass
        return self._is_empty_df_like(value)

    # ---------- public API ----------
    # cache_manager.py
    import pandas as pd
    import time

    def get_or_fetch(self, endpoint_name: str, fetch_func, *, skip_if=None, **kwargs):
        key = self.get_cache_key(endpoint_name, **kwargs)

        # ---- RAM hit ----
        if key in self.cache:
            val = self.cache[key]
            return val

        # ---- (optional) DISK hit ----
        if hasattr(self, "_disk_read"):
            disk = self._disk_read(endpoint_name, kwargs)
            if disk is not None:
                val, _ = disk
                self.cache[key] = val
                return val

        # ---- fetch ----
        t0 = time.perf_counter()
        value = fetch_func(**kwargs)
        _dt = time.perf_counter() - t0
        # print(f"[StatsCache] FETCH {endpoint_name} took {_dt:.2f}s")

        # ---- do-not-cache rules ----
        try:
            # never cache empty DataFrames
            if isinstance(value, pd.DataFrame) and value.empty:
                # print(f"[StatsCache] not caching EMPTY {endpoint_name}")
                return value
            if skip_if is not None and skip_if(value):
                # print(f"[StatsCache] skip_if true â€” not caching {endpoint_name}")
                return value
        except Exception as e:
            #print(f"[StatsCache] skip_if raised for {endpoint_name}: {e!r} â€” returning without cache")
            return value

        # ---- store ----
        exp = (self._now() + self.ttl) if self.ttl else None
        self.cache[key] = value
        if hasattr(self, "_disk_write"):
            try:
                self._disk_write(endpoint_name, kwargs, value, exp)
            except Exception:
                pass
        return value


    def get_cache_key(self, endpoint_name: str, **kwargs) -> str:
        epoch = os.environ.get("NBA_CACHE_EPOCH", "0")  # NEW
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        h = hashlib.md5(f"{epoch}|{payload}".encode()).hexdigest()  # CHANGED
        return f"{endpoint_name}:{h}"
    
    def _key_parts(self, endpoint_name, kwargs):
        epoch = os.environ.get("NBA_CACHE_EPOCH", "0")  # NEW
        payload = json.dumps(kwargs, sort_keys=True, default=str).encode()
        h = hashlib.md5(f"{epoch}|".encode() + payload).hexdigest()  # CHANGED
        return h, f"{endpoint_name}:{h}"


    # maintenance helpers
    def clear_memory(self):
        with self._lock:
            self.cache.clear()

    def clear_disk(self, endpoint: str | None = None, *, all_workers: bool = False):
        """
        Delete all disk files (or only for one endpoint).
        If all_workers=True, remove the entire NBA_CACHE_ROOT tree.
        """
        if not self.enable_disk:
            return
        base = self.disk_dir
        if all_workers:
            # If disk_dir is ".../cache/process_A", we want its parent (".../cache")
            base = base if base.name != f"process_{os.environ.get('NBA_WORKER','').upper()}" else base.parent
        if endpoint is not None:
            base = base / endpoint
        if not base.exists():
            return
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        for p in sorted(base.rglob("*"), reverse=True):
            try:
                if p.is_dir():
                    p.rmdir()
            except Exception:
                pass

    def clear_all(self, *, all_workers: bool = False):
        """Clear RAM and disk."""
        self.clear_memory()
        self.clear_disk(all_workers=all_workers)


    def purge_empty_dfs(self, endpoint: str | None = None):
        """
        Scan disk cache and delete files whose payload is an empty DataFrame (or list of only-empty DFs).
        """
        if not self.enable_disk:
            return
        base = self.disk_dir if endpoint is None else (self.disk_dir / endpoint)
        if not base.exists():
            return

        import pickle, gzip
        try:
            import pandas as pd  # just to detect DataFrame type
        except Exception:
            pd = None

        removed = 0
        for p in base.rglob("*.pkl.gz"):
            try:
                with gzip.open(p, "rb") as f:
                    value, _exp = pickle.load(f)
                if self._is_empty_df_like(value):
                    p.unlink()
                    removed += 1
            except Exception:
                # corrupted â†’ remove
                try: p.unlink()
                except Exception: pass
                removed += 1


# ---- GLOBAL SINGLETON (survives reloads / notebook re-runs) ----
if not hasattr(builtins, "_STATS_CACHE_SINGLETON"):
    builtins._STATS_CACHE_SINGLETON = StatsCache(
        maxsize=5000,
        ttl=None,               # keep forever unless you set a TTL (seconds)
        verbose=False,
        enable_disk=True,       # <â€” turn on disk
        disk_dir=os.environ.get("NBA_CACHE_ROOT", "cache")
    )

# what other modules import

stats_cache: StatsCache = builtins._STATS_CACHE_SINGLETON


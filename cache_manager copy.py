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
        self.disk_dir = Path(disk_dir)
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
        # one subfolder per endpoint → keeps directories tidy
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
            # corrupted/old file → treat as miss
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

        # Treat None as empty (don’t cache)
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
    def get_or_fetch(self, endpoint_name: str, fetch_func, *, skip_if=None, **kwargs):
        """
        Returns cached value if present; otherwise calls fetch_func(**kwargs).
        - If skip_if(value) is True, the result is returned but NOT cached.
        - Empty pandas DataFrames are never cached.
        """
        key = self.get_cache_key(endpoint_name, **kwargs)
        if key in self.cache:
            # print(f"Cache hit: {endpoint_name}")
            return self.cache[key]

        # print(f"Cache miss: {endpoint_name} -> fetching…")
        value = fetch_func(**kwargs)

        # Do not cache empties / bad results
        try:
            if isinstance(value, pd.DataFrame) and value.empty:
                # print(f"Cache skip (empty DataFrame): {endpoint_name}")
                return value
            if skip_if is not None and skip_if(value):
                # print(f"Cache skip (predicate): {endpoint_name}")
                return value
        except Exception:
            # If the skip check itself fails, just return without caching
            return value

        self.cache[key] = value
        return value

    # maintenance helpers
    def clear_memory(self):
        with self._lock:
            self.cache.clear()

    def clear_disk(self, endpoint: str | None = None):
        """
        Delete all disk files (or only for one endpoint).
        """
        if not self.enable_disk:
            return
        base = self.disk_dir if endpoint is None else self.disk_dir / endpoint
        if not base.exists():
            return
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        # optional: clean empty folders
        for p in sorted(base.rglob("*"), reverse=True):
            try:
                if p.is_dir():
                    p.rmdir()
            except Exception:
                pass

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
                # corrupted → remove
                try: p.unlink()
                except Exception: pass
                removed += 1
        if self.verbose:
            print(f"[StatsCache] purged {removed} empty cache files under {base}")


# ---- GLOBAL SINGLETON (survives reloads / notebook re-runs) ----
if not hasattr(builtins, "_STATS_CACHE_SINGLETON"):
    builtins._STATS_CACHE_SINGLETON = StatsCache(
        maxsize=5000,
        ttl=None,               # keep forever unless you set a TTL (seconds)
        verbose=False,
        enable_disk=True,       # <— turn on disk
        disk_dir="cache"        # <— cache folder at project root (gitignore it)
    )

# what other modules import

stats_cache: StatsCache = builtins._STATS_CACHE_SINGLETON



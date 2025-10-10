# proxy_coord.py
from __future__ import annotations
import os, json, time, random, tempfile
from contextlib import contextmanager

STATE_PATH = os.environ.get("NBA_PROXY_STATE", os.path.join(tempfile.gettempdir(), "nba_proxy_state.json"))
POOL_ENV   = "NBA_PROXY_POOL"   # CSV: "DIRECT,http://user:pass@host:port,..."

# --------- tiny file lock (cross-process) ----------
@contextmanager
def _locked(path: str, retry_delay=0.05, timeout=5.0):
    lock = path + ".lock"
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError("proxy_coord: lock timeout")
            time.sleep(retry_delay)
    try:
        yield
    finally:
        try: os.remove(lock)
        except FileNotFoundError: pass

def _load_state():
    if not os.path.exists(STATE_PATH):
        return {
            "cycle": [],         # randomized cycle order
            "idx": 0,            # next index within cycle
            "in_use": {},        # proxy -> worker label ("A"/"B")
            "seen_cycle": set(), # used to avoid repeats until exhaustion
        }
    with open(STATE_PATH, "r") as f:
        obj = json.load(f)
    # json can't store sets; coerce if present
    if isinstance(obj.get("seen_cycle"), list):
        obj["seen_cycle"] = set(obj["seen_cycle"])
    return obj

def _save_state(st):
    st = dict(st)
    st["seen_cycle"] = list(st.get("seen_cycle", set()))
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_PATH)

def _parse_pool():
    raw = (os.environ.get(POOL_ENV, "") or "").strip()
    items = [p.strip() for p in raw.split(",") if p.strip()]
    # normalize "DIRECT"
    out = []
    seen = set()
    for p in items:
        p = "DIRECT" if p.upper() == "DIRECT" else p
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _reshuffle_cycle(pool):
    cyc = pool[:]
    random.shuffle(cyc)
    return cyc

def acquire(worker: str, *, exclude: str | None = None) -> str | None:
    """Return a proxy not in use by another worker. Non-repeating until cycle exhausted."""
    pool = _parse_pool()
    if not pool:
        return None

    with _locked(STATE_PATH):
        st = _load_state()
        # ensure cycle exists & valid vs pool
        if not st.get("cycle") or set(st["cycle"]) - set(pool):
            st["cycle"] = _reshuffle_cycle(pool)
            st["idx"] = 0
            st["seen_cycle"] = set()

        # try to find a free proxy that is not exclude and not in_use
        tried = 0
        total = len(pool)
        while tried < total:
            if st["idx"] >= len(st["cycle"]):
                st["cycle"] = _reshuffle_cycle(pool)
                st["idx"] = 0
                st["seen_cycle"] = set()

            cand = st["cycle"][st["idx"]]
            st["idx"] += 1
            tried += 1

            if exclude and cand == exclude:
                continue
            if cand in st["in_use"] and st["in_use"][cand] != worker:
                continue  # other worker currently using it

            # OK — reserve it
            st["in_use"][cand] = worker
            st["seen_cycle"].add(cand)
            _save_state(st)
            return cand

        # nothing free right now
        return None

def release(worker: str, proxy: str | None):
    if not proxy:
        return
    with _locked(STATE_PATH):
        st = _load_state()
        # only release if we own it
        if st.get("in_use", {}).get(proxy) == worker:
            st["in_use"].pop(proxy, None)
            _save_state(st)

def current_proxy() -> str:
    return os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "DIRECT"

def apply_proxy(proxy: str | None):
    if not proxy or proxy == "DIRECT":
        for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
            os.environ.pop(k, None)
    else:
        os.environ["HTTP_PROXY"]  = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"]  = proxy
        os.environ["https_proxy"] = proxy

def rotate_on_failure(worker: str):
    prev = current_proxy()
    nxt = acquire(worker, exclude=prev)
    if nxt:
        apply_proxy(nxt)
        print(f"[proxy] rotated ({worker}) → {nxt}")
    else:
        print(f"[proxy] rotate skipped ({worker}) — no free proxy")

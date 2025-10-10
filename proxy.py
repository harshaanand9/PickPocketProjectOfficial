import os, itertools, time, random
import requests

PROXY_POOL = [
    {"host": "dc.decodo.com", "port": 10001, "user": "spbi6ee2j0", "pwd": "jD7Pk~5fMlcV4ty2bt"},
    {"host": "dc.decodo.com", "port": 10002, "user": "spbi6ee2j0", "pwd": "jD7Pk~5fMlcV4ty2bt"},
    {}
    # … add as many ports as you were issued
]

# rotation strategies
def per_run(iterator=False):
    # pick one proxy for the whole run
    p = random.choice(PROXY_POOL)
    while True:
        yield p

def per_request():
    # rotate each request
    while True:
        for p in PROXY_POOL:
            yield p

# choose one
proxy_iter = per_run()  # or per_request()

def build_proxies(p):
    auth = f"{p['user']}:{p['pwd']}"
    url = f"http://{auth}@{p['host']}:{p['port']}"
    return {"http": url, "https": url}

def get(url, **kwargs):
    p = next(proxy_iter)
    proxies = build_proxies(p)
    timeout = kwargs.pop("timeout", (10, 30))
    return requests.get(url, proxies=proxies, timeout=timeout, **kwargs)

def verify_egress():
    r = get("https://api.ipify.org?format=json")
    r.raise_for_status()
    print("Egress IP:", r.json())
    return r.json()

# --- example usage ---
if __name__ == "__main__":
    verify_egress()
    # your normal calls below, still respecting the API’s TOS and rate limits
    # resp = get("https://example.com/health")

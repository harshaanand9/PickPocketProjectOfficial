# === brightdata_test.py (paste-all) ===
import os, pathlib, sys

# 1) Path to Bright Data CA you downloaded
BD_CA = pathlib.Path.home() / "Downloads" / "brightdata_proxy_ca" / "SSL_certificate.crt"

# 2) Where we'll put the combined bundle
COMBINED = pathlib.Path.home() / ".ssl" / "cacert_plus_brightdata.pem"

# 3) Build combined bundle if needed (certifi + Bright Data CA)
def ensure_combined():
    if not BD_CA.exists():
        print(f"!! Bright Data CA not found at: {BD_CA}")
        print("   Fix path or re-download, then retry.")
        sys.exit(1)
    import certifi
    COMBINED.parent.mkdir(exist_ok=True)
    with open(certifi.where(), "rb") as f_base, open(BD_CA, "rb") as f_bd, open(COMBINED, "wb") as f_out:
        f_out.write(f_base.read())
        f_out.write(b"\n")
        f_out.write(f_bd.read())
    print("✅ Combined CA written:", COMBINED)

# Rebuild every run to be sure we’re not accidentally using only the BD cert
ensure_combined()

# 4) FORCE requests/urllib3/nba_api to use the combined bundle — set BEFORE imports
os.environ["REQUESTS_CA_BUNDLE"] = str(COMBINED)
os.environ["SSL_CERT_FILE"]      = str(COMBINED)
os.environ["CURL_CA_BUNDLE"]     = str(COMBINED)
os.environ["CERTIFI_CA_BUNDLE"]  = str(COMBINED)
print("Using CA bundle:", os.environ["REQUESTS_CA_BUNDLE"])

# 5) Now import nba_api and call through your Bright Data proxy
from nba_api.stats.endpoints import LeagueDashTeamStats

# Use the US exit + sticky session you already validated
PROXY = (
    "http://brd-customer-hl_290c33d6-zone-residential_proxy1-country-us-session-test123:"
    "n6qajh9v7q3s@brd.superproxy.io:33335"
)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

df = LeagueDashTeamStats(
    season="2024-25",
    season_type_all_star="Regular Season",
    proxy=PROXY,
    headers=HEADERS,
    timeout=25
).get_data_frames()[0]

print("rows:", len(df))
print(df.head())

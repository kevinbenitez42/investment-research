import os
import sys
from pathlib import Path

from schwab.auth import client_from_manual_flow
from schwab.client import Client

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Quantapp.secrets import load_project_env, require_secret

# ====== Step 1: Set your credentials ======
load_project_env()

CLIENT_ID = require_secret("SCHWAB_CLIENT_ID")
APP_SECRET = require_secret("SCHWAB_APP_SECRET")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
TOKEN_PATH = os.getenv("SCHWAB_TOKEN_PATH", "schwab_token.json")

# ====== Step 2: Start manual OAuth flow ======
# This will print a URL and prompt you to open it in your browser
client = client_from_manual_flow(
    api_key=CLIENT_ID,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=TOKEN_PATH
)

# ====== Step 3: Authenticate ======
# The script will automatically prompt you in the terminal:
# 1. Open the printed URL
# 2. Log in, approve the app
# 3. Paste the redirected URL back into the terminal
# The client will then save tokens to TOKEN_PATH

# ====== Step 4: Fetch your account numbers ======
acct_map = client.get_account_numbers().json()
acct_hash = acct_map["accounts"][0]["hashValue"]  # Use the first account

# ====== Step 5: Fetch all positions ======
positions_resp = client.get_account(acct_hash, fields=Client.Account.Fields.POSITIONS).json()
positions = positions_resp.get("securitiesAccount", {}).get("positions", [])

print("\nALL POSITIONS:")
for p in positions:
    inst = p.get("instrument", {})
    qty = p.get("longQuantity", 0) - p.get("shortQuantity", 0)
    print(f"{inst.get('symbol')} | {inst.get('assetType')} | Qty: {qty}")

# ====== Step 6: Filter only OPTIONS ======
options_positions = [
    p for p in positions if p.get("instrument", {}).get("assetType") == "OPTION"
]

print("\nOPTION POSITIONS:")
for p in options_positions:
    i = p["instrument"]
    qty = p.get("longQuantity", 0) - p.get("shortQuantity", 0)
    print({
        "symbol": i.get("symbol"),
        "putCall": i.get("putCall"),
        "strike": i.get("strikePrice"),
        "expiry": i.get("maturityDate"),
        "qty": qty
    })

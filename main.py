import os
import requests
import websocket
import json
import time

# Get secrets
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DERIV_API_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"

# --- Telegram sender ---
def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Failed to send Telegram:", e)

# --- Deriv connection ---
def get_tick():
    ws = websocket.WebSocket()
    ws.connect(DERIV_API_URL)
    auth = {"authorize": DERIV_API_TOKEN}
    ws.send(json.dumps(auth))
    ws.recv()  # auth response

    request = {"ticks": "R_100"}  # Change to EURUSD later if needed
    ws.send(json.dumps(request))

    response = ws.recv()
    ws.close()
    return response

if __name__ == "__main__":
    try:
        tick_data = get_tick()
        parsed = json.loads(tick_data)
        price = parsed.get("tick", {}).get("quote", "N/A")
        send_telegram_message(f"✅ Connected to Deriv. Last tick price: {price}")
    except Exception as e:
        send_telegram_message(f"❌ Error: {e}")

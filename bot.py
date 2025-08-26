import os
import json
import requests
import websocket

# Load secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"

def send_telegram_message(message: str):
    """Send a message to Telegram chat"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

def get_last_tick(symbol="frxEURUSD"):
    """Fetch latest tick price from Deriv WebSocket"""
    ws = websocket.create_connection(DERIV_WS_URL)

    # Authenticate
    ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
    ws.recv()

    # Request latest tick
    ws.send(json.dumps({"ticks_history": symbol, "count": 1, "end": "latest"}))
    response = json.loads(ws.recv())
    ws.close()

    try:
        return response["history"]["prices"][0]
    except Exception:
        return None

if __name__ == "__main__":
    price = get_last_tick("frxEURUSD")
    if price:
        send_telegram_message(f"✅ Deriv price update:\nEUR/USD = {price}")
    else:
        send_telegram_message("❌ Failed to fetch price from Deriv.")

import os
import requests
import websocket
import json
import time

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DERIV_APP_ID = os.getenv("DERIV_APP_ID")

# List of forex pairs you want to track
SYMBOLS = ["frxEURUSD", "frxAUDUSD", "frxUSDJPY"]

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    requests.post(url, json=payload)

def fetch_price(symbol):
    ws = websocket.create_connection(f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}")
    request = {"ticks": symbol}
    ws.send(json.dumps(request))
    data = json.loads(ws.recv())
    ws.close()

    if "tick" in data:
        return data["tick"]["quote"]
    else:
        return None

def main():
    messages = []
    for symbol in SYMBOLS:
        price = fetch_price(symbol)
        if price:
            messages.append(f"✅ {symbol} = {price}")
        else:
            messages.append(f"❌ Failed to fetch {symbol}")

    send_telegram_message("\n".join(messages))

if __name__ == "__main__":
    main()

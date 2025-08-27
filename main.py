import os
import requests
import websocket
import json
import pandas as pd
import numpy as np
import talib
from datetime import datetime

# ==========================
# Load keys
# ==========================
DERIV_API_KEY = os.getenv("DERIV_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================
# Telegram sender
# ==========================
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram send error: {e}")

# ==========================
# WebSocket to Deriv
# ==========================
def get_candles(symbol, timeframe, count=100):
    url = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    ws = websocket.create_connection(url)

    # authorize
    ws.send(json.dumps({"authorize": DERIV_API_KEY}))
    ws.recv()

    # request candles
    ws.send(json.dumps({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": timeframe,
        "style": "candles"
    }))

    raw = ws.recv()
    ws.close()

    data = json.loads(raw)
    if "candles" not in data:
        return None
    return pd.DataFrame(data["candles"])

# ==========================
# Strategy: MA rejection with candlestick
# ==========================
def generate_signal(df):
    if df is None or len(df) < 50:
        return None

    df['ma'] = talib.SMA(df['close'], timeperiod=20)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # bullish rejection
    if last['low'] <= last['ma'] and last['close'] > last['ma'] and last['close'] > prev['close']:
        return "BUY"

    # bearish rejection
    if last['high'] >= last['ma'] and last['close'] < last['ma'] and last['close'] < prev['close']:
        return "SELL"

    return None

# ==========================
# Main bot loop
# ==========================
def run_bot():
    symbols = ["R_10", "R_50", "R_75", "R_100"]
    timeframes = {  # seconds per candle
        "6min": 360,
        "10min": 600
    }

    # ✅ Only once at startup
    send_telegram_message(f"🤖 Deriv MA Rejection bot is now running: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    for symbol in symbols:
        for tf_name, tf_seconds in timeframes.items():
            df = get_candles(symbol, tf_seconds)
            if df is None:
                continue

            signal = generate_signal(df)
            if signal:
                msg = f"📊 {symbol}\n⏰ {tf_name}\n🎯 {signal}"
                send_telegram_message(msg)

if __name__ == "__main__":
    run_bot()

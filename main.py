import requests
import pandas as pd
import time
import os

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Assets
ASSETS = [
    "R_10",
    "R_50",
    "R_75",
    "R_75_1s",
    "R_100_1s",
    "R_150_1s"
]

# Timeframes in minutes
TIMEFRAMES = [6, 10]

# ==================== Candle Fetch ====================
def get_candles(symbol, timeframe=10, count=50):
    url = "https://api.deriv.com/binary/v1"
    params = {
        "candles": symbol,
        "count": count,
        "style": "candles",
        "granularity": timeframe * 60
    }
    r = requests.get(url, params=params)
    try:
        data = r.json()
    except Exception as e:
        print(f"❌ JSON error for {symbol}: {e}")
        return None

    if "candles" not in data:
        print(f"⚠️ No candle data for {symbol}")
        return None

    df = pd.DataFrame(data["candles"])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# ==================== Signal Logic ====================
def generate_signal(df):
    if df is None or len(df) < 20:
        return None

    # Moving Average
    df["MA"] = df["close"].rolling(window=10).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish rejection: candle closes above MA
    if prev["close"] < prev["MA"] and last["close"] > last["MA"]:
        return "Buy"

    # Bearish rejection: candle closes below MA
    if prev["close"] > prev["MA"] and last["close"] < last["MA"]:
        return "Sell"

    return None

# ==================== Telegram ====================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

# ==================== Main Bot ====================
def run_bot():
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            df = get_candles(asset, tf)
            signal = generate_signal(df)

            if signal:
                msg = f"📊 {asset}\n⏰ {tf}min\n🎯 {signal}"
                print(msg)
                send_telegram(msg)
            else:
                print(f"… No signal for {asset} ({tf}min)")

if __name__ == "__main__":
    run_bot()

import requests
import time
import pandas as pd
from deriv_api import DerivAPI
import asyncio
import os

# Telegram details
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Deriv token
DERIV_TOKEN = os.getenv("DERIV_API_TOKEN")

# Assets and timeframes
ASSETS = [
    "R_50",
    "R_75",
    "R_10",
    "R_75_1S",
    "R_100_1S",
    "R_150_1S",
]
TIMEFRAMES = [6, 10]  # minutes

# --- Helper functions ---
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram error:", e)

def calculate_mas(df):
    # Typical price
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
    # MA1: smoothed 9 on typical
    df["ma1"] = df["typical"].rolling(window=9, min_periods=9).mean()
    # MA2: smoothed 19 on close
    df["ma2"] = df["close"].rolling(window=19, min_periods=19).mean()
    # MA3: simple 25 on MA2
    df["ma3"] = df["ma2"].rolling(window=25, min_periods=25).mean()
    return df

def is_rejection(candle, ma_value):
    body = abs(candle["close"] - candle["open"])
    wick_up = candle["high"] - max(candle["open"], candle["close"])
    wick_down = min(candle["open"], candle["close"]) - candle["low"]

    near_ma = abs(candle["close"] - ma_value) / ma_value < 0.002  # within 0.2%

    # rejection types
    doji = body <= (candle["high"] - candle["low"]) * 0.1
    pinbar = wick_up > 2 * body or wick_down > 2 * body

    return near_ma and (doji or pinbar)

def check_signal(df, timeframe, asset):
    if len(df) < 30:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Trend filters
    uptrend = df["ma1"].iloc[-1] > df["ma2"].iloc[-1] > df["ma3"].iloc[-1]
    downtrend = df["ma1"].iloc[-1] < df["ma2"].iloc[-1] < df["ma3"].iloc[-1]

    # --- Buy signal ---
    if uptrend and (is_rejection(prev, prev["ma1"]) or is_rejection(prev, prev["ma2"])):
        if last["close"] > max(prev["ma1"], prev["ma2"]) and last["close"] > last["open"]:
            return f"✅ BUY {asset} ({timeframe}m)\nPrice: {last['close']}"

    # --- Sell signal ---
    if downtrend and (is_rejection(prev, prev["ma1"]) or is_rejection(prev, prev["ma2"])):
        if last["close"] < min(prev["ma1"], prev["ma2"]) and last["close"] < last["open"]:
            return f"❌ SELL {asset} ({timeframe}m)\nPrice: {last['close']}"

    return None

# --- Main async loop ---
async def run_bot():
    api = DerivAPI(app_id=1089)
    await api.authorize(DERIV_TOKEN)

    while True:
        for asset in ASSETS:
            for tf in TIMEFRAMES:
                candles = await api.candles({
                    "ticks_history": asset,
                    "adjust_start_time": 1,
                    "count": 100,
                    "granularity": tf * 60
                })
                df = pd.DataFrame(candles["candles"])
                df = calculate_mas(df)

                signal = check_signal(df, tf, asset)
                if signal:
                    send_telegram_message(signal)
                    print("Signal:", signal)

        # wait 10 minutes before checking again
        await asyncio.sleep(600)

if __name__ == "__main__":
    asyncio.run(run_bot())

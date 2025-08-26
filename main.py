import requests
import pandas as pd
import numpy as np
import datetime
import os

# Telegram setup
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Your assets
ASSETS = ["R_10", "R_50", "R_75", "R_75_1s", "R_100_1s", "R_150_1s"]

# Moving Average period
MA_PERIOD = 20

# Get candles from Deriv API
def get_candles(symbol, timeframe=10, count=50):
    url = "https://api.deriv.com/api/exchange/v1/candles"
    params = {
        "symbol": symbol,
        "granularity": timeframe * 60,  # seconds
        "count": count,
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "candles" not in data:
        return None
    df = pd.DataFrame(data["candles"])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df

# Simple Moving Average
def sma(series, period):
    return series.rolling(period).mean()

# Detect bullish/bearish engulfing + hammer/shooting star
def detect_pattern(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body_last = abs(last["close"] - last["open"])
    body_prev = abs(prev["close"] - prev["open"])
    wick_upper = last["high"] - max(last["close"], last["open"])
    wick_lower = min(last["close"], last["open"]) - last["low"]

    # Bullish Engulfing
    if last["close"] > last["open"] and prev["close"] < prev["open"]:
        if last["close"] > prev["open"] and last["open"] < prev["close"]:
            return "bullish_engulfing"

    # Bearish Engulfing
    if last["close"] < last["open"] and prev["close"] > prev["open"]:
        if last["close"] < prev["open"] and last["open"] > prev["close"]:
            return "bearish_engulfing"

    # Hammer (long lower wick, small body)
    if wick_lower > 2 * body_last and body_last < (last["high"] - last["low"]) * 0.3:
        return "hammer"

    # Shooting star (long upper wick, small body)
    if wick_upper > 2 * body_last and body_last < (last["high"] - last["low"]) * 0.3:
        return "shooting_star"

    return None

# Check for rejection at MA
def check_signal(df, asset, timeframe):
    df["MA"] = sma(df["close"], MA_PERIOD)
    last = df.iloc[-1]
    pattern = detect_pattern(df)

    if not pattern:
        return None

    # Buy signal
    if pattern in ["bullish_engulfing", "hammer"] and last["low"] <= last["MA"] and last["close"] > last["MA"]:
        return f"📊{asset}\n⏰{timeframe}min\n🎯Buy"

    # Sell signal
    if pattern in ["bearish_engulfing", "shooting_star"] and last["high"] >= last["MA"] and last["close"] < last["MA"]:
        return f"📊{asset}\n⏰{timeframe}min\n🎯Sell"

    return None

# Send to Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=payload)

def run_bot():
    timeframes = [6, 10]  # min
    for asset in ASSETS:
        for tf in timeframes:
            df = get_candles(asset, timeframe=tf)
            if df is not None and len(df) > MA_PERIOD:
                signal = check_signal(df, asset, tf)
                if signal:
                    send_telegram(signal)

if __name__ == "__main__":
    run_bot()

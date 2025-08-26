import requests
import pandas as pd
import talib
from datetime import datetime

# ====== CONFIG ======
DERIV_API_URL = "https://api.deriv.com/api/explorer"
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

ASSETS = {
    "R_10": "Volatility 10",
    "R_50": "Volatility 50",
    "R_75": "Volatility 75",
    "R_75_1s": "Volatility 75 (1s)",
    "R_100_1s": "Volatility 100 (1s)",
    "R_150_1s": "Volatility 150 (1s)"
}

TIMEFRAMES = {
    "6m": 6,
    "10m": 10
}

# ====== HELPERS ======
def fetch_ohlc(symbol, minutes):
    """Fetch OHLC data from Deriv API (last 200 candles)."""
    url = f"https://api.deriv.com/api/explorer/ticks_history?ticks_history={symbol}&style=candles&granularity={minutes*60}&count=200"
    resp = requests.get(url).json()
    if "candles" not in resp:
        return None
    df = pd.DataFrame(resp["candles"])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    return df

def apply_strategy(df):
    """Apply MA rejection + trend rule."""
    if df is None or len(df) < 30:
        return None

    # Moving Averages
    ma1 = talib.SMA((df["high"]+df["low"]+df["close"])/3, timeperiod=9)  # Smoothed approx
    ma2 = talib.SMA(df["close"], timeperiod=19)  # Smoothed approx
    ma3 = talib.SMA(ma2, timeperiod=25)  # Simple on MA2

    # Last two candles
    c1_open, c1_close, c1_high, c1_low = df.iloc[-2][["open", "close", "high", "low"]]
    c2_open, c2_close = df.iloc[-1][["open", "close"]]

    # Last MAs
    ma1_last, ma2_last, ma3_last = ma1.iloc[-2], ma2.iloc[-2], ma3.iloc[-2]

    # Trend check
    if ma1_last > ma2_last > ma3_last:  # Uptrend
        if c1_low <= ma2_last <= c1_high and c1_close > ma2_last:  # Rejection candle
            if c2_close > c2_open and c2_close > ma2_last:
                return "Buy"

    if ma1_last < ma2_last < ma3_last:  # Downtrend
        if c1_low <= ma2_last <= c1_high and c1_close < ma2_last:  # Rejection candle
            if c2_close < c2_open and c2_close < ma2_last:
                return "Sell"

    return None

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=payload)

# ====== MAIN LOOP ======
def main():
    for symbol, name in ASSETS.items():
        results = {}

        for label, minutes in TIMEFRAMES.items():
            df = fetch_ohlc(symbol, minutes)
            signal = apply_strategy(df)
            results[label] = signal

        # Only send if both timeframes agree
        if results["6m"] and results["6m"] == results["10m"]:
            msg = f"📊{name}\n⏰10min\n🎯{results['10m']}"
            send_telegram(msg)
            print(f"Signal sent: {msg}")
        else:
            print(f"No valid signal for {name}")

if __name__ == "__main__":
    main()

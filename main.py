import os
import json
import asyncio
import websockets
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timezone

# ====== ENV ======
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089")   # 1089 = public demo app id (ok for public data)
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "")      # optional for public market data

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    # Don't exit; still run for logs.

# ====== ASSETS & TIMEFRAMES ======
# Your exact Deriv codes:
ASSETS = {
    "R_10": "Volatility 10",
    "R_50": "Volatility 50",
    "R_75": "Volatility 75",
    "1HZ75V": "Volatility 75 (1s)",
    "1HZ100V": "Volatility 100 (1s)",
    "1HZ150V": "Volatility 150 (1s)",
}

TIMEFRAMES = {
    "6m": 360,
    "10m": 600,
}

CANDLES_COUNT = 200  # enough history for MAs

# ====== UTIL: Telegram ======
def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured; message would be:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ Telegram send failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")

# ====== TECHNICALS ======
def typical_price(df):
    return (df["high"] + df["low"] + df["close"]) / 3.0

def sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def smma(series, period):
    # Smoothed MA (Wilder/TradingView's RMA-like)
    values = series.values.astype(float)
    out = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return pd.Series(out, index=series.index)

    # seed as SMA of first 'period'
    seed = np.nanmean(values[:period])
    out[period-1] = seed
    alpha = 1.0 / period
    for i in range(period, len(values)):
        out[i] = out[i-1] + alpha * (values[i] - out[i-1])
    return pd.Series(out, index=series.index)

def detect_doji(c):
    rng = max(abs(c["high"] - c["low"]), 1e-12)
    body = abs(c["close"] - c["open"])
    return body <= 0.1 * rng  # body <=10% of range

def detect_pin_bullish(c):
    # bullish pin: long lower wick, close near/above mid, body small-mid
    total = max(abs(c["high"] - c["low"]), 1e-12)
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    body = abs(c["close"] - c["open"])
    return (lower >= 0.5 * total) and (upper <= 0.2 * total) and (c["close"] >= c["open"]) and (body <= 0.5 * total)

def detect_pin_bearish(c):
    # bearish pin: long upper wick, close near/below mid, body small-mid
    total = max(abs(c["high"] - c["low"]), 1e-12)
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    body = abs(c["close"] - c["open"])
    return (upper >= 0.5 * total) and (lower <= 0.2 * total) and (c["close"] <= c["open"]) and (body <= 0.5 * total)

def candle_is_bullish(c): return c["close"] > c["open"]
def candle_is_bearish(c): return c["close"] < c["open"]

# ====== WebSocket OHLC pull ======
async def ws_fetch_candles(symbol: str, granularity: int, count: int = CANDLES_COUNT):
    url = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
    msg = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest"
    }
    auth_msg = {"authorize": DERIV_API_KEY} if DERIV_API_KEY else None

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            # Optional authorize (not required for public data)
            if auth_msg:
                await ws.send(json.dumps(auth_msg))
                auth_reply = json.loads(await ws.recv())
                if "error" in auth_reply:
                    print(f"⚠️ Authorize error: {auth_reply['error']}")
                else:
                    print("✅ Authorized with Deriv (optional)")

            await ws.send(json.dumps(msg))
            raw = await ws.recv()
            data = json.loads(raw)
            if "error" in data:
                print(f"❌ Deriv error for {symbol}: {data['error']}")
                return None
            if "candles" not in data:
                print(f"⚠️ No 'candles' in reply for {symbol}: {str(data)[:200]}")
                return None

            candles = data["candles"]
            if not candles:
                print(f"⚠️ Empty candles for {symbol}")
                return None

            df = pd.DataFrame(candles)
            # fields are strings; cast
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            # epoch -> datetime if needed
            return df

    except Exception as e:
        print(f"❌ WS exception for {symbol} {granularity}s: {e}")
        return None

# ====== Strategy on one timeframe ======
def evaluate_signal_on_df(df: pd.DataFrame):
    """
    Returns: 'BUY' / 'SELL' / None
    Uses last two closed candles: c[-2] = rejection, c[-1] = confirmation.
    """
    if df is None or len(df) < 50:
        return None, {}

    df = df.copy()
    df["tp"] = typical_price(df)

    # MA1: SMMA(9) on typical price
    df["MA1"] = smma(df["tp"], 9)
    # MA2: SMMA(19) on close
    df["MA2"] = smma(df["close"], 19)
    # MA3: SMA(25) on previous indicator's data (MA2)
    df["MA3"] = sma(df["MA2"], 25)

    if df[["MA1", "MA2", "MA3"]].tail(3).isna().any().any():
        return None, {}

    # Use last two CLOSED candles
    c_rej = df.iloc[-2]  # rejection candle
    c_conf = df.iloc[-1] # confirmation candle

    # Trend filters
    uptrend   = (c_rej["MA1"] >= c_rej["MA2"]) and (c_rej["MA2"] >= c_rej["MA3"])
    downtrend = (c_rej["MA1"] <= c_rej["MA2"]) and (c_rej["MA2"] <= c_rej["MA3"])

    # Which MA is being retested? nearest of MA1 or MA2 to rejection candle close
    ma1_val = c_rej["MA1"]
    ma2_val = c_rej["MA2"]
    target_ma = "MA1" if abs(c_rej["close"] - ma1_val) <= abs(c_rej["close"] - ma2_val) else "MA2"
    target_val_rej = c_rej[target_ma]
    target_val_conf = c_conf[target_ma]

    # Rejection candlestick types
    is_doji   = detect_doji(c_rej)
    is_pin_b  = detect_pin_bullish(c_rej)
    is_pin_s  = detect_pin_bearish(c_rej)

    reasons = {
        "uptrend": uptrend,
        "downtrend": downtrend,
        "target_ma": target_ma,
        "rej_is_doji": bool(is_doji),
        "rej_is_pin_bullish": bool(is_pin_b),
        "rej_is_pin_bearish": bool(is_pin_s)
    }

    # ===== BUY logic =====
    buy_rejection = (
        uptrend and
        ((is_doji or is_pin_b)) and
        # rejection candle closes ABOVE target MA (not below)
        (c_rej["close"] >= target_val_rej) and
        # confirmation: bullish candle that closes above same MA
        candle_is_bullish(c_conf) and
        (c_conf["close"] >= target_val_conf)
    )
    if buy_rejection:
        return "BUY", reasons

    # ===== SELL logic =====
    sell_rejection = (
        downtrend and
        ((is_doji or is_pin_s)) and
        # rejection candle closes BELOW target MA (not above)
        (c_rej["close"] <= target_val_rej) and
        # confirmation: bearish candle that closes below same MA
        candle_is_bearish(c_conf) and
        (c_conf["close"] <= target_val_conf)
    )
    if sell_rejection:
        return "SELL", reasons

    return None, reasons

# ====== Orchestrate per asset ======
async def analyze_asset(symbol: str, pretty_name: str):
    results = {}  # {"6m": "BUY"/"SELL"/None, "10m": ...}
    details = {}

    for tf_label, gran in TIMEFRAMES.items():
        df = await ws_fetch_candles(symbol, granularity=gran, count=CANDLES_COUNT)
        if df is None:
            results[tf_label] = None
            continue
        sig, info = evaluate_signal_on_df(df)
        results[tf_label] = sig
        details[tf_label] = info

    # Single-direction resolution:
    sig6 = results.get("6m")
    sig10 = results.get("10m")

    final_signal = None
    display_tf = None
    strength = ""

    if sig6 and sig10:
        if sig6 == sig10:
            final_signal = sig10  # both agree; display 10m
            display_tf = "10min"
            strength = " (STRONG)"
        else:
            # conflict → no trade for this asset (respect your rule)
            final_signal = None
    elif sig10:
        final_signal = sig10; display_tf = "10min"
    elif sig6:
        final_signal = sig6; display_tf = "6min"

    if final_signal:
        msg = f"📊 {pretty_name}\n⏰ {display_tf}\n🎯 {final_signal}{strength}"
        send_telegram(msg)
        print("SENT:", msg)
    else:
        print(f"{pretty_name}: No trade (no valid rejection or TF conflict).")

# ====== MAIN ======
async def main():
    start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    send_telegram(f"🤖 Deriv MA Rejection bot run started: {start}")

    tasks = []
    for sym, name in ASSETS.items():
        tasks.append(analyze_asset(sym, name))
    await asyncio.gather(*tasks)

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    send_telegram(f"✅ Run completed: {end}")

if __name__ == "__main__":
    asyncio.run(main())

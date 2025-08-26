# main.py
import os
import json
import tim
import math
from datetime import datetime, timezone
from typing import List, Dict, Optional

import websocket
import pandas as pd

from bot import send_telegram_message, send_telegram_block

# =========================
# CONFIG
# =========================
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089")  # public sample app_id; can be your own
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")    # optional for public synthetic data, supported if you have it

ASSETS = ["R_50", "R_75", "R_10", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = {"6m": 360, "10m": 600}
CANDLES_PER_REQUEST = 200  # enough to compute indicators robustly

# Range avoidance: require some slope on MA9 over last K bars
SLOPE_LOOKBACK = 6
MIN_REL_SLOPE = 0.0003  # 0.03% (gentle; raise to be stricter)

# Rejection proximity to MA for "touch" (as fraction of price)
MA_TOUCH_TOL = 0.0015  # 0.15%

# =========================
# WEBSOCKET / DATA
# =========================
DERIV_WS = f"wss://ws.deriv.com/websockets/v3?app_id={DERIV_APP_ID}"

def ws_request(payload: dict) -> dict:
    """
    Open a short-lived WS, (optionally) authorize, send payload, return JSON response.
    Designed for ticks_history style requests.
    """
    ws = websocket.create_connection(DERIV_WS, timeout=20)
    try:
        if DERIV_API_TOKEN:
            ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth = json.loads(ws.recv())
            if "error" in auth:
                raise RuntimeError(f"Authorize failed: {auth['error']}")
        ws.send(json.dumps(payload))
        raw = json.loads(ws.recv())
        if "error" in raw:
            raise RuntimeError(f"Deriv error: {raw['error']}")
        return raw
    finally:
        try:
            ws.close()
        except Exception:
            pass

def fetch_candles(symbol: str, granularity: int, count: int) -> pd.DataFrame:
    """
    Returns pandas DataFrame with columns: time, open, high, low, close (floats), epoch (int)
    """
    req = {
        "ticks_history": symbol,
        "end": "latest",
        "count": count,
        "style": "candles",
        "granularity": granularity
    }
    data = ws_request(req)
    candles = data.get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    # normalize types
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("int64")
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    return df[["time", "epoch", "open", "high", "low", "close"]]

# =========================
# INDICATORS
# =========================
def smma(series: pd.Series, period: int) -> pd.Series:
    """
    Smoothed Moving Average (SMMA) aka RMA/SMMA:
    SMMA[i] = (SMMA[i-1]*(N-1) + price[i]) / N ; initialized with SMA of first N.
    """
    s = series.copy().astype(float)
    out = pd.Series(index=s.index, dtype=float)
    if len(s) < period:
        return out
    sma0 = s.iloc[:period].mean()
    out.iloc[period-1] = sma0
    alpha = 1.0 / period
    for i in range(period, len(s)):
        prev = out.iloc[i-1]
        out.iloc[i] = prev + alpha * (s.iloc[i] - prev)
    return out

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0

# =========================
# CANDLE CLASSIFIERS
# =========================
def is_doji(o, h, l, c) -> bool:
    body = abs(c - o)
    rng = max(h - l, 1e-12)
    return body <= 0.2 * rng  # body <= 20% of range

def is_pinbar(o, h, l, c) -> (bool, str):
    """
    Returns (True, 'bull'/'bear') for pin bars.
    bull-pin: long lower wick; bear-pin: long upper wick.
    """
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    # Use 2x body as minimum wick dominance
    if lower >= 2 * body and lower >= upper:
        return True, "bull"
    if upper >= 2 * body and upper >= lower:
        return True, "bear"
    return False, ""

def is_inverted_pinbar(o, h, l, c) -> (bool, str):
    # In practice same detection as pinbar but we’ll tag by wick side
    return is_pinbar(o, h, l, c)

def candle_rejection_type(row) -> Optional[str]:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if is_doji(o, h, l, c):
        return "doji"
    pin, side = is_pinbar(o, h, l, c)
    if pin:
        return f"pin_{side}"
    inv, side2 = is_inverted_pinbar(o, h, l, c)
    if inv:
        return f"inverted_{side2}"
    return None

# =========================
# STRATEGY LOGIC
# =========================
def ma_order_state(ma1, ma2, ma3) -> str:
    if ma1 > ma2 and ma2 > ma3:
        return "up"
    if ma1 < ma2 and ma2 < ma3:
        return "down"
    return "flat"

def near_ma(price: float, ma: float, tol_frac: float) -> bool:
    if ma == 0: 
        return False
    return abs(price - ma) / abs(ma) <= tol_frac

def slope_ok(ma9: pd.Series) -> bool:
    if ma9.isna().sum() > 0 or len(ma9) < SLOPE_LOOKBACK + 1:
        return False
    a = ma9.iloc[-SLOPE_LOOKBACK-1]
    b = ma9.iloc[-1]
    if a == 0 or pd.isna(a) or pd.isna(b):
        return False
    rel = abs((b - a) / a)
    return rel >= MIN_REL_SLOPE

def analyze_symbol_tf(df: pd.DataFrame, tf_label: str, symbol: str) -> Optional[str]:
    """
    Returns 'BUY' or 'SELL' or None for no signal on this timeframe.
    Implements: single rejection candle (MA1/MA2) + immediate next-candle confirmation; trend-following; range filter.
    """
    if df.empty or len(df) < 60:
        return None

    # Indicators
    tp = typical_price(df)
    df["MA1"] = smma(tp, 9)            # smoothed 9 on typical price
    df["MA2"] = smma(df["close"], 19)  # smoothed 19 on close
    df["MA3"] = sma(df["MA2"], 25)     # simple 25 on previous indicator (MA2)

    # need last two candles fully computed
    last = df.iloc[-2]     # last closed candle
    confirm = df.iloc[-1]  # the next (currently closed if GH Action runs post-close)

    if any(pd.isna(last[["MA1", "MA2", "MA3"]])) or any(pd.isna(confirm[["MA1", "MA2", "MA3"]])):
        return None

    # Range filter via MA1 slope
    if not slope_ok(df["MA1"]):
        return None

    trend = ma_order_state(last["MA1"], last["MA2"], last["MA3"])
    if trend == "flat":
        return None

    # Rejection recognition on last closed candle
    rej = candle_rejection_type(last)
    if rej is None:
        return None

    # Check which MA is tested (MA1 or MA2), and apply rule side + close location
    tested = None
    # prefer closest between MA1 & MA2
    dist1 = abs(last["close"] - last["MA1"])
    dist2 = abs(last["close"] - last["MA2"])
    candidate = "MA1" if dist1 <= dist2 else "MA2"
    if near_ma(last["close"], last[candidate], MA_TOUCH_TOL) or near_ma((last["high"]+last["low"])/2.0, last[candidate], MA_TOUCH_TOL):
        tested = candidate
    else:
        # if not close, try the other
        other = "MA2" if candidate == "MA1" else "MA1"
        if near_ma(last["close"], last[other], MA_TOUCH_TOL) or near_ma((last["high"]+last["low"])/2.0, last[other], MA_TOUCH_TOL):
            tested = other

    if tested is None:
        return None

    # BUY logic (trend up): rejection candle must close ABOVE tested MA, and confirm candle bullish above that MA
    if trend == "up":
        if last["close"] >= last[tested]:
            # next candle must be bullish and close above the same MA and above last high (confirmation)
            if confirm["close"] > confirm["open"] and confirm["close"] >= confirm[tested] and confirm["close"] > last["high"]:
                return "BUY"
    # SELL logic (trend down): rejection candle must close BELOW tested MA, next candle bearish below that MA and below last low
    if trend == "down":
        if last["close"] <= last[tested]:
            if confirm["close"] < confirm["open"] and confirm["close"] <= confirm[tested] and confirm["close"] < last["low"]:
                return "SELL"

    return None

def run_once() -> None:
    results: Dict[str, Dict[str, str]] = {}  # symbol -> {tf_label: signal}

    for symbol in ASSETS:
        symbol_signals: Dict[str, str] = {}
        for tf_label, gran in TIMEFRAMES.items():
            try:
                df = fetch_candles(symbol, gran, CANDLES_PER_REQUEST)
                sig = analyze_symbol_tf(df, tf_label, symbol)
                if sig:
                    symbol_signals[tf_label] = sig
            except Exception as e:
                print(f"{symbol} {tf_label} error: {e}")
        if symbol_signals:
            results[symbol] = symbol_signals

    # Send messages
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not results:
        # silent by design; uncomment to always notify:
        # send_telegram_message(f"ℹ️ {now} — No valid setups detected.")
        return

    for symbol, sigs in results.items():
        sides = set(sigs.values())
        if len(sides) == 1:
            side = sides.pop()
            labs = ", ".join(sorted(sigs.keys()))
            send_telegram_message(f"🔔 STRONG {side} — {symbol} — {labs} — {now}")
        else:
            parts = ", ".join(f"{tf}:{side}" for tf, side in sorted(sigs.items()))
            send_telegram_message(f"⚠️ WEAK/MIXED — {symbol} — {parts} — {now}")

if __name__ == "__main__":
    run_once()

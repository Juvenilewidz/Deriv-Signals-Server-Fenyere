# main.py
import os
import json
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
import websocket  # pip install websocket-client

from bot import (
    send_single_timeframe_signal,
    send_strong_signal,           # strong (both TFs agree)
    send_telegram_message,
)

# -------- Env / constants --------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ==========================
# Assets & Timeframes (HARD-CODED)
# ==========================
ASSETS = ["R_10", "R_50", "R_75", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [300, 600]  # 5m, 10m
CANDLES_N = 120          # enough history for the MAs

# ==========================
# Helpers: math & MAs
# ==========================
def sma(series: List[float], period: int) -> List[Optional[float]]:
    out = [None] * (period - 1)
    if len(series) < period:
        out.extend([None] * (len(series) - len(out)))
        return out
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        out.append(sum(window) / period)
    return out

def smma(series: List[float], period: int) -> List[Optional[float]]:
    """
    Wilder's/Smoothed MA: seed with SMA, then recursive:
    smma[i] = (smma[i-1]*(period-1) + price[i]) / period
    """
    n = len(series)
    if n == 0:
        return []
    if n < period:
        return [None] * n
    seed = sum(series[:period]) / period
    out: List[Optional[float]] = [None] * (period - 1)
    out.append(seed)
    prev = seed
    for i in range(period, n):
        val = (prev * (period - 1) + series[i]) / period
        out.append(val)
        prev = val
    return out

def typical_price(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0

# ==========================
# Candle helpers & patterns
# ==========================
def candle_body(o: float, c: float) -> float:
    return abs(c - o)

def candle_range(h: float, l: float) -> float:
    return max(1e-12, h - l)

def upper_wick(h: float, o: float, c: float) -> float:
    return h - max(o, c)

def lower_wick(l: float, o: float, c: float) -> float:
    return min(o, c) - l

def is_bullish(o: float, c: float) -> bool:
    return c > o

def is_bearish(o: float, c: float) -> bool:
    return c < o

def is_doji(o: float, h: float, l: float, c: float, thresh: float = 0.25) -> bool:
    rng = candle_range(h, l)
    return candle_body(o, c) <= thresh * rng

def is_bullish_pin(o: float, h: float, l: float, c: float) -> bool:
    rng = candle_range(h, l)
    lw  = lower_wick(l, o, c)
    bdy = candle_body(o, c)
    return lw >= 0.6 * rng and lw >= 1.2 * bdy

def is_bearish_pin(o: float, h: float, l: float, c: float) -> bool:
    rng = candle_range(h, l)
    uw  = upper_wick(h, o, c)
    bdy = candle_body(o, c)
    return uw >= 0.6 * rng and uw >= 1.2 * bdy

def is_bullish_engulf(prev: Dict, cur: Dict) -> bool:
    return (prev["close"] < prev["open"]) and (cur["close"] > cur["open"]) and \
           (cur["open"] <= prev["close"]) and (cur["close"] >= prev["open"])

def is_bearish_engulf(prev: Dict, cur: Dict) -> bool:
    return (prev["close"] > prev["open"]) and (cur["close"] < cur["open"]) and \
           (cur["open"] >= prev["close"]) and (cur["close"] <= prev["open"])

def pattern_name_buy(prev, cur) -> Optional[str]:
    o, h, l, c = cur["open"], cur["high"], cur["low"], cur["close"]
    if is_bullish_engulf(prev, cur): return "Bullish Engulfing"
    if is_bullish_pin(o, h, l, c):  return "Pin Bar"
    if is_bearish_pin(o, h, l, c):  return "Inverted Pin Bar"
    if is_doji(o, h, l, c):         return "Doji"
    return None

def pattern_name_sell(prev, cur) -> Optional[str]:
    o, h, l, c = cur["open"], cur["high"], cur["low"], cur["close"]
    if is_bearish_engulf(prev, cur): return "Bearish Engulfing"
    if is_bearish_pin(o, h, l, c):   return "Pin Bar"
    if is_bullish_pin(o, h, l, c):   return "Inverted Pin Bar"
    if is_doji(o, h, l, c):          return "Doji"
    return None

# ==========================
# Deriv data fetch (candles)
# ==========================
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
    try:
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        _ = json.loads(ws.recv())

        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "adjust_start_time": 1
        }
        ws.send(json.dumps(req))
        resp = json.loads(ws.recv())
        if "candles" not in resp:
            return []
        out = []
        for c in resp["candles"]:
            out.append({
                "epoch": int(c["epoch"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
            })
        return out
    finally:
        try:
            ws.close()
        except:
            pass

# ==========================
# MAs & trend
# ==========================
def compute_mas(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    hlc3    = [typical_price(h, l, c) for h, l, c in zip(highs, lows, closes)]

    # MA1: SMMA(9) on HLC/3
    ma1 = smma(hlc3, 9)
    # MA2: SMMA(19) on Close
    ma2 = smma(closes, 19)
    # MA3: SMA(25) on previous indicator's data (MA2)
    ma2_vals = [x for x in ma2 if x is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None] * len(ma2_vals)
    ma3: List[Optional[float]] = []
    j = 0
    for i in range(len(ma2)):
        if ma2[i] is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

def stacked_up(ma1, ma2, ma3, i, tol) -> bool:
    return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
            and ma1[i] >= ma2[i] - tol and ma2[i] >= ma3[i] - tol)

def stacked_down(ma1, ma2, ma3, i, tol) -> bool:
    return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
            and ma1[i] <= ma2[i] + tol and ma2[i] <= ma3[i] + tol)

     # =========================
           # Unstrict signal logic
           # ==========================
def signal_for_timeframe(candles, tf_label):
    """
    Uses:
      MA1 = SMMA(9) on HLC/3
      MA2 = SMMA(19) on Close
      MA3 = SMA(25) over MA2 (previous indicator's data)

    Rules:
      1) Pinbar/Doji (even tiny) => valid high-priority context.
      2) Reject oversized spikes: range > 2.2 * ATR(14).
      3) Require stable momentum: |ΔMA1| <= 0.8 * ATR(14).
      4) Reject exhaustion: |close - MA1| > 3 * ATR(14).
      5) Fire immediately when all checks align with trend + pullback near MA1/MA2.
    """
    reasons = []
    n = len(candles)
    if n < 30:
        return None, []

    # ----- Series -----
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    hlc3   = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]

    # ----- MAs (your parameters) -----
    ma1_list = smma(hlc3, 9)            # SMMA(9) on HLC/3
    ma2_list = smma(closes, 19)         # SMMA(19) on Close
    # SMA(25) over MA2, aligned with MA2
    ma2_vals = [x for x in ma2_list if x is not None]
    ma3_raw  = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None] * len(ma2_vals)
    ma3_list = []
    j = 0
    for i in range(len(ma2_list)):
        if ma2_list[i] is None:
            ma3_list.append(None)
        else:
            ma3_list.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1

    i_last = n - 1
    i_prev = n - 2
    ma1 = ma1_list[i_last]; ma2 = ma2_list[i_last]; ma3 = ma3_list[i_last]
    ma1p = ma1_list[i_prev] if i_prev >= 0 else None

    if any(v is None for v in (ma1, ma2, ma3, ma1p)):
        return None, []

    last = candles[i_last]
    prev = candles[i_prev]

    rng  = max(last["high"] - last["low"], 1e-12)
    body = abs(last["close"] - last["open"])
    uw   = last["high"] - max(last["open"], last["close"])
    lw   = min(last["open"], last["close"]) - last["low"]

    # ----- ATR(14) -----
    atr_len = 14
    if n < atr_len + 1:
        return None, []
    trs = []
    for k in range(n - atr_len, n):
        h = candles[k]["high"]
        l = candles[k]["low"]
        pc = candles[k - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs) / atr_len

    # ----- RULES -----
    # 1) Pinbar / Doji (even tiny) => high-priority note (doesn't force direction alone)
    is_doji   = body <= 0.15 * rng
    bull_pin  = lw >= 2.0 * body
    bear_pin  = uw >= 2.0 * body
    if is_doji or bull_pin or bear_pin:
        reasons.append("Rejection: Pin/Doji (tiny allowed)")

    # 2) Avoid oversized candles (fake spikes)
    if rng > 2.2 * atr:
        return None, []

    # 3) Momentum must be stable (MA1 slope not too steep vs ATR)
    if abs(ma1 - ma1p) > 0.8 * atr:
        return None, []

    # 4) Exhaustion: price too far from MA1
    if abs(last["close"] - ma1) > 3.0 * atr:
        return None, []

    # Pullback band (allow near-miss to MA1/MA2)
    band = 0.35 * atr

    # Trend stacking
    up_trend   = (ma1 >= ma2 >= ma3)
    down_trend = (ma1 <= ma2 <= ma3)

    # Near MA1/MA2 checks
    near_ma1 = (min(last["high"], ma1 + band) >= max(last["low"], ma1 - band))
    near_ma2 = (min(last["high"], ma2 + band) >= max(last["low"], ma2 - band))
    near_any = near_ma1 or near_ma2
    if near_ma1:
        reasons.append("Pullback near MA1 (±band)")
    elif near_ma2:
        reasons.append("Pullback near MA2 (±band)")

    # Direction & confirmation on the same (last) candle
    direction = None
    if up_trend and near_any and last["close"] >= ma1:
        direction = "BUY"
        reasons.append("Trend UP: MA1 ≥ MA2 ≥ MA3; bullish close ≥ MA1")
    elif down_trend and near_any and last["close"] <= ma1:
        direction = "SELL"
        reasons.append("Trend DOWN: MA1 ≤ MA2 ≤ MA3; bearish close ≤ MA1")

    return (direction, reasons) if direction else (None, [])
# ==========================
# Orchestrate: per asset, both TFs, resolve conflicts, notify
# ==========================
def analyze_and_notify():
    for symbol in ASSETS:
        results: Dict[int, Tuple[Optional[str], Optional[List[str]]]] = {}

        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles:
                results[tf] = (None, None)
                continue

            direction, reasons = signal_for_timeframe(candles, tf)
            results[tf] = (direction, reasons)

        # Pull 5-min (300s) and 10-min (600s) results
        sig5, rsn5   = results.get(300, (None, None))
        sig10, rsn10 = results.get(600, (None, None))

        # If both TFs agree → Strong
        if sig5 and sig10 and sig5 == sig10:
            send_strong_signal(symbol, sig5, {300: rsn5, 600: rsn10})

        # Single-timeframe signals (no conflict)
        elif sig5 and not sig10:
            send_single_timeframe_signal(symbol, 300, sig5, rsn5)
        elif sig10 and not sig5:
            send_single_timeframe_signal(symbol, 600, sig10, rsn10)
        else:
            # either both None or conflict → do nothing
            pass

if __name__ == "__main__":
    try:
        # ONE startup alert when the script process launches
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                "✅ Bot started successfully and is now live!"
            )

        analyze_and_notify()

    except Exception as e:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                f"❌ Bot crashed: {e}"
            )
        raise

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

def is_bullish_engulf(prev: Dict, cur: Dict]) -> bool:
    return (prev["close"] < prev["open"]) and (cur["close"] > cur["open"]) and \
           (cur["open"] <= prev["close"]) and (cur["close"] >= prev["open"])

def is_bearish_engulf(prev: Dict, cur: Dict]) -> bool:
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

# ==========================
# Unstrict signal logic
# ==========================
def signal_for_timeframe(candles: List[Dict], tf: int) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Unstrict ASPMI logic:
      - MA1 = SMMA(9) on HLC/3
      - MA2 = SMMA(19) on Close
      - MA3 = SMA(25) on MA2
      - Trend: stacked with a small tolerance
      - Pullback: touch OR near miss (ATR band)
      - Rejection: any of (pin/doji/engulf) OR simply wick > 0.5*body with close on the right side of MA
      - Confirmation: next candle closes in trend direction and beyond target MA
    Returns (direction, reasons) or (None, None)
    """
    if len(candles) < 60:
        return (None, None)

    ma1, ma2, ma3 = compute_mas(candles)
    i_rej = len(candles) - 2  # rejection candle index
    i_con = len(candles) - 1  # confirmation candle index

    # Need valid MAs at last two bars
    if any(x is None for x in (ma1[i_rej], ma2[i_rej], ma3[i_rej], ma1[i_con], ma2[i_con], ma3[i_con])):
        return (None, None)

    # arrays for convenience
    o_rej, h_rej, l_rej, c_rej = (candles[i_rej][k] for k in ("open", "high", "low", "close"))
    o_con, h_con, l_con, c_con = (candles[i_con][k] for k in ("open", "high", "low", "close"))

    # ATR band (near-miss allowed)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    rng = highs - lows
    atr14 = float(np.mean(rng[-14:])) if len(rng) >= 14 else float(np.mean(rng[-len(rng):]))
    band = max(0.30 * atr14, 0.0005 * closes[-1])  # 30% ATR or 5 bps of price
    tiny = max(0.05 * band, 1e-9)

    reasons: List[str] = []

    # Trend with small tolerance
    up = stacked_up(ma1, ma2, ma3, i_rej, tiny)
    dn = stacked_down(ma1, ma2, ma3, i_rej, tiny)
    if not (up or dn):
        return (None, None)

    if up:
        reasons.append("Trend UP: MA1 ≥ MA2 ≥ MA3 (tolerant).")
        # which MA is being retested (closer to the dip)
        d1 = abs(l_rej - ma1[i_rej])
        d2 = abs(l_rej - ma2[i_rej])
        ma_label, target = ("MA1", ma1[i_rej]) if d1 <= d2 else ("MA2", ma2[i_rej])

        # near/ touch check
        near = (l_rej <= target + band) or (abs(c_rej - target) <= band)
        if near:
            reasons.append(f"Pullback near {ma_label} (±band).")

            # rejection: pattern or simple wick bias with close back above MA
            patt = pattern_name_buy(candles[i_rej - 1], candles[i_rej]) if i_rej > 0 else None
            lower_w = lower_wick(l_rej, o_rej, c_rej)
            body = candle_body(o_rej, c_rej)
            rej_ok = (patt is not None) or (lower_w > 0.5 * body and c_rej >= target - tiny)

            if rej_ok:
                if patt:
                    reasons.append(f"Rejection pattern: {patt}.")
                else:
                    reasons.append("Rejection: long lower wick / close above MA.")
                # confirmation candle
                if (c_con > o_con) and (c_con >= max(target, ma1[i_con], ma2[i_con]) - tiny):
                    reasons.append("Confirmation: next candle bullish, closing above MA.")
                    return ("BUY", reasons)

    if dn:
        reasons.append("Trend DOWN: MA1 ≤ MA2 ≤ MA3 (tolerant).")
        # which MA is being retested (closer to the rally)
        d1 = abs(h_rej - ma1[i_rej])
        d2 = abs(h_rej - ma2[i_rej])
        ma_label, target = ("MA1", ma1[i_rej]) if d1 <= d2 else ("MA2", ma2[i_rej])

        near = (h_rej >= target - band) or (abs(c_rej - target) <= band)
        if near:
            reasons.append(f"Pullback near {ma_label} (±band).")

            patt = pattern_name_sell(candles[i_rej - 1], candles[i_rej]) if i_rej > 0 else None
            upper_w = upper_wick(h_rej, o_rej, c_rej)
            body = candle_body(o_rej, c_rej)
            rej_ok = (patt is not None) or (upper_w > 0.5 * body and c_rej <= target + tiny)

            if rej_ok:
                if patt:
                    reasons.append(f"Rejection pattern: {patt}.")
                else:
                    reasons.append("Rejection: long upper wick / close below MA.")
                if (c_con < o_con) and (c_con <= min(target, ma1[i_con], ma2[i_con]) + tiny):
                    reasons.append("Confirmation: next candle bearish, closing below MA.")
                    return ("SELL", reasons)

    return (None, None)

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

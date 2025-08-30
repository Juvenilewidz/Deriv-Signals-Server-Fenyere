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

     # ==========================
           # Unstrict signal logic
           # ==========================
def signal_for_timeframe(candles, tf):
    """
    ASPMI signal logic with enhanced filters:
    - Sideways/consolidation filter
    - Exhaustion filter
    - Momentum check
    - Oversized candle filter
    """

    if len(candles) < 60:
        return None, None

    opens  = np.array([c["open"]  for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    typical = (highs + lows + closes) / 3.0

    # ---------- MAs ----------
    def smoothed(values, period):
        vals = np.asarray(values, dtype=float)
        res = [np.mean(vals[:period])]
        for i in range(period, len(vals)):
            res.append((res[-1] * (period - 1) + vals[i]) / period)
        return np.concatenate([np.full(len(vals)-len(res), np.nan), np.array(res)])

    def sma_prev(values, period):
        v = np.asarray(values, dtype=float)
        out = np.full_like(v, np.nan)
        window = []
        for i in range(len(v)):
            window.append(v[i])
            if len(window) > period:
                window.pop(0)
            last = window[-period:] if len(window) >= period else None
            if last is not None and np.isfinite(last).all():
                out[i] = float(np.mean(last))
        return out

    ma1 = smoothed(typical, 9)     # HLC/3
    ma2 = smoothed(closes, 19)     # close
    ma3 = sma_prev(ma2, 25)        # SMA of MA2

    i_rej = len(candles) - 2
    i_con = len(candles) - 1

    if any(math.isnan(x) for x in (ma1[i_rej], ma2[i_rej], ma3[i_rej],
                                    ma1[i_con], ma2[i_con], ma3[i_con])):
        return None, "Invalid MA values"

    # ---------- Filters ----------
    def atr14():
        rng = highs - lows
        return float(np.mean(rng[-14:]))

    atr = atr14()
    tiny = max(0.05 * atr, 1e-9)

    def is_doji(i):
        body = abs(closes[i] - opens[i])
        rng = max(highs[i] - lows[i], 1e-9)
        return body <= 0.2 * rng

    def wick_dominates(i, side):
        body = abs(closes[i] - opens[i])
        rng = max(highs[i] - lows[i], 1e-9)
        upper = highs[i] - max(opens[i], closes[i])
        lower = min(opens[i], closes[i]) - lows[i]
        if side == "BUY":
            return lower >= 1.5 * body and lower >= 0.6 * rng
        else:
            return upper >= 1.5 * body and upper >= 0.6 * rng

    def stack_up(i):   return (ma1[i] >= ma2[i] - tiny) and (ma2[i] >= ma3[i] - tiny)
    def stack_down(i): return (ma1[i] <= ma2[i] + tiny) and (ma2[i] <= ma3[i] + tiny)

    def candle_bits(i):
        o,h,l,c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        rng  = max(h - l, tiny)
        return dict(o=o,h=h,l=l,c=c,body=body,rng=rng)

    rej = candle_bits(i_rej)
    con = candle_bits(i_con)

    # ===== FILTERS =====
    if abs(ma1[i_rej] - ma2[i_rej]) < 0.15 * atr:   # consolidation
        return None, "Sideways market"

    if abs(closes[i_rej] - ma3[i_rej]) > 3 * atr:   # exhaustion
        return None, "Exhaustion move"

    if rej["rng"] > 2.5 * atr or con["rng"] > 2.5 * atr:   # oversized candle
        return None, "Oversized candle"

    if con["rng"] < 0.8 * atr:   # weak momentum
        return None, "Weak momentum"

    # ========== BUY SETUP ==========
    if stack_up(i_rej):
        ma_tag, ma_val = ("MA1", ma1[i_rej]) if abs(lows[i_rej] - ma1[i_rej]) <= abs(lows[i_rej] - ma2[i_rej]) else ("MA2", ma2[i_rej])
        if lows[i_rej] <= ma_val + tiny and closes[i_rej] >= ma_val - tiny:
            if wick_dominates(i_rej, "BUY"):
                if not is_doji(i_con) and closes[i_con] > opens[i_con] and closes[i_con] > ma_val:
                    return "BUY", f"Trend Up | Rejected {ma_tag} | Momentum OK"

    # ========== SELL SETUP ==========
    if stack_down(i_rej):
        ma_tag, ma_val = ("MA1", ma1[i_rej]) if abs(highs[i_rej] - ma1[i_rej]) <= abs(highs[i_rej] - ma2[i_rej]) else ("MA2", ma2[i_rej])
        if highs[i_rej] >= ma_val - tiny and closes[i_rej] <= ma_val + tiny:
            if wick_dominates(i_rej, "SELL"):
                if not is_doji(i_con) and closes[i_con] < opens[i_con] and closes[i_con] < ma_val:
                    return "SELL", f"Trend Down | Rejected {ma_tag} | Momentum OK"

    return None, None
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

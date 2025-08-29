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
    out = [None]*(period-1)
    for i in range(period-1, len(series)):
        window = series[i-period+1:i+1]
        out.append(sum(window)/period)
    return out

def smma(series: List[float], period: int) -> List[Optional[float]]:
    """
    Wilder's/Smoothed MA: seed with SMA, then recursive:
    smma[i] = (smma[i-1]*(period-1) + price[i]) / period
    """
    if len(series) < period:
        return [None]*len(series)
    seed = sum(series[:period]) / period
    out: List[Optional[float]] = [None]*(period-1)
    out.append(seed)
    prev = seed
    for i in range(period, len(series)):
        val = (prev*(period-1) + series[i]) / period
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

def is_doji(o: float, h: float, l: float, c: float, thresh: float = 0.1) -> bool:
    rng = candle_range(h, l)
    return candle_body(o, c) <= thresh * rng

def is_bullish_pin(o: float, h: float, l: float, c: float) -> bool:
    rng = candle_range(h, l)
    lw  = lower_wick(l, o, c)
    bdy = candle_body(o, c)
    return lw >= 0.6 * rng and lw >= 2.0 * bdy

def is_bearish_pin(o: float, h: float, l: float, c: float) -> bool:
    rng = candle_range(h, l)
    uw  = upper_wick(h, o, c)
    bdy = candle_body(o, c)
    return uw >= 0.6 * rng and uw >= 2.0 * bdy

def is_bullish_engulf(prev, cur) -> bool:
    return (prev["close"] < prev["open"]) and (cur["close"] > cur["open"]) and \
           (cur["open"] <= prev["close"]) and (cur["close"] >= prev["open"])

def is_bearish_engulf(prev, cur) -> bool:
    return (prev["close"] > prev["open"]) and (cur["close"] < cur["open"]) and \
           (cur["open"] >= prev["close"]) and (cur["close"] <= prev["open"])

def pattern_name_buy(prev, cur) -> Optional[str]:
    o,h,l,c = cur["open"],cur["high"],cur["low"],cur["close"]
    if is_bullish_engulf(prev, cur): return "Bullish Engulfing"
    if is_bullish_pin(o,h,l,c):      return "Pin Bar"
    if is_bearish_pin(o,h,l,c):      return "Inverted Pin Bar"
    if is_doji(o,h,l,c):             return "Doji"
    return None

def pattern_name_sell(prev, cur) -> Optional[str]:
    o,h,l,c = cur["open"],cur["high"],cur["low"],cur["close"]
    if is_bearish_engulf(prev, cur): return "Bearish Engulfing"
    if is_bearish_pin(o,h,l,c):      return "Pin Bar"
    if is_bullish_pin(o,h,l,c):      return "Inverted Pin Bar"
    if is_doji(o,h,l,c):             return "Doji"
    return None

# ==========================
# Rejection vs MA checks
# ==========================
def touches_or_near_ma(c, ma_value: float) -> bool:
    """
    True if MA lies within candle high-low OR
    if close is near MA within 20% of candle range (for slight miss).
    """
    h, l, close = c["high"], c["low"], c["close"]
    rng = candle_range(h, l)
    if l <= ma_value <= h:
        return True
    return abs(close - ma_value) <= 0.2 * rng

def rejection_candle_ok_buy(prev_candle, rej_candle, ma_val) -> Tuple[bool, Optional[str]]:
    # Must touch/near MA and CLOSE ABOVE it
    if not touches_or_near_ma(rej_candle, ma_val): return (False, None)
    if rej_candle["close"] < ma_val:               return (False, None)
    name = pattern_name_buy(prev_candle, rej_candle)
    return (name is not None, name)

def rejection_candle_ok_sell(prev_candle, rej_candle, ma_val) -> Tuple[bool, Optional[str]]:
    # Must touch/near MA and CLOSE BELOW it
    if not touches_or_near_ma(rej_candle, ma_val): return (False, None)
    if rej_candle["close"] > ma_val:               return (False, None)
    name = pattern_name_sell(prev_candle, rej_candle)
    return (name is not None, name)

def confirmation_buy(candle, ma_val) -> bool:
    return is_bullish(candle["open"], candle["close"]) and (candle["close"] > ma_val)

def confirmation_sell(candle, ma_val) -> bool:
    return is_bearish(candle["open"], candle["close"]) and (candle["close"] < ma_val)
#================================================================================================================================================================# =======================================
# Softer rules + reasons
# ==========================

REJ_WICK_RATIO = 1.2      # wick must be >= 1.2x body to count as a pin-style rejection
DOJI_BODY_MAX = 0.2       # doji if body <= 20% of range
NEAR_FRAC = 0.25          # how close to MA counts as “near” (fraction of recent ATR/range)
WIGGLE_FRAC = 0.15        # let MAs be a bit out of perfect order (15% of recent range)

def _bullish(c): return c["close"] > c["open"]
def _bearish(c): return c["close"] < c["open"]

def _body(c):   return abs(c["close"] - c["open"])
def _range(c):  return c["high"] - c["low"]
def _upper_wick(c): return c["high"] - max(c["open"], c["close"])
def _lower_wick(c): return min(c["open"], c["close"]) - c["low"]

def _avg_range(candles, n=10):
    n = min(n, len(candles))
    if n <= 1: return 0.0
    return sum(_range(c) for c in candles[-n:]) / n

def _near(price, ref, tol):  # absolute distance check
    return abs(price - ref) <= tol

def trend_up(ma1, ma2, ma3, i):
    """Uptrend with wiggle-room: ma1 >= ma2 >= ma3 allowing small tolerance."""
    if i == 0: return False
    return (ma1[i] >= ma2[i] - WIGGLE and ma2[i] >= ma3[i] - WIGGLE)

def trend_down(ma1, ma2, ma3, i):
    """Downtrend with wiggle-room: ma1 <= ma2 <= ma3 allowing small tolerance."""
    if i == 0: return False
    return (ma1[i] <= ma2[i] + WIGGLE and ma2[i] <= ma3[i] + WIGGLE)

def _closest_ma(ma1, ma2, i, price):
    """Return ('MA1' or 'MA2', value) for the MA closer to price."""
    d1 = abs(price - ma1[i])
    d2 = abs(price - ma2[i])
    return ("MA1", ma1[i]) if d1 <= d2 else ("MA2", ma2[i])

def _is_doji(c):
    r = _range(c)
    return r > 0 and _body(c) <= DOJI_BODY_MAX * r

def _rejection_candle(c, target_ma, side, tol):
    """
    True if c 'rejects' target_ma in the intended direction:
    - BUY: candle probes below/near MA and **closes above** it
    - SELL: candle probes above/near MA and **closes below** it
    Accepts pin/hammer/doji-like bodies (long wick).
    """
    r = _range(c)
    if r <= 0: return False, "range=0"

    if side == "BUY":
        if c["low"] <= target_ma + tol and c["close"] >= target_ma - tol:
            # long lower wick OR doji near MA
            lw = _lower_wick(c); uw = _upper_wick(c); b = _body(c)
            if lw >= REJ_WICK_RATIO * b or _is_doji(c):
                return True, "rejection: wick/close above MA"
        return False, "no BUY-style rejection"

    if side == "SELL":
        if c["high"] >= target_ma - tol and c["close"] <= target_ma + tol:
            uw = _upper_wick(c); lw = _lower_wick(c); b = _body(c)
            if uw >= REJ_WICK_RATIO * b or _is_doji(c):
                return True, "rejection: wick/close below MA"
        return False, "no SELL-style rejection"

    return False, "unknown side"
    #========================================================================================================================


    
# ========================================================================================================================================================
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
        try: ws.close()
        except: pass

# ==========================
# Strategy core (ASPMI logic)
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
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None]*len(ma2_vals)
    ma3: List[Optional[float]] = []
    j = 0
    for i in range(len(ma2)):
        if ma2[i] is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

def trend_up(ma1, ma2, ma3, idx: int) -> bool:
    return (ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None
            and ma1[idx] >= ma2[idx] >= ma3[idx])

def trend_down(ma1, ma2, ma3, idx: int) -> bool:
    return (ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None
            and ma1[idx] <= ma2[idx] <= ma3[idx])

def choose_rejected_ma(ma1_val: float, ma2_val: float, candle: Dict) -> Tuple[str, float]:
    d1 = abs(candle["close"] - ma1_val)
    d2 = abs(candle["close"] - ma2_val)
    if d1 <= d2:
        return ("MA1", ma1_val)
    return ("MA2", ma2_val)
    #===============================================================================================================================
def signal_for_timeframe(candles: List[Dict], tf: int) -> Tuple[Optional[str], Optional[str]]:
    """
    ASPMI: Detects trend (MA stack), pullback to MA1/MA2, rejection candle, confirmation candle.
    MA1 = SMMA(9) on HLC/3
    MA2 = SMMA(19) on Close
    MA3 = SMA(25) on MA2 (previous indicator's data)
    Returns (direction, reason_text) or (None, None)
    """
    if len(candles) < 60:
        return None, None

    # ---- arrays ----
    opens  = np.array([c["open"]  for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    typical = (highs + lows + closes) / 3.0

    # ---- MAs (no pandas) ----
    def smoothed(values, period):
        v = np.asarray(values, dtype=float)
        if len(v) < period:  # not enough data
            return np.full_like(v, np.nan)
        seed = np.mean(v[:period])
        out = [np.nan]*(len(v)-period) + [seed]
        prev = seed
        for i in range(period, len(v)):
            prev = (prev*(period-1) + v[i]) / period
            out.append(prev)
        return np.array(out[-len(v):], dtype=float)

    def sma_prev(values, period):
        v = np.asarray(values, dtype=float)
        out = np.full_like(v, np.nan)
        window = []
        for i in range(len(v)):
            window.append(v[i])
            if len(window) > period:
                window.pop(0)
            if len(window) == period and np.isfinite(window).all():
                out[i] = float(np.mean(window))
        return out

    ma1 = smoothed(typical, 9)   # HLC/3
    ma2 = smoothed(closes, 19)   # close
    ma3 = sma_prev(ma2, 25)      # previous indicator's data = MA2

    i_rej = len(candles) - 2  # rejection candle
    i_con = len(candles) - 1  # confirmation candle

    # Need valid values on the last two bars
    needed = (ma1[i_rej], ma2[i_rej], ma3[i_rej], ma1[i_con], ma2[i_con], ma3[i_con])
    if any([not np.isfinite(x) for x in needed]):
        return None, "Invalid MA values"

    # ---- helpers ----
    def atr14():
        rng = highs - lows
        n = min(14, len(rng))
        return float(np.mean(rng[-n:])) if n > 0 else 0.0

    atr  = atr14()
    band = max(0.25 * atr, 0.0005 * closes[-1])  # soft 'near' band
    tiny = max(0.05 * band, 1e-12)

    def stacked_up(i):
        return (ma1[i] >= ma2[i] - tiny) and (ma2[i] >= ma3[i] - tiny)

    def stacked_down(i):
        return (ma1[i] <= ma2[i] + tiny) and (ma2[i] <= ma3[i] + tiny)

    def pattern_name(prev, cur, side: str) -> Optional[str]:
        o,h,l,c = cur["open"],cur["high"],cur["low"],cur["close"]
        if side == "BUY":
            if is_bullish_engulf(prev, cur): return "Bullish Engulfing"
            if is_bullish_pin(o,h,l,c):      return "Pin Bar"
            if is_bearish_pin(o,h,l,c):      return "Inverted Pin Bar"
            if is_doji(o,h,l,c):             return "Doji"
        else:
            if is_bearish_engulf(prev, cur): return "Bearish Engulfing"
            if is_bearish_pin(o,h,l,c):      return "Pin Bar"
            if is_bullish_pin(o,h,l,c):      return "Inverted Pin Bar"
            if is_doji(o,h,l,c):             return "Doji"
        return None

    def near_price_to(value, target):
        return abs(value - target) <= band

    # which MA is being retested (MA1 vs MA2)
    def pick_ma_for_buy(i):
        d1 = abs(lows[i]  - ma1[i])
        d2 = abs(lows[i]  - ma2[i])
        return ("MA1", ma1[i]) if d1 <= d2 else ("MA2", ma2[i])

    def pick_ma_for_sell(i):
        d1 = abs(highs[i] - ma1[i])
        d2 = abs(highs[i] - ma2[i])
        return ("MA1", ma1[i]) if d1 <= d2 else ("MA2", ma2[i])

    # ---- BUY path ----
    if stacked_up(i_rej):
        ma_tag, ma_val = pick_ma_for_buy(i_rej)

        # rejection candle must be near/touch MA and close above it
        rej = candles[i_rej]
        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else rej
        pat = pattern_name(prev, rej, "BUY")

        if pat and (rej["low"] <= ma_val + band) and (rej["close"] >= ma_val - tiny):
            # confirmation: bullish, closes above same MA
            con = candles[i_con]
            if (con["close"] > con["open"]) and (con["close"] > ma_val):
                reason = (
                    f"TF {tf//60}m | BUY\n"
                    f"• Trend: MA1 ≥ MA2 ≥ MA3 (uptrend)\n"
                    f"• Rejection: {ma_tag} touched/near & close above\n"
                    f"• Pattern: {pat} on rejection candle\n"
                    f"• Confirmation: bullish close above {ma_tag}"
                )
                return "BUY", reason

    # ---- SELL path ----
    if stacked_down(i_rej):
        ma_tag, ma_val = pick_ma_for_sell(i_rej)

        # rejection candle near/touch MA and close below it
        rej = candles[i_rej]
        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else rej
        pat = pattern_name(prev, rej, "SELL")

        if pat and (rej["high"] >= ma_val - band) and (rej["close"] <= ma_val + tiny):
            # confirmation: bearish, closes below same MA
            con = candles[i_con]
            if (con["close"] < con["open"]) and (con["close"] < ma_val):
                reason = (
                    f"TF {tf//60}m | SELL\n"
                    f"• Trend: MA1 ≤ MA2 ≤ MA3 (downtrend)\n"
                    f"• Rejection: {ma_tag} touched/near & close below\n"
                    f"• Pattern: {pat} on rejection candle\n"
                    f"• Confirmation: bearish close below {ma_tag}"
                )
                return "SELL", reason

    return None, None
    #=========================================================================================================
def analyze_and_notify():
    for symbol in ASSETS:
        results = {}

        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles:
                results[tf] = (None, None)
                continue

            res = signal_for_timeframe(candles, tf)

            # Normalize any possible shape
            if res is None:
                direction, reasons = (None, None)
            elif isinstance(res, dict):
                direction, reasons = res.get("signal"), res.get("reasons")
            else:
                try:
                    direction, reasons = res
                except Exception:
                    direction, reasons = (None, None)

            results[tf] = (direction, reasons)

        # Resolve per-TF outcomes (6m vs 10m)
        sig5, rsn5   = results.get(300, (None, None))
        sig10, rsn10 = results.get(600, (None, None))

        if sig5 and sig10 and sig5 == sig10:
            send_strong_signal(symbol, sig5, {300: rsn5, 600: rsn10})
        elif sig5 and not sig10:
            send_single_timeframe_signal(symbol, 300, sig5, rsn5)
        elif sig10 and not sig5:
            send_single_timeframe_signal(symbol, 600, sig10, rsn10)
        else:
            pass
    #================================


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

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
TIMEFRAMES = [360, 600]  # 6m, 10m
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
def signal_for_timeframe(candles, tf):
    """
    ASPMI signal logic using:
      MA1 = Smoothed(9) on Typical price (HLC/3)
      MA2 = Smoothed(19) on Close
      MA3 = Simple(25) on MA2  (previous indicator's data)
    Detects: trend (stacking), pullback to MA1/MA2, rejection candle, confirmation candle.
    Returns None or {"signal": "BUY"/"SELL", "reasons": [...]}
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
        # Wilder-style smoothed MA (EMA-like), no pandas
        vals = np.asarray(values, dtype=float)
        res = [np.mean(vals[:period])]
        for i in range(period, len(vals)):
            res.append((res[-1] * (period - 1) + vals[i]) / period)
        # pad front with NaNs for alignment
        return np.concatenate([np.full(len(vals)-len(res), np.nan), np.array(res)])

    def sma_prev(values, period):
        # SMA over 'previous indicator's data' (e.g., MA2). Needs contiguous data.
        v = np.asarray(values, dtype=float)
        out = np.full_like(v, np.nan)
        window = []
        for i in range(len(v)):
            window.append(v[i])
            if len(window) > period: window.pop(0)
            # only compute when the last 'period' values are all finite
            last = window[-period:] if len(window) >= period else None
            if last is not None and np.isfinite(last).all():
                out[i] = float(np.mean(last))
        return out

    ma1 = smoothed(typical, 9)     # HLC/3
    ma2 = smoothed(closes, 19)     # close
    ma3 = sma_prev(ma2, 25)        # previous indicator's data = MA2

    i_rej = len(candles) - 2  # rejection candle index
    i_con = len(candles) - 1  # confirmation candle index

    # Need valid MA values at the last two bars
    if any(math.isnan(x) for x in (ma1[i_rej], ma2[i_rej], ma3[i_rej], ma1[i_con], ma2[i_con], ma3[i_con])):
        return None

    # ---------- helpers ----------
    def atr14():
        rng = highs - lows
        return float(np.mean(rng[-14:]))

    atr = atr14()
    # soft-touch band around an MA: allow near-miss
    band = max(0.25 * atr, 0.0005 * closes[-1])   # ~25% ATR or 5 bps of price
    tiny = max(0.05 * band, 1e-9)                 # tiny tolerance for above/below checks

    def stack_up(i):
        # Uptrend: MA1 above MA2 above MA3 (allow tiny tolerance)
        return (ma1[i] >= ma2[i] - tiny) and (ma2[i] >= ma3[i] - tiny)

    def stack_down(i):
        # Downtrend: MA1 below MA2 below MA3 (allow tiny tolerance)
        return (ma1[i] <= ma2[i] + tiny) and (ma2[i] <= ma3[i] + tiny)

    def candle_bits(i):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        rng  = max(h - l, tiny)
        upper = h - max(o, c)
        lower = min(o, c) - l
        is_bull = c > o
        is_bear = o > c
        is_doji = body <= 0.25 * rng
        pin_low = (lower > body) and (lower > upper)           # bullish pin
        pin_high = (upper > body) and (upper > lower)          # bearish pin
        engulf_bull = (i > 0 and c > o and o <= closes[i-1] and c >= opens[i-1] and body >= 0.6 * rng)
        engulf_bear = (i > 0 and o > c and c <= closes[i-1] and o >= opens[i-1] and body >= 0.6 * rng)
        return dict(o=o,h=h,l=l,c=c,body=body,rng=rng,upper=upper,lower=lower,
                    is_bull=is_bull,is_bear=is_bear,is_doji=is_doji,
                    pin_low=pin_low,pin_high=pin_high,
                    engulf_bull=engulf_bull,engulf_bear=engulf_bear)

    def near(val, target):
        return abs(val - target) <= band

    # pick which MA (1 or 2) is being retested (closer to the relevant extreme)
    def pick_ma_for_buy(i):
        # dip towards MA: compare lows to MA1/MA2
        d1 = abs(lows[i] - ma1[i])
        d2 = abs(lows[i] - ma2[i])
        return ("MA1", ma1[i]) if d1 <= d2 else ("MA2", ma2[i])

    def pick_ma_for_sell(i):
        # rally towards MA: compare highs to MA1/MA2
        d1 = abs(highs[i] - ma1[i])
        d2 = abs(highs[i] - ma2[i])
        return ("MA1", ma1[i]) if d1 <= d2 else ("MA2", ma2[i])
        
    #=========================================================================================================
def analyze_and_notify():
    for symbol in ASSETS:
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles:
                continue

            direction, reason = signal_for_timeframe(candles, tf)
            if direction:
                msg = f"{symbol} | TF {tf//60}m | {direction} | {reason}"
                notify(msg)

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

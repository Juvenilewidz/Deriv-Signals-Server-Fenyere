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

# =========================================
# Assets & Timeframes (HARD-CODED)
# ==========================
ASSETS = ["R_10", "R_50", "R_75", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [600, 900, 1200]  # 10m, 15m, 20m
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
    """
    Fetch the latest candles INCLUDING the live (still-forming) candle.
    """
    ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
    try:
        # authorize
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        _ = json.loads(ws.recv())

        # request candles
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "subscribe": 1  # ensures live forming candle
        }
        ws.send(json.dumps(req))

        resp = json.loads(ws.recv())
        if "candles" not in resp:
            return []

        out = []
        for c in resp["candles"]:
            out.append({
                "epoch": int(c["epoch"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
            })

        # try to grab one tick update for the live candle
        try:
            update = json.loads(ws.recv())
            if "candles" in update and update["candles"]:
                live_c = update["candles"][-1]
                out[-1] = {  # replace last candle with latest forming one
                    "epoch": int(live_c["epoch"]),
                    "open": float(live_c["open"]),
                    "high": float(live_c["high"]),
                    "low": float(live_c["low"]),
                    "close": float(live_c["close"]),
                }

            # after appending new candle
            if len(out) >= 2:
                i_rej = len(out) - 2
                i_con = len(out) - 1
                direction, reason = signal_for_timeframe(out, granularity, i_rej, i_con)
                if direction:
                    send_single_timeframe_signal(symbol, granularity, direction, reason)

        except Exception:
            pass

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
def signal_for_timeframe(candles, granularity, i_rej, i_con):
    """
    ASPMI hard-coded strategy:
      - MA1: SMMA(9) on HLC/3 (typical price)
      - MA2: SMMA(19) on Close
      - MA3: SMA(25) on MA2 (previous indicator data)
    Returns (direction, reason) where direction is "BUY"/"SELL"/None.
    """
    import math
    import numpy as np

    # minimal candles required
    if not candles or len(candles) < 60:
        return None, "insufficient history"

    # === params / tolerances ===
    REJ_WICK_RATIO = 0.2      # treat even small pinbars as valid (1.0 = equal wick >= body)
    OVERSIZED_MULT  = 1.2     # reject candles with body or range > OVERSIZED_MULT * ATR
    MOMENTUM_ATR_FRAC = 0.015 # ATR must be <= 1.5% of price roughly (avoid high vol)
    EXHAUSTION_ATR_MULT = 1.8 # price must be within this ATR distance to rejected MA
    WIGGLE_FRAC = 0.25        # allow MAs slight unsorted wiggle as fraction of ATR
    CONSISTENT_DIR_PCT = 0.5  # last N bars must have >=60% same direction to be clear-moving
    CONSISTENT_BARS = 8

    # === OHLC arrays ===
    opens  = np.array([c["open"]  for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    typical = (highs + lows + closes) / 3.0
    last_price = float(closes[-1])

    # === utility MA functions ===
    def smma_array(values, period):
        vals = np.asarray(values, dtype=float)
        if len(vals) < period:
            return np.full_like(vals, np.nan)
        seed = float(np.mean(vals[:period]))
        out = [np.nan] * (period - 1) + [seed]
        prev = seed
        for i in range(period, len(vals)):
            prev = (prev * (period - 1) + float(vals[i])) / period
            out.append(prev)
        return np.array(out, dtype=float)

    def sma_array_prev_indicator(values, period):
        # SMA computed over a series that may contain NaNs; compute only when window has finite values
        vals = np.asarray(values, dtype=float)
        out = np.full(len(vals), np.nan)
        window = []
        for i in range(len(vals)):
            window.append(vals[i])
            if len(window) > period:
                window.pop(0)
            if len(window) == period and np.isfinite(window).all():
                out[i] = float(np.mean(window))
        return out

    # === compute MAs ===
    ma1 = smma_array(typical, 9)   # MA1 on HLC/3
    ma2 = smma_array(closes, 19)   # MA2 on Close
    ma3 = sma_array_prev_indicator(ma2, 25)  # MA3 on MA2 values


    # require valid MA values
    try:
        ma1_rej, ma2_rej, ma3_rej = float(ma1[i_rej]), float(ma2[i_rej]), float(ma3[i_rej])
        ma1_con, ma2_con, ma3_con = float(ma1[i_con]), float(ma2[i_con]), float(ma3[i_con])
    except Exception:
        return None, "invalid/insufficient MA data"

    # === ATR & basic stats ===
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    # avoid extremely high volatility
    if atr > MOMENTUM_ATR_FRAC * last_price:
        return None, "momentum too volatile (ATR too large)"

    # consecutive direction check (ensure price moving one clear direction)
    directions = (closes[-CONSISTENT_BARS:] - opens[-CONSISTENT_BARS:]) > 0
    pct_up = float(np.sum(directions)) / CONSISTENT_BARS
    pct_down = 1.0 - pct_up
    consistent_up = pct_up >= CONSISTENT_DIR_PCT
    consistent_down = pct_down >= CONSISTENT_DIR_PCT

    # also compute average body/range to detect oversized / choppy
    bodies = np.abs(closes - opens)
    avg_body_10 = float(np.mean(bodies[-10:])) if len(bodies) >= 10 else float(np.mean(bodies))
    avg_range_10 = float(np.mean(rngs[-10:])) if len(rngs) >= 10 else float(np.mean(rngs))

    # detect overall "ranging" as unusually large recent bodies (user said range characterized by long candles)
    if avg_body_10 > OVERSIZED_MULT * atr:
        return None, "ranging / oversized candles (avg body too large)"

    # === candle bit helpers (accept even tiny pin/doji) ===
    def candle_bits_at(i):
        o,h,l,c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
        body = abs(c - o)
        r = max(h - l, 1e-12)
        upper = h - max(o,c)
        lower = min(o,c) - l
        is_bull = c > o
        is_bear = c < o
        # doji: very small body relative to range (we accept even tiny)
        is_doji = body <= 0.35 * r   # loose threshold so tiny dojis count
        pin_low = (lower >= REJ_WICK_RATIO * body) and (lower > upper)
        pin_high = (upper >= REJ_WICK_RATIO * body) and (upper > lower)
        # engulfing relative to previous bar
        engulf_bull = False
        engulf_bear = False
        if i > 0:
            prev_o, prev_c = float(opens[i-1]), float(closes[i-1])
            prev_body = abs(prev_c - prev_o)
            if (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o) and (body >= 0.5 * r):
                engulf_bull = True
            if (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o) and (body >= 0.5 * r):
                engulf_bear = True
        return {
            "o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
            "upper": upper, "lower": lower,
            "is_bull": is_bull, "is_bear": is_bear,
            "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
            "engulf_bull": engulf_bull, "engulf_bear": engulf_bear
        }

    prev_candle = candle_bits_at(i_rej - 1) if i_rej - 1 >= 0 else None
    rej = candle_bits_at(i_rej)
    con = candle_bits_at(i_con)

    # reject if either rejection or confirmation candle is oversized / spike
    if rej["body"] > OVERSIZED_MULT * atr or rej["range"] > OVERSIZED_MULT * atr:
        return None, "rejection candle oversized"
    if con["body"] > OVERSIZED_MULT * atr or con["range"] > OVERSIZED_MULT * atr:
        return None, "confirmation candle oversized"

    # === trend check with wiggle ===
    wiggle = WIGGLE_FRAC * atr
    def is_trend_up(idx):
        return (ma1[idx] >= ma2[idx] - wiggle) and (ma2[idx] >= ma3[idx] - wiggle)
    def is_trend_down(idx):
        return (ma1[idx] <= ma2[idx] + wiggle) and (ma2[idx] <= ma3[idx] + wiggle)

    uptrend = is_trend_up(i_rej)
    downtrend = is_trend_down(i_rej)

    # ensure "price moving in one clear direction" to avoid choppy market
    if not (consistent_up or consistent_down):
        # still allow signals if trend is strongly stacked though
        if not (uptrend or downtrend):
            return None, "market too choppy / not moving in one direction"

     # === choose which MA is being retested (MA1 or MA2) ===
    # ===== DYNAMIC S/R CORE HELPERS (drop-in replacement) =============================$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def _atr_at(i):
    # Works whether 'atr' is a scalar or an array-like
    try:
        return float(atr[i])
    except Exception:
        return float(atr)

def _slope_ok_up(i, lookback=2):
    if i - lookback < 0:
        return False
    return (ma1[i] > ma1[i - lookback]) and (ma2[i] > ma2[i - lookback]) and (ma3[i] > ma3[i - lookback])

def _slope_ok_down(i, lookback=2):
    if i - lookback < 0:
        return False
    return (ma1[i] < ma1[i - lookback]) and (ma2[i] < ma2[i - lookback]) and (ma3[i] < ma3[i - lookback])

def _sep_ok(i, atr_mult=0.20):
    a = _atr_at(i)
    return (abs(ma1[i] - ma2[i]) > atr_mult * a) and (abs(ma2[i] - ma3[i]) > atr_mult * a)

def is_trend_up(i):
    # Strict uptrend: ordered + sloped + separated
    return (ma1[i] > ma2[i] > ma3[i]) and _slope_ok_up(i) and _sep_ok(i)

def is_trend_down(i):
    # Strict downtrend: ordered + sloped + separated
    return (ma1[i] < ma2[i] < ma3[i]) and _slope_ok_down(i) and _sep_ok(i)

def _nearest_ma_for_buy(i):
    """
    In an uptrend, price rejects 'dynamic support'.
    Choose whichever of MA1/MA2 is NEARER to the candle's low/close.
    """
    c = candle_bits_at(i)
    d1 = abs(c["l"] - ma1[i])
    d2 = abs(c["l"] - ma2[i])
    if d1 <= d2:
        return "MA1", float(ma1[i])
    return "MA2", float(ma2[i])

def _nearest_ma_for_sell(i):
    """
    In a downtrend, price rejects 'dynamic resistance'.
    Choose whichever of MA1/MA2 is NEARER to the candle's high/close.
    """
    c = candle_bits_at(i)
    d1 = abs(c["h"] - ma1[i])
    d2 = abs(c["h"] - ma2[i])
    if d1 <= d2:
        return "MA1", float(ma1[i])
    return "MA2", float(ma2[i])

# Keep public names if other parts of your code call them
def pick_ma_for_buy(i):
    return _nearest_ma_for_buy(i)

def pick_ma_for_sell(i):
    return _nearest_ma_for_sell(i)

# ===== REJECTION AT/NEAR DYNAMIC S/R (NO CONFIRMATION) =====
# NOTE: we keep the signature (prev_c, rej_c, ma_val, i) so existing calls still work.
# We ignore passed ma_val and re-pick the nearer MA1/MA2 dynamically inside the function.

def rejection_ok_buy(prev_c, rej_c, ma_val, i):
    """
    BUY: Only when
      - Uptrend (MA1 > MA2 > MA3, sloped, separated)
      - Rejection candle is Pin/Doji/Engulf Bull
      - Candle forms AT or NEAR dynamic support (MA1 or MA2) within ATR-based buffer
      - Fires on the rejection candle close (no next-candle confirmation)
    """
    if not is_trend_up(i):
        return False, "not clean uptrend"

    # choose MA zone dynamically (MA1 or MA2)
    ma_name, zone = _nearest_ma_for_buy(i)
    a = _atr_at(i)
    buffer_ = max(WIGGLE_FRAC * a, 0.0)  # dynamic S/R acts as a zone

    # proximity: low not far below zone; close back above/near zone
    near_zone = abs(rej_c["l"] - zone) <= buffer_
    close_side = (rej_c["c"] >= (zone - buffer_))

    # pattern filter: pin/doji/engulf bull only
    pattern_ok = bool(rej_c.get("is_doji") or rej_c.get("pin_low") or rej_c.get("engulf_bull"))

    if near_zone and close_side and pattern_ok:
        return True, f"{ma_name} dynamic support rejection (pin/doji/engulf) within {round(buffer_, 5)}"
    return False, "no valid rejection at/near MA zone"

def rejection_ok_sell(prev_c, rej_c, ma_val, i):
    """
    SELL: Only when
      - Downtrend (MA1 < MA2 < MA3, sloped, separated)
      - Rejection candle is Pin/Doji/Engulf Bear
      - Candle forms AT or NEAR dynamic resistance (MA1 or MA2) within ATR-based buffer
      - Fires on the rejection candle close (no next-candle confirmation)
    """
    if not is_trend_down(i):
        return False, "not clean downtrend"

    # choose MA zone dynamically (MA1 or MA2)
    ma_name, zone = _nearest_ma_for_sell(i)
    a = _atr_at(i)
    buffer_ = max(WIGGLE_FRAC * a, 0.0)  # dynamic S/R acts as a zone

    # proximity: high not far above zone; close back below/near zone
    near_zone = abs(rej_c["h"] - zone) <= buffer_
    close_side = (rej_c["c"] <= (zone + buffer_))

    # pattern filter: pin/doji/engulf bear only
    pattern_ok = bool(rej_c.get("is_doji") or rej_c.get("pin_high") or rej_c.get("engulf_bear"))

    if near_zone and close_side and pattern_ok:
        return True, f"{ma_name} dynamic resistance rejection (pin/doji/engulf) within {round(buffer_, 5)}"
    return False, "no valid rejection at/near MA zone"
        #==========================================================================================================

    # === confirmation checks ===
   # def confirmation_ok_buy(con_c, ma_val):
        #return con_c["is_bull"] and (con_c["c"] >= ma_val + tiny)

    #def confirmation_ok_sell(con_c, ma_val):
        #return con_c["is_bear"] and (con_c["c"] <= ma_val - tiny)

    # === exhaustion filter: price must not be stretched far from MA ===
    # (reject if rejection candle's close is > EXHAUSTION_ATR_MULT * ATR away from chosen MA)
    # Also ensure price hasn't touched MA3 recently (we require price "above MA3 but never touched it" for buys)
    #recent_lows_touch_ma3 = np.any(lows[-10:] <= ma3_rej + tiny)
    #recent_highs_touch_ma3 = np.any(highs[-10:] >= ma3_rej - tiny)  # symmetric check for sell

    # ---------------- BUY path ----------------
    if uptrend:
        # price must be above MA3 and should not have touched MA3 recently
        if not (closes[-1] > ma3_rej + tiny):
            return None, "buy rejected: price not above MA3"
        #if recent_lows_touch_ma3:
            #return None, "buy rejected: price touched MA3 recently"

        which_ma, ma_val = pick_ma_for_buy(i_rej)
        # ensure MA1 is near price packing order: MA1 close to price then MA2 then MA3
        # check order with wiggle
        if not (ma1_rej >= ma2_rej - wiggle and ma2_rej >= ma3_rej - wiggle):
            return None, "buy rejected: MAs not stacked up"

        rej_ok, rej_reason = rejection_ok_buy(prev_candle, rej, ma_val)
        if not rej_ok:
            return None, f"buy rejected: {rej_reason}"

        # exhaustion: rejection close must be near MA (not far)
        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "buy rejected: exhaustion (too far from MA)"

        # confirmation
        #if not confirmation_ok_buy(con, ma_val):
            #return None, "buy rejected: confirmation candle fail"

        # extra momentum check: average recent range shouldn't be huge (avoid spikes)
        if avg_range_10 > OVERSIZED_MULT * atr:
            return None, "buy rejected: recent range too large (spiky)"

        # passed all -> BUY
        reason = f"BUY | Trend=UP | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_low'] else ('Engulf' if rej['engulf_bull'] else 'Unknown')) } | confirm_close={con['c']:.5f}"
        return "BUY", reason

    # ---------------- SELL path ----------------
    if downtrend:
        if not (closes[-1] < ma3_rej - tiny):
            return None, "sell rejected: price not below MA3"
        #if recent_highs_touch_ma3:
            #return None, "sell rejected: price touched MA3 recently"

        which_ma, ma_val = pick_ma_for_sell(i_rej)
        if not (ma1_rej <= ma2_rej + wiggle and ma2_rej <= ma3_rej + wiggle):
            return None, "sell rejected: MAs not stacked down"

        rej_ok, rej_reason = rejection_ok_sell(prev_candle, rej, ma_val)
        if not rej_ok:
            return None, f"sell rejected: {rej_reason}"

        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "sell rejected: exhaustion (too far from MA)"

       # if not confirmation_ok_sell(con, ma_val):
            #return None, "sell rejected: confirmation candle fail"

        if avg_range_10 > OVERSIZED_MULT * atr:
            return None, "sell rejected: recent range too large (spiky)"

        reason = f"SELL | Trend=DOWN | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_high'] else ('Engulf' if rej['engulf_bear'] else 'Unknown')) } | confirm_close={con['c']:.5f}"
        return "SELL", reason

    # otherwise no clear trend
    return None, "no clear trend / signal"

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

            # define indices before calling signal_for_timeframe
            if len(candles) >= 2:
                i_rej = len(candles) - 2  # 2nd last candle = rejection
                i_con = len(candles) - 1  # last candle = confirmation
            else:
                i_rej = i_con = None  # fallback in case not enough candles

            direction, reasons = signal_for_timeframe(candles, tf, i_rej, i_con)
            results[tf] = (direction, reasons)

        # Pull 10-min (600s), 15-min (600s) and 20-min (1200) results
        sig10, rsn10 = results.get(600, (None, None))
        sig15, rsn15 = results.get(900, (None, None))
        sig20, rsn20 = results.get(1200, (None, None))

        # If both TFs agree → Strong
        if sig10 and sig15 and sig10 == sig15:
            send_strong_signal(symbol, sig10, {600: rsn10, 900: rsn15})

        if sig15 and sig20 and sig15 == sig20:
            send_strong_signal(symbol, sig15, {900: rsn15, 1200: rsn20})

        if sig10 and sig20 and sig10 == sig20:
            send_strong_signal(symbol, sig10, {600: rsn10, 1200: rsn20})

        # Single-timeframe signals (no conflict)
        elif sig10 and not sig15:
            send_single_timeframe_signal(symbol, 600, sig10, rsn10)
        elif sig15 and not sig20:
            send_single_timeframe_signal(symbol, 900, sig15, rsn15)
        elif sig20 and not sig15:
            send_single_timeframe_signal(symbol, 1200, sig20, rsn20)
        else:
            # either both None or conflict → do nothing
            pass

if __name__ == "__main__":
    try:
        # ONE startup alert when the script process launches
        # (commented out to stop repeated "bot is live" messages every 10 min)
        # if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        #     send_telegram_message(
        #         TELEGRAM_BOT_TOKEN,
        #         TELEGRAM_CHAT_ID,
        #         "✅ Bot started successfully and is now live!"
        #     )

        analyze_and_notify()

    except Exception as e:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                f"❌ Bot crashed: {e}"
            )
        raise

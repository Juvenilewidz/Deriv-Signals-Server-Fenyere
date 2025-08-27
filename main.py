# main.py
import os
import json
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

def signal_for_timeframe(candles: List[Dict], tf_seconds: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (direction, reason) where direction ∈ {"BUY","SELL",None}
    Reason explains exactly which ASPMI rule triggered.
    Uses last two closed candles: rejection = -2, confirmation = -1
    """
    if len(candles) < 60:
        return (None, None)

    ma1, ma2, ma3 = compute_mas(candles)
    i_rej = len(candles) - 2
    i_con = len(candles) - 1

    if any(x is None for x in (ma1[i_rej], ma2[i_rej], ma3[i_rej], ma1[i_con], ma2[i_con], ma3[i_con])):
        return (None, None)

    prev_candle = candles[i_rej - 1]
    rej_candle  = candles[i_rej]
    con_candle  = candles[i_con]

    # === BUY side (Uptrend; rejection on MA1 or MA2; rej closes above MA; confirm bullish above same MA)
    if trend_up(ma1, ma2, ma3, i_rej):
        which, ma_val = choose_rejected_ma(ma1[i_rej], ma2[i_rej], rej_candle)
        ok, patt = rejection_candle_ok_buy(prev_candle, rej_candle, ma_val)
        if ok and confirmation_buy(con_candle, ma_val):
            mins = 6 if tf_seconds == 360 else 10
            reason = f"Uptrend (MA9≥MA19≥SMA25), rejection on {which} with {patt} closing above, then bullish confirmation closing above on {mins}m."
            return ("BUY", reason)

    # === SELL side (Downtrend; rejection on MA1 or MA2; rej closes below MA; confirm bearish below same MA)
    if trend_down(ma1, ma2, ma3, i_rej):
        which, ma_val = choose_rejected_ma(ma1[i_rej], ma2[i_rej], rej_candle)
        ok, patt = rejection_candle_ok_sell(prev_candle, rej_candle, ma_val)
        if ok and confirmation_sell(con_candle, ma_val):
            mins = 6 if tf_seconds == 360 else 10
            reason = f"Downtrend (MA9≤MA19≤SMA25), rejection on {which} with {patt} closing below, then bearish confirmation closing below on {mins}m."
            return ("SELL", reason)

    # else: no signal
    return (None, None)

# ==========================
# Orchestrate: per asset, both TFs, resolve conflicts, notify
# ==========================
def analyze_and_notify():
    for symbol in ASSETS:
        results: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles:
                results[tf] = (None, None)
                continue
            direction, reason = signal_for_timeframe(candles, tf)
            results[tf] = (direction, reason)

        sig6, rsn6 = results.get(360, (None, None))
        sig10, rsn10 = results.get(600, (None, None))

        # If both TFs agree → Strong
        if sig6 and sig10 and sig6 == sig10:
            reasons = {360: rsn6, 600: rsn10}
            send_strong_signal(symbol, sig6, reasons)
        # Single-timeframe signals (no conflict)
        elif sig6 and not sig10:
            send_single_timeframe_signal(symbol, 360, sig6, rsn6)
        elif sig10 and not sig6:
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

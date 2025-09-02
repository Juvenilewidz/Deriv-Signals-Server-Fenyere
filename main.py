# main.py
import os
import json
import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import websocket  # pip install websocket-client

from bot import (
    send_single_timeframe_signal,
    send_strong_signal,
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

# Global cache to avoid duplicate alerts while the process runs
# Keyed by symbol -> (direction, reason)
last_sent_signal_by_symbol: Dict[str, Tuple[str, str]] = {}

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
    This function only fetches and returns candles. It does NOT run signal logic (no duplicates).
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

        # try to grab one tick update for the live candle (update last candle only)
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
# Signal logic (full, consolidation-first, duplicates suppressed)
# =========================
def signal_for_timeframe(candles, granularity, i_rej, i_con) -> Tuple[Optional[str], str]:
    """
    Returns (direction, reason) where direction in {"BUY","SELL"} or None.
    Consolidation filter runs first.
    """
    import numpy as np

    # safety
    if not candles or len(candles) < 60:
        return None, "insufficient history"

    # parameters
    REJ_WICK_RATIO = 0.2
    OVERSIZED_MULT  = 1.5
    MOMENTUM_ATR_FRAC = 0.03   # slightly looser
    EXHAUSTION_ATR_MULT = 2.0
    WIGGLE_FRAC = 0.25
    CONSISTENT_DIR_PCT = 0.6
    CONSISTENT_BARS = 8

    opens  = np.array([c["open"]  for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    typical = (highs + lows + closes) / 3.0
    last_price = float(closes[-1])

    # compute MAs (use local helpers)
    ma1, ma2, ma3 = compute_mas(candles)

    # require usable MA values at indices
    try:
        ma1_rej = float(ma1[i_rej]); ma2_rej = float(ma2[i_rej]); ma3_rej = float(ma3[i_rej])
    except Exception:
        return None, "invalid/insufficient MA data"

    # ATR
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    # volatility guard
    if atr > MOMENTUM_ATR_FRAC * last_price:
        return None, "momentum too volatile"

    # === Consolidation-first check (strict) ===
    # 1) MA flatness: check if MA slopes are tiny
    lookback = 5
    if i_rej - lookback >= 0:
        flat_ma = (
            abs(ma1[i_rej] - ma1[i_rej - lookback]) < 0.2 * atr and
            abs(ma2[i_rej] - ma2[i_rej - lookback]) < 0.2 * atr and
            (ma3[i_rej] is not None and abs(ma3[i_rej] - ma3[i_rej - lookback]) < 0.2 * atr)
        )
    else:
        flat_ma = False

    # 2) Box compression: last 8 candles total range small compared to ATR
    box_period = 8
    if len(highs) >= box_period:
        box_high = float(np.max(highs[-box_period:]))
        box_low = float(np.min(lows[-box_period:]))
        box_range = box_high - box_low
        compressed = box_range < 1.2 * atr
    else:
        compressed = False

    if flat_ma or compressed:
        return None, "consolidation detected (flat MA or compressed box)"

    # consecutive direction consistency
    if len(opens) >= CONSISTENT_BARS:
        directions = (closes[-CONSISTENT_BARS:] - opens[-CONSISTENT_BARS:]) > 0
        pct_up = float(np.sum(directions)) / CONSISTENT_BARS
        pct_down = 1.0 - pct_up
        consistent_up = pct_up >= CONSISTENT_DIR_PCT
        consistent_down = pct_down >= CONSISTENT_DIR_PCT
    else:
        consistent_up = consistent_down = False

    if not (consistent_up or consistent_down):
        # still allow if strict trend stacking exists on MAs
        wiggle = WIGGLE_FRAC * atr
        stacked_up_flag = (ma1_rej >= ma2_rej - wiggle and ma2_rej >= ma3_rej - wiggle)
        stacked_down_flag = (ma1_rej <= ma2_rej + wiggle and ma2_rej <= ma3_rej + wiggle)
        if not (stacked_up_flag or stacked_down_flag):
            return None, "no clear direction / choppy"

    # helper to extract candle bits
    def candle_bits_at(idx):
        o = float(opens[idx]); h = float(highs[idx]); l = float(lows[idx]); c = float(closes[idx])
        body = abs(c - o); r = max(h - l, 1e-12)
        upper = h - max(o, c); lower = min(o, c) - l
        is_doji = body <= 0.35 * r
        pin_low = (lower >= REJ_WICK_RATIO * body) and (lower > upper)
        pin_high = (upper >= REJ_WICK_RATIO * body) and (upper > lower)
        engulf_bull = False; engulf_bear = False
        if idx > 0:
            prev_o = float(opens[idx-1]); prev_c = float(closes[idx-1])
            if (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o):
                engulf_bull = True
            if (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o):
                engulf_bear = True
        return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
                "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
                "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

    prev_candle = candle_bits_at(i_rej - 1) if i_rej - 1 >= 0 else None
    rej = candle_bits_at(i_rej)
    con = candle_bits_at(i_con) if i_con is not None else {"c": closes[-1]}

    # oversized checks
    if rej["body"] > OVERSIZED_MULT * atr or rej["range"] > OVERSIZED_MULT * atr:
        return None, "rejection candle oversized"
    if con["body"] > OVERSIZED_MULT * atr or con["range"] > OVERSIZED_MULT * atr:
        return None, "confirmation candle oversized"

    # trend helpers (loose wiggle allowed)
    wiggle = WIGGLE_FRAC * atr
    def ma_trend_up_at(idx):
        return (ma1[idx] >= ma2[idx] - wiggle) and (ma2[idx] >= ma3[idx] - wiggle)
    def ma_trend_down_at(idx):
        return (ma1[idx] <= ma2[idx] + wiggle) and (ma2[idx] <= ma3[idx] + wiggle)

    uptrend = ma_trend_up_at(i_rej)
    downtrend = ma_trend_down_at(i_rej)

    # pick nearest MA zone (MA1 or MA2)
    def pick_ma_for_buy(idx):
        c = candle_bits_at(idx)
        d1 = abs(c["l"] - ma1[idx])
        d2 = abs(c["l"] - ma2[idx])
        return ("MA1", float(ma1[idx])) if d1 <= d2 else ("MA2", float(ma2[idx]))

    def pick_ma_for_sell(idx):
        c = candle_bits_at(idx)
        d1 = abs(c["h"] - ma1[idx])
        d2 = abs(c["h"] - ma2[idx])
        return ("MA1", float(ma1[idx])) if d1 <= d2 else ("MA2", float(ma2[idx]))

    # rejection checks (near MA zone, pattern filtered)
    def rejection_ok_buy_local(rej_c, idx):
        if not ma_trend_up_at(idx):
            return False, "not uptrend"
        ma_name, zone = pick_ma_for_buy(idx)
        buffer_ = max(WIGGLE_FRAC * atr, 0.0)
        near_zone = abs(rej_c["l"] - zone) <= buffer_
        close_side = (rej_c["c"] >= (zone - buffer_))
        pattern_ok = rej_c["is_doji"] or rej_c["pin_low"] or rej_c["engulf_bull"]
        if near_zone and close_side and pattern_ok:
            return True, f"{ma_name} support rejection"
        return False, "no valid buy rejection"

    def rejection_ok_sell_local(rej_c, idx):
        if not ma_trend_down_at(idx):
            return False, "not downtrend"
        ma_name, zone = pick_ma_for_sell(idx)
        buffer_ = max(WIGGLE_FRAC * atr, 0.0)
        near_zone = abs(rej_c["h"] - zone) <= buffer_
        close_side = (rej_c["c"] <= (zone + buffer_))
        pattern_ok = rej_c["is_doji"] or rej_c["pin_high"] or rej_c["engulf_bear"]
        if near_zone and close_side and pattern_ok:
            return True, f"{ma_name} resistance rejection"
        return False, "no valid sell rejection"

    # BUY path
    if uptrend:
        rej_ok, rej_reason = rejection_ok_buy_local(rej, i_rej)
        if not rej_ok:
            return None, f"buy rejected: {rej_reason}"
        # exhaustion
        which_ma, ma_val = pick_ma_for_buy(i_rej)
        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "buy rejected: exhaustion"
        reason = f"BUY | Trend=UP | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_low'] else ('Engulf' if rej['engulf_bull'] else 'Unknown')) } | close={rej['c']:.5f}"
        return "BUY", reason

    # SELL path
    if downtrend:
        rej_ok, rej_reason = rejection_ok_sell_local(rej, i_rej)
        if not rej_ok:
            return None, f"sell rejected: {rej_reason}"
        which_ma, ma_val = pick_ma_for_sell(i_rej)
        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "sell rejected: exhaustion"
        reason = f"SELL | Trend=DOWN | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_high'] else ('Engulf' if rej['engulf_bear'] else 'Unknown')) } | close={rej['c']:.5f}"
        return "SELL", reason

    return None, "no clear trend / signal"

# ==========================
# Orchestrate: per asset, both TFs, resolve conflicts, notify
# ==========================
def analyze_and_notify():
    # priority order for sending single signal per symbol: highest timeframe first (20m -> 15m -> 10m)
    tf_priority = sorted(TIMEFRAMES, reverse=True)

    for symbol in ASSETS:
        results: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

        # fetch candles for each TF and evaluate signals
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 3:
                results[tf] = (None, "no candles")
                continue

            i_rej = len(candles) - 2  # 2nd last candle = rejection
            i_con = len(candles) - 1  # last candle = confirmation (we don't wait on it, but keep for info)

            direction, reason = signal_for_timeframe(candles, tf, i_rej, i_con)
            results[tf] = (direction, reason)

        # pick ONE signal to send per symbol: check tf_priority and choose first TF that has a signal
        chosen = None  # (tf, direction, reason)
        for tf in tf_priority:
            dirc, rsn = results.get(tf, (None, None))
            if dirc:
                chosen = (tf, dirc, rsn)
                break

        if chosen is None:
            # nothing to send for this symbol
            continue

        tf_chosen, direction_chosen, reason_chosen = chosen

        # Deduplicate across runs: if we already sent the exact same signal for this symbol, skip
        last = last_sent_signal_by_symbol.get(symbol)
        if last and last == (direction_chosen, reason_chosen):
            # skip duplicate
            continue

        # send single-timeframe signal
        send_single_timeframe_signal(symbol, tf_chosen, direction_chosen, reason_chosen)
        # update last sent cache
        last_sent_signal_by_symbol[symbol] = (direction_chosen, reason_chosen)

# Entry
if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                f"❌ Bot crashed: {e}"
            )
        raise

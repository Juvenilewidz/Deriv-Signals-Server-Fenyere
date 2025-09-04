# main.py
import os, json, math, time, traceback
from typing import List, Dict, Optional, Tuple
import websocket
import numpy as np

from bot import send_photo_with_caption, send_simple_text  # bot.py below

# ========== CONFIG ==========
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
if not DERIV_API_KEY:
    raise RuntimeError("DERIV_API_KEY env var required")

# assets mapping (display name -> deriv symbol)
ASSETS = [
    ("V10", "R_10"),
    ("V50", "R_50"),
    ("V75", "R_75"),
    ("V75(1s)", "1HZ75V"),
    ("V100(1s)", "1HZ100V"),
    ("V150(1s)", "1HZ150V"),
]

TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = 220  # more history for smaller candles + padding

# thresholds to control spam
SEND_THRESHOLD = 0.35   # normalized [0..1] => only send signals with score >= this
REJECTED_MIN = 0.28     # send rejected notice only if score >= this (otherwise silence)

# slight preference for 5m
TF_BOOST = {300: 1.10, 600: 1.05, 900: 1.00}

# ========= Helpers (MAs & candles) =========
def typical_price(h,l,c): return (h + l + c) / 3.0

def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    out = [None]*n
    if n < period:
        return out
    seed = sum(series[:period]) / period
    out[period-1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev*(period-1) + series[i]) / period
        out[i] = prev
    return out

def sma_prev_indicator(values: List[Optional[float]], period: int) -> List[Optional[float]]:
    n = len(values)
    out = [None]*n
    window = []
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            window.clear()
            continue
        window.append(v)
        if len(window) > period:
            window.pop(0)
        if len(window) == period:
            out[i] = sum(window)/period
    return out

# ========= fetch candles from Deriv =========
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    return list of dicts with open/high/low/close/epoch
    """
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        _ = json.loads(ws.recv())
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "subscribe": 0
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
        return out
    except Exception as e:
        print("fetch_candles exception", e)
        return []
    finally:
        try: ws.close()
        except: pass

# ========= pattern helpers =========
def body(o,c): return abs(c-o)
def rng(h,l): return max(h-l, 1e-12)
def upper_wick(h,o,c): return h - max(o,c)
def lower_wick(l,o,c): return min(o,c) - l

def is_doji(o,h,l,c,th=0.30): return body(o,c) <= th * rng(h,l)
def is_bull_pin(o,h,l,c):
    r = rng(h,l); lw = lower_wick(l,o,c); b = body(o,c)
    return lw >= 0.55*r and lw >= 1.0*b
def is_bear_pin(o,h,l,c):
    r = rng(h,l); uw = upper_wick(h,o,c); b = body(o,c)
    return uw >= 0.55*r and uw >= 1.0*b

def is_bull_engulf(prev, cur):
    return (prev["close"] < prev["open"]) and (cur["close"] > cur["open"]) and (cur["open"] <= prev["close"]) and (cur["close"] >= prev["open"])
def is_bear_engulf(prev, cur):
    return (prev["close"] > prev["open"]) and (cur["close"] < cur["open"]) and (cur["open"] >= prev["close"]) and (cur["close"] <= prev["open"])

# ========= compute MAs =========
def compute_mas(candles: List[Dict]):
    closes = [c["close"] for c in candles]
    highs  = [c["high"] for c in candles]
    lows   = [c["low"] for c in candles]
    hlc3   = [typical_price(h,l,c) for h,l,c in zip(highs,lows,closes)]
    ma1 = smma(hlc3, 9)
    ma2 = smma(closes, 19)
    ma3 = sma_prev_indicator(ma2, 25)
    return ma1, ma2, ma3

# ========= scoring & evaluate single TF =========
def evaluate_tf(candles: List[Dict], tf: int) -> Tuple[Optional[str], str, float, Dict]:
    """
    Returns: (direction, reason, score_norm_0_1, debug)
    direction: "BUY"/"SELL"/None
    """
    debug = {}
    if len(candles) < 60:
        return None, "insufficient candles", 0.0, debug

    o = np.array([c["open"] for c in candles], dtype=float)
    h = np.array([c["high"] for c in candles], dtype=float)
    l = np.array([c["low"] for c in candles], dtype=float)
    cl = np.array([c["close"] for c in candles], dtype=float)
    ma1, ma2, ma3 = compute_mas(candles)

    i_rej = len(candles) - 2
    i_con = len(candles) - 1
    if i_rej < 1:
        return None, "too early", 0.0, debug

    # ATR
    ranges = h - l
    atr = float(np.mean(ranges[-14:])) if len(ranges) >= 14 else float(np.mean(ranges))
    wiggle = 0.25 * atr

    # trend checks
    def finite_at(i, arr): 
        return arr[i] is not None and not (isinstance(arr[i], float) and math.isnan(arr[i]))
    stacked_up = finite_at(i_rej, ma1) and finite_at(i_rej, ma2) and finite_at(i_rej, ma3) and (ma1[i_rej] > ma2[i_rej] > ma3[i_rej])
    stacked_down = finite_at(i_rej, ma1) and finite_at(i_rej, ma2) and finite_at(i_rej, ma3) and (ma1[i_rej] < ma2[i_rej] < ma3[i_rej])

    # slopes & separation
    def slope_up(i, look=2):
        if i - look < 0: return False
        return ma1[i] > ma1[i-look] and ma2[i] > ma2[i-look] and ma3[i] > ma3[i-look]
    def slope_down(i, look=2):
        if i - look < 0: return False
        return ma1[i] < ma1[i-look] and ma2[i] < ma2[i-look] and ma3[i] < ma3[i-look]
    def sep_ok(i):
        a = atr if atr > 0 else 1e-9
        return abs(ma1[i]-ma2[i]) > 0.2*a and abs(ma2[i]-ma3[i]) > 0.2*a

    trend_up = stacked_up and slope_up(i_rej) and sep_ok(i_rej)
    trend_down = stacked_down and slope_down(i_rej) and sep_ok(i_rej)

    # patterns
    prev = {"open": o[i_rej-1], "close": cl[i_rej-1]} if i_rej-1 >=0 else None
    rej_bar = {"open": o[i_rej], "high": h[i_rej], "low": l[i_rej], "close": cl[i_rej] }
    pat_buy = is_doji(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) or is_bull_pin(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) or (prev and is_bull_engulf(prev, rej_bar))
    pat_sell= is_doji(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) or is_bear_pin(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) or (prev and is_bear_engulf(prev, rej_bar))

    # pick nearer MA for zone
    def nearer_buy(i):
        m1 = ma1[i] if finite_at(i, ma1) else float('nan')
        m2 = ma2[i] if finite_at(i, ma2) else float('nan')
        d1 = abs(l[i] - m1)
        d2 = abs(l[i] - m2)
        return ("MA1", m1) if d1 <= d2 else ("MA2", m2)

    def nearer_sell(i):
        m1 = ma1[i] if finite_at(i, ma1) else float('nan')
        m2 = ma2[i] if finite_at(i, ma2) else float('nan')
        d1 = abs(h[i] - m1)
        d2 = abs(h[i] - m2)
        return ("MA1", m1) if d1 <= d2 else ("MA2", m2)

    direction = None
    reason = "no setup"
    score = 0.0

    # BUY path
    if trend_up and pat_buy:
        name, zone = nearer_buy(i_rej)
        if zone is not None and not math.isnan(zone):
            buf = max(0.25*atr, 1e-9)
            near_zone = abs(l[i_rej] - zone) <= buf and (cl[i_rej] >= zone - 0.1*atr)
            if near_zone:
                # score components
                sep = min(1.0, (abs(ma1[i_rej]-ma2[i_rej]) + abs(ma2[i_rej]-ma3[i_rej]))/(2*atr + 1e-9))
                prox = max(0.0, 1.0 - abs(cl[i_rej]-zone)/(1.5*atr + 1e-9))
                patw = 1.0 if is_bull_pin(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) else (0.9 if is_doji(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) else 0.8)
                base = 0.6*sep + 0.3*prox + 0.1*patw
                score = base
                direction = "BUY"
                reason = f"{name} dynamic support rejection"

    # SELL path
    if trend_down and pat_sell and direction is None:
        name, zone = nearer_sell(i_rej)
        if zone is not None and not math.isnan(zone):
            buf = max(0.25*atr, 1e-9)
            near_zone = abs(h[i_rej] - zone) <= buf and (cl[i_rej] <= zone + 0.1*atr)
            if near_zone:
                sep = min(1.0, (abs(ma1[i_rej]-ma2[i_rej]) + abs(ma2[i_rej]-ma3[i_rej]))/(2*atr + 1e-9))
                prox = max(0.0, 1.0 - abs(cl[i_rej]-zone)/(1.5*atr + 1e-9))
                patw = 1.0 if is_bear_pin(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) else (0.9 if is_doji(o[i_rej],h[i_rej],l[i_rej],cl[i_rej]) else 0.8)
                base = 0.6*sep + 0.3*prox + 0.1*patw
                score = base
                direction = "SELL"
                reason = f"{name} dynamic resistance rejection"

    # If nothing accepted, produce a “rejected candidate” score only if pattern exists and trend is present
    if direction is None:
        if (trend_up and pat_buy) or (trend_down and pat_sell):
            # weaker score so we can optionally send a rejected notice
            score = 0.45 * (1.0 if (trend_up and pat_buy) or (trend_down and pat_sell) else 0.0)
            reason = "Rejected: pattern/zone not clean enough"
        else:
            score = 0.0
            reason = "No valid setup"

    # apply TF boost
    score = min(1.0, score * TF_BOOST.get(tf, 1.0))
    debug = {"ma1": ma1, "ma2": ma2, "ma3": ma3, "i_rej": i_rej, "i_con": i_con, "atr": atr, "candles": candles}
    return direction, reason, float(score), debug

# ========= pick best TF per asset & send if above threshold =========
def analyze_asset_and_send(display_name: str, deriv_symbol: str):
    best_payload = None  # (tf, direction, reason, score, debug)
    for tf in TIMEFRAMES:
        candles = fetch_candles(deriv_symbol, tf, CANDLES_N)
        if not candles or len(candles) < 60:
            continue
        direction, reason, score, debug = evaluate_tf(candles, tf)
        if best_payload is None or score > best_payload[3]:
            best_payload = (tf, direction, reason, score, debug)

    if not best_payload:
        return  # nothing meaningful (silence)

    tf, direction, reason, score, debug = best_payload

    # decide whether to send: threshold control to avoid spam
    if direction is not None and score >= SEND_THRESHOLD:
        # accepted signal
        label = "BUY" if direction == "BUY" else "SELL"
        emoji = "✅"
    else:
        # maybe send rejected if score is above rejected-min
        if score >= REJECTED_MIN:
            label = "REJECTED"
            emoji = "❌"
        else:
            return  # skip tiny/redundant events

    # build caption and call bot to draw & send chart
    tf_label = "5m" if tf == 300 else ("10m" if tf == 600 else "15m")
    caption = f"{emoji} {display_name} · {tf_label}\n{label}\n{reason}\nScore: {score:.2f}"
    try:
        ma1 = debug["ma1"]; ma2 = debug["ma2"]; ma3 = debug["ma3"]
        i_rej = debug["i_rej"]; i_con = debug["i_con"]; candles = debug["candles"]
        # send chart & caption (bot handles chart rendering)
        send_photo_with_caption(display_name, tf, caption, candles, ma1, ma2, ma3, i_rej, i_con)
    except Exception as e:
        print("send failed", e)

# ========= main loop (single run) =========
def main():
    for display_name, symbol in ASSETS:
        try:
            analyze_asset_and_send(display_name, symbol)
        except Exception as e:
            tb = traceback.format_exc()
            try:
                send_simple_text(f"❌ Bot error analyzing {display_name}:\n{str(e)[:120]}")
            except:
                pass
            print("Error", display_name, e, tb)

if __name__ == "__main__":
    main()

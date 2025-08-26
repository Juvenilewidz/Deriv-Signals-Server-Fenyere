import os
import json
import math
import time
import statistics
import requests
import websocket
from typing import List, Dict, Tuple, Optional

# =========================
# ENV & CONSTANTS
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")  # read-only is enough
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089")

if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and DERIV_API_TOKEN):
    print("Missing one or more env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DERIV_API_TOKEN")
    # Do not raise — just exit quietly to keep Action green without spamming.
    raise SystemExit(0)

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Your exact assets
ASSETS = ["R_50", "R_75", "R_10", "1HZ75V", "1HZ100V", "1HZ150V"]
# 6 minutes and 10 minutes (Deriv granularity in seconds)
TIMEFRAMES = [360, 600]

# Candles per request (enough for MAs, ATR, pivots & patterns)
HISTORY_COUNT = 400

# Trend / ranging filter knobs
ATR_PERIOD = 14
SLOPE_LOOKBACK = 10  # bars for MA1 slope
# market considered "trending" if slope magnitude over SLOPE_LOOKBACK bars is at least 0.25 * ATR
TREND_MIN_SLOPE_ATR_MULT = 0.25

# Rejection heuristics
DOJI_BODY_TO_RANGE_MAX = 0.2    # body <= 20% of range
PIN_TAIL_MIN_FRACTION  = 0.66   # long tail ~66% of range
BODY_MAX_OF_RANGE      = 0.35   # pin bars have smallish bodies

# Dedup: within one run we avoid duplicate messages per symbol/TF; across runs rely on your Action schedule
# =========================
# UTILITIES
# =========================

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram send failed:", e)

def ws_call(payload: dict) -> dict:
    """One-shot WebSocket call: connect, authorize, send request, read one response, close."""
    ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
    try:
        ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
        ws.recv()  # auth response
        ws.send(json.dumps(payload))
        raw = ws.recv()
        return json.loads(raw)
    finally:
        ws.close()

def get_candles(symbol: str, granularity: int, count: int) -> List[dict]:
    """Return list of candles dicts: {o,h,l,c,epoch} oldest->newest."""
    resp = ws_call({"candles": symbol, "granularity": granularity, "count": count})
    candles = resp.get("candles", [])
    # Some symbols/timeframes might return fewer candles during quiet times; handle gracefully
    candles = sorted(candles, key=lambda x: x["epoch"])
    return candles

def typical_price(c):
    return (float(c["h"]) + float(c["l"]) + float(c["c"])) / 3.0

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out, q = [], []
    s = 0.0
    for v in values:
        q.append(v)
        s += v
        if len(q) > period:
            s -= q.pop(0)
        out.append(s / period if len(q) == period else None)
    return out

def smma(values: List[float], period: int) -> List[Optional[float]]:
    """Smoothed MA (Wilder/SMMA). seed with SMA, then: smma = (prev_smma*(p-1)+price)/p"""
    out: List[Optional[float]] = [None]*len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period-1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (prev*(period-1) + values[i]) / period
        out[i] = prev
    return out

def true_range(prev_close, high, low):
    return max(high - low, abs(high - prev_close), abs(prev_close - low))

def atr(candles: List[dict], period: int) -> List[Optional[float]]:
    if len(candles) < period + 1:
        return [None]*len(candles)
    trs = [None]
    for i in range(1, len(candles)):
        prev_c = float(candles[i-1]["c"])
        h = float(candles[i]["h"]); l = float(candles[i]["l"])
        trs.append(true_range(prev_c, h, l))
    return smma([x if x is not None else 0.0 for x in trs[1:]], period)  # align approx

def pivot_high(candles, i, left=2, right=2):
    if i < left or i+right >= len(candles): return False
    price = float(candles[i]["h"])
    for j in range(i-left, i+right+1):
        if j == i: continue
        if float(candles[j]["h"]) >= price: return False
    return True

def pivot_low(candles, i, left=2, right=2):
    if i < left or i+right >= len(candles): return False
    price = float(candles[i]["l"])
    for j in range(i-left, i+right+1):
        if j == i: continue
        if float(candles[j]["l"]) <= price: return False
    return True

def detect_double_top_bottom(candles) -> Tuple[Optional[Tuple[int,int,float]], Optional[Tuple[int,int,float]]]:
    """Return (double_top_info, double_bottom_info) where each info=(i1,i2,level)."""
    pivots_h = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
    pivots_l = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
    # crude: last two highs nearly equal within 0.15% and spaced >= 5 bars
    dt = None
    if len(pivots_h) >= 2:
        i1,p1 = pivots_h[-2]; i2,p2 = pivots_h[-1]
        if abs(p1-p2)/((p1+p2)/2.0) <= 0.0015 and (i2-i1) >= 5:
            dt = (i1,i2,(p1+p2)/2.0)
    db = None
    if len(pivots_l) >= 2:
        i1,p1 = pivots_l[-2]; i2,p2 = pivots_l[-1]
        if abs(p1-p2)/((p1+p2)/2.0) <= 0.0015 and (i2-i1) >= 5:
            db = (i1,i2,(p1+p2)/2.0)
    return dt, db

def detect_triangles(candles) -> Tuple[bool,bool]:
    """Very light triangle check: lower highs with flat lows -> descending; higher lows with flat highs -> ascending."""
    highs = [float(c["h"]) for c in candles]
    lows  = [float(c["l"]) for c in candles]
    last = len(candles)
    if last < 40: return (False, False)
    # recent window
    H = highs[-30:]; L = lows[-30:]
    # flatness via small std of min edge, and monotonic trend on other edge
    flat_low = statistics.pstdev(L) <= (sum(L)/len(L))*0.001
    flat_high= statistics.pstdev(H) <= (sum(H)/len(H))*0.001
    lower_highs = all(H[i] > H[i+1] for i in range(len(H)-1))
    higher_lows = all(L[i] < L[i+1] for i in range(len(L)-1))
    descending = flat_low and lower_highs
    ascending  = flat_high and higher_lows
    return ascending, descending

def detect_head_shoulders(candles) -> Tuple[bool,bool]:
    """Toy H&S from last ~50 bars using three pivot highs/lows."""
    highs = [float(c["h"]) for c in candles]
    lows  = [float(c["l"]) for c in candles]
    n = len(candles)
    if n < 50: return (False, False)
    # find last 5 pivot highs/lows and check for head in middle
    pivH = [i for i in range(2, n-2) if pivot_high(candles, i)]
    pivL = [i for i in range(2, n-2) if pivot_low(candles, i)]
    inv = False; reg = False
    if len(pivH) >= 3:
        a,b,c = pivH[-3], pivH[-2], pivH[-1]
        if highs[b] > highs[a] and highs[b] > highs[c] and abs(highs[a]-highs[c]) / highs[b] < 0.02:
            reg = True
    if len(pivL) >= 3:
        a,b,c = pivL[-3], pivL[-2], pivL[-1]
        if lows[b] < lows[a] and lows[b] < lows[c] and abs(lows[a]-lows[c]) / lows[b] < 0.02:
            inv = True
    return (reg, inv)

def is_doji(c):
    o = float(c["o"]); c_ = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    body = abs(c_-o)
    return body <= DOJI_BODY_TO_RANGE_MAX * rng

def is_bullish(c): return float(c["c"]) > float(c["o"])
def is_bearish(c): return float(c["c"]) < float(c["o"])

def is_pin_bar(c, bullish=True):
    o = float(c["o"]); cc = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    body = abs(cc-o)
    # body smallish
    if body > BODY_MAX_OF_RANGE * rng: return False
    upper = h - max(o,cc)
    lower = min(o,cc) - l
    if bullish:
        # long lower tail
        return lower >= PIN_TAIL_MIN_FRACTION * rng and upper <= (1-PIN_TAIL_MIN_FRACTION)*rng
    else:
        # long upper tail
        return upper >= PIN_TAIL_MIN_FRACTION * rng and lower <= (1-PIN_TAIL_MIN_FRACTION)*rng

def rejection_on_ma(c, ma_value, direction: str):
    """direction='above' for bullish rejection (close above MA), 'below' for bearish."""
    close = float(c["c"]); o = float(c["o"]); h=float(c["h"]); l=float(c["l"])
    if direction == "above":
        # candle must NOT close below MA, and wick pierces or kisses MA
        cond_close = close >= ma_value
        cond_touch = (l <= ma_value <= h) or (min(o,close) <= ma_value <= max(o,close))
        return cond_close and cond_touch
    else:
        cond_close = close <= ma_value
        cond_touch = (l <= ma_value <= h) or (min(o,close) <= ma_value <= max(o,close))
        return cond_close and cond_touch

def nearest_sr(candles, direction: str) -> Optional[float]:
    """Return nearest swing S/R as TP zone in trend direction."""
    pivH = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
    pivL = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
    if direction == "buy":
        # nearest resistance above last close
        last_close = float(candles[-1]["c"])
        above = [p for _,p in pivH if p > last_close]
        return min(above) if above else None
    else:
        last_close = float(candles[-1]["c"])
        below = [p for _,p in pivL if p < last_close]
        return max(below) if below else None

# =========================
# SIGNAL ENGINE
# =========================

def analyze_symbol_tf(symbol: str, tf: int) -> Optional[str]:
    candles = get_candles(symbol, tf, HISTORY_COUNT)
    if len(candles) < 60:
        return None

    closes = [float(c["c"]) for c in candles]
    highs  = [float(c["h"]) for c in candles]
    lows   = [float(c["l"]) for c in candles]
    typ    = [typical_price(c) for c in candles]

    # Indicators
    ma1 = smma(typ, 9)            # smoothed on typical
    ma2 = smma(closes, 19)        # smoothed on close
    # MA3 applied to previous indicator data (MA2 values): simple 25
    ma2_clean = [v for v in ma2]  # may contain None initially
    # replace initial None by earliest available seed to allow SMA to start later
    first_valid = next((i for i,v in enumerate(ma2_clean) if v is not None), None)
    if first_valid is None:
        return None
    for i in range(first_valid):
        ma2_clean[i] = ma2_clean[first_valid]
    ma3 = sma(ma2_clean, 25)

    # ATR and slope (range filter)
    atr_vals = atr(candles, ATR_PERIOD)
    i_last = len(candles) - 1
    if i_last < SLOPE_LOOKBACK + 1: 
        return None
    # slope of MA1 over last SLOPE_LOOKBACK bars
    # use last valid ma1 indices
    last_indices = [i for i in range(i_last-SLOPE_LOOKBACK, i_last+1) if ma1[i] is not None]
    if len(last_indices) < 2: 
        return None
    start_i, end_i = last_indices[0], last_indices[-1]
    slope = ma1[end_i] - ma1[start_i]
    atr_now = atr_vals[-1] if atr_vals[-1] is not None else None
    if atr_now is None or abs(slope) < TREND_MIN_SLOPE_ATR_MULT * atr_now:
        # Ranging → skip
        return None

    # Trend state
    last_ma1 = ma1[-1]; last_ma2 = ma2[-1]; last_ma3 = ma3[-1]
    last_close = closes[-1]
    # "MA1 closest to price" check
    d1 = abs(last_close - last_ma1) if last_ma1 else 1e9
    d2 = abs(last_close - last_ma2) if last_ma2 else 1e9
    d3 = abs(last_close - last_ma3) if last_ma3 else 1e9
    ma1_nearest = d1 <= d2 and d1 <= d3

    uptrend   = ma1_nearest and last_ma2 is not None and last_ma3 is not None and (last_ma2 < last_ma1) and (last_ma3 < last_ma1) and (last_ma2 >= last_ma3*0.95)
    downtrend = ma1_nearest and last_ma2 is not None and last_ma3 is not None and (last_ma2 > last_ma1) and (last_ma3 > last_ma1) and (last_ma2 <= last_ma3*1.05)

    # Rejection & confirmation logic using MA1 or MA2 (the two valid retest lines)
    # Use last two completed candles: [-2] rejection, [-1] confirmation
    rej_c = candles[-2]; conf_c = candles[-1]
    msg_parts = []
    signal = None
    used_ma_name = None

    def check_buy_on(ma_val, name):
        nonlocal signal, used_ma_name
        # rejection candle closes above MA and is one of (doji/pin)
        is_rej = rejection_on_ma(rej_c, ma_val, "above") and (is_doji(rej_c) or is_pin_bar(rej_c, bullish=True) or is_pin_bar(rej_c, bullish=False))
        is_conf= is_bullish(conf_c) and float(conf_c["c"]) >= ma_val
        if is_rej and is_conf:
            signal = "BUY"; used_ma_name = name

    def check_sell_on(ma_val, name):
        nonlocal signal, used_ma_name
        is_rej = rejection_on_ma(rej_c, ma_val, "below") and (is_doji(rej_c) or is_pin_bar(rej_c, bullish=True) or is_pin_bar(rej_c, bullish=False))
        is_conf= is_bearish(conf_c) and float(conf_c["c"]) <= ma_val
        if is_rej and is_conf:
            signal = "SELL"; used_ma_name = name

    if uptrend:
        if ma1[-2] and ma2[-2]:
            # prefer MA1 retest; else MA2
            if ma1[-2]: check_buy_on(ma1[-2], "MA1")
            if signal is None and ma2[-2]: check_buy_on(ma2[-2], "MA2")
    elif downtrend:
        if ma1[-2] and ma2[-2]:
            if ma1[-2]: check_sell_on(ma1[-2], "MA1")
            if signal is None and ma2[-2]: check_sell_on(ma2[-2], "MA2")

    if signal is None:
        return None  # no valid MA+rejection+confirmation per your strict rules

    # Pattern alignment (boost probability; if pattern contradicts, drop)
    asc, desc = detect_triangles(candles)
    dt, db = detect_double_top_bottom(candles)
    hs, ihs = detect_head_shoulders(candles)

    pattern_ok = True
    patterns = []
    if signal == "BUY":
        if asc: patterns.append("Ascending Triangle")
        if db:  patterns.append("Double Bottom")
        if ihs: patterns.append("Inverse H&S")
        # if clearly bearish pattern exists, invalidate
        bearish_flags = (dt is not None) or desc or hs
        if bearish_flags: pattern_ok = False
    else:  # SELL
        if desc: patterns.append("Descending Triangle")
        if dt:  patterns.append("Double Top")
        if hs:  patterns.append("Head & Shoulders")
        bullish_flags = (db is not None) or asc or ihs
        if bullish_flags: pattern_ok = False

    if not pattern_ok:
        return None

    # Break & retest on MA: ensure recent cross and return
    # Check last 6 bars: price crossed MA1 or MA2 then came back to reject
    def crossed_back(series_ma):
        lookback = 6
        cross = False
        for i in range(len(candles)-lookback-1, len(candles)-2):
            if i < 1 or series_ma[i] is None: continue
            pc = float(candles[i]["c"])
            prev = float(candles[i-1]["c"])
            if signal == "BUY":
                if prev < series_ma[i-1] and pc > series_ma[i]:
                    cross = True; break
            else:
                if prev > series_ma[i-1] and pc < series_ma[i]:
                    cross = True; break
        return cross

    br_ok = crossed_back(ma1) or crossed_back(ma2)
    if not br_ok:
        return None

    # TP zone: nearest S/R in signal direction
    tp = nearest_sr(candles, "buy" if signal=="BUY" else "sell")

    # Build message
    tf_label = "6m" if tf == 360 else "10m"
    price_now = closes[-1]
    patt_txt = (", ".join(patterns)) if patterns else "MA confluence"
    used_ma_print = used_ma_name or "MA"
    tp_txt = f"{tp:.2f}" if tp else "—"

    text = (
        f"📣 <b>{symbol}</b> | TF: <b>{tf_label}</b>\n"
        f"Signal: <b>{signal}</b>\n"
        f"Reason: {patt_txt} + rejection on <b>{used_ma_print}</b> with confirmation\n"
        f"Price: {price_now:.2f}\n"
        f"TP zone (nearest S/R): {tp_txt}"
    )
    return text

# =========================
# MAIN
# =========================

def main():
    messages = []
    for s in ASSETS:
        for tf in TIMEFRAMES:
            try:
                txt = analyze_symbol_tf(s, tf)
                if txt:
                    messages.append(txt)
            except Exception as e:
                # Keep going even if one pair/tf fails
                print(f"Error on {s}@{tf}: {e}")
                continue

    # Per your instruction: send ONLY if valid signal(s)
    if messages:
        send_telegram("✅ <b>Signals</b>\n\n" + "\n\n".join(messages))
    else:
        # stay silent if no signal (as requested)
        print("No valid signals this run.")

if __name__ == "__main__":
    main()

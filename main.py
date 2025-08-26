import os
import json
import math
import statistics
import requests
import websocket
from typing import List, Optional, Tuple, Dict

# =========================
# ENV & CONSTANTS
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DERIV_API_TOKEN    = os.getenv("DERIV_API_TOKEN")
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089")

if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and DERIV_API_TOKEN):
    print("Missing one or more env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DERIV_API_TOKEN")
    raise SystemExit(0)

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Assets & timeframes (granularity in seconds)
ASSETS = ["R_50", "R_75", "R_10", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [360, 600]  # 6m, 10m

HISTORY_COUNT = 400      # enough for indicators/patterns
ATR_PERIOD = 14
SLOPE_LOOKBACK = 10
TREND_MIN_SLOPE_ATR_MULT = 0.25

# Rejection heuristics
DOJI_BODY_TO_RANGE_MAX = 0.2
PIN_TAIL_MIN_FRACTION  = 0.66
BODY_MAX_OF_RANGE      = 0.35

# Rejection proximity tolerance if candle didn’t literally touch the MA:
# allow within 5% of ATR from the MA to count as “near miss” rejection
REJECT_ATR_TOL = 0.05

# =========================
# HELPERS
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
    ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
    try:
        ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
        ws.recv()  # auth resp
        ws.send(json.dumps(payload))
        raw = ws.recv()
        return json.loads(raw)
    finally:
        ws.close()

def get_candles(symbol: str, granularity: int, count: int) -> List[dict]:
    resp = ws_call({"candles": symbol, "granularity": granularity, "count": count})
    candles = resp.get("candles", [])
    candles = sorted(candles, key=lambda x: x["epoch"])
    return candles

def typical_price(c):  # HLC/3
    return (float(c["h"]) + float(c["l"]) + float(c["c"])) / 3.0

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out, q, s = [], [], 0.0
    for v in values:
        q.append(v); s += v
        if len(q) > period:
            s -= q.pop(0)
        out.append(s / period if len(q) == period else None)
    return out

def smma(values: List[float], period: int) -> List[Optional[float]]:
    out = [None]*len(values)
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
    # Drop the first None to align roughly
    tr_vals = [t for t in trs[1:]]
    return smma(tr_vals, period)

def pivot_high(candles, i, L=2, R=2):
    if i < L or i+R >= len(candles): return False
    price = float(candles[i]["h"])
    for j in range(i-L, i+R+1):
        if j == i: continue
        if float(candles[j]["h"]) >= price: return False
    return True

def pivot_low(candles, i, L=2, R=2):
    if i < L or i+R >= len(candles): return False
    price = float(candles[i]["l"])
    for j in range(i-L, i+R+1):
        if j == i: continue
        if float(candles[j]["l"]) <= price: return False
    return True

def detect_double_top_bottom(candles) -> Tuple[Optional[Tuple[int,int,float]], Optional[Tuple[int,int,float]]]:
    pivH = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
    pivL = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
    dt = None; db = None
    if len(pivH) >= 2:
        i1,p1 = pivH[-2]; i2,p2 = pivH[-1]
        if abs(p1-p2)/max((p1+p2)/2.0,1e-9) <= 0.0015 and (i2-i1) >= 5:
            dt = (i1,i2,(p1+p2)/2.0)
    if len(pivL) >= 2:
        i1,p1 = pivL[-2]; i2,p2 = pivL[-1]
        if abs(p1-p2)/max((p1+p2)/2.0,1e-9) <= 0.0015 and (i2-i1) >= 5:
            db = (i1,i2,(p1+p2)/2.0)
    return dt, db

def detect_triangles(candles) -> Tuple[bool,bool]:
    n = len(candles)
    if n < 40: return (False, False)
    H = [float(c["h"]) for c in candles[-30:]]
    L = [float(c["l"]) for c in candles[-30:]]
    mH = sum(H)/len(H); mL = sum(L)/len(L)
    flat_low  = statistics.pstdev(L) <= mL*0.001
    flat_high = statistics.pstdev(H) <= mH*0.001
    lower_highs = all(H[i] > H[i+1] for i in range(len(H)-1))
    higher_lows = all(L[i] < L[i+1] for i in range(len(L)-1))
    descending = flat_low and lower_highs
    ascending  = flat_high and higher_lows
    return ascending, descending

def detect_head_shoulders(candles) -> Tuple[bool,bool]:
    n = len(candles)
    if n < 50: return (False, False)
    highs = [float(c["h"]) for c in candles]
    lows  = [float(c["l"]) for c in candles]
    pivH = [i for i in range(2, n-2) if pivot_high(candles, i)]
    pivL = [i for i in range(2, n-2) if pivot_low(candles, i)]
    reg = False; inv = False
    if len(pivH) >= 3:
        a,b,c = pivH[-3], pivH[-2], pivH[-1]
        if highs[b] > highs[a] and highs[b] > highs[c] and abs(highs[a]-highs[c]) / max(highs[b],1e-9) < 0.02:
            reg = True
    if len(pivL) >= 3:
        a,b,c = pivL[-3], pivL[-2], pivL[-1]
        if lows[b] < lows[a] and lows[b] < lows[c] and abs(lows[a]-lows[c]) / max(abs(lows[b]),1e-9) < 0.02:
            inv = True
    return reg, inv

def is_doji(c):
    o = float(c["o"]); cl = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    return abs(cl-o) <= DOJI_BODY_TO_RANGE_MAX * rng

def is_bullish(c): return float(c["c"]) > float(c["o"])
def is_bearish(c): return float(c["c"]) < float(c["o"])

def is_pin_bar(c, bullish=True):
    o = float(c["o"]); cl = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    body = abs(cl-o)
    if body > BODY_MAX_OF_RANGE * rng: return False
    upper = h - max(o,cl)
    lower = min(o,cl) - l
    if bullish:
        return lower >= PIN_TAIL_MIN_FRACTION * rng and upper <= (1-PIN_TAIL_MIN_FRACTION)*rng
    else:
        return upper >= PIN_TAIL_MIN_FRACTION * rng and lower <= (1-PIN_TAIL_MIN_FRACTION)*rng

def rejection_on_ma(c, ma_value, direction: str, atr_val: Optional[float]):
    close = float(c["c"]); o = float(c["o"]); h=float(c["h"]); l=float(c["l"])
    if ma_value is None: return False
    # literal touch?
    touched = (l <= ma_value <= h) or (min(o,close) <= ma_value <= max(o,close))
    # or near-miss within tolerance of ATR
    near = False
    if atr_val is not None:
        near = abs(min(abs(h-ma_value), abs(l-ma_value), abs(close-ma_value), abs(o-ma_value))) <= REJECT_ATR_TOL * atr_val
    if direction == "above":
        return (close >= ma_value) and (touched or near)
    else:
        return (close <= ma_value) and (touched or near)

def nearest_sr(candles, direction: str) -> Optional[float]:
    pivH = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
    pivL = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
    last_close = float(candles[-1]["c"])
    if direction == "buy":
        above = [p for _,p in pivH if p > last_close]
        return min(above) if above else None
    else:
        below = [p for _,p in pivL if p < last_close]
        return max(below) if below else None

# =========================
# STRATEGY
# =========================
def analyze_symbol_tf(symbol: str, tf: int) -> Tuple[str, Optional[str]]:
    """
    Returns (signal, message)
    signal in {"BUY","SELL","NONE"}
    message (if BUY/SELL) is a human-readable explanation for Telegram.
    """
    candles = get_candles(symbol, tf, HISTORY_COUNT)
    if len(candles) < 60:
        return "NONE", None

    closes = [float(c["c"]) for c in candles]
    highs  = [float(c["h"]) for c in candles]
    lows   = [float(c["l"]) for c in candles]
    typ    = [typical_price(c) for c in candles]

    # MA1: SMMA(9) on typical; MA2: SMMA(19) on close; MA3: SMA(25) on MA2 values
    ma1 = smma(typ, 9)
    ma2 = smma(closes, 19)
    ma2_clean = ma2[:]
    fv = next((i for i,v in enumerate(ma2_clean) if v is not None), None)
    if fv is None:
        return "NONE", None
    for i in range(fv):
        ma2_clean[i] = ma2_clean[fv]
    ma3 = sma(ma2_clean, 25)

    # Range filter: MA1 slope vs ATR
    atr_vals = atr(candles, ATR_PERIOD)
    last_i = len(candles) - 1
    # slope across MA1 over SLOPE_LOOKBACK bars
    idxs = [i for i in range(last_i - SLOPE_LOOKBACK, last_i + 1) if i >= 0 and ma1[i] is not None]
    if len(idxs) < 2: return "NONE", None
    slope = ma1[idxs[-1]] - ma1[idxs[0]]
    atr_now = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else None
    if (atr_now is None) or (abs(slope) < TREND_MIN_SLOPE_ATR_MULT * atr_now):
        return "NONE", None  # avoid ranging markets

    # Trend definition per your rules
    last_ma1 = ma1[-1]; last_ma2 = ma2[-1]; last_ma3 = ma3[-1]
    last_close = closes[-1]

    # "MA1 nearest to price"
    d1 = abs(last_close - last_ma1) if last_ma1 is not None else 1e9
    d2 = abs(last_close - last_ma2) if last_ma2 is not None else 1e9
    d3 = abs(last_close - last_ma3) if last_ma3 is not None else 1e9
    ma1_nearest = d1 <= d2 and d1 <= d3

    uptrend   = ma1_nearest and last_ma2 is not None and last_ma3 is not None and (last_ma2 < last_ma1) and (last_ma3 < last_ma1) and (last_ma2 >= last_ma3*0.95)
    downtrend = ma1_nearest and last_ma2 is not None and last_ma3 is not None and (last_ma2 > last_ma1) and (last_ma3 > last_ma1) and (last_ma2 <= last_ma3*1.05)

    # Use last two completed candles: [-2] rejection, [-1] confirmation
    rej = candles[-2]; conf = candles[-1]

    # rejection must be doji/pin/inverted pin and close relative to MA side; allow MA1 or MA2 as retest line
    def check_buy_on(ma_val, name):
        atr_for_rej = atr_vals[-2] if len(atr_vals) >= 2 else None
        is_rej = rejection_on_ma(rej, ma_val, "above", atr_for_rej) and (is_doji(rej) or is_pin_bar(rej, True) or is_pin_bar(rej, False))
        is_conf= (is_bullish(conf) and float(conf["c"]) >= ma_val)
        return is_rej and is_conf, name

    def check_sell_on(ma_val, name):
        atr_for_rej = atr_vals[-2] if len(atr_vals) >= 2 else None
        is_rej = rejection_on_ma(rej, ma_val, "below", atr_for_rej) and (is_doji(rej) or is_pin_bar(rej, True) or is_pin_bar(rej, False))
        is_conf= (is_bearish(conf) and float(conf["c"]) <= ma_val)
        return is_rej and is_conf, name

    signal = "NONE"; used_ma = None

    if uptrend:
        ok, nm = check_buy_on(ma1[-2], "MA1") if ma1[-2] else (False, None)
        if ok: signal, used_ma = "BUY", nm
        elif ma2[-2]:
            ok2, nm2 = check_buy_on(ma2[-2], "MA2")
            if ok2: signal, used_ma = "BUY", nm2

    if signal == "NONE" and downtrend:
        ok, nm = check_sell_on(ma1[-2], "MA1") if ma1[-2] else (False, None)
        if ok: signal, used_ma = "SELL", nm
        elif ma2[-2]:
            ok2, nm2 = check_sell_on(ma2[-2], "MA2")
            if ok2: signal, used_ma = "SELL", nm2

    if signal == "NONE":
        return "NONE", None

    # Pattern alignment filter
    asc, desc = detect_triangles(candles)
    dt, db = detect_double_top_bottom(candles)
    hs, ihs = detect_head_shoulders(candles)

    pattern_ok = True
    patterns = []
    if signal == "BUY":
        if asc: patterns.append("Ascending Triangle")
        if db:  patterns.append("Double Bottom")
        if ihs: patterns.append("Inverse H&S")
        if (dt is not None) or desc or hs:
            pattern_ok = False
    else:
        if desc: patterns.append("Descending Triangle")
        if dt:  patterns.append("Double Top")
        if hs:  patterns.append("Head & Shoulders")
        if (db is not None) or asc or ihs:
            pattern_ok = False

    if not pattern_ok:
        return "NONE", None

    # Break & Retest across last few bars relative to MA1/MA2
    def crossed_back(series_ma):
        lookback = 6
        for i in range(len(candles)-lookback-1, len(candles)-2):
            if i < 1 or series_ma[i] is None or series_ma[i-1] is None: 
                continue
            pc = float(candles[i]["c"]); prev = float(candles[i-1]["c"])
            if signal == "BUY" and (prev < series_ma[i-1] and pc > series_ma[i]):
                return True
            if signal == "SELL" and (prev > series_ma[i-1] and pc < series_ma[i]):
                return True
        return False

    if not (crossed_back(ma1) or crossed_back(ma2)):
        return "NONE", None

    # TP zone
    tp = nearest_sr(candles, "buy" if signal=="BUY" else "sell")
    tf_lbl = "6m" if tf == 360 else "10m"
    price_now = closes[-1]
    patt_txt = (", ".join(patterns)) if patterns else "MA confluence"
    used_ma_print = used_ma or "MA"
    tp_txt = f"{tp:.2f}" if tp else "—"

    msg = (
        f"📊 <b>{symbol}</b> | TF <b>{tf_lbl}</b>\n"
        f"Signal: <b>{signal}</b>\n"
        f"Reason: {patt_txt} + rejection on <b>{used_ma_print}</b> with confirmation\n"
        f"Price: {price_now:.2f}\n"
        f"TP zone: {tp_txt}"
    )
    return signal, msg

# =========================
# MTF ORCHESTRATION
# =========================
def analyze_symbol_both_tfs(symbol: str) -> Optional[str]:
    # per-TF signals
    results: Dict[int, Tuple[str, Optional[str]]] = {}
    for tf in TIMEFRAMES:
        try:
            sig, msg = analyze_symbol_tf(symbol, tf)
            results[tf] = (sig, msg)
        except Exception as e:
            print(f"Error {symbol}@{tf}: {e}")
            results[tf] = ("NONE", None)

    sig6, msg6 = results[360]
    sig10, msg10 = results[600]

    # STRONG agreement
    if sig6 == sig10 and sig6 in ("BUY","SELL"):
        label = "🔥 STRONG BUY" if sig6 == "BUY" else "🔥 STRONG SELL"
        # prefer to include both TF details below
        parts = [f"{label} — <b>{symbol}</b> (6m + 10m agree)"]
        if msg6:  parts.append("— 6m —\n" + msg6)
        if msg10: parts.append("— 10m —\n" + msg10)
        return "\n\n".join(parts)

    # No agreement → still send individual valid TF messages
    pieces = []
    if sig6 in ("BUY","SELL") and msg6: pieces.append(msg6)
    if sig10 in ("BUY","SELL") and msg10: pieces.append(msg10)
    if pieces:
        return "\n\n".join(pieces)

    return None

# =========================
# MAIN
# =========================
def main():
    all_msgs = []
    for sym in ASSETS:
        txt = analyze_symbol_both_tfs(sym)
        if txt:
            all_msgs.append(txt)

    # Send only if we have valid signals (as per your rule)
    if all_msgs:
        send_telegram("✅ <b>Signals</b>\n\n" + "\n\n" .join(all_msgs))
    else:
        print("No valid signals this run.")

if __name__ == "__main__":
    main()

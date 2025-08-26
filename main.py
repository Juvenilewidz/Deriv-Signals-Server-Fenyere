import os, json, statistics
from typing import List, Optional, Tuple, Dict
import websocket

from bot import send_telegram

# =========================
# ENV & CONSTANTS
# =========================
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_APP_ID    = os.getenv("DERIV_APP_ID", "1089")
if not DERIV_API_TOKEN:
    print("Missing DERIV_API_TOKEN")
    raise SystemExit(0)

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Exact assets & TFs
ASSETS = ["R_50", "R_75", "R_10", "1HZ75V", "1HZ100V", "1HZ150V"]
TF_6M, TF_10M = 360, 600
TIMEFRAMES = [TF_6M, TF_10M]

HISTORY_COUNT = 400

# Range filter
ATR_PERIOD = 14
SLOPE_LOOKBACK = 10
TREND_MIN_SLOPE_ATR_MULT = 0.25  # avoid ranging

# Rejection tolerances
DOJI_BODY_TO_RANGE_MAX = 0.20
BODY_MAX_OF_RANGE      = 0.35
PIN_TAIL_MIN_FRACTION  = 0.66
REJECT_ATR_TOL         = 0.05   # near-miss distance to MA allowed (fraction of ATR)

# =========================
# UTIL / WS
# =========================
def ws_call(payload: dict) -> dict:
    ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
    try:
        ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
        ws.recv()  # auth response
        ws.send(json.dumps(payload))
        data = json.loads(ws.recv())
        return data
    finally:
        ws.close()

def get_candles(symbol: str, granularity: int, count: int) -> List[dict]:
    resp = ws_call({"candles": symbol, "granularity": granularity, "count": count})
    arr = resp.get("candles", [])
    return sorted(arr, key=lambda x: x["epoch"])

def typical_price(c):  # HLC/3
    return (float(c["h"]) + float(c["l"]) + float(c["c"])) / 3.0

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out, q, s = [], [], 0.0
    for v in values:
        q.append(v); s += v
        if len(q) > period:
            s -= q.pop(0)
        out.append(s/period if len(q)==period else None)
    return out

def smma(values: List[float], period: int) -> List[Optional[float]]:
    out = [None]*len(values)
    if len(values) < period: return out
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
    if len(candles) < period + 1: return [None]*len(candles)
    trs = [None]
    for i in range(1, len(candles)):
        prev_c = float(candles[i-1]["c"])
        h = float(candles[i]["h"]); l = float(candles[i]["l"])
        trs.append(true_range(prev_c, h, l))
    tr_vals = trs[1:]
    return smma(tr_vals, period)

# =========================
# CANDLE / PATTERN CHECKS
# =========================
def is_bullish(c): return float(c["c"]) > float(c["o"])
def is_bearish(c): return float(c["c"]) < float(c["o"])

def is_doji(c):
    o = float(c["o"]); cl = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    return abs(cl-o) <= DOJI_BODY_TO_RANGE_MAX * rng

def is_pin_bar(c, bullish=True):
    o = float(c["o"]); cl = float(c["c"]); h=float(c["h"]); l=float(c["l"])
    rng = max(h-l, 1e-9)
    body = abs(cl-o)
    if body > BODY_MAX_OF_RANGE * rng: return False
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    if bullish:
        return lower >= PIN_TAIL_MIN_FRACTION * rng and upper <= (1-PIN_TAIL_MIN_FRACTION)*rng
    else:
        return upper >= PIN_TAIL_MIN_FRACTION * rng and lower <= (1-PIN_TAIL_MIN_FRACTION)*rng

def pivot_high(candles, i, L=2, R=2):
    if i < L or i+R >= len(candles): return False
    val = float(candles[i]["h"])
    for j in range(i-L, i+R+1):
        if j==i: continue
        if float(candles[j]["h"]) >= val: return False
    return True

def pivot_low(candles, i, L=2, R=2):
    if i < L or i+R >= len(candles): return False
    val = float(candles[i]["l"])
    for j in range(i-L, i+R+1):
        if j==i: continue
        if float(candles[j]["l"]) <= val: return False
    return True

def detect_double_top_bottom(candles):
    pivH = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
    pivL = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
    dt = db = None
    if len(pivH) >= 2:
        i1,p1 = pivH[-2]; i2,p2 = pivH[-1]
        if abs(p1-p2)/max((p1+p2)/2.0,1e-9) <= 0.0015 and (i2-i1) >= 5:
            dt = (i1,i2,(p1+p2)/2.0)
    if len(pivL) >= 2:
        i1,p1 = pivL[-2]; i2,p2 = pivL[-1]
        if abs(p1-p2)/max((p1+p2)/2.0,1e-9) <= 0.0015 and (i2-i1) >= 5:
            db = (i1,i2,(p1+p2)/2.0)
    return dt, db

def detect_triangles(candles):
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

def detect_head_shoulders(candles):
    n = len(candles)
    if n < 50: return (False, False)
    highs = [float(c["h"]) for c in candles]
    lows  = [float(c["l"]) for c in candles]
    pivH = [i for i in range(2, n-2) if pivot_high(candles, i)]
    pivL = [i for i in range(2, n-2) if pivot_low(candles, i)]
    reg = inv = False
    if len(pivH) >= 3:
        a,b,c = pivH[-3], pivH[-2], pivH[-1]
        if highs[b] > highs[a] and highs[b] > highs[c] and abs(highs[a]-highs[c]) / max(highs[b],1e-9) < 0.02:
            reg = True
    if len(pivL) >= 3:
        a,b,c = pivL[-3], pivL[-2], pivL[-1]
        if lows[b] < lows[a] and lows[b] < lows[c] and abs(lows[a]-lows[c]) / max(abs(lows[b]),1e-9) < 0.02:
            inv = True
    return reg, inv

# =========================
# STRATEGY CORE
# =========================
def analyze_symbol_tf(symbol: str, tf: int) -> Tuple[str, Optional[str]]:
    """
    Returns (signal, message)
    signal in {"BUY","SELL","NONE"}
    message is the formatted explanation for Telegram if BUY/SELL
    """
    candles = get_candles(symbol, tf, HISTORY_COUNT)
    if len(candles) < 60: return "NONE", None

    closes = [float(c["c"]) for c in candles]
    highs  = [float(c["h"]) for c in candles]
    lows   = [float(c["l"]) for c in candles]
    typ    = [typical_price(c) for c in candles]

    # MAs: MA1 = SMMA(9) on typical, MA2 = SMMA(19) on close, MA3 = SMA(25) on MA2 (previous indicator)
    ma1 = smma(typ, 9)
    ma2 = smma(closes, 19)
    ma2_filled = ma2[:]
    fv = next((i for i,v in enumerate(ma2_filled) if v is not None), None)
    if fv is None: return "NONE", None
    for i in range(fv):  # forward-fill to allow SMA(25) start
        ma2_filled[i] = ma2_filled[fv]
    ma3 = sma(ma2_filled, 25)

    atr_vals = atr(candles, ATR_PERIOD)
    # Range filter: compare MA1 slope to ATR
    last_i = len(candles) - 1
    idxs = [i for i in range(last_i - SLOPE_LOOKBACK, last_i + 1) if i >= 0 and ma1[i] is not None]
    if len(idxs) < 2: return "NONE", None
    slope = ma1[idxs[-1]] - ma1[idxs[0]]
    atr_now = atr_vals[-1] if len(atr_vals)>0 else None
    if (atr_now is None) or (abs(slope) < TREND_MIN_SLOPE_ATR_MULT * atr_now):
        return "NONE", None  # avoid ranging markets

    last_close = closes[-1]
    last_ma1, last_ma2, last_ma3 = ma1[-1], ma2[-1], ma3[-1]

    # "MA1 nearest to price"
    def dist(a, b): return abs(a-b) if (a is not None and b is not None) else 1e9
    ma1_nearest = dist(last_close, last_ma1) <= min(dist(last_close, last_ma2), dist(last_close, last_ma3))

    # Uptrend/Downtrend stacking (as described)
    uptrend   = ma1_nearest and (last_ma2 is not None and last_ma3 is not None) and (last_ma1 > last_ma2 > last_ma3)
    downtrend = ma1_nearest and (last_ma2 is not None and last_ma3 is not None) and (last_ma1 < last_ma2 < last_ma3)

    # Work with last two completed bars: [-2] rejection candlestick, [-1] confirmation
    rej = candles[-2]; conf = candles[-1]

    def reject_towards(ma_val, side: str) -> bool:
        if ma_val is None: return False
        o=float(rej["o"]); c=float(rej["c"]); h=float(rej["h"]); l=float(rej["l"])
        touched = (l <= ma_val <= h) or (min(o,c) <= ma_val <= max(o,c))
        near = False
        if atr_vals[-2] is not None:
            near = min(abs(h-ma_val), abs(l-ma_val), abs(c-ma_val), abs(o-ma_val)) <= REJECT_ATR_TOL * atr_vals[-2]
        # candlestick type
        is_rej_candle = is_doji(rej) or is_pin_bar(rej, True) or is_pin_bar(rej, False)
        if side == "buy":
            # rejection must close ABOVE the MA being tested
            return is_rej_candle and (c >= ma_val) and (touched or near)
        else:
            # rejection must close BELOW the MA being tested
            return is_rej_candle and (c <= ma_val) and (touched or near)

    # Confirmation candle must close on same side of the MA
    def confirm_on(ma_val, direction: str) -> bool:
        if ma_val is None: return False
        cc = float(conf["c"]); oo=float(conf["o"])
        if direction == "buy":
            return (cc >= ma_val) and (cc > oo)  # bullish
        else:
            return (cc <= ma_val) and (cc < oo)  # bearish

    # Break & Retest dynamic behavior around MA1/MA2 within last few bars
    def crossed_back(series_ma, direction: str) -> bool:
        look = 6
        for i in range(len(candles)-look-1, len(candles)-2):
            if i < 1: continue
            m0 = series_ma[i-1]; m1 = series_ma[i]
            if m0 is None or m1 is None: continue
            pc_prev = float(candles[i-1]["c"]); pc = float(candles[i]["c"])
            if direction == "buy"  and (pc_prev < m0 and pc > m1): return True
            if direction == "sell" and (pc_prev > m0 and pc < m1): return True
        return False

    signal = "NONE"; used_ma = None

    if uptrend:
        # MA1 preferred, then MA2
        if reject_towards(ma1[-2], "buy") and confirm_on(ma1[-2], "buy") and crossed_back(ma1, "buy"):
            signal, used_ma = "BUY", "MA1"
        elif reject_towards(ma2[-2], "buy") and confirm_on(ma2[-2], "buy") and crossed_back(ma2, "buy"):
            signal, used_ma = "BUY", "MA2"

    if signal == "NONE" and downtrend:
        if reject_towards(ma1[-2], "sell") and confirm_on(ma1[-2], "sell") and crossed_back(ma1, "sell"):
            signal, used_ma = "SELL", "MA1"
        elif reject_towards(ma2[-2], "sell") and confirm_on(ma2[-2], "sell") and crossed_back(ma2, "sell"):
            signal, used_ma = "SELL", "MA2"

    if signal == "NONE":
        return "NONE", None

    # Chart pattern alignment (heuristics)
    asc, desc = detect_triangles(candles)
    dt, db   = detect_double_top_bottom(candles)
    hs, ihs  = detect_head_shoulders(candles)

    pattern_ok = True
    patterns = []
    if signal == "BUY":
        if asc: patterns.append("Ascending Triangle")
        if db:  patterns.append("Double Bottom")
        if ihs: patterns.append("Inverse H&S")
        # invalidate if bearish structures appear
        if (dt is not None) or desc or hs:
            pattern_ok = False
    else:
        if desc: patterns.append("Descending Triangle")
        if dt:   patterns.append("Double Top")
        if hs:   patterns.append("Head & Shoulders")
        # invalidate if bullish structures appear
        if (db is not None) or asc or ihs:
            pattern_ok = False

    if not pattern_ok:
        return "NONE", None

    # TP zone via nearest swing S/R
    def nearest_sr(direction: str) -> Optional[float]:
        ph = [(i, float(c["h"])) for i,c in enumerate(candles) if pivot_high(candles, i)]
        pl = [(i, float(c["l"])) for i,c in enumerate(candles) if pivot_low(candles, i)]
        last_c = float(candles[-1]["c"])
        if direction == "buy":
            above = [p for _,p in ph if p > last_c]
            return min(above) if above else None
        else:
            below = [p for _,p in pl if p < last_c]
            return max(below) if below else None

    tp = nearest_sr("buy" if signal=="BUY" else "sell")
    tf_lbl = "6m" if tf == TF_6M else "10m"
    price_now = float(candles[-1]["c"])
    patt_txt = (", ".join(patterns)) if patterns else "MA confluence"
    tp_txt = f"{tp:.2f}" if tp else "—"

    msg = (
        f"📊 <b>{symbol}</b> | TF <b>{tf_lbl}</b>\n"
        f"Signal: <b>{signal}</b>\n"
        f"Reason: {patt_txt} + rejection on <b>{used_ma}</b> (with confirmation)\n"
        f"Price: {price_now:.2f}\n"
        f"TP zone: {tp_txt}"
    )
    return signal, msg

# =========================
# MTF ORCHESTRATOR
# =========================
def analyze_symbol_both_tfs(symbol: str) -> Optional[str]:
    r6  = analyze_symbol_tf(symbol, TF_6M)
    r10 = analyze_symbol_tf(symbol, TF_10M)
    sig6, msg6 = r6
    sig10, msg10 = r10

    # strong agreement
    if sig6 == sig10 and sig6 in ("BUY","SELL"):
        label = "🔥 STRONG BUY" if sig6 == "BUY" else "🔥 STRONG SELL"
        parts = [f"{label} — <b>{symbol}</b> (6m + 10m agree)"]
        if msg6:  parts.append("— 6m —\n" + msg6)
        if msg10: parts.append("— 10m —\n" + msg10)
        return "\n\n".join(parts)

    # no agreement → still report valid TF signals
    pieces = []
    if sig6 in ("BUY","SELL") and msg6:   pieces.append(msg6)
    if sig10 in ("BUY","SELL") and msg10: pieces.append(msg10)
    return "\n\n".join(pieces) if pieces else None

# =========================
# MAIN
# =========================
def main():
    all_msgs = []
    for sym in ASSETS:
        try:
            txt = analyze_symbol_both_tfs(sym)
            if txt:
                all_msgs.append(txt)
        except Exception as e:
            print(f"Error analyzing {sym}: {e}")

    if all_msgs:  # only send when signal(s) valid
        send_telegram("✅ <b>Signals</b>\n\n" + "\n\n".join(all_msgs))
    else:
        print("No valid signals this run.")

if __name__ == "__main__":
    main()

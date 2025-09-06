#!/usr/bin/env python3
"""
main_1s.py — text-first alerts for 1s synthetic indices (5m TF only by default)
Same MA & ATR signal engine as main.py but optimized for 1s indices (less chart emphasis).
"""

import os, json, time, math, tempfile, traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import websocket
import numpy as np

# Use your bot helper if present
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    try:
        import requests
    except Exception:
        requests = None
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM]", text); return False, "no-token"
    def send_telegram_photo(token, chat_id, caption, photo_path):
        print("[TELEGRAM PHOTO]", caption, photo_path); return False, "no-token"

# -------------------------
# Config
# -------------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

ASSETS = [s.strip() for s in os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",") if s.strip()]
TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES", "300").split(",")]  # default 5m
CANDLES_N = int(os.getenv("CANDLES_N", "120"))

NEAR_ATR_MULT = float(os.getenv("NEAR_ATR_MULT", "0.6"))
PCT_FALLBACK  = float(os.getenv("PCT_FALLBACK", "0.002"))

TMPDIR = tempfile.gettempdir()
ALERT_PERSIST_PATH = os.getenv("ALERT_PERSIST_PATH", os.path.join(TMPDIR, "fenyere_alerts_1s.json"))

DISABLE_HEARTBEAT = os.getenv("DISABLE_HEARTBEAT", "1") in ("1", "true", "True")
SLEEP_WINDOW = os.getenv("SLEEP_WINDOW", "")
TZ_OFFSET = os.getenv("TZ_OFFSET", "+02:00")

MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# Reuse MA & ATR implementations from main.py (SMMA/SMA)
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series); out = [None]*n
    if n < period or period <= 0:
        return out
    seed = sum(series[:period]) / period
    out[period-1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev*(period-1) + float(series[i])) / period
        out[i] = prev
    return out

def sma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series); out = [None]*n
    if period <= 0 or n < period:
        return out
    for i in range(period-1, n):
        out[i] = sum(series[i-period+1:i+1]) / period
    return out

def compute_mas_for_chart(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes = [float(c["close"]) for c in candles]
    highs  = [float(c["high"])  for c in candles]
    lows   = [float(c["low"])   for c in candles]
    hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    ma1 = smma(hlc3, 9)
    ma2 = smma(closes, 19)
    ma2_vals = [v for v in ma2 if v is not None]
    if len(ma2_vals) >= 25:
        ma3_raw = sma(ma2_vals, 25)
    else:
        ma3_raw = [None] * len(ma2_vals)
    ma3 = []
    j = 0
    for v in ma2:
        if v is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

def compute_atr(candles: List[Dict], period: int = 14) -> float:
    if not candles:
        return 0.0
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows  = np.array([float(c["low"]) for c in candles], dtype=float)
    closes= np.array([float(c["close"]) for c in candles], dtype=float)
    trs = []
    for i in range(len(candles)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    if len(trs) >= period:
        return float(np.mean(trs[-period:]))
    return float(np.mean(trs))

def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    tried = [count, max(100, count//2), 80, 50, 25]
    for ccount in tried:
        for attempt in range(2):
            ws = None
            try:
                ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                try: _ = ws.recv()
                except: pass
                req = {"ticks_history": symbol, "style": "candles", "granularity": granularity,
                       "count": ccount, "end": "latest", "subscribe": 0}
                ws.send(json.dumps(req))
                raw = ws.recv()
                resp = json.loads(raw)
                if isinstance(resp, dict) and resp.get("candles"):
                    parsed = []
                    for cc in resp["candles"]:
                        try:
                            parsed.append({"epoch": int(cc["epoch"]), "open": float(cc["open"]),
                                           "high": float(cc["high"]), "low": float(cc["low"]), "close": float(cc["close"])})
                        except Exception:
                            continue
                    parsed.sort(key=lambda x: x["epoch"])
                    return parsed
            except Exception as e:
                log("fetch err", symbol, granularity, ccount, "attempt", attempt+1, e)
            finally:
                try:
                    if ws: ws.close()
                except: pass
            time.sleep(0.6)
    log("fetch_candles final fail", symbol, granularity)
    return []

# Candle families
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o); rng = max(1e-12, h - l)
    upper = h - max(o, c); lower = min(o, c) - l
    is_doji = body <= 0.40 * rng
    pin_low = (lower >= 0.3 * rng) and (lower > upper)
    pin_high = (upper >= 0.3 * rng) and (upper > lower)
    engulf_bull = engulf_bear = False
    if prev:
        po = float(prev["open"]); pc = float(prev["close"])
        if pc < po and c > o and o <= pc and c >= po:
            engulf_bull = True
        if pc > po and c < o and o >= pc and c <= po:
            engulf_bear = True
    return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": rng,
            "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
            "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

def is_near_ma(price_extreme: float, ma_value: Optional[float], atr: float) -> Tuple[bool, float]:
    if ma_value is None or atr is None:
        return False, float('inf')
    tol = max(atr * NEAR_ATR_MULT, abs(price_extreme) * PCT_FALLBACK)
    dist = abs(price_extreme - ma_value)
    return (dist <= tol), float(dist/atr if atr>0 else float('inf'))

def detect_signal_from_candles(candles: List[Dict], tf: int, symbol: str) -> Optional[Dict]:
    if not candles or len(candles) < 12:
        return None
    ma1, ma2, ma3 = compute_mas_for_chart(candles)
    atr = compute_atr(candles)
    idx = len(candles) - 1
    prev = candles[idx-1] if idx-1 >= 0 else None
    candle = candles[idx]
    bits = candle_bits(candle, prev)
    family = None
    if bits["is_doji"]:
        family = "DOJI"
    elif bits["pin_low"] or bits["pin_high"]:
        family = "PIN"
    elif bits["engulf_bull"] or bits["engulf_bear"]:
        family = "ENGULF"
    elif bits["body"] <= 0.15 * bits["range"]:
        family = "TINY"
    if family is None:
        return None

    low = float(candle["low"]); high = float(candle["high"]); close = float(candle["close"])
    near_low_ma1, d1 = is_near_ma(low, ma1[idx] if idx<len(ma1) else None, atr)
    near_low_ma2, d2 = is_near_ma(low, ma2[idx] if idx<len(ma2) else None, atr)
    near_high_ma1, d3 = is_near_ma(high, ma1[idx] if idx<len(ma1) else None, atr)
    near_high_ma2, d4 = is_near_ma(high, ma2[idx] if idx<len(ma2) else None, atr)

    reasons=[]; direction=None
    if near_low_ma1 or near_low_ma2:
        reasons.append(f"Rejection {family} near MA1/MA2 (low) distATR={min(d1,d2):.2f}"); direction="BUY"
    if near_high_ma1 or near_high_ma2:
        reasons.append(f"Rejection {family} near MA1/MA2 (high) distATR={min(d3,d4):.2f}"); direction="SELL"

    # check MA3 cross + retest
    def recent_ma3_cross(candles, ma3, lookback=8):
        n=len(candles)
        for i in range(max(1,n-lookback), n):
            if i-1<0: continue
            prevc=float(candles[i-1]["close"]); curc=float(candles[i]["close"])
            ma_prev = ma3[i-1] if i-1 < len(ma3) else None
            ma_cur  = ma3[i]   if i < len(ma3) else None
            if ma_prev is None or ma_cur is None: continue
            if prevc < ma_prev and curc > ma_cur: return (i, "BUY")
            if prevc > ma_prev and curc < ma_cur: return (i, "SELL")
        return None
    cross = recent_ma3_cross(candles, ma3, lookback=8)
    if cross:
        cross_idx, cross_dir = cross
        if cross_idx < idx:
            if direction is None:
                # if retest matched
                if (cross_dir == "BUY" and (near_low_ma1 or near_low_ma2)):
                    reasons.append("MA3-break+retest (BUY)"); direction="BUY"
                if (cross_dir == "SELL" and (near_high_ma1 or near_high_ma2)):
                    reasons.append("MA3-break+retest (SELL)"); direction="SELL"

    if not reasons:
        return None

    epoch = int(candle["epoch"])
    return {"symbol": symbol, "tf": tf, "direction": direction, "reason": " | ".join(reasons),
            "price": close, "epoch": epoch}

# Persistence/dedupe
def load_alerts(path=ALERT_PERSIST_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_alerts(d, path=ALERT_PERSIST_PATH):
    try:
        with open(path, "w") as f:
            json.dump(d, f)
    except Exception:
        pass

def already_sent(symbol: str, tf: int, epoch: int, side: str) -> bool:
    key = f"{symbol}|{tf}"
    d = load_alerts()
    rec = d.get(key)
    if not rec:
        return False
    return rec.get("epoch") == int(epoch) and rec.get("side") == side

def mark_sent(symbol: str, tf: int, epoch: int, side: str):
    key = f"{symbol}|{tf}"
    d = load_alerts()
    d[key] = {"epoch": int(epoch), "side": side, "sent_at": int(time.time())}
    save_alerts(d)

# Sleep window
def in_sleep_window(sleep_window: str, tz_offset: str = "+00:00") -> bool:
    if not sleep_window:
        return False
    try:
        start_s, end_s = sleep_window.split("-")
        sign = 1 if tz_offset[0] != "-" else -1
        tz_h = int(tz_offset[1:3]) * sign
        tz_m = int(tz_offset[4:6]) * sign if len(tz_offset) >= 6 else 0
        offset = timedelta(hours=tz_h, minutes=tz_m)
        now_utc = datetime.utcnow()
        local_now = now_utc + offset
        sh, sm = [int(x) for x in start_s.split(":")]
        eh, em = [int(x) for x in end_s.split(":")]
        start_dt = local_now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end_dt = local_now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start_dt < end_dt:
            return start_dt <= local_now <= end_dt
        else:
            return local_now >= start_dt or local_now <= end_dt
    except Exception:
        return False

# Runner
def run_once():
    if in_sleep_window(SLEEP_WINDOW, TZ_OFFSET):
        log("Within sleep window -> skipping run"); return

    if not DISABLE_HEARTBEAT:
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"✅ main_1s started - tracking {len(ASSETS)} symbols")
        except Exception:
            log("heartbeat failed")

    sent = 0
    for symbol in ASSETS:
        if sent >= MAX_SIGNALS_PER_RUN:
            break
        for tf in TIMEFRAMES:
            try:
                candles = fetch_candles(symbol, tf, count=CANDLES_N)
                if not candles:
                    continue
                sig = detect_signal_from_candles(candles, tf, symbol)
                if not sig:
                    continue
                epoch = int(sig["epoch"])
                side = sig["direction"]
                if already_sent(symbol, tf, epoch, side):
                    log("already sent", symbol, tf, epoch, side); continue
                msg = f"🔔 {symbol} | {tf//60}m | {side}\nReason: {sig['reason']}\nPrice: {sig['price']}"
                try:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
                except Exception as e:
                    log("send err", e)
                mark_sent(symbol, tf, epoch, side)
                sent += 1
            except Exception as e:
                log("error", symbol, e)
                if DEBUG: traceback.print_exc()
    log("Run finished. sent:", sent)

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Fatal", e)
        if DEBUG: traceback.print_exc()
        try: send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"1s bot crash: {e}")
        except: pass
        raise

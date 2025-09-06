#!/usr/bin/env python3
# main_1s.py - text-only paste-and-cruise for 1s synthetic indices.
# Uses the same signal rules (rejection families inclusive), sends text alerts only.
# Avoids building charts to prevent previous "no candles" crashes.

import os, json, time, math, tempfile, traceback
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import websocket, numpy as np

# Try import bot helpers, fallback to prints
try:
    from bot import send_telegram_message, send_single_timeframe_signal
except Exception:
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM TEXT]", text); return True, "fallback"
    def send_single_timeframe_signal(symbol, tf, direction, reason, chart_path=None):
        print("[SIG]", symbol, tf, direction, reason, chart_path); return True

# -------------------------
# Config
# -------------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
# for 1s bot we'll recommend 5m only for signals (you insisted)
TIMEFRAMES = [300]  # 5m only

CANDLES_N = int(os.getenv("CANDLES_N", "100"))
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))
DIST_TOL = float(os.getenv("DIST_TOL", "1.0"))

TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")

last_sent_cache: Dict[str, Dict] = {}

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

def reason_skip(symbol: str, tf: int, reason: str):
    log(f"[SKIP] {symbol} @{tf}s -> {reason}")

def load_cache():
    global last_sent_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_cache = json.load(f)
        else:
            last_sent_cache = {}
    except Exception:
        last_sent_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
    except Exception:
        pass

def can_send(symbol: str, direction: str, tf: int) -> bool:
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec and rec.get("direction") == direction and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
        return False
    return True

def mark_sent(symbol: str, direction: str, tf: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time())}
    save_cache()

# -------------------------
# MA & pattern helpers (same as main)
# -------------------------
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if n == 0 or period <= 0:
        return []
    if n < period:
        return [None] * n
    seed = sum(series[:period]) / period
    out = [None] * (period - 1)
    out.append(seed)
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + float(series[i])) / period
        out.append(prev)
    return out

def sma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if period <= 0 or n < period:
        return [None] * n
    out = [None] * (period - 1)
    for i in range(period - 1, n):
        out.append(sum(series[i - period + 1:i + 1]) / period)
    return out

def compute_mas_for_chart(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    hlc3    = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
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

def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o)
    r = max(1e-12, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l
    is_doji = body <= 0.15 * r
    pin_low = (lower >= 0.3 * r) and (lower > upper)
    pin_high = (upper >= 0.3 * r) and (upper > lower)
    engulf_bull = engulf_bear = False
    if prev:
        po = float(prev["open"]); pc = float(prev["close"])
        if pc < po and c > o and o <= pc and c >= po:
            engulf_bull = True
        if pc > po and c < o and o >= pc and c <= po:
            engulf_bear = True
    return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
            "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
            "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

def compute_atr_estimate(candles: List[Dict], period: int = 14) -> float:
    rngs = [c["high"] - c["low"] for c in candles if c["high"] >= c["low"]]
    if not rngs:
        return 1.0
    return float(np.mean(rngs[-period:])) if len(rngs) >= period else float(np.mean(rngs))

def is_rejection_candle(candle: Dict, prev: Optional[Dict], atr: float, direction: str) -> bool:
    bits = candle_bits(candle, prev)
    if atr > 0 and bits["body"] <= 0.25 * atr:
        if direction == "BUY":
            return candle["close"] >= candle["open"]
        else:
            return candle["close"] <= candle["open"]
    if bits["is_doji"]:
        if direction == "BUY":
            return candle["close"] >= candle["open"]
        else:
            return candle["close"] <= candle["open"]
    if direction == "BUY" and bits["pin_low"]:
        return True
    if direction == "SELL" and bits["pin_high"]:
        return True
    if direction == "BUY" and bits["engulf_bull"]:
        return True
    if direction == "SELL" and bits["engulf_bear"]:
        return True
    return False

# -------------------------
# Fetch candles (snapshot + ticks fallback), same as main but smaller count
# -------------------------
def aggregate_ticks_to_candles(ticks, granularity, count):
    buckets = {}
    for t in ticks:
        ts = int(t.get("epoch") or t.get("time") or 0)
        price = float(t.get("quote") or t.get("price") or t.get("tick"))
        bucket = (ts // granularity) * granularity
        b = buckets.get(bucket)
        if not b:
            buckets[bucket] = {"epoch": bucket, "open": price, "high": price, "low": price, "close": price}
        else:
            b["high"] = max(b["high"], price)
            b["low"] = min(b["low"], price)
            b["close"] = price
    out = sorted(buckets.values(), key=lambda x: x["epoch"])
    if len(out) > count:
        out = out[-count:]
    return out

def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES+1):
        ws = None
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            try:
                _ = ws.recv()
            except Exception:
                pass
            req = {"ticks_history": symbol, "style": "candles", "granularity": granularity, "count": count, "end": "latest", "subscribe": 0}
            ws.send(json.dumps(req))
            raw = ws.recv()
            resp = json.loads(raw)
            if isinstance(resp, dict) and resp.get("candles"):
                parsed = []
                for c in resp["candles"]:
                    try:
                        parsed.append({"epoch": int(c["epoch"]), "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])})
                    except Exception:
                        continue
                parsed.sort(key=lambda x: x["epoch"])
                return parsed
            # fallback ticks
            if DEBUG:
                log(f"{symbol}@{granularity}s: no candles, trying ticks fallback (attempt {attempt})")
            req2 = {"ticks_history": symbol, "end": "latest", "count": max(500, granularity * count // 5), "style": "ticks"}
            ws.send(json.dumps(req2))
            raw2 = ws.recv()
            resp2 = json.loads(raw2)
            if isinstance(resp2, dict):
                if "history" in resp2 and "times" in resp2["history"] and "prices" in resp2["history"]:
                    times = resp2["history"]["times"]; prices = resp2["history"]["prices"]
                    ticks = [{"epoch": int(t), "quote": p} for t, p in zip(times, prices)]
                    cand = aggregate_ticks_to_candles(ticks, granularity, count)
                    if cand:
                        return cand
                elif resp2.get("candles"):
                    parsed = []
                    for c in resp2["candles"]:
                        try:
                            parsed.append({"epoch": int(c["epoch"]), "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])})
                        except Exception:
                            continue
                    parsed.sort(key=lambda x: x["epoch"])
                    return parsed
        except Exception as e:
            if DEBUG:
                log("fetch_candles error:", e)
        finally:
            try:
                if ws: ws.close()
            except Exception:
                pass
        time.sleep(1)
    if DEBUG:
        log(f"fetch_candles final fail for {symbol}@{granularity}")
    return []

# -------------------------
# Signal evaluation - reuse main's logic condensed (no chart)
# -------------------------
def find_recent_ma3_cross(candles: List[Dict], ma3: List[Optional[float]], lookback: int = 6) -> Tuple[Optional[int], Optional[str]]:
    n = len(candles)
    if n < 3:
        return None, None
    start = max(1, n - lookback)
    for i in range(start, n):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        prev_above = candles[i-1]["close"] > ma3[i-1]
        now_above = candles[i]["close"] > ma3[i]
        if prev_above != now_above:
            return i, ("BUY" if now_above else "SELL")
    return None, None

def evaluate_signals_for_asset(symbol: str, candles: List[Dict], tf: int) -> List[Dict]:
    out = []
    n = len(candles)
    if n < 6:
        reason_skip(symbol, tf, f"insufficient candles ({n})")
        return out
    ma1, ma2, ma3 = compute_mas_for_chart(candles)
    atr = compute_atr_estimate(candles)
    cand_indices = []
    if n >= 2:
        cand_indices.append(n-2)
    if n >= 3:
        cand_indices.append(n-3)
    def near_ma(idx, side):
        try:
            price_point = candles[idx]["low"] if side == "BUY" else candles[idx]["high"]
            d1 = abs(price_point - (ma1[idx] if ma1[idx] is not None else float("inf")))
            d2 = abs(price_point - (ma2[idx] if ma2[idx] is not None else float("inf")))
            dist = min(d1, d2)
            return dist <= (DIST_TOL * atr), dist
        except Exception:
            return False, float("inf")
    def trend_up(idx):
        try:
            return ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and (ma1[idx] > ma2[idx] > ma3[idx]) and (ma1[idx] > ma1[max(0, idx-3)])
        except Exception:
            return False
    def trend_down(idx):
        try:
            return ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and (ma1[idx] < ma2[idx] < ma3[idx]) and (ma1[idx] < ma1[max(0, idx-3)])
        except Exception:
            return False
    # continuation
    for idx in cand_indices:
        if trend_up(idx):
            ok, dist = near_ma(idx, "BUY")
            if ok and is_rejection_candle(candles[idx], candles[idx-1] if idx-1>=0 else None, atr, "BUY"):
                out.append({"symbol": symbol, "tf": tf, "direction": "BUY", "reason": "Continuation: MA support rejection", "candles": candles, "i_rej": idx, "atr": atr, "dist": dist})
                return out
        if trend_down(idx):
            ok, dist = near_ma(idx, "SELL")
            if ok and is_rejection_candle(candles[idx], candles[idx-1] if idx-1>=0 else None, atr, "SELL"):
                out.append({"symbol": symbol, "tf": tf, "direction": "SELL", "reason": "Continuation: MA resistance rejection", "candles": candles, "i_rej": idx, "atr": atr, "dist": dist})
                return out
    # reversal (MA3 breakout) with mandatory retest
    cross_idx, cross_dir = find_recent_ma3_cross(candles, ma3, lookback=6)
    if cross_idx is not None:
        for idx in cand_indices:
            ok, dist = near_ma(idx, cross_dir)
            if ok and is_rejection_candle(candles[idx], candles[idx-1] if idx-1>=0 else None, atr, cross_dir):
                out.append({"symbol": symbol, "tf": tf, "direction": cross_dir, "reason": f"Reversal: MA3 breakout + retest", "candles": candles, "i_rej": idx, "atr": atr, "cross_idx": cross_idx, "dist": dist})
                return out
    # near-ma loose acceptance
    for idx in cand_indices:
        ok_b, dist_b = near_ma(idx, "BUY")
        if ok_b and is_rejection_candle(candles[idx], candles[idx-1] if idx-1>=0 else None, atr, "BUY"):
            out.append({"symbol": symbol, "tf": tf, "direction": "BUY", "reason": "Near-MA rejection (loose)", "candles": candles, "i_rej": idx, "atr": atr, "dist": dist_b})
            return out
        ok_s, dist_s = near_ma(idx, "SELL")
        if ok_s and is_rejection_candle(candles[idx], candles[idx-1] if idx-1>=0 else None, atr, "SELL"):
            out.append({"symbol": symbol, "tf": tf, "direction": "SELL", "reason": "Near-MA rejection (loose)", "candles": candles, "i_rej": idx, "atr": atr, "dist": dist_s})
            return out
    return out

# -------------------------
# Runner
# -------------------------
def run_once():
    load_cache()
    signals_sent = 0
    for symbol in ASSETS:
        symbol = symbol.strip()
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            break
        log("Scanning", symbol)
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 6:
                reason_skip(symbol, tf, "insufficient candles")
                continue
            candidates = evaluate_signals_for_asset(symbol, candles, tf)
            if not candidates:
                continue
            for c in candidates:
                direction = c["direction"]
                if not can_send(symbol, direction, tf):
                    log(f"{symbol} -> cooldown suppression for {direction}")
                    continue
                # send text-only
                price = c["candles"][c["i_rej"]]["close"]
                text = f"🔔 {symbol} | {tf//60}m | {direction}\nReason: {c['reason']}\nPrice: {price:.5f}"
                try:
                    ok = send_single_timeframe_signal(symbol, tf, direction, c["reason"])
                    # fall back to send_telegram_message if needed
                    if not ok and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
                except Exception as e:
                    log("send error:", e)
                mark_sent(symbol, direction, tf)
                signals_sent += 1
                break
            if signals_sent >= MAX_SIGNALS_PER_RUN:
                break
    save_cache()
    log("Run complete. signals_sent:", signals_sent)

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Fatal:", e)
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ 1s Bot crashed: {e}")
        except Exception:
            pass
        raise

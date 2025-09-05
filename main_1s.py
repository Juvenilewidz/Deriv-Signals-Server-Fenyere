#!/usr/bin/env python3
# main_1s.py
# Paste-and-cruise runner specialized for 1s synthetic indices (5m TF).
# Sends text-only Telegram alerts when signals match the combined logic:
#  - MA3 breakout + retest (reversal)
#  - Trend continuation: MA1/MA2 rejections during aligned trend
#
# Configuration via environment variables at top of file.

import os
import json
import time
import math
import tempfile
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import websocket
import numpy as np
# No requests — use stdlib for Telegram to avoid dependency issues
import urllib.parse
import urllib.request

# -------------------------
# Config (change via env)
# -------------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# Only 1s assets by default (override with ASSETS env)
ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")

# Timeframes (seconds) - single 5m only as requested
TIMEFRAMES = [300]  # 5m only

# Candle history
CANDLES_N = int(os.getenv("CANDLES_N", "100"))  # default changed to 100
MIN_CANDLES_REQUIRED = int(os.getenv("MIN_CANDLES_REQUIRED", "60"))

# Dispatch/cooldown
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))  # default 10 minutes
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "5"))

# Heartbeat / startup
STARTUP_HEARTBEAT = os.getenv("STARTUP_HEARTBEAT", "0") in ("1", "true", "True", "yes")

# Persist files (tmp)
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")

# -------------------------
# Utilities
# -------------------------
def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

def safe_json_load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_json_dump(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

# -------------------------
# Telegram (stdlib)
# -------------------------
def send_telegram_text(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    """Send message using Telegram Bot API via urllib (no requests dependency)."""
    try:
        base = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        url = base + "?" + urllib.parse.urlencode(payload)
        with urllib.request.urlopen(url, timeout=15) as r:
            resp = r.read().decode("utf-8")
        return True, resp
    except Exception as e:
        log("telegram send failed:", e)
        return False, str(e)

# -------------------------
# Persistence: cooldowns
# -------------------------
last_sent_cache: Dict[str, Dict] = {}

def load_cache():
    global last_sent_cache
    last_sent_cache = safe_json_load(CACHE_FILE) or {}

def save_cache():
    safe_json_dump(CACHE_FILE, last_sent_cache)

def can_send(symbol: str, direction: str, tf: int) -> bool:
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec:
        if rec.get("direction") == direction and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
            return False
    return True

def mark_sent(symbol: str, direction: str, tf: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time())}
    save_cache()

# -------------------------
# Moving averages
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

def compute_mas(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes = [c["close"] for c in candles]
    highs  = [c["high"] for c in candles]
    lows   = [c["low"] for c in candles]
    hlc3   = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
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

# -------------------------
# Fetch candles (Deriv websocket)
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N, max_retries: int = 3) -> List[Dict]:
    """
    Fetch candle history via Deriv ticks_history (snapshot subscribe=0).
    Returns list of candles sorted by epoch ascending.
    """
    for attempt in range(1, max_retries + 1):
        ws = None
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
            # authorize
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            _ = ws.recv()
            req = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": "latest",
                "subscribe": 0
            }
            ws.send(json.dumps(req))
            raw = ws.recv()
            resp = json.loads(raw)
            if isinstance(resp, dict) and "candles" in resp and resp["candles"]:
                candles = []
                for c in resp["candles"]:
                    try:
                        candles.append({
                            "epoch": int(c.get("epoch", 0)),
                            "open": float(c.get("open", 0.0)),
                            "high": float(c.get("high", 0.0)),
                            "low": float(c.get("low", 0.0)),
                            "close": float(c.get("close", 0.0))
                        })
                    except Exception:
                        continue
                candles.sort(key=lambda x: x["epoch"])
                return candles
        except Exception as e:
            log(f"fetch_candles error for {symbol} @{granularity}s attempt {attempt}:", e)
        finally:
            try:
                if ws:
                    ws.close()
            except Exception:
                pass
        time.sleep(1)
    return []

# -------------------------
# Candle helpers & rejection detection
# -------------------------
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o)
    rng = max(1e-12, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l
    is_doji = body <= 0.35 * rng
    pin_low = (lower >= 0.2 * rng) and (lower > upper)
    pin_high = (upper >= 0.2 * rng) and (upper > lower)
    engulf_bull = engulf_bear = False
    if prev:
        po = float(prev["open"]); pc = float(prev["close"])
        if pc < po and c > o and o <= pc and c >= po:
            engulf_bull = True
        if pc > po and c < o and o >= pc and c <= po:
            engulf_bear = True
    return {
        "o": o, "h": h, "l": l, "c": c, "body": body, "range": rng,
        "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
        "engulf_bull": engulf_bull, "engulf_bear": engulf_bear
    }

def compute_atr_estimate(candles: List[Dict], period: int = 14) -> float:
    rngs = [c["high"] - c["low"] for c in candles if c["high"] >= c["low"]]
    if not rngs:
        return 1.0
    if len(rngs) >= period:
        return float(np.mean(rngs[-period:]))
    return float(np.mean(rngs))

# Consider a candle a "rejection" in direction (uptrend -> BUY) if:
#  - small body (doji-like) OR pin with tail in opposite direction OR very small body relative to ATR
#  - close is in direction of the trend (e.g. uptrend: close > open)
def is_rejection_candle(candle: Dict, prev: Optional[Dict], atr: float, direction: str) -> bool:
    b = candle_bits(candle, prev)
    tiny_body_threshold = max(0.15 * atr, 0.0001)
    # Accept tiny bodies (flat candles) as valid rejection
    if b["body"] <= tiny_body_threshold:
        # also ensure close is in direction (or neutral)
        if direction == "BUY":
            return candle["close"] >= candle["open"]
        else:
            return candle["close"] <= candle["open"]
    # Pin-checks
    if direction == "BUY" and b["pin_low"]:
        return True
    if direction == "SELL" and b["pin_high"]:
        return True
    # Doji acceptance as rejection if closing in direction
    if b["is_doji"]:
        if direction == "BUY":
            return candle["close"] >= candle["open"]
        else:
            return candle["close"] <= candle["open"]
    # Engulf might be considered rejection if in direction
    if direction == "BUY" and b["engulf_bull"]:
        return True
    if direction == "SELL" and b["engulf_bear"]:
        return True
    return False

# -------------------------
# Signal evaluation (combined reversal + continuation)
# -------------------------
def evaluate_signals_for_asset(symbol: str, candles: List[Dict], tf: int) -> List[Dict]:
    """
    Returns list of candidate signals: dict with keys symbol, tf, direction, reason, price, extras
    """
    out = []
    n = len(candles)
    if n < MIN_CANDLES_REQUIRED:
        log(f"{symbol} -> insufficient candles ({n} < {MIN_CANDLES_REQUIRED})")
        return out

    ma1, ma2, ma3 = compute_mas(candles)
    # get last indices
    last_idx = n - 1
    prev_idx = n - 2 if n >= 2 else None

    # ATR estimate
    atr = compute_atr_estimate(candles, period=14)

    # Helper: check recent alignment (for trend)
    def is_trend_up(i: int = last_idx, lookback: int = 3) -> bool:
        # require MA1 > MA2 > MA3 and slopes positive over lookback
        try:
            for k in range(i - lookback + 1, i + 1):
                if k < 0 or ma1[k] is None or ma2[k] is None or ma3[k] is None:
                    return False
            # alignment at last candle
            if not (ma1[i] > ma2[i] > ma3[i]):
                return False
            # slopes (compare older to newer)
            if not (ma1[i] > ma1[i - lookback + 1] and ma2[i] > ma2[i - lookback + 1] and ma3[i] > ma3[i - lookback + 1]):
                return False
            return True
        except Exception:
            return False

    def is_trend_down(i: int = last_idx, lookback: int = 3) -> bool:
        try:
            for k in range(i - lookback + 1, i + 1):
                if k < 0 or ma1[k] is None or ma2[k] is None or ma3[k] is None:
                    return False
            if not (ma1[i] < ma2[i] < ma3[i]):
                return False
            if not (ma1[i] < ma1[i - lookback + 1] and ma2[i] < ma2[i - lookback + 1] and ma3[i] < ma3[i - lookback + 1]):
                return False
            return True
        except Exception:
            return False

    # Distance helper: how far price is from an MA (absolute)
    def dist_to_ma(price: float, ma_val: Optional[float]) -> float:
        if ma_val is None:
            return float("inf")
        return abs(price - ma_val)

    last_candle = candles[last_idx]
    prev_candle = candles[prev_idx] if prev_idx is not None else None

    # Evaluate continuation signals first: trend aligned and last candle a rejection near MA1 or MA2
    if is_trend_up():
        # distance threshold: allow near-touch up to X * ATR
        price = last_candle["close"]
        # prefer MA1 then MA2
        d1 = dist_to_ma(price, ma1[last_idx])
        d2 = dist_to_ma(price, ma2[last_idx])
        # acceptance thresholds
        accept_thresh = max(0.5 * atr, 1e-6)  # you can tune; 0.5 ATR allows shallow near-touches
        near_ma = "MA1" if d1 <= accept_thresh else ("MA2" if d2 <= accept_thresh else None)
        if near_ma:
            if is_rejection_candle(last_candle, prev_candle, atr, "BUY"):
                out.append({
                    "symbol": symbol, "tf": tf, "direction": "BUY",
                    "reason": f"MA{ '1' if near_ma=='MA1' else '2' } support rejection (trend continuation)",
                    "price": last_candle["close"], "atr": atr
                })
    elif is_trend_down():
        price = last_candle["close"]
        d1 = dist_to_ma(price, ma1[last_idx])
        d2 = dist_to_ma(price, ma2[last_idx])
        accept_thresh = max(0.5 * atr, 1e-6)
        near_ma = "MA1" if d1 <= accept_thresh else ("MA2" if d2 <= accept_thresh else None)
        if near_ma:
            if is_rejection_candle(last_candle, prev_candle, atr, "SELL"):
                out.append({
                    "symbol": symbol, "tf": tf, "direction": "SELL",
                    "reason": f"MA{ '1' if near_ma=='MA1' else '2' } resistance rejection (trend continuation)",
                    "price": last_candle["close"], "atr": atr
                })

    # Evaluate reversal (MA3 breakout + retest):
    # If price crossed MA3 in the last few bars and now there is a rejection near MA1/MA2 in the new side direction.
    cross_window = 6  # look back window for cross
    # detect crosses: find last index where sign(prev close - ma3) != sign(close - ma3)
    cross_idx = None
    cross_dir = None
    for i in range(max(1, last_idx - cross_window + 1), last_idx + 1):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        prev_above = candles[i-1]["close"] > ma3[i-1]
        now_above = candles[i]["close"] > ma3[i]
        if prev_above != now_above:
            cross_idx = i
            cross_dir = "BUY" if now_above else "SELL"
    if cross_idx is not None:
        # If cross happened recently and we now have a rejection candle near MA1/MA2 in new direction -> reversal
        # look for a retest: check last candle distance to MA1/MA2
        price = last_candle["close"]
        d1 = dist_to_ma(price, ma1[last_idx])
        d2 = dist_to_ma(price, ma2[last_idx])
        accept_thresh = max(0.6 * atr, 1e-6)  # slightly wider for retest acceptance
        near_ma = "MA1" if d1 <= accept_thresh else ("MA2" if d2 <= accept_thresh else None)
        if near_ma:
            if is_rejection_candle(last_candle, prev_candle, atr, cross_dir):
                out.append({
                    "symbol": symbol, "tf": tf, "direction": cross_dir,
                    "reason": f"MA3 breakout + retest at {near_ma} (reversal)",
                    "price": last_candle["close"], "atr": atr, "cross_idx": cross_idx
                })

    return out

# -------------------------
# Runner
# -------------------------
def run_once():
    load_cache()
    reports = []
    signals_sent = 0

    for symbol in ASSETS:
        symbol = symbol.strip()
        if not symbol:
            continue
        for tf in TIMEFRAMES:
            log(f"Fetching {CANDLES_N} candles for {symbol} @{tf}s")
            candles = fetch_candles(symbol, tf, count=CANDLES_N, max_retries=3)
            if not candles or len(candles) < MIN_CANDLES_REQUIRED:
                log(f"{symbol} @{tf}s -> insufficient candles ({len(candles) if candles else 0})")
                continue
            candidates = evaluate_signals_for_asset(symbol, candles, tf)
            for c in candidates:
                if signals_sent >= MAX_SIGNALS_PER_RUN:
                    log("Reached max signals per run")
                    break
                direction = c["direction"]
                if not can_send(symbol, direction, tf):
                    log(f"{symbol} -> cooldown: skip {direction}")
                    continue
                # Build message
                price = c.get("price")
                reason = c.get("reason", "")
                atr = c.get("atr", None)
                text = f"🔔 *{symbol} | {tf//60}m | {direction}*\nReason: {reason}\nPrice: {price:.2f}\nATR: {atr:.4f}"
                ok, resp = send_telegram_text(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
                if ok:
                    mark_sent(symbol, direction, tf)
                    signals_sent += 1
                    log(f"Sent alert for {symbol} {direction}: {reason}")
                else:
                    log(f"Failed sending alert for {symbol}: {resp}")
            if signals_sent >= MAX_SIGNALS_PER_RUN:
                break
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            break

    # summary
    log("Run complete. Signals sent:", signals_sent)
    return signals_sent

def send_startup_heartbeat():
    if not STARTUP_HEARTBEAT:
        return
    try:
        text = "✅ main_1s.py startup heartbeat (5m timeframe, 1s indices)"
        send_telegram_text(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        log(">>> DEBUG main_1s.py starting...")
        log("TELEGRAM_BOT_TOKEN length:", len(TELEGRAM_BOT_TOKEN))
        log("TELEGRAM_CHAT_ID:", "***" if TELEGRAM_CHAT_ID else "")
        send_startup_heartbeat()
        start = time.time()
        run_once()
        log("Done in", time.time() - start, "seconds")
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        try:
            send_telegram_text(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"main_1s.py error: {e}")
        except Exception:
            pass
        raise

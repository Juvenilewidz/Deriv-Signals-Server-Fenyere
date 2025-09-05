# main_1s.py
# Text-only 1s indices runner (5m TF only).
# - No charts, only text alerts to Telegram
# - Robust 5m fetch with fallbacks; notifies when candles missing
# - Uses your bot.py helpers (must exist in repo)

import os
import sys
import json
import time
import math
import tempfile
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import websocket
import numpy as np

# ensure repo dir on path so "from bot import ..." works in GH Actions
sys.path.append(os.path.dirname(__file__) or ".")

# Strictly require your bot.py helpers so messages go to Telegram (no silent fallback)
try:
    from bot import send_telegram_message, send_single_timeframe_signal, send_strong_signal
except Exception as e:
    raise ImportError("Failed to import bot.py. Ensure bot.py is in the repository and exports "
                      "send_telegram_message, send_single_timeframe_signal, send_strong_signal.") from e

# -------------------------
# Config
# -------------------------
DEBUG = True

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
ASSETS = [a.strip() for a in ASSETS if a.strip()]

# Only 5 minute timeframe (text alerts will reference 5m)
TIMEFRAMES = [300]  # seconds

# desired number of 5m candles
CANDLES_N = int(os.getenv("CANDLES_N", "100"))

# persistence & control
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")
HEART_FILE = os.path.join(TMPDIR, "fenyere_last_heartbeat_1s.json")

MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "0"))

last_sent_cache: Dict[str, Dict] = {}


def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)


# -------------------------
# Cache helpers
# -------------------------
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
    if rec:
        if rec.get("direction") == direction and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
            return False
    return True


def mark_sent(symbol: str, direction: str, tf: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time())}
    save_cache()


# -------------------------
# MA helpers
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
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
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


# -------------------------
# Candle patterns & scoring
# -------------------------
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o)
    r = max(1e-12, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l
    is_doji = body <= 0.35 * r
    pin_low = (lower >= 0.2 * body) and (lower > upper)
    pin_high = (upper >= 0.2 * body) and (upper > lower)
    engulf_bull = False; engulf_bear = False
    if prev:
        po = float(prev["open"]); pc = float(prev["close"])
        if pc < po and c > o and o <= pc and c >= po:
            engulf_bull = True
        if pc > po and c < o and o >= pc and c <= po:
            engulf_bear = True
    return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
            "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
            "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}


def compute_score_for_rejection(candles: List[Dict], i_rej: int, ma1: List[Optional[float]],
                                ma2: List[Optional[float]], ma3: List[Optional[float]], tf: int) -> Tuple[int, Dict]:
    n = len(candles)
    if i_rej < 0 or i_rej >= n:
        return 0, {}
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)

    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    def slope_ok(i, lookback=2, up=True):
        if i - lookback < 0:
            return False
        try:
            if up:
                return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None and
                        ma1[i] > ma1[i - lookback] and ma2[i] > ma2[i - lookback] and ma3[i] > ma3[i - lookback])
            else:
                return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None and
                        ma1[i] < ma1[i - lookback] and ma2[i] < ma2[i - lookback] and ma3[i] < ma3[i - lookback])
        except Exception:
            return False

    trend_score = 0
    if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
        if ma1[i_rej] > ma2[i_rej] > ma3[i_rej] and slope_ok(i_rej, up=True):
            trend_score = 3
        elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej] and slope_ok(i_rej, up=False):
            trend_score = 3

    prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
    bits = candle_bits(candles[i_rej], prev)

    if bits["is_doji"] or bits["pin_low"] or bits["pin_high"]:
        pattern_score = 2
    elif bits["engulf_bull"] or bits["engulf_bear"]:
        pattern_score = 1
    else:
        pattern_score = 0

    try:
        d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
        d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
        d1h = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
        d2h = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
    except Exception:
        d1 = d2 = d1h = d2h = float("inf")

    dist = min(d1, d2, d1h, d2h)
    if dist <= 0.1 * atr:
        proximity_score = 2
    elif dist <= 0.25 * atr:
        proximity_score = 1
    else:
        proximity_score = 0

    tf_weight = {300: 1}.get(tf, 1)
    body_small = 1 if bits["body"] <= 0.35 * bits["range"] else 0

    total = trend_score + pattern_score + proximity_score + tf_weight + body_small
    details = {"trend": trend_score, "pattern": pattern_score, "prox": proximity_score, "tf_w": tf_weight,
               "body_small": body_small, "dist": dist}
    return int(total), details


# -------------------------
# Websocket helper
# -------------------------
def _ws_auth_and_send(req: dict, timeout: int = 18) -> Optional[dict]:
    ws = None
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=timeout)
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        # try to read auth response
        try:
            _ = ws.recv()
        except Exception:
            pass
        ws.send(json.dumps(req))
        raw = ws.recv()
        resp = json.loads(raw)
        return resp
    except Exception as e:
        log("WS error:", e)
        return None
    finally:
        try:
            if ws:
                ws.close()
        except Exception:
            pass


# -------------------------
# Robust fetch for 5m (granularity=300)
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Try multiple counts (requested -> fallbacks). Return oldest-first list or [].
    """
    if granularity != 300:
        granularity = 300

    requested = max(1, int(count))
    fallbacks = [requested, 200, 150, 100, 50, 25]
    # keep unique, preserve order
    seen = set()
    fallbacks = [x for x in fallbacks if x > 0 and (x not in seen and not seen.add(x)) is None or True][:6]

    for fc in fallbacks:
        for attempt in range(1, 4):
            req = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": fc,
                "end": "latest",
                "subscribe": 0
            }
            resp = _ws_auth_and_send(req)
            if resp and "candles" in resp and resp["candles"]:
                parsed = []
                for c in resp["candles"]:
                    try:
                        parsed.append({
                            "epoch": int(c["epoch"]),
                            "open": float(c["open"]),
                            "high": float(c["high"]),
                            "low": float(c["low"]),
                            "close": float(c["close"])
                        })
                    except Exception:
                        continue
                parsed.sort(key=lambda x: x["epoch"])
                log(f"{symbol} -> fetched {len(parsed)} candles @{granularity}s using count={fc} (attempt {attempt})")
                return parsed
            else:
                log(f"{symbol} -> no candles (attempt {attempt}) with count={fc}")
            time.sleep(0.8)
    log(f"{symbol} -> all fetch attempts failed for granularity {granularity}")
    return []


# -------------------------
# Text-only alert composition & send
# -------------------------
def compose_alert_text(symbol: str, tf: int, direction: str, reason: str, score: int, details: Dict, price: float, ts_epoch: int) -> str:
    t = datetime.utcfromtimestamp(ts_epoch).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"🔔 {symbol} | {tf//60}m | {direction}",
        f"Score: {score}",
        f"Reason: {reason}",
        f"Price: {price}",
        f"Time: {t}",
        f"Details: {details}"
    ]
    return "\n".join(lines)


# -------------------------
# Analysis & notify (text-only)
# -------------------------
def analyze_and_notify():
    load_cache()
    signals_sent = 0
    reports: List[str] = []

    for symbol in ASSETS:
        symbol = symbol.strip()
        if not symbol:
            continue

        rejections = []
        accepted = []

        for tf in sorted(TIMEFRAMES):
            log(f"{symbol}: fetching {CANDLES_N} candles for {tf}s")
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 6:
                msg = f"⚠️ {symbol} @{tf//60}m -> insufficient candles ({len(candles) if candles else 0})"
                log(msg)
                # send status message so you're aware
                try:
                    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
                except Exception as e:
                    log("Failed to send insufficient-candles message:", e)
                continue

            i_rej = len(candles) - 2
            ma1, ma2, ma3 = compute_mas_for_chart(candles)

            direction = None
            reason = ""
            try:
                if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                    # BUY
                    if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                        d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        dist_low = min(d1, d2)
                        atr = float(np.mean([c["high"] - c["low"] for c in candles][-14:])) if len(candles) >= 14 else float(np.mean([c["high"] - c["low"] for c in candles]))
                        near = dist_low <= 0.25 * atr
                        pattern_ok = bits["is_doji"] or bits["pin_low"] or bits["engulf_bull"]
                        if near and pattern_ok:
                            direction = "BUY"
                            reason = "MA support rejection"
                    # SELL
                    elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        d1 = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                        d2 = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        dist_high = min(d1, d2)
                        atr = float(np.mean([c["high"] - c["low"] for c in candles][-14:])) if len(candles) >= 14 else float(np.mean([c["high"] - c["low"] for c in candles]))
                        near = dist_high <= 0.25 * atr
                        pattern_ok = bits["is_doji"] or bits["pin_high"] or bits["engulf_bear"]
                        if near and pattern_ok:
                            direction = "SELL"
                            reason = "MA resistance rejection"
            except Exception:
                if DEBUG:
                    traceback.print_exc()

            score, details = compute_score_for_rejection(candles, i_rej, ma1, ma2, ma3, tf)

            if direction:
                accepted.append({"tf": tf, "direction": direction, "reason": reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})
            else:
                prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                bits = candle_bits(candles[i_rej], prev)
                pat = "Doji" if bits["is_doji"] else ("PinLow" if bits["pin_low"] else ("PinHigh" if bits["pin_high"] else ("EngulfB" if bits["engulf_bear"] else ("EngulfU" if bits["engulf_bull"] else "None"))))
                rej_reason = f"Rejected({pat})"
                rejections.append({"tf": tf, "reason": rej_reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})

        # choose best
        chosen_alert = None
        alert_type = None
        if accepted:
            chosen_alert = max(accepted, key=lambda x: x["score"])
            alert_type = "accepted"
        elif rejections:
            chosen_alert = max(rejections, key=lambda x: x["score"])
            alert_type = "rejection"

        if chosen_alert and signals_sent < MAX_SIGNALS_PER_RUN:
            tf_ch = chosen_alert["tf"]
            direction_ch = chosen_alert.get("direction", "REJECTED")
            reason_ch = chosen_alert.get("reason", "")
            candles_ch = chosen_alert["candles"]
            i_rej_ch = chosen_alert.get("i_rej", len(candles_ch) - 2)
            price = candles_ch[i_rej_ch]["close"] if candles_ch and 0 <= i_rej_ch < len(candles_ch) else None
            score_val = chosen_alert.get("score", 0)
            details = chosen_alert.get("details", {})

            if can_send(symbol, direction_ch, tf_ch):
                text = compose_alert_text(symbol, tf_ch, direction_ch, reason_ch, score_val, details, price, candles_ch[i_rej_ch]["epoch"] if candles_ch and i_rej_ch < len(candles_ch) else int(time.time()))
                try:
                    ok, info = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
                    if ok:
                        mark_sent(symbol, direction_ch, tf_ch)
                        signals_sent += 1
                        reports.append(f"{symbol} -> SENT {direction_ch} @{tf_ch//60}m score={score_val}")
                        log(f"Sent text alert for {symbol} @{tf_ch//60}m {direction_ch} s={score_val}")
                    else:
                        reports.append(f"{symbol} -> FAILED SEND (telegram replied false)")
                        log(f"Telegram send returned False for {symbol}")
                except Exception as e:
                    reports.append(f"{symbol} -> FAILED SEND EXC")
                    log("Error sending telegram message:", e)
            else:
                reports.append(f"{symbol} -> SKIPPED cooldown for {direction_ch} @{tf_ch//60}m")
                log(f"Skipped {symbol} due cooldown for {direction_ch}")
        else:
            reports.append(f"{symbol} -> no alerts chosen")
            log(f"{symbol} -> no alerts chosen")

        if signals_sent >= MAX_SIGNALS_PER_RUN:
            log("Reached MAX_SIGNALS_PER_RUN for this run")
            break

    # heartbeat if nothing sent and heartbeat enabled
    if signals_sent == 0 and HEARTBEAT_INTERVAL_HOURS > 0:
        try:
            last_h = 0
            if os.path.exists(HEART_FILE):
                try:
                    with open(HEART_FILE, "r") as f:
                        last_h = int(json.load(f).get("ts", 0))
                except Exception:
                    last_h = 0
            now = int(time.time())
            if now - last_h >= int(HEARTBEAT_INTERVAL_HOURS * 3600):
                checked = ", ".join(ASSETS)
                hb = f"🤖 1s Bot heartbeat – alive\n⏰ No signals right now.\n📊 Checked: {checked}\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                try:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, hb)
                except Exception:
                    pass
                with open(HEART_FILE, "w") as f:
                    json.dump({"ts": now}, f)
        except Exception:
            pass

    save_cache()
    if DEBUG:
        log("Run complete. Reports:", reports)


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    print(">>> DEBUG main_1s.py starting")
    print("TELEGRAM_BOT_TOKEN length:", len(TELEGRAM_BOT_TOKEN))
    print("TELEGRAM_CHAT_ID:", TELEGRAM_CHAT_ID)
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            ok, info = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                             "✅ main_1s.py startup heartbeat (5m timeframe, text-only alerts)")
            print(">>> DEBUG: heartbeat send result:", ok, info)
        else:
            print(">>> DEBUG: TELEGRAM credentials missing")
    except Exception as e:
        print(">>> DEBUG: heartbeat failed:", e)

    try:
        analyze_and_notify()
    except Exception as e:
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ 1s Bot crashed: {e}")
        except Exception:
            pass
        log("Fatal error in analyze_and_notify:", e)
        traceback.print_exc()
        raise

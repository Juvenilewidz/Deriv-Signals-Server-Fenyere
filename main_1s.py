# main_1s.py
# Paste-and-cruise 1s-specialized bot (5m only, robust 1s->5m aggregation)
# - Defaults: ASSETS = 1HZ75V,1HZ100V,1HZ150V
# - TIMEFRAMES = [300] (5m only)
# - CANDLES_N default = 100 (number of 5m candles)
# - DEBUG forced on, startup heartbeat to Telegram

import os
import json
import time
import math
import tempfile
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Try to import telegram helpers from your existing bot.py (preferred)
try:
    from bot import (
        send_telegram_message,
        send_single_timeframe_signal,
        send_strong_signal,
        send_telegram_photo
    )
except Exception:
    # safe fallbacks so file can be tested locally without bot.py
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM TEXT]", text)
        return True, "fallback"

    def send_single_timeframe_signal(symbol, tf, direction, reason, chart_path=None):
        print("[SIG]", symbol, tf, direction, reason, chart_path)
        return True

    def send_strong_signal(symbol, direction, details, chart_path=None):
        print("[STRONG]", symbol, direction, details, chart_path)
        return True

    def send_telegram_photo(token, chat_id, caption, photo_path):
        print("[PHOTO]", caption, photo_path)
        return True, "fallback"


# -------------------------
# Config (1s specialized)
# -------------------------
DEBUG = True  # forced on so runs are visible in logs

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# Only 1s assets by default (can be overridden via ASSETS env var)
ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
ASSETS = [a.strip() for a in ASSETS if a.strip()]

# Only 5 minute timeframe (user requested only 5m charts)
TIMEFRAMES = [300]  # seconds: 5m only

# Candle history defaults: number of 5m candles desired
CANDLES_N = int(os.getenv("CANDLES_N", "100"))  # default 100 5m bars
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "220"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))

# Chart drawing
CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))

# Controls
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))

# Heartbeat
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "0"))

# Temp files for caches
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")
HEART_FILE = os.path.join(TMPDIR, "fenyere_last_heartbeat_1s.json")

# In-memory cache
last_sent_cache: Dict[str, Dict] = {}


def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)


# -------------------------
# Persistence helpers
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
# MA implementations
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
# Aggregation helper (1s -> target granularity)
# -------------------------
def aggregate_1s_to_tf(candles_1s: List[Dict], target_granularity: int) -> List[Dict]:
    """
    Aggregate list of 1s candles into target_granularity (seconds).
    Returns sorted list of aggregated candles (oldest-first).
    """
    if not candles_1s:
        return []

    buckets: Dict[int, Dict] = {}
    for c in candles_1s:
        epoch = int(c["epoch"])
        bucket = epoch - (epoch % target_granularity)
        if bucket not in buckets:
            buckets[bucket] = {
                "epoch": bucket,
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
            }
        else:
            b = buckets[bucket]
            b["high"] = max(b["high"], c["high"])
            b["low"] = min(b["low"], c["low"])
            b["close"] = c["close"]

    aggregated = sorted(buckets.values(), key=lambda x: x["epoch"])
    return aggregated


# -------------------------
# Robust fetch with fallbacks
# -------------------------
def _ws_auth_and_send(req: dict) -> Optional[dict]:
    """Helper: open ws, authorize, send req, return parsed json or None."""
    ws = None
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        # try reading auth response (non-fatal)
        try:
            _ = ws.recv()
        except Exception:
            pass
        ws.send(json.dumps(req))
        raw = ws.recv()
        resp = json.loads(raw)
        return resp
    except Exception as e:
        log("WS helper error:", e)
        return None
    finally:
        try:
            if ws:
                ws.close()
        except Exception:
            pass


def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Robust fetch for 1s indices:
    - If symbol looks like 1s index, fetch raw 1s candles and aggregate to requested granularity (5m).
    - Tries fallback counts for 1s fetch: [count_in_seconds, 15000, 7200, 3600, 1800, 900]
    - For non-1s symbols, behaves as normal (asks Deriv for requested granularity).
    Returns oldest-first list of candles (open/high/low/close/epoch).
    """
    is_1s = symbol.startswith("1HZ") or symbol.startswith("1HZ")  # keep simple prefix check

    # If requesting a higher TF but symbol is 1s, we'll fetch raw 1s and aggregate
    if is_1s:
        # count here is number of desired 5m bars; convert to seconds (1s candles)
        desired_5m_bars = max(1, int(count))
        needed_1s = desired_5m_bars * granularity  # granularity is 300 (5m) as we call only 5m
        # prepare fallback counts (number of 1s candles to request)
        fallback_counts = [
            needed_1s,
            min(15000, needed_1s),
            15000,
            7200,
            3600,
            1800,
            900,
            300,
        ]
        # eliminate duplicates and sort descending
        seen = set()
        fc = []
        for x in fallback_counts:
            if x is None or x <= 0:
                continue
            if x in seen:
                continue
            seen.add(x)
            fc.append(int(x))

        for fc_count in fc:
            # cap fc_count to a reasonable max to avoid huge requests
            if fc_count > 30000:
                fc_count = 30000
            for attempt in range(1, 4):
                req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": 1,  # raw 1s
                    "count": fc_count,
                    "end": "latest",
                    "subscribe": 0
                }
                resp = _ws_auth_and_send(req)
                if resp and "candles" in resp and resp["candles"]:
                    # parse 1s candles
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
                    log(f"{symbol} -> fetched {len(parsed)} raw 1s candles (requested {fc_count}) on attempt {attempt}")
                    # aggregate into target granularity (e.g., 300s)
                    aggregated = aggregate_1s_to_tf(parsed, granularity)
                    if aggregated:
                        # ensure we return at most 'count' many 5m candles (most recent)
                        if len(aggregated) > desired_5m_bars:
                            aggregated = aggregated[-desired_5m_bars:]
                        log(f"{symbol} -> aggregated to {len(aggregated)} {granularity}s candles")
                        return aggregated
                    else:
                        log(f"{symbol} -> aggregation returned 0 buckets")
                else:
                    log(f"{symbol} -> no candles (attempt {attempt}) with raw1s count={fc_count}")
                time.sleep(0.9)
        log(f"{symbol} -> all 1s fetch attempts failed for granularity {granularity}")
        return []

    # Non-1s symbols (shouldn't be used in this file normally)
    for attempt in range(1, 4):
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
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
            log(f"{symbol} -> fetched {len(parsed)} candles @ {granularity}s")
            return parsed
        else:
            log(f"{symbol} -> no candles (attempt {attempt}) for granularity {granularity} count={count}")
            time.sleep(1)
    log(f"{symbol} -> failed to fetch candles for granularity {granularity}")
    return []


# -------------------------
# Charting (compact)
# -------------------------
def make_chart(candles: List[Dict], ma1: List[Optional[float]], ma2: List[Optional[float]],
               ma3: List[Optional[float]], rej_index: int, reason: str, symbol: str, tf: int,
               last_n: int = LAST_N_CHART, pad: int = PAD_CANDLES) -> Optional[str]:
    try:
        total = len(candles)
        if total == 0:
            return None
        show_n = min(last_n, total)
        start = max(0, total - show_n)
        chosen = candles[start: total]
        xs = [datetime.utcfromtimestamp(c["epoch"]) for c in chosen]
        opens = [c["open"] for c in chosen]
        highs = [c["high"] for c in chosen]
        lows = [c["low"] for c in chosen]
        closes = [c["close"] for c in chosen]

        fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
        ax.set_title(f"{symbol} | {tf // 60}m | {reason}", fontsize=10)

        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = "#2ca02c" if c >= o else "#d62728"
            ax.plot([i, i], [l, h], color="black", linewidth=0.6, zorder=1)
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - CANDLE_WIDTH / 2.0, lower), CANDLE_WIDTH, height,
                             facecolor=color, edgecolor="black", linewidth=0.35, zorder=2)
            ax.add_patch(rect)

        ma1_vals = []
        ma2_vals = []
        ma3_vals = []
        for i in range(start, total):
            ma1_vals.append(ma1[i] if i < len(ma1) else None)
            ma2_vals.append(ma2[i] if i < len(ma2) else None)
            ma3_vals.append(ma3[i] if i < len(ma3) else None)

        def plot_ma(vals, label):
            y = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float('nan') for v in vals]
            ax.plot(list(range(len(y))), y, label=label, linewidth=1.0, zorder=3)

        try:
            plot_ma(ma1_vals, "MA1")
            plot_ma(ma2_vals, "MA2")
            plot_ma(ma3_vals, "MA3")
            ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        if start <= rej_index < total:
            idx = rej_index - start
            price = chosen[idx]["close"]
            marker_color = "red" if ("SELL" in reason or "sell" in reason.lower()) else "green"
            ax.scatter([idx], [price], marker="v" if "SELL" in reason else "^",
                       color=marker_color, s=120, zorder=6, edgecolors="black")

        ax.set_xlim(-1, len(chosen) - 1 + pad)
        ymin, ymax = min(lows), max(highs)
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        step = max(1, len(chosen) // 8)
        ax.set_xticks(range(0, len(chosen), step))
        ax.set_xticklabels([xs[i].strftime("%H:%M\n%m-%d") for i in range(0, len(xs), step)], rotation=25, fontsize=8)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.tight_layout()
        fig.savefig(tmp.name, dpi=120)
        plt.close(fig)
        return tmp.name
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return None


# -------------------------
# Candle bits & scoring
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

    tf_weight = {300: 1}.get(tf, 1)  # only 5m used here
    body_small = 1 if bits["body"] <= 0.35 * bits["range"] else 0

    total = trend_score + pattern_score + proximity_score + tf_weight + body_small
    details = {"trend": trend_score, "pattern": pattern_score, "prox": proximity_score, "tf_w": tf_weight,
               "body_small": body_small, "dist": dist}
    return int(total), details


# -------------------------
# Main analysis & notify
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
            log(f"{symbol}: fetching target {CANDLES_N} candles for {tf}s (will use 1s->5m aggregation if needed)")
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 6:
                log(f"{symbol} @{tf}s -> insufficient candles ({len(candles) if candles else 0}), skipping")
                continue

            i_rej = len(candles) - 2
            ma1, ma2, ma3 = compute_mas_for_chart(candles)

            direction = None
            reason = ""
            try:
                if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                    # BUY check (support rejection)
                    if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        try:
                            d1 = abs(bits["l"] - ma1[i_rej])
                            d2 = abs(bits["l"] - ma2[i_rej])
                        except Exception:
                            d1 = d2 = float("inf")
                        dist_low = min(d1, d2)
                        atr = float(np.mean([c["high"] - c["low"] for c in candles][-14:])) if len(candles) >= 14 else float(np.mean([c["high"] - c["low"] for c in candles]))
                        near = dist_low <= 0.25 * atr
                        pattern_ok = bits["is_doji"] or bits["pin_low"] or bits["engulf_bull"]
                        if near and pattern_ok:
                            direction = "BUY"
                            reason = "MA support rejection"
                    # SELL check (resistance rejection)
                    elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        try:
                            d1 = abs(bits["h"] - ma1[i_rej])
                            d2 = abs(bits["h"] - ma2[i_rej])
                        except Exception:
                            d1 = d2 = float("inf")
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

        # choose best alert or rejection summary
        chosen_alert = None
        alert_type = None
        if accepted:
            chosen_alert = max(accepted, key=lambda x: x["score"])
            alert_type = "accepted"
        elif rejections:
            chosen_alert = max(rejections, key=lambda x: x["score"])
            alert_type = "rejection"

        other_summary_lines = []
        if rejections:
            sorted_rej = sorted(rejections, key=lambda x: x["score"], reverse=True)
            for r in sorted_rej[:6]:
                other_summary_lines.append(f"{r['tf']//60}m → {r['reason']} (s={r['score']})")

        if chosen_alert and signals_sent < MAX_SIGNALS_PER_RUN:
            tf_ch = chosen_alert["tf"]
            direction_ch = chosen_alert.get("direction", "REJECTED")
            reason_ch = chosen_alert.get("reason", "")
            candles_ch = chosen_alert["candles"]
            i_rej_ch = chosen_alert.get("i_rej", len(candles_ch) - 2)

            if can_send(symbol, direction_ch, tf_ch):
                ma1c, ma2c, ma3c = compute_mas_for_chart(candles_ch)
                chart_path = make_chart(candles_ch, ma1c, ma2c, ma3c, i_rej_ch, f"{direction_ch}", symbol, tf_ch, last_n=LAST_N_CHART, pad=PAD_CANDLES)

                emoji = "✅" if alert_type == "accepted" else "❌"
                caption_lines = [
                    f"{emoji} {symbol} | {tf_ch//60}m | {direction_ch} | s={chosen_alert.get('score',0)}",
                    f"Reason: {reason_ch}"
                ]
                if other_summary_lines:
                    caption_lines.append("")
                    caption_lines.append("Other TFs: " + " | ".join(other_summary_lines))

                caption = "\n".join(caption_lines)

                sent_ok = False
                try:
                    if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        ok, info = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                        sent_ok = ok
                        try:
                            os.unlink(chart_path)
                        except Exception:
                            pass
                    else:
                        send_single_timeframe_signal(symbol, tf_ch, direction_ch, reason_ch)
                        sent_ok = True
                except Exception as e:
                    sent_ok = False
                    log(f"Error sending telegram for {symbol}: {e}")

                if sent_ok:
                    mark_sent(symbol, direction_ch, tf_ch)
                    signals_sent += 1
                    reports.append(f"{symbol} -> SENT {direction_ch} @{tf_ch//60}m score={chosen_alert['score']}")
                    log(f"Sent alert for {symbol} @{tf_ch//60}m {direction_ch} (s={chosen_alert['score']})")
                else:
                    reports.append(f"{symbol} -> FAILED SEND {direction_ch} @{tf_ch//60}m")
                    log(f"Failed to send alert for {symbol}")
            else:
                reports.append(f"{symbol} -> SKIPPED cooldown for {direction_ch} @{tf_ch//60}m")
                log(f"Skipped {symbol} due cooldown for {direction_ch}")
        else:
            reports.append(f"{symbol} -> no alerts chosen")
            log(f"{symbol} -> no alerts chosen")

        if not chosen_alert and rejections and DEBUG:
            sorted_rej = sorted(rejections, key=lambda x: x["score"], reverse=True)
            summary_lines = [f"{r['tf']//60}m:{r['reason']} (s={r['score']})" for r in sorted_rej[:6]]
            log(f"{symbol} rejections summary (no chart sent):", " | ".join(summary_lines))

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
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
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
# Entry point
# -------------------------
if __name__ == "__main__":
    # startup debug info + heartbeat so Actions logs and Telegram confirm runner health
    print(">>> DEBUG main_1s.py starting")
    print("TELEGRAM_BOT_TOKEN length:", len(TELEGRAM_BOT_TOKEN))
    print("TELEGRAM_CHAT_ID:", TELEGRAM_CHAT_ID)
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            ok, info = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                                             "✅ main_1s.py startup heartbeat (5m timeframe, 1s indices)")
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

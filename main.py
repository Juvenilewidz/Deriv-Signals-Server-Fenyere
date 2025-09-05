# main.py
"""
Paste-and-cruise main runner (candles, MAs per spec, single-chart per asset,
strongest TF chosen, compact summaries, muted heartbeat by default).
"""

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

# Try to import bot helpers (these should be in your bot.py)
try:
    from bot import (
        send_telegram_message,
        send_single_timeframe_signal,
        send_strong_signal,
        send_telegram_photo
    )
except Exception:
    # minimal fallback (prints) so file can be tested locally without Telegram
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
# Configuration
# -------------------------
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# Assets & TFs (leave TFs as requested: 5m/10m/15m)
# Default assets used historically in your logs — override with ASSETS env if needed
ASSETS = os.getenv("ASSETS", "R_10,R_50,R_75,1HZ75V,1HZ100V,1HZ150V").split(",")
TIMEFRAMES = [300, 600, 900]  # seconds: 5m, 10m, 15m

# Charting/history
CANDLES_N = int(os.getenv("CANDLES_N", "400"))  # how many candles to fetch (history)
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "220"))  # how many last candles to show in chart (more history)
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))  # right padding as requested

# Drawing
CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))  # smaller candles

# Dispatch / cooldown
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "3"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))

# Heartbeat: user asked to mute heartbeat -> default 0 (disable).
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "0"))

# Persist files (tmp)
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "fenyere_last_heartbeat.json")

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
    except Exception as e:
        log("load_cache failed:", e)
        last_sent_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
    except Exception as e:
        log("save_cache failed:", e)

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
# MA implementations as you requested
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
    """
    MA1: SMMA(9) on HLC/3
    MA2: SMMA(19) on Close
    MA3: SMA(25) on MA2 values (previous indicator)
    """
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    hlc3    = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    ma1 = smma(hlc3, 9)        # MA1: SMMA(9) on HLC/3
    ma2 = smma(closes, 19)     # MA2: SMMA(19) on Close
    # MA3: apply SMA(25) on the MA2 values (previous indicator). Keep alignment with candles.
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
# Fetch candles (Deriv)
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Robust fetch for Deriv candles:
     - Uses subscribe=0 for history snapshot requests (faster)
     - Retries a few times on failure
     - If the single request returns fewer than `count`, will page backwards
       (using 'end' = oldest_epoch - 1) to accumulate up to `count`.
    Returns list of candles oldest-first or [] on failure.
    """
    MAX_RETRIES = 3
    PAGE_SIZE = min(200, count)  # fetch pages up to this size (200 safer than huge counts)
    all_candles: List[Dict] = []

    def _request_history(ws, end_param, this_count):
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": this_count,
            "end": end_param,
            "subscribe": 0
        }
        ws.send(json.dumps(req))
        try:
            raw = ws.recv()
            return json.loads(raw)
        except Exception as e:
            log("fetch_candles recv/json failed:", e)
            return {}

    # Try to connect (once). If connect fails, return [].
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
    except Exception as e:
        log("WS connect failed:", e)
        return []

    try:
        # authorize
        try:
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            auth_raw = ws.recv()
            auth = json.loads(auth_raw)
            if isinstance(auth, dict) and auth.get("error"):
                log("Deriv auth error:", auth)
                return []
        except Exception as e:
            log("Auth failure:", e)
            return []

        # page backwards and/or retry until we have enough candles or attempts exhausted
        attempts = 0
        end = "latest"
        remaining = count

        while len(all_candles) < count and attempts < MAX_RETRIES:
            attempts += 1
            this_fetch = min(PAGE_SIZE if PAGE_SIZE > 0 else count, remaining)
            resp = _request_history(ws, end, this_fetch)
            if not resp or "candles" not in resp or not resp["candles"]:
                log(f"fetch attempt {attempts} for {symbol} tf={granularity} returned no candles (end={end})")
                # small backoff
                time.sleep(0.6 * attempts)
                continue

            # Deriv returns newest-first? In practice in your earlier code you treated the returned list as
            # chronological (oldest-first). Ensure we normalize: if candles appear descending by epoch, reverse.
            received = resp["candles"]
            if len(received) >= 2 and received[0] and received[-1] and received[0].get("epoch", 0) > received[-1].get("epoch", 0):
                received = list(reversed(received))

            # convert to our canonical dicts (oldest-first)
            parsed = []
            for c in received:
                try:
                    parsed.append({
                        "epoch": int(c["epoch"]),
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"])
                    })
                except Exception:
                    # ignore malformed candle
                    continue

            if not parsed:
                log("parsed empty after conversion, skipping attempt", attempts)
                time.sleep(0.4)
                continue

            # If this is the first page and end == "latest", just accept what we got (it will be newest->oldest maybe)
            # We want all_candles to be oldest-first. Prepend pages because we're fetching newest-first pages going back.
            # If we fetched with end="latest" and got oldest-first list, then appending is correct. To be safe, use epochs.
            if not all_candles:
                all_candles = parsed[:]  # start
            else:
                # insert older page before current list (since we are paging backwards)
                # Choose correct merging based on epoch values:
                if parsed[-1]["epoch"] < all_candles[0]["epoch"]:
                    # parsed are older than existing -> prepend
                    all_candles = parsed + all_candles
                else:
                    # fallback: append
                    all_candles = all_candles + parsed

            # update remaining and end param for next page
            remaining = max(0, count - len(all_candles))
            # set next end to oldest_epoch - 1 to page further back
            oldest_epoch = all_candles[0]["epoch"] if all_candles else parsed[0]["epoch"]
            end = oldest_epoch - 1

            # small delay to be courteous
            time.sleep(0.25)

        # final sanity: ensure order oldest->newest
        all_candles.sort(key=lambda x: x["epoch"])
        # If still too few, still return what we have (calling code only requires len >= 6)
        return all_candles

    except Exception as e:
        log("fetch_candles error:", e)
        return []
    finally:
        try:
            ws.close()
        except Exception:
            pass

# -------------------------
# Candlestick chart: small candles + padding + MA overlays
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

        fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
        ax.set_title(f"{symbol} | {tf//60}m | {reason}", fontsize=10)

        # draw candles (small width)
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = "#2ca02c" if c >= o else "#d62728"
            # wick
            ax.plot([i, i], [l, h], color="black", linewidth=0.6, zorder=1)
            # body
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - CANDLE_WIDTH / 2.0, lower), CANDLE_WIDTH, height,
                             facecolor=color, edgecolor="black", linewidth=0.35, zorder=2)
            ax.add_patch(rect)

        # prepare MA windows aligned with chosen window
        ma1_vals = []
        ma2_vals = []
        ma3_vals = []
        for i in range(start, total):
            ma1_vals.append(ma1[i] if i < len(ma1) else None)
            ma2_vals.append(ma2[i] if i < len(ma2) else None)
            ma3_vals.append(ma3[i] if i < len(ma3) else None)

        def plot_ma(vals, label, color):
            y = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float('nan') for v in vals]
            ax.plot(list(range(len(y))), y, label=label, color=color, linewidth=1.0, zorder=3)

        try:
            plot_ma(ma1_vals, "MA1", "#1f77b4")
            plot_ma(ma2_vals, "MA2", "#ff7f0e")
            plot_ma(ma3_vals, "MA3", "#2ca02c")
            ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        # mark rejection index if within shown window
        if start <= rej_index < total:
            idx = rej_index - start
            price = chosen[idx]["close"]
            marker_color = "red" if ("SELL" in reason or "sell" in reason.lower()) else "green"
            ax.scatter([idx], [price], marker="v" if "SELL" in reason else "^",
                       color=marker_color, s=120, zorder=6, edgecolors="black")

        # axis limits + right padding
        ax.set_xlim(-1, len(chosen) - 1 + pad)
        ymin, ymax = min(lows), max(highs)
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # xticks: show 8 ticks at most
        step = max(1, len(chosen)//8)
        ax.set_xticks(range(0, len(chosen), step))
        ax.set_xticklabels([xs[i].strftime("%H:%M\n%m-%d") for i in range(0, len(xs), step)], rotation=25, fontsize=8)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.tight_layout()
        fig.savefig(tmp.name, dpi=120)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        log("make_chart error:", e)
        traceback.print_exc()
        return None

# -------------------------
# Candle pattern & scoring helpers
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

def compute_score_for_rejection(candles: List[Dict], i_rej: int, ma1: List[Optional[float]], ma2: List[Optional[float]],
                                ma3: List[Optional[float]], tf: int) -> Tuple[int, Dict]:
    """Return integer score and details dict. Higher = stronger."""
    n = len(candles)
    if i_rej < 0 or i_rej >= n:
        return 0, {}
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)

    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    # trend stacking & slope
    def slope_ok(i, lookback=2, up=True):
        if i - lookback < 0: return False
        try:
            if up:
                return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
                        and ma1[i] > ma1[i - lookback] and ma2[i] > ma2[i - lookback] and ma3[i] > ma3[i - lookback])
            else:
                return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
                        and ma1[i] < ma1[i - lookback] and ma2[i] < ma2[i - lookback] and ma3[i] < ma3[i - lookback])
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

    # proximity to MAs
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

    tf_weight = {300: 1, 600: 2, 900: 3}.get(tf, 1)
    body_small = 1 if bits["body"] <= 0.35 * bits["range"] else 0

    total = trend_score + pattern_score + proximity_score + tf_weight + body_small
    details = {"trend": trend_score, "pattern": pattern_score, "prox": proximity_score, "tf_w": tf_weight, "body_small": body_small, "dist": dist}
    return int(total), details

# -------------------------
# Main logic
# -------------------------
def analyze_and_notify():
    load_cache()
    signals_sent = 0
    reports = []

    for symbol in ASSETS:
        log("Scanning", symbol)
        rejections = []
        accepted = []

        for tf in sorted(TIMEFRAMES):
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 6:
                # skip quietly on insufficient data (user requested less spam)
                log(symbol, tf, "insufficient candles or no data, skipping")
                continue

            i_rej = len(candles) - 2  # candidate rejection candle
            ma1, ma2, ma3 = compute_mas_for_chart(candles)

            direction = None
            reason = ""
            try:
                # check MA stacking + pattern near MA
                if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                    # BUY candidate
                    if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        # distance to lower MA near low
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
                    # SELL candidate
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
                log("detector error for", symbol, tf, traceback.format_exc())

            score, details = compute_score_for_rejection(candles, i_rej, ma1, ma2, ma3, tf)

            if direction:
                accepted.append({"tf": tf, "direction": direction, "reason": reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})
                log("Accepted candidate", symbol, tf, direction, "score", score)
            else:
                prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                bits = candle_bits(candles[i_rej], prev)
                pat = "Doji" if bits["is_doji"] else ("PinLow" if bits["pin_low"] else ("PinHigh" if bits["pin_high"] else ("EngulfB" if bits["engulf_bear"] else ("EngulfU" if bits["engulf_bull"] else "None"))))
                rej_reason = f"Rejected({pat})"
                rejections.append({"tf": tf, "reason": rej_reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})
                log("Rejection captured", symbol, tf, rej_reason, "score", score)

        # choose single alert per asset: highest accepted OR highest rejection
        chosen_alert = None
        alert_type = None
        if accepted:
            chosen_alert = max(accepted, key=lambda x: x["score"])
            alert_type = "accepted"
        elif rejections:
            chosen_alert = max(rejections, key=lambda x: x["score"])
            alert_type = "rejection"

        # prepare compact other-TF summary that will be included in the caption (one message per asset)
        other_summary_lines = []
        if rejections:
            sorted_rej = sorted(rejections, key=lambda x: x["score"], reverse=True)
            for r in sorted_rej[:6]:
                emoji = "✅" if ("BUY" in r.get("reason","") or "buy" in r.get("reason","").lower()) else "❌"
                other_summary_lines.append(f"{r['tf']//60}m → {r['reason']} (s={r['score']})")

        # send chosen_alert chart+caption
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
                    caption_lines.append("")  # blank line
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
                        # fallback: simple text send
                        send_single_timeframe_signal(symbol, tf_ch, direction_ch, reason_ch)
                        sent_ok = True
                except Exception as e:
                    log("send error:", e)
                    sent_ok = False

                if sent_ok:
                    mark_sent(symbol, direction_ch, tf_ch)
                    signals_sent += 1
                    reports.append(f"{symbol} -> SENT {direction_ch} @{tf_ch//60}m score={chosen_alert['score']}")
                else:
                    reports.append(f"{symbol} -> FAILED SEND {direction_ch} @{tf_ch//60}m")
            else:
                reports.append(f"{symbol} -> SKIPPED cooldown for {direction_ch} @{tf_ch//60}m")
        else:
            reports.append(f"{symbol} -> no alerts chosen")

        # send compact summary if there were no chosen alert but there are rejections and debug enabled
        # (avoid spamming when not requested). We already included other-TFs in the caption when sending chart.
        # If nothing was sent and there are rejections, optionally log to console for debugging.
        if not chosen_alert and rejections and DEBUG:
            sorted_rej = sorted(rejections, key=lambda x: x["score"], reverse=True)
            summary_lines = [f"{r['tf']//60}m:{r['reason']} (s={r['score']})" for r in sorted_rej[:6]]
            log(f"{symbol} rejections summary (no chart sent):", " | ".join(summary_lines))

        if signals_sent >= MAX_SIGNALS_PER_RUN:
            log("Reached MAX_SIGNALS_PER_RUN:", MAX_SIGNALS_PER_RUN)
            break

    # heartbeat: muted by default, only send if HEARTBEAT_INTERVAL_HOURS > 0
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
                hb = f"🤖 Bot heartbeat – alive\n⏰ No signals right now.\n📊 Checked: {checked}\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, hb)
                with open(HEART_FILE, "w") as f:
                    json.dump({"ts": now}, f)
        except Exception:
            log("heartbeat exception", traceback.format_exc())

    save_cache()
    log("Run complete. Reports:", reports)

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        log("Fatal:", e)
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

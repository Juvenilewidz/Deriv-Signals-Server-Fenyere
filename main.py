# main.py
"""
Paste-and-cruise main runner (updated: picks highest-probability rejection per asset,
sends single chart per asset, but still summarises all other rejections compactly).
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
    # Provide minimal fallback stubs so file doesn't crash when importing
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

# Assets & TFs (5m,10m,15m)
ASSETS = os.getenv("ASSETS", "R_10,R_50,R_75,1HZ75V,1HZ100V,1HZ150V").split(",")
TIMEFRAMES = [300, 600, 900]  # seconds: 5m, 10m, 15m

CANDLES_N = int(os.getenv("CANDLES_N", "240"))  # history to fetch
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "80"))  # last-n candles shown in chart
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))  # right padding in candles

# Dispatch / cooldown
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "3"))  # max assets alerted per run
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))  # avoid duplicate alert for same symbol

# Heartbeat
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))  # 0 disables

# Persist files (tmp)
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "ai_forex_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "ai_forex_last_heartbeat.json")

# In-memory cache
last_sent_cache: Dict[str, Dict] = {}  # symbol -> {"direction": str, "tf": int, "ts": int}

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
# Utilities: moving averages
# -------------------------
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if n == 0:
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
    if n < period:
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
    ma2_vals = [x for x in ma2 if x is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None] * len(ma2_vals)
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
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
    except Exception as e:
        log("WS connect failed:", e)
        return []
    try:
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        _ = json.loads(ws.recv())

        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "subscribe": 1
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
                "close": float(c["close"])
            })
        # try to pull one live update for the forming candle
        try:
            update = json.loads(ws.recv())
            if "candles" in update and update["candles"]:
                live_c = update["candles"][-1]
                out[-1] = {
                    "epoch": int(live_c["epoch"]),
                    "open": float(live_c["open"]),
                    "high": float(live_c["high"]),
                    "low": float(live_c["low"]),
                    "close": float(live_c["close"])
                }
        except Exception:
            pass
        return out
    except Exception as e:
        log("fetch_candles error:", e)
        return []
    finally:
        try:
            ws.close()
        except Exception:
            pass

# -------------------------
# Candlestick chart function (candles small + right padding)
# -------------------------
def make_chart(candles: List[Dict], ma1: List[Optional[float]], ma2: List[Optional[float]],
               ma3: List[Optional[float]], rej_index: int, reason: str, symbol: str, tf: int,
               last_n: int = LAST_N_CHART, pad: int = PAD_CANDLES) -> Optional[str]:
    try:
        total = len(candles)
        show_n = min(last_n, total)
        start = max(0, total - show_n)
        chosen = candles[start: total]
        xs = [datetime.utcfromtimestamp(c["epoch"]) for c in chosen]
        opens = [c["open"] for c in chosen]
        highs = [c["high"] for c in chosen]
        lows = [c["low"] for c in chosen]
        closes = [c["close"] for c in chosen]

        fig, ax = plt.subplots(figsize=(11, 4), dpi=110)
        ax.set_title(f"{symbol} | {tf//60}m | {reason[:30]}")
        # draw candles
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = "#2ca02c" if c >= o else "#d62728"
            # wick
            ax.plot([i, i], [l, h], color="black", linewidth=0.6, zorder=1)
            # body
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - 0.3, lower), 0.6, height, facecolor=color, edgecolor="black", linewidth=0.3, zorder=2)
            ax.add_patch(rect)

        # MA overlay: align indices to the chosen window
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

        # limits + right padding
        ax.set_xlim(-1, len(chosen) - 1 + pad)
        ymin, ymax = min(lows), max(highs)
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_xticks(range(0, len(chosen), max(1, len(chosen)//8)))
        ax.set_xticklabels([xs[i].strftime("%H:%M\n%m-%d") for i in range(0, len(xs), max(1, len(xs)//8))], rotation=25)

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
# Rejection / pattern helpers & scoring
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
    """
    Score a rejection. Higher is better.
    Components:
      - trend_score (MA stacking + slope) -> up to 3
      - pattern_score (doji/pin:2, engulf:1) -> up to 2
      - proximity_score to nearest MA (<=0.1*ATR:2, <=0.25*ATR:1)
      - timeframe_weight: 5m=1,10m=2,15m=3
      - small_body_bonus (body small relative to ATR) -> 1
    """
    n = len(candles)
    if i_rej < 0 or i_rej >= n:
        return 0, {}
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    opens = [c["open"] for c in candles]
    closes = [c["close"] for c in candles]

    # ATR-like
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    # trend stacking & slope & separation
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

    # pattern
    prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
    bits = candle_bits(candles[i_rej], prev)
    if bits["is_doji"] or bits["pin_low"] or bits["pin_high"]:
        pattern_score = 2
    elif bits["engulf_bull"] or bits["engulf_bear"]:
        pattern_score = 1
    else:
        pattern_score = 0

    # proximity to nearest MA1/MA2
    zone = None
    try:
        d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
    except Exception:
        d1 = float("inf")
    try:
        d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
    except Exception:
        d2 = float("inf")
    # choose low/high check depending on pattern: use l for buy-like, h for sell-like
    # We'll compute distance relative to both low and high and pick the smaller.
    try:
        d1h = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
        d2h = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
    except Exception:
        d1h = d2h = float("inf")

    dist = min(d1, d2, d1h, d2h)
    if dist <= 0.1 * atr:
        proximity_score = 2
    elif dist <= 0.25 * atr:
        proximity_score = 1
    else:
        proximity_score = 0

    # timeframe weight
    tf_weight = {300: 1, 600: 2, 900: 3}.get(tf, 1)

    # small body bonus
    body_small = 1 if bits["body"] <= 0.35 * bits["range"] else 0

    total = trend_score + pattern_score + proximity_score + tf_weight + body_small

    details = {"trend": trend_score, "pattern": pattern_score, "prox": proximity_score, "tf_w": tf_weight, "body_small": body_small, "dist": dist}
    return int(total), details

# -------------------------
# Main logic: scan and pick highest-probability per asset
# -------------------------
def analyze_and_notify():
    load_cache()
    load_ts = int(time.time())
    signals_sent = 0
    reports = []

    for symbol in ASSETS:
        log("Scanning", symbol)
        rejections = []  # list of dicts with tf, reason, candles, score, details
        accepted = []    # list of accepted signals (tf, dir, reason, candles, score)
        # check each timeframe
        for tf in sorted(TIMEFRAMES):
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 4:
                log(symbol, tf, "insufficient candles")
                continue
            i_rej = len(candles) - 2
            i_con = len(candles) - 1

            # compute mas and signal
            ma1, ma2, ma3 = compute_mas_for_chart(candles)
            # Reuse signal_for_timeframe logic but implemented inline to keep single-file
            # We'll call a concise detector (similar to earlier behavior)
            # For brevity we call the same scoring to evaluate rejections
            # Determine if it qualifies as an accepted signal:
            direction = None; reason = ""
            # quick check: detect trend stacking roughly
            try:
                if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                    if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                        # potential buy rejection if pattern near MA
                        # check pattern
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        near = False
                        # distance to nearer MA (low)
                        try:
                            d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                            d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        except Exception:
                            d1 = d2 = float("inf")
                        dist_low = min(d1, d2)
                        rngs = np.array([c["high"] - c["low"] for c in candles], dtype=float)
                        atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
                        if dist_low <= 0.25 * atr:
                            near = True
                        pattern_ok = bits["is_doji"] or bits["pin_low"] or bits["engulf_bull"]
                        if near and pattern_ok:
                            direction = "BUY"
                            reason = "MA support rejection"
                    elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej]:
                        prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                        bits = candle_bits(candles[i_rej], prev)
                        try:
                            d1 = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                            d2 = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        except Exception:
                            d1 = d2 = float("inf")
                        dist_high = min(d1, d2)
                        rngs = np.array([c["high"] - c["low"] for c in candles], dtype=float)
                        atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
                        near = dist_high <= 0.25 * atr
                        pattern_ok = bits["is_doji"] or bits["pin_high"] or bits["engulf_bear"]
                        if near and pattern_ok:
                            direction = "SELL"
                            reason = "MA resistance rejection"
            except Exception:
                pass

            # Score the rejection / accepted signals
            # Score uses the same compute_score_for_rejection
            score, details = compute_score_for_rejection(candles, i_rej, ma1, ma2, ma3, tf)

            if direction:
                accepted.append({"tf": tf, "direction": direction, "reason": reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})
                log("Accepted candidate", symbol, tf, direction, "score", score)
            else:
                # capture rejection reason for summary
                # build a short reason string
                prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
                bits = candle_bits(candles[i_rej], prev)
                pat = "Doji" if bits["is_doji"] else ("PinLow" if bits["pin_low"] else ("PinHigh" if bits["pin_high"] else ("EngulfB" if bits["engulf_bear"] else ("EngulfU" if bits["engulf_bull"] else "None"))))
                rej_reason = f"Rejected({pat})"
                rejections.append({"tf": tf, "reason": rej_reason, "candles": candles, "score": score, "details": details, "i_rej": i_rej})
                log("Rejection captured", symbol, tf, rej_reason, "score", score)

        # Decide what to send:
        # 1) If there are accepted signals: pick highest-scoring accepted (if multiple) and send it.
        # 2) Else pick the single highest-scoring rejection and send its chart (others only summarized).
        # This ensures one chart per asset while still summarizing all rejections (no spam).
        chosen_alert = None
        if accepted:
            chosen_alert = max(accepted, key=lambda x: x["score"])
            alert_type = "accepted"
        elif rejections:
            chosen_alert = max(rejections, key=lambda x: x["score"])
            alert_type = "rejection"
        else:
            chosen_alert = None
            alert_type = None

        # send chosen_alert (one chart) if allowed by cooldown and limit
        if chosen_alert and signals_sent < MAX_SIGNALS_PER_RUN:
            tf_ch = chosen_alert["tf"]
            direction_ch = chosen_alert.get("direction", "REJECTED")
            reason_ch = chosen_alert.get("reason", "")
            candles_ch = chosen_alert["candles"]
            i_rej_ch = chosen_alert.get("i_rej", len(candles_ch) - 2)
            # cooldown
            if can_send(symbol, direction_ch, tf_ch):
                # make chart
                ma1c, ma2c, ma3c = compute_mas_for_chart(candles_ch)
                chart_path = make_chart(candles_ch, ma1c, ma2c, ma3c, i_rej_ch, f"{direction_ch}", symbol, tf_ch, last_n=LAST_N_CHART, pad=PAD_CANDLES)
                caption = f"{symbol} | {tf_ch//60}m | {direction_ch}\nScore: {chosen_alert['score']}\nReason: {reason_ch}"
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
                        # fallback: text only send helper
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

        # send compact summary of all rejections for this asset (no charts) so you still "get them all"
        if rejections:
            # sort rejections by score desc
            sorted_rej = sorted(rejections, key=lambda x: x["score"], reverse=True)
            summary_lines = [f"{r['tf']//60}m:{r['reason']} (s={r['score']})" for r in sorted_rej]
            summary_text = f"🔍 {symbol} rejections: " + " | ".join(summary_lines[:6])  # cap to 6 items
            # send only if at least one was scored moderately high or user wants full logging
            try:
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, summary_text)
                else:
                    log(summary_text)
            except Exception as e:
                log("summary send failed:", e)

        # stop early if we've reached MAX_SIGNALS_PER_RUN
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            log("Reached MAX_SIGNALS_PER_RUN:", MAX_SIGNALS_PER_RUN)
            break

    # heartbeat: short & concise (only when NOTHING was sent this run)
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

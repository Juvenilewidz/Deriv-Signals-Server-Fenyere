# main.py
"""
Paste-and-cruise runner:
- Scans assets across TFs (5m,10m,15m)
- Detects MA rejections (fires on rejection candle close)
- Scores each TF, picks the highest-probability TF per asset
- Sends exactly ONE Telegram message per asset containing:
    * chart (candlesticks + MAs)
    * main result (BUY/SELL/REJECTED) + score + reason
    * blended short summary of other TFs (pattern, bias, score)
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
        send_telegram_photo,
        send_heartbeat
    )
except Exception:
    # minimal fallbacks to avoid import errors during editing
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
    def send_heartbeat(checked_assets):
        print("[HEARTBEAT]", checked_assets)
        return True, "fallback"

# -------------------
# Config (env-overrides)
# -------------------
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

ASSETS = os.getenv("ASSETS", "R_10,R_50,R_75,1HZ75V,1HZ100V,1HZ150V").split(",")
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m

CANDLES_N = int(os.getenv("CANDLES_N", "240"))
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "80"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))

MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "3"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))

HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))

TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "ai_forex_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "ai_forex_last_heartbeat.json")
last_sent_cache: Dict[str, Dict] = {}

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# -------------------------
# Persistence & cooldown
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
    except Exception as e:
        log("save_cache error", e)

def can_send(symbol: str, direction: str) -> bool:
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec and rec.get("direction") == direction and (now - rec.get("ts",0)) < ALERT_COOLDOWN_SECS:
        return False
    return True

def mark_sent(symbol: str, direction: str):
    last_sent_cache[symbol] = {"direction": direction, "ts": int(time.time())}
    save_cache()

# -------------------------
# MAs & helpers
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
# Candle pattern helpers
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
# Chart function (candles + MAs + padding)
# -------------------------
def make_chart(candles: List[Dict],
               rej_index: int,
               symbol: str,
               tf: int,
               last_n: int = 80,
               pad: int = 10) -> Optional[str]:
    """
    Draw candlesticks + MAs using your exact MA params:
      - MA1 = SMMA(9) on HLC/3 (typical price)
      - MA2 = SMMA(19) on Close
      - MA3 = SMA(25) on MA2 (previous indicator's data)

    Important: MAs are computed on CLOSED bars only (we drop the last forming candle).
    Inputs:
      candles: list of dicts with keys 'epoch','open','high','low','close'
      rej_index: index in the original candles list pointing to the rejection candle (usually len-2)
      symbol, tf: used for title
      last_n: how many candles to display (history)
      pad: extra blank candles to the right for breathing room
    Returns:
      path to PNG image (temp file) or None on error.
    """
    try:
        # minimal imports if called standalone (main.py normally already imports these)
        import math
        import tempfile
        from matplotlib.patches import Rectangle
        from datetime import datetime
        import matplotlib.pyplot as plt
        import numpy as np

        total = len(candles)
        if total == 0:
            return None

        # Drop the still-forming last candle for MA calculation and plotting (match MT5 closed-bar MAs)
        # Note: this also matches MT5 behavior where indicators are calculated on closed bars only.
        closed_candles = candles[:-1] if total >= 2 else candles[:]  # ensure at least 1 bar stays
        if len(closed_candles) < 5:
            # not enough closed bars to meaningfully plot
            return None

        # prepare arrays
        epochs = np.array([c["epoch"] for c in closed_candles], dtype=float)
        opens  = np.array([float(c["open"])  for c in closed_candles], dtype=float)
        highs  = np.array([float(c["high"])  for c in closed_candles], dtype=float)
        lows   = np.array([float(c["low"])   for c in closed_candles], dtype=float)
        closes = np.array([float(c["close"]) for c in closed_candles], dtype=float)

        # === SMMA implementation ===
        def smma_array(values: np.ndarray, period: int) -> np.ndarray:
            vals = np.asarray(values, dtype=float)
            n = len(vals)
            out = np.full(n, np.nan, dtype=float)
            if n < period:
                return out
            # seed = SMA of first 'period' values
            seed = float(np.mean(vals[:period]))
            out[period - 1] = seed
            prev = seed
            for i in range(period, n):
                prev = (prev * (period - 1) + float(vals[i])) / period
                out[i] = prev
            return out

        def sma_array(values: np.ndarray, period: int) -> np.ndarray:
            vals = np.asarray(values, dtype=float)
            n = len(vals)
            out = np.full(n, np.nan, dtype=float)
            if n < period:
                return out
            s = np.cumsum(vals, dtype=float)
            out[period - 1] = float(s[period - 1]) / period
            for i in range(period, n):
                out[i] = float((s[i] - s[i - period]) / period)
            return out

        # === compute MA inputs ===
        # typical price HLC/3 for MA1
        tp = (highs + lows + closes) / 3.0

        ma1_arr = smma_array(tp, 9)     # SMMA(9) on HLC/3
        ma2_arr = smma_array(closes, 19)# SMMA(19) on Close
        # MA3: SMA(25) on MA2 values (use only finite entries of ma2)
        # but keep alignment: compute SMA over the ma2_arr itself (NaNs handle missing early periods)
        ma3_arr = sma_array(np.nan_to_num(ma2_arr, nan=np.nan), 25)
        # Note: sma_array given NaNs becomes weird; safer to compute SMA only where ma2 has finite values:
        # Recompute ma3 properly:
        ma2_valid = np.where(np.isfinite(ma2_arr), ma2_arr, np.nan)
        # rolling SMA mindful of NaNs:
        def rolling_sma_on_valid(arr: np.ndarray, period: int) -> np.ndarray:
            n = len(arr)
            out = np.full(n, np.nan, dtype=float)
            # maintain a queue of last valid values
            vals = []
            for i in range(n):
                v = arr[i]
                if not math.isnan(v):
                    vals.append(v)
                else:
                    vals.append(np.nan)
                # compute window
                window = arr[max(0, i - period + 1):i + 1]
                window = np.array([x for x in window if not math.isnan(x)], dtype=float)
                if len(window) == period:
                    out[i] = float(np.mean(window))
                else:
                    out[i] = np.nan
            return out

        ma3_arr = rolling_sma_on_valid(ma2_valid, 25)

        # === select window for plotting (last_n closed candles) ===
        show_n = min(last_n, len(closed_candles))
        start = len(closed_candles) - show_n
        plot_opens  = opens[start:]
        plot_highs  = highs[start:]
        plot_lows   = lows[start:]
        plot_closes = closes[start:]
        plot_epochs = epochs[start:]
        plot_ma1 = ma1_arr[start:]
        plot_ma2 = ma2_arr[start:]
        plot_ma3 = ma3_arr[start:]

        # build x positions 0..N-1
        N = len(plot_closes)
        xs = list(range(N))

        # === Plotting ===
        fig, ax = plt.subplots(figsize=(11, 4), dpi=110)
        title = f"{symbol} | {tf//60}m"
        ax.set_title(title, fontsize=10)

        # candle width small so many candles fit
        width = max(0.12, 0.6 * (80.0 / max(1.0, float(N))))
        half_w = width / 2.0

        for i in range(N):
            o = plot_opens[i]; h = plot_highs[i]; l = plot_lows[i]; c = plot_closes[i]
            color = "#2ca02c" if c >= o else "#d62728"
            # wick
            ax.plot([i, i], [l, h], color="black", linewidth=0.5, zorder=1)
            # body rect
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - half_w, lower), width, height, facecolor=color, edgecolor="black", linewidth=0.25, zorder=2)
            ax.add_patch(rect)

        # MA lines: convert NaN to numpy NaN for plotting
        def safe_list(vs):
            return [float(x) if (x is not None and not (isinstance(x, float) and math.isnan(x))) else float('nan') for x in vs]

        try:
            ax.plot(xs, safe_list(plot_ma1), label="MA1 SMMA(9,HLC/3)", linewidth=1.0, zorder=3)
            ax.plot(xs, safe_list(plot_ma2), label="MA2 SMMA(19,Close)", linewidth=1.0, zorder=3)
            ax.plot(xs, safe_list(plot_ma3), label="MA3 SMA(25 of MA2)", linewidth=1.0, zorder=3)
            ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        # mark rejection candle if it falls within closed_candles window
        # Note: rej_index is index in original candles; after dropping last, map it if possible
        if rej_index is not None:
            if rej_index < len(closed_candles):
                if rej_index >= start:
                    idx = rej_index - start
                    price = plot_closes[idx]
                    # decide marker shape by small heuristic (if close < open -> sell-like marker)
                    marker = "v" if plot_closes[idx] < plot_opens[idx] else "^"
                    color = "red" if plot_closes[idx] < plot_opens[idx] else "green"
                    ax.scatter([idx], [price], marker=marker, color=color, s=120, zorder=6, edgecolors="black")

        # x-limits + right pad
        ax.set_xlim(-1, N - 1 + pad)
        ymin = float(np.nanmin(plot_lows)) if len(plot_lows) > 0 else 0.0
        ymax = float(np.nanmax(plot_highs)) if len(plot_highs) > 0 else 1.0
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # x ticks: show roughly 6 labels
        step = max(1, N // 6)
        ticks = list(range(0, N, step))
        labels = []
        for t in ticks:
            ts = datetime.utcfromtimestamp(plot_epochs[t])
            labels.append(ts.strftime("%m-%d\n%H:%M"))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=25, fontsize=8)

        fig.tight_layout()
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpf.name, dpi=120)
        plt.close(fig)
        return tmpf.name

    except Exception as e:
        # if anything goes wrong, log (if you have a logger) and return None
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        return None
# -------------------------
# Scoring helper (same logic as before)
# -------------------------
def compute_score_for_rejection(candles: List[Dict], i_rej: int, ma1: List[Optional[float]], ma2: List[Optional[float]],
                                ma3: List[Optional[float]], tf: int) -> Tuple[int, Dict]:
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

    try:
        d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
    except Exception:
        d1 = float("inf")
    try:
        d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
    except Exception:
        d2 = float("inf")
    try:
        d1h = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
        d2h = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
    except Exception:
        d1h = d2h = float("inf")

    dist = min(d1, d2, d1h, d2h)
    if atr > 0:
        if dist <= 0.1 * atr:
            proximity_score = 2
        elif dist <= 0.25 * atr:
            proximity_score = 1
        else:
            proximity_score = 0
    else:
        proximity_score = 0

    tf_weight = {300: 1, 600: 2, 900: 3}.get(tf, 1)
    body_small = 1 if bits["body"] <= 0.35 * bits["range"] else 0
    total = trend_score + pattern_score + proximity_score + tf_weight + body_small
    details = {"trend": trend_score, "pattern": pattern_score, "prox": proximity_score, "tf_w": tf_weight, "body_small": body_small, "dist": dist}
    return int(total), details

# -------------------------
# Build blended single message caption
# -------------------------
def build_caption_for_asset(symbol: str, chosen: Dict, per_tf_info: Dict[int, Dict]) -> str:
    tf_main = chosen["tf"]
    main_dir = chosen.get("direction", "REJECTED")
    main_score = chosen.get("score", 0)
    main_pat = chosen.get("pattern", "None")
    main_reason = chosen.get("reason", "")

    # pick emoji
    if main_dir in ("BUY", "SELL"):
        emoji = "✅"
    else:
        emoji = "❌"

    title = f"{emoji} {symbol} | {tf_main//60}m | {main_dir} | s={main_score} | {main_pat}"

    # build other TFs blended lines
    other_lines = []
    for tf in sorted(per_tf_info.keys()):
        info = per_tf_info[tf]
        if tf == tf_main:
            continue
        pat = info.get("pattern", "None")
        bias = info.get("bias", "Neutral")
        s = info.get("score", 0)
        label = info.get("direction") or "Rejected"
        emoji_sub = "✅" if label in ("BUY", "SELL") else "❌"
        other_lines.append(f"{emoji_sub} {tf//60}m → {label} ({pat}, {bias}) s={s}")

    footer = "\n".join(other_lines) if other_lines else ""
    caption = f"{title}\nReason: {main_reason}\n\nOther TFs:\n{footer}" if footer else f"{title}\nReason: {main_reason}"
    return caption
# -------------------------
# Main scanning & sending (single message per asset)
# -------------------------
def analyze_and_notify():
    load_cache()
    signals_sent = 0
    reports = []
    checked_assets = []

    for symbol in ASSETS:
        log("Scanning", symbol)
        checked_assets.append(symbol)
        per_tf_info: Dict[int, Dict] = {}
        candidates = []  # accepted or rejection candidates

        for tf in sorted(TIMEFRAMES):
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 4:
                per_tf_info[tf] = {"direction": None, "score": 0, "pattern": "None", "bias": "Neutral", "reason": "no data"}
                continue

            i_rej = len(candles) - 2
            i_con = len(candles) - 1
            ma1, ma2, ma3 = compute_mas_for_chart(candles)
            prev = candles[i_rej - 1] if i_rej - 1 >= 0 else None
            bits = candle_bits(candles[i_rej], prev)

            # determine bias from pattern and MA stacking
            bias = "Neutral"
            try:
                if bits["pin_low"] or bits["engulf_bull"]:
                    bias = "BUY"
                elif bits["pin_high"] or bits["engulf_bear"]:
                    bias = "SELL"
                else:
                    # fallback to MA stacking
                    if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                        if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                            bias = "BUY"
                        elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej]:
                            bias = "SELL"
            except Exception:
                bias = "Neutral"

            # detect accepted BUY/SELL candidate (rejection at MA)
            direction = None
            reason = ""
            try:
                if ma1[i_rej] is not None and ma2[i_rej] is not None and ma3[i_rej] is not None:
                    # buy case
                    if ma1[i_rej] > ma2[i_rej] > ma3[i_rej]:
                        d1 = abs(bits["l"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                        d2 = abs(bits["l"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        dist_low = min(d1, d2)
                        rngs = np.array([c["high"] - c["low"] for c in candles], dtype=float)
                        atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
                        pattern_ok = bits["is_doji"] or bits["pin_low"] or bits["engulf_bull"]
                        if dist_low <= 0.25 * atr and pattern_ok:
                            direction = "BUY"
                            reason = "MA support rejection"
                    # sell case
                    elif ma1[i_rej] < ma2[i_rej] < ma3[i_rej]:
                        d1 = abs(bits["h"] - ma1[i_rej]) if ma1[i_rej] is not None else float("inf")
                        d2 = abs(bits["h"] - ma2[i_rej]) if ma2[i_rej] is not None else float("inf")
                        dist_high = min(d1, d2)
                        rngs = np.array([c["high"] - c["low"] for c in candles], dtype=float)
                        atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
                        pattern_ok = bits["is_doji"] or bits["pin_high"] or bits["engulf_bear"]
                        if dist_high <= 0.25 * atr and pattern_ok:
                            direction = "SELL"
                            reason = "MA resistance rejection"
            except Exception:
                pass

            score, details = compute_score_for_rejection(candles, i_rej, ma1, ma2, ma3, tf)
            pattern_name = "Doji" if bits["is_doji"] else ("PinLow" if bits["pin_low"] else ("PinHigh" if bits["pin_high"] else ("EngulfBull" if bits["engulf_bull"] else ("EngulfBear" if bits["engulf_bear"] else "None"))))

            per_tf_info[tf] = {
                "direction": direction,
                "reason": reason or f"Rejected({pattern_name})",
                "score": score,
                "pattern": pattern_name,
                "bias": bias,
                "candles": candles,
                "i_rej": i_rej
            }

            # include candidate (accepted or not) for selection by score
            candidates.append({"tf": tf, "score": score, "direction": direction or "REJECTED", "pattern": pattern_name, "bias": bias, "candles": candles, "i_rej": i_rej, "reason": per_tf_info[tf]["reason"]})

        # pick highest-scoring candidate for chart (tie-break: prefer higher TF weight, then lower TF if equal)
        if not candidates:
            log(symbol, "no candidates")
            continue

        # sort by (score desc, tf weight desc (900>600>300), then tf asc to prefer lower TF on tie)
        def tf_weight(t): return {300:1, 600:2, 900:3}.get(t,1)
        candidates_sorted = sorted(candidates, key=lambda x: (x["score"], tf_weight(x["tf"]), -x["tf"]), reverse=True)
        chosen = candidates_sorted[0]

        tf_ch = chosen["tf"]
        chosen_info = per_tf_info[tf_ch]
        chosen_direction = chosen_info["direction"] or "REJECTED"
        chosen_score = chosen_info["score"]
        chosen_pattern = chosen_info["pattern"]
        chosen_reason = chosen_info["reason"]
        i_rej_ch = chosen_info["i_rej"]
        candles_ch = chosen_info["candles"]

        # build single caption containing both main and other TFs
        caption = build_caption_for_asset(symbol, {"tf": tf_ch, "direction": chosen_direction, "score": chosen_score, "pattern": chosen_pattern, "reason": chosen_reason}, per_tf_info)

        # send single message (one chart) per asset
        if signals_sent < MAX_SIGNALS_PER_RUN and can_send(symbol, chosen_direction):
            chart_path = None
            try:
                ma1c, ma2c, ma3c = compute_mas_for_chart(candles_ch)
                chart_path = make_chart(candles_ch, ma1c, ma2c, ma3c, i_rej_ch, caption_title=f"{symbol} | {tf_ch//60}m | {chosen_direction} s={chosen_score}", symbol=symbol, tf=tf_ch, last_n=LAST_N_CHART, pad=PAD_CANDLES)
            except Exception as e:
                log("chart creation error", e)
                chart_path = None

            sent_ok = False
            try:
                if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    ok, info = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                    sent_ok = ok
                else:
                    # fallback plain text
                    send_single_timeframe_signal(symbol, tf_ch, chosen_direction, caption)
                    sent_ok = True
            except Exception as e:
                log("send error", e)
                sent_ok = False

            if chart_path:
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

            if sent_ok:
                mark_sent(symbol, chosen_direction)
                signals_sent += 1
                reports.append(f"{symbol} -> SENT {chosen_direction} @{tf_ch//60}m s={chosen_score}")
            else:
                reports.append(f"{symbol} -> FAILED SEND {chosen_direction} @{tf_ch//60}m")
        else:
            reports.append(f"{symbol} -> SKIPPED (cooldown/limit)")

        # continue to next asset
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            log("max signals reached")
            break

    # heartbeat if nothing sent
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
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    send_heartbeat(ASSETS)
                with open(HEART_FILE, "w") as f:
                    json.dump({"ts": now}, f)
        except Exception:
            log("heartbeat error", traceback.format_exc())

    save_cache()
    log("Run complete. Reports:", reports)

# Entry
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

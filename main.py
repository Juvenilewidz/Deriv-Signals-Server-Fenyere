# main.py
import os
import json
import math
import tempfile
import time
import websocket  # pip install websocket-client
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# try to reuse bot helpers
try:
    from bot import (
        send_single_timeframe_signal,
        send_strong_signal,
        send_telegram_message,
        send_telegram_photo,
    )
except Exception:
    # offline fallback stubs (so the script still runs if bot isn't present)
    send_single_timeframe_signal = None
    send_strong_signal = None
    send_telegram_message = None
    send_telegram_photo = None

# ---------------- Env / constants ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Operational env
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes", "on")
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))  # default 2 hours
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN_SECS", "300"))  # avoid repeat alerts in secs

# Assets requested by user (volatility indices + 1s variants)
ASSETS = ["V10", "V50", "V75", "1HZ75V", "1HZ100V", "1HZ150V"]

# Timeframes (seconds): 5m, 10m, 15m as requested
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = 240  # more history for charting / smoothing

# Chart settings
CHART_HISTORY = 120   # number of candles shown (will be adjusted if less available)
CHART_PADDING_CANDLES = 10  # extra candles to the right
FIGSIZE = (10, 3.5)
DPI = 150

# emoji / icons
EMOJI_ACCEPT = "✅"
EMOJI_REJECT = "❌"
EMOJI_HEART = "💓"
EMOJI_CLOCK = "⏰"
EMOJI_CHART = "📊"

# cache files
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "deriv_bot_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "deriv_bot_last_heartbeat.json")

# in-memory
last_sent_signal_by_symbol: Dict[str, Tuple[str, int]] = {}  # symbol -> (direction, ts)
LAST_ALERT: Dict[Tuple[str,int], int] = {}  # ((symbol, tf), ts)

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{ts}]", *args, **kwargs)

# ==========================
# Math / MAs (user-specified)
# MA1: SMMA(9) on HLC3 (typical price)
# MA2: SMMA(19) on Close
# MA3: SMA(25) on MA2 values
# ==========================
def sma(series: List[float], period: int) -> List[Optional[float]]:
    out = [None] * (period - 1)
    if len(series) < period:
        out.extend([None] * (len(series) - len(out)))
        return out
    for i in range(period - 1, len(series)):
        window = series[i - period + 1:i + 1]
        out.append(sum(window) / period)
    return out

def smma(series: List[float], period: int) -> List[Optional[float]]:
    if not series:
        return []
    n = len(series)
    if n < period:
        return [None] * n
    seed = sum(series[:period]) / period
    out: List[Optional[float]] = [None] * (period - 1)
    out.append(seed)
    prev = seed
    for i in range(period, n):
        val = (prev * (period - 1) + series[i]) / period
        out.append(val)
        prev = val
    return out

def typical_price(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0

# ==========================
# Candle helpers
# ==========================
def candle_body(o: float, c: float) -> float:
    return abs(c - o)

def candle_range(h: float, l: float) -> float:
    return max(1e-12, h - l)

# ==========================
# Fetch candles (Deriv)
# ==========================
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict[str, Any]]:
    """
    Fetch latest candles including live candle (subscribe).
    Returns list of dicts with epoch, open, high, low, close.
    """
    ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
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
                "close": float(c["close"]),
            })

        # attempt one update tick for live candle
        try:
            update = json.loads(ws.recv())
            if "candles" in update and update["candles"]:
                live_c = update["candles"][-1]
                out[-1] = {
                    "epoch": int(live_c["epoch"]),
                    "open": float(live_c["open"]),
                    "high": float(live_c["high"]),
                    "low": float(live_c["low"]),
                    "close": float(live_c["close"]),
                }
        except Exception:
            pass

        return out
    finally:
        try:
            ws.close()
        except Exception:
            pass

# ==========================
# Compute MA arrays using USER parameters
# ==========================
def compute_mas(candles: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = np.array([c["close"] for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    hlc3   = (highs + lows + closes) / 3.0

    # MA1: SMMA(9) on HLC3
    ma1_list = smma(list(hlc3), 9)
    ma1 = np.array([np.nan if v is None else float(v) for v in ma1_list], dtype=float)

    # MA2: SMMA(19) on Close
    ma2_list = smma(list(closes), 19)
    ma2 = np.array([np.nan if v is None else float(v) for v in ma2_list], dtype=float)

    # MA3: SMA(25) on MA2 values (only where MA2 is finite)
    # compute moving average ignoring leading NaNs
    ma2_vals = [v for v in ma2_list if v is not None]
    if len(ma2_vals) >= 25:
        ma3_raw = sma(ma2_vals, 25)
        # re-insert None values to align with ma2_list
        ma3: List[Optional[float]] = []
        j = 0
        for v in ma2_list:
            if v is None:
                ma3.append(None)
            else:
                ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
                j += 1
        ma3_arr = np.array([np.nan if v is None else float(v) for v in ma3], dtype=float)
    else:
        ma3_arr = np.full_like(ma2, np.nan, dtype=float)

    return ma1, ma2, ma3_arr

# ==========================
# Candlestick chart builder
# ==========================
def make_chart(candles: List[Dict[str, float]],
               ma1: np.ndarray, ma2: np.ndarray, ma3: np.ndarray,
               i_rej: int, direction: Optional[str], symbol: str,
               tf: int, reason: str,
               last_n: int = 120,
               padding: int = CHART_PADDING_CANDLES) -> str:
    """
    Draw candlestick chart + MA overlays and return filepath to PNG.
    - last_n: number of candles to show
    - padding: candles of empty space on the right
    """
    n = len(candles)
    if n == 0:
        raise ValueError("no candles to chart")

    last_n = min(last_n, n)
    start = max(0, n - last_n)
    window = candles[start:n]
    idxs = list(range(len(window)))
    opens = [c["open"] for c in window]
    highs = [c["high"] for c in window]
    lows  = [c["low"] for c in window]
    closes = [c["close"] for c in window]
    times = [datetime.utcfromtimestamp(c["epoch"]) for c in window]

    # slice MA arrays to the same window
    ma_idx_slice = slice(start, n)
    ma1_win = ma1[ma_idx_slice]
    ma2_win = ma2[ma_idx_slice]
    ma3_win = ma3[ma_idx_slice]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    # candlestick drawing: rectangle body + wicks
    candle_width = 0.6
    wick_width = 0.6 * 0.3

    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = "green" if c >= o else "red"
        lower = min(o, c)
        height = abs(c - o)
        # body
        rect = Rectangle((i - candle_width/2, lower), candle_width, height, color=color, alpha=0.85)
        ax.add_patch(rect)
        # wick
        ax.plot([i, i], [l, h], color="black", linewidth=0.6)

    # plot MA lines (ignore NaNs)
    ax.plot(np.arange(len(ma1_win)), ma1_win, label="MA1 (SMMA9 HLC3)", linewidth=1.2, color="#1f77b4")
    ax.plot(np.arange(len(ma2_win)), ma2_win, label="MA2 (SMMA19 Close)", linewidth=1.2, color="#ff7f0e")
    ax.plot(np.arange(len(ma3_win)), ma3_win, label="MA3 (SMA25 of MA2)", linewidth=1.2, color="#2ca02c")

    # highlight rejection candle if inside window
    if i_rej is not None and start <= i_rej < n:
        i_local = i_rej - start
        # mark with triangle
        ax.scatter([i_local], [closes[i_local]], s=80, marker='v' if direction == "SELL" else '^', color="black", zorder=5)

    # aesthetics
    ax.set_xlim(-1, len(window) - 1 + padding)
    y_min = min(lows)
    y_max = max(highs)
    y_pad = 0.06 * (y_max - y_min) if (y_max - y_min) > 0 else 1.0
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # x axis as date ticks
    xticks = np.linspace(0, max(0, len(window)-1), min(6, len(window))).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i].strftime("%m-%d %H:%M") for i in xticks], rotation=30, fontsize=8)

    ax.set_title(f"{symbol} | {tf//60}m | {direction or 'REJECTED'}", fontsize=10)
    ax.set_ylabel("Price")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)

    # save
    fname = os.path.join(TMPDIR, f"chart_{symbol}_{tf}_{int(time.time())}.png")
    plt.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# ==========================
# Signal detection core (simplified & consistent with MA defs)
# Returns (direction, reason)
# ==========================
def signal_for_timeframe(candles: List[Dict[str, float]], tf: int, i_rej: int, i_con: int) -> Tuple[Optional[str], str]:
    # minimal history
    if not candles or len(candles) < 30:
        return None, "no candles or insufficient"

    ma1, ma2, ma3 = compute_mas(candles)

    # bounds
    n = len(candles)
    try:
        if i_rej is None or i_con is None:
            return None, "indexes missing"
        # ensure ints
        i_rej = int(i_rej)
        i_con = int(i_con)
    except Exception:
        return None, "invalid indices"

    # ensure indices valid
    if i_rej < 0 or i_con < 0 or i_rej >= n or i_con >= n:
        return None, "index out of range"

    # compute ATR approximation
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows  = np.array([c["low"]  for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))

    # simple checks: trend (MA stacking)
    def is_trend_up(i):
        return not np.isnan(ma1[i]) and not np.isnan(ma2[i]) and not np.isnan(ma3[i]) and (ma1[i] > ma2[i] > ma3[i])
    def is_trend_down(i):
        return not np.isnan(ma1[i]) and not np.isnan(ma2[i]) and not np.isnan(ma3[i]) and (ma1[i] < ma2[i] < ma3[i])

    uptr = is_trend_up(i_rej)
    downtr = is_trend_down(i_rej)

    # candles
    def bits(i):
        o = float(candles[i]["open"]); c = float(candles[i]["close"]); h = float(candles[i]["high"]); l = float(candles[i]["low"])
        body = abs(c - o)
        r = max(h - l, 1e-12)
        upper = h - max(o, c)
        lower = min(o, c) - l
        is_bull = c > o
        is_bear = c < o
        is_doji = body <= 0.35 * r
        pin_low = (lower >= 0.6 * r) and (lower > upper)
        pin_high = (upper >= 0.6 * r) and (upper > lower)
        engulf_bull = False
        engulf_bear = False
        if i > 0:
            prev_o = float(candles[i-1]["open"]); prev_c = float(candles[i-1]["close"])
            if (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o): engulf_bull = True
            if (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o): engulf_bear = True
        return {"o":o, "h":h, "l":l, "c":c, "body":body, "range":r, "upper":upper, "lower":lower,
                "is_bull":is_bull, "is_bear":is_bear, "is_doji":is_doji, "pin_low":pin_low, "pin_high":pin_high,
                "engulf_bull":engulf_bull, "engulf_bear":engulf_bear}

    rej = bits(i_rej)
    con = bits(i_con)

    # reject oversized candles
    OVERSIZED_MULT = 1.8
    if rej["body"] > OVERSIZED_MULT * atr or rej["range"] > OVERSIZED_MULT * atr:
        return None, "rejection candle oversized"

    # Buy path
    if uptr:
        # require rejection pattern near MA1/MA2
        # choose closer MA to candle low
        zone1 = ma1[i_rej] if not np.isnan(ma1[i_rej]) else None
        zone2 = ma2[i_rej] if not np.isnan(ma2[i_rej]) else None
        zone, ma_name = (zone1, "MA1") if (zone1 is not None and zone2 is not None and abs(rej["l"] - zone1) <= abs(rej["l"] - zone2)) else (zone2, "MA2")
        if zone is None:
            return None, "no MA zone"
        buffer_ = max(0.25 * atr, 0.0)
        near_zone = abs(rej["l"] - zone) <= buffer_
        close_side = (rej["c"] >= (zone - buffer_))
        pattern_ok = bool(rej["is_doji"] or rej["pin_low"] or rej["engulf_bull"])
        if near_zone and close_side and pattern_ok:
            return "BUY", f"{ma_name} support rejection"
        return None, "buy rejected: no valid buy rejection (pivot/pullback failed)"

    # Sell path
    if downtr:
        zone1 = ma1[i_rej] if not np.isnan(ma1[i_rej]) else None
        zone2 = ma2[i_rej] if not np.isnan(ma2[i_rej]) else None
        zone, ma_name = (zone1, "MA1") if (zone1 is not None and zone2 is not None and abs(rej["h"] - zone1) <= abs(rej["h"] - zone2)) else (zone2, "MA2")
        if zone is None:
            return None, "no MA zone"
        buffer_ = max(0.25 * atr, 0.0)
        near_zone = abs(rej["h"] - zone) <= buffer_
        close_side = (rej["c"] <= (zone + buffer_))
        pattern_ok = bool(rej["is_doji"] or rej["pin_high"] or rej["engulf_bear"])
        if near_zone and close_side and pattern_ok:
            return "SELL", f"{ma_name} resistance rejection"
        return None, "sell rejected: no valid sell rejection (pivot/pullback failed)"

    return None, "no clear trend / signal"

# ==========================
# Persistence helpers (cache)
# ==========================
def load_persistent_cache():
    global last_sent_signal_by_symbol
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_signal_by_symbol = json.load(f)
            # convert keys properly (json saved tuples as lists)
            for k, v in list(last_sent_signal_by_symbol.items()):
                last_sent_signal_by_symbol[k] = tuple(v)
        else:
            last_sent_signal_by_symbol = {}
        log("Loaded cache:", last_sent_signal_by_symbol)
    except Exception as e:
        log("Failed loading cache:", e)
        last_sent_signal_by_symbol = {}

def save_persistent_cache():
    try:
        serial = {k:list(v) for k,v in last_sent_signal_by_symbol.items()}
        with open(CACHE_FILE, "w") as f:
            json.dump(serial, f)
        log("Saved cache to", CACHE_FILE)
    except Exception as e:
        log("Failed saving cache:", e)

def last_heartbeat_time() -> float:
    try:
        if os.path.exists(HEART_FILE):
            with open(HEART_FILE, "r") as f:
                j = json.load(f)
                return float(j.get("last", 0))
    except Exception:
        pass
    return 0.0

def set_heartbeat_time(ts: float):
    try:
        with open(HEART_FILE, "w") as f:
            json.dump({"last": ts}, f)
    except Exception:
        pass

# ==========================
# Orchestration
# ==========================
def analyze_and_notify():
    checked_assets = []
    any_sent_this_run = False
    report_lines = []

    load_persistent_cache()

    for symbol in ASSETS:
        checked_assets.append(symbol)
        per_asset_report = []
        results: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles:
                results[tf] = (None, "no candles")
                continue

            if len(candles) >= 2:
                i_rej = len(candles) - 2
                i_con = len(candles) - 1
            else:
                i_rej = i_con = None

            direction, reason = signal_for_timeframe(candles, tf, i_rej, i_con)
            results[tf] = (direction, reason)

        # choose main TF (prefer lower timeframes first)
        main_tf = TIMEFRAMES[0]  # 5m preferred (user said prefer lower Tfs)
        main_sig, main_reason = results.get(main_tf, (None, None))

        # Build single message per asset: if any accepted in main -> send accepted; else send rejected for main TF with other TF details below
        caption_lines = []
        # Build "Other TFs" lines
        other_lines = []
        for tf in TIMEFRAMES:
            sig, rsn = results.get(tf, (None, None))
            tf_label = f"{tf//60}m"
            if sig:
                other_lines.append(f"{tf_label} -> {sig} ({rsn})")
            else:
                other_lines.append(f"{tf_label} -> Rejected ({rsn})")

        # prepare chart & send for main_tf (even rejected we send chart)
        candles_main = fetch_candles(symbol, main_tf, CANDLES_N)
        if candles_main:
            ma1_arr, ma2_arr, ma3_arr = compute_mas(candles_main)
            i_rej = len(candles_main) - 2 if len(candles_main) >= 2 else None
            try:
                chart_path = make_chart(candles_main, ma1_arr, ma2_arr, ma3_arr, i_rej, main_sig, symbol, main_tf, main_reason, last_n=CHART_HISTORY, padding=CHART_PADDING_CANDLES)
            except Exception as e:
                log("Chart build failed:", e)
                chart_path = None
        else:
            chart_path = None

        # build caption
        if main_sig:
            head = f"{EMOJI_ACCEPT} {symbol} | {main_tf//60}m | {main_sig} | Reason: {main_reason}"
        else:
            head = f"{EMOJI_REJECT} {symbol} | {main_tf//60}m | REJECTED | Reason: {main_reason}"

        caption = head + "\n\nOther TFs:\n" + "\n".join(other_lines)

        # dedupe: don't send identical signal repeatedly within cooldown
        key = (symbol, main_tf)
        last_ts = LAST_ALERT.get(key, 0)
        now_ts = int(time.time())
        allow_send = (now_ts - last_ts) > ALERT_COOLDOWN

        sent_ok = False
        sent_msg = ""
        if allow_send:
            if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and send_telegram_photo:
                try:
                    ok = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                    sent_ok = True
                    sent_msg = "sent photo"
                except Exception as e:
                    sent_ok = False
                    sent_msg = str(e)
            else:
                # fallback to text helper
                if send_single_timeframe_signal:
                    try:
                        send_single_timeframe_signal(symbol, main_tf, main_sig or "REJECTED", main_reason)
                        sent_ok = True
                        sent_msg = "sent via helper"
                    except Exception as e:
                        sent_ok = False
                        sent_msg = str(e)
                else:
                    # as final fallback, try direct telegram text function if available
                    if send_telegram_message:
                        try:
                            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                            sent_ok = True
                            sent_msg = "sent text direct"
                        except Exception as e:
                            sent_ok = False
                            sent_msg = str(e)

            if sent_ok:
                LAST_ALERT[key] = now_ts
                any_sent_this_run = True
                report_lines.append(f"{symbol}: SENT {main_sig or 'REJECTED'} @ {main_tf//60}m -> {main_reason}")
            else:
                report_lines.append(f"{symbol}: failed send -> {sent_msg}")
        else:
            report_lines.append(f"{symbol}: suppressed duplicate (cooldown)")

    save_persistent_cache()

    # heartbeat if nothing sent
    if not any_sent_this_run:
        last_hb = last_heartbeat_time()
        now = time.time()
        if now - last_hb >= HEARTBEAT_INTERVAL_HOURS * 3600:
            # build heartbeat short message
            try:
                body = (
                    "🤖 Bot heartbeat – alive\n"
                    "⏰ No signals right now.\n"
                    f"📊 Checked: {', '.join(checked_assets)}\n"
                    f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                if send_telegram_message:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, body)
                set_heartbeat_time(now)
                log("Heartbeat sent")
            except Exception as e:
                log("Heartbeat send failed:", e)
        else:
            log("Heartbeat suppressed (interval not reached)")
    else:
        log("Signals were sent this run; skipping heartbeat.")

# -------------- run logic --------------
if __name__ == "__main__":
    try:
        # timezone sleep window - Zimbabwe example 21:00-06:00 (UTC offset varies; we use UTC hours)
        # You can change this behavior by editing these lines if needed.
        now_hour_utc = datetime.utcnow().hour
        sleep_start = 21  # 21:00 local (user wanted Zimbabwe TZ earlier - adapt in env if necessary)
        sleep_end = 6
        # NOTE: we are using UTC hour; if you want Zimbabwe local, compute with pytz and check local hour.
        if (now_hour_utc >= sleep_start) or (now_hour_utc < sleep_end):
            log("Bot sleeping (night hours). Exiting run.")
        else:
            analyze_and_notify()
    except Exception as e:
        log("Fatal:", e)
        # try to notify
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and send_telegram_message:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

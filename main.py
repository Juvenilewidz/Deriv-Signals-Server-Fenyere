# main.py
"""
AI Forex trading helper - main runner

Features:
- Fetch candles from Deriv via websocket (live forming candle)
- Compute MAs and detect rejection setups (fires on rejection candle close)
- Send signals & rejection charts via bot.py helpers
- Charting: shows last_n candles (default 100) and +10 candles padding on right
- Dispatch: prefer lower TF, send at most MAX_SIGNALS_PER_RUN per run
- Rejected setups: one per symbol per run (lowest TF preferred)
- Simple persistent cache to avoid duplicate alerts across runs
"""

import os
import json
import time
import tempfile
from typing import List, Dict, Tuple, Optional
import math

# third-party
import requests
import websocket  # pip install websocket-client
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

# --- bot helpers (send) ---
try:
    from bot import (
        send_single_timeframe_signal,
        send_strong_signal,
        send_rejection_with_chart,
        send_telegram_message,
        send_heartbeat,
    )
except Exception:
    # graceful fallbacks if bot isn't available in this env
    send_single_timeframe_signal = None
    send_strong_signal = None
    send_rejection_with_chart = None
    send_telegram_message = None
    send_heartbeat = None

# -------- Env / constants --------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes", "on")

# Trading configuration (user requested)
ASSETS = ["R_10", "R_50", "R_75", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = 240  # history to fetch (keep ample so MA stabilize)

# Dispatch controls
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "1"))  # at most this many accepted signals per run
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "900"))  # 15 minutes default cooldown

# Heartbeat (disabled by default to avoid spam)
HEARTBEAT_ENABLED = os.getenv("HEARTBEAT_ENABLED", "0") in ("1", "true", "True", "yes")
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))

# files for simple persistence across runs
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "ai_forex_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "ai_forex_last_heartbeat.json")

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# --------------------------
# Persistence: last sent alerts to avoid duplicates
# --------------------------
last_sent_cache: Dict[str, Dict] = {}  # symbol -> {"direction":str, "tf":int, "ts":int}

def load_cache():
    global last_sent_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_cache = json.load(f)
        else:
            last_sent_cache = {}
    except Exception as e:
        log("load_cache error:", e)
        last_sent_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
    except Exception as e:
        log("save_cache error:", e)

def can_send_alert(symbol: str, direction: str, tf: int) -> bool:
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec:
        if rec.get("direction") == direction and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
            return False
    return True

def mark_sent(symbol: str, direction: str, tf: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time())}
    save_cache()

# --------------------------
# Fetch candles (Deriv)
# --------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Connects to Deriv websocket and requests candles (including live forming).
    Returns list of candles ordered oldest..newest.
    """
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

        # optionally grab one live update to refresh last forming candle
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

# --------------------------
# MAs
# --------------------------
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

def compute_mas(candles: List[Dict]):
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    ma1 = smma(hlc3, 9)
    ma2 = smma(closes, 19)
    # SMA(25) on ma2 values - compute over ma2 finite values
    ma2_vals = [v for v in ma2 if v is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None] * len(ma2_vals)
    # expand ma3 to align with ma2 indices
    ma3 = []
    j = 0
    for v in ma2:
        if v is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

# --------------------------
# Signal logic (fires on rejection candle close)
# --------------------------
def signal_for_timeframe(candles: List[Dict], granularity: int, i_rej: int, i_con: int) -> Tuple[Optional[str], str]:
    """
    Returns ("BUY"/"SELL", reason) or (None, short_reason).
    Fires on the rejection candle (i_rej) close, not waiting further confirmation.
    Basic dynamic S/R rejection logic tuned to your description.
    """

    import numpy as np
    if not candles or len(candles) < 50:
        return None, "insufficient history"

    opens = np.array([c["open"] for c in candles], dtype=float)
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows  = np.array([c["low"] for c in candles], dtype=float)
    closes= np.array([c["close"] for c in candles], dtype=float)

    ma1, ma2, ma3 = compute_mas(candles)

    # ensure valid MA at i_rej
    try:
        ma1_r = float(ma1[i_rej]); ma2_r = float(ma2[i_rej]); ma3_r = float(ma3[i_rej])
    except Exception:
        return None, "invalid MA data"

    # ATR-like measure: average range
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    # trending stacking with wiggle
    wiggle = 0.25 * atr
    stacked_up = (ma1_r >= ma2_r - wiggle) and (ma2_r >= ma3_r - wiggle)
    stacked_down = (ma1_r <= ma2_r + wiggle) and (ma2_r <= ma3_r + wiggle)

    # candle helpers
    def candle_bits(idx):
        o = float(opens[idx]); h = float(highs[idx]); l = float(lows[idx]); c = float(closes[idx])
        body = abs(c - o); r = max(h - l, 1e-12)
        upper = h - max(o, c); lower = min(o, c) - l
        is_doji = body <= 0.35 * r
        pin_low = (lower >= 0.2 * body) and (lower > upper)
        pin_high = (upper >= 0.2 * body) and (upper > lower)
        engulf_bull = False; engulf_bear = False
        if idx > 0:
            po = float(opens[idx-1]); pc = float(closes[idx-1])
            if pc < po and c > o and o <= pc and c >= po:
                engulf_bull = True
            if pc > po and c < o and o >= pc and c <= po:
                engulf_bear = True
        return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
                "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
                "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

    rej = candle_bits(i_rej)
    # pattern and proximity checks
    def near_zone_buy(idx):
        c = candle_bits(idx)
        d1 = abs(c["l"] - ma1[idx])
        d2 = abs(c["l"] - ma2[idx])
        zone = ma1[idx] if d1 <= d2 else ma2[idx]
        return abs(c["l"] - zone) <= 0.25 * atr, zone

    def near_zone_sell(idx):
        c = candle_bits(idx)
        d1 = abs(c["h"] - ma1[idx])
        d2 = abs(c["h"] - ma2[idx])
        zone = ma1[idx] if d1 <= d2 else ma2[idx]
        return abs(c["h"] - zone) <= 0.25 * atr, zone

    # buy
    if stacked_up:
        near, zone = near_zone_buy(i_rej)
        pattern_ok = rej["is_doji"] or rej["pin_low"] or rej["engulf_bull"]
        if near and pattern_ok and (closes[-1] > ma3_r + tiny):
            return "BUY", "MA support rejection"
        return None, "buy rejected: no valid rejection"

    # sell
    if stacked_down:
        near, zone = near_zone_sell(i_rej)
        pattern_ok = rej["is_doji"] or rej["pin_high"] or rej["engulf_bear"]
        if near and pattern_ok and (closes[-1] < ma3_r - tiny):
            return "SELL", "MA resistance rejection"
        return None, "sell rejected: no valid rejection"

    return None, "no clear trend"

# --------------------------
# Make chart (candlesticks), zoomed: last_n default 100, right padding 10 bars
# --------------------------
def make_chart(candles, ma1, ma2, ma3, i_rej_idx, direction, symbol, tf, reason):
    """
    Create a candlestick chart PNG and return the path.
    - candles: list of dicts with 'epoch','open','high','low','close'
    - ma1/ma2/ma3: lists/arrays (same length)
    - i_rej_idx: index (into candles) of rejection candle (0..len-1)
    - direction: 'BUY' or 'SELL' or 'REJECTED'
    This implementation uses only matplotlib (no extra deps).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import tempfile
    import math
    from datetime import datetime

    # prepare arrays
    times = [datetime.utcfromtimestamp(c["epoch"]) for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows  = [c["low"] for c in candles]
    closes= [c["close"] for c in candles]

    x = mdates.date2num(times)
    # candle width relative to time step (small so candles appear slim)
    if len(x) >= 2:
        width = (x[1] - x[0]) * 0.5
    else:
        width = 0.02

    # figure styling: wide and short for long history look
    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    ax.set_facecolor('white')

    # draw candlesticks (wick + body)
    for xi, o, h, l, c in zip(x, opens, highs, lows, closes):
        color = '#2ca02c' if c >= o else '#d62728'  # green/red
        # wick
        ax.vlines(xi, l, h, linewidth=0.6, color='black', zorder=1)
        # body as rectangle
        lower = min(o, c)
        height = max(1e-8, abs(c - o))
        rect = Rectangle((xi - width*0.5, lower), width*0.9, height,
                         facecolor=color, edgecolor='black', linewidth=0.4, zorder=2, alpha=0.9)
        ax.add_patch(rect)

    # helper to plot MA arrays (skip None / nan values)
    def _plot_ma(arr, label, color, lw=1.2):
        safe_y = []
        for v in arr:
            if v is None:
                safe_y.append(float('nan'))
            else:
                # handle numpy nan
                try:
                    if math.isnan(v):
                        safe_y.append(float('nan'))
                    else:
                        safe_y.append(v)
                except Exception:
                    safe_y.append(v)
        ax.plot(x, safe_y, label=label, color=color, linewidth=lw, zorder=3)

    try:
        _plot_ma(ma1, "MA1 (SMMA9 HLC/3)", color='#1f77b4')
        _plot_ma(ma2, "MA2 (SMMA19 Close)", color='#ff7f0e')
        _plot_ma(ma3, "MA3 (SMA25 of MA2)", color='#2ca02c')
    except Exception:
        pass

    # mark rejection candle with a marker
    if isinstance(i_rej_idx, int) and 0 <= i_rej_idx < len(x):
        mark_y = closes[i_rej_idx]
        color = 'red' if direction == 'SELL' else 'green'
        ax.scatter([x[i_rej_idx]], [mark_y], s=60, marker='v' if direction == 'SELL' else '^',
                   color=color, zorder=6, edgecolors='black')

    # formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=25)
    ax.set_ylabel('Price')
    ax.set_title(f"{symbol} | {int(tf/60)}m | {direction or 'REJECTED'}", fontsize=10)

    # set x-limits to include right-side padding (10 candles)
    if len(x) >= 2:
        dx = (x[-1] - x[0]) / max(1, len(x)-1)
        ax.set_xlim(x[0] - dx*1.5, x[-1] + dx*10)

    plt.tight_layout(pad=1.0)

    # write temporary png
    tmpf = tempfile.NamedTemporaryFile(prefix="chart_", suffix=".png", delete=False)
    fig.savefig(tmpf.name, bbox_inches='tight')
    plt.close(fig)
    return tmpf.name
# --------------------------
# Orchestrator: main runner
# --------------------------
def analyze_and_notify():
    load_cache()
    signals_sent = 0
    last_rejections: Dict[Tuple[str,int], Tuple[List[Dict], str]] = {}

    for symbol in ASSETS:
        log("Scanning", symbol)
        # collect per-symbol results
        per_tf_results: Dict[int, Tuple[Optional[str], str, List[Dict]]] = {}
        for tf in sorted(TIMEFRAMES):
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 3:
                per_tf_results[tf] = (None, "no candles", candles)
                continue

            i_rej = len(candles) - 2
            i_con = len(candles) - 1
            direction, reason = signal_for_timeframe(candles, tf, i_rej, i_con)
            per_tf_results[tf] = (direction, reason, candles)
            log(f" {symbol} {tf//60}m -> {direction} | {reason}")

            if direction is None and reason:
                last_rejections[(symbol, tf)] = (candles, reason)

        # prefer lower timeframe accepted signal
        chosen = None
        for tf in sorted(TIMEFRAMES):
            dirc, rsn, cnd = per_tf_results.get(tf, (None, None, None))
            if dirc:
                chosen = (tf, dirc, rsn, cnd)
                break

        if chosen and signals_sent < MAX_SIGNALS_PER_RUN:
            tf_chosen, direction_chosen, reason_chosen, chosen_candles = chosen
            if can_send_alert(symbol, direction_chosen, tf_chosen):
                ma1, ma2, ma3 = compute_mas(chosen_candles)
                i_rej_chart = len(chosen_candles) - 2
                chart_path = make_chart(chosen_candles, ma1, ma2, ma3, i_rej_chart,
                                        direction_chosen, symbol, tf_chosen, last_n=100)
                caption = f"{symbol} | {tf_chosen//60}m | {direction_chosen}\n{reason_chosen}"
                ok = False
                # prefer bot helper if available
                if chart_path and send_single_timeframe_signal:
                    try:
                        send_single_timeframe_signal(symbol, tf_chosen, direction_chosen, reason_chosen, chart_path)
                        ok = True
                    except Exception as e:
                        log("send_single_timeframe_signal error:", e)
                        ok = False
                else:
                    # fallback direct telegram photo/text
                    if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        try:
                            send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                            ok = True
                        except Exception as e:
                            log("send photo fallback error:", e)
                            ok = False
                    else:
                        try:
                            if send_telegram_message:
                                send_telegram_message(caption)
                                ok = True
                            else:
                                send_telegram_text_direct(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                                ok = True
                        except Exception as e:
                            log("send text fallback error:", e)
                            ok = False

                if ok:
                    signals_sent += 1
                    mark_sent(symbol, direction_chosen, tf_chosen)
                    log("SENT signal", symbol, tf_chosen, direction_chosen)
                try:
                    if chart_path:
                        os.unlink(chart_path)
                except Exception:
                    pass

        # continue to next symbol

    # --- send rejections: one per symbol, prefer lowest TF ---
    handled = set()
    for (sym, tf) in sorted(last_rejections.keys(), key=lambda x: x[1]):  # sorted by tf ascending (lower first)
        if sym in handled:
            continue
        candles, reason = last_rejections[(sym, tf)]
        # skip if symbol already had an accepted signal this run (persisted mark)
        rec = last_sent_cache.get(sym)
        if rec and (int(time.time()) - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
            continue

        # prepare short caption + chart
        ma1, ma2, ma3 = compute_mas(candles)
        i_rej_chart = len(candles) - 2
        chart_path = make_chart(candles, ma1, ma2, ma3, i_rej_chart, "REJECTED", sym, tf, last_n=100)
        short_reason = str(reason).split(":")[-1].strip()
        caption = f"❌ Rejected\n{sym} | {tf//60}m\nReason: {short_reason}"

        sent_ok = False
        # use bot helper if available
        if chart_path and send_rejection_with_chart:
            try:
                send_rejection_with_chart(sym, tf, candles, short_reason)
                sent_ok = True
            except Exception as e:
                log("send_rejection_with_chart error:", e)
                sent_ok = False
        else:
            # fallback: send photo then caption text
            if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                    sent_ok = True
                except Exception as e:
                    log("fallback send rejection photo error:", e)
                    sent_ok = False
            else:
                try:
                    if send_telegram_message:
                        send_telegram_message(caption)
                        sent_ok = True
                    else:
                        send_telegram_text_direct(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                        sent_ok = True
                except Exception as e:
                    log("fallback send rejection text error:", e)
                    sent_ok = False

        if sent_ok:
            handled.add(sym)
            mark_sent(sym, f"REJECTED:{short_reason}", tf)
        try:
            if chart_path:
                os.unlink(chart_path)
        except Exception:
            pass

    # heartbeat (optional)
    if HEARTBEAT_ENABLED:
        last_hb = last_heartbeat_time()
        now = int(time.time())
        if now - last_hb >= int(HEARTBEAT_INTERVAL_HOURS * 3600):
            try:
                if send_heartbeat:
                    send_heartbeat(ASSETS)
            except Exception:
                pass
            set_heartbeat_time(now)

# --------------------------
# telegram helpers (fallbacks)
# --------------------------
def send_telegram_photo(token: str, chat_id: str, caption: str, filepath: str) -> Tuple[bool, str]:
    if not token or not chat_id or not filepath:
        return False, "missing args"
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(filepath, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=20)
        return (r.status_code == 200, r.text)
    except Exception as e:
        return False, str(e)

def send_telegram_text_direct(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "missing args"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
        return (r.status_code == 200, r.text)
    except Exception as e:
        return False, str(e)

# heartbeat persistence helpers
def last_heartbeat_time():
    try:
        if os.path.exists(HEART_FILE):
            with open(HEART_FILE, "r") as f:
                return int(json.load(f).get("last", 0))
    except Exception:
        pass
    return 0

def set_heartbeat_time(ts=None):
    try:
        t = int(ts or time.time())
        with open(HEART_FILE, "w") as f:
            json.dump({"last": t}, f)
    except Exception:
        pass

# --------------------------
# entry
# --------------------------
if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        log("Fatal error:", e)
        # send crash to telegram if possible
        try:
            if send_telegram_message:
                send_telegram_message(f"❌ Bot crashed: {e}")
            else:
                send_telegram_text_direct(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

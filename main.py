# main.py (updated)
import os
import json
import math
import time
import tempfile
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
import websocket  # pip install websocket-client
from datetime import datetime
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- try to reuse bot helpers ----------
try:
    from bot import (
        send_single_timeframe_signal,
        send_strong_signal,
        send_telegram_message,
    )
except Exception:
    send_single_timeframe_signal = None
    send_strong_signal = None
    send_telegram_message = None

# -------- Env / constants --------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Operational env
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes", "on")
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))

# =========================================
# Assets & Timeframes
ASSETS = ["R_10", "R_50", "R_75", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = 120

# cache files (persist across runs while runner exists)
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "deriv_bot_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "deriv_bot_last_heartbeat.json")

# in-memory cache used during run (initially loaded from file)
last_sent_signal_by_symbol: Dict[str, Tuple[str, str]] = {}
LAST_ALERT: Dict[Tuple, int] = {}
ALERT_COOLDOWN = 60  # seconds base

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC]", *args, **kwargs)

def load_persistent_cache():
    global last_sent_signal_by_symbol
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_signal_by_symbol = json.load(f)
                # ensure tuple values
                for k, v in list(last_sent_signal_by_symbol.items()):
                    if isinstance(v, list):
                        last_sent_signal_by_symbol[k] = tuple(v)
        else:
            last_sent_signal_by_symbol = {}
        log("Loaded cache:", last_sent_signal_by_symbol)
    except Exception as e:
        log("Failed loading cache:", e)
        last_sent_signal_by_symbol = {}

def save_persistent_cache():
    try:
        serial = {k: list(v) for k, v in last_sent_signal_by_symbol.items()}
        with open(CACHE_FILE, "w") as f:
            json.dump(serial, f)
        log("Saved cache to", CACHE_FILE)
    except Exception as e:
        log("Failed saving cache:", e)

def last_heartbeat_time():
    try:
        if os.path.exists(HEART_FILE):
            with open(HEART_FILE, "r") as f:
                data = json.load(f)
            return int(data.get("last", 0))
    except Exception as e:
        log("hb read err", e)
    return 0

def set_heartbeat_time(ts=None):
    try:
        t = int(ts or time.time())
        with open(HEART_FILE, "w") as f:
            json.dump({"last": t}, f)
        log("updated heartbeat file")
    except Exception as e:
        log("hb write err", e)

# ==========================
# Helpers: math & MAs
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
    n = len(series)
    if n == 0:
        return []
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
# Candle helpers & patterns
# ==========================
def candle_body(o: float, c: float) -> float:
    return abs(c - o)

def candle_range(h: float, l: float) -> float:
    return max(1e-12, h - l)

def upper_wick(h: float, o: float, c: float) -> float:
    return h - max(o, c)

def lower_wick(l: float, o: float, c: float) -> float:
    return min(o, c) - l

def is_doji(o: float, h: float, l: float, c: float, thresh: float = 0.25) -> bool:
    rng = candle_range(h, l)
    return candle_body(o, c) <= thresh * rng

# ==========================
# Deriv data fetch (candles)
# ==========================
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=12)
    except Exception as e:
        log("ws connect error:", e)
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
                "close": float(c["close"]),
            })

        # optional live update
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
        except:
            pass

# ==========================
# MAs & trend
# ==========================
def compute_mas(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    hlc3    = [typical_price(h, l, c) for h, l, c in zip(highs, lows, closes)]

    ma1 = smma(hlc3, 9)       # MA1: SMMA(9) on HLC/3
    ma2 = smma(closes, 19)    # MA2: SMMA(19) on Close
    ma2_vals = [x for x in ma2 if x is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None] * len(ma2_vals)
    ma3: List[Optional[float]] = []
    j = 0
    for i in range(len(ma2)):
        if ma2[i] is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

# ==========================
# Consolidation helper (strict)
# ==========================
def is_consolidating(candles, atr, *, lookback=12, range_mult=0.9, body_mult=0.45, overlap_pct=0.72):
    import numpy as np
    if not candles or len(candles) < lookback:
        return False, None

    highs = np.array([c["high"] for c in candles[-lookback:]], dtype=float)
    lows  = np.array([c["low"]  for c in candles[-lookback:]], dtype=float)
    opens = np.array([c["open"] for c in candles[-lookback:]], dtype=float)
    closes= np.array([c["close"] for c in candles[-lookback:]], dtype=float)

    highest = float(np.max(highs))
    lowest  = float(np.min(lows))
    total_range = highest - lowest
    avg_body = float(np.mean(np.abs(closes - opens)))

    median_h = float(np.median(highs))
    median_l = float(np.median(lows))
    overlapping = np.logical_and(highs <= median_h, lows >= median_l)
    overlap_ratio = float(np.sum(overlapping)) / float(len(overlapping))

    if total_range <= max(0.0, range_mult * atr) and avg_body <= body_mult * atr and overlap_ratio >= overlap_pct:
        reason = f"consolidation: range {total_range:.5f} <= {range_mult}*ATR and avg_body {avg_body:.5f} <= {body_mult}*ATR and overlap {overlap_ratio:.2f}"
        return True, reason
    return False, None

# ==========================
# Pivot helpers
# ==========================
def pivot_low(idx, lows, k=2):
    if idx - k < 0 or idx + k >= len(lows):
        return False
    return all(lows[idx] <= lows[idx - j] for j in range(1, k + 1)) and \
           all(lows[idx] <  lows[idx + j] for j in range(1, k + 1))

def pivot_high(idx, highs, k=2):
    if idx - k < 0 or idx + k >= len(highs):
        return False
    return all(highs[idx] >= highs[idx - j] for j in range(1, k + 1)) and \
           all(highs[idx] >  highs[idx + j] for j in range(1, k + 1))

def recent_pullback_bias(idx, opens, closes, bars=3):
    left = max(0, idx - bars)
    chg = sum((closes[i] - opens[i]) for i in range(left, idx))
    if chg < 0: return "down"
    if chg > 0: return "up"
    return None

# ==========================
# Chart rendering + send helper
# (same as you have; unchanged)
# ... make_chart() and send_telegram_photo() ...
def make_chart(candles: List[Dict], ma1: List[Optional[float]], ma2: List[Optional[float]],
               ma3: List[Optional[float]], i_rej: int, direction: str, symbol: str, tf: int,
               last_n: int = 60) -> Optional[str]:
    try:
        n = min(last_n, len(candles))
        offset = len(candles) - n
        subset = candles[-n:]

        times = [datetime.utcfromtimestamp(c["epoch"]) for c in subset]
        opens = [c["open"] for c in subset]
        highs = [c["high"] for c in subset]
        lows = [c["low"] for c in subset]
        closes = [c["close"] for c in subset]

        ma1_vals = [ma1[i] if (i is not None and i < len(ma1) and not (ma1[i] is None)) else np.nan for i in range(offset, offset + n)]
        ma2_vals = [ma2[i] if (i is not None and i < len(ma2) and not (ma2[i] is None)) else np.nan for i in range(offset, offset + n)]
        ma3_vals = [ma3[i] if (i is not None and i < len(ma3) and not (ma3[i] is None)) else np.nan for i in range(offset, offset + n)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"{symbol} {tf//60}m | {direction}")
        ax.set_ylabel("Price")

        width = 0.6
        for i in range(n):
            o = opens[i]; h = highs[i]; l = lows[i]; c = closes[i]
            x = i
            color = "green" if c >= o else "red"
            ax.plot([x, x], [l, h], color="black", linewidth=0.6, zorder=1)
            lower = min(o, c); height = abs(c - o)
            rect = patches.Rectangle((x - width/2, lower), width, height if height!=0 else 0.0000001,
                                     facecolor=color, edgecolor="black", linewidth=0.4, zorder=2)
            ax.add_patch(rect)

        xs = list(range(n))
        ax.plot(xs, ma1_vals, label="MA1 (SMMA9 HLC/3)", linewidth=1.25)
        ax.plot(xs, ma2_vals, label="MA2 (SMMA19 Close)", linewidth=1.0)
        ax.plot(xs, ma3_vals, label="MA3 (SMA25 of MA2)", linewidth=0.9)
        ax.legend(loc="upper left", fontsize="small")

        if offset <= i_rej < offset + n:
            idx = i_rej - offset
            price = closes[idx]
            marker_color = "red" if direction == "SELL" else "green"
            ax.scatter([idx], [price], marker="v" if direction == "SELL" else "^", color=marker_color, s=120, zorder=5)
            ax.plot([idx, idx], [price, price + (max(highs) - min(lows))*0.02], color=marker_color, linewidth=1.2)

        ax.set_xticks(xs[::max(1, n//8)])
        ax.set_xticklabels([t.strftime("%H:%M") for t in times[::max(1, n//8)]], rotation=30)

        ax.set_xlim(-1, n)
        ymin, ymax = min(lows), max(highs)
        pad = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad, ymax + pad)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.tight_layout()
        plt.savefig(tmp.name, dpi=140)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        log("make_chart error:", e)
        return None

def send_telegram_photo(token: str, chat_id: str, caption: str, filepath: str):
    if not token or not chat_id or not filepath:
        return False, "missing token/chat/file"
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    try:
        with open(filepath, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.status_code == 200:
            return True, "sent"
        else:
            return False, f"tg status {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

def send_telegram_text_direct(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "missing token/chat"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
        return (resp.status_code == 200), resp.text
    except Exception as e:
        return False, str(e)

# ==========================
# Core signal_for_timeframe
# (kept mostly same as before; consolidation-first)
def signal_for_timeframe(candles, granularity, i_rej, i_con) -> Tuple[Optional[str], str]:
    import numpy as np

    if not candles or len(candles) < 60:
        return None, "insufficient history"

    # params
    REJ_WICK_RATIO = 0.2
    OVERSIZED_MULT  = 1.5
    MOMENTUM_ATR_FRAC = 0.03
    EXHAUSTION_ATR_MULT = 2.0
    WIGGLE_FRAC = 0.25
    CONSISTENT_DIR_PCT = 0.6
    CONSISTENT_BARS = 8

    opens  = np.array([c["open"]  for c in candles], dtype=float)
    highs  = np.array([c["high"]  for c in candles], dtype=float)
    lows   = np.array([c["low"]   for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    typical = (highs + lows + closes) / 3.0
    last_price = float(closes[-1])

    ma1, ma2, ma3 = compute_mas(candles)

    try:
        ma1_rej = float(ma1[i_rej]); ma2_rej = float(ma2[i_rej]); ma3_rej = float(ma3[i_rej])
    except Exception:
        return None, "invalid/insufficient MA data"

    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05 * atr)

    if atr > MOMENTUM_ATR_FRAC * last_price:
        return None, "momentum too volatile"

    consolidating, cons_reason = is_consolidating(candles, atr, lookback=12, range_mult=0.9, body_mult=0.45, overlap_pct=0.72)
    if consolidating:
        return None, cons_reason

    if len(opens) >= CONSISTENT_BARS:
        directions = (closes[-CONSISTENT_BARS:] - opens[-CONSISTENT_BARS:]) > 0
        pct_up = float(np.sum(directions)) / CONSISTENT_BARS
        pct_down = 1.0 - pct_up
        consistent_up = pct_up >= CONSISTENT_DIR_PCT
        consistent_down = pct_down >= CONSISTENT_DIR_PCT
    else:
        consistent_up = consistent_down = False

    if not (consistent_up or consistent_down):
        wiggle = WIGGLE_FRAC * atr
        stacked_up_flag = (ma1_rej >= ma2_rej - wiggle and ma2_rej >= ma3_rej - wiggle)
        stacked_down_flag = (ma1_rej <= ma2_rej + wiggle and ma2_rej <= ma3_rej + wiggle)
        if not (stacked_up_flag or stacked_down_flag):
            return None, "no clear direction / choppy"

    # candle bits
    def candle_bits_at(idx):
        o = float(opens[idx]); h = float(highs[idx]); l = float(lows[idx]); c = float(closes[idx])
        body = abs(c - o); r = max(h - l, 1e-12)
        upper = h - max(o, c); lower = min(o, c) - l
        is_doji = body <= 0.35 * r
        pin_low = (lower >= REJ_WICK_RATIO * body) and (lower > upper)
        pin_high = (upper >= REJ_WICK_RATIO * body) and (upper > lower)
        engulf_bull = False; engulf_bear = False
        if idx > 0:
            prev_o = float(opens[idx-1]); prev_c = float(closes[idx-1])
            if (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o):
                engulf_bull = True
            if (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o):
                engulf_bear = True
        return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
                "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
                "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

    prev_candle = candle_bits_at(i_rej - 1) if i_rej - 1 >= 0 else None
    rej = candle_bits_at(i_rej)
    con = candle_bits_at(i_con) if i_con is not None else {"c": float(closes[-1])}

    # oversize check
    if rej["body"] > OVERSIZED_MULT * atr or rej["range"] > OVERSIZED_MULT * atr:
        return None, "rejection candle oversized"
    if con["body"] > OVERSIZED_MULT * atr or con["range"] > OVERSIZED_MULT * atr:
        return None, "confirmation candle oversized"

    # trend helper
    wiggle = WIGGLE_FRAC * atr
    def ma_trend_up_at(idx):
        return (ma1[idx] >= ma2[idx] - wiggle) and (ma2[idx] >= ma3[idx] - wiggle)
    def ma_trend_down_at(idx):
        return (ma1[idx] <= ma2[idx] + wiggle) and (ma2[idx] <= ma3[idx] + wiggle)

    uptrend = ma_trend_up_at(i_rej)
    downtrend = ma_trend_down_at(i_rej)

    def pick_ma_for_buy(idx):
        cbs = candle_bits_at(idx)
        d1 = abs(cbs["l"] - ma1[idx])
        d2 = abs(cbs["l"] - ma2[idx])
        return ("MA1", float(ma1[idx])) if d1 <= d2 else ("MA2", float(ma2[idx]))

    def pick_ma_for_sell(idx):
        cbs = candle_bits_at(idx)
        d1 = abs(cbs["h"] - ma1[idx])
        d2 = abs(cbs["h"] - ma2[idx])
        return ("MA1", float(ma1[idx])) if d1 <= d2 else ("MA2", float(ma2[idx]))

    def rejection_ok_buy_local(rej_c, idx):
        if not ma_trend_up_at(idx):
            return False, "not uptrend"
        ma_name, zone = pick_ma_for_buy(idx)
        buffer_ = max(WIGGLE_FRAC * atr, 0.0)
        near_zone = abs(rej_c["l"] - zone) <= buffer_
        close_side = (rej_c["c"] >= (zone - buffer_))
        pattern_ok = rej_c["is_doji"] or rej_c["pin_low"] or rej_c["engulf_bull"]
        is_pivot = pivot_low(idx, list(lows), k=2)
        pullback_ok = recent_pullback_bias(idx, list(opens), list(closes), bars=3) == "down"
        if near_zone and close_side and pattern_ok and is_pivot and pullback_ok:
            return True, f"{ma_name} support rejection (pivot)"
        return False, "no valid buy rejection (pivot/pullback failed)"

    def rejection_ok_sell_local(rej_c, idx):
        if not ma_trend_down_at(idx):
            return False, "not downtrend"
        ma_name, zone = pick_ma_for_sell(idx)
        buffer_ = max(WIGGLE_FRAC * atr, 0.0)
        near_zone = abs(rej_c["h"] - zone) <= buffer_
        close_side = (rej_c["c"] <= (zone + buffer_))
        pattern_ok = rej_c["is_doji"] or rej_c["pin_high"] or rej_c["engulf_bear"]
        is_pivot = pivot_high(idx, list(highs), k=2)
        pullback_ok = recent_pullback_bias(idx, list(opens), list(closes), bars=3) == "up"
        if near_zone and close_side and pattern_ok and is_pivot and pullback_ok:
            return True, f"{ma_name} resistance rejection (pivot)"
        return False, "no valid sell rejection (pivot/pullback failed)"

    # BUY path
    if uptrend:
        rej_ok, rej_reason = rejection_ok_buy_local(rej, i_rej)
        if not rej_ok:
            return None, f"buy rejected: {rej_reason}"
        which_ma, ma_val = pick_ma_for_buy(i_rej)
        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "buy rejected: exhaustion"
        reason = f"BUY | Trend=UP | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_low'] else ('Engulf' if rej['engulf_bull'] else 'Unknown')) } | close={rej['c']:.5f}"
        return "BUY", reason

    # SELL path
    if downtrend:
        rej_ok, rej_reason = rejection_ok_sell_local(rej, i_rej)
        if not rej_ok:
            return None, f"sell rejected: {rej_reason}"
        which_ma, ma_val = pick_ma_for_sell(i_rej)
        if abs(rej["c"] - ma_val) > EXHAUSTION_ATR_MULT * atr:
            return None, "sell rejected: exhaustion"
        reason = f"SELL | Trend=DOWN | MA={which_ma} rejected | pattern={ 'Doji' if rej['is_doji'] else ('Pin' if rej['pin_high'] else ('Engulf' if rej['engulf_bear'] else 'Unknown')) } | close={rej['c']:.5f}"
        return "SELL", reason

    return None, "no clear trend / signal"

# ==========================
# Orchestrate: per asset, both TFs, resolve conflicts, notify (prefer lower TF)
# ==========================
def _can_send_alert(key, cooldown_seconds):
    now = int(time.time())
    last = LAST_ALERT.get(key)
    if last is None or (now - last) >= cooldown_seconds:
        LAST_ALERT[key] = now
        return True
    return False

def analyze_and_notify():
    load_persistent_cache()
    tf_priority = sorted(TIMEFRAMES)
    any_sent_this_run = False
    report_lines = []

    for symbol in ASSETS:
        results: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        log("Processing", symbol)

        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 3:
                results[tf] = (None, "no candles")
                log(f" {symbol} {tf//60}m -> no candles or insufficient")
                continue

            i_rej = len(candles) - 2
            i_con = len(candles) - 1
            direction, reason = signal_for_timeframe(candles, tf, i_rej, i_con)
            results[tf] = (direction, reason)
            log(f" {symbol} {tf//60}m -> {direction} | {reason}")

        # choose single tf (prefer lower)
        chosen = None
        for tf in tf_priority:
            dirc, rsn = results.get(tf, (None, None))
            if dirc:
                chosen = (tf, dirc, rsn)
                break

        if chosen is None:
            report_lines.append(f"{symbol}: no signal (details: { {tf:results[tf] for tf in TIMEFRAMES} })")
            continue

        tf_chosen, direction_chosen, reason_chosen = chosen

        # dedupe persistent
        last = last_sent_signal_by_symbol.get(symbol)
        if last and last == (direction_chosen, reason_chosen):
            log(f"Skipping duplicate for {symbol}: {direction_chosen}")
            continue

        # create chart if possible and send
        chosen_candles = fetch_candles(symbol, tf_chosen, CANDLES_N)
        ma1, ma2, ma3 = compute_mas(chosen_candles)
        i_rej_chart = len(chosen_candles) - 2

        chart_path = make_chart(chosen_candles, ma1, ma2, ma3, i_rej_chart, direction_chosen, symbol, tf_chosen, last_n=60)
        caption = f"{symbol} | {tf_chosen//60}m | {direction_chosen}\n{reason_chosen}"

        sent_ok, sent_msg = False, "no send attempted"
        if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            ok, info = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            sent_ok, sent_msg = ok, info
            try:
                os.unlink(chart_path)
            except Exception:
                pass
        else:
            # fallback to bot helper text send
            if send_single_timeframe_signal:
                try:
                    send_single_timeframe_signal(symbol, tf_chosen, direction_chosen, reason_chosen)
                    sent_ok, sent_msg = True, "sent via bot helper"
                except Exception as e:
                    sent_ok, sent_msg = False, str(e)
            else:
                # as last fallback, try direct Telegram text
                ok, info = send_telegram_text_direct(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                sent_ok, sent_msg = ok, info

        if sent_ok:
            any_sent_this_run = True
            last_sent_signal_by_symbol[symbol] = (direction_chosen, reason_chosen)
            report_lines.append(f"{symbol}: SENT {direction_chosen} @ {tf_chosen//60}m -> {reason_chosen}")
            log("Sent:", symbol, direction_chosen, tf_chosen, sent_msg)
        else:
            log("Send failed for", symbol, sent_msg)
            report_lines.append(f"{symbol}: failed send: {sent_msg}")

    # persist cache
    save_persistent_cache()

    # heartbeat logic: if no signal sent, send heartbeat text each HEARTBEAT_INTERVAL_HOURS
    # heartbeat logic: if no signal sent, send heartbeat text each HEARTBEAT_INTERVAL_HOURS
    if not any_sent_this_run:
        last_hb = last_heartbeat_time()
        now = int(time.time())
        if now - last_hb >= int(HEARTBEAT_INTERVAL_HOURS * 3600):
        # send clean heartbeat with checked assets
            send_heartbeat(ASSETS)
            set_heartbeat_time(now)
            log("Heartbeat sent (no signals).")
        else:
            log("Heartbeat not due yet.")
    else:
        log("Signals were sent this run; skipping heartbeat.")
# Entry sleep mode (Zimbabwe TZ)
if __name__ == "__main__":
    try:
        tz = pytz.timezone("Africa/Harare")
        now = datetime.now(tz)
        hour = now.hour
        if 21 <= hour or hour < 6:
            log("⏸ Bot sleeping (Zimbabwe 21:00–06:00)")
        else:
            analyze_and_notify()
    except Exception as e:
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

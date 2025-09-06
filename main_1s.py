#!/usr/bin/env python3
# main_1s.py — paste-and-cruise
"""
1s-indices runner (evaluates 5m TF only). Robust caching + dedup + charting.
Designed to run in scheduled GitHub Action (snapshot fetch).
Env vars:
  DERIV_API_KEY (required)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional)
  ASSETS (csv) default: "1HZ75V,1HZ100V,1HZ150V"
  CANDLES_N default: 100
  DISABLE_HEARTBEAT=1  -> disables startup heartbeat
  SLEEP_WINDOW e.g. "21:00-06:00" (local_time offset applied via TZ_OFFSET)
  TZ_OFFSET e.g. "+02:00" (Zimbabwe +02:00 default)
  MAX_SIGNALS_PER_RUN default 3
  ALERT_COOLDOWN_SECS default 120 (used only for identical direction duplicates if desired)
"""
import os
import time
import json
import tempfile
import math
import traceback
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------
# Config from env
# ---------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True")
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
TIMEFRAMES = [300]  # only 5m

CANDLES_N = int(os.getenv("CANDLES_N", "100"))  # default 100 as requested
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "220"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))

CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))

MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "3"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "120"))

TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")
PROCESSED_FILE = os.path.join(TMPDIR, "fenyere_processed_1s.json")

DISABLE_HEARTBEAT = os.getenv("DISABLE_HEARTBEAT", "0") in ("1", "true", "True")
SLEEP_WINDOW = os.getenv("SLEEP_WINDOW", "")  # e.g. "21:00-06:00"
TZ_OFFSET = os.getenv("TZ_OFFSET", "+02:00")  # Zimbabwe default +02:00

# ---------------------
# Utilities & minimal bot fallback
# ---------------------
def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# Telegram helper — try to import from bot.py, fallback to simple HTTP using requests if available
try:
    from bot import send_telegram_message, send_single_timeframe_signal, send_strong_signal, send_telegram_photo
except Exception:
    try:
        import requests
    except Exception:
        requests = None

    def send_telegram_message(token, chat_id, text):
        if not token or not chat_id:
            print("[TELEGRAM TEXT]", text)
            return False, "no-token"
        if requests is None:
            print("[TELEGRAM TEXT fallback-no-requests]", text)
            return False, "no-requests"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
        return resp.ok, resp.text

    def send_telegram_photo(token, chat_id, caption, photo_path):
        if not token or not chat_id:
            print("[TELEGRAM PHOTO]", caption, photo_path)
            return False, "no-token"
        if requests is None:
            print("[TELEGRAM PHOTO fallback-no-requests]", caption, photo_path)
            return False, "no-requests"
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                resp = requests.post(url, data={"chat_id": chat_id, "caption": caption}, files=files)
            return resp.ok, resp.text
        except Exception as e:
            return False, str(e)

    def send_single_timeframe_signal(symbol, tf, direction, reason, chart_path=None):
        text = f"{symbol} | {tf//60}m | {direction}\nReason: {reason}"
        ok, resp = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
        if chart_path:
            send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text, chart_path)
        return ok

    def send_strong_signal(symbol, direction, details, chart_path=None):
        text = f"STRONG {symbol} | {direction}\n{details}"
        ok, resp = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
        if chart_path:
            send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text, chart_path)
        return ok

# ---------------------
# Persistence helpers
# ---------------------
last_sent_cache: Dict[str, Dict] = {}
processed_cache: Dict[str, int] = {}

def load_cache():
    global last_sent_cache, processed_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_cache = json.load(f)
        else:
            last_sent_cache = {}
    except Exception:
        last_sent_cache = {}
    try:
        if os.path.exists(PROCESSED_FILE):
            with open(PROCESSED_FILE, "r") as f:
                processed_cache = json.load(f)
        else:
            processed_cache = {}
    except Exception:
        processed_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
    except Exception:
        pass
    try:
        with open(PROCESSED_FILE, "w") as f:
            json.dump(processed_cache, f)
    except Exception:
        pass

def can_send_for_candle(symbol: str, candle_epoch: int) -> bool:
    last = processed_cache.get(symbol)
    if last is None:
        return True
    return candle_epoch > int(last)

def mark_processed(symbol: str, candle_epoch: int):
    processed_cache[symbol] = int(candle_epoch)
    save_cache()

def mark_sent(symbol: str, direction: str, tf: int, candle_epoch: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time()), "epoch": int(candle_epoch)}
    save_cache()

# ---------------------
# MA helpers
# ---------------------
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

# ---------------------
# Candle fetch (snapshot mode)
# ---------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Robust snapshot fetch; tries some fallback smaller counts on failure.
    Returns list of candle dicts sorted by epoch asc, keys: epoch, open, high, low, close
    """
    attempts_counts = [count, max(100, count // 2), max(50, count // 4), 25, 10]
    tried = set()
    for c in attempts_counts:
        if c in tried:
            continue
        tried.add(c)
        for attempt in range(3):
            ws = None
            try:
                ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                _ = ws.recv()
                req = {
                    "ticks_history": symbol,
                    "style": "candles",
                    "granularity": granularity,
                    "count": c,
                    "end": "latest",
                    "subscribe": 0
                }
                ws.send(json.dumps(req))
                raw = ws.recv()
                resp = json.loads(raw)
                if "candles" in resp and resp["candles"]:
                    candles = [{
                        "epoch": int(candle["epoch"]),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"])
                    } for candle in resp["candles"]]
                    candles.sort(key=lambda x: x["epoch"])
                    return candles
            except Exception as e:
                log(f"[fetch_candles] error for {symbol}@{granularity}s count={c} attempt={attempt+1}: {e}")
            finally:
                try:
                    if ws:
                        ws.close()
                except:
                    pass
            time.sleep(0.7)
    return []

# ---------------------
# Charting
# ---------------------
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
        ax.set_title(f"{symbol} | {tf//60}m | {reason}", fontsize=10)

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

        def safe_plot(y, label, c):
            yy = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float('nan') for v in y]
            ax.plot(list(range(len(yy))), yy, label=label, linewidth=1.0)

        try:
            safe_plot(ma1_vals, "MA1", None)
            safe_plot(ma2_vals, "MA2", None)
            safe_plot(ma3_vals, "MA3", None)
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

        step = max(1, len(chosen)//8)
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

# ---------------------
# Pattern detection / scoring (flexible)
# ---------------------
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o) if not math.isnan(c - o) else 0.0
    r = max(1e-12, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l
    is_doji = body <= 0.25 * r  # smaller threshold to accept small-bodied doji
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

def compute_atr(candles: List[Dict], lookback=14) -> float:
    if not candles:
        return 0.0
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    rngs = highs - lows
    if len(rngs) >= lookback:
        return float(np.mean(rngs[-lookback:]))
    return float(np.mean(rngs))

def is_near(value: float, target: float, atr: float, factor=0.6) -> bool:
    # near measured relative to ATR or percentage
    if atr > 1e-9:
        return abs(value - target) <= max(atr * factor, atr * 0.4)
    # fallback percent
    return abs(value - target) <= max(0.002 * target, 0.5)

def detect_rejection_on_ma(candles: List[Dict], idx: int, ma1: List, ma2: List, ma3: List) -> Tuple[bool, str]:
    """
    Accept many rejection forms:
      - pin bars (upper/lower wick)
      - doji / tiny body near MA
      - small engulfers
    Accept near-MA distance up to ATR * factor
    Returns (True, reason_str) if rejection detected at index idx (closed candle)
    """
    if idx <= 0 or idx >= len(candles):
        return False, ""
    candle = candles[idx]
    prev = candles[idx - 1] if idx - 1 >= 0 else None
    bits = candle_bits(candle, prev)
    atr = compute_atr(candles[:idx+1])
    close = candle["close"]
    o = candle["open"]
    # check distance to MA1 or MA2
    ma1_val = ma1[idx] if idx < len(ma1) else None
    ma2_val = ma2[idx] if idx < len(ma2) else None
    near_ma1 = ma1_val is not None and is_near(close, ma1_val, atr, factor=0.6)
    near_ma2 = ma2_val is not None and is_near(close, ma2_val, atr, factor=0.6)
    # direction: if candle rejects down -> bullish (close > open? not necessary)
    # detect pin / doji / small body: treat as rejection if near MA and price action shows rejection (wick opposite to direction)
    if (near_ma1 or near_ma2):
        # prefer pin or doji or tiny bodies
        if bits["is_doji"]:
            return True, "Doji/Small body near MA"
        # pin low (bullish rejection)
        if bits["pin_low"]:
            return True, "PinLow near MA (bullish)"
        if bits["pin_high"]:
            return True, "PinHigh near MA (bearish)"
        # engulfers as strong rejection
        if bits["engulf_bull"]:
            return True, "Engulfing bullish near MA"
        if bits["engulf_bear"]:
            return True, "Engulfing bearish near MA"
        # also accept tiny bodies even without long wick
        if bits["body"] <= 0.15 * bits["range"]:
            return True, "Tiny body near MA (loose)"
    return False, ""

def detect_ma3_break(candles: List[Dict], ma3: List[Optional[float]], lookback=6) -> Optional[int]:
    """
    Detect recent MA3 crossing (break). Returns index of crossing candle (where cross happened) or None.
    A cross occurs when close crosses MA3 between i-1 and i.
    We lookback a few candles for a recent crossing.
    """
    n = len(candles)
    for i in range(max(1, n - lookback), n):
        if i - 1 < 0:
            continue
        prev_close = candles[i - 1]["close"]
        curr_close = candles[i]["close"]
        ma_prev = ma3[i - 1] if i - 1 < len(ma3) else None
        ma_curr = ma3[i] if i < len(ma3) else None
        if ma_prev is None or ma_curr is None:
            continue
        # bullish break
        if prev_close < ma_prev and curr_close > ma_curr:
            return i
        # bearish break
        if prev_close > ma_prev and curr_close < ma_curr:
            return i
    return None

# ---------------------
# Signal definition and run logic
# ---------------------
def analyze_symbol(symbol: str, tf: int = 300) -> Optional[Dict]:
    """
    Analyze latest closed candle only and return a signal dict if found:
    { symbol, tf, direction, reason, candle_epoch, chart_path (optional) }
    """
    candles = fetch_candles(symbol, tf, count=CANDLES_N)
    if not candles:
        log(f"[{symbol}] no candles fetched")
        return None
    ma1, ma2, ma3 = compute_mas_for_chart(candles)
    # index of latest closed candle (last element)
    idx = len(candles) - 1
    last_epoch = candles[idx]["epoch"]
    # If we've already processed this candle, skip (dedup)
    if not can_send_for_candle(symbol, last_epoch):
        log(f"[{symbol}] candle {last_epoch} already processed; skipping")
        return None

    # Basic guard: require at least some candles and MA values
    if idx < 10:
        log(f"[{symbol}] insufficient history for analysis (n={len(candles)})")
        mark_processed(symbol, last_epoch)  # still mark processed to avoid repeated attempts
        return None

    # compute ATR for thresholds
    atr = compute_atr(candles)
    # Detect rejection on the latest closed candle
    rej_ok, rej_reason = detect_rejection_on_ma(candles, idx, ma1, ma2, ma3)
    # Detect MA3 breakout within recent lookback
    cross_idx = detect_ma3_break(candles, ma3, lookback=8)

    # Determine direction of rejection:
    if not rej_ok:
        # nothing to do — but still mark processed to avoid repeated re-checks
        mark_processed(symbol, last_epoch)
        return None

    # Decide direction by looking at the candle shape
    bits = candle_bits(candles[idx], candles[idx - 1] if idx - 1 >= 0 else None)
    # if pin low or body close > open => BUY; vice versa for SELL (this is flexible)
    direction = "BUY" if (bits["pin_low"] or (candles[idx]["close"] > candles[idx]["open"]) or bits["engulf_bull"]) else "SELL"

    # If MA3 breakout exists, ensure the retest happened after cross (this confirms trend change)
    if cross_idx is not None:
        # require cross index < idx (cross happened earlier) and rejection occurred after cross
        if cross_idx >= idx:
            # cross is not older than rejection -> skip
            mark_processed(symbol, last_epoch)
            return None
        # else we consider this a valid retest after cross -> strengthen reason
        reason = f"MA3-break + {rej_reason}"
    else:
        # No MA3 breakout: still allow near-MA rejections that ride trend (user asked to catch retests)
        reason = f"Near-MA rejection (loose): {rej_reason}"

    # Build chart
    chart_path = make_chart(candles, ma1, ma2, ma3, idx, f"{symbol} | {direction} | {reason}", symbol, tf)

    # Ready to send
    signal = {
        "symbol": symbol,
        "tf": tf,
        "direction": direction,
        "reason": reason,
        "price": candles[idx]["close"],
        "candle_epoch": last_epoch,
        "chart_path": chart_path
    }
    return signal

# ---------------------
# Time window helpers
# ---------------------
def in_sleep_window(sleep_window: str, tz_offset: str = "+00:00") -> bool:
    if not sleep_window:
        return False
    try:
        start_s, end_s = sleep_window.split("-")
        # parse TZ offset to hours
        sign = 1 if tz_offset[0] != "-" else -1
        tz_hours = int(tz_offset[1:3]) * sign
        tz_minutes = int(tz_offset[4:6]) if len(tz_offset) >= 6 else 0
        offset = timedelta(hours=tz_hours, minutes=tz_minutes * sign)
        now_utc = datetime.utcnow()
        local_now = now_utc + offset
        start_h, start_m = [int(x) for x in start_s.split(":")]
        end_h, end_m = [int(x) for x in end_s.split(":")]
        start_t = local_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end_t = local_now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if start_t < end_t:
            return start_t <= local_now <= end_t
        else:
            # overnight window like 21:00-06:00
            return local_now >= start_t or local_now <= end_t
    except Exception:
        return False

# ---------------------
# Main runner
# ---------------------
def run_once():
    load_cache()
    if not DERIV_API_KEY:
        raise RuntimeError("Missing DERIV_API_KEY env var")
    if in_sleep_window(SLEEP_WINDOW, TZ_OFFSET):
        log("Within sleep window, skipping run")
        return

    # Optional startup heartbeat (disable via DISABLE_HEARTBEAT=1)
    if not DISABLE_HEARTBEAT:
        hb_text = f"main_1s.py startup heartbeat (5m timeframe, {len(ASSETS)} indices)"
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"[TELEGRAM TEXT] {hb_text}")

    signals_sent = 0
    reports = []
    for sym in ASSETS:
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            break
        try:
            sig = analyze_symbol(sym, tf=300)
            if not sig:
                reports.append(f"{sym} -> no alerts chosen")
                continue
            # final dedup: ensure not sent already for this candle
            last = last_sent_cache.get(sym)
            if last and int(last.get("epoch", 0)) >= int(sig["candle_epoch"]):
                reports.append(f"{sym} -> already sent for epoch {sig['candle_epoch']}")
                mark_processed(sym, sig["candle_epoch"])  # still mark processed
                continue
            # send
            msg = f"🔔 {sig['symbol']} | {sig['tf']//60}m | {sig['direction']}\nReason: {sig['reason']}\nPrice: {sig['price']}"
            ok_text, _ = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
            # also send chart if available
            if sig.get("chart_path"):
                send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, sig["chart_path"])
            mark_sent(sym, sig['direction'], sig['tf'], sig['candle_epoch'])
            mark_processed(sym, sig['candle_epoch'])
            signals_sent += 1
            reports.append(f"{sym} -> signal sent: {sig['direction']} ({sig['reason']})")
        except Exception as e:
            log(f"[run_once] error for {sym}: {e}")
            traceback.print_exc()
            reports.append(f"{sym} -> error")
    # Summary
    log("Run complete. Reports:", reports)
    # final save
    save_cache()

if __name__ == "__main__":
    try:
        run_once()
    except Exception as exc:
        log("Fatal error:", exc)
        traceback.print_exc()
        # notify crash if possible
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"Bot crash: {exc}")
        except:
            pass
        raise

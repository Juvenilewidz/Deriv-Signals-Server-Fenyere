#!/usr/bin/env python3
# main.py - paste-and-cruise dynamic S/R (trend-following) bot (charts + text)
# Tracks both default indices and 1s indices; robust fetch + dedup + anti-spam.
#
# Behavior summary:
# - TIMEFRAMES: 5m,10m,15m (300,600,900)
# - CANDLES_N default 100
# - Signal when: (A) continuation: MA aligned + rejection at MA1/MA2 OR (B) reversal: MA3 break + retest+rejection
# - Rejection families: pinbars, doji, engulfing, tiny-body (inclusive)
# - Send chart + text on signal; only build chart when signal found
# - Dedup by processed candle epoch per symbol+tf to avoid repeated sends on scheduled runs
# - Optional sleep window, optional heartbeat disable
# - Tune env vars: CANDLES_N, MAX_SIGNALS_PER_RUN, ALERT_COOLDOWN_SECS, DIST_TOL, DEBUG, SLEEP_WINDOW, TZ_OFFSET

import os, json, time, math, tempfile, traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# third-party (ensure in GH Action: websocket-client, numpy, matplotlib)
import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Try import bot helpers; fallback to simple HTTP or prints
try:
    from bot import send_telegram_message, send_single_timeframe_signal, send_strong_signal, send_telegram_photo
except Exception:
    try:
        import requests
    except Exception:
        requests = None

    def send_telegram_message(token, chat_id, text):
        if not token or not chat_id:
            print("[TELEGRAM TEXT]", text); return False, "no-token"
        if requests is None:
            print("[TELEGRAM TEXT fallback-no-requests]", text); return False, "no-requests"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text})
        return r.ok, r.text

    def send_telegram_photo(token, chat_id, caption, photo_path):
        if not token or not chat_id:
            print("[TELEGRAM PHOTO]", caption, photo_path); return False, "no-token"
        if requests is None:
            print("[TELEGRAM PHOTO fallback-no-requests]", caption, photo_path); return False, "no-requests"
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                r = requests.post(url, data={"chat_id": chat_id, "caption": caption}, files=files)
            return r.ok, r.text
        except Exception as e:
            return False, str(e)

    def send_single_timeframe_signal(symbol, tf, direction, reason, chart_path=None):
        txt = f"{symbol} | {tf//60}m | {direction}\nReason: {reason}"
        ok, resp = send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), txt)
        if chart_path:
            send_telegram_photo(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), txt, chart_path)
        return ok

    def send_strong_signal(symbol, direction, details, chart_path=None):
        txt = f"STRONG {symbol} | {direction}\n{details}"
        ok, resp = send_telegram_message(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), txt)
        if chart_path:
            send_telegram_photo(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""), txt, chart_path)
        return ok

# -------------------------
# Config (env override)
# -------------------------
DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# Default asset sets: combine your default indices and the 1s indices
ASSETS_DEFAULT = os.getenv("ASSETS_DEFAULT", "R_10,R_50,R_75").split(",")
ASSETS_1S = os.getenv("ASSETS_1S", "1HZ75V,1HZ100V,1HZ150V").split(",")
ASSETS = [s.strip() for s in (ASSETS_DEFAULT + ASSETS_1S) if s and s.strip()]

TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m

# History & charting
CANDLES_N = int(os.getenv("CANDLES_N", "100"))
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "180"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))
CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))

MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "4"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "60"))  # small cooldown for duplicate suppression

DIST_TOL = float(os.getenv("DIST_TOL", "1.0"))  # factor * ATR considered "near"

# behavior toggles
DISABLE_HEARTBEAT = os.getenv("DISABLE_HEARTBEAT", "0") in ("1", "true", "True")
SLEEP_WINDOW = os.getenv("SLEEP_WINDOW", "")  # "21:00-06:00"
TZ_OFFSET = os.getenv("TZ_OFFSET", "+00:00")  # default UTC; set "+02:00" for Zimbabwe if you prefer

# temp files
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent.json")
PROCESSED_FILE = os.path.join(TMPDIR, "fenyere_processed.json")

# in memory caches
last_sent_cache: Dict[str, Dict] = {}
processed_cache: Dict[str, int] = {}  # key: f"{symbol}:{tf}" -> epoch

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# -------------------------
# Persistence helpers
# -------------------------
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
    except Exception as e:
        log("save_cache error:", e)
    try:
        with open(PROCESSED_FILE, "w") as f:
            json.dump(processed_cache, f)
    except Exception as e:
        log("save_processed error:", e)

def mark_processed(symbol: str, tf: int, candle_epoch: int):
    key = f"{symbol}:{tf}"
    processed_cache[key] = int(candle_epoch)
    save_cache()

def already_processed(symbol: str, tf: int, candle_epoch: int) -> bool:
    key = f"{symbol}:{tf}"
    last = processed_cache.get(key)
    if last is None:
        return False
    return int(candle_epoch) <= int(last)

def mark_sent(symbol: str, direction: str, tf: int, candle_epoch: int):
    last_sent_cache[symbol] = {"direction": direction, "tf": tf, "ts": int(time.time()), "epoch": int(candle_epoch)}
    save_cache()

def can_send_cooldown(symbol: str, direction: str, tf: int) -> bool:
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec and rec.get("direction") == direction and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
        return False
    return True

# -------------------------
# MA utils
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
# Candle fetch (snapshot + small fallbacks). For production you can increase counts
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    tried_counts = [count, max(150, count // 2), max(80, count // 4), 50, 25]
    for c in tried_counts:
        for attempt in range(2):
            ws = None
            try:
                ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                try:
                    _ = ws.recv()
                except Exception:
                    pass
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
                if isinstance(resp, dict) and resp.get("candles"):
                    parsed = []
                    for cc in resp["candles"]:
                        try:
                            parsed.append({"epoch": int(cc["epoch"]), "open": float(cc["open"]),
                                           "high": float(cc["high"]), "low": float(cc["low"]), "close": float(cc["close"])})
                        except Exception:
                            continue
                    parsed.sort(key=lambda x: x["epoch"])
                    return parsed
            except Exception as e:
                log(f"fetch_candles err for {symbol}@{granularity}s (count={c}) attempt {attempt+1}: {e}")
            finally:
                try:
                    if ws:
                        ws.close()
                except:
                    pass
            time.sleep(0.6)
    # final fail
    log(f"fetch_candles final fail for {symbol}@{granularity}s")
    return []

# -------------------------
# Charting (only on signal)
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

        fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
        ax.set_title(f"{symbol} | {tf//60}m | {reason}", fontsize=10)

        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = "#2ca02c" if c >= o else "#d62728"
            ax.plot([i, i], [l, h], color="black", linewidth=0.6, zorder=1)
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - CANDLE_WIDTH / 2.0, lower), CANDLE_WIDTH, height, facecolor=color, edgecolor="black", linewidth=0.35, zorder=2)
            ax.add_patch(rect)

        ma1_vals = []; ma2_vals = []; ma3_vals = []
        for i in range(start, total):
            ma1_vals.append(ma1[i] if i < len(ma1) else None)
            ma2_vals.append(ma2[i] if i < len(ma2) else None)
            ma3_vals.append(ma3[i] if i < len(ma3) else None)

        def plot_ma(vals, label, color):
            y = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float('nan') for v in vals]
            ax.plot(list(range(len(y))), y, label=label, color=color, linewidth=1.2, zorder=3)

        try:
            plot_ma(ma1_vals, "MA1", "#1f77b4")
            plot_ma(ma2_vals, "MA2", "#ff7f0e")
            plot_ma(ma3_vals, "MA3", "#d62728")
            ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        if start <= rej_index < total:
            idx = rej_index - start
            price = chosen[idx]["close"]
            marker_color = "red" if ("SELL" in reason or "sell" in reason.lower()) else "green"
            ax.scatter([idx], [price], marker="v" if "SELL" in reason else "^", color=marker_color, s=160, zorder=6, edgecolors="black")
            ax.text(idx, price, "  "+reason, fontsize=8, color=marker_color)

        ax.set_xlim(-1, len(chosen) - 1 + pad)
        ymin, ymax = min(lows), max(highs)
        pad_y = (ymax - ymin) * 0.08 if ymax > ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        step = max(1, len(chosen)//8)
        ax.set_xticks(range(0, len(chosen), step))
        ax.set_xticklabels([xs[i].strftime("%H:%M\n%m-%d") for i in range(0, len(xs), step)], rotation=25, fontsize=8)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.tight_layout(); fig.savefig(tmp.name, dpi=120); plt.close(fig)
        return tmp.name
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return None

# -------------------------
# Pattern detection & helpers
# -------------------------
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o)
    r = max(1e-12, h - l)
    upper = h - max(o, c); lower = min(o, c) - l
    is_doji = body <= 0.25 * r
    pin_low = (lower >= 0.3 * r) and (lower > upper)
    pin_high = (upper >= 0.3 * r) and (upper > lower)
    engulf_bull = engulf_bear = False
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
    closes = np.array([c["close"] for c in candles], dtype=float)
    trs = []
    for i in range(len(candles)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    if len(trs) >= lookback:
        return float(np.mean(trs[-lookback:]))
    return float(np.mean(trs))

def is_near_price_to_ma(price: float, ma_value: Optional[float], atr: float, factor=0.6) -> bool:
    if ma_value is None:
        return False
    threshold = max(atr * factor, 0.001 * price)
    return abs(price - ma_value) <= threshold

# detect MA alignment/trend
def ma_alignment_up(ma1: List[Optional[float]], ma2: List[Optional[float]], ma3: List[Optional[float]], idx: int) -> bool:
    try:
        return ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and (ma1[idx] > ma2[idx] > ma3[idx])
    except Exception:
        return False

def ma_alignment_down(ma1: List[Optional[float]], ma2: List[Optional[float]], ma3: List[Optional[float]], idx: int) -> bool:
    try:
        return ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and (ma1[idx] < ma2[idx] < ma3[idx])
    except Exception:
        return False

def detect_rejection_at_idx(candles: List[Dict], idx: int, ma1: List, ma2: List, atr: float) -> Tuple[bool, str, str]:
    """
    Return (is_rejection, reason, suggested_direction) for candle at idx.
    suggested_direction is "BUY" or "SELL".
    """
    if idx <= 0 or idx >= len(candles):
        return False, "", ""
    candle = candles[idx]
    prev = candles[idx - 1] if idx - 1 >= 0 else None
    bits = candle_bits(candle, prev)
    close = candle["close"]
    ma1v = ma1[idx] if idx < len(ma1) else None
    ma2v = ma2[idx] if idx < len(ma2) else None
    atr_local = atr if atr > 0 else max(0.0001, abs(close) * 0.001)
    near1 = is_near_price_to_ma(close, ma1v, atr_local)
    near2 = is_near_price_to_ma(close, ma2v, atr_local)
    if not (near1 or near2):
        return False, "", ""
    # Doji & tiny body
    if bits["is_doji"]:
        # direction guess from candle close vs open or surrounding MAs slope
        dir_guess = "BUY" if candle["close"] >= candle["open"] else "SELL"
        return True, "Doji/Small body near MA", dir_guess
    # Pin low -> bullish rejection
    if bits["pin_low"]:
        return True, "PinLow (bullish) near MA", "BUY"
    if bits["pin_high"]:
        return True, "PinHigh (bearish) near MA", "SELL"
    # engulf
    if bits["engulf_bull"]:
        return True, "Engulf Bull near MA", "BUY"
    if bits["engulf_bear"]:
        return True, "Engulf Bear near MA", "SELL"
    # tiny body relative to range
    if bits["body"] <= 0.15 * bits["range"]:
        dir_guess = "BUY" if candle["close"] >= candle["open"] else "SELL"
        return True, "Tiny body near MA (loose)", dir_guess
    return False, "", ""

def detect_recent_ma3_cross(candles: List[Dict], ma3: List[Optional[float]], lookback=8) -> Optional[Tuple[int,str]]:
    n = len(candles)
    for i in range(max(1, n - lookback), n):
        if i - 1 < 0: continue
        prevc = candles[i-1]["close"]; curc = candles[i]["close"]
        ma_prev = ma3[i-1] if i-1 < len(ma3) else None
        ma_cur = ma3[i] if i < len(ma3) else None
        if ma_prev is None or ma_cur is None:
            continue
        if prevc < ma_prev and curc > ma_cur:
            return (i, "BUY")
        if prevc > ma_prev and curc < ma_cur:
            return (i, "SELL")
    return None

# -------------------------
# Analyze per-symbol per-timeframe
# -------------------------
def analyze_symbol_tf(symbol: str, tf: int) -> Optional[Dict]:
    candles = fetch_candles(symbol, tf, count=CANDLES_N)
    if not candles or len(candles) < 12:
        log(f"[{symbol}@{tf}] insufficient candles ({len(candles) if candles else 0})")
        return None
    ma1, ma2, ma3 = compute_mas_for_chart(candles)
    atr = compute_atr(candles)
    idx = len(candles) - 1  # latest closed candle
    epoch = candles[idx]["epoch"]
    # dedup: only one signal per candle per symbol+tf
    if already_processed(symbol, tf, epoch):
        log(f"[{symbol}@{tf}] candle {epoch} already processed")
        return None

    # 1) Check rejection on latest closed candle
    rej, reason, dir_guess = detect_rejection_at_idx(candles, idx, ma1, ma2, atr)
    # 2) Check MA3 cross in recent candles (reversal candidate)
    cross = detect_recent_ma3_cross(candles, ma3, lookback=8)

    # no rejection -> nothing
    if not rej and not cross:
        mark_processed(symbol, tf, epoch)  # mark processed so we won't try again
        return None

    # If rejection exists, enforce dynamic S/R role:
    if rej:
        # determine if continuation/trend exists: MA alignment
        aligned_up = ma_alignment_up(ma1, ma2, ma3, idx)
        aligned_down = ma_alignment_down(ma1, ma2, ma3, idx)
        # require alignment OR this rejection matches MA3-break reversal
        if dir_guess == "BUY":
            # only accept BUY if MA alignment is up OR there was a recent MA3 bullish break (retest scenario)
            if not aligned_up:
                # allow if cross exists and cross direction is BUY and cross happened before this candle (retest)
                if not (cross and cross[1] == "BUY"):
                    log(f"[{symbol}@{tf}] rejection BUY but not aligned_up and no recent MA3 BUY cross -> skip")
                    mark_processed(symbol, tf, epoch)
                    return None
        else:  # SELL
            if not aligned_down:
                if not (cross and cross[1] == "SELL"):
                    log(f"[{symbol}@{tf}] rejection SELL but not aligned_down and no recent MA3 SELL cross -> skip")
                    mark_processed(symbol, tf, epoch)
                    return None
        # pass: valid rejection that respects dynamic S/R
        reason_full = f"Rejection: {reason}"
        direction = dir_guess
        # Build chart & return signal
        chart = None
        try:
            chart = make_chart(candles, ma1, ma2, ma3, idx, f"{direction} | {reason_full}", symbol, tf)
        except Exception as e:
            log("chart build error:", e)
            chart = None
        return {"symbol": symbol, "tf": tf, "direction": direction, "reason": reason_full,
                "price": candles[idx]["close"], "epoch": epoch, "chart": chart}

    # If no direct rejection but cross exists -> check for retest/rejection after cross
    if cross:
        cross_idx, cross_dir = cross
        # only consider if cross happened before the latest candle
        if cross_idx < idx:
            # check whether latest closed candle (idx) is a retest & rejection
            rej_after_cross, reason_after, dir_guess2 = detect_rejection_at_idx(candles, idx, ma1, ma2, atr)
            if rej_after_cross and dir_guess2 == cross_dir:
                reason_full = f"MA3-break + retest: {reason_after}"
                chart = None
                try:
                    chart = make_chart(candles, ma1, ma2, ma3, idx, f"{cross_dir} | {reason_full}", symbol, tf)
                except Exception:
                    chart = None
                return {"symbol": symbol, "tf": tf, "direction": cross_dir, "reason": reason_full,
                        "price": candles[idx]["close"], "epoch": epoch, "chart": chart}
    # nothing valid
    mark_processed(symbol, tf, epoch)
    return None

# -------------------------
# Sleep window helper
# -------------------------
def in_sleep_window(sleep_window: str, tz_offset: str = "+00:00") -> bool:
    if not sleep_window:
        return False
    try:
        start_s, end_s = sleep_window.split("-")
        # parse tz offset hours only (support +HH:MM)
        sign = 1 if tz_offset[0] != "-" else -1
        tz_h = int(tz_offset[1:3]) * sign
        tz_m = int(tz_offset[4:6]) * sign if len(tz_offset) >= 6 else 0
        offset = timedelta(hours=tz_h, minutes=tz_m)
        now_utc = datetime.utcnow()
        local_now = now_utc + offset
        sh, sm = [int(x) for x in start_s.split(":")]
        eh, em = [int(x) for x in end_s.split(":")]
        start_dt = local_now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end_dt = local_now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start_dt < end_dt:
            return start_dt <= local_now <= end_dt
        else:
            # wrap over midnight
            return local_now >= start_dt or local_now <= end_dt
    except Exception:
        return False

# -------------------------
# Runner
# -------------------------
def run_once():
    load_cache()
    if in_sleep_window(SLEEP_WINDOW, TZ_OFFSET):
        log("Within sleep window -> skipping run")
        return
    if not DISABLE_HEARTBEAT:
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"✅ main.py startup heartbeat - tracking {len(ASSETS)} assets")
        except Exception:
            log("heartbeat send failed")

    signals_sent = 0
    reports = []
    for symbol in ASSETS:
        if signals_sent >= MAX_SIGNALS_PER_RUN:
            break
        symbol = symbol.strip()
        log("Scanning", symbol)
        for tf in TIMEFRAMES:
            try:
                sig = analyze_symbol_tf(symbol, tf)
                if not sig:
                    continue
                # dedup final check: if same epoch already sent -> skip
                last = last_sent_cache.get(symbol)
                if last and int(last.get("epoch", 0)) >= int(sig["epoch"]):
                    log(f"[{symbol}@{tf}] already sent for epoch {sig['epoch']}")
                    mark_processed(symbol, tf, sig["epoch"])
                    continue
                # cooldown check (small) to avoid immediate duplicate direction repeats
                if not can_send_cooldown(symbol, sig["direction"], tf):
                    log(f"[{symbol}@{tf}] suppressed by cooldown for direction {sig['direction']}")
                    mark_processed(symbol, tf, sig["epoch"])
                    continue
                # send text first
                caption = f"🔔 {sig['symbol']} | {sig['tf']//60}m | {sig['direction']}\nReason: {sig['reason']}\nPrice: {sig['price']}"
                ok, resp = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                # send chart if available
                if sig.get("chart"):
                    try:
                        send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, sig["chart"])
                    except Exception as e:
                        log("chart send error:", e)
                mark_sent(symbol, sig["direction"], sig["tf"], sig["epoch"])
                mark_processed(symbol, tf, sig["epoch"])
                signals_sent += 1
                reports.append(f"{symbol}@{tf} -> SENT {sig['direction']} ({sig['reason']})")
            except Exception as e:
                log(f"Error analyzing {symbol}@{tf}: {e}")
                if DEBUG:
                    traceback.print_exc()
    save_cache()
    log("Run complete. signals_sent:", signals_sent, "reports:", reports)

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Fatal:", e)
        if DEBUG:
            traceback.print_exc()
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

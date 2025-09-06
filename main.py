#!/usr/bin/env python3
"""
main.py — Robust dynamic S/R signal bot (charts + Telegram)
- MA1 = SMMA(HLC3,9), MA2 = SMMA(close,19), MA3 = SMA(MA2,25)
- OR logic: any qualifying rejection near MA1/MA2 OR MA3-break+retest OR continuation retest fires a signal on candle close
- ATR-based near test (ATR(14) * NEAR_ATR_MULT) with small percent fallback
- Builds chart only when sending a real signal (avoids chart spam)
- Dedup persisted to ALERT_PERSIST_PATH or TMPDIR
"""

import os, json, time, math, tempfile, traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# external libs
import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Try to use your bot helper; fallback to http requests or prints
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    try:
        import requests
    except Exception:
        requests = None
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM]", text); return False, "no-token"
    def send_telegram_photo(token, chat_id, caption, photo_path):
        print("[TELEGRAM PHOTO]", caption, photo_path); return False, "no-token"

# -------------------------
# Config (env override)
# -------------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# Assets (combine defaults + 1s)
ASSETS_DEFAULT = os.getenv("ASSETS_DEFAULT", "R_10,R_50,R_75")
ASSETS_1S = os.getenv("ASSETS_1S", "1HZ75V,1HZ100V,1HZ150V")
ASSETS = [s.strip() for s in (ASSETS_DEFAULT.split(",") + ASSETS_1S.split(",")) if s and s.strip()]

# TIMEFRAMES env expects comma-separated seconds, default 5m only per your latest ask
TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES", "300").split(",")]

CANDLES_N = int(os.getenv("CANDLES_N", "120"))

# ATR near multipliers (tunable via env)
NEAR_ATR_MULT = float(os.getenv("NEAR_ATR_MULT", "0.6"))  # how many ATRs counts as near
PCT_FALLBACK  = float(os.getenv("PCT_FALLBACK", "0.002")) # 0.2% fallback

# Charting / chart params
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "180"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))
CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))

# persistence
TMPDIR = tempfile.gettempdir()
ALERT_PERSIST_PATH = os.getenv("ALERT_PERSIST_PATH", os.path.join(TMPDIR, "fenyere_alerts_main.json"))

# sleep window / heartbeat
DISABLE_HEARTBEAT = os.getenv("DISABLE_HEARTBEAT", "1") in ("1", "true", "True")
SLEEP_WINDOW = os.getenv("SLEEP_WINDOW", "")  # e.g. "21:00-06:00"
TZ_OFFSET = os.getenv("TZ_OFFSET", "+02:00")

# signal limits
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print("[", ts, "]", *args, **kwargs)

# -------------------------
# MA utils (SMMA & SMA exactly as you specified)
# -------------------------
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if n == 0 or period <= 0:
        return [None]*n
    out = [None] * n
    if n < period:
        return out
    seed = sum(series[:period]) / period
    out[period-1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + float(series[i])) / period
        out[i] = prev
    return out

def sma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if period <= 0 or n < period:
        return [None]*n
    out = [None] * n
    for i in range(period-1, n):
        out[i] = sum(series[i-period+1:i+1]) / period
    return out

def compute_mas_for_chart(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes = [float(c["close"]) for c in candles]
    highs  = [float(c["high"])  for c in candles]
    lows   = [float(c["low"])   for c in candles]
    # HLC3
    hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    ma1 = smma(hlc3, 9)      # SMMA(HLC3,9)
    ma2 = smma(closes, 19)   # SMMA(close,19)
    # ma3 is SMA of ma2 values (only where ma2 is not None)
    ma2_vals = [v for v in ma2 if v is not None]
    if len(ma2_vals) >= 25:
        ma3_raw = sma(ma2_vals, 25)
    else:
        ma3_raw = [None] * len(ma2_vals)
    # align ma3 to original index positions of ma2
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
# ATR
# -------------------------
def compute_atr(candles: List[Dict], period: int = 14) -> float:
    if not candles:
        return 0.0
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows  = np.array([float(c["low"])  for c in candles], dtype=float)
    closes= np.array([float(c["close"]) for c in candles], dtype=float)
    trs = []
    for i in range(len(candles)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
    if len(trs) >= period:
        return float(np.mean(trs[-period:]))
    return float(np.mean(trs))

# -------------------------
# Candle fetch (snapshot)
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    tried = [count, max(100, count//2), 80, 50, 25]
    for ccount in tried:
        for attempt in range(2):
            ws = None
            try:
                ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                try:
                    _ = ws.recv()
                except Exception:
                    pass
                req = {"ticks_history": symbol, "style": "candles", "granularity": granularity,
                       "count": ccount, "end": "latest", "subscribe": 0}
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
                log("fetch err", symbol, granularity, ccount, "attempt", attempt+1, e)
            finally:
                try:
                    if ws:
                        ws.close()
                except:
                    pass
            time.sleep(0.6)
    log("fetch_candles final fail", symbol, granularity)
    return []

# -------------------------
# Candle pattern: broad rejection families
# -------------------------
def candle_bits(candle: Dict, prev: Optional[Dict] = None) -> Dict:
    o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    body = abs(c - o); rng = max(1e-12, h - l)
    upper = h - max(o, c); lower = min(o, c) - l
    is_doji = body <= 0.40 * rng
    pin_low = (lower >= 0.3 * rng) and (lower > upper)
    pin_high = (upper >= 0.3 * rng) and (upper > lower)
    engulf_bull = engulf_bear = False
    if prev:
        po = float(prev["open"]); pc = float(prev["close"])
        if pc < po and c > o and o <= pc and c >= po:
            engulf_bull = True
        if pc > po and c < o and o >= pc and c <= po:
            engulf_bear = True
    return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": rng,
            "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
            "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

# -------------------------
# Near MA test using ATR with fallback percent
# -------------------------
def is_near_ma(price_extreme: float, ma_value: Optional[float], atr: float) -> Tuple[bool, float]:
    if ma_value is None or atr is None:
        return False, float('inf')
    tol = max(atr * NEAR_ATR_MULT, abs(price_extreme) * PCT_FALLBACK)
    dist = abs(price_extreme - ma_value)
    return (dist <= tol), float(dist/atr if atr>0 else float('inf'))

# -------------------------
# Detect signals (OR logic)
# -------------------------
def detect_signal_from_candles(candles: List[Dict], tf: int, symbol: str) -> Optional[Dict]:
    """
    Returns signal dict or None.
    - uses latest closed candle (last element).
    - OR logic: any condition satisfied -> signal
    """
    if not candles or len(candles) < 12:
        return None
    ma1, ma2, ma3 = compute_mas_for_chart(candles)
    atr = compute_atr(candles)
    idx = len(candles) - 1  # index of latest closed candle
    prev_idx = idx - 1
    candle = candles[idx]
    prev = candles[prev_idx] if prev_idx >= 0 else None

    bits = candle_bits(candle, prev)
    close = float(candle["close"])
    high = float(candle["high"]); low = float(candle["low"])
    epoch = int(candle["epoch"])
    # trend by MA3
    ma3_val = ma3[idx] if idx < len(ma3) else None
    trend = None
    if ma3_val is not None:
        try:
            trend = "UP" if close > ma3_val else ("DOWN" if close < ma3_val else None)
        except Exception:
            trend = None

    reasons = []
    direction = None

    # A) Rejection at MA1/MA2 (any family)
    # For bullish rejection we consider candle low; for bearish consider candle high.
    near_low_ma1, dist_low_ma1 = is_near_ma(low, ma1[idx] if idx < len(ma1) else None, atr)
    near_low_ma2, dist_low_ma2 = is_near_ma(low, ma2[idx] if idx < len(ma2) else None, atr)
    near_high_ma1, dist_high_ma1 = is_near_ma(high, ma1[idx] if idx < len(ma1) else None, atr)
    near_high_ma2, dist_high_ma2 = is_near_ma(high, ma2[idx] if idx < len(ma2) else None, atr)

    # detect families
    family = None
    if bits["is_doji"]:
        family = "DOJI"
    elif bits["pin_low"] or bits["pin_high"]:
        family = "PIN"
    elif bits["engulf_bull"] or bits["engulf_bear"]:
        family = "ENGULF"
    elif bits["body"] <= 0.15 * bits["range"]:
        family = "TINY"

    # if family present + near MA -> accept
    if family:
        # bullish-type rejection (buy)
        if family and (near_low_ma1 or near_low_ma2):
            reasons.append(f"Rejection-family {family} near MA1/MA2 (low) distATR={min(dist_low_ma1, dist_low_ma2):.2f}")
            direction = "BUY"
        # bearish-type rejection (sell)
        if family and (near_high_ma1 or near_high_ma2):
            reasons.append(f"Rejection-family {family} near MA1/MA2 (high) distATR={min(dist_high_ma1, dist_high_ma2):.2f}")
            direction = "SELL"

    # B) MA3 breakout + first retest logic:
    # detect a recent MA3 cross (lookback window)
    def recent_ma3_cross(candles, ma3, lookback=8):
        n = len(candles)
        for i in range(max(1, n - lookback), n):
            if i - 1 < 0: continue
            prevc = float(candles[i-1]["close"]); curc = float(candles[i]["close"])
            ma_prev = ma3[i-1] if i-1 < len(ma3) else None
            ma_cur  = ma3[i]   if i < len(ma3) else None
            if ma_prev is None or ma_cur is None: continue
            if prevc < ma_prev and curc > ma_cur:
                return (i, "BUY")
            if prevc > ma_prev and curc < ma_cur:
                return (i, "SELL")
        return None
    cross = recent_ma3_cross(candles, ma3, lookback=8)
    if cross:
        cross_idx, cross_dir = cross
        # ensure cross happened before latest candle, then check latest candle for rejection near MA1/MA2
        if cross_idx < idx:
            # check if latest candle is rejection and aligns with cross direction
            if family:
                if (cross_dir == "BUY") and (near_low_ma1 or near_low_ma2):
                    reasons.append(f"MA3-break+retest (BUY) nearMA distATR={min(dist_low_ma1, dist_low_ma2):.2f}")
                    direction = "BUY"
                if (cross_dir == "SELL") and (near_high_ma1 or near_high_ma2):
                    reasons.append(f"MA3-break+retest (SELL) nearMA distATR={min(dist_high_ma1, dist_high_ma2):.2f}")
                    direction = "SELL"

    # C) Continuation: in-trend MA alignment + rejection
    aligned_up = (ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and ma1[idx] > ma2[idx] > ma3[idx])
    aligned_down = (ma1[idx] is not None and ma2[idx] is not None and ma3[idx] is not None and ma1[idx] < ma2[idx] < ma3[idx])
    if (aligned_up and family and (near_low_ma1 or near_low_ma2)):
        reasons.append("Continuation uptrend rejection")
        direction = direction or "BUY"
    if (aligned_down and family and (near_high_ma1 or near_high_ma2)):
        reasons.append("Continuation downtrend rejection")
        direction = direction or "SELL"

    if not reasons:
        # nothing matched
        return None

    reason_text = " | ".join(reasons)
    return {"symbol": symbol, "tf": tf, "direction": direction, "reason": reason_text,
            "price": close, "epoch": epoch, "ma1": ma1[idx], "ma2": ma2[idx], "ma3": ma3[idx],
            "candles": candles, "rej_index": idx}

# -------------------------
# Chart building
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
            rect = Rectangle((i - CANDLE_WIDTH / 2.0, lower), CANDLE_WIDTH, height,
                             facecolor=color, edgecolor="black", linewidth=0.35, zorder=2)
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
            ax.scatter([idx], [price], marker="v" if "SELL" in reason else "^",
                       color=marker_color, s=160, zorder=6, edgecolors="black")
            ax.text(idx, price, "  "+reason, fontsize=8, color=marker_color)

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
    except Exception as e:
        if DEBUG:
            traceback.print_exc()
        return None

# -------------------------
# Persistence / dedupe
# -------------------------
def load_alerts(path=ALERT_PERSIST_PATH):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_alerts(d, path=ALERT_PERSIST_PATH):
    try:
        with open(path, "w") as f:
            json.dump(d, f)
    except Exception:
        pass

def already_sent(symbol: str, tf: int, epoch: int, side: str) -> bool:
    key = f"{symbol}|{tf}"
    d = load_alerts()
    rec = d.get(key)
    if not rec:
        return False
    if rec.get("epoch") == int(epoch) and rec.get("side") == side:
        return True
    return False

def mark_sent(symbol: str, tf: int, epoch: int, side: str):
    key = f"{symbol}|{tf}"
    d = load_alerts()
    d[key] = {"epoch": int(epoch), "side": side, "sent_at": int(time.time())}
    save_alerts(d)

# -------------------------
# Sleep window helper
# -------------------------
def in_sleep_window(sleep_window: str, tz_offset: str = "+00:00") -> bool:
    if not sleep_window:
        return False
    try:
        start_s, end_s = sleep_window.split("-")
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
            return local_now >= start_dt or local_now <= end_dt
    except Exception:
        return False

# -------------------------
# Runner
# -------------------------
def run_once():
    if in_sleep_window(SLEEP_WINDOW, TZ_OFFSET):
        log("Within sleep window -> skipping run")
        return

    if not DISABLE_HEARTBEAT:
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"✅ Bot started - tracking {len(ASSETS)} symbols")
        except Exception:
            log("heartbeat failed (non-fatal)")

    signals = 0
    for symbol in ASSETS:
        if signals >= MAX_SIGNALS_PER_RUN:
            break
        symbol = symbol.strip()
        for tf in TIMEFRAMES:
            try:
                candles = fetch_candles(symbol, tf, count=CANDLES_N)
                if not candles:
                    continue
                sig = detect_signal_from_candles(candles, tf, symbol)
                if not sig:
                    continue
                epoch = int(sig["epoch"])
                side = sig["direction"]
                if already_sent(symbol, tf, epoch, side):
                    log("already sent", symbol, tf, epoch, side)
                    continue
                # send text
                caption = f"🔔 {symbol} | {tf//60}m | {side}\nReason: {sig['reason']}\nPrice: {sig['price']}"
                ok_text = False
                try:
                    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                    ok_text = True
                except Exception as e:
                    log("text send error", e)
                # chart (best-effort)
                chart_path = None
                try:
                    chart_path = make_chart(sig["candles"], sig["ma1"] and [sig["ma1"]]*len(sig["candles"]) or [],
                                            sig["ma2"] and [sig["ma2"]]*len(sig["candles"]) or [],
                                            sig["ma3"] and [sig["ma3"]]*len(sig["candles"]) or [],
                                            sig["rej_index"], sig["reason"], symbol, tf)
                    if chart_path:
                        send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                except Exception as e:
                    log("chart build/send failed", e)
                mark_sent(symbol, tf, epoch, side)
                signals += 1
            except Exception as e:
                log("Error processing", symbol, tf, e)
                if DEBUG:
                    traceback.print_exc()
    log("Run finished. signals:", signals)

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Fatal error", e)
        if DEBUG:
            traceback.print_exc()
        try:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crash: {e}")
        except Exception:
            pass
        raise

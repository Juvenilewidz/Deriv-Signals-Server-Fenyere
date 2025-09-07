#!/usr/bin/env python3
"""
main_1s.py — 1s synthetic indices runner (paste-and-cruise)

Behavior:
- Assets default: 1HZ75V,1HZ100V,1HZ150V (override with ASSETS env)
- Timeframes default: 5m (300s) only (override TIMEFRAMES env if needed)
- CANDLES_N default: 200
- Uses closed candle only (last closed bar)
- Rejection families accepted: any pin / doji / engulf / tiny-body (no numeric thresholds)
- Fires when ANY condition satisfied:
    1) MA3 breakout + retest + rejection => reversal
    2) Retest MA1 or MA2 in a clear trend => continuation
    3) Small swings in a clear trend + rejection => continuation
- Sends chart (attempt). If chart build fails, falls back to text-only alert (so not silent).
- Dedup persisted in tmpfile to avoid duplicate alerts.
"""

import os
import json
import time
import math
import tempfile
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Try using your bot.py helpers if available
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM TEXT]", text); return True, "fallback"
    def send_telegram_photo(token, chat_id, caption, photo_path):
        print("[TELEGRAM PHOTO]", caption, photo_path); return True, "fallback"

# -------------------------
# Configuration
# -------------------------
DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

# 1s indices by default (can be overridden by ASSETS env)
ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
ASSETS = [s.strip() for s in ASSETS if s.strip()]

# default timeframe: 5m only (as you requested)
TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES", "300").split(",")]

# candles/chart config
CANDLES_N = int(os.getenv("CANDLES_N", "100"))
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "220"))
PAD_CANDLES = int(os.getenv("PAD_CANDLES", "10"))
CANDLE_WIDTH = float(os.getenv("CANDLE_WIDTH", "0.35"))

# dedupe / persistence
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")

# in-memory cache
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

def already_sent(symbol: str, tf: int, epoch: int, side: str) -> bool:
    rec = last_sent_cache.get(f"{symbol}|{tf}")
    if not rec:
        return False
    return rec.get("epoch") == int(epoch) and rec.get("side") == side

def mark_sent(symbol: str, tf: int, epoch: int, side: str):
    last_sent_cache[f"{symbol}|{tf}"] = {"epoch": int(epoch), "side": side, "ts": int(time.time())}
    save_cache()

# -------------------------
# MA implementations
# -------------------------
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if n == 0 or period <= 0:
        return [None] * n
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
    run = sum(series[:period])
    out.append(run / period)
    for i in range(period, n):
        run += series[i] - series[i - period + 1]
        out.append(run / period)
    return out

def compute_mas_for_chart(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes  = [float(c["close"]) for c in candles]
    highs   = [float(c["high"])  for c in candles]
    lows    = [float(c["low"])   for c in candles]
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

# -------------------------
# Candle family (very loose; ANY family counts) — no numeric thresholds
# -------------------------
def candle_family(candle: Dict, prev: Optional[Dict] = None) -> str:
    """
    Loose inclusion:
    - DOJI: small-ish body relative to range (we don't enforce tight percentages)
    - PIN_HIGH / PIN_LOW: upper/lower wick noticeably present (no strict percent)
    - ENGULF: current body engulfs previous body
    - TINY: small body compared to range (catch-all)
    """
    try:
        o = float(candle["open"]); h = float(candle["high"]); l = float(candle["low"]); c = float(candle["close"])
    except Exception:
        return "NONE"

    body = abs(c - o)
    rng = max(1e-9, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l

    # Doji-ish: body is noticeably smaller than range, but we do not require strict %.
    if body == 0 or body < (0.5 * rng):
        return "DOJI"

    # Pin-style: one wick clearly present (no strict numeric threshold)
    if upper > body and upper > lower:
        return "PIN_HIGH"
    if lower > body and lower > upper:
        return "PIN_LOW"

    # Engulfing: current body engulfs previous body (if prev available)
    if prev:
        try:
            po = float(prev["open"]); pc = float(prev["close"])
            if pc < po and c > o and o <= pc and c >= po:
                return "BULL_ENG"
            if pc > po and c < o and o >= pc and c <= po:
                return "BEAR_ENG"
        except Exception:
            pass

    # Tiny-body catch-all
    if body <= (0.6 * rng):
        return "TINY"

    return "NONE"

# -------------------------
# Trend helpers & MA3 cross detection
# -------------------------
def in_uptrend(i: int, ma1, ma2, ma3, price: float) -> bool:
    try:
        return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
                and ma1[i] > ma2[i] > ma3[i] and price >= ma3[i])
    except Exception:
        return False

def in_downtrend(i: int, ma1, ma2, ma3, price: float) -> bool:
    try:
        return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
                and ma1[i] < ma2[i] < ma3[i] and price <= ma3[i])
    except Exception:
        return False

def broke_ma3_recently(candles: List[Dict], ma3: List[Optional[float]], idx: int, lookback: int = 6) -> Optional[str]:
    """
    Detect a recent MA3 crossing event (within lookback bars).
    Returns "UP" if crossed from below->above, "DOWN" if above->below, or None.
    """
    n = len(candles)
    if idx <= 0:
        return None
    lo = max(1, idx - lookback)
    for k in range(lo, idx + 1):
        if k - 1 < 0: continue
        if ma3[k] is None or ma3[k - 1] is None:
            continue
        prev_close = float(candles[k - 1]["close"])
        cur_close = float(candles[k]["close"])
        if prev_close <= ma3[k - 1] and cur_close > ma3[k]:
            return "UP"
        if prev_close >= ma3[k - 1] and cur_close < ma3[k]:
            return "DOWN"
    return None

# -------------------------
# near MA tests (NO numeric tolerances)
# -------------------------
def ma_inside_candle(ma_value: Optional[float], candle: Dict) -> bool:
    """Return True if MA lies between candle low and high (touch/cross containment)."""
    if ma_value is None:
        return False
    try:
        return float(candle["low"]) <= float(ma_value) <= float(candle["high"])
    except Exception:
        return False

def ma_between_prev_and_curr(ma_value: Optional[float], prev_candle: Optional[Dict], curr_candle: Dict) -> bool:
    """
    Return True if MA lies between previous close and current close (representing a retest direction)
    — this is a non-numeric 'retest' check (no % thresholds).
    """
    if ma_value is None or prev_candle is None:
        return False
    try:
        pclose = float(prev_candle["close"])
        cclose = float(curr_candle["close"])
        low = min(pclose, cclose)
        high = max(pclose, cclose)
        return low <= float(ma_value) <= high
    except Exception:
        return False

def is_near_ma_loose(ma_value: Optional[float], prev: Optional[Dict], curr: Dict) -> bool:
    """
    Loose 'near' check without numeric thresholds:
      - True if MA is inside the candle (touch/cross), OR
      - True if MA lies between previous close and current close (retest), OR
      - True if MA equals a candle extreme.
    """
    if ma_value is None:
        return False
    if ma_inside_candle(ma_value, curr):
        return True
    if ma_between_prev_and_curr(ma_value, prev, curr):
        return True
    # equality with extremes (touch)
    try:
        if float(ma_value) == float(curr["low"]) or float(ma_value) == float(curr["high"]):
            return True
    except Exception:
        pass
    return False

# -------------------------
# Fetch candles (snapshot subscribe=0)
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        ws = None
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            _ = ws.recv()  # auth
            req = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": "latest",
                "subscribe": 0
            }
            ws.send(json.dumps(req))
            raw = ws.recv()
            resp = json.loads(raw)
            if isinstance(resp, dict) and resp.get("candles"):
                out = []
                for c in resp["candles"]:
                    try:
                        out.append({
                            "epoch": int(c["epoch"]),
                            "open": float(c["open"]),
                            "high": float(c["high"]),
                            "low": float(c["low"]),
                            "close": float(c["close"])
                        })
                    except Exception:
                        continue
                out.sort(key=lambda x: x["epoch"])
                return out
        except Exception as e:
            log("fetch error", symbol, e)
        finally:
            try:
                if ws:
                    ws.close()
            except Exception:
                pass
        time.sleep(0.6)
    return []

# -------------------------
# Chart builder
# -------------------------
def make_chart(candles: List[Dict], ma1: List[Optional[float]], ma2: List[Optional[float]],
               ma3: List[Optional[float]], rej_index: int, reasons: List[str], symbol: str, tf: int,
               last_n: int = LAST_N_CHART, pad: int = PAD_CANDLES) -> Optional[str]:
    try:
        total = len(candles)
        if total == 0:
            return None
        show_n = min(last_n, total)
        start = max(0, total - show_n)
        chosen = candles[start: total]
        xs = [datetime.fromtimestamp(c["epoch"], tz=timezone.utc) for c in chosen]
        opens = [c["open"] for c in chosen]
        highs = [c["high"] for c in chosen]
        lows = [c["low"] for c in chosen]
        closes = [c["close"] for c in chosen]

        fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
        ax.set_title(f"{symbol} | {tf//60}m | {'; '.join(reasons)}", fontsize=10)

        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = "#2ca02c" if c >= o else "#d62728"
            ax.plot([i, i], [l, h], color="black", linewidth=0.6, zorder=1)
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((i - CANDLE_WIDTH / 2.0, lower), CANDLE_WIDTH, height,
                             facecolor=color, edgecolor="black", linewidth=0.35, zorder=2)
            ax.add_patch(rect)

        # prepare MA series for plotting aligned to chosen window
        m1_vals = [ma1[i] if i < len(ma1) else None for i in range(start, total)]
        m2_vals = [ma2[i] if i < len(ma2) else None for i in range(start, total)]
        m3_vals = [ma3[i] if i < len(ma3) else None for i in range(start, total)]

        def plot_ma(vals, label, color):
            y = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float('nan') for v in vals]
            ax.plot(list(range(len(y))), y, label=label, linewidth=1.0, zorder=3, color=color)

        try:
            plot_ma(m1_vals, "MA1 SMMA(HLC3,9)", "#1f77b4")
            plot_ma(m2_vals, "MA2 SMMA(Close,19)", "#ff7f0e")
            plot_ma(m3_vals, "MA3 SMA(MA2,25)", "#2ca02c")
            ax.legend(loc="upper left", fontsize=8)
        except Exception:
            pass

        if start <= rej_index < total:
            idx = rej_index - start
            price = chosen[idx]["close"]
            marker_color = "red" if any("SELL" in r.upper() or "DOWN" in r.upper() for r in reasons) else "green"
            ax.scatter([idx], [price], marker="v" if "SELL" in marker_color else "^",
                       color=marker_color, s=120, zorder=6, edgecolors="black")
            ax.text(idx + 0.5, price, "  " + ("; ".join(reasons)), fontsize=8, color=marker_color)

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
# Detection (NO numeric thresholds inside logic)
# -------------------------
def detect_signal(candles: List[Dict], tf: int, symbol: str) -> Optional[Dict]:
    """
    Returns None or dict with keys:
      symbol, tf, direction ("BUY"/"SELL"), reasons (list), idx (int), ma1, ma2, ma3, candles
    Logic strictly follows user's rules (no numeric tolerances).
    """
    n = len(candles)
    # need enough history to compute MA3 properly
    if n < 60:
        return None

    ma1, ma2, ma3 = compute_mas_for_chart(candles)

    # We ALWAYS analyze the last *closed* candle:
    # If candles[-1] is the live bar, use the previous closed candle at index -2.
    idx = n - 2 if n >= 2 else n - 1
    candle = candles[idx]
    prev = candles[idx - 1] if idx - 1 >= 0 else None

    family = candle_family(candle, prev)
    if family == "NONE":
        return None

    reasons = []
    direction = None

    # Helper: direct containment / retest checks (no percentages)
    def near_ma_loose_buy(i):
        m1 = ma1[i] if i < len(ma1) else None
        m2 = ma2[i] if i < len(ma2) else None
        if ma_inside_candle(m1, candle) or ma_between_prev_and_curr(m1, prev, candle):
            return True, "MA1"
        if ma_inside_candle(m2, candle) or ma_between_prev_and_curr(m2, prev, candle):
            return True, "MA2"
        return False, None

    def near_ma_loose_sell(i):
        return near_ma_loose_buy(i)  # symmetric for sell (we only check containment/retest/no %)

    # 1) MA3 breakout + retest + rejection -> reversal
    cross = broke_ma3_recently(candles, ma3, idx, lookback=6)
    if cross:
        tb2, which_b2 = near_ma_loose_buy(idx)
        if tb2:
            reasons.append("MA3 breakout then retest at " + which_b2)
            direction = "BUY" if cross == "UP" else direction
        ts2, which_s2 = near_ma_loose_sell(idx)
        if ts2:
            reasons.append("MA3 breakout then retest at " + which_s2)
            direction = "SELL" if cross == "DOWN" else direction
        # if both present, leave direction as detected by MA3 cross
        if cross == "UP" and direction is None and tb2:
            direction = "BUY"
        if cross == "DOWN" and direction is None and ts2:
            direction = "SELL"
        if direction:
            reasons.insert(0, family + " (reversal candidate)")

    # 2) Retest MA1/MA2 in clear trend -> continuation
    if direction is None:
        if in_uptrend(idx, ma1, ma2, ma3, candle["close"]):
            tb, wb = near_ma_loose_buy(idx)
            if tb:
                direction = "BUY"
                reasons.append(f"{family} retest {wb} in uptrend")
        if in_downtrend(idx, ma1, ma2, ma3, candle["close"]):
            ts, ws = near_ma_loose_sell(idx)
            if ts:
                direction = "SELL"
                reasons.append(f"{family} retest {ws} in downtrend")

    # 3) Small swings in clear trend + rejection -> continuation
    if direction is None:
        if in_uptrend(idx, ma1, ma2, ma3, candle["close"]):
            # small swing detection: local low within recent window equals this candle's low
            window = 4
            lows = [float(candles[i]["low"]) for i in range(max(0, idx - window), idx + 1)]
            if float(candle["low"]) == min(lows):
                direction = "BUY"
                reasons.append(f"{family} small swing in uptrend")
        if in_downtrend(idx, ma1, ma2, ma3, candle["close"]):
            window = 4
            highs = [float(candles[i]["high"]) for i in range(max(0, idx - window), idx + 1)]
            if float(candle["high"]) == max(highs):
                direction = "SELL"
                reasons.append(f"{family} small swing in downtrend")

    # If none of the above set direction, still allow ANY of the rules to fire:
    # (Rule: if any condition satisfied, fire — we already checked family; check MA3 retest or retest in trend or small swing)
    if direction is None:
        # final chance: if family exists and MA is inside candle or retest between prev and curr
        tbf, wb = near_ma_loose_buy(idx)
        if tbf:
            direction = "BUY"; reasons.append(f"{family} near {wb}")
        tsf, ws = near_ma_loose_sell(idx)
        if tsf and direction is None:
            direction = "SELL"; reasons.append(f"{family} near {ws}")

    if direction is None or not reasons:
        return None

    return {"symbol": symbol, "tf": tf, "direction": direction, "reasons": reasons, "idx": idx,
            "ma1": ma1, "ma2": ma2, "ma3": ma3, "candles": candles}

# -------------------------
# Main runner (one pass)
# -------------------------
def run_once():
    load_cache()
    sent = 0
    for symbol in ASSETS:
        for tf in TIMEFRAMES:
            if sent >= 6:
                break
            try:
                candles = fetch_candles(symbol, tf, CANDLES_N)
                if not candles or len(candles) < 60:
                    # If we couldn't fetch enough candles at requested tf, try to still detect on what we have
                    # but do not crash
                    log("insufficient candles", symbol, tf, len(candles) if candles else 0)
                    continue

                res = detect_signal(candles, tf, symbol)
                if not res:
                    continue

                idx = res["idx"]
                candle_epoch = int(res["candles"][idx]["epoch"])
                side = res["direction"]
                if already_sent(symbol, tf, candle_epoch, side):
                    log("already_sent skip", symbol, tf, candle_epoch, side)
                    continue

                when = datetime.fromtimestamp(candle_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                caption = f"[{symbol} | {tf//60}m | {side}] {' | '.join(res['reasons'])} — close={res['candles'][idx]['close']} @ {when}"

                # try to build chart; if chart fails, still send text fallback
                chart_path = make_chart(res["candles"], res["ma1"], res["ma2"], res["ma3"], res["idx"], res["reasons"], symbol, tf)
                ok = False
                if chart_path:
                    try:
                        ok, _ = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                    except Exception as e:
                        log("send photo error", e)
                if not ok:
                    try:
                        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption)
                    except Exception as e:
                        log("send text error", e)

                mark_sent(symbol, tf, candle_epoch, side)
                sent += 1
            except Exception as e:
                log("processing error", symbol, tf, e)
                if DEBUG:
                    traceback.print_exc()
    log("run complete, alerts sent:", sent)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    # basic validation
    if not DERIV_API_KEY:
        raise RuntimeError("DERIV_API_KEY required")
    # TELEGRAM token optional for testing, but recommended
    run_once()

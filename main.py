# main.py
"""
Paste-and-cruise main bot file.

Features:
 - 5m / 10m / 15m TFs (change TIMEFRAMES list below if you'd like different).
 - dynamic S/R rejection logic (MA1/MA2 zones).
 - stricter consolidation filter / ATR checks.
 - dedupe & cooldown for signals.
 - small, zoomed candlestick charts with 10-candle padding and MA overlays.
 - send charts for rejected setups & for accepted signals.
 - concise heartbeat message (configurable / can be disabled).
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

# try to import bot helpers (send_telegram_message, send_single_timeframe_signal, send_strong_signal)
try:
    from bot import (
        send_telegram_message,
        send_single_timeframe_signal,
        send_strong_signal,
        send_telegram_photo
    )
except Exception:
    # fallback stubs so importing main doesn't crash if bot.py hasn't been patched yet.
    def send_telegram_message(token, chat_id, text): print("[TELEGRAM]", text)
    def send_single_timeframe_signal(symbol, tf, direction, reason): print("SIG", symbol, tf, direction, reason)
    def send_strong_signal(symbol, direction, details): print("STRONG", symbol, direction, details)
    def send_telegram_photo(token, chat_id, caption, path): print("PHOTO", path, caption); return True, "ok"

# ===== ENV & GLOBALS =====
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

DEBUG = os.getenv("DEBUG", "1") in ("1", "true", "True", "yes", "on")
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "2"))  # 0 to disable
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN_SECS", "300"))  # per symbol/tf cooldown seconds

# Assets / timeframes
ASSETS = os.getenv("ASSETS", "R_10,R_50,R_75,1HZ75V,1HZ100V,1HZ150V").split(",")
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = int(os.getenv("CANDLES_N", "240"))  # more history (used for charting)

TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "deriv_bot_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "deriv_bot_last_heartbeat.json")

# in-memory state
last_sent_signal_by_symbol: Dict[str, Tuple[int, str, int]] = {}  # symbol -> (tf, direction, ts)
last_heartbeat_sent = 0

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{ts}]", *args, **kwargs)

# ===== MAs & helpers =====
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
    out = [None] * (period - 1)
    out.append(seed)
    prev = seed
    for i in range(period, n):
        val = (prev * (period - 1) + series[i]) / period
        out.append(val)
        prev = val
    return out

def typical_price(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0

# ===== Candle fetching (Deriv websocket client) =====
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
    """
    Fetch candles including the live (forming) candle. Returns list of dicts with keys:
    epoch, open, high, low, close
    """
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=10)
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        _ = json.loads(ws.recv())
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "subscribe": 0
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
        try:
            # try a quick update to include latest forming candle
            ws.settimeout(1.0)
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
        finally:
            try:
                ws.close()
            except:
                pass
        return out
    except Exception as e:
        log("fetch_candles error:", e)
        return []

# ===== Charting (candlesticks) =====
def make_chart(candles: List[Dict], ma1: List[Optional[float]], ma2: List[Optional[float]],
               ma3: List[Optional[float]], rej_index: int, reason: str, symbol: str, tf: int,
               last_n: int = 80, pad: int = 10) -> Optional[str]:
    """
    Draw candlestick chart PNG and return path. last_n = number of candles to show (excluding padding),
    pad = additional candles of padding at the right (future area).
    """
    try:
        # choose slice
        total = len(candles)
        show_n = min(last_n, total)
        start = max(0, total - show_n - pad)
        chosen = candles[start: total]
        xs = [datetime.utcfromtimestamp(c["epoch"]) for c in chosen]

        opens = [c["open"] for c in chosen]
        highs = [c["high"] for c in chosen]
        lows = [c["low"] for c in chosen]
        closes = [c["close"] for c in chosen]

        # figure size: wider to make candles visually small; dpi smaller to reduce file size
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        ax.set_title(f"{symbol} | {tf//60}m | {reason[:20]}".center(60))
        ax.grid(axis='y', alpha=0.25)

        # candlesticks: draw rectangles + wicks
        bar_width = 0.6 * (1.0)  # relative
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            x = i
            color = "#2ECC71" if c >= o else "#E74C3C"  # green/red
            # wick
            ax.plot([x, x], [l, h], color=color, linewidth=0.8)
            # body
            lower = min(o, c)
            height = max(1e-9, abs(c - o))
            rect = Rectangle((x - 0.3, lower), 0.6, height, facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        # convert MA arrays corresponding to chosen window
        ma1_vals = []
        ma2_vals = []
        ma3_vals = []
        for i in range(start, total):
            ma1_vals.append(ma1[i] if i < len(ma1) else None)
            ma2_vals.append(ma2[i] if i < len(ma2) else None)
            ma3_vals.append(ma3[i] if i < len(ma3) else None)

        # plot MA lines if finite
        def plot_ma(vals, label, color):
            y = [v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else None for v in vals]
            xs2 = list(range(len(y)))
            if any(v is not None for v in y):
                ax.plot(xs2, y, label=label, linewidth=1.2, color=color, alpha=0.9)

        plot_ma(ma1_vals, "MA1 (SMMA9 HLC3)", "#1f77b4")
        plot_ma(ma2_vals, "MA2 (SMMA19 Close)", "#ff7f0e")
        plot_ma(ma3_vals, "MA3 (SMA25 MA2)", "#2ca02c")

        # aesthetics: x axis ticks and limits
        ax.set_xlim(-1, len(chosen) + pad/2)
        ax.set_xticks(range(0, len(chosen), max(1, len(chosen)//8)))
        ax.set_xticklabels([xs[i].strftime("%H:%M\n%m-%d") for i in range(0, len(xs), max(1, len(xs)//8))])
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=8)

        # save
        out_path = os.path.join(TMPDIR, f"chart_{symbol}_{tf}_{int(time.time())}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
    except Exception as e:
        log("make_chart error:", e)
        traceback.print_exc()
        return None

# ===== Dynamic S/R / rejection logic (core) =====
def signal_for_timeframe(candles, granularity, i_rej, i_con):
    """
    Returns (direction, reason, extra) where direction in {"BUY","SELL", None}
    """
    try:
        if not candles or len(candles) < 60:
            return None, "insufficient_history", {}

        opens = np.array([c["open"] for c in candles], dtype=float)
        highs = np.array([c["high"] for c in candles], dtype=float)
        lows  = np.array([c["low"] for c in candles], dtype=float)
        closes= np.array([c["close"] for c in candles], dtype=float)
        typical = (highs + lows + closes) / 3.0
        last_price = float(closes[-1])

        # compute MAs
        ma1 = smma(list(typical), 9)
        ma2 = smma(list(closes), 19)
        ma3 = sma([x for x in ma2 if x is not None], 25)
        # produce array-aligned ma3
        ma3_full = []
        j = 0
        for v in ma2:
            if v is None:
                ma3_full.append(None)
            else:
                ma3_full.append(ma3[j] if j < len(ma3) else None)
                j += 1

        # get indices
        if i_rej is None or i_con is None:
            return None, "invalid_indices", {}
        if i_rej < 0 or i_rej >= len(candles):
            return None, "invalid_rejection_index", {}
        if i_con < 0 or i_con >= len(candles):
            return None, "invalid_confirmation_index", {}

        ma1_rej = ma1[i_rej] if i_rej < len(ma1) else None
        ma2_rej = ma2[i_rej] if i_rej < len(ma2) else None
        ma3_rej = ma3_full[i_rej] if i_rej < len(ma3_full) else None

        # ATR
        rngs = highs - lows
        atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
        tiny = max(1e-9, 0.05 * atr)

        # strict consolidation filter: recent avg body must be significantly smaller than ATR
        bodies = np.abs(closes - opens)
        avg_body_10 = float(np.mean(bodies[-10:])) if len(bodies) >= 10 else float(np.mean(bodies))
        if avg_body_10 > 0.9 * atr:   # stricter: if average body almost ATR -> choppy/big
            return None, "consolidation_failed_oversized", {}

        # consecutive direction check
        CONSISTENT_BARS = 8
        CONSISTENT_DIR_PCT = 0.6
        if len(closes) >= CONSISTENT_BARS:
            dirs = (closes[-CONSISTENT_BARS:] - opens[-CONSISTENT_BARS:]) > 0
            pct_up = float(np.sum(dirs)) / CONSISTENT_BARS
            consistent_up = pct_up >= CONSISTENT_DIR_PCT
            consistent_down = (1.0 - pct_up) >= CONSISTENT_DIR_PCT
        else:
            consistent_up = consistent_down = False

        # candle bits
        def candle_bits_at(i):
            o,h,l,c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            body = abs(c - o)
            r = max(h - l, 1e-12)
            upper = h - max(o, c)
            lower = min(o, c) - l
            is_doji = body <= 0.35 * r
            pin_low = (lower >= 0.2 * body) and (lower > upper)
            pin_high = (upper >= 0.2 * body) and (upper > lower)
            engulf_bull = engulf_bear = False
            if i > 0:
                prev_o, prev_c = float(opens[i-1]), float(closes[i-1])
                if (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o):
                    engulf_bull = True
                if (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o):
                    engulf_bear = True
            return {"o": o, "h": h, "l": l, "c": c, "body": body, "range": r,
                    "is_doji": is_doji, "pin_low": pin_low, "pin_high": pin_high,
                    "engulf_bull": engulf_bull, "engulf_bear": engulf_bear}

        rej = candle_bits_at(i_rej)
        con = candle_bits_at(i_con)
        prev_candle = candle_bits_at(i_rej - 1) if i_rej - 1 >= 0 else None

        # trend rules using stricter ordering & slope & separation
        def slope_ok_up(i, lookback=2):
            if i - lookback < 0: return False
            return (ma1[i] is not None and ma2[i] is not None and ma3_full[i] is not None and
                    ma1[i] > ma1[i-lookback] and ma2[i] > ma2[i-lookback] and ma3_full[i] > ma3_full[i-lookback])

        def slope_ok_down(i, lookback=2):
            if i - lookback < 0: return False
            return (ma1[i] is not None and ma2[i] is not None and ma3_full[i] is not None and
                    ma1[i] < ma1[i-lookback] and ma2[i] < ma2[i-lookback] and ma3_full[i] < ma3_full[i-lookback])

        def sep_ok(i, atr_mult=0.18):
            a = atr
            return (ma1[i] is not None and ma2[i] is not None and ma3_full[i] is not None and
                    (abs(ma1[i] - ma2[i]) > atr_mult * a) and (abs(ma2[i] - ma3_full[i]) > atr_mult * a))

        def is_trend_up(i):
            return (ma1[i] is not None and ma2[i] is not None and ma3_full[i] is not None and
                    (ma1[i] > ma2[i] > ma3_full[i]) and slope_ok_up(i) and sep_ok(i))

        def is_trend_down(i):
            return (ma1[i] is not None and ma2[i] is not None and ma3_full[i] is not None and
                    (ma1[i] < ma2[i] < ma3_full[i]) and slope_ok_down(i) and sep_ok(i))

        uptrend = is_trend_up(i_rej)
        downtrend = is_trend_down(i_rej)

        # -------- BUY path: rejection near MA dynamic support (MA1/MA2) ----------
        if uptrend:
            # choose nearest of MA1/MA2 to candle low
            zone1 = ma1[i_rej]; zone2 = ma2[i_rej]
            if zone1 is None and zone2 is None:
                return None, "no_ma_zone", {}
            zone = zone1 if zone1 is not None and (zone2 is None or abs(rej["l"] - zone1) <= abs(rej["l"] - zone2)) else zone2
            buffer_ = max(0.18 * atr, 0.0)

            near_zone = abs(rej["l"] - zone) <= buffer_
            close_side = rej["c"] >= (zone - buffer_)
            pattern_ok = rej["is_doji"] or rej["pin_low"] or rej["engulf_bull"]

            if near_zone and close_side and pattern_ok:
                extra = {"ma_zone": float(zone), "pattern": ("doji" if rej["is_doji"] else ("pin" if rej["pin_low"] else ("engulf" if rej["engulf_bull"] else "unknown")))}
                return "BUY", f"MA dynamic support rejection ({extra['pattern']})", extra
            return None, "buy_rejected:no_valid_rejection", {}

        # -------- SELL path ----------
        if downtrend:
            zone1 = ma1[i_rej]; zone2 = ma2[i_rej]
            if zone1 is None and zone2 is None:
                return None, "no_ma_zone", {}
            zone = zone1 if zone1 is not None and (zone2 is None or abs(rej["h"] - zone1) <= abs(rej["h"] - zone2)) else zone2
            buffer_ = max(0.18 * atr, 0.0)
            near_zone = abs(rej["h"] - zone) <= buffer_
            close_side = rej["c"] <= (zone + buffer_)
            pattern_ok = rej["is_doji"] or rej["pin_high"] or rej["engulf_bear"]
            if near_zone and close_side and pattern_ok:
                extra = {"ma_zone": float(zone), "pattern": ("doji" if rej["is_doji"] else ("pin" if rej["pin_high"] else ("engulf" if rej["engulf_bear"] else "unknown")))}
                return "SELL", f"MA dynamic resistance rejection ({extra['pattern']})", extra
            return None, "sell_rejected:no_valid_rejection", {}

        return None, "no_clear_trend", {}
    except Exception as e:
        log("signal_for_timeframe error:", e)
        traceback.print_exc()
        return None, "internal_error", {}

# ===== helpers to persist sent cache =====
def load_cache():
    global last_sent_signal_by_symbol
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                obj = json.load(f)
                # ensure proper types
                for k, v in obj.items():
                    last_sent_signal_by_symbol[k] = tuple(v)
            log("Loaded cache:", last_sent_signal_by_symbol)
    except Exception as e:
        log("load_cache failed:", e)

def save_cache():
    try:
        serial = {k: list(v) for k, v in last_sent_signal_by_symbol.items()}
        with open(CACHE_FILE, "w") as f:
            json.dump(serial, f)
    except Exception as e:
        log("save_cache failed:", e)

# ===== orchestrator & notifications =====
def analyze_and_notify():
    global last_heartbeat_sent

    load_cache()
    any_sent_this_run = False
    report_lines = []

    # iterate assets, prefer lower tf signals first (user requested)
    for symbol in ASSETS:
        checked = {}
        per_symbol_sent = False
        # Query TFs in ascending order (lower first)
        for tf in sorted(TIMEFRAMES):
            log("Processing", symbol, tf)
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles or len(candles) < 4:
                log(symbol, tf, "no candles or insufficient")
                checked[tf] = (None, "no candles")
                continue
            i_rej = len(candles) - 2
            i_con = len(candles) - 1
            direction, reason, extra = signal_for_timeframe(candles, tf, i_rej, i_con)
            checked[tf] = (direction, reason)
            log(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] | {symbol} {tf//60}m -> {direction} | {reason}")

            # cooldown & dedupe: skip if we sent same symbol/tf/direction recently
            last = last_sent_signal_by_symbol.get(symbol)
            now = int(time.time())
            if direction:
                allow_send = True
                if last and last[0] == tf and last[1] == direction and (now - last[2]) < ALERT_COOLDOWN:
                    allow_send = False
                    log("Skipping duplicate (cooldown):", symbol, tf, direction)
                if allow_send and not per_symbol_sent:
                    # create chart and send
                    ma1, ma2, ma3 = compute_mas_for_chart(candles)
                    chart_path = make_chart(candles, ma1, ma2, ma3, i_rej, "ACCEPT", symbol, tf, last_n=80, pad=10)
                    caption = f"{symbol} | {tf//60}m | {direction}\nReason: {reason}"
                    ok = False
                    try:
                        if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                            ok, info = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                            try:
                                os.unlink(chart_path)
                            except:
                                pass
                        else:
                            # fallback: use text helper
                            send_single_timeframe_signal(symbol, tf, direction, reason)
                            ok = True
                    except Exception as e:
                        log("send error:", e)
                    if ok:
                        last_sent_signal_by_symbol[symbol] = (tf, direction, now)
                        save_cache()
                        report_lines.append(f"{symbol}: SENT {direction} @ {tf//60}m -> {reason}")
                        any_sent_this_run = True
                        per_symbol_sent = True
                    else:
                        report_lines.append(f"{symbol}: failed send {direction} @ {tf//60}m")
                else:
                    report_lines.append(f"{symbol}: skipped duplicate or already sent this symbol {direction} @ {tf//60}m")
            else:
                # send chart for rejected high-probability setups (single timeframe) if reason indicates near-MA but failed pivot/pullback
                if reason and ("no_valid_rejection" in reason or "pivot/pullback failed" in reason or "consolidation_failed" in reason):
                    # still produce a small chart for later review
                    try:
                        ma1, ma2, ma3 = compute_mas_for_chart(candles)
                        chart_path = make_chart(candles, ma1, ma2, ma3, i_rej, "REJECTED", symbol, tf, last_n=80, pad=10)
                        caption = f"❌ Rejected\n{symbol} | {tf//60}m\nReason: {reason}"
                        if chart_path and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                            send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                            try:
                                os.unlink(chart_path)
                            except:
                                pass
                    except Exception as e:
                        log("failed to send rejected chart:", e)
                report_lines.append(f"{symbol}: {tf//60}m -> None | {reason}")

        # prefer lower tf signals: if we already sent on a lower tf, we skip sending for higher (already enforced per_symbol_sent)
        # done

    # heartbeat: short & concise
    #if not any_sent_this_run and HEARTBEAT_INTERVAL_HOURS > 0:
      #  now = int(time.time())
      #  since = now - last_heartbeat_sent
       # if since >= int(HEARTBEAT_INTERVAL_HOURS * 3600):
     #       checked_assets = ", ".join(ASSETS)
      #      msg = (
     #           "🤖 Bot heartbeat – alive\n"
      #          "⏰ No signals right now.\n"
      #          f"📊 Checked: {checked_assets}\n"
     #           f"⏱ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
     #       )
     #       try:
    #            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    #                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
     #           last_heartbeat_sent = now
     #           with open(HEART_FILE, "w") as f:
      #              json.dump({"ts": now}, f)
     #       except Exception as e:
      #          log("heartbeat send failed:", e)
    # persist
    save_cache()
    log("Analyze run complete.")

# small helper to produce MA arrays for charting
def compute_mas_for_chart(candles):
    closes  = [c["close"] for c in candles]
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    hlc3    = [typical_price(h, l, c) for h, l, c in zip(highs, lows, closes)]
    ma1 = smma(hlc3, 9)
    ma2 = smma(closes, 19)
    # produce MA3 aligned
    ma2_vals = [x for x in ma2 if x is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else [None]*len(ma2_vals)
    ma3 = []
    j = 0
    for i in range(len(ma2)):
        if ma2[i] is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

# ===== main run (timezone sleep support) =====
if __name__ == "__main__":
    try:
        # Sleep window in Zimbabwe time (21:00 - 06:00) -> skip runs during that local time
        try:
            import pytz
            tz = pytz.timezone("Africa/Harare")
            now_local = datetime.now(tz)
            if 21 <= now_local.hour or now_local.hour < 6:
                log("Bot sleeping Zimbabwe 21:00-06:00 local")
            else:
                analyze_and_notify()
        except Exception:
            # if pytz not available, just run
            analyze_and_notify()
    except Exception as e:
        log("Fatal error in main:", e)
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, f"❌ Bot crashed: {e}")
            except:
                pass
        raise

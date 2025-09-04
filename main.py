# main.py
"""
Paste-and-cruise single-file bot:
- MT5-style SMMA/SMA moving averages (MA1=SMMA(9) on HLC/3, MA2=SMMA(19) on Close, MA3=SMA(25) on MA2)
- Robust fetch_candles() for Deriv via websocket + normalization
- Per-asset per-TF fallback: skip unsupported TFs, analyze whichever TFs returned data (so 1s assets won't be ignored)
- Proper candlesticks via mplfinance, fig ratio/scale configurable
- Sends Telegram chart + compact caption (one message per asset per run, choosen strongest TF)
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
import requests
import numpy as np
import pandas as pd
import mplfinance as mpf

# -------------------------
# CONFIG
# -------------------------
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not DERIV_API_KEY:
    raise RuntimeError("DERIV_API_KEY env var is required")

# Assets (use the Deriv symbols that match your MT5 indices)
ASSETS = [
    "R_10",      # V10
    "R_50",      # V50
    "R_75",      # V75
    "1HZ75V",    # V75(1s)
    "1HZ100V",   # V100(1s)
    "1HZ150V",   # V150(1s)
]

# TIMEFRAMES: main desired TFs (seconds): 5m, 10m, 15m
DESIRED_TIMEFRAMES = [300, 600, 900]
TF_LABEL = {300: "5m", 600: "10m", 900: "15m"}

# If a TF request returns no data (common on 1s assets for 10m/15m),
# the code will skip that TF and use others that returned data.
CANDLES_FETCH_COUNT = int(os.getenv("CANDLES_N", "300"))  # history to fetch
CHART_LOOKBACK = int(os.getenv("LAST_N_CHART", "220"))    # how many candles to show on chart
RIGHT_PADDING = int(os.getenv("PAD_CANDLES", "10"))       # 10-candle right padding

# Chart sizing: change these to control aspect/size
# These are the mplfinance arguments the bot will use:
FIGRATIO = (9, 6)   # shape: (width, height). bigger second number -> taller chart
FIGSCALE = 1.4      # overall scale: >1 -> bigger image
FIG_DPI = 140

# Alerting behavior
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "600"))  # prevent duplicates per symbol/tf/dir
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN", "6"))

# Heartbeat (muted by default)
HEARTBEAT_INTERVAL_HOURS = float(os.getenv("HEARTBEAT_INTERVAL_HOURS", "0"))

# temp cache file for last-sent
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "aspmibot_last_sent.json")
HEART_FILE = os.path.join(TMPDIR, "aspmibot_last_heartbeat.json")

DEBUG = os.getenv("DEBUG", "0") in ("1", "true", "True", "yes", "on")

def log(*args, **kwargs):
    if DEBUG:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC]", *args, **kwargs)

# -------------------------
# Telegram helpers
# -------------------------
def send_telegram_text(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram credentials missing; not sending text")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=payload, timeout=15)
        if not r.ok:
            log("Telegram sendText failed:", r.status_code, r.text)
        return r.ok
    except Exception as e:
        log("Telegram sendText exception:", e)
        return False

def send_telegram_photo(photo_path: str, caption: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram credentials missing; not sending photo")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
            r = requests.post(url, data=data, files=files, timeout=30)
        if not r.ok:
            log("Telegram sendPhoto failed:", r.status_code, r.text)
        return r.ok
    except Exception as e:
        log("Telegram sendPhoto exception:", e)
        return False

# -------------------------
# Persistence for duplicate suppression
# -------------------------
last_sent_cache: Dict[str, float] = {}

def load_cache():
    global last_sent_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_cache = json.load(f)
                log("Loaded last sent cache")
    except Exception as e:
        log("load_cache failed:", e)
        last_sent_cache = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
            log("Saved last sent cache")
    except Exception as e:
        log("save_cache failed:", e)

def should_send(symbol: str, tf: int, direction: str) -> bool:
    key = f"{symbol}:{tf}:{direction}"
    now = time.time()
    last = last_sent_cache.get(key)
    if last and (now - last) < ALERT_COOLDOWN_SECS:
        log("suppress duplicate", key)
        return False
    last_sent_cache[key] = now
    save_cache()
    return True

# -------------------------
# Helpers: MT5-style SMMA / SMA implementations
# -------------------------
def smma_series(values: np.ndarray, period: int) -> np.ndarray:
    """
    MT5-style SMMA:
      - seed at index period-1 = SMA(values[:period])
      - then for i >= period: smma[i] = (smma[i-1]*(period-1) + val[i]) / period
    Returns numpy array length = len(values), with np.nan for indices < period-1.
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    seed = float(np.mean(vals[:period]))
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + float(vals[i])) / period
        out[i] = prev
    return out

def sma_series(values: np.ndarray, period: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    cumsum = np.cumsum(vals)
    for i in range(period - 1, n):
        out[i] = float((cumsum[i] - (cumsum[i - period] if i - period >= 0 else 0.0)) / period)
    return out

# compute MAs aligned with candles (same length arrays)
def compute_mas_np(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    typical = (highs + lows + closes) / 3.0
    ma1 = smma_series(typical, 9)   # MA1 SMMA(9) on typical
    ma2 = smma_series(closes, 19)   # MA2 SMMA(19) on close
    ma3 = sma_series(ma2, 25)       # MA3 SMA(25) on MA2 values
    return ma1, ma2, ma3

# -------------------------
# Normalize raw payload into DataFrame with DatetimeIndex & open,high,low,close
# -------------------------
def _normalize_candles(raw) -> Optional[pd.DataFrame]:
    try:
        if raw is None:
            return None
        # if it's a dict with 'candles' key
        if isinstance(raw, dict) and "candles" in raw:
            rows = raw["candles"]
        elif isinstance(raw, list):
            rows = raw
        elif isinstance(raw, pd.DataFrame):
            df = raw.copy()
            # ensure columns
            if all(c in df.columns for c in ("open","high","low","close")):
                # try to use epoch column if present
                if "epoch" in df.columns:
                    df.index = pd.to_datetime(df["epoch"].astype(float), unit="s", origin="unix")
                return df[["open","high","low","close"]]
            rows = df.to_dict("records")
        else:
            return None

        if not rows:
            return None
        df = pd.DataFrame(rows)

        # canonical epoch column
        if "epoch" not in df.columns:
            for k in ("time", "timestamp", "start_time"):
                if k in df.columns:
                    df["epoch"] = df[k]
                    break

        # epoch may be string/float/ms -> detect and convert to seconds
        if "epoch" in df.columns:
            # coerce to float
            df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
            # if values are huge (ms)
            if df["epoch"].dropna().max() > 1e12:
                df["epoch_s"] = df["epoch"] / 1000.0
                df.index = pd.to_datetime(df["epoch_s"], unit="s", origin="unix")
            else:
                df.index = pd.to_datetime(df["epoch"].astype(float), unit="s", origin="unix")
        else:
            # fallback: try to parse any datetime-like column
            assigned = False
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df.index = pd.to_datetime(df[col])
                        assigned = True
                        break
                    except Exception:
                        pass
            if not assigned:
                # impossible to create datetime index
                return None

        # map variant names to canonical ones
        rename_map = {}
        for a, b in (("open", "open"), ("opens", "open"), ("OPEN", "open"),
                     ("high", "high"), ("highs", "high"), ("HIGH", "high"),
                     ("low", "low"), ("lows", "low"), ("LOW", "low"),
                     ("close", "close"), ("closes", "close"), ("CLOSE", "close")):
            if a in df.columns and b not in df.columns:
                rename_map[a] = b
        df.rename(columns=rename_map, inplace=True)

        # sometimes OHLC appear nested or under other keys; try to extract
        required = ["open","high","low","close"]
        for col in required:
            if col not in df.columns:
                # try extracting from nested dicts
                for c in df.columns:
                    if df[c].apply(lambda v: isinstance(v, dict)).any():
                        df[col] = df[c].apply(lambda v: v.get(col) if isinstance(v, dict) else None)
                        if col in df.columns:
                            break

        if not all(c in df.columns for c in required):
            return None

        # numeric convert and drop NA rows
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=required)
        if df.empty:
            return None

        # ensure sorted unique index
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df[required].copy()
    except Exception as e:
        log("normalize_candles failed:", e)
        return None

# -------------------------
# Fetch candles from Deriv (websocket) and return normalized DataFrame or None
# -------------------------
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_FETCH_COUNT, timeout=12) -> Optional[pd.DataFrame]:
    """
    Uses Deriv websocket ticks_history candles API.
    Returns pandas.DataFrame indexed by DatetimeIndex with columns open, high, low, close.
    """
    try:
        ws = websocket.create_connection(DERIV_WS_URL, timeout=timeout)
    except Exception as e:
        log("websocket.connect failed:", e)
        return None

    try:
        # authorize
        ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        auth = json.loads(ws.recv())
        # request candles
        req = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": "latest",
            "subscribe": 0  # not subscribing live updates here; we only need snapshot
        }
        ws.send(json.dumps(req))
        resp = json.loads(ws.recv())
        if not resp or "candles" not in resp or not resp["candles"]:
            # try to detect error message
            # if the API rejects granularity it may include an error or empty response
            return None
        df = _normalize_candles(resp)
        return df
    except Exception as e:
        log("fetch_candles exception:", e)
        return None
    finally:
        try:
            ws.close()
        except Exception:
            pass

# -------------------------
# Pattern helpers (pin, doji, engulf)
# -------------------------
def is_doji(o,h,l,c,th=0.25):
    rng = max(h-l, 1e-12)
    return abs(c-o) <= th * rng

def is_bull_pin(o,h,l,c):
    rng = max(h-l, 1e-12)
    lw = min(o,c) - l
    bdy = abs(c-o)
    return (lw >= 0.6*rng) and (lw >= 1.2*bdy)

def is_bear_pin(o,h,l,c):
    rng = max(h-l, 1e-12)
    uw = h - max(o,c)
    bdy = abs(c-o)
    return (uw >= 0.6*rng) and (uw >= 1.2*bdy)

def is_engulf_bull(prev_o, prev_c, o, c):
    return (prev_c < prev_o) and (c > o) and (o <= prev_c) and (c >= prev_o)

def is_engulf_bear(prev_o, prev_c, o, c):
    return (prev_c > prev_o) and (c < o) and (o >= prev_c) and (c <= prev_o)

# -------------------------
# Decide signal for a single TF (checks rejection on last-closed candle)
# Returns dict with accepted True/False, direction, score, reason, and required plotting data
# -------------------------
def evaluate_tf(df: pd.DataFrame, tf: int) -> Optional[Dict]:
    """
    df: DataFrame with index DatetimeIndex and columns open,high,low,close
    Uses last closed candle as rejection (i_rej = -2) and last forming candle as current (-1)
    """
    if df is None or len(df) < 10:
        return None

    n = len(df)
    i_rej = n - 2
    i_con = n - 1

    opens = df['open'].to_numpy(dtype=float)
    highs = df['high'].to_numpy(dtype=float)
    lows  = df['low'].to_numpy(dtype=float)
    closes= df['close'].to_numpy(dtype=float)

    ma1, ma2, ma3 = compute_mas_np(highs, lows, closes)

    # require ma values present at i_rej
    if math.isnan(ma1[i_rej]) or math.isnan(ma2[i_rej]) or math.isnan(ma3[i_rej]):
        return None

    # basic ATR
    rngs = highs - lows
    atr = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny = max(1e-9, 0.05*atr)

    # pattern checks at rejection candle
    prev_o, prev_c = opens[i_rej - 1], closes[i_rej - 1] if i_rej - 1 >= 0 else (None, None)
    o,h,l,c = opens[i_rej], highs[i_rej], lows[i_rej], closes[i_rej]
    pattern_ok = False
    patname = "None"
    if is_doji(o,h,l,c):
        pattern_ok = True; patname = "Doji"
    if is_bull_pin(o,h,l,c):
        pattern_ok = True; patname = "BullPin"
    if is_bear_pin(o,h,l,c):
        pattern_ok = True; patname = "BearPin"
    if i_rej - 1 >= 0:
        if is_engulf_bull(prev_o, prev_c, o, c):
            pattern_ok = True; patname = "EngulfBull"
        if is_engulf_bear(prev_o, prev_c, o, c):
            pattern_ok = True; patname = "EngulfBear"

    # trend stacking
    uptrend = (ma1[i_rej] > ma2[i_rej] > ma3[i_rej])
    downtrend = (ma1[i_rej] < ma2[i_rej] < ma3[i_rej])

    # proximity to MA zone: pick nearer of MA1/MA2
    if uptrend:
        # use lows proximity
        d1 = abs(l - ma1[i_rej])
        d2 = abs(l - ma2[i_rej])
        zone_name = "MA1" if d1 <= d2 else "MA2"
        zone_val = float(ma1[i_rej] if zone_name=="MA1" else ma2[i_rej])
        near_zone = abs(l - zone_val) <= 0.25*atr
        close_side = (c >= zone_val - tiny)
        accepted = uptrend and pattern_ok and near_zone and close_side and (closes[-1] > ma3[i_rej] + tiny)
        direction = "BUY" if accepted else ("BUY" if uptrend else None)
    elif downtrend:
        # highs proximity
        d1 = abs(h - ma1[i_rej])
        d2 = abs(h - ma2[i_rej])
        zone_name = "MA1" if d1 <= d2 else "MA2"
        zone_val = float(ma1[i_rej] if zone_name=="MA1" else ma2[i_rej])
        near_zone = abs(h - zone_val) <= 0.25*atr
        close_side = (c <= zone_val + tiny)
        accepted = downtrend and pattern_ok and near_zone and close_side and (closes[-1] < ma3[i_rej] - tiny)
        direction = "SELL" if accepted else ("SELL" if downtrend else None)
    else:
        # no clear trend
        return {"tf": tf, "accepted": False, "direction": None, "score": 0.0, "reason": "no clear trend", "df": df, "ma1": ma1, "ma2": ma2, "ma3": ma3, "i_rej": i_rej}

    # simple scoring: trend + pattern + proximity
    score = 0
    score += 3 if (uptrend or downtrend) else 0
    score += 2 if pattern_ok else 0
    score += 2 if near_zone else 0
    # weight TF: prefer higher TF (10m/15m) a bit
    tf_weight = {300:1, 600:1.2, 900:1.4}.get(tf, 1.0)
    score = float(score) * tf_weight

    reason = f"{zone_name} rejection; pattern={patname}; near_zone={near_zone}"
    return {"tf": tf, "accepted": bool(accepted), "direction": direction, "score": score, "reason": reason, "df": df, "ma1": ma1, "ma2": ma2, "ma3": ma3, "i_rej": i_rej}

# -------------------------
# Chart builder using mplfinance
# -------------------------
def build_chart_png(evald: Dict, symbol: str, outdir: str = None) -> Optional[str]:
    """
    evald: dict returned by evaluate_tf: contains df, ma1, ma2, ma3, i_rej etc
    Returns path to PNG.
    """
    try:
        df = evald["df"].copy()
        # trim to last CHART_LOOKBACK candles
        if CHART_LOOKBACK and len(df) > CHART_LOOKBACK:
            df = df.iloc[-CHART_LOOKBACK:]
        # add moving averages columns with alignment (they are arrays aligned to original df)
        ma1 = evald["ma1"][-len(df):] if evald.get("ma1") is not None else None
        ma2 = evald["ma2"][-len(df):] if evald.get("ma2") is not None else None
        ma3 = evald["ma3"][-len(df):] if evald.get("ma3") is not None else None

        # prepare DataFrame for mplfinance
        mpf_df = df.copy()
        # ensure columns Open/High/Low/Close names for mplfinance
        mpf_df = mpf_df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
        # attach MA columns for addplot (must be numeric series indexed same)
        addplots = []
        idx = mpf_df.index
        if ma1 is not None:
            s1 = pd.Series(ma1[-len(mpf_df):], index=idx)
            addplots.append(mpf.make_addplot(s1, color="#1f77b4", width=1.2))
        if ma2 is not None:
            s2 = pd.Series(ma2[-len(mpf_df):], index=idx)
            addplots.append(mpf.make_addplot(s2, color="#ff7f0e", width=1.2))
        if ma3 is not None:
            s3 = pd.Series(ma3[-len(mpf_df):], index=idx)
            addplots.append(mpf.make_addplot(s3, color="#2ca02c", width=1.2))

        # title
        tf = evald.get("tf")
        verdict = "✅" if evald.get("accepted") else "❌"
        direction = evald.get("direction") or "-"
        title = f"{symbol} {TF_LABEL.get(tf, str(tf))} {verdict} {direction} s={evald.get('score', 0):.2f}"

        # output path
        if outdir is None:
            outdir = tempfile.gettempdir()
        fname = f"{symbol}_{TF_LABEL.get(tf,tf)}_{int(time.time())}.png"
        outpath = os.path.join(outdir, fname)

        mpf.plot(
            mpf_df,
            type="candle",
            style="charles",
            title=title,
            addplot=addplots,
            figratio=FIGRATIO,
            figscale=FIGSCALE,
            volume=False,
            tight_layout=True,
            savefig=dict(fname=outpath, dpi=FIG_DPI, bbox_inches="tight")
        )

        return outpath
    except Exception as e:
        log("build_chart_png failed:", e)
        return None

# -------------------------
# Orchestration: analyze all assets, pick strongest TF per asset, send one message per asset
# -------------------------

def analyze_and_notify():
    load_cache()
    signals_sent = 0
    summary = []

    for symbol in ASSETS:
        try:
            log("Analyzing", symbol)
            per_tf_results = []

            # loop over all timeframes (no extra try)
            for tf in DESIRED_TIMEFRAMES:
                df = fetch_candles(symbol, tf, CANDLES_FETCH_COUNT)
                if df is None or len(df) < 8:
                    log(f"{symbol} {TF_LABEL.get(tf)}: no data or insufficient rows -> skip")
                    continue

                res = evaluate_tf(df, tf)
                if res:
                    # Apply weighting to prioritize 5m slightly
                    tf_weight = {300: 1.3, 600: 1.1, 900: 1.0}.get(tf, 1.0)
                    res["score"] = res.get("score", 0.0) * tf_weight
                    per_tf_results.append(res)

            if not per_tf_results:
                summary.append(f"{symbol}: no TF returned data")
                continue

            # choose best: prefer accepted True highest score, else highest rejection
            accepted = [r for r in per_tf_results if r.get("accepted")]
            if accepted:
                best = max(accepted, key=lambda x: x.get("score", 0.0))
            else:
                best = max(per_tf_results, key=lambda x: x.get("score", 0.0))

            # build a compact other-TF summary
            other_lines = []
            for r in sorted(per_tf_results, key=lambda x: x.get("tf")):
                tag = "✅" if r.get("accepted") else "❌"
                other_lines.append(f"{TF_LABEL.get(r['tf'])} {tag} (score {r.get('score', 0.2):.2f})")

            # send alert for best (if not suppressed by cooldown)
            tf_best = best["tf"]
            direction = best.get("direction") or "-"
            score = best.get("score", 0.0)
            reason = best.get("reason", "-")
            emoji = "✅" if best.get("accepted") else "❌"

            if should_send(symbol, tf_best, direction):
                chart = build_chart_png(best, symbol)
                caption = f"{emoji} {symbol} | {TF_LABEL.get(tf_best)} | {direction} | s={score:.2f}\nReason: {reason}\nOther TFs:\n" + "\n".join(other_lines)
                sent = False

                if chart:
                    try:
                        sent = send_tg_photo(chart, caption)
                        os.remove(chart)
                    except Exception:
                        pass
                else:
                    sent = send_tg_text(caption)

                if sent:
                    summary.append(f"{symbol}: sent {direction} ({TF_LABEL.get(tf_best)}) score={score:.2f}")
                    signals_sent += 1
                else:
                    summary.append(f"{symbol}: failed send {direction} ({TF_LABEL.get(tf_best)})")
            else:
                summary.append(f"{symbol}: suppressed duplicate or cooldown")

            if signals_sent >= MAX_SIGNALS_PER_RUN:
                log("Reached MAX_SIGNALS_PER_RUN")
                break

        except Exception as e:
            log("Error analyzing", symbol, e)
            traceback.print_exc()
            try:
                send_tg_text(f"⚠️ {symbol} error: {e}")
            except Exception:
                pass
    # optionally send heartbeat if muted is disabled and nothing sent
    if signals_sent == 0 and HEARTBEAT_INTERVAL_HOURS > 0:
        try:
            last_ts = 0
            if os.path.exists(HEART_FILE):
                try:
                    with open(HEART_FILE, "r") as f:
                        last_ts = int(json.load(f).get("ts", 0))
                except Exception:
                    last_ts = 0
            now = int(time.time())
            if now - last_ts >= int(HEARTBEAT_INTERVAL_HOURS * 3600):
                checked = ", ".join(ASSETS)
                hb = f"🤖 Bot heartbeat – alive\n⏰ No signals right now.\n📊 Checked: {checked}\n🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                send_telegram_text(hb)
                with open(HEART_FILE, "w") as f:
                    json.dump({"ts": now}, f)
        except Exception:
            log("heartbeat exception", traceback.format_exc())

    log("Run summary:", summary)


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        log("Fatal error:", e)
        try:
            send_telegram_text(f"❌ Bot crashed: {e}")
        except Exception:
            pass
        raise

# main.py  — Dynamic S/R with charts (5m only)
# Paste-and-cruise. No placeholders. No heartbeats. No ATR. Fire at candle close only.

import os, json, time, math, tempfile, traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# === Telegram helpers (must exist in your repo in bot.py) ===
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    # Fallback for local testing
    def send_telegram_message(token, chat_id, text):
        print("[TELEGRAM TEXT]", text); return True, "local"
    def send_telegram_photo(token, chat_id, caption, photo_path):
        print("[TELEGRAM PHOTO]", caption, photo_path); return True, "local"

# === ENV / Defaults ===
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Default assets for the charting bot (non-1s). Adjust via env if you like.
ASSETS = os.getenv("ASSETS", "R_10,R_25,R_50,R_75,R_100,BOOM500,CRASH500,BOOM1000,CRASH1000").split(",")

# Strategy constants (hard-coded to your spec)
TF_SEC = 300  # 5 minutes only
CANDLES_N = int(os.getenv("CANDLES_N", "160"))  # history depth for MAs & chart
LAST_N_CHART = int(os.getenv("LAST_N_CHART", "180"))
PAD_CANDLES = 10
CANDLE_WIDTH = 0.35

# Near tolerance (% of price). No ATR. Tiny retests in trend allowed within this %.
NEAR_PCT = float(os.getenv("NEAR_PCT", "0.001"))  # 0.10% default

# Dedupe / spam control
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", "0"))  # 0 = no cooldown besides dedupe
MAX_ALERTS_PER_RUN = int(os.getenv("MAX_ALERTS_PER_RUN", "12"))   # safety cap per run
TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def utc_ts() -> int:
    return int(time.time())

def log(msg: str):
    # Quiet by default; uncomment if needed:
    # print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}")
    pass

# ------------------------------------------------------------
# Persistence for dedupe
# ------------------------------------------------------------
_last_sent = {}  # symbol -> {"epoch": int, "dir": "BUY/SELL"}

def load_cache():
    global _last_sent
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                _last_sent = json.load(f)
        else:
            _last_sent = {}
    except Exception:
        _last_sent = {}

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_last_sent, f)
    except Exception:
        pass

def can_send(symbol: str, direction: str, epoch: int) -> bool:
    rec = _last_sent.get(symbol)
    now = utc_ts()
    if rec:
        if rec.get("epoch") == epoch and rec.get("dir") == direction:
            return False
        if ALERT_COOLDOWN_SECS > 0 and (now - rec.get("ts", 0)) < ALERT_COOLDOWN_SECS:
            # cooldown blocks only same symbol irrespective of epoch/dir
            return False
    return True

def mark_sent(symbol: str, direction: str, epoch: int):
    _last_sent[symbol] = {"epoch": epoch, "dir": direction, "ts": utc_ts()}
    save_cache()

# ------------------------------------------------------------
# MA Calculations (Your exact stack)
# ------------------------------------------------------------
def smma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if period <= 0 or n == 0: return [None]*n
    if n < period: return [None]*n
    seed = sum(series[:period]) / period
    out = [None]*(period-1) + [seed]
    prev = seed
    for i in range(period, n):
        prev = (prev*(period-1) + float(series[i])) / period
        out.append(prev)
    return out

def sma(series: List[float], period: int) -> List[Optional[float]]:
    n = len(series)
    if period <= 0 or n < period: return [None]*n
    out = [None]*(period-1)
    run = sum(series[:period])
    out.append(run/period)
    for i in range(period, n):
        run += series[i] - series[i-period+1]
        out.append(run/period)
    return out

def build_mas(candles: List[Dict]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    hlc3   = [(h+l+c)/3.0 for h,l,c in zip(highs,lows,closes)]
    ma1 = smma(hlc3, 9)               # MA1: SMMA(HLC3, 9)
    ma2 = smma(closes, 19)            # MA2: SMMA(Close, 19)
    # MA3 = SMA(MA2, 25)
    ma2_vals = [v for v in ma2 if v is not None]
    ma3_raw = sma(ma2_vals, 25) if len(ma2_vals) >= 25 else []
    ma3 = []
    j = 0
    for v in ma2:
        if v is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[j] if j < len(ma3_raw) else None)
            j += 1
    return ma1, ma2, ma3

# ------------------------------------------------------------
# Candle family detection (broad)
# ------------------------------------------------------------
def candle_family(c: Dict, prev: Optional[Dict]) -> str:
    o = float(c["open"]); h = float(c["high"]); l = float(c["low"]); cclose = float(c["close"])
    body = abs(cclose - o)
    rng  = max(1e-12, h - l)
    upper = h - max(o, cclose)
    lower = min(o, cclose) - l

    # Doji family (any) — body tiny vs range
    if body <= 0.15 * rng:
        return "DOJI"

    # Pin bars (any) — one wick dominates
    if upper >= 0.55 * rng and body <= 0.45 * rng:
        return "PIN_HIGH"
    if lower >= 0.55 * rng and body <= 0.45 * rng:
        return "PIN_LOW"

    # Engulfing (bull/bear)
    if prev is not None:
        po, pc = float(prev["open"]), float(prev["close"])
        # bullish engulfing
        if pc < po and cclose > o and o <= pc and cclose >= po:
            return "BULL_ENG"
        # bearish engulfing
        if pc > po and cclose < o and o >= pc and cclose <= po:
            return "BEAR_ENG"

    # Tiny-body catch-all
    if body <= 0.08 * rng:
        return "TINY"

    return "NONE"

# ------------------------------------------------------------
# Trend + proximity logic (No ATR)
# ------------------------------------------------------------
def in_uptrend(i: int, ma1, ma2, ma3, price: float) -> bool:
    return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
            and ma1[i] > ma2[i] > ma3[i] and price >= ma3[i])

def in_downtrend(i: int, ma1, ma2, ma3, price: float) -> bool:
    return (ma1[i] is not None and ma2[i] is not None and ma3[i] is not None
            and ma1[i] < ma2[i] < ma3[i] and price <= ma3[i])

def near(value: float, ref: float, price: float) -> bool:
    # Accept touch/cross or within NEAR_PCT of price
    if ref is None: return False
    if value == ref: return True
    tol = max(price * NEAR_PCT, 0.0)
    return abs(value - ref) <= tol

def broke_ma3_recently(closes: List[float], ma3, i: int, lookback: int = 5) -> Optional[str]:
    """Return 'UP' if recently crossed above MA3, 'DOWN' if crossed below, else None."""
    if i <= 0 or ma3[i] is None: return None
    lo = max(1, i - lookback)
    for k in range(lo, i+1):
        if ma3[k] is None or ma3[k-1] is None: continue
        # cross up
        if closes[k-1] <= ma3[k-1] and closes[k] > ma3[k]:
            return "UP"
        # cross down
        if closes[k-1] >= ma3[k-1] and closes[k] < ma3[k]:
            return "DOWN"
    return None

# ------------------------------------------------------------
# Deriv fetch (snapshot; no subscribe; robust retries)
# ------------------------------------------------------------
def fetch_candles(symbol: str, granularity: int, count: int) -> List[Dict]:
    MAX_TRIES = 3
    for _ in range(MAX_TRIES):
        ws = None
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=18)
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
            _ = ws.recv()
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
            if "candles" in resp and resp["candles"]:
                cds = [{
                    "epoch": int(c["epoch"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                } for c in resp["candles"]]
                cds.sort(key=lambda x: x["epoch"])
                return cds
        except Exception:
            time.sleep(1)
        finally:
            try:
                if ws: ws.close()
            except Exception:
                pass
    return []

# ------------------------------------------------------------
# Charting
# ------------------------------------------------------------
def make_chart(symbol: str, tf_sec: int, candles: List[Dict], ma1, ma2, ma3, idx: int, reasons: List[str]) -> Optional[str]:
    try:
        total = len(candles)
        if total == 0: return None
        start = max(0, total - LAST_N_CHART)
        ch = candles[start:total]
        xs  = [datetime.fromtimestamp(c["epoch"], tz=timezone.utc) for c in ch]
        opn = [c["open"] for c in ch]
        hig = [c["high"] for c in ch]
        low = [c["low"] for c in ch]
        cls = [c["close"] for c in ch]

        fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
        ax.set_title(f"{symbol} | {tf_sec//60}m | {', '.join(reasons)}", fontsize=11)

        # candles
        for i,(o,h,l,c) in enumerate(zip(opn,hig,low,cls)):
            color = "#2ca02c" if c>=o else "#d62728"
            ax.plot([i,i],[l,h], color="black", linewidth=0.6, zorder=1)
            lower = min(o,c)
            height = max(1e-9, abs(c-o))
            ax.add_patch(Rectangle((i - CANDLE_WIDTH/2.0, lower), CANDLE_WIDTH, height,
                                   facecolor=color, edgecolor="black", linewidth=0.35, zorder=2))

        # MAs
        m1 = [ma1[j] for j in range(start, total)]
        m2 = [ma2[j] for j in range(start, total)]
        m3 = [ma3[j] for j in range(start, total)]
        def sanitize(x): return [v if (v is not None and not (isinstance(v,float) and math.isnan(v))) else float('nan') for v in x]
        ax.plot(range(len(m1)), sanitize(m1), linewidth=1.1, label="MA1 SMMA(HLC3,9)")
        ax.plot(range(len(m2)), sanitize(m2), linewidth=1.1, label="MA2 SMMA(Close,19)")
        ax.plot(range(len(m3)), sanitize(m3), linewidth=1.1, label="MA3 SMA(MA2,25)")
        ax.legend(loc="upper left", fontsize=8)

        # marker
        if start <= idx < total:
            midx = idx - start
            ax.scatter([midx], [ch[midx]["close"]],
                       marker="^" if ch[midx]["close"] >= ch[midx]["open"] else "v",
                       s=140, zorder=6, edgecolors="black")

        ax.set_xlim(-1, len(ch)+PAD_CANDLES)
        ymin, ymax = min(low), max(hig)
        pad_y = (ymax-ymin)*0.08 if ymax>ymin else 1e-6
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        step = max(1, len(ch)//8)
        ax.set_xticks(range(0, len(ch), step))
        ax.set_xticklabels([xs[i].strftime("%m-%d\n%H:%M") for i in range(0, len(xs), step)], fontsize=8)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.tight_layout(); fig.savefig(tmp.name, dpi=120); plt.close(fig)
        return tmp.name
    except Exception:
        return None

# ------------------------------------------------------------
# Signal evaluation
# ------------------------------------------------------------
def evaluate_signal(symbol: str, candles: List[Dict]) -> Optional[Dict]:
    n = len(candles)
    if n < 60: return None  # need enough history for MAs

    ma1, ma2, ma3 = build_mas(candles)
    i = n - 1  # last closed candle as per Deriv snapshot

    c = candles[i]
    p = candles[i-1] if i-1 >= 0 else None

    family = candle_family(c, p)
    if family == "NONE":
        return None  # rejection family required

    # Location: near/touch MA1 or MA2 (no ATR)
    price = c["close"]
    low   = c["low"]
    high  = c["high"]

    near_ma_buy  = near(low,  ma1[i], price) or near(low,  ma2[i], price)
    near_ma_sell = near(high, ma1[i], price) or near(high, ma2[i], price)

    closes = [x["close"] for x in candles]
    cross = broke_ma3_recently(closes, ma3, i, lookback=5)

    reasons = []
    direction = None

    # --- Continuation (trend follow) with tiny retests allowed ---
    if in_uptrend(i, ma1, ma2, ma3, price) and (near_ma_buy or (ma1[i] and price >= ma1[i] and (low >= ma2[i] if ma2[i] else True))):
        direction = "BUY"; reasons.append(f"{family} near MA1/MA2 (trend)")
    if in_downtrend(i, ma1, ma2, ma3, price) and (near_ma_sell or (ma1[i] and price <= ma1[i] and (high <= ma2[i] if ma2[i] else True))):
        # tiny retests in downtrend
        direction = "SELL"; reasons.append(f"{family} near MA1/MA2 (trend)")

    # --- Reversal: MA3 breakout then retest MA1/MA2 with rejection ---
    if cross == "UP" and (near_ma_buy):
        direction = "BUY"; reasons.append("Reversal: broke above MA3 then rejection at MA1/MA2")
    if cross == "DOWN" and (near_ma_sell):
        direction = "SELL"; reasons.append("Reversal: broke below MA3 then rejection at MA1/MA2")

    if direction is None:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "reasons": reasons,
        "idx": i,
        "ma1": ma1, "ma2": ma2, "ma3": ma3
    }

# ------------------------------------------------------------
# Run one pass
# ------------------------------------------------------------
def run_once():
    load_cache()
    alerts = 0
    for sym in ASSETS:
        if alerts >= MAX_ALERTS_PER_RUN:
            break
        candles = fetch_candles(sym, TF_SEC, CANDLES_N)
        if not candles or len(candles) < 60:
            continue
        sig = evaluate_signal(sym, candles)
        if not sig: 
            continue

        last_epoch = candles[sig["idx"]]["epoch"]
        if not can_send(sym, sig["direction"], last_epoch):
            continue

        # Build caption
        close = candles[sig["idx"]]["close"]
        when  = datetime.fromtimestamp(last_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cap = f"[{sym} | {TF_SEC//60}m | {sig['direction']}] " \
              f"{' | '.join(sig['reasons'])} — close={close} @ {when}"

        # Chart with marker
        chart = make_chart(sym, TF_SEC, candles, sig["ma1"], sig["ma2"], sig["ma3"], sig["idx"], sig["reasons"])
        ok = False
        if chart:
            ok, _ = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, cap, chart)
        if not chart or not ok:
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, cap)

        mark_sent(sym, sig["direction"], last_epoch)
        alerts += 1

if __name__ == "__main__":
    if not DERIV_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Missing required ENV: DERIV_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    run_once()

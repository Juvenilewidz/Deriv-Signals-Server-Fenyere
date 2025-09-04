# main.py — ASPMI dynamic S/R (paste-and-run)
import os, io, json, math, time, datetime, tempfile, requests
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import websocket  # pip install websocket-client

# =========================
# Env / constants
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# =========================
# Assets & TFs (your exact list)
# =========================
# Deriv symbols for your assets:
ASSETS = [
    "R_10",     # V10
    "R_50",     # V50
    "R_75",     # V75
    "1HZ75V",   # V75(1s)
    "1HZ100V",  # V100(1s)
    "1HZ150V",  # 150(1s)
]
# Timeframes: 5m, 10m, 15m
TIMEFRAMES = [300, 600, 900]
TF_LABEL = {300: "5m", 600: "10m", 900: "15m"}

# How many candles to fetch and how many to show on chart
CANDLES_N_FETCH = 300        # more data for better MAs & context
CHART_LOOKBACK  = 200        # show more history (smaller candles)
RIGHT_PADDING   = 10         # 10-candle right padding (blank space)

# =========================
# Telegram helpers
# =========================
def send_telegram_message(text: str, photo_path: Optional[str] = None):
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"{base}/sendPhoto"
            with open(photo_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
                requests.post(url, data=data, files=files, timeout=20)
        else:
            url = f"{base}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=20)
    except Exception as e:
        # Best-effort fallback — do not crash the bot for Telegram issues
        print(f"[WARN] Telegram send failed: {e}")

# =========================
# Candle/MA utilities (your parameters)
# =========================
def typical_price(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0

def smma_array(values: np.ndarray, period: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        return np.array([])
    if n < period:
        out = np.full(n, np.nan)
        if n > 0:
            out[-1] = np.nanmean(vals)  # graceful
        return out
    seed = float(np.mean(vals[:period]))
    out = [np.nan] * (period - 1) + [seed]
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + float(vals[i])) / period
        out.append(prev)
    return np.array(out, dtype=float)

def sma_prev_indicator(values: np.ndarray, period: int) -> np.ndarray:
    # SMA over the MA2 array (previous indicator's data)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    window = []
    for i in range(n):
        window.append(vals[i])
        if len(window) > period:
            window.pop(0)
        if len(window) == period and np.isfinite(window).all():
            out[i] = float(np.mean(window))
    return out

def compute_mas(candles: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    highs   = np.array([c["high"]  for c in candles], dtype=float)
    lows    = np.array([c["low"]   for c in candles], dtype=float)
    closes  = np.array([c["close"] for c in candles], dtype=float)
    hlc3    = (highs + lows + closes) / 3.0

    # MA1: Smoothed(9) on typical price (HLC/3)
    ma1 = smma_array(hlc3, 9)
    # MA2: Smoothed(19) on Close
    ma2 = smma_array(closes, 19)
    # MA3: Simple(25) on previous indicator's data (MA2)
    ma3 = sma_prev_indicator(ma2, 25)
    return ma1, ma2, ma3

# =========================
# Patterns & helpers
# =========================
def bits(o,h,l,c):
    body = abs(c-o); rng = max(h-l, 1e-12)
    upper = h - max(o,c); lower = min(o,c) - l
    return {
        "o":o,"h":h,"l":l,"c":c,"body":body,"rng":rng,
        "upper":upper,"lower":lower,
        "is_bull": c>o, "is_bear": c<o,
        "is_doji": body <= 0.35*rng,
        "pin_low": (lower >= 0.6*rng and lower >= 1.2*body),
        "pin_high":(upper >= 0.6*rng and upper >= 1.2*body),
    }

def engulf_bull(prev, cur):
    return (prev["c"] < prev["o"]) and (cur["c"] > cur["o"]) and (cur["o"] <= prev["c"]) and (cur["c"] >= prev["o"])

def engulf_bear(prev, cur):
    return (prev["c"] > prev["o"]) and (cur["c"] < cur["o"]) and (cur["o"] >= prev["c"]) and (cur["c"] <= prev["o"])

def pattern_ok_buy(prev_c, rej_c):
    return rej_c["is_doji"] or rej_c["pin_low"] or engulf_bull(prev_c, rej_c)

def pattern_ok_sell(prev_c, rej_c):
    return rej_c["is_doji"] or rej_c["pin_high"] or engulf_bear(prev_c, rej_c)

# strict trend (ordered + sloped + separated by ATR fraction)
def trend_flags(ma1, ma2, ma3, i, atr, lookback=2, sep_mult=0.20):
    if i - lookback < 0 or any(np.isnan([ma1[i],ma2[i],ma3[i]])):
        return False, False
    sep_ok = (abs(ma1[i]-ma2[i]) > sep_mult*atr) and (abs(ma2[i]-ma3[i]) > sep_mult*atr)
    up  = (ma1[i] > ma2[i] > ma3[i]) and (ma1[i] > ma1[i-lookback]) and (ma2[i] > ma2[i-lookback]) and (ma3[i] > ma3[i-lookback]) and sep_ok
    dn  = (ma1[i] < ma2[i] < ma3[i]) and (ma1[i] < ma1[i-lookback]) and (ma2[i] < ma2[i-lookback]) and (ma3[i] < ma3[i-lookback]) and sep_ok
    return up, dn

# pick the nearest MA (MA1 or MA2) as dynamic S/R
def pick_zone_for_buy(rej_c, ma1, ma2, i):
    d1 = abs(rej_c["l"] - ma1[i]); d2 = abs(rej_c["l"] - ma2[i])
    return ("MA1", float(ma1[i])) if d1 <= d2 else ("MA2", float(ma2[i]))

def pick_zone_for_sell(rej_c, ma1, ma2, i):
    d1 = abs(rej_c["h"] - ma1[i]); d2 = abs(rej_c["h"] - ma2[i])
    return ("MA1", float(ma1[i])) if d1 <= d2 else ("MA2", float(ma2[i]))

# =========================
# Data fetch (Deriv)
# =========================
def fetch_candles(symbol: str, granularity: int, count: int) -> List[Dict]:
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

        # one live update for last candle
        try:
            update = json.loads(ws.recv())
            if "candles" in update and update["candles"]:
                live = update["candles"][-1]
                out[-1] = {
                    "epoch": int(live["epoch"]),
                    "open": float(live["open"]),
                    "high": float(live["high"]),
                    "low": float(live["low"]),
                    "close": float(live["close"]),
                }
        except Exception:
            pass

        return out
    finally:
        try: ws.close()
        except: pass

# =========================
# Scoring + decision per TF
# =========================
def evaluate_timeframe(candles: List[Dict], tf: int) -> Dict:
    if len(candles) < 60:
        return {"tf": tf, "accepted": False, "score": -1, "reason": "insufficient history"}

    opens  = np.array([c["open"] for c in candles], dtype=float)
    highs  = np.array([c["high"] for c in candles], dtype=float)
    lows   = np.array([c["low"]  for c in candles], dtype=float)
    closes = np.array([c["close"]for c in candles], dtype=float)
    rngs   = highs - lows
    atr    = float(np.mean(rngs[-14:])) if len(rngs) >= 14 else float(np.mean(rngs))
    tiny   = max(1e-9, 0.05 * atr)

    ma1, ma2, ma3 = compute_mas(candles)

    # indices: use closed candle as rejection, last as "current"
    i_con = len(candles) - 1       # forming/last
    i_rej = len(candles) - 2       # last closed candle

    prev = bits(opens[i_rej-1], highs[i_rej-1], lows[i_rej-1], closes[i_rej-1]) if i_rej-1 >= 0 else None
    rej  = bits(opens[i_rej], highs[i_rej], lows[i_rej], closes[i_rej])
    if prev is None:
        return {"tf": tf, "accepted": False, "score": -1, "reason": "not enough bars"}

    # trend
    up, dn = trend_flags(ma1, ma2, ma3, i_rej, atr)

    # proximity buffer (zone) ~ fraction of ATR
    buffer_ = 0.25 * atr

    direction = None
    zone_name = None
    zone_val  = None
    reason    = None
    accepted  = False
    score     = -1.0

    # BUY path
    if up and (closes[i_con] > (ma3[i_rej] + tiny)):
        zone_name, zone_val = pick_zone_for_buy(rej, ma1, ma2, i_rej)
        near_zone  = abs(rej["l"] - zone_val) <= buffer_
        close_side = (rej["c"] >= zone_val - buffer_)
        pat_ok     = pattern_ok_buy(prev, rej)

        # score components
        s_trend = 1.0
        s_prox  = max(0.0, 1.0 - abs(rej["l"] - zone_val) / max(1e-9, atr))
        s_pat   = 1.0 if pat_ok else 0.0
        score   = 0.5*s_trend + 0.35*s_prox + 0.15*s_pat

        if near_zone and close_side and pat_ok:
            accepted  = True
            direction = "BUY"
            reason    = f"MA pack up; {zone_name} rejection; pattern OK"
        else:
            accepted  = False
            direction = "BUY"
            reason    = f"Uptrend but rejection not clean (zone={zone_name})"

    # SELL path
    elif dn and (closes[i_con] < (ma3[i_rej] - tiny)):
        zone_name, zone_val = pick_zone_for_sell(rej, ma1, ma2, i_rej)
        near_zone  = abs(rej["h"] - zone_val) <= buffer_
        close_side = (rej["c"] <= zone_val + buffer_)
        pat_ok     = pattern_ok_sell(prev, rej)

        s_trend = 1.0
        s_prox  = max(0.0, 1.0 - abs(rej["h"] - zone_val) / max(1e-9, atr))
        s_pat   = 1.0 if pat_ok else 0.0
        score   = 0.5*s_trend + 0.35*s_prox + 0.15*s_pat

        if near_zone and close_side and pat_ok:
            accepted  = True
            direction = "SELL"
            reason    = f"MA pack down; {zone_name} rejection; pattern OK"
        else:
            accepted  = False
            direction = "SELL"
            reason    = f"Downtrend but rejection not clean (zone={zone_name})"
    else:
        # no clean trend; score stays low
        reason = "No clean trend / MA pack not aligned"

    return {
        "tf": tf,
        "accepted": accepted,
        "direction": direction,  # may be None if no trend
        "score": float(score),
        "reason": reason,
        "rej_index": i_rej,
        "ma_zone": zone_name,
        "ma_value": float(zone_val) if zone_val is not None else None,
        "ma1": ma1, "ma2": ma2, "ma3": ma3,
        "opens": opens, "highs": highs, "lows": lows, "closes": closes,
        "atr": atr,
    }

# =========================
# Chart builder (candlesticks + MAs; zoomed + padding)
# =========================
def build_chart_png(symbol: str, tf: int, evald: Dict, outdir: str) -> str:
    opens  = evald["opens"]; highs = evald["highs"]; lows = evald["lows"]; closes = evald["closes"]
    ma1 = evald["ma1"]; ma2 = evald["ma2"]; ma3 = evald["ma3"]
    n = len(closes)

    look_start = max(0, n - CHART_LOOKBACK)
    o = opens [look_start:n]
    h = highs [look_start:n]
    l = lows  [look_start:n]
    c = closes[look_start:n]
    m1 = ma1[look_start:n]
    m2 = ma2[look_start:n]
    m3 = ma3[look_start:n]

    x = np.arange(len(o))
    fig = plt.figure(figsize=(12, 6), dpi=120)
    ax  = fig.add_subplot(111)

    # candlesticks (simple drawing)
    width = 0.6
    for i in range(len(o)):
        col_up = c[i] >= o[i]
        # wick
        ax.vlines(i, l[i], h[i], linewidth=1)
        # body
        lower = min(o[i], c[i]); upper = max(o[i], c[i])
        ax.add_patch(plt.Rectangle((i - width/2, lower), width, max(upper - lower, 1e-9),
                                   fill=True, alpha=0.6 if col_up else 0.6))

    # plot MAs
    ax.plot(x, m1, linewidth=1.3, label="MA1 Smoothed(9) HLC/3")
    ax.plot(x, m2, linewidth=1.3, label="MA2 Smoothed(19) Close")
    ax.plot(x, m3, linewidth=1.3, label="MA3 SMA(25) on MA2")

    # x-limits with RIGHT PADDING
    ax.set_xlim(-1, len(x) - 1 + RIGHT_PADDING)

    # title
    tf_text = TF_LABEL.get(tf, str(tf))
    verdict = "✅" if evald["accepted"] else "❌"
    dir_txt = evald.get("direction") or "-"
    score   = evald.get("score", 0.0)
    ax.set_title(f"{symbol}  {tf_text}  {verdict} {dir_txt}  score={score:.2f}")

    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    outpath = os.path.join(outdir, f"{symbol}_{tf_text}.png")
    fig.savefig(outpath)
    plt.close(fig)
    return outpath

# =========================
# Assemble + send per asset (only strongest TF, include commentary for others)
# =========================
def analyze_asset(symbol: str):
    best: Optional[Dict] = None
    others: List[Dict] = []

    # evaluate each timeframe
    evaluations = []
    for tf in TIMEFRAMES:
        candles = fetch_candles(symbol, tf, CANDLES_N_FETCH)
        if len(candles) < 3:
            evaluations.append({"tf": tf, "accepted": False, "score": -1, "reason": "no data"})
            continue
        ev = evaluate_timeframe(candles, tf)
        evaluations.append(ev)

    # choose strongest: prefer accepted True; otherwise highest score
    accepted_only = [e for e in evaluations if e.get("accepted")]
    if accepted_only:
        best = max(accepted_only, key=lambda x: x["score"])
        others = [e for e in evaluations if e is not best]
    else:
        # none accepted -> take the highest score rejection
        best = max(evaluations, key=lambda x: x.get("score", -1.0))
        others = [e for e in evaluations if e is not best]

    # build a quick, human-friendly commentary for other TFs
    lines = []
    for e in sorted(others, key=lambda x: x["tf"]):
        tag = "✅" if e.get("accepted") else "❌"
        d   = e.get("direction") or "-"
        lines.append(f"{tag} {TF_LABEL.get(e['tf'], e['tf'])} {d} (score {e.get('score',0):.2f})")

    commentary = "\n".join(lines) if lines else "-"

    # chart for best
    with tempfile.TemporaryDirectory() as td:
        img = build_chart_png(symbol, best["tf"], best, td)

        tag = "✅" if best["accepted"] else "❌"
        d   = best.get("direction") or "-"
        tf_txt = TF_LABEL.get(best["tf"], best["tf"])
        zone = best.get("ma_zone") or "-"
        reason = best.get("reason") or "-"

        text = (
            f"{tag} <b>{symbol}</b> — <b>{tf_txt}</b> {d}\n"
            f"• Reason: {reason}\n"
            f"• Zone: {zone}\n"
            f"• Score: {best.get('score',0):.2f}\n"
            f"• Other TFs:\n{commentary}"
        )
        send_telegram_message(text, photo_path=img)

# =========================
# Main orchestration
# =========================
def main():
    for sym in ASSETS:
        try:
            analyze_asset(sym)
            # small pause between assets to be gentle on API
            time.sleep(1.0)
        except Exception as e:
            send_telegram_message(f"❌ Bot crashed while analyzing {sym}: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        send_telegram_message(f"❌ Bot crashed: {e}")
        raise

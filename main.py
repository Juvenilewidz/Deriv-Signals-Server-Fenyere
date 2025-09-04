# main.py
import os, json, math, asyncio, tempfile, time, traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mplfinance as mpf
import websockets

from bot import send_text, send_photo, notify_crash

# --------------------
# Config
# --------------------
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()  # public test id is fine
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

ASSETS = os.getenv("ASSETS", "R_10,R_50,R_75,1HZ75V,1HZ100V,1HZ150V").split(",")

# seconds (5m,10m,15m)
TIMEFRAMES = list(map(int, os.getenv("TIMEFRAMES", "300,600,900").split(",")))

# Candles to fetch for scoring/plot; more history => nicer chart
CANDLES_N = int(os.getenv("CANDLES_N", "400"))

# one alert per asset per run (muted heartbeat by design)
MAX_SIGNALS_PER_RUN = 99

# Slight preference for 5m; others neutral
TF_WEIGHT = {300: 1.10, 600: 1.00, 900: 1.00}

# Minimum candles needed per TF to evaluate
MIN_ROWS = 60

# --------------------
# Helpers
# --------------------
def utc(ts: int) -> datetime:
    return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc)

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0

def smma(series: pd.Series, period: int) -> pd.Series:
    """MT5-style Smoothed MA: SMMA[i] = SMMA[i-1] + (price[i] - SMMA[i-1]) / period, seeded with SMA."""
    s = series.astype(float).values
    out = np.empty_like(s, dtype=float)
    if len(s) == 0:
        return pd.Series([], dtype=float)
    p = int(period)
    seed = s[:p].mean() if len(s) >= p else s.mean()
    out[0] = seed
    for i in range(1, len(s)):
        out[i] = out[i-1] + (s[i] - out[i-1]) / p
    return pd.Series(out, index=series.index)

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def slope_last(series: pd.Series, lookback: int = 8) -> float:
    if len(series) < 2:
        return 0.0
    y = series[-lookback:].values
    x = np.arange(len(y))
    # simple slope
    denom = (x - x.mean())
    denom = np.sum(denom * denom)
    if denom == 0:
        return 0.0
    m = np.sum((x - x.mean()) * (y - y.mean())) / denom
    return float(m)

def allowed_tfs_for_symbol(symbol: str) -> List[int]:
    """1-second indices: only 5m (300). Others: 5/10/15."""
    if symbol.startswith("1HZ"):
        return [300]
    return [tf for tf in TIMEFRAMES if tf in (300,600,900)]

# --------------------
# Deriv fetch (WebSocket, candles)
# --------------------
async def _fetch_candles_ws(symbol: str, granularity: int, count: int) -> Optional[pd.DataFrame]:
    req = {
        "ticks_history": symbol,
        "granularity": granularity,
        "style": "candles",
        "adjust_start_time": 1,
        "end": "latest",
        "count": count
    }
    async with websockets.connect(DERIV_WS_URL, ping_interval=20, close_timeout=5) as ws:
        await ws.send(json.dumps(req))
        raw = await ws.recv()
        data = json.loads(raw)
        if "error" in data:
            print("[DERIV ERR]", symbol, granularity, data["error"])
            return None
        candles = data.get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles)
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","epoch":"time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
        return df

def fetch_candles(symbol: str, tf: int, count: int) -> Optional[pd.DataFrame]:
    try:
        return asyncio.run(_fetch_candles_ws(symbol, tf, count))
    except Exception as e:
        print("[FETCH EXC]", symbol, tf, e)
        return None

# --------------------
# Signal logic
# --------------------
def compute_mas(df: pd.DataFrame) -> Tuple[pd.Series,pd.Series,pd.Series]:
    tp = typical_price(df)
    ma1 = smma(tp, 9)           # SMMA(9) on HLC/3
    ma2 = smma(df["close"], 19) # SMMA(19) on Close
    ma3 = sma(ma2, 25)          # SMA(25) of MA2
    return ma1, ma2, ma3

def evaluate(df: pd.DataFrame, tf: int) -> Optional[Dict]:
    if df is None or len(df) < MIN_ROWS:
        return None
    ma1, ma2, ma3 = compute_mas(df)

    close = df["close"]
    c = float(close.iloc[-1])

    # Basic trend by MA3 slope
    s3 = slope_last(ma3)
    trend = "up" if s3 > 0 else "down" if s3 < 0 else "flat"

    # Rejection logic: price pulls back to MA bands then resumes trend
    # Use distance from MA2 and candle direction
    prev_c = float(close.iloc[-2]) if len(close) > 1 else c
    bull = c > prev_c
    bear = c < prev_c

    near_ma2 = abs(c - float(ma2.iloc[-1])) <= max(0.0001, 0.003 * c)  # ~0.3%
    above_all = c > float(ma1.iloc[-1]) and c > float(ma2.iloc[-1]) and c > float(ma3.iloc[-1])
    below_all = c < float(ma1.iloc[-1]) and c < float(ma2.iloc[-1]) and c < float(ma3.iloc[-1])

    accepted = False
    reason = "None"
    direction = "-"

    # Decide direction and acceptance
    if trend == "up":
        direction = "BUY"
        if above_all and near_ma2 and bull:
            accepted = True
            reason = "MA support rejection"
        else:
            reason = "Uptrend but rejection not clean"
    elif trend == "down":
        direction = "SELL"
        if below_all and near_ma2 and bear:
            accepted = True
            reason = "MA resistance rejection"
        else:
            reason = "Downtrend but rejection not clean"
    else:
        direction = "NEUTRAL"
        accepted = False
        reason = "Flat trend"

    # Score: trend strength + confluence
    s1 = abs(slope_last(ma1))
    s2 = abs(slope_last(ma2))
    strength = (abs(s3)*2 + s2 + 0.5*s1)
    confluence = 0.0
    if near_ma2: confluence += 0.5
    if above_all or below_all: confluence += 0.5
    if accepted: confluence += 0.5
    score = float(strength + confluence)

    return {
        "tf": tf,
        "accepted": accepted,
        "direction": direction,
        "reason": reason,
        "score": score,
        "ma1": ma1, "ma2": ma2, "ma3": ma3,
        "df": df
    }

# --------------------
# Charting
# --------------------
def plot_chart(symbol: str, res: Dict) -> Optional[str]:
    try:
        df = res["df"][["open","high","low","close"]].copy()
        ma1, ma2, ma3 = res["ma1"], res["ma2"], res["ma3"]

        addplots = [
            mpf.make_addplot(ma1, color="#1f77b4", width=1.2),
            mpf.make_addplot(ma2, color="#ff7f0e", width=1.2),
            mpf.make_addplot(ma3, color="#2ca02c", width=1.6),
        ]

        title = f"{symbol} | {res['tf']//60}m | {'ACCEPTED' if res['accepted'] else 'REJECTED'}"
        style = mpf.make_mpf_style(gridstyle="-", gridaxis="y")
        fig, axlist = mpf.plot(
            df,
            type="candle",
            addplot=addplots,
            returnfig=True,
            style=style,
            figsize=(13, 4.8),   # wide but taller than default; tweak here if you want
            tight_layout=True,
            datetime_format="%H:%M\n%m-%d"
        )
        fig.suptitle(title, fontsize=12)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, dpi=140)
        plt.close(fig)
        return tmp.name
    except Exception as e:
        print("[CHART EXC]", e)
        return None

# --------------------
# Orchestration
# --------------------
def analyze_and_send() -> None:
    signals_sent = 0

    for symbol in ASSETS:
        try:
            per_tf: List[Dict] = []
            for tf in allowed_tfs_for_symbol(symbol):
                df = fetch_candles(symbol, tf, CANDLES_N)
                if df is None or len(df) < MIN_ROWS:
                    continue
                res = evaluate(df, tf)
                if not res:
                    continue
                # weight 5m slightly
                res["score"] *= TF_WEIGHT.get(tf, 1.0)
                per_tf.append(res)

            if not per_tf:
                # nothing to say for this asset this run
                continue

            accepted = [r for r in per_tf if r["accepted"]]
            if accepted:
                best = max(accepted, key=lambda r: r["score"])
            else:
                best = max(per_tf, key=lambda r: r["score"])

            # Compact other-TFs summary (no spam)
            lines = []
            for r in sorted(per_tf, key=lambda x: x["tf"]):
                tag = "✅" if r["accepted"] else "❌"
                lines.append(f"{r['tf']//60}m → {tag} ({r['reason']}) (s={r['score']:.1f})")
            others = " | ".join(lines)

            # Build caption + chart
            emoji = "✅" if best["accepted"] else "❌"
            caption = (
                f"{emoji} <b>{symbol}</b> | <b>{best['tf']//60}m</b> | <b>{best['direction']}</b>\n"
                f"Reason: {best['reason']}\n\n"
                f"<i>Other TFs:</i> {others}"
            )

            chart = plot_chart(symbol, best)
            ok = False
            if chart:
                ok = send_photo(caption, chart)
                try: os.remove(chart)
                except: pass
            if not ok:
                ok = send_text(caption)

            if ok:
                signals_sent += 1
                if signals_sent >= MAX_SIGNALS_PER_RUN:
                    break

        except Exception as e:
            notify_crash(f"analyzing {symbol}", e)

# --------------------
# Entry
# --------------------
if __name__ == "__main__":
    try:
        analyze_and_send()
    except Exception as e:
        notify_crash("main", e)

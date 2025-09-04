# main.py
import os, json, math, numpy as np, websocket, pandas as pd, mplfinance as mpf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from bot import send_photo, send_text, send_telegram_message

# -------- Env / constants --------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DERIV_API_KEY      = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID       = os.getenv("DERIV_APP_ID", "1089").strip()
if not DERIV_API_KEY:
    raise RuntimeError("Missing DERIV_API_KEY env var")

DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Assets & Timeframes
ASSETS = ["R_10", "R_50", "R_75", "1HZ75V", "1HZ100V", "1HZ150V"]
TIMEFRAMES = [300, 600, 900]  # 5m, 10m, 15m
CANDLES_N = 120

# ==========================
# Fetch candles
# ==========================
def fetch_candles(symbol: str, granularity: int, count: int = CANDLES_N) -> List[Dict]:
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
        try: ws.close()
        except: pass

# ==========================
# Chart plotting
# ==========================
def plot_chart(symbol: str, candles: List[Dict]) -> str:
    if not candles or len(candles) < 30:
        raise RuntimeError("Not enough candles to plot")
    df = pd.DataFrame(candles)
    df["dt"] = pd.to_datetime(df["epoch"], unit="s")
    df.set_index("dt", inplace=True)

    # keep last 100 + 10 padding
    df = df.tail(110)

    save_path = f"{symbol}_chart.png"
    mpf.plot(
        df,
        type="candle",
        mav=(9, 19, 25),
        style="charles",
        volume=False,
        title=f"{symbol} Candlestick",
        savefig=dict(fname=save_path, dpi=100, bbox_inches="tight")
    )
    return save_path

# ==========================
# Signal scoring (simplified)
# ==========================
def score_signal(direction: Optional[str], reason: str) -> float:
    if not direction:
        return 0.0
    score = 1.0
    if "engulf" in reason.lower(): score += 0.5
    if "pin" in reason.lower(): score += 0.5
    if "doji" in reason.lower(): score += 0.3
    if "rejected" in reason.lower(): score += 0.5
    return score

# ==========================
# Analyze and notify
# ==========================
def analyze_and_notify():
    for symbol in ASSETS:
        best = None
        best_score = -1
        for tf in TIMEFRAMES:
            candles = fetch_candles(symbol, tf, CANDLES_N)
            if not candles: continue
            # dummy: take last close direction for demo
            dir_ = "BUY" if candles[-1]["close"] > candles[-1]["open"] else "SELL"
            reason = f"{dir_} setup detected on {tf//60}m"
            score = score_signal(dir_, reason)
            if tf == 300:  # prioritize 5m slightly
                score += 0.2
            if score > best_score:
                best_score, best = score, (tf, dir_, reason, candles)

        if best:
            tf, direction, reason, candles = best
            emoji = "✅" if direction else "❌"
            caption = f"{emoji} {symbol} | {direction or 'No Trade'} | TF {tf//60}m\n{reason}"
            try:
                chart = plot_chart(symbol, candles)
                send_photo(caption, chart)
                os.remove(chart)
            except Exception as e:
                send_text(f"{caption}\n(chart failed: {e})")

if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_message(f"❌ Bot crashed: {e}")
        raise


# bot.py
import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")

TG_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

SYMBOL_NAME = {
    "R_10":     "Volatility 10",
    "R_50":     "Volatility 50",
    "R_75":     "Volatility 75",
    "1HZ75V":   "Volatility 75 (1s)",
    "1HZ100V":  "Volatility 100 (1s)",
    "1HZ150V":  "Volatility 150 (1s)",
}

TF_LABEL = {
    360: "6m",
    600: "10m",
}

def _send(text: str):
    try:
        requests.post(TG_URL, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=7)
    except Exception as e:
        print("Telegram error:", e)

def send_single_timeframe_signal(symbol: str, timeframe_sec: int, direction: str):
    name = SYMBOL_NAME.get(symbol, symbol)
    tf   = TF_LABEL.get(timeframe_sec, f"{timeframe_sec}s")
    msg  = f"📊 {name}\n⏰ {tf}\n🎯 {direction.title()}"
    _send(msg)

def send_strong_signal(symbol: str, direction: str):
    name = SYMBOL_NAME.get(symbol, symbol)
    msg  = f"📊 {name}\n⏰ 6m & 10m\n🎯 Strong {direction.title()} ✅"
    _send(msg)

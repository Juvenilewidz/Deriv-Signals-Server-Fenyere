# bot.py
import os
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def _send_text(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram bot token or chat id")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        if resp.status_code != 200:
            print("Telegram sendMessage error:", resp.text)
            return False
        return True
    except Exception as e:
        print("Telegram text send exception:", e)
        return False

def _send_photo(caption: str, filepath: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram bot token or chat id")
        return False
    if not filepath or not os.path.exists(filepath):
        print("⚠️ Chart file missing:", filepath)
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(filepath, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            resp = requests.post(url, data=data, files=files, timeout=20)
        if resp.status_code != 200:
            print("Telegram sendPhoto error:", resp.text)
            return False
        return True
    except Exception as e:
        print("Telegram photo send exception:", e)
        return False

# -------------------
# Public helpers used by main.py
# -------------------

def send_telegram_message(token: str, chat_id: str, message: str) -> None:
    """Generic send text (used for crash/heartbeat)."""
    if not token or not chat_id:
        print("⚠️ Missing Telegram config")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=15)
        if resp.status_code != 200:
            print("send_telegram_message error:", resp.text)
    except Exception as e:
        print("send_telegram_message exception:", e)

def send_single_timeframe_signal(symbol: str, tf: int, direction: str, reason: str,
                                 chart_path: str = None) -> None:
    """Send a signal for one timeframe, with optional chart."""
    caption = f"📊 {symbol} | {tf//60}m | {direction}\n{reason}\n\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    sent = False
    if chart_path:
        sent = _send_photo(caption, chart_path)
    if not sent:
        _send_text(caption)

def send_strong_signal(symbol: str, direction: str, reasons: dict,
                       chart_path: str = None) -> None:
    """Send a strong signal (multi-TF agree), with optional chart."""
    tf_reasons = "\n".join([f"{tf//60}m: {rsn}" for tf, rsn in reasons.items()])
    caption = f"💪 STRONG SIGNAL\n{symbol} | {direction}\n{tf_reasons}\n\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    sent = False
    if chart_path:
        sent = _send_photo(caption, chart_path)
    if not sent:
        _send_text(caption)

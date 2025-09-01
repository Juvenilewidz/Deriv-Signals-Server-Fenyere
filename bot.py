# bot.py
import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def _tf_label(seconds: int) -> str:
    return "10min" if seconds == 600 else "15min" if seconds == 900 else "20min" if seconds == 1200 else f"{seconds}s"

def _sym_label(symbol: str) -> str:
    mapping = {
        "R_10": "Volatility 10",
        "R_50": "Volatility 50",
        "R_75": "Volatility 75",
        "1HZ75V": "Volatility 75 (1s)",
        "1HZ100V": "Volatility 100 (1s)",
        "1HZ150V": "Volatility 150 (1s)",
    }
    return mapping.get(symbol, symbol)

def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        requests.post(url, data=payload, timeout=15)
    except Exception:
        # swallow to avoid crashing the run on Telegram hiccups
        pass

def send_single_timeframe_signal(symbol: str, timeframe: int, direction: str, reason: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym = _sym_label(symbol)
    tf  = _tf_label(timeframe)
    text = (
        f"📊 <b>{sym}</b>\n"
        f"⏰ {tf}\n"
        f"🎯 <b>{direction.title()}</b>\n"
        f"🧠 Reason: {reason}"
    )
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)

def send_strong_signal(symbol: str, direction: str, reasons_by_tf: dict) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym = _sym_label(symbol)
    r10  = reasons_by_tf.get(600) or "-"
    r15 = reasons_by_tf.get(900) or "-"
    text = (
        f"📊 <b>{sym}</b>\n"
        f"⏰ 10min & 15min AGREE\n"
        f"💪 <b>STRONG {direction.upper()}</b>\n"
        f"🧠 10m: {r10}\n"
        f"🧠 15m: {r15}"
    )
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)

def send_strong_signal(symbol: str, direction: str, reasons_by_tf: dict) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym = _sym_label(symbol)
    r15  = reasons_by_tf.get(900) or "-"
    r20 = reasons_by_tf.get(1200) or "-"
    text = (
        f"📊 <b>{sym}</b>\n"
        f"⏰ 15min & 20min AGREE\n"
        f"💪 <b>STRONG {direction.upper()}</b>\n"
        f"🧠 15m: {r15}\n"
        f"🧠 20m: {r20}"
    )

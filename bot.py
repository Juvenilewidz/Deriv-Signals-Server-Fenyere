# bot.py
import os
import io
import requests
import matplotlib.pyplot as plt
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def _tf_label(seconds: int) -> str:
    mapping = {
        300: "5min",
        600: "10min",
        900: "15min",
    }
    return mapping.get(seconds, f"{seconds}s")

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
        pass

def send_telegram_chart(token: str, chat_id: str, candles, ma1, ma2, ma3, signal_idx, direction, symbol, tf_label):
    try:
        closes = [c["close"] for c in candles]
        times  = [datetime.fromtimestamp(c["epoch"]) for c in candles]

        plt.figure(figsize=(8,4))
        plt.plot(times, closes, label="Price", color="black")
        if ma1: plt.plot(times, ma1, label="MA1", color="blue", linewidth=1)
        if ma2: plt.plot(times, ma2, label="MA2", color="orange", linewidth=1)
        if ma3: plt.plot(times, ma3, label="MA3", color="green", linewidth=1)

        # highlight rejection candle
        if 0 <= signal_idx < len(candles):
            plt.axvline(times[signal_idx], color="red" if direction=="SELL" else "green", linestyle="--", linewidth=1.2)

        plt.title(f"{_sym_label(symbol)} | {tf_label} | {direction}")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        files = {"photo": buf}
        data = {"chat_id": chat_id}
        requests.post(url, files=files, data=data, timeout=20)
    except Exception:
        pass

def send_single_timeframe_signal(symbol: str, timeframe: int, direction: str, reason: str,
                                 candles=None, ma1=None, ma2=None, ma3=None, signal_idx=None) -> None:
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
    if candles:
        send_telegram_chart(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, candles, ma1, ma2, ma3, signal_idx, direction, symbol, tf)

def send_strong_signal(symbol: str, direction: str, reasons_by_tf: dict,
                       candles=None, ma1=None, ma2=None, ma3=None, signal_idx=None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    sym = _sym_label(symbol)
    parts = []
    for tf, reason in reasons_by_tf.items():
        parts.append(f"🧠 {_tf_label(tf)}: {reason}")
    tf_labels = " & ".join([_tf_label(tf) for tf in reasons_by_tf.keys()])

    text = (
        f"📊 <b>{sym}</b>\n"
        f"⏰ {tf_labels} AGREE\n"
        f"💪 <b>STRONG {direction.upper()}</b>\n"
        + "\n".join(parts)
    )
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, text)
    if candles:
        send_telegram_chart(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, candles, ma1, ma2, ma3, signal_idx, direction, symbol, tf_labels)

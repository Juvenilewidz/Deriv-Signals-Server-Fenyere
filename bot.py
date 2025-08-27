# bot.py
import json
import urllib.parse
import urllib.request

def send_single_timeframe_signal(symbol, interval, signal, tf, direction):
    message = f"""
📊 Signal Alert!
Symbol: {symbol}
Timeframe: {tf}
Direction: {direction}
Signal: {signal}
"""
    send_telegram_message(message)

def _tg_post(token: str, method: str, payload: dict) -> None:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(payload).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=15) as _:
        pass  # best-effort; ignore body

def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    _tg_post(token, "sendMessage", {"chat_id": chat_id, "text": text})

def send_single_timeframe_signal(token: str, chat_id: str, symbol: str, tf: int, direction: str):
    def send_single_timeframe_signal(token: str, chat_id: str,
                                 symbol: str, tf: int, direction: str,
                                 reason: str = "") -> None:
    tf_label = f"M{tf // 60}"
    msg = f"📊 {symbol} | {tf_label}\nSignal: {direction}"
    if reason:
        msg += f"\nReason: {reason}"
    send_telegram_message(token, chat_id, msg)
    send_telegram_message(token, chat_id, msg)

def send_strong_signal(token: str, chat_id: str, symbol: str, direction: str, reason: str = ""):
    msg = f"💥 STRONG {direction.upper()} | {symbol}" + (f"\n{reason}" if reason else "")
    send_telegram_message(token, chat_id, msg)

# Backward-compat for earlier import typo
send_strongs_signal = send_strong_signal
#

# ====== Reasoning ============
def send_single_timeframe_signal(token: str, chat_id: str,
                                 symbol: str, tf: int, direction: str,
                                 reason: str = "") -> None:
    tf_label = f"M{tf // 60}"
    msg = f"📊 {symbol} | {tf_label}\nSignal: {direction}"
    if reason:
        msg += f"\nReason: {reason}"
    send_telegram_message(token, chat_id, msg)

def send_strong_signal(token: str, chat_id: str,
                       symbol: str, direction: str,
                       reason: str = "") -> None:
    msg = f"💥 STRONG {direction.upper()} | {symbol}"
    if reason:
        msg += f"\nReason: {reason}"
    send_telegram_message(token, chat_id, msg)

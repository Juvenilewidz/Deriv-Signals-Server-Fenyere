# bot.py
import requests
import os
from typing import Dict, Any

# Telegram send utility
def send_telegram_message(bot_token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send telegram message: {e}")


# ============ SIGNAL FUNCTIONS ============

# Single timeframe signal
def send_single_timeframe_signal(bot_token: str, chat_id: str, symbol: str, timeframe: str, signal: Dict[str, Any]):
    """
    Sends a trading signal based on a single timeframe analysis.
    """
    text = (
        f"📊 *Single-Timeframe Signal*\n"
        f"Symbol: `{symbol}`\n"
        f"Timeframe: `{timeframe}`\n"
        f"Signal: *{signal['type'].upper()}*\n"
        f"Reason: {signal.get('reason', 'N/A')}\n"
    )
    send_telegram_message(bot_token, chat_id, text)


# Multi-timeframe / Strong confluence signal
def send_strongs_signal(bot_token: str, chat_id: str, symbol: str, signals: Dict[str, Any]):
    """
    Sends a 'strong' signal when multiple timeframe confirmations align.
    """
    text = f"🚀 *Strong Multi-Timeframe Signal*\nSymbol: `{symbol}`\n\n"
    for tf, sig in signals.items():
        text += f"TF {tf}: *{sig['type'].upper()}* ({sig.get('reason', 'N/A')})\n"

    send_telegram_message(bot_token, chat_id, text)

# bot.py
import os
import json
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str) -> bool:
    """
    Sends a plain text Telegram message. Returns True on success.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": str(TELEGRAM_CHAT_ID), "text": text, "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"Telegram error: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        print(f"Telegram exception: {e}")
        return False

def send_telegram_block(header: str, lines: list) -> None:
    """
    Sends a formatted block message (header + bullet lines).
    """
    msg = header + "\n" + "\n".join(f"• {ln}" for ln in lines)
    send_telegram_message(msg)

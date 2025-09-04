# bot.py
import os
import time
import traceback
from typing import Optional
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_text(text: str):
    """Send plain text message"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Telegram text send failed:", e)

def send_photo(photo_path: str, caption: str = ""):
    """Send photo with optional caption"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as img:
        files = {"photo": img}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        try:
            requests.post(url, data=data, files=files, timeout=20)
        except Exception as e:
            print("Telegram photo send failed:", e)

def send_telegram_message(text: str, photo_path: str = None):
    """Unified sender: if photo is provided, send with chart, else send plain text"""
    if photo_path:
        send_photo(photo_path, caption=text)
    else:
        send_text(text)

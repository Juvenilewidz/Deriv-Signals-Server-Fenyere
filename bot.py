# bot.py
"""
Small helper for Telegram messaging used by main.py
Requires `requests` in your environment.
"""

import requests
import traceback

def send_telegram_message(token: str, chat_id: str, text: str):
    """Send a simple message. Returns (ok, info)"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        r = requests.post(url, data=payload, timeout=20)
        return (r.ok, r.text if not r.ok else "ok")
    except Exception as e:
        return (False, str(e))

def send_telegram_photo(token: str, chat_id: str, caption: str, photo_path: str):
    """Send a photo with caption. Returns (ok, info)"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            r = requests.post(url, files=files, data=data, timeout=30)
        return (r.ok, r.text if not r.ok else "ok")
    except Exception as e:
        return (False, str(e))

def send_single_timeframe_signal(symbol: str, tf: int, direction: str, reason: str, chart_path: str = None):
    """Fallback textual signal helper (keeps backward compatibility)"""
    text = f"{symbol} | {tf//60}m | {direction}\nReason: {reason}"
    # In main.py fallback this function is only called when TELEGRAM creds are missing.
    print("[SINGLE TF SIGNAL]", text, "chart:", chart_path)
    return True

def send_strong_signal(symbol: str, direction: str, details: str, chart_path: str = None):
    text = f"STRONG: {symbol} | {direction}\n{details}"
    print("[STRONG SIGNAL]", text, "chart:", chart_path)
    return True

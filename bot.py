# bot.py - helper functions to send Telegram messages/photos for the Deriv signal bot.

import os
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"

def _safe_post(url: str, files=None, data=None, timeout=10):
    try:
        if files:
            r = requests.post(url, files=files, data=data, timeout=timeout)
        else:
            r = requests.post(url, json=data, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)

def send_telegram_text_direct(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "missing token/chat_id"
    url = TELEGRAM_API_BASE.format(token=token, method="sendMessage")
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    ok, info = _safe_post(url, data=data)
    return ok, info

def send_telegram_message(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    return send_telegram_text_direct(token, chat_id, text)

def send_telegram_photo(token: str, chat_id: str, caption: str, photo_path: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "missing token/chat_id"
    url = TELEGRAM_API_BASE.format(token=token, method="sendPhoto")
    try:
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
            ok, info = _safe_post(url, files=files, data=data)
            return ok, info
    except Exception as e:
        return False, str(e)

def send_single_timeframe_signal(symbol: str, tf: int, direction: str, reason: str) -> Tuple[bool, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "telegram env not set"
    tfm = f"{int(tf/60)}m"
    emoji = "🔸" if direction in ("BUY","SELL") else "⚠️"
    text = (
        f"<b>{symbol}</b>\n"
        f"⏰ {tfm}\n\n"
        f"{emoji} <b>{direction}</b>\n\n"
        f"🧠 {reason}"
    )
    return send_telegram_text_direct(token, chat_id, text)

def send_strong_signal(symbol: str, direction: str, details: Dict[int, Tuple[Optional[str], Optional[str]]]) -> Tuple[bool, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "telegram env not set"
    lines = []
    for tf, val in sorted(details.items()):
        sig, rsn = val if isinstance(val, (list,tuple)) else (None, None)
        lines.append(f"{int(tf/60)}m: {sig or 'None'} -> {rsn or ''}")
    text = (
        f"🚀 <b>Strong signal</b>\n"
        f"{symbol} — <b>{direction}</b>\n\n"
        + "\n".join(lines)
    )
    return send_telegram_text_direct(token, chat_id, text)

def send_heartbeat(checked_assets: List[str]):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "telegram env not set"
    ts = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        "🤖 Bot heartbeat – alive\n"
        "⏰ No signals right now.\n"
        f"📊 Checked: {', '.join(checked_assets)}\n"
        f"🕒 {ts}"
    )
    return send_telegram_text_direct(token, chat_id, msg)

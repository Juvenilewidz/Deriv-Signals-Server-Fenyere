# bot.py
import os
import time
import traceback
from typing import Optional
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

_API = "https://api.telegram.org/bot{}/".format(TELEGRAM_BOT_TOKEN)

def _post(method: str, data: dict, files=None) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Local dry-run fallback prints so you can test without Telegram
        print("[TG:{}] {}".format(method, {**data, **({'_has_file': bool(files)} if files else {})}))
        return True
    try:
        url = _API + method
        r = requests.post(url, data=data, files=files, timeout=30)
        ok = r.ok and r.json().get("ok", False)
        if not ok:
            print("[TG ERROR]", r.text[:300])
        return ok
    except Exception as e:
        print("[TG EXC]", e)
        return False

def send_text(text: str) -> bool:
    return _post("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})

def send_photo(caption: str, photo_path: str) -> bool:
    files = {"photo": open(photo_path, "rb")}
    try:
        return _post("sendPhoto", {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}, files)
    finally:
        try: files["photo"].close()
        except: pass

def notify_crash(where: str, err: Exception) -> None:
    msg = f"⚠️ <b>Bot crashed</b> in <code>{where}</code>\n<code>{repr(err)}</code>\n<pre>{traceback.format_exc()[-1500:]}</pre>"
    try:
        send_text(msg)
    except:
        print("[CRASH-NOTIFY-FAIL]")

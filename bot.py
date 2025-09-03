# bot.py
"""
Bot helpers: small, robust Telegram helpers used by main.py
- send_single_timeframe_signal(symbol, tf, direction, reason, chart_path=None)
- send_strong_signal(symbol, direction, reasons, chart_path=None)
- send_rejection_with_chart(symbol, tf, candles, reason)
- send_telegram_message(text)
- send_heartbeat(checked_assets)
"""

import os
import requests
from datetime import datetime
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def _send_text(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram token/chat")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print("send text err:", e)
        return False

def _send_photo(caption: str, filepath: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram token/chat")
        return False
    if not filepath or not os.path.exists(filepath):
        print("⚠️ missing chart file:", filepath)
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(filepath, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            r = requests.post(url, data=data, files=files, timeout=20)
        return r.status_code == 200
    except Exception as e:
        print("send photo err:", e)
        return False

def send_telegram_message(text: str):
    """Public send text wrapper used by main.py for heartbeat/crash messages."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    body = f"{text}\n\n{now}"
    _send_text(body)

def send_heartbeat(checked_assets: list):
    """Short heartbeat showing checked assets (used by main)."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        "🤖 Bot heartbeat – alive\n"
        "⏰ No signals right now.\n"
        f"📊 Checked: {', '.join(checked_assets)}\n"
        f"🕒 {now}"
    )
    _send_text(msg)

def send_single_timeframe_signal(symbol: str, tf: int, direction: str, reason: str, chart_path: str = None):
    caption = f"📊 {symbol} | {tf//60}m | {direction}\n{reason}"
    if chart_path:
        ok = _send_photo(caption, chart_path)
        if ok:
            return True
    return _send_text(caption)

def send_strong_signal(symbol: str, direction: str, reasons: dict, chart_path: str = None):
    tf_reasons = "\n".join([f"{tf//60}m: {r}" for tf, r in reasons.items()])
    caption = f"💪 STRONG SIGNAL\n{symbol} | {direction}\n{tf_reasons}"
    if chart_path:
        ok = _send_photo(caption, chart_path)
        if ok:
            return True
    return _send_text(caption)

def send_rejection_with_chart(symbol: str, tf: int, candles: list, reason: str):
    """Send a short rejection caption + chart (chart created here)."""
    short_reason = str(reason).split(":")[-1].strip()
    caption = f"❌ Rejected\n{symbol} | {tf//60}m\nReason: {short_reason}"
    # build small chart (reuse a simple line snapshot to keep it light)
    try:
        times = [datetime.utcfromtimestamp(c["epoch"]) for c in candles]
        closes = [c["close"] for c in candles]
        # show last 100 candles
        last_n = 100
        if len(times) > last_n:
            times = times[-last_n:]
            closes = closes[-last_n:]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, closes, linewidth=1)
        ax.set_title(f"{symbol} {tf//60}m | Rejected")
        ax.set_xlim(times[0], times[-1] + (times[-1] - times[0]) / max(1, int(last_n/10)))
        ax.set_ylabel("Price")
        plt.xticks(rotation=25)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.tight_layout()
        plt.savefig(tmp.name, dpi=120)
        plt.close(fig)
        sent = _send_photo(caption, tmp.name)
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        if not sent:
            _send_text(caption)
    except Exception as e:
        print("send_rejection_with_chart error:", e)
        _send_text(caption)

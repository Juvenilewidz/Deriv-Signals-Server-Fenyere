# bot.py
import os
import io
import math
import requests
from typing import List, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timezone

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ============== Telegram ==============
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})

def send_telegram_photo(caption: str, image_bytes: bytes):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes)}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    requests.post(url, files=files, data=data)

# ============== Chart (candlesticks) ==============
def _draw_candles(ax, candles, color_up="#26a69a", color_down="#ef5350"):
    """
    Lightweight candlestick drawer (no external libs).
    """
    for i, c in enumerate(candles):
        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
        color = color_up if cl >= o else color_down
        # wick
        ax.vlines(i, l, h, linewidth=1, color=color)
        # body
        body_low  = min(o, cl)
        body_high = max(o, cl)
        ax.add_patch(plt.Rectangle((i - 0.35, body_low), 0.7, body_high - body_low, linewidth=0.6,
                                   edgecolor=color, facecolor=color, alpha=0.9))

def make_signal_chart(asset_alias: str,
                      tf: int,
                      candles: List[dict],
                      ma1: List[Optional[float]],
                      ma2: List[Optional[float]],
                      ma3: List[Optional[float]],
                      highlight_index: Optional[int] = None,
                      pad_right: int = 10) -> bytes:
    """
    Returns PNG bytes. Shows lots of history with compact bars + right padding.
    """
    n = len(candles)
    # figure wide, so bars look smaller; height moderate
    fig, ax = plt.subplots(figsize=(13, 5), dpi=140)

    _draw_candles(ax, candles)

    # plot MAs where available
    xs = range(n)
    y1 = [ma1[i] if ma1[i] is not None else float("nan") for i in range(n)]
    y2 = [ma2[i] if ma2[i] is not None else float("nan") for i in range(n)]
    y3 = [ma3[i] if ma3[i] is not None else float("nan") for i in range(n)]
    ax.plot(xs, y1, linewidth=1.2, label="MA1(9) SMMA HLC3")
    ax.plot(xs, y2, linewidth=1.2, label="MA2(19) SMMA Close")
    ax.plot(xs, y3, linewidth=1.2, label="MA3(25) SMA on MA2")

    # highlight rejection bar
    if highlight_index is not None and 0 <= highlight_index < n:
        c = candles[highlight_index]
        ax.vlines(highlight_index, c["low"], c["high"], linewidth=3, color="#ffd54f", alpha=0.9)
        ax.add_patch(plt.Rectangle((highlight_index - 0.45, min(c["open"], c["close"])),
                                   0.9, abs(c["close"] - c["open"]), linewidth=1.4,
                                   edgecolor="#ffa000", facecolor="none"))

    ax.set_xlim(-1, n + pad_right)  # right padding for future space
    # y-lims with a little headroom
    lows  = [c["low"] for c in candles]
    highs = [c["high"] for c in candles]
    ymin, ymax = min(lows), max(highs)
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)

    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    tf_min = tf // 60
    ax.set_title(f"{asset_alias}  •  {tf_min}m  •  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()

# ============== Final message helpers ==============
def _tf_label(tf_sec: int) -> str:
    return f"{tf_sec // 60}m"

def send_accepted_signal(asset_alias: str, tf: int, direction: str, score: float, reason: str, png_bytes: bytes):
    emoji = "✅"
    caption = (
        f"{emoji} <b>{asset_alias}</b> | <b>{_tf_label(tf)}</b>\n"
        f"<b>{direction}</b>  •  score {score:.2f}\n"
        f"{reason}"
    )
    send_telegram_photo(caption, png_bytes)

def send_rejected_signal(asset_alias: str, tf: int, score: float, reason: str, png_bytes: bytes):
    emoji = "❌"
    caption = (
        f"{emoji} <b>{asset_alias}</b> | <b>{_tf_label(tf)}</b>\n"
        f"Rejected  •  score {score:.2f}\n"
        f"{reason}"
    )
    send_telegram_photo(caption, png_bytes)

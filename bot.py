# bot.py
import os, io, math, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ========= low-level send photo =========
def _send_photo_bytes(png_bytes: bytes, caption: str):
    url = f"{TELEGRAM_API}/sendPhoto"
    files = {"photo": ("chart.png", png_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, files=files, timeout=30)
        if r.status_code != 200:
            print("TG send error", r.status_code, r.text)
    except Exception as e:
        print("TG exception", e)

def send_simple_text(msg: str):
    url = f"{TELEGRAM_API}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data, timeout=20)
    except Exception as e:
        print("TG text fail", e)

# ========= chart renderer =========
def _plot_candles(candles, ma1, ma2, ma3, i_rej, i_con, title_label):
    opens  = [c["open"] for c in candles]
    highs  = [c["high"] for c in candles]
    lows   = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    n = len(candles)

    # figure tall portrait (adjust for bigger image if required)
    fig = plt.figure(figsize=(8, 11), dpi=120)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title_label, fontsize=12, pad=8)

    # candle width adapt to n
    width = max(0.25, min(0.6, 0.6 * (200.0 / max(200, n))))

    for i in range(n):
        o,h,l,c = opens[i], highs[i], lows[i], closes[i]
        color = "#2ca02c" if c >= o else "#d62728"
        ax.vlines(i, l, h, linewidth=1.0, color=color)
        # body
        bottom = min(o,c); height = max(1e-9, abs(c-o))
        rect = plt.Rectangle((i - width/2, bottom), width, height, facecolor=color, edgecolor=color, linewidth=0.6)
        ax.add_patch(rect)

    # plot MAs (skip Nones)
    def plot_ma(series, color, lab=None):
        xs, ys = [], []
        for idx, val in enumerate(series):
            if val is not None and not (isinstance(val,float) and math.isnan(val)):
                xs.append(idx); ys.append(val)
            else:
                if xs:
                    ax.plot(xs, ys, color=color, linewidth=1.4, label=lab)
                    lab = None
                    xs, ys = [], []
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.4, label=lab)

    plot_ma(ma1, "#1f77b4", "MA1 SMMA(9) HLC/3")
    plot_ma(ma2, "#ff7f0e", "MA2 SMMA(19) Close")
    plot_ma(ma3, "#9467bd", "MA3 SMA(25) on MA2")

    # mark rejection / confirmation bars
    if 0 <= i_rej < n:
        ax.axvline(i_rej, color="black", linestyle="--", linewidth=0.9, alpha=0.8)
    if 0 <= i_con < n:
        ax.axvline(i_con, color="black", linestyle=":", linewidth=0.9, alpha=0.7)

    # padding forward by 10 bars
    ax.set_xlim(-5, n + 10)
    ax.grid(True, linewidth=0.25, alpha=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ========= public helper used by main =========
def send_photo_with_caption(asset_name: str, timeframe: int, caption: str,
                            candles, ma1, ma2, ma3, i_rej: int, i_con: int):
    tf_label = "5m" if timeframe == 300 else ("10m" if timeframe == 600 else "15m")
    title = f"{asset_name} | {tf_label}"
    png = _plot_candles(candles, ma1, ma2, ma3, i_rej, i_con, title)
    _send_photo_bytes(png, caption)

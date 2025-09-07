import os, json, time, websocket
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Rectangle

# Telegram helpers (your bot.py)
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    def send_telegram_message(token, chat_id, text): print("TEXT:", text)
    def send_telegram_photo(token, chat_id, caption, photo): print("PHOTO:", caption)

# Config
DERIV_API_KEY = os.getenv("DERIV_API_KEY","").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID","").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()
ASSETS = [s.strip() for s in os.getenv("ASSETS","R_50,R_75,V_75,V_100").split(",")]
TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES","300").split(",")]
CANDLES_N = 100
LAST_N_CHART = 80
PAD_CANDLES = 10
CANDLE_WIDTH = 0.4
NEAR_FACTOR = 0.30
TMPDIR = os.path.expanduser("~")
ALERT_FILE = os.path.join(TMPDIR,"dsr_last_sent_main.json")

# Persistence
def load_persist():
    try: return json.load(open(ALERT_FILE))
    except: return {}
def save_persist(d):
    try: json.dump(d, open(ALERT_FILE,"w"))
    except: pass
def already_sent(sym, tf, epoch, side):
    rec = load_persist().get(f"{sym}|{tf}")
    return bool(rec and rec.get("epoch")==epoch and rec.get("side")==side)
def mark_sent(sym, tf, epoch, side):
    d=load_persist(); d[f"{sym}|{tf}"]={"epoch":epoch,"side":side}; save_persist(d)

# MAs
def smma(series, period):
    n=len(series)
    if n<period: return [None]*n
    out=[None]*(period-1); out.append(sum(series[:period])/period)
    for i in range(period, n):
        prev=out[-1]; out.append((prev*(period-1)+series[i])/period)
    return out
def sma(series, period):
    n=len(series)
    if n<period: return [None]*n
    out=[None]*(period-1)
    for i in range(period, n+1):
        out.append(sum(series[i-period:i])/period)
    return out
def compute_mas(candles):
    closes=[c["close"] for c in candles]; hlc3=[(c["high"]+c["low"]+c["close"])/3 for c in candles]
    ma1=smma(hlc3,9); ma2=smma(closes,19); ma3_raw=sma(ma2,25)
    ma3=[ma3_raw[j] if j<len(ma3_raw) else None for j in range(len(ma2))]
    return ma1,ma2,ma3

# Candle families (rejections)
def detect(candles, tf, sym):
    n = len(candles)
    if n < 30:
        return None
    i = n - 1
    prev = candles[i - 1] if i > 0 else None
    c = candles[i]

    # compute MAs
    ma1, ma2, ma3 = compute_mas(candles)
    reasons = []
    side = None

    # classify current candle
    rej = candle_family(c, prev)

    # rule 1: Break MA3 + retest + rejection candlestick → reversal
    if rej != "NONE" and broke_ma3_recently(candles, ma3, i):
        if c["close"] > ma3[i]:
            side = "BUY"
            reasons.append(f"{rej} after MA3 breakout + retest")
        elif c["close"] < ma3[i]:
            side = "SELL"
            reasons.append(f"{rej} after MA3 breakout + retest")

    # rule 2: Retest MA1/MA2 in a clear trend
    if rej != "NONE":
        if in_uptrend(i, ma1, ma2, ma3, c["close"]):
            if near_ma(c, ma1[i]) or near_ma(c, ma2[i]):
                side = "BUY"
                reasons.append(f"{rej} retest MA1/MA2 in uptrend")
        if in_downtrend(i, ma1, ma2, ma3, c["close"]):
            if near_ma(c, ma1[i]) or near_ma(c, ma2[i]):
                side = "SELL"
                reasons.append(f"{rej} retest MA1/MA2 in downtrend")

    # rule 3: Small swings in clear trend + rejection
    if rej != "NONE":
        if in_uptrend(i, ma1, ma2, ma3, c["close"]):
            side = "BUY"
            reasons.append(f"{rej} small swing in uptrend")
        if in_downtrend(i, ma1, ma2, ma3, c["close"]):
            side = "SELL"
            reasons.append(f"{rej} small swing in downtrend")

    if not reasons or not side:
        return None

    return {
        "symbol": sym,
        "tf": tf,
        "side": side,
        "reasons": reasons,
        "epoch": c["epoch"],
        "ma1": ma1,
        "ma2": ma2,
        "ma3": ma3,
    }

# Trend helpers
def in_uptrend(i,m1,m2,m3,price): return m1[i] and m2[i] and m3[i] and m1[i]>m2[i]>m3[i] and price>m1[i]
def in_downtrend(i,m1,m2,m3,price): return m1[i] and m2[i] and m3[i] and m1[i]<m2[i]<m3[i] and price<m1[i]
def broke_ma3_recently(candles,ma3,idx,look=6):
    for k in range(max(0,idx-look),idx+1):
        if ma3[k] and ma3[k-1]:
            if candles[k]["close"]>ma3[k] and candles[k-1]["close"]<=ma3[k-1]: return True
            if candles[k]["close"]<ma3[k] and candles[k-1]["close"]>=ma3[k-1]: return True
    return False

# NEW Detection logic
def detect(candles, tf, sym):
    n = len(candles)
    if n < 30: return None
    i = n - 1
    prev = candles[i - 1] if i > 0 else None
    c = candles[i]

    ma1, ma2, ma3 = compute_mas(candles)
    reasons, side = [], None
    rej = candle_family(c, prev)

    # 1. Break MA3 + retest + rejection
    if rej != "NONE" and broke_ma3_recently(candles, ma3, i):
        if (abs(c["close"] - ma1[i]) < NEAR_FACTOR * (c["high"] - c["low"])) or \
           (abs(c["close"] - ma2[i]) < NEAR_FACTOR * (c["high"] - c["low"])):
            side = "BUY" if c["close"] > ma3[i] else "SELL"
            reasons.append(f"{rej} MA3 breakout+retest")

    # 2. Retest MA1/MA2 in clear trend
    if rej != "NONE":
        if in_uptrend(i, ma1, ma2, ma3, c["close"]):
            if abs(c["close"] - ma1[i]) < NEAR_FACTOR * (c["high"] - c["low"]):
                side = "BUY"; reasons.append(f"{rej} near MA1 in uptrend")
            if abs(c["close"] - ma2[i]) < NEAR_FACTOR * (c["high"] - c["low"]):
                side = "BUY"; reasons.append(f"{rej} near MA2 in uptrend")
        if in_downtrend(i, ma1, ma2, ma3, c["close"]):
            if abs(c["close"] - ma1[i]) < NEAR_FACTOR * (c["high"] - c["low"]):
                side = "SELL"; reasons.append(f"{rej} near MA1 in downtrend")
            if abs(c["close"] - ma2[i]) < NEAR_FACTOR * (c["high"] - c["low"]):
                side = "SELL"; reasons.append(f"{rej} near MA2 in downtrend")

    # 3. Tiny swings in trend + rejection
    if rej != "NONE":
        rng = c["high"] - c["low"]; body = abs(c["close"] - c["open"])
        if rng > 0 and body / rng < 0.25:
            if in_uptrend(i, ma1, ma2, ma3, c["close"]):
                side = "BUY"; reasons.append(f"{rej} tiny swing uptrend")
            elif in_downtrend(i, ma1, ma2, ma3, c["close"]):
                side = "SELL"; reasons.append(f"{rej} tiny swing downtrend")

    if not reasons or not side: return None
    return {"symbol":sym,"tf":tf,"side":side,"reasons":reasons,"epoch":c["epoch"],"ma1":ma1,"ma2":ma2,"ma3":ma3}

# Fetch
def fetch_candles(sym, tf, count=CANDLES_N):
    for _ in range(3):
        try:
            ws=websocket.create_connection(DERIV_WS_URL,timeout=18)
            ws.send(json.dumps({"authorize":DERIV_API_KEY})); ws.recv()
            ws.send(json.dumps({"ticks_history":sym,"style":"candles","granularity":tf,"count":count}))
            resp=json.loads(ws.recv()); ws.close()
            if "candles" in resp: return resp["candles"]
        except: time.sleep(1)
    return []

# Chart
def make_chart(candles,ma1,ma2,ma3,i,reasons,sym,tf):
    n=len(candles); start=max(0,n-LAST_N_CHART); ch=candles[start:]
    xs=[datetime.fromtimestamp(c["epoch"]).strftime("%H:%M") for c in ch]
    fig,ax=plt.subplots(figsize=(10,6))
    for j,c in enumerate(ch):
        o,h,l,cl=c["open"],c["high"],c["low"],c["close"]; col="g" if cl>=o else "r"
        ax.plot([j,j],[l,h],c=col,lw=0.6); ax.add_patch(Rectangle((j-CANDLE_WIDTH/2,min(o,cl)),CANDLE_WIDTH,abs(cl-o),facecolor=col,edgecolor=col))
    ax.plot(ma1[start:],label="MA1"); ax.plot(ma2[start:],label="MA2"); ax.plot(ma3[start:],label="MA3")
    if reasons: ax.scatter(len(ch)-1,ch[-1]["close"],marker="^" if "BUY" in reasons[0] else "v",color="orange",s=120)
    ax.legend(); plt.title(sym+" "+str(tf))
    fn=os.path.join(TMPDIR,f"{sym}_{tf}.png"); plt.savefig(fn); plt.close(); return fn

# Runner
def run_once():
    for sym in ASSETS:
        for tf in TIMEFRAMES:
            candles=fetch_candles(sym,tf)
            if len(candles)<60: continue
            res=detect(candles,tf,sym)
            if not res: continue
            ires=res; epoch=candles[-1]["epoch"]; side=ires["side"]
            if already_sent(sym,tf,epoch,side): continue
            caption=f"[{sym} {tf//60}m {side}] " + " | ".join(ires["reasons"]) + f" @ {candles[-1]['close']}"
            chart=make_chart(candles,ires["ma1"],ires["ma2"],ires["ma3"],len(candles)-1,ires["reasons"],sym,tf)
            send_telegram_photo(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,caption,chart)
            mark_sent(sym,tf,epoch,side)

if __name__=="__main__": run_once()

#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (strictly following user instructions)

- Detects any rejection candlestick from families: pinbars (any), dojis (any), bullish/bearish engulfings, tiny bodies
- Fires signals exactly as specified
- Uses shorthand -> Deriv symbol mapping
- Supports per-symbol timeframes (1s for (1s) assets)
- Deduplicates alerts, sends chart and Telegram message
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
import websocket, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Telegram helpers (fallback to print)
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    def send_telegram_message(token, chat_id, text): print("[TEXT]", text); return True, "local"
    def send_telegram_photo(token, chat_id, caption, photo): print("[PHOTO]", caption, photo); return True, "local"

# -------------------------
# Config
# -------------------------
DERIV_API_KEY = os.getenv("DERIV_API_KEY","").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID","1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()

TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES","300").split(",") if x.strip().isdigit()]
DEBUG = os.getenv("DEBUG","0") == "1"
TEST_MODE = os.getenv("TEST_MODE","0") == "1"

CANDLES_N = 480
LAST_N_CHART = 180
PAD_CANDLES = 10
CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")
MIN_CANDLES = 100
LOOKBACK_BROKE_MA3 = 12
NEAR_FACTOR = 0.30

# -------------------------
# Shorthand -> Deriv symbols
# -------------------------
SYMBOL_MAP = {
    "V10": "R_10",
    "V50": "R_50",
    "V75": "R_75",
    "V75(1s)": "1HZ75V",
    "V100(1s)": "1HZ100V",
    "V150(1s)": "1HZ150V"
}

SYMBOL_TF_MAP = {
    "V75(1s)": 1,
    "V100(1s)": 1,
    "V150(1s)": 1
}

# -------------------------
# Persistence
# -------------------------
def load_persist():
    try:
        return json.load(open(ALERT_FILE))
    except Exception:
        return {}
def save_persist(d):
    try:
        json.dump(d, open(ALERT_FILE,"w"))
    except Exception:
        pass
def already_sent(shorthand, tf, epoch, side):
    if TEST_MODE:
        return False
    rec = load_persist().get(f"{shorthand}|{tf}")
    return bool(rec and rec.get("epoch")==epoch and rec.get("side")==side)
def mark_sent(shorthand, tf, epoch, side):
    d=load_persist(); d[f"{shorthand}|{tf}"]={"epoch":epoch,"side":side}; save_persist(d)

# -------------------------
# Moving averages
# -------------------------
def smma(series, period):
    n=len(series)
    if n<period: return [None]*n
    out=[None]*(period-1)
    seed=sum(series[:period])/period
    out.append(seed)
    prev=seed
    for i in range(period,n):
        prev=(prev*(period-1)+series[i])/period
        out.append(prev)
    return out

def sma(series, period):
    n=len(series)
    if n<period: return [None]*n
    out=[None]*(period-1)
    run=sum(series[:period])
    out.append(run/period)
    for i in range(period,n):
        run += series[i]-series[i-period]
        out.append(run/period)
    return out

def compute_mas(candles):
    closes = [c["close"] for c in candles]
    hlc3 = [(c["high"]+c["low"]+c["close"])/3.0 for c in candles]
    ma1 = smma(hlc3,9)
    ma2 = smma(closes,19)
    ma2_vals = [v for v in ma2 if v is not None]
    ma3_raw = sma(ma2_vals,25) if len(ma2_vals)>=25 else []
    ma3=[]
    idx_raw=0
    for v in ma2:
        if v is None:
            ma3.append(None)
        else:
            ma3.append(ma3_raw[idx_raw] if idx_raw<len(ma3_raw) else None)
            idx_raw += 1
    return ma1, ma2, ma3

# -------------------------
# Candle family detection (any family, no numeric restrictions)
# -------------------------
def candle_family(c, prev):
    o,h,l,cl = c["open"], c["high"], c["low"], c["close"]
    body = abs(cl-o)
    
    # Doji: exact open==close
    if cl == o:
        return "DOJI"

    # Pinbar: any long wick relative to opposite side
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    if upper_wick > lower_wick and upper_wick > 0:
        return "PINBAR"
    if lower_wick > upper_wick and lower_wick > 0:
        return "PINBAR"

    # Engulfing
    if prev:
        po, pc = prev["open"], prev["close"]
        if pc < po and cl > o and o <= pc and cl >= po:
            return "BULL_ENG"
        if pc > po and cl < o and o >= pc and cl <= po:
            return "BEAR_ENG"

    # Tiny body: any small body (including dojis)
    if body == 0:
        return "TINY"

    return "NONE"

# -------------------------
# Trend helpers
# -------------------------
def in_uptrend(i,m1,m2,m3,price):
    try:
        return m1[i] and m2[i] and m3[i] and m1[i]>m2[i]>m3[i] and price>=m3[i]
    except: return False

def in_downtrend(i,m1,m2,m3,price):
    try:
        return m1[i] and m2[i] and m3[i] and m1[i]<m2[i]<m3[i] and price<=m3[i]
    except: return False

def broke_ma3_recently(candles, ma3, idx, look=LOOKBACK_BROKE_MA3):
    start = max(1, idx-look)
    for k in range(start, idx+1):
        if ma3[k] is None or ma3[k-1] is None: continue
        if candles[k-1]["close"]<=ma3[k-1] and candles[k]["close"]>ma3[k]:
            return "UP"
        if candles[k-1]["close"]>=ma3[k-1] and candles[k]["close"]<ma3[k]:
            return "DOWN"
    return None

def near(price, ma, rng):
    if ma is None: return False
    return abs(price-ma) <= max(1e-9, rng*NEAR_FACTOR)

# -------------------------
# Fetch candles
# -------------------------
def fetch_candles(sym, tf, count=CANDLES_N):
    for _ in range(3):
        try:
            ws=websocket.create_connection(DERIV_WS_URL,timeout=18)
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize":DERIV_API_KEY}))
                _ = ws.recv()
            ws.send(json.dumps({"ticks_history":sym,"style":"candles","granularity":tf,"count":count,"end":"latest"}))
            resp=json.loads(ws.recv()); ws.close()
            if "candles" in resp:
                return [{"epoch":int(c["epoch"]),"open":float(c["open"]),"high":float(c["high"]),
                         "low":float(c["low"]),"close":float(c["close"])} for c in resp["candles"]]
        except: time.sleep(1)
    return []

# -------------------------
# Charting
# -------------------------
def make_chart(candles, ma1, ma2, ma3, i, reasons, shorthand, tf):
    n=len(candles)
    start=max(0,n-LAST_N_CHART)
    ch=candles[start:]
    fig,ax=plt.subplots(figsize=(10,6))
    for j,c in enumerate(ch):
        o,h,l,cl=c["open"],c["high"],c["low"],c["close"]
        col="g" if cl>=o else "r"
        ax.plot([j,j],[l,h],c="k",lw=0.6)
        ax.add_patch(Rectangle((j-CANDLE_WIDTH/2,min(o,cl)),CANDLE_WIDTH,max(1e-9,abs(cl-o)),fc=col,ec="k",lw=0.3))
    def plot_ma(vals,label):
        arr=[vals[k] if k<len(vals) else None for k in range(start,n)]
        ax.plot(range(len(ch)), arr, lw=1, label=label)
    plot_ma(ma1,"MA1"); plot_ma(ma2,"MA2"); plot_ma(ma3,"MA3"); ax.legend()
    idx=i-start
    cl=ch[idx]["close"]
    side="SELL" if any("SELL" in r or "DOWN" in r for r in reasons) else "BUY"
    ax.scatter([idx],[cl],c=("red" if side=="SELL" else "green"),marker=("v" if side=="SELL" else "^"),s=120)
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png"); plt.savefig(tmp.name); plt.close(); return tmp.name

# -------------------------
# Detection
# -------------------------
def detect(candles,tf,shorthand):
    n=len(candles); i=n-1; prev=candles[i-1] if i>0 else None; c=candles[i]
    fam=candle_family(c,prev)
    if fam=="NONE": return None
    l,h,cl=c["low"],c["high"],c["close"]; rng=h-l or 1e-9
    ma1,ma2,ma3=compute_mas(candles); reasons=[]; side=None
    if near(l,ma1[i],rng) or near(l,ma2[i],rng): reasons.append(f"{fam} near MA1/MA2"); side="BUY"
    if near(h,ma1[i],rng) or near(h,ma2[i],rng): reasons.append(f"{fam} near MA1/MA2"); side="SELL"
    cross=broke_ma3_recently(candles,ma3,i)
    if cross=="UP" and (near(l,ma1[i],rng) or near(l,ma2[i],rng)): reasons.append("MA3 breakout+retest"); side="BUY"
    if cross=="DOWN" and (near(h,ma1[i],rng) or near(h,ma2[i],rng)): reasons.append("MA3 breakout+retest"); side="SELL"
    if in_uptrend(i,ma1,ma2,ma3,cl) and (near(l,ma1[i],rng) or near(l,ma2[i],rng)): reasons.append("Continuation up"); side="BUY"
    if in_downtrend(i,ma1,ma2,ma3,cl) and (near(h,ma1[i],rng) or near(h,ma2[i],rng)): reasons.append("Continuation down"); side="SELL"
    if side: return {"symbol":shorthand,"tf":tf,"side":side,"reasons":reasons,"idx":i,"ma1":ma1,"ma2":ma2,"ma3":ma3,"candles":candles}
    return None

# -------------------------
# Runner
# -------------------------
def get_tf(shorthand):
    return SYMBOL_TF_MAP.get(shorthand,TIMEFRAMES[0])

def run_once():
    for shorthand, deriv_sym in SYMBOL_MAP.items():
        tf=get_tf(shorthand)
        candles=fetch_candles(deriv_sym,tf)
        if len(candles)<MIN_CANDLES: continue
        res=detect(candles,tf,shorthand)
        if not res: continue
        i=res["idx"]; epoch=candles[i]["epoch"]; side=res["side"]
        if already_sent(shorthand,tf,epoch,side): continue
        caption=f"[{shorthand} {tf//60 if tf>=60 else tf}s {side}] {' | '.join(res['reasons'])} @ {candles[i]['close']}"
        chart=make_chart(res["candles"],res["ma1"],res["ma2"],res["ma3"],i,res["reasons"],shorthand,tf)
        send_telegram_photo(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,caption,chart)
        mark_sent(shorthand,tf,epoch,side)

if __name__=="__main__":
    try:
        run_once()
    except Exception as e:
        traceback.print_exc()

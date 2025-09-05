#!/usr/bin/env python3
# main.py
# Default indices bot with charts.
# Signal logic: trend continuation (MA1/2 rejection) + reversal (MA3 breakout + retest).
# Sends charts to Telegram only on true setups, no spam updates.

import os, json, time, math, tempfile, traceback
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import websocket, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Config
# =========================
DEBUG = os.getenv("DEBUG", "0") in ("1","true","True","yes")

DERIV_API_KEY = os.getenv("DERIV_API_KEY","").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID","1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()

# Default indices (not 1s)
ASSETS = os.getenv("ASSETS","R_50,R_75,R_100,V_50,V_75,V_100").split(",")
TIMEFRAMES = [300,600,900]  # 5m, 10m, 15m

CANDLES_N = int(os.getenv("CANDLES_N","400"))
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS","600"))
MAX_SIGNALS_PER_RUN = int(os.getenv("MAX_SIGNALS_PER_RUN","5"))

TMPDIR = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR,"fenyere_last_sent.json")

# =========================
# Telegram
# =========================
from bot import send_telegram_message, send_telegram_photo

# =========================
# Cooldown persistence
# =========================
last_sent_cache: Dict[str, Dict] = {}

def load_cache():
    global last_sent_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE,"r") as f: last_sent_cache=json.load(f)
    except: last_sent_cache={}

def save_cache():
    try:
        with open(CACHE_FILE,"w") as f: json.dump(last_sent_cache,f)
    except: pass

def can_send(symbol,direction,tf):
    rec=last_sent_cache.get(symbol)
    now=int(time.time())
    if rec and rec.get("direction")==direction and (now-rec.get("ts",0))<ALERT_COOLDOWN_SECS:
        return False
    return True

def mark_sent(symbol,direction,tf):
    last_sent_cache[symbol]={"direction":direction,"tf":tf,"ts":int(time.time())}
    save_cache()

# =========================
# Moving Averages
# =========================
def smma(series,period):
    n=len(series)
    if n<period: return [None]*n
    seed=sum(series[:period])/period
    out=[None]*(period-1)+[seed]
    prev=seed
    for i in range(period,n):
        prev=(prev*(period-1)+series[i])/period
        out.append(prev)
    return out

def sma(series,period):
    n=len(series)
    if n<period: return [None]*n
    out=[None]*(period-1)
    for i in range(period-1,n):
        out.append(sum(series[i-period+1:i+1])/period)
    return out

def compute_mas(candles):
    closes=[c["close"] for c in candles]
    highs=[c["high"] for c in candles]
    lows=[c["low"] for c in candles]
    hlc3=[(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
    ma1=smma(hlc3,9)
    ma2=smma(closes,19)
    ma2_vals=[v for v in ma2 if v is not None]
    ma3_raw=sma(ma2_vals,25) if len(ma2_vals)>=25 else [None]*len(ma2_vals)
    ma3=[]
    j=0
    for v in ma2:
        if v is None: ma3.append(None)
        else: ma3.append(ma3_raw[j] if j<len(ma3_raw) else None); j+=1
    return ma1,ma2,ma3

# =========================
# Fetch candles
# =========================
def fetch_candles(symbol,granularity,count=CANDLES_N):
    for attempt in range(3):
        ws=None
        try:
            ws=websocket.create_connection(DERIV_WS_URL,timeout=18)
            ws.send(json.dumps({"authorize":DERIV_API_KEY})); _=ws.recv()
            req={"ticks_history":symbol,"style":"candles","granularity":granularity,
                 "count":count,"end":"latest","subscribe":0}
            ws.send(json.dumps(req)); resp=json.loads(ws.recv())
            if "candles" in resp:
                candles=[{"epoch":int(c["epoch"]),"open":float(c["open"]),
                          "high":float(c["high"]),"low":float(c["low"]),
                          "close":float(c["close"])} for c in resp["candles"]]
                candles.sort(key=lambda x:x["epoch"])
                return candles
        except Exception as e:
            if DEBUG: print("fetch error:",e)
        finally:
            if ws:
                try: ws.close()
                except: pass
        time.sleep(1)
    return []

# =========================
# Helpers
# =========================
def candle_bits(c,prev=None):
    o,h,l,cl=c["open"],c["high"],c["low"],c["close"]
    body=abs(cl-o); rng=max(1e-12,h-l)
    upper=h-max(o,cl); lower=min(o,cl)-l
    is_doji=body<=0.35*rng
    pin_low=(lower>=0.2*rng and lower>upper)
    pin_high=(upper>=0.2*rng and upper>lower)
    engulf_bull=engulf_bear=False
    if prev:
        po,pc=prev["open"],prev["close"]
        if pc<po and cl>o and o<=pc and cl>=po: engulf_bull=True
        if pc>po and cl<o and o>=pc and cl<=po: engulf_bear=True
    return {"body":body,"is_doji":is_doji,"pin_low":pin_low,"pin_high":pin_high,
            "engulf_bull":engulf_bull,"engulf_bear":engulf_bear}

def compute_atr(candles,n=14):
    rngs=[c["high"]-c["low"] for c in candles]
    return float(np.mean(rngs[-n:])) if len(rngs)>=n else float(np.mean(rngs))

def is_rejection(c,prev,atr,direction):
    b=candle_bits(c,prev)
    tiny=b["body"]<=0.25*atr
    if tiny: return True
    if direction=="BUY":
        return b["pin_low"] or b["engulf_bull"] or (b["is_doji"] and c["close"]>=c["open"])
    else:
        return b["pin_high"] or b["engulf_bear"] or (b["is_doji"] and c["close"]<=c["open"])

# =========================
# Charting
# =========================
def make_chart(candles,ma1,ma2,ma3,idx,reason,symbol,tf):
    xs=[datetime.utcfromtimestamp(c["epoch"]) for c in candles]
    opens=[c["open"] for c in candles]
    highs=[c["high"] for c in candles]
    lows =[c["low"] for c in candles]
    closes=[c["close"] for c in candles]
    fig,ax=plt.subplots(figsize=(12,4),dpi=110)
    ax.set_title(f"{symbol} | {tf//60}m | {reason}",fontsize=10)

    for i,(o,h,l,cl) in enumerate(zip(opens,highs,lows,closes)):
        color="#2ca02c" if cl>=o else "#d62728"
        ax.plot([i,i],[l,h],color="black",lw=0.6)
        rect=Rectangle((i-0.35/2,min(o,cl)),0.35,max(1e-9,abs(cl-o)),
                       facecolor=color,edgecolor="black",lw=0.35)
        ax.add_patch(rect)
    ax.plot(ma1,label="MA1"); ax.plot(ma2,label="MA2"); ax.plot(ma3,label="MA3")
    ax.legend()

    # Mark rejection candle
    ax.scatter([idx],[closes[idx]],color="blue",s=120,zorder=5)
    ax.text(idx,closes[idx],reason,fontsize=8,color="blue")

    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png")
    fig.tight_layout(); fig.savefig(tmp.name,dpi=120); plt.close(fig)
    return tmp.name

# =========================
# Signal Evaluation
# =========================
def evaluate(symbol,candles,tf):
    out=[]; n=len(candles)
    if n<30: return out
    ma1,ma2,ma3=compute_mas(candles)
    atr=compute_atr(candles)
    last,prev=candles[-2],candles[-3]

    def trend_up():
        return ma1[-1] and ma1[-1]>ma2[-1]>ma3[-1] and ma1[-1]>ma1[-4]
    def trend_down():
        return ma1[-1] and ma1[-1]<ma2[-1]<ma3[-1] and ma1[-1]<ma1[-4]

    # continuation
    if trend_up():
        if min(abs(last["close"]-ma1[-1]),abs(last["close"]-ma2[-1]))<=0.5*atr:
            if is_rejection(last,prev,atr,"BUY"):
                out.append(("BUY","Continuation: MA1/2 support rejection",last,ma1,ma2,ma3))
    if trend_down():
        if min(abs(last["close"]-ma1[-1]),abs(last["close"]-ma2[-1]))<=0.5*atr:
            if is_rejection(last,prev,atr,"SELL"):
                out.append(("SELL","Continuation: MA1/2 resistance rejection",last,ma1,ma2,ma3))

    # reversal
    if ma3[-3] and ma3[-2]:
        if candles[-3]["close"]<ma3[-3] and candles[-2]["close"]>ma3[-2]:
            if min(abs(last["close"]-ma1[-1]),abs(last["close"]-ma2[-1]))<=0.6*atr:
                if is_rejection(last,prev,atr,"BUY"):
                    out.append(("BUY","Reversal: MA3 breakout + retest",last,ma1,ma2,ma3))
        if candles[-3]["close"]>ma3[-3] and candles[-2]["close"]<ma3[-2]:
            if min(abs(last["close"]-ma1[-1]),abs(last["close"]-ma2[-1]))<=0.6*atr:
                if is_rejection(last,prev,atr,"SELL"):
                    out.append(("SELL","Reversal: MA3 breakout + retest",last,ma1,ma2,ma3))
    return out

# =========================
# Runner
# =========================
def run_once():
    load_cache(); signals_sent=0
    for symbol in ASSETS:
        for tf in TIMEFRAMES:
            candles=fetch_candles(symbol,tf)
            if not candles: continue
            candidates=evaluate(symbol,candles,tf)
            for direction,reason,last,ma1,ma2,ma3 in candidates:
                if signals_sent>=MAX_SIGNALS_PER_RUN: break
                if not can_send(symbol,direction,tf): continue
                chart=make_chart(candles,ma1,ma2,ma3,len(candles)-2,reason,symbol,tf)
                text=f"🔔 {symbol} | {tf//60}m | {direction}\nReason: {reason}\nPrice: {last['close']:.2f}"
                send_telegram_photo(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,text,chart)
                mark_sent(symbol,direction,tf); signals_sent+=1
    return signals_sent

if __name__=="__main__":
    try:
        run_once()
    except Exception as e:
        if DEBUG: traceback.print_exc()
        send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,f"main.py error: {e}")

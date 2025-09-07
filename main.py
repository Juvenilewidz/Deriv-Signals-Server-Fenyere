#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (final strategy implementation)

Rules:
- MA1 = SMMA(HLC3, 9)
- MA2 = SMMA(Close, 19)
- MA3 = SMA(MA2, 25)

Valid rejection candles:
- Pinbars (any type)
- Doji (any type)
- Engulfing (bullish/bearish)
- Tiny-body

Signal fires immediately at candle close if ANY of the 3 conditions holds:
1. Reversal: Price breaks MA3, then retests MA1/MA2 with rejection candlestick.
2. Continuation: Market trending (MA1>MA2>MA3 or reverse), price retests MA1/MA2 with rejection candlestick.
3. Imperfect Retest: Market trending, rejection candlestick forms near MA1/MA2 (looser distance, small pullback away).
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import websocket, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Telegram helpers
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    def send_telegram_message(token, chat_id, text): print("[TEXT]", text); return True, "local"
    def send_telegram_photo(token, chat_id, caption, photo): print("[PHOTO]", caption, photo); return True, "local"

# Config
DERIV_API_KEY = os.getenv("DERIV_API_KEY","").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID","1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()

ASSETS = [s.strip() for s in os.getenv("ASSETS","V10,V50,V75,V75(1s),V100(1s),V150(1s)").split(",") if s.strip()]
TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES","300").split(",")]  # default 5m

CANDLES_N = 180; LAST_N_CHART = 180; CANDLE_WIDTH = 0.35
NEAR_FACTOR = float(os.getenv("NEAR_FACTOR","0.30"))   # strict near
LOOSE_FACTOR = float(os.getenv("LOOSE_FACTOR","0.60")) # imperfect retests
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

# ---------------- Persistence ----------------
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

# ---------------- Moving averages ----------------
def smma(series, period):
    n=len(series); 
    if n<period: return [None]*n
    seed=sum(series[:period])/period; out=[None]*(period-1)+[seed]; prev=seed
    for i in range(period,n): prev=(prev*(period-1)+series[i])/period; out.append(prev)
    return out
def sma(series, period):
    n=len(series); 
    if n<period: return [None]*n
    out=[None]*(period-1); run=sum(series[:period]); out.append(run/period)
    for i in range(period,n): run+=series[i]-series[i-period]; out.append(run/period)
    return out
def compute_mas(candles):
    closes=[c["close"] for c in candles]; hlc3=[(c["high"]+c["low"]+c["close"])/3 for c in candles]
    ma1=smma(hlc3,9); ma2=smma(closes,19)
    ma2_vals=[v for v in ma2 if v is not None]; ma3_raw=sma(ma2_vals,25) if len(ma2_vals)>=25 else []
    ma3=[]; j=0
    for v in ma2:
        if v is not None and j<len(ma3_raw): ma3.append(ma3_raw[j]); j+=1
        else: ma3.append(None)
    return ma1,ma2,ma3

# ---------------- Candle families ----------------
def candle_family(c, prev):
    o,h,l,cl=c["open"],c["high"],c["low"],c["close"]; body=abs(cl-o); rng=max(1e-9,h-l)
    up=h-max(o,cl); lo=min(o,cl)-l
    if body<=0.15*rng: return "DOJI"
    if up>=0.55*rng and body<=0.45*rng: return "PIN_HIGH"
    if lo>=0.55*rng and body<=0.45*rng: return "PIN_LOW"
    if prev:
        po,pc=prev["open"],prev["close"]
        if pc<po and cl>o and o<=pc and cl>=po: return "BULL_ENG"
        if pc>po and cl<o and o>=pc and cl<=po: return "BEAR_ENG"
    if body<=0.08*rng: return "TINY"
    return "NONE"
def is_rejection(fam): return fam in ("PIN_HIGH","PIN_LOW","DOJI","BULL_ENG","BEAR_ENG","TINY")

# ---------------- Trend helpers ----------------
def in_uptrend(i,m1,m2,m3,price): return m1[i] and m2[i] and m3[i] and m1[i]>m2[i]>m3[i] and price>=m3[i]
def in_downtrend(i,m1,m2,m3,price): return m1[i] and m2[i] and m3[i] and m1[i]<m2[i]<m3[i] and price<=m3[i]
def broke_ma3_recently(candles, ma3, idx, look=6):
    for k in range(max(1,idx-look),idx+1):
        if ma3[k] and ma3[k-1]:
            if candles[k-1]["close"]<=ma3[k-1] and candles[k]["close"]>ma3[k]: return "UP"
            if candles[k-1]["close"]>=ma3[k-1] and candles[k]["close"]<ma3[k]: return "DOWN"
    return None

# ---------------- Near logic ----------------
def near_ma(candle, m, rng, factor): 
    if m is None: return False
    l,h,cl=c["low"],c["high"],c["close"]; dist=min(abs(l-m),abs(h-m),abs(cl-m))
    return dist <= factor * rng

# ---------------- Fetch ----------------
def fetch_candles(sym, tf, count=CANDLES_N):
    for _ in range(3):
        try:
            ws=websocket.create_connection(DERIV_WS_URL,timeout=18)
            ws.send(json.dumps({"authorize":DERIV_API_KEY})); ws.recv()
            ws.send(json.dumps({"ticks_history":sym,"style":"candles","granularity":tf,"count":count,"end":"latest"}))
            resp=json.loads(ws.recv()); ws.close()
            if "candles" in resp:
                return [{"epoch":int(c["epoch"]),"open":float(c["open"]),"high":float(c["high"]),
                         "low":float(c["low"]),"close":float(c["close"])} for c in resp["candles"]]
        except: time.sleep(1)
    return []

# ---------------- Chart ----------------
def make_chart(candles,ma1,ma2,ma3,i,reasons,sym,tf):
    n=len(candles); start=max(0,n-LAST_N_CHART); ch=candles[start:]
    fig,ax=plt.subplots(figsize=(10,6))
    for j,c in enumerate(ch):
        o,h,l,cl=c["open"],c["high"],c["low"],c["close"]; col="g" if cl>=o else "r"
        ax.plot([j,j],[l,h],c="k",lw=0.6)
        ax.add_patch(Rectangle((j-CANDLE_WIDTH/2,min(o,cl)),CANDLE_WIDTH,max(1e-9,abs(cl-o)),fc=col,ec="k",lw=0.3))
    def plot_ma(vals,label,col): ax.plot(range(len(ch)),[vals[k] if k<len(vals) else None for k in range(start,n)],c=col,lw=1,label=label)
    plot_ma(ma1,"MA1", "b"); plot_ma(ma2,"MA2","orange"); plot_ma(ma3,"MA3","red"); ax.legend()
    idx=i-start; cl=ch[idx]["close"]
    side="SELL" if any("SELL" in r or "DOWN" in r for r in reasons) else "BUY"
    ax.scatter([idx],[cl],c=("red" if side=="SELL" else "green"),marker=("v" if side=="SELL" else "^"),s=120)
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".png"); plt.savefig(tmp.name); plt.close(); return tmp.name

# ---------------- Detection ----------------
def detect(candles,tf,sym):
    n=len(candles); i=n-1; prev=candles[i-1] if i>0 else None; c=candles[i]
    fam=candle_family(c,prev); 
    if not is_rejection(fam): return None
    l,h,cl=c["low"],c["high"],c["close"]; rng=max(1e-9,h-l)
    ma1,ma2,ma3=compute_mas(candles); reasons=[]; side=None

    # Condition 1: Reversal (MA3 breakout + retest at/near MA1/MA2 with rejection)
    cross=broke_ma3_recently(candles,ma3,i)
    if cross=="UP" and (near_ma(c,ma1[i],rng,NEAR_FACTOR) or near_ma(c,ma2[i],rng,NEAR_FACTOR)):
        reasons.append("Reversal: MA3 breakout + retest with rejection"); side="BUY"
    elif cross=="DOWN" and (near_ma(c,ma1[i],rng,NEAR_FACTOR) or near_ma(c,ma2[i],rng,NEAR_FACTOR)):
        reasons.append("Reversal: MA3 breakout + retest with rejection"); side="SELL"

    # Condition 2: Continuation (trend + strict retest at/near MA1/MA2)
    if side is None:
        if in_uptrend(i,ma1,ma2,ma3,cl) and (near_ma(c,ma1[i],rng,NEAR_FACTOR) or near_ma(c,ma2[i],rng,NEAR_FACTOR)):
            reasons.append("Continuation: Trend retest MA1/MA2 with rejection"); side="BUY"
        elif in_downtrend(i,ma1,ma2,ma3,cl) and (near_ma(c,ma1[i],rng,NEAR_FACTOR) or near_ma(c,ma2[i],rng,NEAR_FACTOR)):
            reasons.append("Continuation: Trend retest MA1/MA2 with rejection"); side="SELL"

    # Condition 3: Imperfect retest (trend + rejection near but not exactly at MAs, looser distance)
    if side is None:
        if in_uptrend(i,ma1,ma2,ma3,cl) and (near_ma(c,ma1[i],rng,LOOSE_FACTOR) or near_ma(c,ma2[i],rng,LOOSE_FACTOR)):
            reasons.append("Continuation: Imperfect retest (tiny swing near MAs)"); side="BUY"
        elif in_downtrend(i,ma1,ma2,ma3,cl) and (near_ma(c,ma1[i],rng,LOOSE_FACTOR) or near_ma(c,ma2[i],rng,LOOSE_FACTOR)):
            reasons.append("Continuation: Imperfect retest (tiny swing near MAs)"); side="SELL"

    if side: return {"symbol":sym,"tf":tf,"side":side,"reasons":reasons,"idx":i,"ma1":ma1,"ma2":ma2,"ma3":ma3,"candles":candles}
    return None

# ---------------- Runner ----------------
def run_once():
    for sym in ASSETS:
        for tf in TIMEFRAMES:
            try:
                candles=fetch_candles(sym,tf)
                if len(candles)<60: continue
                res=detect(candles,tf,sym)
                if not res: continue
                i=res["idx"]; epoch=candles[i]["epoch"]; side=res["side"]
                if already_sent(sym,tf,epoch,side): continue
                caption=f"[{sym} {tf//60}m {side}] {' | '.join(res['reasons'])} @ {candles[i]['close']}"
                chart=make_chart(res["candles"],res["ma1"],res["ma2"],res["ma3"],i,res["reasons"],sym,tf)
                send_telegram_photo(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,caption,chart)
                mark_sent(sym,tf,epoch,side)
            except Exception: traceback.print_exc(); time.sleep(0.5)

if __name__=="__main__": run_once()

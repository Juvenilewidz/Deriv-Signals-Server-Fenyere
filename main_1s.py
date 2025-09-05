# main_1s.py
"""
1s Indices Bot (text-only alerts, no charts).
- Monitors 1HZ75V, 1HZ100V, 1HZ150V at 5m timeframe.
- Sends BUY/SELL alerts to Telegram when conditions are met.
- Paste-and-cruise: works even without bot.py (has fallback sender).
"""

import os
import json
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

import websocket
import numpy as np

# ---------------------------
# Telegram Support (with fallback)
# ---------------------------
try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

def send_telegram_message(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        print("[TELEGRAM] Missing token/chat_id, message:", text)
        return False, "missing creds"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        return resp.ok, resp.text
    except Exception as e:
        print("[TELEGRAM ERROR]", e)
        return False, str(e)

# ---------------------------
# Config
# ---------------------------
DEBUG = True

DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID  = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL  = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ASSETS = os.getenv("ASSETS", "1HZ75V,1HZ100V,1HZ150V").split(",")
ASSETS = [a.strip() for a in ASSETS if a.strip()]

TIMEFRAME = 300   # 5 minutes
CANDLES_N = int(os.getenv("CANDLES_N", "100"))

TMPDIR     = tempfile.gettempdir()
CACHE_FILE = os.path.join(TMPDIR, "fenyere_last_sent_1s.json")

last_sent_cache: Dict[str, Dict] = {}

def log(*a): 
    if DEBUG: 
        print("[", datetime.utcnow().strftime("%H:%M:%S"), "]", *a)

# ---------------------------
# Cache (for cooldown)
# ---------------------------
def load_cache():
    global last_sent_cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                last_sent_cache = json.load(f)
    except: 
        pass

def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(last_sent_cache, f)
    except: 
        pass

def can_send(symbol, direction):
    rec = last_sent_cache.get(symbol)
    now = int(time.time())
    if rec and rec["direction"] == direction and now - rec["ts"] < 600:  # 10 min cooldown
        return False
    return True

def mark_sent(symbol, direction):
    last_sent_cache[symbol] = {"direction": direction, "ts": int(time.time())}
    save_cache()

# ---------------------------
# Indicators
# ---------------------------
def smma(vals: List[float], p: int):
    if len(vals) < p: return [None]*len(vals)
    s = sum(vals[:p])/p
    out = [None]*(p-1)+[s]
    prev=s
    for v in vals[p:]:
        prev=(prev*(p-1)+v)/p
        out.append(prev)
    return out

def sma(vals: List[float], p: int):
    if len(vals) < p: return [None]*len(vals)
    out=[None]*(p-1)
    for i in range(p-1, len(vals)):
        out.append(sum(vals[i-p+1:i+1])/p)
    return out

def candle_bits(c, prev=None):
    o,h,l,cx=float(c["open"]),float(c["high"]),float(c["low"]),float(c["close"])
    rng=max(1e-9,h-l); body=abs(cx-o)
    return {"o":o,"h":h,"l":l,"c":cx,
            "is_doji": body<=0.35*rng,
            "pin_low": l<o and l<cx,
            "pin_high": h>o and h>cx}

def analyze(symbol, candles):
    if len(candles)<30: 
        return None
    closes=[c["close"] for c in candles]; highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]
    hlc3=[(h+l+c)/3 for h,l,c in zip(highs,lows,closes)]
    ma1=smma(hlc3,9); ma2=smma(closes,19)
    ma2_vals=[v for v in ma2 if v]; ma3=sma(ma2_vals,25) if len(ma2_vals)>=25 else []
    # align ma3 with ma2
    ma3_full=[]; j=0
    for v in ma2:
        if v is None: ma3_full.append(None)
        else: ma3_full.append(ma3[j] if j<len(ma3) else None); j+=1
    i=-2
    if ma1[i] and ma2[i] and ma3_full[i]:
        bits=candle_bits(candles[i],candles[i-1])
        if ma1[i]>ma2[i]>ma3_full[i] and (bits["is_doji"] or bits["pin_low"]):
            return "BUY","MA support rejection"
        if ma1[i]<ma2[i]<ma3_full[i] and (bits["is_doji"] or bits["pin_high"]):
            return "SELL","MA resistance rejection"
    return None

# ---------------------------
# Candle fetch
# ---------------------------
def fetch(symbol, gran=TIMEFRAME, count=CANDLES_N):
    try:
        ws=websocket.create_connection(DERIV_WS_URL,timeout=15)
        ws.send(json.dumps({"authorize":DERIV_API_KEY})); ws.recv()
        req={"ticks_history":symbol,"style":"candles","granularity":gran,"count":count,"end":"latest"}
        ws.send(json.dumps(req))
        data=json.loads(ws.recv()); ws.close()
        if "candles" in data:
            return [{"epoch":int(c["epoch"]),
                     "open":float(c["open"]),
                     "high":float(c["high"]),
                     "low":float(c["low"]),
                     "close":float(c["close"])} for c in data["candles"]]
    except Exception as e:
        log("fetch error",e)
    return []

# ---------------------------
# Main loop
# ---------------------------
def run():
    load_cache()
    for sym in ASSETS:
        candles=fetch(sym)
        if not candles:
            send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,f"⚠️ {sym} no candles")
            continue
        res=analyze(sym,candles)
        if res:
            d,r=res
            if can_send(sym,d):
                msg=f"🔔 {sym} | 5m | {d}\nReason: {r}\nPrice: {candles[-2]['close']}"
                send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,msg)
                mark_sent(sym,d)

#if __name__=="__main__":
   # send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,"✅ 1s bot (text-only) started")
 #   run()

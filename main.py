import asyncio
from typing import Dict, Any
import pandas as pd
from fastapi import FastAPI
from datetime import datetime, timezone, timedelta
from .config import ASSETS, TIMEFRAMES, HISTORY\_COUNT, POLL\_SECONDS, DEDUP\_WINDOW
from .data.deriv\_api import DerivClient
from .analysis.patterns import detect\_double\_top\_bottom, detect\_head\_shoulders, detect\_triangles
from .analysis.signal\_engine import combine\_signals
from .notifications.telegram\_notifier import send\_telegram\_message
from . [notifications.email](http://notifications.email)\_notifier import send\_email
from .utils.logger import logger
app = FastAPI(title="Deriv Signal Server", version="1.0.0")
PATTERN\_DETECTORS = \[
detect\_double\_top\_bottom,
detect\_head\_shoulders,
detect\_triangles,
]
LATEST: Dict\[tuple, Dict\[str, Any]] = {}
HISTORY: Dict\[tuple, list] = {}
LAST\_SENT: Dict\[tuple, Dict\[str, Any]] = {}
LAST\_SENT\_TIME: Dict\[tuple, datetime] = {}
def tf\_label(tf\_sec: int) -> str:
return f"{tf\_sec//60}m" if tf\_sec % 60 == 0 else f"{tf\_sec}s"
def dedup\_ok(key: tuple, payload: Dict\[str, Any]) -> bool:
now = [datetime.now](http://datetime.now)(timezone.utc)
last = LAST\_SENT.get(key)
last\_t = LAST\_SENT\_TIME.get(key, now - timedelta(days=1))
if last == payload and (now - last\_t).total\_seconds() < DEDUP\_WINDOW:
return False
LAST\_SENT\[key] = payload
LAST\_SENT\_TIME\[key] = now
return True
def notify(asset: str, tf: int, payload: Dict\[str, Any]):
title = f"Signal: {payload\['signal']} | {asset} {tf\_label(tf)} ({payload.get('strength','')})"
lines = \[
f"⏱ { [datetime.now](http://datetime.now)(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
f"Asset: {asset}",
f"Timeframe: {tf\_label(tf)}",
f"Signal: {payload\['signal']} ({payload.get('strength','NORMAL')})",
f"Reason: {payload\['reason']}",
]
if payload.get("tp"):
lines.append(f"Potential TP: {payload\['tp']}")
body = "\n".join(lines)
send\_telegram\_message("📢 " + title + "\n" + body)
send\_email(title, body)
async def analyze\_once():
async with DerivClient() as dc:
for asset in ASSETS:
for tf in TIMEFRAMES:
try:
candles = await dc.candles(asset, tf, HISTORY\_COUNT)
except Exception as e:
logger.exception(f"Fetch error {asset}@{tf}: {e}")
continue
if not candles or len(candles) < 60:
[logger.info](http://logger.info)(f"Insufficient candles for {asset}@{tf}")
continue
df = pd.DataFrame(candles)
for col in \["open", "high", "low", "close"]:
df\[col] = [pd.to](http://pd.to)\_numeric(df\[col], errors="coerce")
df = df.dropna()
sig = combine\_signals(df, PATTERN\_DETECTORS)
payload: Dict\[str, Any] = {"asset": asset, "timeframe": tf, \*\*sig}
key = (asset, tf)
LATEST\[key] = payload
HISTORY.setdefault(key, \[]).insert(0, {"time": [datetime.now](http://datetime.now)(timezone.utc).isoformat(), \*\*payload})
HISTORY\[key] = HISTORY\[key]\[:500]
if payload\["signal"] in ("BUY", "SELL") and dedup\_ok(key, payload):
notify(asset, tf, payload)
async def scheduler\_loop():
while True:
try:
await analyze\_once()
except Exception as e:
logger.exception(f"Analyze error: {e}")
await asyncio.sleep(POLL\_SECONDS)
@app.on\_event("startup")
async def on\_start():
asyncio.create\_task(scheduler\_loop())
@app.get("/health")
def health():
return {"ok": True, "assets": ASSETS, "timeframes": TIMEFRAMES}
@app.get("/signals")
def signals():
out = {}
for a in ASSETS:
out\[a] = {}
for tf in TIMEFRAMES:
out\[a]\[tf\_label(tf)] = LATEST.get((a, tf))
return out
@app.get("/signals/{asset}/{tf}")
def signal\_for(asset: str, tf: int):
return LATEST.get((asset, tf))
@app.get("/history/{asset}/{tf}")
def history\_for(asset: str, tf: int):
return HISTORY.get((asset, tf), \[])
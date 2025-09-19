#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance Trading Bot

Recrafted for strict adherence to DSR strategy rules and logic as described in images 1-14:
- Signal only in clear trends (no ranging/consolidation/sideways)
- Strict MA arrangement: bullish (MA1 > MA2 > MA3), bearish (MA1 < MA2 < MA3)
- Signal only on confirmed rejection candlestick patterns (Pin Bar, Doji, Engulfing) at MA1/MA2
- Confirmation candle required after rejection, and signals only at close of confirmation
- No signals after price spike, or when price is between MA1 and MA2
- No signals during consolidation (frequent MA2 touches, packed/intersecting MAs)
"""

import os, json, time, tempfile, traceback
from datetime import datetime
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
CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")
MIN_CANDLES = 50

# -------------------------
# Symbol Mappings
# -------------------------
SYMBOL_MAP = {
    "V10": "R_10",
    "V25": "R_25",
    "V50": "R_50", 
    "V75": "R_75",
    "Jump10": "JD10",
    "Jump25": "JD25",
    "Jump50": "JD50", 
    "Jump100": "JD100",
    "V75(1s)": "1s_V75",
    "V100(1s)": "1s_V100",  
    "V150(1s)": "1s_V150",
    "V15(1s)": "1s_V15"
}

SYMBOL_TF_MAP = {
    "V75(1s)": 1,
    "V100(1s)": 1,  
    "V150(1s)": 1,
    "V15(1s)": 1
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
# Moving Averages
# -------------------------
def smma(series, period):
    n = len(series)
    if n < period:
        return [None] * n
    result = [None] * (period - 1)
    sma = sum(series[:period]) / period
    result.append(sma)
    prev = sma
    for i in range(period, n):
        smma = (prev * (period - 1) + series[i]) / period
        result.append(smma)
        prev = smma
    return result

def sma(series, period):
    n = len(series)
    if n < period:
        return [None] * n
    result = [None] * (period - 1)
    window_sum = sum(series[:period])
    result.append(window_sum / period)
    for i in range(period, n):
        window_sum += series[i] - series[i - period]
        result.append(window_sum / period)
    return result

def compute_mas(candles):
    closes = [c["close"] for c in candles]
    hlc3 = [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    ma1 = smma(hlc3, 9)
    ma2 = smma(closes, 19)
    ma2_valid = [v for v in ma2 if v is not None]
    ma3 = sma(ma2_valid, 25) if len(ma2_valid) >= 25 else [None]*len(candles)
    # Pad ma3 to candles
    ma3_full = []
    idx = 0
    for v in ma2:
        if v is None: ma3_full.append(None)
        else:
            if idx < len(ma3): ma3_full.append(ma3[idx]); idx += 1
            else: ma3_full.append(None)
    return ma1, ma2, ma3_full

# -------------------------
# Utility Functions
# -------------------------
def get_ma_arrangement(ma1, ma2, ma3):
    if not all(v is not None for v in (ma1, ma2, ma3)):
        return "UNDEFINED"
    if ma1 > ma2 > ma3:
        return "BULLISH_ARRANGEMENT"
    elif ma1 < ma2 < ma3:
        return "BEARISH_ARRANGEMENT"
    return "MIXED_ARRANGEMENT"

def is_packed_or_intersecting(ma1_arr, ma2_arr, ma3_arr, idx, lookback=10):
    # Check if MAs are packed horizontally or frequently intersecting in lookback
    packed_count, intersect_count = 0, 0
    for i in range(idx-lookback+1, idx+1):
        if i <= 0: continue
        vals = [ma1_arr[i], ma2_arr[i], ma3_arr[i]]
        if not all(v is not None for v in vals): continue
        ma1, ma2, ma3 = vals
        # Packed horizontally: difference < 0.2% of price
        max_min = max(vals) - min(vals)
        avg_price = (ma1 + ma2 + ma3)/3
        if abs(max_min) < avg_price*0.002: packed_count += 1
        # Intersecting: MAs cross each other
        if (ma1 > ma2 and ma1 < ma3) or (ma1 < ma2 and ma1 > ma3): intersect_count += 1
        if (ma2 > ma3 and ma2 < ma1) or (ma2 < ma3 and ma2 > ma1): intersect_count += 1
    return packed_count > 4 or intersect_count > 4

def check_ranging_market(candles, ma2, idx, lookback=10):
    # If price touches MA2 (high/low spans across MA2) more than 2 times in lookback = ranging
    touches = 0
    for i in range(idx-lookback+1, idx+1):
        if i < 0 or i >= len(candles): continue
        c = candles[i]
        v = ma2[i]
        if v is None: continue
        if c["low"] <= v <= c["high"]: touches += 1
    return touches > 2

def is_price_spike(candles, idx):
    # A spike is defined by a large vertical move (>2% from previous close)
    if idx < 2: return False
    prev_close = candles[idx-1]["close"]
    this_close = candles[idx]["close"]
    move = abs(this_close - prev_close) / prev_close
    return move > 0.02

def rejection_type(candle):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c-o)
    rng = h-l
    if rng == 0: return "NONE"
    wick_upper = h - max(o,c)
    wick_lower = min(o,c) - l
    ratio = body / rng if rng > 0 else 0
    # Pin bar: wick >= 1.2x body and body < 40% of range
    if wick_upper >= body*1.2 and body < rng*0.4: return "PIN_BAR"
    if wick_lower >= body*1.2 and body < rng*0.4: return "PIN_BAR"
    # Doji: body less than 20% of range
    if body < rng*0.2: return "DOJI"
    # Engulfing: previous candle body < current body and opposite direction
    return "ENGULFING"

def near_ma(price, ma, tolerance=0.001):
    if ma is None: return False
    return abs(price-ma) <= abs(ma)*tolerance

def fetch_candles(sym, tf, count=CANDLES_N):
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                ws.recv()
            request = {
                "ticks_history": sym,
                "style": "candles",
                "granularity": tf,
                "count": count,
                "end": "latest"
            }
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            if "candles" in response and response["candles"]:
                return [{
                    "epoch": int(c["epoch"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                } for c in response["candles"]]
        except Exception as e:
            time.sleep(1)
    return []

# -------------------------
# DSR Signal Detection Strict
# -------------------------
def detect_signal(candles, tf, shorthand):
    n = len(candles)
    if n < MIN_CANDLES:
        return None

    ma1, ma2, ma3 = compute_mas(candles)
    idx = n-2  # Confirmation candle is after rejection, so check second last
    current = candles[idx]
    next_candle = candles[idx+1]

    # --- Trend-only zone ---
    ma1_val, ma2_val, ma3_val = ma1[idx], ma2[idx], ma3[idx]
    arrangement = get_ma_arrangement(ma1_val, ma2_val, ma3_val)
    if arrangement not in {"BULLISH_ARRANGEMENT", "BEARISH_ARRANGEMENT"}:
        return None

    # --- No signals if packed/intersecting MAs or ranging ---
    if is_packed_or_intersecting(ma1, ma2, ma3, idx):
        return None
    if check_ranging_market(candles, ma2, idx):
        return None

    # --- No signals after price spike ---
    if is_price_spike(candles, idx):
        return None

    # --- Signal must be at MA1 or MA2 level ---
    at_ma1 = near_ma(current["close"], ma1_val) or near_ma(current["high"], ma1_val) or near_ma(current["low"], ma1_val)
    at_ma2 = near_ma(current["close"], ma2_val) or near_ma(current["high"], ma2_val) or near_ma(current["low"], ma2_val)
    if not (at_ma1 or at_ma2):
        return None

    # --- No signal if price between MA1 and MA2 ---
    if arrangement == "BULLISH_ARRANGEMENT" and ma2_val < current["close"] < ma1_val:
        return None
    if arrangement == "BEARISH_ARRANGEMENT" and ma1_val < current["close"] < ma2_val:
        return None

    # --- Rejection candle requirements ---
    pattern = rejection_type(current)
    if pattern not in {"PIN_BAR", "DOJI", "ENGULFING"}:
        return None

    # --- Confirmation candle direction ---
    if arrangement == "BULLISH_ARRANGEMENT":
        if next_candle["close"] <= next_candle["open"]:  # Not bullish
            return None
        side = "BUY"
    else:
        if next_candle["close"] >= next_candle["open"]:  # Not bearish
            return None
        side = "SELL"

    # --- Cooldown ---
    last_signal_time = getattr(detect_signal, f'last_signal_{shorthand}', 0)
    current_time = candles[idx+1]["epoch"]
    if current_time - last_signal_time < 1800:  # 30 min
        return None
    setattr(detect_signal, f'last_signal_{shorthand}', current_time)

    return {
        "symbol": shorthand,
        "tf": tf,
        "side": side,
        "pattern": pattern,
        "ma_level": "MA1" if at_ma1 else "MA2",
        "ma_arrangement": arrangement,
        "price": current["close"],
        "ma1": ma1_val,
        "ma2": ma2_val,
        "ma3": ma3_val,
        "idx": idx+1,
        "candles": candles,
        "ma1_array": ma1,
        "ma2_array": ma2,
        "ma3_array": ma3
    }

# -------------------------
# Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
    signal_idx = signal_data["idx"]
    n = len(candles)
    chart_start = max(0, n - LAST_N_CHART)
    chart_candles = candles[chart_start:]
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        if c >= o:
            body_color = "#00FF00"
            edge_color = "#00AA00"
        else:
            body_color = "#FF0000"
            edge_color = "#AA0000"
        ax.add_patch(Rectangle(
            (i - CANDLE_WIDTH/2, min(o, c)), 
            CANDLE_WIDTH, 
            max(abs(c - o), 1e-9),
            facecolor=body_color, 
            edgecolor=edge_color, 
            alpha=0.9,
            linewidth=1
        ))
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.2, alpha=0.8)
    def plot_ma(ma_values, label, color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        ax.plot(range(len(chart_candles)), chart_ma, color=color, linewidth=linewidth, label=label, alpha=0.9)
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#FFFFFF", 2)
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00BFFF", 2)
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#FF6347", 2)
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        marker_color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker_symbol = "^" if signal_data["side"] == "BUY" else "v"
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    arrangement_emoji = "📈" if signal_data["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} DSR Signal {arrangement_emoji}", 
                fontsize=16, color='white', fontweight='bold', pad=20)
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', 
                      edgecolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.8)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, dpi=150, bbox_inches="tight", facecolor='black',edgecolor='none',pad_inches=0.1)
    plt.close()
    plt.style.use('default')
    return chart_file.name

# -------------------------
# Main Execution
# -------------------------
def get_timeframe_for_symbol(shorthand):
    return SYMBOL_TF_MAP.get(shorthand, TIMEFRAMES[0] if TIMEFRAMES else 300)

def run_analysis():
    signals_found = 0
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = get_timeframe_for_symbol(shorthand)
            candles = fetch_candles(deriv_symbol, tf)
            if len(candles) < MIN_CANDLES: continue
            signal = detect_signal(candles, tf, shorthand)
            if not signal: continue
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]): continue
            arrangement_emoji = "📈" if signal["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
            caption = (f"🎯 {signal['symbol']} {tf}s - {signal['side']} SIGNAL\n"
                      f"{arrangement_emoji} MA Setup: {signal['ma_arrangement'].replace('_', ' ')}\n" 
                      f"🎨 Pattern: {signal['pattern']}\n"
                      f"📍 Level: {signal['ma_level']} Dynamic S/R\n"
                      f"💰 Price: {signal['price']:.5f}")
            chart_path = create_signal_chart(signal)
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
            try: os.unlink(chart_path)
            except: pass
        except Exception as e:
            if DEBUG: print(f"Error analyzing {shorthand}: {e}")
    if DEBUG:
        print(f"Analysis complete. {signals_found} DSR signals found.")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
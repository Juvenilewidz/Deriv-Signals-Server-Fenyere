#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (M5 Swing Trend Focused)

DESIGNED FOR M5 TIMEFRAME SWING TRENDS:
- Detects short-term swing trends (not long-term trends)
- MA3 breaks with pullback retests to MA1/MA2
- Works with completed candles only
- Focuses on the exact setups you've highlighted
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
import websocket, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

TIMEFRAMES = [int(x) for x in os.getenv("TIMEFRAMES","300").split(",") if x.strip().isdigit()]
DEBUG = os.getenv("DEBUG","0") == "1"
TEST_MODE = os.getenv("TEST_MODE","0") == "1"

CANDLES_N = 300
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

# M5 SWING PARAMETERS (Optimized for short-term trends)
SWING_LOOKBACK = 15        # Look back 15 candles for swing detection
MA3_BREAK_LOOKBACK = 12    # MA3 break detection window
RETEST_PROXIMITY = 0.004   # 0.4% proximity for MA retest
MIN_SWING_SIZE = 0.006     # 0.6% minimum swing size for M5

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

def load_persist():
    try: return json.load(open(ALERT_FILE))
    except: return {}

def save_persist(d):
    try: json.dump(d, open(ALERT_FILE,"w"))
    except: pass

def already_sent(shorthand, tf, epoch, side):
    if TEST_MODE: return False
    rec = load_persist().get(f"{shorthand}|{tf}")
    return bool(rec and rec.get("epoch")==epoch and rec.get("side")==side)

def mark_sent(shorthand, tf, epoch, side):
    d=load_persist(); d[f"{shorthand}|{tf}"]={"epoch":epoch,"side":side}; save_persist(d)

# Moving Averages
def smma_correct(series, period):
    n = len(series)
    if n < period: return [None] * n
    result = [None] * (period - 1)
    first_sma = sum(series[:period]) / period
    result.append(first_sma)
    prev_smma = first_sma
    for i in range(period, n):
        current_smma = (prev_smma * (period - 1) + series[i]) / period
        result.append(current_smma)
        prev_smma = current_smma
    return result

def sma(series, period):
    n = len(series)
    if n < period: return [None] * n
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
    ma1 = smma_correct(hlc3, 9)
    ma2 = smma_correct(closes, 19)
    ma2_valid = [v for v in ma2 if v is not None]
    if len(ma2_valid) >= 25:
        ma3_calc = sma(ma2_valid, 25)
        ma3 = []
        valid_idx = 0
        for v in ma2:
            if v is None:
                ma3.append(None)
            else:
                if valid_idx < len(ma3_calc):
                    ma3.append(ma3_calc[valid_idx])
                else:
                    ma3.append(None)
                valid_idx += 1
    else:
        ma3 = [None] * len(candles)
    return ma1, ma2, ma3

# M5 SWING TREND DETECTION
def detect_swing_trend(candles, ma1, ma2, ma3, current_idx):
    """Detect swing trends suitable for M5 timeframe"""
    if current_idx < SWING_LOOKBACK:
        return None, None
    
    # Look for recent swing high/low
    swing_high_idx = None
    swing_low_idx = None
    swing_high_price = 0
    swing_low_price = float('inf')
    
    # Find swing points in recent candles
    for i in range(max(0, current_idx - SWING_LOOKBACK), current_idx):
        price_high = candles[i]["high"]
        price_low = candles[i]["low"]
        
        if price_high > swing_high_price:
            swing_high_price = price_high
            swing_high_idx = i
            
        if price_low < swing_low_price:
            swing_low_price = price_low
            swing_low_idx = i
    
    if swing_high_idx is None or swing_low_idx is None:
        return None, None
    
    current_price = candles[current_idx]["close"]
    
    # Determine swing trend direction
    if swing_low_idx < swing_high_idx:
        # Low came before high = potential uptrend
        swing_size = (swing_high_price - swing_low_price) / swing_low_price
        if swing_size >= MIN_SWING_SIZE:
            return "UP_SWING", {"high_idx": swing_high_idx, "low_idx": swing_low_idx, "size": swing_size}
    else:
        # High came before low = potential downtrend  
        swing_size = (swing_high_price - swing_low_price) / swing_high_price
        if swing_size >= MIN_SWING_SIZE:
            return "DOWN_SWING", {"high_idx": swing_high_idx, "low_idx": swing_low_idx, "size": swing_size}
    
    return None, None

# MA3 BREAK DETECTION FOR M5
def find_ma3_break(candles, ma3, current_idx):
    """Find recent MA3 break for swing trends"""
    if current_idx < MA3_BREAK_LOOKBACK:
        return None
    
    # Look for MA3 break in recent candles
    for i in range(max(1, current_idx - MA3_BREAK_LOOKBACK), current_idx):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        
        prev_close = candles[i-1]["close"]
        curr_close = candles[i]["close"]
        
        # Break above MA3
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            return {"idx": i, "direction": "UP", "level": ma3[i]}
        
        # Break below MA3
        if prev_close >= ma3[i-1] and curr_close < ma3[i]:
            return {"idx": i, "direction": "DOWN", "level": ma3[i]}
    
    return None

# RETEST DETECTION AT MA LEVELS
def detect_ma_retest(candles, ma1, ma2, current_idx, break_direction):
    """Detect retest of MA1 or MA2 after MA3 break"""
    current_candle = candles[current_idx]
    
    if not all(v is not None for v in [ma1[current_idx], ma2[current_idx]]):
        return False, "NONE"
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    current_close = current_candle["close"]
    
    # Check proximity to MA levels
    ma1_distance = abs(current_candle["low"] - current_ma1) / current_close
    ma2_distance = abs(current_candle["low"] - current_ma2) / current_close
    ma1_distance_high = abs(current_candle["high"] - current_ma1) / current_close
    ma2_distance_high = abs(current_candle["high"] - current_ma2) / current_close
    
    if break_direction == "UP":
        # Look for pullback to MA1/MA2 from above
        if (ma1_distance <= RETEST_PROXIMITY or ma2_distance <= RETEST_PROXIMITY):
            return True, "PULLBACK_RETEST"
    else:
        # Look for bounce to MA1/MA2 from below
        if (ma1_distance_high <= RETEST_PROXIMITY or ma2_distance_high <= RETEST_PROXIMITY):
            return True, "BOUNCE_RETEST"
    
    return False, "NONE"

# REJECTION PATTERN DETECTION
def detect_rejection_pattern(candle, prev_candle):
    """Detect rejection patterns for M5 timeframe"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # Doji patterns
    if body_size <= total_range * 0.25:
        return True, "DOJI"
    
    # Pinbar patterns
    if upper_wick >= total_range * 0.35:
        return True, "PINBAR"
    if lower_wick >= total_range * 0.35:
        return True, "PINBAR"
    
    # Engulfing patterns
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        if (prev_c < prev_o and c > o and o <= prev_c and c >= prev_o):
            return True, "BULL_ENGULF"
        if (prev_c > prev_o and c < o and o >= prev_c and c <= prev_o):
            return True, "BEAR_ENGULF"
    
    # Strong rejection (close away from extreme)
    if c <= h * 0.6 + l * 0.4:  # Close in lower 40%
        return True, "HIGH_REJECT"
    if c >= h * 0.4 + l * 0.6:  # Close in upper 60%
        return True, "LOW_REJECT"
    
    return False, "NONE"

# DATA FETCHING
def fetch_completed_candles_only(sym, tf):
    """Fetch completed candles with timeframe optimization"""
    count = 200 if tf == 1 else CANDLES_N
    timeout = 25 if tf == 1 else 20
    
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=timeout)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
            
            request = {
                "ticks_history": sym,
                "style": "candles",
                "granularity": tf,
                "count": count + 2,
                "end": "latest"
            }
            
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            
            if "candles" in response and len(response["candles"]) > 2:
                candles_data = [{
                    "epoch": int(c["epoch"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                } for c in response["candles"]]
                
                # Use second-to-last candle (fully completed)
                completed_candles = candles_data[:-1]
                
                if DEBUG:
                    print(f"Fetched {len(completed_candles)} completed candles for {sym} ({tf}s)")
                
                return completed_candles
        
        except Exception as e:
            if DEBUG:
                print(f"Attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(0.5 if tf == 1 else 1)
    
    return []

# M5 SWING SIGNAL DETECTION
def detect_m5_swing_signal(candles, tf, shorthand):
    """Detect swing trend signals optimized for M5 timeframe"""
    n = len(candles)
    min_candles = 100 if tf == 1 else 120
    
    if n < min_candles:
        return None
    
    # Use completed candle
    current_idx = n - 1
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    if not all(v is not None for v in [ma1[current_idx], ma2[current_idx], ma3[current_idx]]):
        return None
    
    # Step 1: Detect swing trend
    swing_direction, swing_info = detect_swing_trend(candles, ma1, ma2, ma3, current_idx)
    if swing_direction is None:
        return None
    
    # Step 2: Find MA3 break
    ma3_break = find_ma3_break(candles, ma3, current_idx)
    if not ma3_break:
        return None
    
    # Step 3: Check if break direction matches swing direction
    if ((swing_direction == "UP_SWING" and ma3_break["direction"] != "UP") or
        (swing_direction == "DOWN_SWING" and ma3_break["direction"] != "DOWN")):
        return None
    
    # Step 4: Detect MA retest
    is_retest, retest_type = detect_ma_retest(candles, ma1, ma2, current_idx, ma3_break["direction"])
    if not is_retest:
        return None
    
    # Step 5: Check for rejection pattern
    has_rejection, pattern_type = detect_rejection_pattern(current_candle, prev_candle)
    if not has_rejection:
        return None
    
    # Step 6: Final validation
    current_close = current_candle["close"]
    signal_side = "BUY" if ma3_break["direction"] == "UP" else "SELL"
    
    # Price positioning check
    if signal_side == "BUY" and current_close <= ma3[current_idx]:
        return None
    if signal_side == "SELL" and current_close >= ma3[current_idx]:
        return None
    
    return {
        "symbol": shorthand,
        "tf": tf,
        "side": signal_side,
        "pattern": pattern_type,
        "swing_info": swing_info,
        "ma3_break": ma3_break,
        "retest_type": retest_type,
        "idx": current_idx,
        "ma1": ma1,
        "ma2": ma2,
        "ma3": ma3,
        "candles": candles
    }

# CHART GENERATION
def create_swing_chart(signal_data):
    """Chart for swing trend signals"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1"], signal_data["ma2"], signal_data["ma3"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - 120)  # Show recent 120 candles
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        body_color = "#00FF00" if c >= o else "#FF0000"
        edge_color = "#00AA00" if c >= o else "#AA0000"
        
        ax.add_patch(Rectangle(
            (i - 0.35/2, min(o, c)), 0.35, max(abs(c - o), 1e-9),
            facecolor=body_color, edgecolor=edge_color, alpha=0.9, linewidth=1
        ))
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.2, alpha=0.8)
    
    # Plot moving averages
    def plot_ma(ma_values, label, color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        ax.plot(range(len(chart_candles)), chart_ma, 
                color=color, linewidth=linewidth, label=label, alpha=0.9)
    
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#FFFFFF", 2)
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00BFFF", 2)
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#FF6347", 2)
    
    # Mark signal point
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        marker_color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker_symbol = "^" if signal_data["side"] == "BUY" else "v"
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    
    # Mark MA3 break point
    ma3_break_idx = signal_data["ma3_break"]["idx"] - chart_start
    if 0 <= ma3_break_idx < len(chart_candles):
        break_price = signal_data["ma3_break"]["level"]
        ax.scatter([ma3_break_idx], [break_price], 
                  color="#FFFF00", marker="o", s=150, 
                  edgecolor="#FFFFFF", linewidth=2, zorder=8, alpha=0.8)
    
    tf_label = f"{signal_data['tf']}s" if signal_data['tf'] < 60 else f"{signal_data['tf']//60}m"
    ax.set_title(f"{signal_data['symbol']} {tf_label} - {signal_data['side']} SWING SIGNAL", 
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
    plt.savefig(chart_file.name, dpi=150, bbox_inches="tight", 
                facecolor='black', edgecolor='none', pad_inches=0.1)
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

# MAIN EXECUTION
def run_swing_analysis():
    """Run swing trend analysis for M5 and 1s timeframes"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = SYMBOL_TF_MAP.get(shorthand, TIMEFRAMES[0] if TIMEFRAMES else 300)
            
            if DEBUG:
                print(f"M5 Swing analysis for {shorthand} on {tf}s...")
            
            candles = fetch_completed_candles_only(deriv_symbol, tf)
            if len(candles) < (100 if tf == 1 else 120):
                continue
            
            signal = detect_m5_swing_signal(candles, tf, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                continue
            
            # Create alert
            tf_display = "1s" if tf == 1 else f"{tf}s" if tf < 60 else f"{tf//60}m"
            swing_size = signal["swing_info"]["size"] * 100
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} SWING\n"
                      f"📊 Pattern: {signal['pattern']}\n"
                      f"📈 Swing Size: {swing_size:.1f}%\n"
                      f"🔄 MA3 Break: {signal['ma3_break']['direction']} at candle {signal['ma3_break']['idx']}\n"
                      f"✅ Retest: {signal['retest_type']}\n"
                      f"💰 Price: {signal['candles'][signal['idx']]['close']}\n"
                      f"📝 M5 Swing Trend Detected")
            
            chart_path = create_swing_chart(signal)
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"✅ M5 SWING signal sent: {shorthand} {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error in {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"M5 Swing analysis complete. {signals_found} signals found.")

if __name__ == "__main__":
    try:
        run_swing_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

#!/usr/bin/env python3
"""
Pure DSR Trading Bot - M10 Timeframe Only
100% Core DSR Strategy - ZERO Numeric Thresholds - Pure Trend Following
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

DEBUG = os.getenv("DEBUG","0") == "1"
TEST_MODE = os.getenv("TEST_MODE","0") == "1"

CANDLES_N = 480
LAST_N_CHART = 180
CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_pure_core.json")
MIN_CANDLES = 50

# M10 TIMEFRAME ONLY
TIMEFRAME = 600  # 10 minutes only

# -------------------------
# Symbol Mappings (Removed 1s indices)
# -------------------------
SYMBOL_MAP = {
    "V10": "R_10",
    "V25": "R_25",
    "V50": "R_50", 
    "V75": "R_75",
    "Jump10": "JD10",
    "Jump25": "JD25",
    "Jump75": "JD75",
    "Jump50": "JD50", 
    "Jump100": "JD100",
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

def already_sent(shorthand, epoch, side):
    if TEST_MODE:
        return False
    rec = load_persist().get(shorthand)
    return bool(rec and rec.get("epoch")==epoch and rec.get("side")==side)

def mark_sent(shorthand, epoch, side):
    d=load_persist(); d[shorthand]={"epoch":epoch,"side":side}; save_persist(d)

# -------------------------
# Moving Averages
# -------------------------
def smma_correct(series, period):
    """Proper SMMA calculation"""
    n = len(series)
    if n < period:
        return [None] * n
    
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
    """Standard SMA calculation"""
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
    """Compute MAs exactly as per strategy"""
    closes = [c["close"] for c in candles]
    hlc3 = [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    
    # MA1 → SMMA of HLC3, period 9
    ma1 = smma_correct(hlc3, 9)
    
    # MA2 → SMMA of Close, period 19  
    ma2 = smma_correct(closes, 19)
    
    # MA3 → SMA of MA2, period 25
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

# -------------------------
# Core DSR Trend Detection - COMPLETELY DYNAMIC
# -------------------------
def detect_trend_direction(ma1, ma2, ma3, current_idx, lookback=10):
    """Detect trend direction using MA arrangement and MA3 slope - NO thresholds"""
    if current_idx < lookback:
        return "NONE"
    
    # Get current MA values
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx] 
    current_ma3 = ma3[current_idx]
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return "NONE"
    
    # Get MA3 values for slope calculation
    ma3_values = []
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i >= 0 and i < len(ma3) and ma3[i] is not None:
            ma3_values.append(ma3[i])
    
    if len(ma3_values) < lookback // 2:
        return "NONE"
    
    # Calculate MA3 slope - COMPLETELY DYNAMIC
    ma3_slope = ma3_values[-1] - ma3_values[0]  # Simple slope
    
    # Check MA arrangement AND slope alignment - NO numeric thresholds
    if current_ma1 > current_ma2 > current_ma3 and ma3_slope > 0:
        return "UPTREND"
    elif current_ma1 < current_ma2 < current_ma3 and ma3_slope < 0:
        return "DOWNTREND"
    else:
        return "NONE"

def ma1_closest_to_price(candle, ma1_val, ma2_val):
    """Check if MA1 is closest to price - COMPLETELY DYNAMIC"""
    if ma1_val is None or ma2_val is None:
        return False
    
    close_price = candle["close"]
    
    # MA1 must be closer to price than MA2 - NO numeric thresholds
    distance_to_ma1 = abs(close_price - ma1_val)
    distance_to_ma2 = abs(close_price - ma2_val)
    
    return distance_to_ma1 < distance_to_ma2

def detect_ma1_action(candle, ma1_val, trend_direction):
    """Detect MA1 rejection or break - COMPLETELY DYNAMIC"""
    if ma1_val is None:
        return False, "NONE"
    
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    
    # Check if price interacted with MA1
    ma1_in_range = (l <= ma1_val <= h)
    if not ma1_in_range:
        return False, "NONE"
    
    if trend_direction == "UPTREND":
        # In uptrend: Look for bullish action at MA1
        lower_wick = min(o, c) - l
        
        # MA1 rejection: Price touched MA1 and rejected upward
        if c > ma1_val and lower_wick > 0:
            return True, "MA1_REJECTION_UP"
        
        # MA1 break: Price closed above MA1 
        if c > ma1_val:
            return True, "MA1_BREAK_UP"
    
    elif trend_direction == "DOWNTREND":
        # In downtrend: Look for bearish action at MA1
        upper_wick = h - max(o, c)
        
        # MA1 rejection: Price touched MA1 and rejected downward
        if c < ma1_val and upper_wick > 0:
            return True, "MA1_REJECTION_DOWN"
        
        # MA1 break: Price closed below MA1
        if c < ma1_val:
            return True, "MA1_BREAK_DOWN"
    
    return False, "NONE"

# -------------------------
# Data Fetching
# -------------------------
def fetch_candles(sym, count=CANDLES_N):
    """Fetch candles from Deriv API - M10 only"""
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
            
            request = {
                "ticks_history": sym,
                "style": "candles", 
                "granularity": TIMEFRAME,
                "count": count,
                "end": "latest"
            }
            
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            
            if DEBUG:
                print(f"Fetched {len(response.get('candles', []))} candles for {sym}")
            
            if "candles" in response and response["candles"]:
                return [{
                    "epoch": int(c["epoch"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                } for c in response["candles"]]
        
        except Exception as e:
            if DEBUG:
                print(f"Attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(1)
    
    return []

# -------------------------
# CORE DSR Signal Detection - 100% Pure
# -------------------------
def detect_core_dsr_signal(candles, shorthand):
    """100% CORE DSR Strategy - NO numeric thresholds - Pure trend following"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    current_idx = n - 1
    current_candle = candles[current_idx]
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    # CORE DSR STEP 1: Confirm trend direction (MA arrangement + MA3 slope)
    trend_direction = detect_trend_direction(ma1, ma2, ma3, current_idx)
    if trend_direction == "NONE":
        if DEBUG:
            print(f"No defined trend for {shorthand}")
        return None
    
    # CORE DSR STEP 2: MA1 must be closest to price (dynamic S/R)
    if not ma1_closest_to_price(current_candle, current_ma1, current_ma2):
        if DEBUG:
            print(f"MA1 not closest to price for {shorthand}")
        return None
    
    # CORE DSR STEP 3: Detect MA1 action (rejection or break)
    has_action, action_type = detect_ma1_action(current_candle, current_ma1, trend_direction)
    if not has_action:
        if DEBUG:
            print(f"No MA1 action for {shorthand}")
        return None
    
    # CORE DSR STEP 4: Generate signal based on trend and action
    signal_side = None
    
    if trend_direction == "UPTREND":
        if action_type in ["MA1_REJECTION_UP", "MA1_BREAK_UP"]:
            signal_side = "BUY"
    
    elif trend_direction == "DOWNTREND":
        if action_type in ["MA1_REJECTION_DOWN", "MA1_BREAK_DOWN"]:
            signal_side = "SELL"
    
    if not signal_side:
        return None
    
    # Cooldown
    cooldown_key = f'last_signal_{shorthand}'
    last_signal_time = getattr(detect_core_dsr_signal, cooldown_key, 0)
    current_time = current_candle["epoch"]
    
    if current_time - last_signal_time < 1800:  # 30 minutes
        return None
    
    setattr(detect_core_dsr_signal, cooldown_key, current_time)
    
    if DEBUG:
        print(f"CORE DSR: {signal_side} - {action_type} in {trend_direction}")
        print(f"MA1: {current_ma1:.5f}, MA2: {current_ma2:.5f}, Price: {current_candle['close']:.5f}")
    
    return {
        "symbol": shorthand,
        "side": signal_side,
        "pattern": action_type,
        "trend": trend_direction,
        "price": current_candle["close"],
        "ma1": current_ma1,
        "ma2": current_ma2, 
        "ma3": current_ma3,
        "idx": current_idx,
        "candles": candles,
        "ma1_array": ma1,
        "ma2_array": ma2,
        "ma3_array": ma3
    }

# -------------------------
# Chart Generation
# -------------------------
def create_core_dsr_chart(signal_data):
    """Create clean core DSR chart"""
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
    
    # Plot candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        
        if c >= o:
            body_color = "#00FF88"
            edge_color = "#00CC66"
        else:
            body_color = "#FF3366"
            edge_color = "#CC1144"
        
        ax.add_patch(Rectangle(
            (i - CANDLE_WIDTH/2, min(o, c)), 
            CANDLE_WIDTH, 
            max(abs(c - o), 1e-9),
            facecolor=body_color, 
            edgecolor=edge_color, 
            alpha=0.9,
            linewidth=1.2
        ))
        
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.5, alpha=0.9)
    
    # Plot moving averages
    def plot_ma(ma_values, label, color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        
        valid_points = [(i, v) for i, v in enumerate(chart_ma) if v is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            ax.plot(x_vals, y_vals, color=color, linewidth=linewidth, 
                   label=label, alpha=0.95, linestyle='-')
    
    # Plot all MAs
    plot_ma(ma1, "MA1 (SMMA HLC3-9) - Dynamic S/R", "#FFFFFF", 3)
    plot_ma(ma2, "MA2 (SMMA Close-19) - Dynamic S/R", "#00BFFF", 2.5)
    plot_ma(ma3, "MA3 (SMA MA2-25) - Trend Filter", "#FF6347", 2)
    
    # Mark signal point
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        
        if signal_data["side"] == "BUY":
            marker_color = "#00FF88"
            marker_symbol = "^"
        else:
            marker_color = "#FF3366" 
            marker_symbol = "v"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=400, edgecolor="#FFFFFF", linewidth=4, zorder=10)
        
        # Add signal annotation
        ax.annotate(f'{signal_data["side"]} DSR Signal', 
                   xy=(signal_chart_idx, signal_price),
                   xytext=(signal_chart_idx + 15, signal_price),
                   fontsize=12, color='white', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=marker_color, alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='white', lw=1))
    
    # Clean title
    trend_emoji = "📈" if signal_data["trend"] == "UPTREND" else "📉"
    
    ax.set_title(f"{signal_data['symbol']} M10 - {signal_data['side']} DSR Signal {trend_emoji}\n"
                f"Core Trend Following Strategy | {signal_data['trend']} Confirmed", 
                fontsize=14, color='white', fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', 
                      edgecolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Grid and styling
    ax.grid(True, alpha=0.2, color='gray', linestyle=':', linewidth=0.8)
    ax.tick_params(colors='white', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, 
                dpi=180, 
                bbox_inches="tight", 
                facecolor='black',
                edgecolor='none',
                pad_inches=0.2)
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

# -------------------------
# Main Execution
# -------------------------
def run_core_dsr_analysis():
    """Core DSR analysis - 100% trend following"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            if DEBUG:
                print(f"Analyzing {shorthand} on M10...")
            
            candles = fetch_candles(deriv_symbol)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"Insufficient candles for {shorthand}: {len(candles)}")
                continue
            
            signal = detect_core_dsr_signal(candles, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, current_epoch, signal["side"]):
                if DEBUG:
                    print(f"Signal already sent for {shorthand}")
                continue
            
            # Create core DSR alert message
            trend_emoji = "📈" if signal["trend"] == "UPTREND" else "📉"
            
            # Signal type description
            signal_desc = {
                "MA1_REJECTION_UP": "MA1 Rejection (Bullish)",
                "MA1_BREAK_UP": "MA1 Break (Bullish)", 
                "MA1_REJECTION_DOWN": "MA1 Rejection (Bearish)",
                "MA1_BREAK_DOWN": "MA1 Break (Bearish)"
            }.get(signal["pattern"], signal["pattern"])
            
            caption = (f"🎯 {signal['symbol']} M10 - {signal['side']} DSR SIGNAL {trend_emoji}\n\n"
                      f"📊 Trend Direction: {signal['trend']}\n"
                      f"🎨 Signal Type: {signal_desc}\n"
                      f"📍 Level: MA1 (Dynamic S/R)\n"
                      f"💰 Entry Price: {signal['price']:.5f}\n"
                      f"🔄 MA1: {signal['ma1']:.5f} | MA2: {signal['ma2']:.5f}\n\n"
                      f"💎 Core DSR: MA1 Action in Confirmed Trends\n"
                      f"✅ MA3 Slope + MA Arrangement + MA1 Proximity")
            
            chart_path = create_core_dsr_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"Core DSR signal sent for {shorthand}: {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                    
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"Core DSR Analysis complete. {signals_found} pure signals found.")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        print("Starting Core DSR Trading Bot - M10 Timeframe Only")
        print("Strategy: 100% Pure Trend Following - ZERO Numeric Thresholds")
        print("Requirements: MA3 Slope + MA Arrangement + MA1 Proximity + MA1 Action")
        
        run_core_dsr_analysis()
        
    except Exception as e:
        print(f"Critical error in Core DSR Bot: {e}")
        traceback.print_exc()

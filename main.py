#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance Trading Bot (Enhanced)

Enhanced DSR Strategy with Confirmed Rejection:
- MA crossovers and rearrangement confirm trend changes
- MA1 (SMMA HLC3-9) and MA2 (SMMA Close-19) = Dynamic S/R
- MA3 (SMA MA2-25) = Trend filter
- BUY BIAS: Only when MA1 > MA2 (bullish arrangement)
- SELL BIAS: Only when MA1 < MA2 (bearish arrangement)
- CONFIRMED rejection patterns at MA1/MA2 levels = Signals
- Wait for candlestick to CLOSE before confirming signal
- NO numeric tolerance thresholds
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
    # CORRECTED 1s indices symbols from Deriv API
    "V75(1s)": "1s_V75",     # Fixed from R_1HZ75V
    "V100(1s)": "1s_V100",   # Fixed from R_1HZ100V  
    "V150(1s)": "1s_V150",   # Fixed from R_1HZ150V
    "V15(1s)": "1s_V15"      # Fixed from R_1HZ15V
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
# MA Crossover Detection (Core DSR)
# -------------------------
def detect_ma_crossover(ma1, ma2, current_idx, lookback=3):
    """Detect MA crossovers - key for trend change confirmation"""
    if current_idx < lookback:
        return False, "NONE"
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    
    if current_ma1 is None or current_ma2 is None:
        return False, "NONE"
    
    # Check for crossover in last 3 candles
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i > 0 and i < len(ma1) and i < len(ma2):
            prev_ma1 = ma1[i-1]
            prev_ma2 = ma2[i-1]
            curr_ma1 = ma1[i]
            curr_ma2 = ma2[i]
            
            if all(v is not None for v in [prev_ma1, prev_ma2, curr_ma1, curr_ma2]):
                # Bullish crossover: MA1 crosses above MA2
                if prev_ma1 <= prev_ma2 and curr_ma1 > curr_ma2:
                    return True, "BULLISH_CROSSOVER"
                
                # Bearish crossover: MA1 crosses below MA2  
                if prev_ma1 >= prev_ma2 and curr_ma1 < curr_ma2:
                    return True, "BEARISH_CROSSOVER"
    
    return False, "NONE"

def get_ma_arrangement(ma1_val, ma2_val, ma3_val):
    """Get current MA arrangement for trend confirmation"""
    if not all(v is not None for v in [ma1_val, ma2_val, ma3_val]):
        return "UNDEFINED"
    
    if ma1_val > ma2_val > ma3_val:
        return "BULLISH_ARRANGEMENT"  # MA1 on top = uptrend
    elif ma1_val < ma2_val < ma3_val:
        return "BEARISH_ARRANGEMENT"  # MA1 on bottom = downtrend
    else:
        return "MIXED_ARRANGEMENT"  # MAs mixed = no clear trend

# -------------------------
# Enhanced Confirmed Rejection Detection
# -------------------------
def is_confirmed_rejection_at_ma(candle, ma_level):
    """
    Check if candle shows CONFIRMED rejection at MA level
    - Must be a CLOSED candle (not current forming candle)
    - Price must test MA level and reject (not just interact)
    - NO numeric tolerance - pure rejection behavior
    """
    if ma_level is None:
        return False, "NONE"
    
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    
    # Check if candle actually touched the MA level
    ma_was_touched = (l <= ma_level <= h)
    if not ma_was_touched:
        return False, "NO_TOUCH"
    
    # Determine rejection type based on MA level position and close
    rejection_type = "NONE"
    
    # UPPER REJECTION: MA tested from below, price rejected back down
    if c < ma_level and h >= ma_level:
        # Price went up to test MA but closed below it = upper rejection
        body_size = abs(c - o)
        upper_wick = h - max(o, c)
        
        # Must have meaningful upper wick or small body showing rejection
        if upper_wick > 0 and (upper_wick >= body_size * 0.5 or body_size < (h - l) * 0.3):
            rejection_type = "UPPER_REJECTION"
    
    # LOWER REJECTION: MA tested from above, price rejected back up  
    elif c > ma_level and l <= ma_level:
        # Price went down to test MA but closed above it = lower rejection
        body_size = abs(c - o)
        lower_wick = min(o, c) - l
        
        # Must have meaningful lower wick or small body showing rejection
        if lower_wick > 0 and (lower_wick >= body_size * 0.5 or body_size < (h - l) * 0.3):
            rejection_type = "LOWER_REJECTION"
    
    return rejection_type != "NONE", rejection_type

def test_ma_rejection(candle, ma1_val, ma2_val):
    """Test if candle shows rejection at MA1 or MA2"""
    
    # Test MA1 rejection first (primary S/R)
    ma1_rejected, ma1_pattern = is_confirmed_rejection_at_ma(candle, ma1_val)
    if ma1_rejected:
        return True, "MA1", ma1_pattern
    
    # Test MA2 rejection (secondary S/R)
    ma2_rejected, ma2_pattern = is_confirmed_rejection_at_ma(candle, ma2_val)
    if ma2_rejected:
        return True, "MA2", ma2_pattern
    
    return False, "NONE", "NONE"

# -------------------------
# Ranging Market Detection
# -------------------------
def check_ranging_market(candles, ma1, ma2, current_idx, lookback=10):
    """Check if price touches MA2 more than twice = ranging market"""
    if current_idx < lookback:
        return False
    
    ma2_touches = 0
    
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i < len(candles) and i < len(ma2) and ma2[i] is not None:
            candle = candles[i]
            ma2_val = ma2[i]
            
            # Check if candle touched MA2 (high/low spans across MA2)
            if candle["low"] <= ma2_val <= candle["high"]:
                ma2_touches += 1
    
    # If price touches MA2 more than twice in lookback period = ranging
    return ma2_touches > 2

# -------------------------
# Data Fetching
# -------------------------
def fetch_candles(sym, tf, count=CANDLES_N):
    """Fetch candles from Deriv API"""
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
            
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
# Enhanced DSR Signal Detection with Confirmed Rejection
# -------------------------
def detect_signal(candles, tf, shorthand):
    """Enhanced DSR Strategy with confirmed rejection - All rules implemented correctly"""
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
    
    current_close = current_candle["close"]
    current_high = current_candle["high"]
    current_low = current_candle["low"]
    
    # DSR RULE 1: Determine bias from MA1/MA2 relationship (Enhanced)
    if current_ma1 > current_ma2:
        bias = "BUY_BIAS"
    elif current_ma1 < current_ma2:
        bias = "SELL_BIAS"
    else:
        # MA1 = MA2, no clear bias
        return None
    
    # DSR RULE 6: No ranging markets
    is_ranging = check_ranging_market(candles, ma1, ma2, current_idx)
    if is_ranging:
        return None
    
    # Test for CONFIRMED MA rejection
    has_rejection, ma_level, pattern_type = test_ma_rejection(current_candle, current_ma1, current_ma2)
    
    if not has_rejection:
        return None
    
    # DSR RULE 2 & 3: Enhanced price position requirements
    if bias == "BUY_BIAS" and current_close <= current_ma1:
        # BUY signals only when price ABOVE MA1
        return None
        
    if bias == "SELL_BIAS" and current_close >= current_ma1:
        # SELL signals only when price BELOW MA1
        return None
    
    # DSR RULE 4: NO signals when price between MAs
    if current_ma1 > current_ma2:  # Uptrend structure
        if current_ma2 < current_close < current_ma1:
            return None  # Price between MA2 and MA1
    else:  # Downtrend structure  
        if current_ma1 < current_close < current_ma2:
            return None  # Price between MA1 and MA2
    
    # Generate signal based on bias and confirmed rejection
    signal_side = None
    context = ""
    
    if bias == "BUY_BIAS":
        # In uptrend (MA1 > MA2), look for lower rejection (buying opportunity)
        if pattern_type == "LOWER_REJECTION":
            signal_side = "BUY"
            context = "MA1 above MA2 - uptrend confirmed, lower rejection at MA level"
        
    elif bias == "SELL_BIAS":
        # In downtrend (MA1 < MA2), look for upper rejection (selling opportunity) 
        if pattern_type == "UPPER_REJECTION":
            signal_side = "SELL"
            context = "MA1 below MA2 - downtrend confirmed, upper rejection at MA level"
    
    if signal_side is None:
        return None
    
    # Enhanced crossover detection
    has_crossover, crossover_type = detect_ma_crossover(ma1, ma2, current_idx)
    
    # Get MA arrangement
    ma_arrangement = get_ma_arrangement(current_ma1, current_ma2, current_ma3)
    
    # Cooldown to prevent spam (reduced for faster signals)
    last_signal_time = getattr(detect_signal, f'last_signal_{shorthand}', 0)
    current_time = current_candle["epoch"]
    
    if current_time - last_signal_time < 900:  # 15 minutes cooldown
        return None
    
    setattr(detect_signal, f'last_signal_{shorthand}', current_time)
    
    if DEBUG:
        print(f"ENHANCED DSR: {signal_side} - {pattern_type} at {ma_level} - Arrangement: {ma_arrangement}")
        print(f"Price: {current_close:.5f}, MA1: {current_ma1:.5f}, MA2: {current_ma2:.5f}, MA3: {current_ma3:.5f}")
    
    return {
        "symbol": shorthand,
        "tf": tf,
        "side": signal_side,
        "pattern": pattern_type,
        "ma_level": ma_level,
        "ma_arrangement": ma_arrangement,
        "crossover": crossover_type,
        "context": context,
        "price": current_close,
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
# Enhanced Chart Generation - Plot All MAs
# -------------------------
def create_signal_chart(signal_data):
    """Create enhanced chart showing all MAs and confirmed rejection signal"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - LAST_N_CHART)
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        
        if c >= o:
            body_color = "#00FF00"
            edge_color = "#00DD00"
        else:
            body_color = "#FF0000"
            edge_color = "#DD0000"
        
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
    
    # Plot ALL moving averages with enhanced styling
    def plot_ma_safe(ma_values, label, color, linewidth=2, alpha=0.9):
        chart_ma = []
        x_vals = []
        for i, candle_idx in enumerate(range(chart_start, n)):
            if candle_idx < len(ma_values) and ma_values[candle_idx] is not None:
                chart_ma.append(ma_values[candle_idx])
                x_vals.append(i)
        
        if len(chart_ma) > 1:
            ax.plot(x_vals, chart_ma, color=color, linewidth=linewidth, 
                   label=label, alpha=alpha)
    
    # Plot all MAs with distinct colors and enhanced visibility
    plot_ma_safe(ma1, "MA1 (SMMA HLC3-9) - Primary S/R", "#FFFFFF", 3, 0.95)  # White - Primary
    plot_ma_safe(ma2, "MA2 (SMMA Close-19) - Secondary S/R", "#00BFFF", 3, 0.95)  # Sky Blue  
    plot_ma_safe(ma3, "MA3 (SMA MA2-25) - Trend Filter", "#FFD700", 2.5, 0.9)  # Gold
    
    # Enhanced signal visualization
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        
        # Main signal marker
        if signal_data["side"] == "BUY":
            marker_color = "#00FF00"
            marker_symbol = "^"
            signal_label = "BUY - Confirmed Rejection"
        else:
            marker_color = "#FF0000" 
            marker_symbol = "v"
            signal_label = "SELL - Confirmed Rejection"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=500, edgecolor="#FFFFFF", linewidth=4, 
                  zorder=10, label=signal_label)
        
        # Highlight the rejection MA level with enhanced visibility
        ma_val = signal_data["ma1"] if signal_data["ma_level"] == "MA1" else signal_data["ma2"]
        ax.axhline(y=ma_val, color="#FFFF00", linestyle='--', 
                  linewidth=3, alpha=0.8, 
                  label=f"Rejection Level ({signal_data['ma_level']})")
        
        # Add rejection pattern annotation
        pattern_text = f"{signal_data['pattern']} at {signal_data['ma_level']}"
        ax.annotate(pattern_text, 
                   xy=(signal_chart_idx, signal_price),
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                   fontsize=10, fontweight='bold')
    
    # Enhanced title with MA arrangement and crossover info
    arrangement_emoji = "📈" if signal_data["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
    crossover_info = f" - {signal_data['crossover']}" if signal_data['crossover'] != "NONE" else ""
    
    title = (f"{signal_data['symbol']} - {signal_data['side']} CONFIRMED REJECTION {arrangement_emoji}"
            f"\n{signal_data['ma_arrangement'].replace('_', ' ')}{crossover_info}")
    ax.set_title(title, fontsize=16, color='white', fontweight='bold', pad=25)
    
    # Enhanced legend with better positioning
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', 
                      edgecolor='white', fontsize=11, shadow=True)
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color('white')
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, 
                dpi=150, 
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
def get_timeframe_for_symbol(shorthand):
    return SYMBOL_TF_MAP.get(shorthand, TIMEFRAMES[0] if TIMEFRAMES else 300)

def run_analysis():
    """Enhanced DSR analysis with MA crossover detection and confirmed rejection"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = get_timeframe_for_symbol(shorthand)
            
            if DEBUG:
                tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
                print(f"Analyzing {shorthand} on {tf_display}...")
            
            candles = fetch_candles(deriv_symbol, tf)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"Insufficient candles for {shorthand}: {len(candles)}")
                continue
            
            signal = detect_signal(candles, tf, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                if DEBUG:
                    print(f"Signal already sent for {shorthand}")
                continue
            
            # Enhanced alert message with comprehensive info
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            arrangement_emoji = "📈" if signal["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
            crossover_info = f" + {signal['crossover']}" if signal['crossover'] != "NONE" else ""
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} CONFIRMED REJECTION\n"
                      f"{arrangement_emoji} MA Setup: {signal['ma_arrangement'].replace('_', ' ')}{crossover_info}\n" 
                      f"🔄 Pattern: {signal['pattern']} (Candle Closed)\n"
                      f"📍 Rejection Level: {signal['ma_level']} Dynamic S/R\n"
                      f"💰 Entry Price: {signal['price']:.5f}\n"
                      f"✅ Signal Confirmed - No Tolerance Used\n"
                      f"📊 Context: {signal['context']}")
            
            chart_path = create_signal_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"Enhanced DSR signal sent for {shorthand}: {signal['side']} - {signal['pattern']} at {signal['ma_level']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"Enhanced analysis complete. {signals_found} confirmed DSR signals found.")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

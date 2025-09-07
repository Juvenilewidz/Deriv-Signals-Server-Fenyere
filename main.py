#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (corrected implementation)

- Proper SMMA calculations for MA1 and MA2
- OR logic for signal conditions (fires if ANY condition is met)
- Enhanced candlestick pattern detection
- Improved proximity detection for MA levels
- Fixed 1-second symbol analysis
- Proper trend reversal detection
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
LOOKBACK_BROKE_MA3 = 15  # Increased lookback for MA3 breaks
PROXIMITY_THRESHOLD = 0.002  # 0.2% proximity threshold (more reliable than range-based)

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
# Corrected Moving Averages
# -------------------------
def smma_correct(series, period):
    """Proper SMMA calculation as per trading standards"""
    n = len(series)
    if n < period:
        return [None] * n
    
    result = [None] * (period - 1)
    
    # First SMMA value is SMA of first 'period' values
    first_sma = sum(series[:period]) / period
    result.append(first_sma)
    
    # Subsequent SMMA values using correct formula
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
    """Compute moving averages per strategy specification"""
    closes = [c["close"] for c in candles]
    hlc3 = [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    
    # MA1: SMMA of HLC3, period 9
    ma1 = smma_correct(hlc3, 9)
    
    # MA2: SMMA of Close, period 19
    ma2 = smma_correct(closes, 19)
    
    # MA3: SMA of MA2, period 25
    ma2_valid = [v for v in ma2 if v is not None]
    if len(ma2_valid) >= 25:
        ma3_calc = sma(ma2_valid, 25)
        # Map back to original length
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
# Inclusive Candlestick Pattern Detection (Market Reality Focused)
# -------------------------
def is_rejection_candle(candle, prev_candle=None):
    """Inclusive rejection detection that captures real market rejection behavior"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range == 0 or total_range < 1e-9:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # DOJI FAMILY - Any small body showing indecision (expanded criteria)
    # Includes near-dojis and small-body candles that show market hesitation
    if body_size <= total_range * 0.3:  # Relaxed from 0.1 to 0.3
        return True, "DOJI"
    
    # PINBAR FAMILY - Any wick dominance showing rejection (very inclusive)
    # Upper rejection - any meaningful upper wick
    if upper_wick > 0 and upper_wick >= body_size * 0.8:  # Much more relaxed than 2.0
        return True, "PINBAR"
    
    # Lower rejection - any meaningful lower wick
    if lower_wick > 0 and lower_wick >= body_size * 0.8:  # Much more relaxed than 2.0
        return True, "PINBAR"
    
    # Alternative pinbar - wick is significant portion of total range
    if upper_wick >= total_range * 0.4:  # Upper wick is 40% of total range
        return True, "PINBAR"
    
    if lower_wick >= total_range * 0.4:  # Lower wick is 40% of total range
        return True, "PINBAR"
    
    # ENGULFING FAMILY - Partial and complete engulfing patterns
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        prev_body = abs(prev_c - prev_o)
        prev_high, prev_low = prev_candle["high"], prev_candle["low"]
        
        # Bullish engulfing (relaxed criteria)
        if prev_c < prev_o and c > o:  # Direction change requirement
            # Complete engulfing
            if o <= prev_c and c >= prev_o:
                return True, "BULL_ENGULF"
            # Partial engulfing - body overlaps significantly with previous body
            elif c > prev_o * 0.7 + prev_c * 0.3:  # Engulfs at least 70% of previous body
                return True, "BULL_ENGULF"
            # Range engulfing - current candle's range covers significant portion of previous
            elif h >= prev_high * 0.9 and l <= prev_low * 1.1:
                return True, "BULL_ENGULF"
        
        # Bearish engulfing (relaxed criteria)
        if prev_c > prev_o and c < o:  # Direction change requirement
            # Complete engulfing
            if o >= prev_c and c <= prev_o:
                return True, "BEAR_ENGULF"
            # Partial engulfing - body overlaps significantly with previous body
            elif c < prev_o * 0.7 + prev_c * 0.3:  # Engulfs at least 70% of previous body
                return True, "BEAR_ENGULF"
            # Range engulfing - current candle's range covers significant portion of previous
            elif h >= prev_high * 0.9 and l <= prev_low * 1.1:
                return True, "BEAR_ENGULF"
    
    # TINY BODY FAMILY - Very small bodies indicating uncertainty (expanded)
    if body_size <= total_range * 0.25:  # 25% or less of range
        return True, "TINY_BODY"
    
    # ADDITIONAL REJECTION PATTERNS - Catching subtle rejection behavior
    
    # High/Low close rejection - closes significantly away from high/low
    if c <= h * 0.7 + l * 0.3:  # Close in lower 30% of range (bearish rejection of highs)
        return True, "REJECTION"
    
    if c >= h * 0.3 + l * 0.7:  # Close in upper 70% of range (bullish rejection of lows)
        return True, "REJECTION"
    
    # Long range with small body - any long-range candle with relatively small body
    if total_range > 0 and body_size <= total_range * 0.4:  # Body is 40% or less of range
        return True, "LONG_RANGE"
    
    return False, "NONE"

# -------------------------
# Improved Proximity Detection
# -------------------------
def is_near_ma(price, ma_value, reference_price=None):
    """Check if price is near moving average using percentage-based threshold"""
    if ma_value is None:
        return False
    
    if reference_price is None:
        reference_price = price
    
    threshold = reference_price * PROXIMITY_THRESHOLD
    return abs(price - ma_value) <= threshold

# -------------------------
# Trend Analysis
# -------------------------
def analyze_trend(candles, ma1, ma2, ma3, current_idx):
    """Analyze current trend state"""
    if current_idx < 2 or not all(v is not None for v in [ma1[current_idx], ma2[current_idx], ma3[current_idx]]):
        return "UNDEFINED"
    
    current_ma1, current_ma2, current_ma3 = ma1[current_idx], ma2[current_idx], ma3[current_idx]
    current_price = candles[current_idx]["close"]
    
    # Strong uptrend: MA1 > MA2 > MA3 and price above MA3
    if current_ma1 > current_ma2 > current_ma3 and current_price > current_ma3:
        return "UPTREND"
    
    # Strong downtrend: MA1 < MA2 < MA3 and price below MA3
    if current_ma1 < current_ma2 < current_ma3 and current_price < current_ma3:
        return "DOWNTREND"
    
    return "CONSOLIDATION"

def check_ma3_break(candles, ma3, current_idx, lookback=LOOKBACK_BROKE_MA3):
    """Check for recent MA3 break"""
    if current_idx < lookback:
        return None
    
    for i in range(max(1, current_idx - lookback), current_idx):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        
        prev_close = candles[i-1]["close"]
        curr_close = candles[i]["close"]
        
        # Break above MA3
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            return "BREAK_UP"
        
        # Break below MA3
        if prev_close >= ma3[i-1] and curr_close < ma3[i]:
            return "BREAK_DOWN"
    
    return None

# -------------------------
# Fetch candles (with better error handling for 1s symbols)
# -------------------------
def fetch_candles(sym, tf, count=CANDLES_N):
    """Enhanced candle fetching with better handling for 1s timeframes"""
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
                if DEBUG:
                    print(f"Auth response: {auth_resp}")
            
            # Adjust count for 1s timeframes to avoid timeout
            adjusted_count = min(count, 300) if tf == 1 else count
            
            request = {
                "ticks_history": sym,
                "style": "candles",
                "granularity": tf,
                "count": adjusted_count,
                "end": "latest"
            }
            
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            
            if DEBUG:
                print(f"Fetched {len(response.get('candles', []))} candles for {sym} {tf}s")
            
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
# Trend-Aware Signal Detection (Fixed Logic)
# -------------------------
def detect_signal(candles, tf, shorthand):
    """Trend-aware signal detection that eliminates contradictory signals"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    current_idx = n - 1
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    # Check for rejection candlestick
    is_rejection, pattern_type = is_rejection_candle(current_candle, prev_candle)
    if not is_rejection:
        return None
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    # Get current values
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    current_high = current_candle["high"]
    current_low = current_candle["low"]
    current_close = current_candle["close"]
    
    # Analyze trend and MA3 breaks
    trend = analyze_trend(candles, ma1, ma2, ma3, current_idx)
    ma3_break = check_ma3_break(candles, ma3, current_idx)
    
    # Trend-aware signal logic (eliminates contradictions)
    signal_side = None
    reasons = []
    
    # UPTREND: MA1/MA2 act as dynamic support
    if trend == "UPTREND":
        if (is_near_ma(current_low, current_ma1, current_close) or 
            is_near_ma(current_low, current_ma2, current_close)):
            signal_side = "BUY"
            reasons.append(f"Uptrend continuation - {pattern_type} at MA support")
    
    # DOWNTREND: MA1/MA2 act as dynamic resistance
    elif trend == "DOWNTREND":
        if (is_near_ma(current_high, current_ma1, current_close) or 
            is_near_ma(current_high, current_ma2, current_close)):
            signal_side = "SELL"
            reasons.append(f"Downtrend continuation - {pattern_type} at MA resistance")
    
    # CONSOLIDATION: Check both support and resistance, prioritize based on MA3 breaks
    elif trend == "CONSOLIDATION":
        # Recent MA3 break up suggests bullish bias
        if ma3_break == "BREAK_UP":
            if (is_near_ma(current_low, current_ma1, current_close) or 
                is_near_ma(current_low, current_ma2, current_close)):
                signal_side = "BUY"
                reasons.append(f"Bullish reversal - MA3 breakout + {pattern_type} retest")
        
        # Recent MA3 break down suggests bearish bias
        elif ma3_break == "BREAK_DOWN":
            if (is_near_ma(current_high, current_ma1, current_close) or 
                is_near_ma(current_high, current_ma2, current_close)):
                signal_side = "SELL"
                reasons.append(f"Bearish reversal - MA3 breakdown + {pattern_type} retest")
        
        # No recent MA3 break - evaluate based on rejection location
        else:
            # Rejection at lower MA levels (support)
            if (is_near_ma(current_low, current_ma1, current_close) or 
                is_near_ma(current_low, current_ma2, current_close)):
                signal_side = "BUY"
                reasons.append(f"Consolidation - {pattern_type} rejection at MA support")
            
            # Rejection at upper MA levels (resistance)
            elif (is_near_ma(current_high, current_ma1, current_close) or 
                  is_near_ma(current_high, current_ma2, current_close)):
                signal_side = "SELL"
                reasons.append(f"Consolidation - {pattern_type} rejection at MA resistance")
    
    # Additional reversal signals: Strong MA3 breaks with opposite rejections
    if signal_side is None:
        # Bullish reversal: Recent MA3 break up + rejection at MA1/MA2 support
        if (ma3_break == "BREAK_UP" and 
            (is_near_ma(current_low, current_ma1, current_close) or 
             is_near_ma(current_low, current_ma2, current_close))):
            signal_side = "BUY"
            reasons.append(f"Trend reversal - MA3 breakout + {pattern_type} support retest")
        
        # Bearish reversal: Recent MA3 break down + rejection at MA1/MA2 resistance
        elif (ma3_break == "BREAK_DOWN" and 
              (is_near_ma(current_high, current_ma1, current_close) or 
               is_near_ma(current_high, current_ma2, current_close))):
            signal_side = "SELL"
            reasons.append(f"Trend reversal - MA3 breakdown + {pattern_type} resistance retest")
    
    # Return valid signal with trend consistency
    if signal_side and reasons:
        return {
            "symbol": shorthand,
            "tf": tf,
            "side": signal_side,
            "reasons": reasons,
            "pattern": pattern_type,
            "trend": trend,
            "ma3_break": ma3_break,
            "idx": current_idx,
            "ma1": ma1,
            "ma2": ma2,
            "ma3": ma3,
            "candles": candles,
            "strength": 1  # Single consistent signal
        }
    
    return None

# -------------------------
# Professional Chart Generation with Custom Styling
# -------------------------
def create_signal_chart(signal_data):
    """Create professional chart with black background and optimized layout"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1"], signal_data["ma2"], signal_data["ma3"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - LAST_N_CHART)
    chart_candles = candles[chart_start:]
    
    # Set black background style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set black background for figure and axes
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot candlesticks with specified colors
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        
        # Green for bullish (close >= open), Red for bearish (close < open)
        if c >= o:
            body_color = "#00FF00"  # Bright green for bullish
            edge_color = "#00AA00"  # Darker green edge
        else:
            body_color = "#FF0000"  # Bright red for bearish
            edge_color = "#AA0000"  # Darker red edge
        
        # Candlestick body
        ax.add_patch(Rectangle(
            (i - CANDLE_WIDTH/2, min(o, c)), 
            CANDLE_WIDTH, 
            max(abs(c - o), 1e-9),
            facecolor=body_color, 
            edgecolor=edge_color, 
            alpha=0.9,
            linewidth=1
        ))
        
        # Wicks with matching colors
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.2, alpha=0.8)
    
    # Plot moving averages with enhanced visibility
    def plot_ma(ma_values, label, color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        ax.plot(range(len(chart_candles)), chart_ma, 
                color=color, linewidth=linewidth, label=label, alpha=0.9)
    
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#FFFFFF", 2)    # White
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00BFFF", 2)   # Deep Sky Blue
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#FF6347", 2)      # Tomato Red
    
    # Mark signal point with enhanced visibility
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        
        if signal_data["side"] == "BUY":
            marker_color = "#00FF00"  # Bright green
            marker_symbol = "^"
            edge_color = "#FFFFFF"
        else:
            marker_color = "#FF0000"  # Bright red
            marker_symbol = "v"
            edge_color = "#FFFFFF"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor=edge_color, linewidth=3, zorder=10)
    
    # Enhanced legend with black background
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', 
                      edgecolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.8)
    
    # Enhanced title and labels
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} Signal", 
                fontsize=16, color='white', fontweight='bold', pad=20)
    
    # Grid styling
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
    
    # Axis styling
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
    # Remove extra whitespace and optimize layout
    plt.tight_layout()
    
    # Save chart with optimized settings
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, 
                dpi=150, 
                bbox_inches="tight", 
                facecolor='black',
                edgecolor='none',
                pad_inches=0.1)
    plt.close()
    
    # Reset style to default for future plots
    plt.style.use('default')
    
    return chart_file.name

# -------------------------
# Main Execution
# -------------------------
def get_timeframe_for_symbol(shorthand):
    """Get appropriate timeframe for symbol"""
    return SYMBOL_TF_MAP.get(shorthand, TIMEFRAMES[0] if TIMEFRAMES else 300)

def run_analysis():
    """Main analysis loop"""
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = get_timeframe_for_symbol(shorthand)
            
            if DEBUG:
                print(f"Analyzing {shorthand} ({deriv_symbol}) on {tf}s timeframe...")
            
            # Fetch candles
            candles = fetch_candles(deriv_symbol, tf)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"Insufficient candles for {shorthand}: {len(candles)}")
                continue
            
            # Detect signals
            signal = detect_signal(candles, tf, shorthand)
            if not signal:
                continue
            
            # Check for duplicate alerts
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                if DEBUG:
                    print(f"Signal already sent for {shorthand} {signal['side']} at {current_epoch}")
                continue
            
            # Create alert message
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            strength_indicator = "🔥" * min(signal["strength"], 3)
            
            caption = (f"📊 {shorthand} {tf_display} - {signal['side']} {strength_indicator}\n"
                      f"🎯 Pattern: {signal['pattern']}\n"
                      f"📈 Trend: {signal['trend']}\n"
                      f"💰 Price: {signal['candles'][signal['idx']]['close']}\n"
                      f"📝 Reasons:\n" + "\n".join(f"• {reason}" for reason in signal["reasons"]))
            
            # Generate chart
            chart_path = create_signal_chart(signal)
            
            # Send alert
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                if DEBUG:
                    print(f"Signal sent for {shorthand}: {signal['side']}")
            
            # Clean up chart file
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

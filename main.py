#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (Perfected Implementation)

- Enhanced trend analysis with proper consolidation detection
- Zone-based pattern detection (patterns only matter at rejection zones)
- Proper confirmation sequencing for trend changes
- Price positioning validation for signal direction
- Reduced signal spam through stringent quality filters
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
MIN_CANDLES = 120
LOOKBACK_BROKE_MA3 = 20
MA3_CONFIRMATION_PERIOD = 8  # Candles to wait after MA3 break before reversal signals

# Enhanced proximity thresholds for zone-based detection
PRIMARY_ZONE_THRESHOLD = 0.0015    # 0.15% - direct MA interaction
SECONDARY_ZONE_THRESHOLD = 0.004   # 0.4% - extended retracement zone
TREND_STRENGTH_THRESHOLD = 0.003   # 0.3% - MA separation for trend confirmation

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
    """Compute moving averages per strategy specification"""
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

# -------------------------
# Zone-Based Pattern Detection (Only at Rejection Zones)
# -------------------------
def is_rejection_candle(candle, prev_candle=None):
    """Rejection pattern detection - only used when price is in rejection zones"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range == 0 or total_range < 1e-9:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # DOJI FAMILY - indecision at key levels
    if body_size <= total_range * 0.25:
        return True, "DOJI"
    
    # PINBAR FAMILY - rejection wicks at key levels
    if upper_wick >= body_size * 1.2 and upper_wick >= total_range * 0.35:
        return True, "PINBAR"
    
    if lower_wick >= body_size * 1.2 and lower_wick >= total_range * 0.35:
        return True, "PINBAR"
    
    # ENGULFING FAMILY - only at key levels
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        prev_body = abs(prev_c - prev_o)
        
        # Bullish engulfing
        if (prev_c < prev_o and c > o and 
            o <= prev_c and c >= prev_o and 
            body_size > prev_body * 1.1):
            return True, "BULL_ENGULF"
        
        # Bearish engulfing
        if (prev_c > prev_o and c < o and 
            o >= prev_c and c <= prev_o and 
            body_size > prev_body * 1.1):
            return True, "BEAR_ENGULF"
    
    # High/Low rejection - close away from extremes at key levels
    if c <= h * 0.6 + l * 0.4:  # Close in lower 40%
        return True, "HIGH_REJECTION"
    
    if c >= h * 0.4 + l * 0.6:  # Close in upper 60%
        return True, "LOW_REJECTION"
    
    return False, "NONE"

# -------------------------
# Enhanced Trend Analysis with Consolidation Detection
# -------------------------
def calculate_ma_separation(ma1, ma2, ma3, idx):
    """Calculate separation between moving averages"""
    if not all(v is not None for v in [ma1[idx], ma2[idx], ma3[idx]]):
        return 0
    
    avg_price = (ma1[idx] + ma2[idx] + ma3[idx]) / 3
    ma1_sep = abs(ma1[idx] - ma2[idx]) / avg_price if avg_price > 0 else 0
    ma2_sep = abs(ma2[idx] - ma3[idx]) / avg_price if avg_price > 0 else 0
    
    return (ma1_sep + ma2_sep) / 2

def analyze_trend_strength(candles, ma1, ma2, ma3, current_idx, lookback=10):
    """Comprehensive trend analysis with consolidation detection"""
    if current_idx < lookback:
        return "UNDEFINED", 0
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx] 
    current_ma3 = ma3[current_idx]
    current_price = candles[current_idx]["close"]
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return "UNDEFINED", 0
    
    # Calculate moving average separation (trend strength indicator)
    ma_separation = calculate_ma_separation(ma1, ma2, ma3, current_idx)
    
    # Direction consistency check
    bullish_alignment = current_ma1 > current_ma2 > current_ma3
    bearish_alignment = current_ma1 < current_ma2 < current_ma3
    
    # Price positioning relative to MA structure
    price_above_all = current_price > max(current_ma1, current_ma2, current_ma3)
    price_below_all = current_price < min(current_ma1, current_ma2, current_ma3)
    
    # Trend consistency over lookback period
    trend_consistency = 0
    for i in range(max(0, current_idx - lookback), current_idx):
        if all(v is not None for v in [ma1[i], ma2[i], ma3[i]]):
            if bullish_alignment and ma1[i] > ma2[i] > ma3[i]:
                trend_consistency += 1
            elif bearish_alignment and ma1[i] < ma2[i] < ma3[i]:
                trend_consistency += 1
    
    trend_consistency_ratio = trend_consistency / lookback
    
    # Determine trend state
    if (bullish_alignment and price_above_all and 
        ma_separation > TREND_STRENGTH_THRESHOLD and 
        trend_consistency_ratio > 0.7):
        return "STRONG_UPTREND", ma_separation
    
    elif (bearish_alignment and price_below_all and 
          ma_separation > TREND_STRENGTH_THRESHOLD and 
          trend_consistency_ratio > 0.7):
        return "STRONG_DOWNTREND", ma_separation
    
    elif (bullish_alignment and current_price > current_ma3 and 
          ma_separation > TREND_STRENGTH_THRESHOLD * 0.5):
        return "UPTREND", ma_separation
    
    elif (bearish_alignment and current_price < current_ma3 and 
          ma_separation > TREND_STRENGTH_THRESHOLD * 0.5):
        return "DOWNTREND", ma_separation
    
    # If MAs are converging or overlapping = consolidation
    elif ma_separation < TREND_STRENGTH_THRESHOLD * 0.5:
        return "CONSOLIDATION", ma_separation
    
    else:
        return "TRANSITION", ma_separation

# -------------------------
# MA3 Break Detection with Confirmation Logic
# -------------------------
def check_ma3_break_with_confirmation(candles, ma3, current_idx, lookback=LOOKBACK_BROKE_MA3):
    """Enhanced MA3 break detection with confirmation requirements"""
    if current_idx < lookback:
        return None, 0
    
    break_candle_idx = None
    break_direction = None
    
    # Look for MA3 break
    for i in range(max(1, current_idx - lookback), current_idx - MA3_CONFIRMATION_PERIOD + 1):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        
        prev_close = candles[i-1]["close"]
        curr_close = candles[i]["close"]
        
        # Break above MA3
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            break_candle_idx = i
            break_direction = "BREAK_UP"
            break
        
        # Break below MA3  
        if prev_close >= ma3[i-1] and curr_close < ma3[i]:
            break_candle_idx = i
            break_direction = "BREAK_DOWN"
            break
    
    if break_candle_idx is None:
        return None, 0
    
    # Calculate candles since break
    candles_since_break = current_idx - break_candle_idx
    
    return break_direction, candles_since_break

# -------------------------
# Zone Detection for Pattern Analysis
# -------------------------
def is_in_rejection_zone(price, ma1_val, ma2_val, reference_price, trend_state):
    """Determine if price is in a rejection zone around MAs"""
    if ma1_val is None or ma2_val is None:
        return False, "NONE"
    
    # Primary zone - direct MA interaction
    primary_threshold = reference_price * PRIMARY_ZONE_THRESHOLD
    
    if (abs(price - ma1_val) <= primary_threshold or 
        abs(price - ma2_val) <= primary_threshold):
        return True, "PRIMARY"
    
    # Secondary zone - extended retracement zone (only in trending markets)
    if "TREND" in trend_state:
        secondary_threshold = reference_price * SECONDARY_ZONE_THRESHOLD
        
        if (abs(price - ma1_val) <= secondary_threshold or 
            abs(price - ma2_val) <= secondary_threshold):
            return True, "SECONDARY"
    
    return False, "NONE"

# -------------------------
# Enhanced Data Fetching
# -------------------------
def fetch_candles(sym, tf, count=CANDLES_N):
    """Enhanced candle fetching with better error handling"""
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
# Master Signal Detection with All Corrections
# -------------------------
def detect_signal(candles, tf, shorthand):
    """Master signal detection with all problem corrections applied"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    current_idx = n - 1
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    current_high = current_candle["high"]
    current_low = current_candle["low"]
    current_close = current_candle["close"]
    
    # Enhanced trend analysis
    trend_state, trend_strength = analyze_trend_strength(candles, ma1, ma2, ma3, current_idx)
    
    # MA3 break analysis with confirmation
    ma3_break, candles_since_break = check_ma3_break_with_confirmation(candles, ma3, current_idx)
    
    # ZONE-BASED PATTERN DETECTION: Only check patterns if price is in rejection zones
    in_zone_high, zone_type_high = is_in_rejection_zone(current_high, current_ma1, current_ma2, current_close, trend_state)
    in_zone_low, zone_type_low = is_in_rejection_zone(current_low, current_ma1, current_ma2, current_close, trend_state)
    
    if not (in_zone_high or in_zone_low):
        return None  # No pattern analysis if not in rejection zones
    
    # Pattern detection (only executed if in zones)
    is_rejection, pattern_type = is_rejection_candle(current_candle, prev_candle)
    if not is_rejection:
        return None
    
    # SIGNAL LOGIC WITH ALL CORRECTIONS APPLIED
    signal_side = None
    reasons = []
    signal_quality = 0
    
    # 1. TRENDING MARKET CONTINUATION SIGNALS
    if trend_state in ["STRONG_UPTREND", "UPTREND"]:
        # BUY signals: Price above MA structure + rejection at support
        if (current_close > current_ma3 and 
            current_close > min(current_ma1, current_ma2) and  # Price positioning validation
            in_zone_low):
            signal_side = "BUY"
            reasons.append(f"{trend_state.replace('_', ' ').title()} continuation - {pattern_type} at MA support")
            signal_quality = 3 if trend_state == "STRONG_UPTREND" else 2
    
    elif trend_state in ["STRONG_DOWNTREND", "DOWNTREND"]:
        # SELL signals: Price below MA structure + rejection at resistance
        if (current_close < current_ma3 and 
            current_close < max(current_ma1, current_ma2) and  # Price positioning validation
            in_zone_high):
            signal_side = "SELL"
            reasons.append(f"{trend_state.replace('_', ' ').title()} continuation - {pattern_type} at MA resistance")
            signal_quality = 3 if trend_state == "STRONG_DOWNTREND" else 2
    
    # 2. TREND REVERSAL SIGNALS (with proper confirmation)
    elif (ma3_break and candles_since_break >= MA3_CONFIRMATION_PERIOD):
        if (ma3_break == "BREAK_UP" and 
            current_close > current_ma3 and  # Price positioning confirmation
            in_zone_low):
            signal_side = "BUY"
            reasons.append(f"Trend reversal confirmed - MA3 breakout + {pattern_type} retest (after {candles_since_break} candles)")
            signal_quality = 4  # High quality reversal signal
        
        elif (ma3_break == "BREAK_DOWN" and 
              current_close < current_ma3 and  # Price positioning confirmation
              in_zone_high):
            signal_side = "SELL" 
            reasons.append(f"Trend reversal confirmed - MA3 breakdown + {pattern_type} retest (after {candles_since_break} candles)")
            signal_quality = 4  # High quality reversal signal
    
    # 3. CONSOLIDATION SIGNALS (limited and high-quality only)
    elif trend_state == "CONSOLIDATION" and zone_type_high == "PRIMARY":
        # Only primary zone signals in consolidation
        if in_zone_low:
            signal_side = "BUY"
            reasons.append(f"Consolidation range - {pattern_type} at MA support")
            signal_quality = 1
        elif in_zone_high:
            signal_side = "SELL"
            reasons.append(f"Consolidation range - {pattern_type} at MA resistance")
            signal_quality = 1
    
    # QUALITY FILTER: Only return high-quality signals
    if signal_side and signal_quality >= 2:  # Minimum quality threshold
        return {
            "symbol": shorthand,
            "tf": tf,
            "side": signal_side,
            "reasons": reasons,
            "pattern": pattern_type,
            "trend": trend_state,
            "trend_strength": trend_strength,
            "ma3_break": ma3_break,
            "candles_since_break": candles_since_break,
            "zone_type": zone_type_high if in_zone_high else zone_type_low,
            "quality": signal_quality,
            "idx": current_idx,
            "ma1": ma1,
            "ma2": ma2,
            "ma3": ma3,
            "candles": candles
        }
    
    return None

# -------------------------
# Enhanced Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    """Enhanced chart with trend state visualization"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1"], signal_data["ma2"], signal_data["ma3"]
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
        
        if signal_data["side"] == "BUY":
            marker_color = "#00FF00"
            marker_symbol = "^"
        else:
            marker_color = "#FF0000"
            marker_symbol = "v"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    
    # Enhanced title with trend state
    quality_stars = "⭐" * signal_data["quality"]
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} Signal {quality_stars}", 
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
    plt.savefig(chart_file.name, 
                dpi=150, 
                bbox_inches="tight", 
                facecolor='black',
                edgecolor='none',
                pad_inches=0.1)
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

# -------------------------
# Main Execution
# -------------------------
def get_timeframe_for_symbol(shorthand):
    return SYMBOL_TF_MAP.get(shorthand, TIMEFRAMES[0] if TIMEFRAMES else 300)

def run_analysis():
    """Main analysis with enhanced signal filtering"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        # Skip 1s symbols in main bot
        if shorthand.endswith("(1s)"):
            continue
            
        try:
            tf = get_timeframe_for_symbol(shorthand)
            
            if DEBUG:
                print(f"Analyzing {shorthand} ({deriv_symbol}) on {tf}s timeframe...")
            
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
            
            # Enhanced alert message
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            quality_indicator = "🔥" * signal["quality"]
            trend_emoji = "📈" if "UP" in signal["trend"] else "📉" if "DOWN" in signal["trend"] else "↔️"
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} {quality_indicator}\n"
                      f"{trend_emoji} Trend: {signal['trend'].replace('_', ' ').title()}\n"
                      f"🎨 Pattern: {signal['pattern']}\n"
                      f"📍 Zone: {signal['zone_type']} rejection zone\n"
                      f"💰 Price: {signal['candles'][signal['idx']]['close']}\n")
            
            if signal.get("candles_since_break", 0) > 0:
                caption += f"⏱️ Confirmation: {signal['candles_since_break']} candles since MA3 break\n"
            
            caption += f"📝 Analysis:\n" + "\n".join(f"• {reason}" for reason in signal["reasons"])
            
            chart_path = create_signal_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"Quality signal sent for {shorthand}: {signal['side']} (Quality: {signal['quality']})")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
    
    if DEBUG:
        print(f"Analysis complete. {signals_found} quality signals found.")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

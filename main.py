#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (FINAL IMPLEMENTATION)

ABSOLUTE RULES (NO EXCEPTIONS):
1. CONSOLIDATION = IMMEDIATE REJECTION (No signals ever)
2. ALL candlesticks must be FULLY CLOSED (use n-2 index)
3. MA3 breaks require PHYSICAL RETEST + REJECTION (not time-based)
4. Pullbacks must show CLEAR SWING MOVEMENT (not just proximity)
5. Price positioning must be VALIDATED before any signal
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

CANDLES_N = 500
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

# STRICT STRATEGY PARAMETERS
MA_SEPARATION_MINIMUM = 0.008      # 0.8% minimum separation for valid trend
SWING_MINIMUM = 0.015              # 1.5% minimum swing for pullback validation  
RETEST_PRECISION = 0.003           # 0.3% precision for MA retest detection
TREND_STRENGTH_PERIODS = 20        # Periods to validate trend consistency
CONSOLIDATION_THRESHOLD = 0.005    # 0.5% MA convergence = consolidation

SYMBOL_MAP = {"V10": "R_10", "V50": "R_50", "V75": "R_75"}

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

# IRONCLAD CONSOLIDATION DETECTION
def is_consolidation_market(candles, ma1, ma2, ma3, current_idx):
    """ABSOLUTE consolidation detection - any consolidation = NO SIGNALS"""
    if current_idx < TREND_STRENGTH_PERIODS:
        return True  # Not enough data = assume consolidation
    
    # Check MA separation over recent periods
    for i in range(current_idx - TREND_STRENGTH_PERIODS + 1, current_idx + 1):
        if not all(v is not None for v in [ma1[i], ma2[i], ma3[i]]):
            continue
        
        avg_price = (ma1[i] + ma2[i] + ma3[i]) / 3
        if avg_price <= 0:
            continue
            
        # Calculate MA separation
        sep1_2 = abs(ma1[i] - ma2[i]) / avg_price
        sep2_3 = abs(ma2[i] - ma3[i]) / avg_price
        avg_separation = (sep1_2 + sep2_3) / 2
        
        # If ANY period shows consolidation characteristics, reject entire setup
        if avg_separation < CONSOLIDATION_THRESHOLD:
            return True
            
        # Check for MA crossovers (indicating consolidation)
        if i > 0:
            prev_ma1, prev_ma2, prev_ma3 = ma1[i-1], ma2[i-1], ma3[i-1]
            if all(v is not None for v in [prev_ma1, prev_ma2, prev_ma3]):
                # MA1 and MA2 crossing = consolidation
                if ((ma1[i] > ma2[i] and prev_ma1 <= prev_ma2) or 
                    (ma1[i] < ma2[i] and prev_ma1 >= prev_ma2)):
                    return True
    
    return False

# STRICT TREND VALIDATION
def validate_trending_market(candles, ma1, ma2, ma3, current_idx):
    """Only allow clear trending markets with proper MA alignment"""
    if current_idx < TREND_STRENGTH_PERIODS:
        return None, 0
    
    # FIRST: Check for consolidation (immediate rejection)
    if is_consolidation_market(candles, ma1, ma2, ma3, current_idx):
        return None, 0
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx] 
    current_ma3 = ma3[current_idx]
    current_price = candles[current_idx]["close"]
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None, 0
    
    # Check MA alignment
    bullish_alignment = current_ma1 > current_ma2 > current_ma3
    bearish_alignment = current_ma1 < current_ma2 < current_ma3
    
    if not (bullish_alignment or bearish_alignment):
        return None, 0
    
    # Validate MA separation
    avg_price = (current_ma1 + current_ma2 + current_ma3) / 3
    separation = abs(current_ma1 - current_ma3) / avg_price
    
    if separation < MA_SEPARATION_MINIMUM:
        return None, 0
    
    # Validate price positioning
    if bullish_alignment and current_price <= current_ma3:
        return None, 0
    if bearish_alignment and current_price >= current_ma3:
        return None, 0
    
    # Check trend consistency
    consistent_periods = 0
    for i in range(max(0, current_idx - TREND_STRENGTH_PERIODS), current_idx):
        if all(v is not None for v in [ma1[i], ma2[i], ma3[i]]):
            if bullish_alignment and ma1[i] > ma2[i] > ma3[i]:
                consistent_periods += 1
            elif bearish_alignment and ma1[i] < ma2[i] < ma3[i]:
                consistent_periods += 1
    
    consistency_ratio = consistent_periods / TREND_STRENGTH_PERIODS
    if consistency_ratio < 0.85:  # 85% consistency required
        return None, 0
    
    trend_state = "UPTREND" if bullish_alignment else "DOWNTREND"
    return trend_state, separation

# PHYSICAL SWING DETECTION
def detect_swing_movement(candles, start_idx, end_idx, direction):
    """Detect actual swing movement between two points"""
    if end_idx <= start_idx or start_idx < 0:
        return False, 0
    
    start_price = candles[start_idx]["close"]
    end_price = candles[end_idx]["close"]
    
    if direction == "UP":
        # For upward swing, find the highest point
        highest = max(candles[i]["high"] for i in range(start_idx, end_idx + 1))
        swing_distance = (highest - start_price) / start_price
        return swing_distance >= SWING_MINIMUM, swing_distance
    else:
        # For downward swing, find the lowest point  
        lowest = min(candles[i]["low"] for i in range(start_idx, end_idx + 1))
        swing_distance = (start_price - lowest) / start_price
        return swing_distance >= SWING_MINIMUM, swing_distance

# PHYSICAL RETEST VALIDATION
def validate_ma3_break_with_physical_retest(candles, ma1, ma2, ma3, current_idx):
    """Validate MA3 break followed by PHYSICAL retest to MA1/MA2"""
    if current_idx < 30:
        return None
    
    # Find MA3 break
    break_info = None
    for i in range(max(1, current_idx - 25), current_idx - 10):
        if ma3[i] is None or ma3[i-1] is None:
            continue
        
        prev_close = candles[i-1]["close"]
        curr_close = candles[i]["close"]
        
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            break_info = {"idx": i, "direction": "UP"}
            break
        elif prev_close >= ma3[i-1] and curr_close < ma3[i]:
            break_info = {"idx": i, "direction": "DOWN"}
            break
    
    if not break_info:
        return None
    
    break_idx = break_info["idx"]
    break_direction = break_info["direction"]
    
    # Validate swing movement after break
    swing_valid, swing_distance = detect_swing_movement(
        candles, break_idx, break_idx + 5, break_direction
    )
    if not swing_valid:
        return None
    
    # Find PHYSICAL retest to MA1 or MA2
    for i in range(break_idx + 6, current_idx + 1):
        if not all(v is not None for v in [ma1[i], ma2[i]]):
            continue
        
        candle = candles[i]
        
        if break_direction == "UP":
            # Look for pullback that physically touches MA1 or MA2
            ma1_distance = abs(candle["low"] - ma1[i]) / candle["close"]
            ma2_distance = abs(candle["low"] - ma2[i]) / candle["close"]
            
            if ma1_distance <= RETEST_PRECISION or ma2_distance <= RETEST_PRECISION:
                # Validate this is a proper pullback (not just random touch)
                pullback_valid, _ = detect_swing_movement(candles, break_idx, i, "DOWN")
                if pullback_valid:
                    return {
                        "break_idx": break_idx,
                        "break_direction": break_direction,
                        "retest_idx": i,
                        "retest_type": "MA_PULLBACK"
                    }
        else:
            # Look for bounce that physically touches MA1 or MA2
            ma1_distance = abs(candle["high"] - ma1[i]) / candle["close"]
            ma2_distance = abs(candle["high"] - ma2[i]) / candle["close"]
            
            if ma1_distance <= RETEST_PRECISION or ma2_distance <= RETEST_PRECISION:
                # Validate this is a proper bounce (not just random touch)
                bounce_valid, _ = detect_swing_movement(candles, break_idx, i, "UP")
                if bounce_valid:
                    return {
                        "break_idx": break_idx,
                        "break_direction": break_direction,
                        "retest_idx": i,
                        "retest_type": "MA_BOUNCE"
                    }
    
    return None

# REJECTION PATTERN AT RETEST ONLY
def validate_rejection_at_retest(candle, prev_candle, ma1_val, ma2_val):
    """Only validate rejection patterns when price is physically at MA levels"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    # Verify physical interaction with MA levels
    ma1_touched = abs(l - ma1_val) <= c * RETEST_PRECISION or abs(h - ma1_val) <= c * RETEST_PRECISION
    ma2_touched = abs(l - ma2_val) <= c * RETEST_PRECISION or abs(h - ma2_val) <= c * RETEST_PRECISION
    
    if not (ma1_touched or ma2_touched):
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # Strong rejection patterns only
    if body_size <= total_range * 0.15:  # Very small body
        return True, "DOJI"
    
    if upper_wick >= total_range * 0.5:  # Upper rejection
        return True, "UPPER_REJECTION"
    
    if lower_wick >= total_range * 0.5:  # Lower rejection
        return True, "LOWER_REJECTION"
    
    # Engulfing at MA levels
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        if (prev_c < prev_o and c > o and o <= prev_c and c >= prev_o):
            return True, "BULL_ENGULF"
        if (prev_c > prev_o and c < o and o >= prev_c and c <= prev_o):
            return True, "BEAR_ENGULF"
    
    return False, "NONE"

# DATA FETCHING (COMPLETED CANDLES ONLY)
def fetch_completed_candles_only(sym, tf, count=CANDLES_N):
    """Fetch candles ensuring we only work with completed ones"""
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
                "count": count + 3,  # Extra candles for safety
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
                
                # Remove last 2 candles (current forming + safety buffer)
                completed_candles = candles_data[:-2]
                
                if DEBUG:
                    print(f"Using {len(completed_candles)} completed candles for {sym}")
                
                return completed_candles
        
        except Exception as e:
            if DEBUG:
                print(f"Attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(1)
    
    return []

# MASTER SIGNAL DETECTION (IRONCLAD RULES)
def detect_signal_ironclad(candles, tf, shorthand):
    """Ironclad signal detection with absolute rule enforcement"""
    n = len(candles)
    if n < 200:  # Need sufficient history
        return None
    
    # Use fully completed candle (n-1 is the last completed candle)
    current_idx = n - 1
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    # RULE 1: ABSOLUTE CONSOLIDATION REJECTION
    trend_state, trend_strength = validate_trending_market(candles, ma1, ma2, ma3, current_idx)
    if trend_state is None:
        if DEBUG:
            print(f"{shorthand}: REJECTED - Consolidation/Invalid market state")
        return None
    
    # RULE 2: VALIDATE PHYSICAL MA3 BREAK + RETEST
    retest_validation = validate_ma3_break_with_physical_retest(candles, ma1, ma2, ma3, current_idx)
    if not retest_validation:
        if DEBUG:
            print(f"{shorthand}: REJECTED - No valid MA3 break + retest")
        return None
    
    # RULE 3: REJECTION PATTERN AT RETEST LOCATION
    is_rejection, pattern_type = validate_rejection_at_retest(
        current_candle, prev_candle, current_ma1, current_ma2
    )
    if not is_rejection:
        if DEBUG:
            print(f"{shorthand}: REJECTED - No rejection pattern at retest")
        return None
    
    # RULE 4: FINAL PRICE POSITIONING VALIDATION
    current_close = current_candle["close"]
    retest_direction = retest_validation["break_direction"]
    
    if retest_direction == "UP" and current_close <= current_ma3:
        if DEBUG:
            print(f"{shorthand}: REJECTED - Price below MA3 for bullish setup")
        return None
    
    if retest_direction == "DOWN" and current_close >= current_ma3:
        if DEBUG:
            print(f"{shorthand}: REJECTED - Price above MA3 for bearish setup")
        return None
    
    # GENERATE VALIDATED SIGNAL
    signal_side = "BUY" if retest_direction == "UP" else "SELL"
    
    return {
        "symbol": shorthand,
        "tf": tf,
        "side": signal_side,
        "pattern": pattern_type,
        "trend": trend_state,
        "trend_strength": trend_strength,
        "retest_validation": retest_validation,
        "idx": current_idx,
        "ma1": ma1,
        "ma2": ma2,
        "ma3": ma3,
        "candles": candles,
        "validation_status": "IRONCLAD_VALIDATED"
    }

# CHART GENERATION
def create_validated_chart(signal_data):
    """Chart generation for validated signals only"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1"], signal_data["ma2"], signal_data["ma3"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - 180)
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        body_color = "#00FF00" if c >= o else "#FF0000"
        edge_color = "#00AA00" if c >= o else "#AA0000"
        
        ax.add_patch(Rectangle(
            (i - 0.35/2, min(o, c)), 0.35, max(abs(c - o), 1e-9),
            facecolor=body_color, edgecolor=edge_color, alpha=0.9, linewidth=1
        ))
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.2, alpha=0.8)
    
    # Moving averages
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
    
    # Signal marker
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        marker_color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker_symbol = "^" if signal_data["side"] == "BUY" else "v"
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} ✓ IRONCLAD VALIDATED", 
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
def run_ironclad_analysis():
    """Main analysis with ironclad rule enforcement"""
    validated_signals = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = TIMEFRAMES[0] if TIMEFRAMES else 300
            
            if DEBUG:
                print(f"IRONCLAD analysis for {shorthand}...")
            
            candles = fetch_completed_candles_only(deriv_symbol, tf)
            if len(candles) < 200:
                if DEBUG:
                    print(f"{shorthand}: Insufficient data")
                continue
            
            signal = detect_signal_ironclad(candles, tf, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                continue
            
            # Create alert
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            retest_info = signal["retest_validation"]
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} ✅ VALIDATED\n"
                      f"📊 Trend: {signal['trend']}\n"
                      f"🎨 Pattern: {signal['pattern']}\n"
                      f"💰 Price: {signal['candles'][signal['idx']]['close']}\n"
                      f"🔄 MA3 Break: {retest_info['break_direction']} at candle {retest_info['break_idx']}\n"
                      f"✅ Physical Retest: {retest_info['retest_type']} at candle {retest_info['retest_idx']}\n"
                      f"📝 Status: IRONCLAD VALIDATION PASSED\n"
                      f"⚡ All rules enforced: Trending market + Physical retest + Rejection pattern")
            
            chart_path = create_validated_chart(signal)
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                validated_signals += 1
                if DEBUG:
                    print(f"✅ IRONCLAD VALIDATED signal sent: {shorthand} {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error in {shorthand}: {e}")
    
    if DEBUG:
        print(f"Analysis complete. {validated_signals} ironclad validated signals found.")

if __name__ == "__main__":
    try:
        run_ironclad_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

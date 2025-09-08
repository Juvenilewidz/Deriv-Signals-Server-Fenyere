#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (Rebuilt with Strict Strategy Adherence)

CORE PRINCIPLES:
1. NO SIGNALS in consolidation markets (consolidation = NO GO ZONE)
2. Breakouts MUST have retest + rejection candlestick (not just time-based)
3. Only work with FULLY CLOSED candlesticks
4. Patterns only matter at MA rejection zones with proper pullback/retest
5. Strict trend validation required before any signal generation
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
import websocket, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Telegram helpers
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

CANDLES_N = 500
LAST_N_CHART = 180
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")
MIN_CANDLES = 150

# Strategy-specific thresholds
MA_SEPARATION_MIN = 0.004          # Minimum MA separation for valid trend (0.4%)
RETEST_ZONE_THRESHOLD = 0.002      # 0.2% proximity for MA retest detection
PULLBACK_MIN_DISTANCE = 0.008      # 0.8% minimum pullback distance
TREND_CONSISTENCY_MIN = 0.8        # 80% directional consistency required

# -------------------------
# Symbol mapping (excluding 1s for main bot)
# -------------------------
SYMBOL_MAP = {
    "V10": "R_10",
    "V50": "R_50", 
    "V75": "R_75"
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
# Strict Trend Analysis (NO CONSOLIDATION SIGNALS)
# -------------------------
def calculate_ma_separation_ratio(ma1, ma2, ma3, idx):
    """Calculate normalized MA separation"""
    if not all(v is not None for v in [ma1[idx], ma2[idx], ma3[idx]]):
        return 0
    
    avg_price = (ma1[idx] + ma2[idx] + ma3[idx]) / 3
    if avg_price <= 0:
        return 0
    
    # Calculate separation between each MA pair
    sep1_2 = abs(ma1[idx] - ma2[idx]) / avg_price
    sep2_3 = abs(ma2[idx] - ma3[idx]) / avg_price
    
    return (sep1_2 + sep2_3) / 2

def validate_trend_structure(candles, ma1, ma2, ma3, current_idx, lookback=15):
    """STRICT trend validation - rejects consolidation completely"""
    if current_idx < lookback:
        return "INVALID", 0
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    current_ma3 = ma3[current_idx]
    current_price = candles[current_idx]["close"]
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return "INVALID", 0
    
    # Calculate MA separation (trend strength indicator)
    ma_separation = calculate_ma_separation_ratio(ma1, ma2, ma3, current_idx)
    
    # REJECT if MA separation is insufficient (indicates consolidation)
    if ma_separation < MA_SEPARATION_MIN:
        return "CONSOLIDATION", ma_separation  # This will be rejected
    
    # Check directional alignment
    bullish_alignment = current_ma1 > current_ma2 > current_ma3
    bearish_alignment = current_ma1 < current_ma2 < current_ma3
    
    if not (bullish_alignment or bearish_alignment):
        return "CONSOLIDATION", ma_separation  # Mixed alignment = consolidation
    
    # Validate price positioning
    if bullish_alignment:
        price_positioned_correctly = current_price > current_ma3
        if not price_positioned_correctly:
            return "INVALID", ma_separation
    else:
        price_positioned_correctly = current_price < current_ma3
        if not price_positioned_correctly:
            return "INVALID", ma_separation
    
    # Check trend consistency over lookback period
    consistent_periods = 0
    for i in range(max(0, current_idx - lookback), current_idx):
        if all(v is not None for v in [ma1[i], ma2[i], ma3[i]]):
            if bullish_alignment and ma1[i] > ma2[i] > ma3[i]:
                consistent_periods += 1
            elif bearish_alignment and ma1[i] < ma2[i] < ma3[i]:
                consistent_periods += 1
    
    consistency_ratio = consistent_periods / lookback
    if consistency_ratio < TREND_CONSISTENCY_MIN:
        return "CONSOLIDATION", ma_separation
    
    # Determine trend strength
    if bullish_alignment and consistency_ratio > 0.9:
        return "STRONG_UPTREND", ma_separation
    elif bearish_alignment and consistency_ratio > 0.9:
        return "STRONG_DOWNTREND", ma_separation
    elif bullish_alignment:
        return "UPTREND", ma_separation
    else:
        return "DOWNTREND", ma_separation

# -------------------------
# MA3 Break and Retest Detection (MANDATORY RETEST)
# -------------------------
def find_ma3_break_and_validate_retest(candles, ma1, ma2, ma3, current_idx, lookback=25):
    """Find MA3 break and validate proper retest occurred"""
    if current_idx < lookback:
        return None
    
    break_info = None
    
    # Find the most recent MA3 break
    for i in range(max(1, current_idx - lookback), current_idx - 5):  # Exclude very recent candles
        if ma3[i] is None or ma3[i-1] is None:
            continue
        
        prev_close = candles[i-1]["close"]
        curr_close = candles[i]["close"]
        
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            break_info = {"idx": i, "direction": "UP", "level": ma3[i]}
            break
        elif prev_close >= ma3[i-1] and curr_close < ma3[i]:
            break_info = {"idx": i, "direction": "DOWN", "level": ma3[i]}
            break
    
    if not break_info:
        return None
    
    # Validate retest occurred after the break
    break_idx = break_info["idx"]
    break_direction = break_info["direction"]
    
    # Look for retest in the period after the break
    retest_found = False
    retest_info = None
    
    for i in range(break_idx + 1, current_idx + 1):
        if not all(v is not None for v in [ma1[i], ma2[i]]):
            continue
        
        candle = candles[i]
        
        # Check if price came back to test MA1 or MA2 levels
        if break_direction == "UP":
            # For upward break, look for pullback to MA1/MA2 from above
            if (abs(candle["low"] - ma1[i]) <= candle["close"] * RETEST_ZONE_THRESHOLD or
                abs(candle["low"] - ma2[i]) <= candle["close"] * RETEST_ZONE_THRESHOLD):
                
                # Validate this is a proper pullback (price came from higher levels)
                pullback_distance = 0
                for j in range(break_idx + 1, i):
                    if candles[j]["high"] > candle["low"] + candle["close"] * PULLBACK_MIN_DISTANCE:
                        pullback_distance = candles[j]["high"] - candle["low"]
                        break
                
                if pullback_distance >= candle["close"] * PULLBACK_MIN_DISTANCE:
                    retest_found = True
                    retest_info = {"idx": i, "type": "PULLBACK_TO_SUPPORT"}
                    break
        
        else:  # break_direction == "DOWN"
            # For downward break, look for retest to MA1/MA2 from below
            if (abs(candle["high"] - ma1[i]) <= candle["close"] * RETEST_ZONE_THRESHOLD or
                abs(candle["high"] - ma2[i]) <= candle["close"] * RETEST_ZONE_THRESHOLD):
                
                # Validate this is a proper bounce (price came from lower levels)
                bounce_distance = 0
                for j in range(break_idx + 1, i):
                    if candles[j]["low"] < candle["high"] - candle["close"] * PULLBACK_MIN_DISTANCE:
                        bounce_distance = candle["high"] - candles[j]["low"]
                        break
                
                if bounce_distance >= candle["close"] * PULLBACK_MIN_DISTANCE:
                    retest_found = True
                    retest_info = {"idx": i, "type": "RETEST_OF_RESISTANCE"}
                    break
    
    if retest_found:
        return {
            "break": break_info,
            "retest": retest_info,
            "validated": True
        }
    
    return None

# -------------------------
# Rejection Pattern Detection (Only at Retest Zones)
# -------------------------
def is_rejection_at_retest_zone(candle, prev_candle, ma1_val, ma2_val):
    """Detect rejection patterns specifically at MA retest zones"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    # Verify candle interacted with MA levels
    interacted_with_ma = (
        abs(l - ma1_val) <= c * RETEST_ZONE_THRESHOLD or
        abs(l - ma2_val) <= c * RETEST_ZONE_THRESHOLD or
        abs(h - ma1_val) <= c * RETEST_ZONE_THRESHOLD or
        abs(h - ma2_val) <= c * RETEST_ZONE_THRESHOLD
    )
    
    if not interacted_with_ma:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # DOJI at MA levels
    if body_size <= total_range * 0.2:
        return True, "DOJI"
    
    # PINBAR with significant rejection wick
    if upper_wick >= total_range * 0.4 and upper_wick > body_size * 1.5:
        return True, "PINBAR"
    
    if lower_wick >= total_range * 0.4 and lower_wick > body_size * 1.5:
        return True, "PINBAR"
    
    # ENGULFING patterns at MA levels
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        
        if (prev_c < prev_o and c > o and 
            o <= prev_c and c >= prev_o):
            return True, "BULL_ENGULF"
        
        if (prev_c > prev_o and c < o and 
            o >= prev_c and c <= prev_o):
            return True, "BEAR_ENGULF"
    
    # Strong rejection closes (close far from tested extreme)
    if h > max(ma1_val, ma2_val) and c < h * 0.7 + l * 0.3:
        return True, "UPPER_REJECTION"
    
    if l < min(ma1_val, ma2_val) and c > h * 0.3 + l * 0.7:
        return True, "LOWER_REJECTION"
    
    return False, "NONE"

# -------------------------
# Data Fetching with Completed Candle Validation
# -------------------------
def fetch_candles_completed_only(sym, tf, count=CANDLES_N):
    """Fetch candles and ensure we work only with completed candles"""
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
            
            # Fetch extra candles to ensure we have enough completed ones
            request = {
                "ticks_history": sym,
                "style": "candles",
                "granularity": tf,
                "count": count + 2,  # Get extra candles
                "end": "latest"
            }
            
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            
            if "candles" in response and len(response["candles"]) > 1:
                candles_data = [{
                    "epoch": int(c["epoch"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                } for c in response["candles"]]
                
                # Remove the last candle (might be incomplete) and return completed candles only
                completed_candles = candles_data[:-1]
                
                if DEBUG:
                    print(f"Fetched {len(completed_candles)} completed candles for {sym}")
                
                return completed_candles
        
        except Exception as e:
            if DEBUG:
                print(f"Attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(1)
    
    return []

# -------------------------
# Master Signal Detection (Strategy-Compliant)
# -------------------------
def detect_signal_strict(candles, tf, shorthand):
    """Strict signal detection following exact strategy requirements"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    # Work with second-to-last candle (ensure it's completely closed)
    current_idx = n - 2  # Use completed candle only
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    # STEP 1: STRICT TREND VALIDATION (Reject consolidation completely)
    trend_state, trend_strength = validate_trend_structure(candles, ma1, ma2, ma3, current_idx)
    
    if trend_state in ["CONSOLIDATION", "INVALID"]:
        return None  # NO SIGNALS in consolidation or invalid states
    
    # STEP 2: VALIDATE MA3 BREAK AND RETEST (Mandatory for reversal signals)
    break_retest_info = find_ma3_break_and_validate_retest(candles, ma1, ma2, ma3, current_idx)
    
    # STEP 3: PATTERN DETECTION (Only at proper retest zones)
    is_rejection, pattern_type = is_rejection_at_retest_zone(
        current_candle, prev_candle, current_ma1, current_ma2
    )
    
    if not is_rejection:
        return None
    
    # STEP 4: SIGNAL GENERATION WITH STRICT VALIDATION
    current_close = current_candle["close"]
    signal_side = None
    reasons = []
    
    # TREND CONTINUATION SIGNALS (in established trends only)
    if trend_state in ["STRONG_UPTREND", "UPTREND"]:
        # Buy signal: Price above MA structure + rejection at support levels
        if (current_close > current_ma3 and 
            current_close > min(current_ma1, current_ma2)):
            
            # Verify this is a pullback retest (not just random rejection)
            pullback_verified = False
            for i in range(max(0, current_idx - 10), current_idx):
                if candles[i]["high"] > current_candle["low"] + current_close * PULLBACK_MIN_DISTANCE:
                    pullback_verified = True
                    break
            
            if pullback_verified:
                signal_side = "BUY"
                reasons.append(f"{trend_state.replace('_', ' ').title()} continuation - {pattern_type} at support retest")
    
    elif trend_state in ["STRONG_DOWNTREND", "DOWNTREND"]:
        # Sell signal: Price below MA structure + rejection at resistance levels
        if (current_close < current_ma3 and 
            current_close < max(current_ma1, current_ma2)):
            
            # Verify this is a bounce retest (not just random rejection)
            bounce_verified = False
            for i in range(max(0, current_idx - 10), current_idx):
                if candles[i]["low"] < current_candle["high"] - current_close * PULLBACK_MIN_DISTANCE:
                    bounce_verified = True
                    break
            
            if bounce_verified:
                signal_side = "SELL"
                reasons.append(f"{trend_state.replace('_', ' ').title()} continuation - {pattern_type} at resistance retest")
    
    # TREND REVERSAL SIGNALS (Only with validated MA3 break + retest)
    if break_retest_info and break_retest_info["validated"]:
        break_direction = break_retest_info["break"]["direction"]
        
        if (break_direction == "UP" and
            current_close > current_ma3 and
            signal_side is None):  # Don't override continuation signals
            signal_side = "BUY"
            reasons.append(f"Trend reversal - MA3 breakout with validated retest + {pattern_type}")
        
        elif (break_direction == "DOWN" and
              current_close < current_ma3 and
              signal_side is None):
            signal_side = "SELL" 
            reasons.append(f"Trend reversal - MA3 breakdown with validated retest + {pattern_type}")
    
    # Return signal only if all validations passed
    if signal_side and reasons:
        return {
            "symbol": shorthand,
            "tf": tf,
            "side": signal_side,
            "reasons": reasons,
            "pattern": pattern_type,
            "trend": trend_state,
            "trend_strength": trend_strength,
            "break_retest": break_retest_info,
            "idx": current_idx,
            "ma1": ma1,
            "ma2": ma2,
            "ma3": ma3,
            "candles": candles,
            "validation": "STRICT_COMPLIANCE"
        }
    
    return None

# -------------------------
# Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    """Professional chart generation"""
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
            (i - 0.35/2, min(o, c)), 
            0.35, 
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
        
        marker_color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker_symbol = "^" if signal_data["side"] == "BUY" else "v"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} Signal ✓ VALIDATED", 
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
def run_analysis():
    """Main analysis with strict strategy compliance"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = TIMEFRAMES[0] if TIMEFRAMES else 300
            
            if DEBUG:
                print(f"Analyzing {shorthand} with strict validation...")
            
            # Fetch only completed candles
            candles = fetch_candles_completed_only(deriv_symbol, tf)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"Insufficient completed candles for {shorthand}: {len(candles)}")
                continue
            
            # Strict signal detection
            signal = detect_signal_strict(candles, tf, shorthand)
            if not signal:
                continue
            
            # Check for duplicates
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                continue
            
            # Create enhanced alert message
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} ✅\n"
                      f"📊 Trend: {signal['trend'].replace('_', ' ').title()}\n"
                      f"🎨 Pattern: {signal['pattern']}\n"
                      f"💰 Price: {signal['candles'][signal['idx']]['close']}\n"
                      f"✓ Validation: STRICT COMPLIANCE\n")
            
            if signal.get("break_retest"):
                caption += f"🔄 Retest: Validated MA3 break + retest confirmed\n"
            
            caption += f"📝 Analysis:\n" + "\n".join(f"• {reason}" for reason in signal["reasons"])
            
            # Generate and send chart
            chart_path = create_signal_chart(signal)
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"VALIDATED signal sent for {shorthand}: {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
    
    if DEBUG:
        print(f"Analysis complete. {signals_found} validated signals found.")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

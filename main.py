#!/usr/bin/env python3
"""
Enhanced DSR Trading Bot - M10 Timeframe Only
Focus on market structure, pullbacks/retracements, and dynamic analysis
Removes fixed tolerances and adds accumulation/distribution detection
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
import websocket, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

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

CANDLES_N = 600  # More candles for better structure analysis
LAST_N_CHART = 200
CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")
MIN_CANDLES = 100  # Need more candles for structure analysis

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
# Market Structure Analysis
# -------------------------
def detect_accumulation_distribution(candles, lookback=20):
    """Detect accumulation/distribution phases using volume-price analysis"""
    if len(candles) < lookback:
        return "UNDEFINED"
    
    recent_candles = candles[-lookback:]
    
    # Analyze price action and candle patterns
    bullish_pressure = 0
    bearish_pressure = 0
    small_body_count = 0
    
    for candle in recent_candles:
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        body_size = abs(c - o)
        total_range = h - l
        
        if total_range > 0:
            body_ratio = body_size / total_range
            
            # Small bodies indicate consolidation/accumulation
            if body_ratio < 0.3:
                small_body_count += 1
            
            # Analyze close position within range
            close_position = (c - l) / total_range if total_range > 0 else 0.5
            
            if close_position > 0.7:  # Close in upper part
                bullish_pressure += 1
            elif close_position < 0.3:  # Close in lower part
                bearish_pressure += 1
    
    # Determine phase
    if small_body_count > lookback * 0.6:  # More than 60% small bodies
        if bullish_pressure > bearish_pressure * 1.5:
            return "ACCUMULATION"
        elif bearish_pressure > bullish_pressure * 1.5:
            return "DISTRIBUTION"
        else:
            return "CONSOLIDATION"
    else:
        return "TRENDING"

def detect_market_structure(candles, ma1, ma2, ma3, current_idx):
    """Enhanced market structure detection"""
    if current_idx < 50:
        return {"structure": "UNDEFINED", "strength": 0}
    
    lookback = min(50, current_idx)
    recent_candles = candles[current_idx - lookback + 1:current_idx + 1]
    recent_ma1 = ma1[current_idx - lookback + 1:current_idx + 1]
    recent_ma2 = ma2[current_idx - lookback + 1:current_idx + 1]
    recent_ma3 = ma3[current_idx - lookback + 1:current_idx + 1]
    
    # Calculate MA slopes (trend strength)
    def calculate_slope(values, periods=10):
        if len(values) < periods:
            return 0
        valid_values = [v for v in values[-periods:] if v is not None]
        if len(valid_values) < periods // 2:
            return 0
        
        # Simple slope calculation
        y_values = np.array(valid_values)
        x_values = np.array(range(len(valid_values)))
        if len(x_values) > 1:
            slope = np.polyfit(x_values, y_values, 1)[0]
            return slope / np.mean(y_values) if np.mean(y_values) != 0 else 0
        return 0
    
    ma1_slope = calculate_slope(recent_ma1)
    ma2_slope = calculate_slope(recent_ma2)
    ma3_slope = calculate_slope(recent_ma3)
    
    # Analyze MA arrangement consistency
    arrangement_consistency = 0
    for i in range(len(recent_ma1)):
        if all(v is not None for v in [recent_ma1[i], recent_ma2[i], recent_ma3[i]]):
            if recent_ma1[i] > recent_ma2[i] > recent_ma3[i]:
                arrangement_consistency += 1
            elif recent_ma1[i] < recent_ma2[i] < recent_ma3[i]:
                arrangement_consistency -= 1
    
    arrangement_strength = abs(arrangement_consistency) / len(recent_ma1)
    trend_strength = (abs(ma1_slope) + abs(ma2_slope) + abs(ma3_slope)) / 3
    
    # Determine structure
    if arrangement_strength > 0.7 and trend_strength > 0.001:
        if arrangement_consistency > 0:
            return {"structure": "STRONG_UPTREND", "strength": trend_strength}
        else:
            return {"structure": "STRONG_DOWNTREND", "strength": trend_strength}
    elif arrangement_strength > 0.4:
        if arrangement_consistency > 0:
            return {"structure": "WEAK_UPTREND", "strength": trend_strength}
        else:
            return {"structure": "WEAK_DOWNTREND", "strength": trend_strength}
    else:
        return {"structure": "CONSOLIDATION", "strength": trend_strength}

def detect_breakout_potential(candles, ma1, ma2, current_idx, lookback=15):
    """Detect potential breakout from consolidation"""
    if current_idx < lookback:
        return False, "NONE"
    
    recent_highs = [c["high"] for c in candles[current_idx - lookback:current_idx + 1]]
    recent_lows = [c["low"] for c in candles[current_idx - lookback:current_idx + 1]]
    
    # Check for range compression
    recent_range = max(recent_highs) - min(recent_lows)
    if recent_range == 0:
        return False, "NONE"
    
    # Calculate average range
    ranges = []
    for i in range(max(0, current_idx - lookback * 2), current_idx - lookback):
        if i < len(candles):
            ranges.append(candles[i]["high"] - candles[i]["low"])
    
    if not ranges:
        return False, "NONE"
    
    avg_range = sum(ranges) / len(ranges)
    
    # Compression detected if recent range is smaller than average
    compression_ratio = recent_range / avg_range if avg_range > 0 else 1
    
    current_candle = candles[current_idx]
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    
    if compression_ratio < 0.7 and current_ma1 is not None and current_ma2 is not None:
        # Check for breakout
        if current_candle["close"] > max(recent_highs[:-1]):
            return True, "BULLISH_BREAKOUT"
        elif current_candle["close"] < min(recent_lows[:-1]):
            return True, "BEARISH_BREAKOUT"
    
    return False, "NONE"

# -------------------------
# Enhanced Rejection Detection
# -------------------------
def detect_rejection_candlestick(candle):
    """COMPLETELY DYNAMIC rejection detection - NO numerical thresholds"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE", 0
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # DYNAMIC ratios - NO fixed thresholds
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    body_ratio = body_size / total_range
    
    # ANY meaningful rejection pattern - completely relative to the candle itself
    
    # 1. Has upper wick AND it's larger than lower wick = Upper rejection
    if upper_wick > 0 and upper_wick >= lower_wick:
        strength = upper_wick_ratio * (2 - body_ratio)  # More strength if smaller body
        return True, "UPPER_REJECTION", strength
    
    # 2. Has lower wick AND it's larger than upper wick = Lower rejection  
    if lower_wick > 0 and lower_wick > upper_wick:
        strength = lower_wick_ratio * (2 - body_ratio)  # More strength if smaller body
        return True, "LOWER_REJECTION", strength
    
    # 3. Body smaller than either wick = Small body pattern
    if body_size < upper_wick or body_size < lower_wick:
        strength = (1 - body_ratio) * (upper_wick_ratio + lower_wick_ratio + 0.5)
        if upper_wick >= lower_wick:
            return True, "SMALL_BODY_UPPER", strength
        else:
            return True, "SMALL_BODY_LOWER", strength
    
    # 4. Both wicks exist = Indecision (regardless of size)
    if upper_wick > 0 and lower_wick > 0:
        strength = (upper_wick_ratio + lower_wick_ratio) * 0.8
        return True, "INDECISION_REJECTION", strength
    
    # 5. ANY wick exists = Some form of rejection
    if upper_wick > 0:
        strength = upper_wick_ratio * 0.6
        return True, "BASIC_UPPER_REJECTION", strength
    
    if lower_wick > 0:
        strength = lower_wick_ratio * 0.6
        return True, "BASIC_LOWER_REJECTION", strength
    
    return False, "NONE", 0

def detect_pullback_quality(candles, ma1, ma2, current_idx, lookback=10):
    """Detect small/quick pullbacks vs large/slow pullbacks"""
    if current_idx < lookback:
        return "UNDEFINED", 0
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    
    if current_ma1 is None or current_ma2 is None:
        return "UNDEFINED", 0
    
    # Determine trend direction
    if current_ma1 > current_ma2:
        trend = "UP"
        key_level = current_ma1
    else:
        trend = "DOWN" 
        key_level = current_ma1
    
    # Analyze recent price action relative to key level
    touches = 0
    max_penetration = 0
    candles_away = 0
    
    for i in range(max(0, current_idx - lookback), current_idx + 1):
        candle = candles[i]
        
        if trend == "UP":
            # Count touches below MA1 (pullback)
            if candle["low"] < key_level:
                touches += 1
                penetration = (key_level - candle["low"]) / key_level
                max_penetration = max(max_penetration, penetration)
                candles_away = current_idx - i
        else:
            # Count touches above MA1 (pullback)
            if candle["high"] > key_level:
                touches += 1
                penetration = (candle["high"] - key_level) / key_level  
                max_penetration = max(max_penetration, penetration)
                candles_away = current_idx - i
    
    # Classify pullback quality
    if max_penetration < 0.002 and candles_away <= 3:  # Very small, quick
        return "EXCELLENT_PULLBACK", 0.9
    elif max_penetration < 0.005 and candles_away <= 5:  # Small, quick
        return "GOOD_PULLBACK", 0.7
    elif max_penetration < 0.01 and candles_away <= 8:  # Medium
        return "AVERAGE_PULLBACK", 0.5
    else:  # Large or slow - likely stop hunt
        return "POOR_PULLBACK", 0.2

# -------------------------
# Dynamic MA Touch Detection
# -------------------------
def dynamic_ma_touch(candle, ma_value):
    """Dynamic detection of MA touch without fixed tolerances"""
    if ma_value is None:
        return False
    
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    
    # Use candle's own range to determine what constitutes a "touch"
    candle_range = h - l
    if candle_range <= 0:
        return False
    
    # Dynamic tolerance based on candle range (more sensitive)
    touch_threshold = candle_range * 0.1  # 10% of candle range
    
    # Check if any part of candle is within threshold of MA
    distance_to_ma = min(
        abs(h - ma_value),
        abs(l - ma_value), 
        abs(c - ma_value),
        abs(o - ma_value)
    )
    
    return distance_to_ma <= touch_threshold

def detect_exhaustion_signals(candles, current_idx, lookback=5):
    """Detect potential exhaustion phases"""
    if current_idx < lookback:
        return False, "NONE"
    
    recent_candles = candles[current_idx - lookback + 1:current_idx + 1]
    
    # Look for decreasing momentum patterns
    decreasing_bodies = 0
    increasing_wicks = 0
    
    for i in range(1, len(recent_candles)):
        curr = recent_candles[i]
        prev = recent_candles[i-1]
        
        curr_body = abs(curr["close"] - curr["open"])
        prev_body = abs(prev["close"] - prev["open"])
        
        curr_total = curr["high"] - curr["low"]
        prev_total = prev["high"] - prev["low"]
        
        if curr_total > 0 and prev_total > 0:
            curr_wick_ratio = (curr_total - curr_body) / curr_total
            prev_wick_ratio = (prev_total - prev_body) / prev_total
            
            if curr_body < prev_body:
                decreasing_bodies += 1
            
            if curr_wick_ratio > prev_wick_ratio:
                increasing_wicks += 1
    
    if decreasing_bodies >= lookback * 0.6 and increasing_wicks >= lookback * 0.6:
        return True, "MOMENTUM_EXHAUSTION"
    
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
# Enhanced DSR Signal Detection
# -------------------------
def detect_dsr_signal(candles, shorthand):
    """Enhanced DSR Strategy with market structure analysis"""
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
    
    # Analyze market structure first
    market_structure = detect_market_structure(candles, ma1, ma2, ma3, current_idx)
    
    # Skip consolidation markets unless it's a clear breakout
    if market_structure["structure"] == "CONSOLIDATION":
        is_breakout, breakout_type = detect_breakout_potential(candles, ma1, ma2, current_idx)
        if not is_breakout:
            return None
    
    # Detect accumulation/distribution phase
    acc_dist_phase = detect_accumulation_distribution(candles)
    
    # Skip if in pure consolidation without clear bias
    if acc_dist_phase == "CONSOLIDATION":
        return None
    
    # ANY rejection pattern is acceptable - NO thresholds
    is_rejection, rejection_type, rejection_strength = detect_rejection_candlestick(current_candle)
    if not is_rejection:  # Just needs to be a rejection - NO strength filter
        return None
    
    # Dynamic MA touch detection (NO fixed tolerances)
    touched_ma1 = dynamic_ma_touch(current_candle, current_ma1)
    touched_ma2 = dynamic_ma_touch(current_candle, current_ma2)
    
    if not (touched_ma1 or touched_ma2):
        return None
    
    # Determine which MA was touched
    ma_level = "MA1" if touched_ma1 else "MA2"
    key_ma = current_ma1 if touched_ma1 else current_ma2
    
    # Analyze pullback quality
    pullback_quality, pullback_score = detect_pullback_quality(candles, ma1, ma2, current_idx)
    
    # Only take high-quality pullbacks
    if pullback_score < 0.6:
        return None
    
    # Check for exhaustion (avoid trend ending signals)
    is_exhaustion, exhaustion_type = detect_exhaustion_signals(candles, current_idx)
    if is_exhaustion:
        return None
    
    # Determine signal direction based on structure and rejection
    signal_side = None
    
    if market_structure["structure"] in ["STRONG_UPTREND", "WEAK_UPTREND"]:
        if rejection_type in ["LOWER_REJECTION", "SMALL_BODY_LOWER", "INDECISION_REJECTION", "BASIC_LOWER_REJECTION"]:
            if current_candle["close"] >= key_ma:  # Closed above MA after rejection
                signal_side = "BUY"
    
    elif market_structure["structure"] in ["STRONG_DOWNTREND", "WEAK_DOWNTREND"]:
        if rejection_type in ["UPPER_REJECTION", "SMALL_BODY_UPPER", "INDECISION_REJECTION", "BASIC_UPPER_REJECTION"]:
            if current_candle["close"] <= key_ma:  # Closed below MA after rejection
                signal_side = "SELL"
    
    if not signal_side:
        return None
    
    # Enhanced cooldown (per symbol)
    cooldown_key = f'last_signal_{shorthand}'
    last_signal_time = getattr(detect_dsr_signal, cooldown_key, 0)
    current_time = current_candle["epoch"]
    
    if current_time - last_signal_time < 3600:  # 1 hour cooldown
        return None
    
    setattr(detect_dsr_signal, cooldown_key, current_time)
    
    if DEBUG:
        print(f"ENHANCED DSR: {signal_side} - {rejection_type} at {ma_level}")
        print(f"Structure: {market_structure['structure']}, Pullback: {pullback_quality}")
        print(f"Phase: {acc_dist_phase}, Rejection Strength: {rejection_strength:.2f}")
    
    return {
        "symbol": shorthand,
        "side": signal_side,
        "pattern": rejection_type,
        "ma_level": ma_level,
        "market_structure": market_structure,
        "pullback_quality": pullback_quality,
        "pullback_score": pullback_score,
        "acc_dist_phase": acc_dist_phase,
        "rejection_strength": rejection_strength,
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
# Enhanced Chart Generation
# -------------------------
def create_enhanced_chart(signal_data):
    """Create enhanced chart with market structure visualization"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - LAST_N_CHART)
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot candlesticks with enhanced colors
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
    
    # Plot moving averages with structure-based colors
    def plot_enhanced_ma(ma_values, label, base_color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        
        valid_points = [(i, v) for i, v in enumerate(chart_ma) if v is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            ax.plot(x_vals, y_vals, color=base_color, linewidth=linewidth, 
                   label=label, alpha=0.95, linestyle='-')
    
    # Enhanced MA plotting
    plot_enhanced_ma(ma1, "MA1 (SMMA HLC3-9) - Dynamic S/R", "#FFFFFF", 3)
    plot_enhanced_ma(ma2, "MA2 (SMMA Close-19) - Dynamic S/R", "#00BFFF", 2.5)
    plot_enhanced_ma(ma3, "MA3 (SMA MA2-25) - Trend Filter", "#FF6347", 2)
    
    # Mark signal point with enhanced visualization
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        
        if signal_data["side"] == "BUY":
            marker_color = "#00FF88"
            marker_symbol = "^"
            arrow_props = dict(arrowstyle='->', color='#00FF88', lw=3)
        else:
            marker_color = "#FF3366" 
            marker_symbol = "v"
            arrow_props = dict(arrowstyle='->', color='#FF3366', lw=3)
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=400, edgecolor="#FFFFFF", linewidth=4, zorder=10)
        
        # Add signal annotation
        ax.annotate(f'{signal_data["side"]} Signal', 
                   xy=(signal_chart_idx, signal_price),
                   xytext=(signal_chart_idx + 10, signal_price),
                   fontsize=12, color='white', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=marker_color, alpha=0.8))
    
    # Enhanced title with structure info
    structure_emoji = {
        "STRONG_UPTREND": "📈💪", "WEAK_UPTREND": "📈",
        "STRONG_DOWNTREND": "📉💪", "WEAK_DOWNTREND": "📉",
        "CONSOLIDATION": "⏸️"
    }.get(signal_data["market_structure"]["structure"], "❓")
    
    pullback_emoji = {"EXCELLENT_PULLBACK": "👑", "GOOD_PULLBACK": "✅", "AVERAGE_PULLBACK": "⚡"}.get(signal_data["pullback_quality"], "")
    
    ax.set_title(f"{signal_data['symbol']} M10 - {signal_data['side']} DSR Signal {structure_emoji}{pullback_emoji}\n"
                f"Structure: {signal_data['market_structure']['structure']} | "
                f"Pullback: {signal_data['pullback_quality']} | "
                f"Phase: {signal_data['acc_dist_phase']}", 
                fontsize=14, color='white', fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', 
                      edgecolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.2, color='gray', linestyle=':', linewidth=0.8)
    ax.tick_params(colors='white', labelsize=11)
    
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, 
                dpi=200, 
                bbox_inches="tight", 
                facecolor='black',
                edgecolor='none',
                pad_inches=0.2)
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

# -------------------------
# Chart Pattern Detection (Additional Enhancement)
# -------------------------
def detect_chart_patterns(candles, current_idx, lookback=30):
    """Detect basic chart patterns - supplementary to DSR"""
    if current_idx < lookback:
        return "NONE", 0
    
    recent_candles = candles[current_idx - lookback:current_idx + 1]
    highs = [c["high"] for c in recent_candles]
    lows = [c["low"] for c in recent_candles]
    
    # Simple double top/bottom detection
    max_high = max(highs)
    min_low = min(lows)
    
    # Find peaks and valleys
    peaks = []
    valleys = []
    
    for i in range(2, len(recent_candles) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
            highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            peaks.append((i, highs[i]))
        
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
            lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            valleys.append((i, lows[i]))
    
    # Double top detection
    if len(peaks) >= 2:
        last_two_peaks = peaks[-2:]
        if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
            return "DOUBLE_TOP", 0.7
    
    # Double bottom detection  
    if len(valleys) >= 2:
        last_two_valleys = valleys[-2:]
        if abs(last_two_valleys[0][1] - last_two_valleys[1][1]) / last_two_valleys[0][1] < 0.02:
            return "DOUBLE_BOTTOM", 0.7
    
    return "NONE", 0

def detect_horizontal_levels(candles, current_idx, lookback=50, min_touches=3):
    """Detect horizontal support/resistance levels"""
    if current_idx < lookback:
        return []
    
    recent_candles = candles[current_idx - lookback:current_idx + 1]
    price_levels = []
    
    # Collect significant price levels
    for candle in recent_candles:
        price_levels.extend([candle["high"], candle["low"], candle["close"]])
    
    # Group similar price levels
    levels = []
    tolerance = 0.001  # 0.1% tolerance for grouping
    
    for price in price_levels:
        found_level = False
        for level in levels:
            if abs(price - level["price"]) / level["price"] < tolerance:
                level["touches"] += 1
                level["price"] = (level["price"] * (level["touches"] - 1) + price) / level["touches"]
                found_level = True
                break
        
        if not found_level:
            levels.append({"price": price, "touches": 1})
    
    # Return levels with minimum touches
    significant_levels = [level for level in levels if level["touches"] >= min_touches]
    return sorted(significant_levels, key=lambda x: x["touches"], reverse=True)[:3]  # Top 3

# -------------------------
# Main Execution - M10 Only
# -------------------------
def run_enhanced_dsr_analysis():
    """Enhanced DSR analysis - M10 timeframe only with market structure"""
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
            
            signal = detect_dsr_signal(candles, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, current_epoch, signal["side"]):
                if DEBUG:
                    print(f"Signal already sent for {shorthand}")
                continue
            
            # Detect supplementary chart patterns
            pattern, pattern_strength = detect_chart_patterns(candles, signal["idx"])
            horizontal_levels = detect_horizontal_levels(candles, signal["idx"])
            
            # Create enhanced alert message
            structure_emoji = {
                "STRONG_UPTREND": "📈💪", "WEAK_UPTREND": "📈",
                "STRONG_DOWNTREND": "📉💪", "WEAK_DOWNTREND": "📉", 
                "CONSOLIDATION": "⏸️"
            }.get(signal["market_structure"]["structure"], "❓")
            
            pullback_emoji = {
                "EXCELLENT_PULLBACK": "👑", 
                "GOOD_PULLBACK": "✅", 
                "AVERAGE_PULLBACK": "⚡"
            }.get(signal["pullback_quality"], "")
            
            phase_emoji = {
                "ACCUMULATION": "📦", 
                "DISTRIBUTION": "📤", 
                "TRENDING": "🚀"
            }.get(signal["acc_dist_phase"], "")
            
            caption = (f"🎯 {signal['symbol']} M10 - {signal['side']} DSR SIGNAL {structure_emoji}{pullback_emoji}\n\n"
                      f"📊 Market Structure: {signal['market_structure']['structure']}\n"
                      f"🔄 Pullback Quality: {signal['pullback_quality']} (Score: {signal['pullback_score']:.1f})\n"
                      f"{phase_emoji} Market Phase: {signal['acc_dist_phase']}\n"
                      f"🎨 Rejection Pattern: {signal['pattern']}\n"
                      f"⚡ Rejection Strength: {signal['rejection_strength']:.2f}\n"
                      f"📍 Dynamic Level: {signal['ma_level']}\n"
                      f"💰 Entry Price: {signal['price']:.5f}\n")
            
            # Add chart pattern info if detected
            if pattern != "NONE":
                caption += f"📈 Chart Pattern: {pattern} (Strength: {pattern_strength:.1f})\n"
            
            # Add horizontal levels info
            if horizontal_levels:
                caption += f"📏 Key Levels: {len(horizontal_levels)} detected\n"
            
            caption += f"\n🔥 This is a HIGH-QUALITY pullback signal!\n💎 DSR Strategy: Trend Following at Dynamic S/R"
            
            chart_path = create_enhanced_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"Enhanced DSR signal sent for {shorthand}: {signal['side']}")
                    print(f"Structure: {signal['market_structure']['structure']}, Quality: {signal['pullback_quality']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                    
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"Enhanced DSR Analysis complete. {signals_found} high-quality signals found.")

# -------------------------
# Additional Utility Functions
# -------------------------
def calculate_atr(candles, period=14):
    """Calculate Average True Range for dynamic analysis"""
    if len(candles) < period + 1:
        return [None] * len(candles)
    
    true_ranges = []
    
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i-1]
        
        tr1 = curr["high"] - curr["low"]
        tr2 = abs(curr["high"] - prev["close"])
        tr3 = abs(curr["low"] - prev["close"])
        
        true_ranges.append(max(tr1, tr2, tr3))
    
    # Calculate ATR using SMA
    atr_values = [None]  # First candle has no ATR
    
    if len(true_ranges) >= period:
        # Initial ATR
        initial_atr = sum(true_ranges[:period]) / period
        atr_values.extend([None] * (period - 1))
        atr_values.append(initial_atr)
        
        # Subsequent ATR values using exponential smoothing
        prev_atr = initial_atr
        for i in range(period, len(true_ranges)):
            current_atr = (prev_atr * (period - 1) + true_ranges[i]) / period
            atr_values.append(current_atr)
            prev_atr = current_atr
    else:
        atr_values.extend([None] * len(true_ranges))
    
    return atr_values

def detect_momentum_divergence(candles, current_idx, lookback=20):
    """Detect momentum divergence for additional confluence"""
    if current_idx < lookback:
        return False, "NONE"
    
    # Simple momentum using close prices
    recent_closes = [c["close"] for c in candles[current_idx - lookback:current_idx + 1]]
    
    # Calculate momentum (rate of change)
    momentum = []
    for i in range(5, len(recent_closes)):
        roc = (recent_closes[i] - recent_closes[i-5]) / recent_closes[i-5]
        momentum.append(roc)
    
    if len(momentum) < 10:
        return False, "NONE"
    
    # Check for divergence patterns
    price_trend = recent_closes[-1] - recent_closes[-10]
    momentum_trend = momentum[-1] - momentum[-5] if len(momentum) >= 5 else 0
    
    # Bullish divergence: price making lower lows, momentum making higher lows
    if price_trend < 0 and momentum_trend > 0:
        return True, "BULLISH_DIVERGENCE"
    
    # Bearish divergence: price making higher highs, momentum making lower highs  
    if price_trend > 0 and momentum_trend < 0:
        return True, "BEARISH_DIVERGENCE"
    
    return False, "NONE"

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        print("Starting Enhanced DSR Trading Bot - M10 Timeframe Only")
        print("Features: Market Structure Analysis | Dynamic Pullback Detection | No Fixed Tolerances")
        print("Focus: High-Quality Retracements at Dynamic S/R Levels")
        
        run_enhanced_dsr_analysis()
        
    except Exception as e:
        print(f"Critical error in Enhanced DSR Bot: {e}")
        traceback.print_exc()

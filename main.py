#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (Trend Following Strategy)

Strategy Logic:
- MA1 (SMMA HLC3-9) and MA2 (SMMA Close-19) act as dynamic support/resistance
- MA3 (SMA of MA2-25) acts as trend filter
- Above MA3 = Buy bias, Below MA3 = Sell bias
- Signals fire on rejection candlesticks at/near MA1 or MA2
- Immediate signal on candle close, no delays
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
    "V50": "R_50", 
    "V75": "R_75",
    "Jump10": "JD10",
    "Jump25": "JD25",
    "Jump50": "JD50", 
    "Jump100": "JD100",
    "V75(1s)": "R_1HZ75V",
    "V100(1s)": "R_1HZ100V",
    "V150(1s)": "R_1HZ150V",
    "V15(1s)": "R_1HZ15V"
}

SYMBOL_TF_MAP = {
    "V75(1s)": 1,
    "V100(1s)": 1,  
    "V150(1s)": 1,
    "V15(1s)": 1
}

# -------------------------
# Persistence (Deduplication)
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
# Moving Averages (Exact as Strategy)
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
# Trend Detection (MA Alignment + MA3 Filter)
# -------------------------
def get_trend_state(candles, ma1, ma2, ma3, idx):
    """Determine trend state - AVOID CONSOLIDATION/RANGING AT ALL COSTS (especially M5 and below)"""
    if idx < len(ma1) and idx < len(ma2) and idx < len(ma3):
        current_ma1 = ma1[idx]
        current_ma2 = ma2[idx]
        current_ma3 = ma3[idx]
        current_price = candles[idx]["close"]
        
        if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
            return "UNDEFINED"
        
        # Check MA alignment for clear trend direction
        uptrend_alignment = current_ma1 > current_ma2 > current_ma3
        downtrend_alignment = current_ma1 < current_ma2 < current_ma3
        
        # MA3 filter for bias
        above_ma3 = current_price > current_ma3
        below_ma3 = current_price < current_ma3
        
        # ENHANCED: Calculate multiple separation metrics to detect ranging
        avg_ma = (current_ma1 + current_ma2 + current_ma3) / 3
        ma1_ma3_separation = abs(current_ma1 - current_ma3) / avg_ma if avg_ma > 0 else 0
        ma1_ma2_separation = abs(current_ma1 - current_ma2) / avg_ma if avg_ma > 0 else 0
        ma2_ma3_separation = abs(current_ma2 - current_ma3) / avg_ma if avg_ma > 0 else 0
        
        # STRICT: Minimum separation thresholds (higher for lower timeframes)
        base_min_separation = 0.003  # 0.3% base minimum
        strict_min_separation = 0.005  # 0.5% for M5 and below
        
        # Use stricter thresholds for lower timeframes
        min_separation = strict_min_separation  # Default to strict
        
        # Check if ALL separations meet minimum requirements
        separations_adequate = (ma1_ma3_separation > min_separation and 
                               ma1_ma2_separation > min_separation * 0.5 and
                               ma2_ma3_separation > min_separation * 0.5)
        
        # ADDITIONAL RANGING DETECTION: Check price oscillation around MAs
        if idx >= 10:  # Look back 10 candles
            ma_crosses = 0
            for i in range(idx - 9, idx + 1):
                if i < len(candles) and i < len(ma1) and ma1[i] is not None and ma2[i] is not None:
                    price = candles[i]["close"]
                    # Count how often price crosses around MA1/MA2
                    if abs(price - ma1[i]) < ma1[i] * 0.01:  # Within 1% of MA1
                        ma_crosses += 1
                    if abs(price - ma2[i]) < ma2[i] * 0.01:  # Within 1% of MA2  
                        ma_crosses += 1
            
            # If price oscillates too much around MAs = ranging
            if ma_crosses > 6:  # More than 60% of time near MAs = ranging
                return "RANGING"
        
        # FINAL TREND DETERMINATION with strict requirements
        if (uptrend_alignment and above_ma3 and separations_adequate):
            return "UPTREND"
        elif (downtrend_alignment and below_ma3 and separations_adequate):
            return "DOWNTREND"
        else:
            # If any condition fails = treat as RANGING (avoid at all costs)
            return "RANGING"
    
    return "UNDEFINED"

# -------------------------
# Rejection Candlestick Detection
# -------------------------
def is_rejection_candle(candle, prev_candle=None):
    """Detect ANY rejection candlestick - flexible, no rigid thresholds"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range == 0 or total_range < 1e-9:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # 1. HAMMER/SHOOTING STAR - Any significant lower/upper wick
    if lower_wick > body_size:  # Lower wick bigger than body = bullish rejection
        return True, "HAMMER_REJECTION"
    
    if upper_wick > body_size:  # Upper wick bigger than body = bearish rejection  
        return True, "SHOOTING_STAR_REJECTION"
    
    # 2. DOJI FAMILY - Small body relative to total range (any size)
    if body_size < total_range * 0.5:  # Body less than 50% of range
        if upper_wick > 0 or lower_wick > 0:  # Has wicks = rejection
            return True, "DOJI_REJECTION"
    
    # 3. SPINNING TOP - Small body with wicks on both sides
    if (body_size < total_range * 0.4 and 
        upper_wick > 0 and lower_wick > 0):
        return True, "SPINNING_TOP_REJECTION"
    
    # 4. ENGULFING PATTERNS - Previous candle context
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        prev_body = abs(prev_c - prev_o)
        
        # Bullish engulfing - current green candle engulfs previous red
        if (prev_c < prev_o and c > o and 
            o <= prev_c and c >= prev_o and 
            body_size > prev_body * 0.8):  # More lenient threshold
            return True, "BULLISH_ENGULFING"
        
        # Bearish engulfing - current red candle engulfs previous green
        if (prev_c > prev_o and c < o and 
            o >= prev_c and c <= prev_o and 
            body_size > prev_body * 0.8):  # More lenient threshold
            return True, "BEARISH_ENGULFING"
    
    # 5. REVERSAL HAMMER - Close away from one extreme
    if c <= l + (total_range * 0.3):  # Close in lower 30% = bearish rejection
        return True, "BEARISH_REVERSAL"
    
    if c >= h - (total_range * 0.3):  # Close in upper 70% = bullish rejection  
        return True, "BULLISH_REVERSAL"
    
    # 6. INDECISION CANDLES - Any candle with meaningful wicks
    if (upper_wick >= total_range * 0.2 or 
        lower_wick >= total_range * 0.2):  # 20% wick threshold
        return True, "INDECISION_REJECTION"
    
    # 7. SMALL BODY REJECTION - Body smaller than average wick
    avg_wick = (upper_wick + lower_wick) / 2
    if body_size < avg_wick and avg_wick > 0:
        return True, "SMALL_BODY_REJECTION"
    
    # 8. EXTREME CLOSE - Close significantly away from one extreme
    close_ratio_from_low = (c - l) / total_range if total_range > 0 else 0.5
    close_ratio_from_high = (h - c) / total_range if total_range > 0 else 0.5
    
    # Close very near low (bearish rejection of higher levels)
    if close_ratio_from_low <= 0.25:
        return True, "LOW_CLOSE_REJECTION"
    
    # Close very near high (bullish rejection of lower levels)  
    if close_ratio_from_high <= 0.25:
        return True, "HIGH_CLOSE_REJECTION"
    
    return False, "NONE"

# -------------------------
# Dynamic Support & Resistance Zone Detection (Adaptive)
# -------------------------
def calculate_dynamic_zone_size(candles, current_idx, lookback=10):
    """Calculate dynamic zone size based on recent market volatility"""
    if current_idx < lookback:
        lookback = current_idx
    
    if lookback <= 0:
        return 0.002  # Fallback 0.2%
    
    # Get recent candle ranges
    recent_ranges = []
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i >= 0 and i < len(candles):
            candle_range = candles[i]["high"] - candles[i]["low"]
            recent_ranges.append(candle_range)
    
    if not recent_ranges:
        return 0.002
    
    # Average recent volatility
    avg_range = sum(recent_ranges) / len(recent_ranges)
    current_price = candles[current_idx]["close"]
    
    # Base zone size = 75% of average candle range
    if current_price > 0:
        base_zone = avg_range / current_price * 0.75
    else:
        base_zone = 0.002
    
    # Maximum bound only - no minimum to allow direct MA touches
    max_zone = 0.008  # 0.8% maximum to prevent excessive zones
    
    return min(base_zone, max_zone)

def assess_trend_strength(candles, ma1, ma2, ma3, current_idx, lookback=8):
    """Assess trend strength for zone expansion"""
    if current_idx < lookback:
        return 1.0  # Neutral multiplier
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return 1.0
    
    # Check MA alignment consistency over lookback period
    alignment_score = 0
    total_checks = 0
    
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if (i < len(ma1) and i < len(ma2) and i < len(ma3) and
            all(v is not None for v in [ma1[i], ma2[i], ma3[i]])):
            
            # Check for proper alignment
            if ma1[i] > ma2[i] > ma3[i]:  # Uptrend alignment
                alignment_score += 1
            elif ma1[i] < ma2[i] < ma3[i]:  # Downtrend alignment  
                alignment_score += 1
            total_checks += 1
    
    if total_checks == 0:
        return 1.0
    
    consistency = alignment_score / total_checks
    
    # Strong trend = wider zones (allows tiny retests further away)
    if consistency >= 0.8:
        return 1.5  # 50% wider zones
    elif consistency >= 0.6:
        return 1.25 # 25% wider zones
    else:
        return 1.0  # Standard zones

def is_in_dsr_zone(price, ma1_val, ma2_val, trend_state, zone_size):
    """Dynamic DSR zone detection - adapts to market volatility"""
    if ma1_val is None or ma2_val is None:
        return False, "NONE"
    
    # Calculate dynamic thresholds
    ma1_threshold = ma1_val * zone_size
    ma2_threshold = ma2_val * zone_size
    
    # In trending markets, MAs act as dynamic S/R
    if trend_state == "UPTREND":
        # MA1 and MA2 act as support in uptrends
        # Check if price is testing these support levels
        if abs(price - ma1_val) <= ma1_threshold:
            return True, "MA1_SUPPORT"
        if abs(price - ma2_val) <= ma2_threshold:
            return True, "MA2_SUPPORT"
    
    elif trend_state == "DOWNTREND":
        # MA1 and MA2 act as resistance in downtrends
        # Check if price is testing these resistance levels  
        if abs(price - ma1_val) <= ma1_threshold:
            return True, "MA1_RESISTANCE"
        if abs(price - ma2_val) <= ma2_threshold:
            return True, "MA2_RESISTANCE"
    
    return False, "NONE"

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
# Signal Detection (Core Strategy Logic)
# -------------------------
def detect_signal(candles, tf, shorthand):
    """Detect signals on LAST CLOSED CANDLE ONLY - confirmed rejection patterns"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    # CRITICAL: Ensure we analyze only CLOSED candles
    current_time = int(time.time())
    
    # Check if last candle is actually closed
    last_candle = candles[-1]
    last_candle_close_time = last_candle["epoch"] + tf
    
    if current_time < last_candle_close_time:
        # Last candle still forming - check second to last if available
        if n < 2:
            return None
        current_idx = n - 2  # Use second to last (definitely closed)
        current_candle = candles[current_idx]
        prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    else:
        # Last candle is closed - use it
        current_idx = n - 1
        current_candle = candles[current_idx] 
        prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    if DEBUG:
        candle_time = datetime.fromtimestamp(current_candle["epoch"], timezone.utc)
        print(f"Analyzing CLOSED candle at index {current_idx}, time: {candle_time}")
    
    # Compute moving averages
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    # STRICT: Get trend state - ZERO TOLERANCE for ranging markets (especially M5 and below)
    trend_state = get_trend_state(candles, ma1, ma2, ma3, current_idx)
    if trend_state not in ["UPTREND", "DOWNTREND"]:
        if DEBUG:
            tf_minutes = tf // 60 if tf >= 60 else f"{tf}s"
            print(f"BLOCKED: {trend_state} detected on {tf_minutes} timeframe - NO RANGING SIGNALS ALLOWED")
        return None  # ZERO SIGNALS in ranging/consolidation/undefined states
    
    # CONFIRMED REJECTION PATTERN CHECK - Only from specified families
    is_rejection, pattern_type = is_rejection_candle(current_candle, prev_candle)
    if not is_rejection:
        return None
    
    # VALIDATED: Accept ANY rejection pattern (no restrictive family list)
    if not is_rejection:
        return None
    
    if DEBUG:
        print(f"Detected rejection pattern: {pattern_type} on CLOSED candle")
    
    # DYNAMIC ZONE CALCULATION - Adapts to market volatility
    base_zone_size = calculate_dynamic_zone_size(candles, current_idx)
    trend_multiplier = assess_trend_strength(candles, ma1, ma2, ma3, current_idx)
    adaptive_zone_size = base_zone_size * trend_multiplier
    
    if DEBUG:
        print(f"Dynamic zone: {adaptive_zone_size:.4f} ({base_zone_size:.4f} * {trend_multiplier:.2f})")
        print(f"Confirmed {pattern_type} rejection pattern on CLOSED candle")
    
    # DYNAMIC DSR ZONE VALIDATION - Captures tiny retests in strong trends
    current_close = current_candle["close"]
    current_low = current_candle["low"] 
    current_high = current_candle["high"]
    
    # Check key price levels against adaptive DSR zones
    in_zone_high, zone_high = is_in_dsr_zone(current_high, current_ma1, current_ma2, trend_state, adaptive_zone_size)
    in_zone_low, zone_low = is_in_dsr_zone(current_low, current_ma1, current_ma2, trend_state, adaptive_zone_size)
    in_zone_close, zone_close = is_in_dsr_zone(current_close, current_ma1, current_ma2, trend_state, adaptive_zone_size)
    
    # DSR Strategy Logic: Price must interact with MA1/MA2 dynamic levels
    signal_side = None
    reason = ""
    zone_info = ""
    
    # BUY SIGNAL: Uptrend + Price tests MA support + CONFIRMED rejection pattern
    if trend_state == "UPTREND":
        # Look for support tests (low or close touching MA support zones)
        if in_zone_low and "SUPPORT" in zone_low:
            signal_side = "BUY"
            zone_info = zone_low
            reason = f"DSR Uptrend - CONFIRMED {pattern_type} bounce from dynamic {zone_low.replace('_', ' ').lower()}"
        elif in_zone_close and "SUPPORT" in zone_close:
            signal_side = "BUY" 
            zone_info = zone_close
            reason = f"DSR Uptrend - CONFIRMED {pattern_type} hold above dynamic {zone_close.replace('_', ' ').lower()}"
        elif in_zone_high and "SUPPORT" in zone_high:  # High touched support (tiny retest)
            signal_side = "BUY"
            zone_info = zone_high
            reason = f"DSR Uptrend - CONFIRMED {pattern_type} tiny retest of dynamic {zone_high.replace('_', ' ').lower()}"
    
    # SELL SIGNAL: Downtrend + Price tests MA resistance + CONFIRMED rejection pattern
    elif trend_state == "DOWNTREND":
        # Look for resistance tests (high or close touching MA resistance zones)
        if in_zone_high and "RESISTANCE" in zone_high:
            signal_side = "SELL"
            zone_info = zone_high
            reason = f"DSR Downtrend - CONFIRMED {pattern_type} rejection at dynamic {zone_high.replace('_', ' ').lower()}"
        elif in_zone_close and "RESISTANCE" in zone_close:
            signal_side = "SELL"
            zone_info = zone_close  
            reason = f"DSR Downtrend - CONFIRMED {pattern_type} failure below dynamic {zone_close.replace('_', ' ').lower()}"
        elif in_zone_low and "RESISTANCE" in zone_low:  # Low touched resistance (tiny retest)
            signal_side = "SELL"
            zone_info = zone_low
            reason = f"DSR Downtrend - CONFIRMED {pattern_type} tiny retest of dynamic {zone_low.replace('_', ' ').lower()}"
    
    # Additional validation: Ensure price action makes sense for DSR
    if signal_side:
        # For BUY: Price should be above MA3 (trend filter)
        if signal_side == "BUY" and current_close <= current_ma3:
            if DEBUG:
                print(f"BUY signal rejected: price {current_close:.5f} below MA3 {current_ma3:.5f}")
            return None
        
        # For SELL: Price should be below MA3 (trend filter)
        if signal_side == "SELL" and current_close >= current_ma3:
            if DEBUG:
                print(f"SELL signal rejected: price {current_close:.5f} above MA3 {current_ma3:.5f}")
            return None
        
        if DEBUG:
            print(f"Valid DSR signal: {signal_side} at {zone_info} with CONFIRMED {pattern_type}")
        
        return {
            "symbol": shorthand,
            "tf": tf,
            "side": signal_side,
            "pattern": pattern_type,
            "trend": trend_state,
            "zone": zone_info,
            "zone_size": f"{adaptive_zone_size:.4f}",
            "reason": reason,
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
    
    return None

# -------------------------
# Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    """Create chart for signal visualization"""
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
    
    # Title
    trend_emoji = "📈" if signal_data["trend"] == "UPTREND" else "📉"
    ax.set_title(f"{signal_data['symbol']} - {signal_data['side']} Signal {trend_emoji}", 
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
    """Main analysis loop - ZERO TOLERANCE for ranging markets on M5 and below"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = get_timeframe_for_symbol(shorthand)
            
            # ENHANCED: Stricter ranging detection for M5 and below
            tf_minutes = tf // 60 if tf >= 60 else tf
            is_low_timeframe = tf <= 300  # M5 and below
            
            if DEBUG:
                tf_display = f"{tf_minutes}m" if tf >= 60 else f"{tf}s"
                ranging_policy = "ZERO TOLERANCE" if is_low_timeframe else "STRICT AVOIDANCE"
                print(f"Analyzing {shorthand} on {tf_display} - Ranging Policy: {ranging_policy}")
            
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
            
            # Create alert message with ranging policy info
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            trend_emoji = "📈" if signal["trend"] == "UPTREND" else "📉"
            policy_indicator = "🚫📊" if is_low_timeframe else "📊"  # Extra ranging warning for low timeframes
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} SIGNAL {policy_indicator}\n"
                      f"{trend_emoji} Trend: {signal['trend']} (RANGING AVOIDED)\n" 
                      f"🎨 Pattern: {signal['pattern']}\n"
                      f"⚡ DSR Zone: {signal['zone'].replace('_', ' ').title()}\n"
                      f"💰 Price: {signal['price']:.5f}\n"
                      f"📝 {signal['reason']}")
            
            chart_path = create_signal_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"TRENDING signal sent for {shorthand}: {signal['side']} (Ranging avoided)")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"Analysis complete. {signals_found} TRENDING signals found (Zero ranging signals).")

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

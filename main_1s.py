#!/usr/bin/env python3
"""
main_1s.py — High-Frequency Dynamic Support & Resistance (1s Indices Only)

Specialized implementation for V75(1s), V100(1s), V150(1s)
- Text-based signals only (no chart generation)
- Optimized for high-frequency data processing
- Reduced computational overhead
- Same strategy logic as main bot
"""

import os, json, time, traceback
from datetime import datetime, timezone
import websocket

# Telegram helpers (fallback to print)
try:
    from bot import send_telegram_message
except Exception:
    def send_telegram_message(token, chat_id, text): print("[1S-TEXT]", text); return True, "local"

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

# Optimized for 1s processing
CANDLES_N = 200  # Reduced from 480 for faster processing
TMPDIR = "/tmp"
ALERT_FILE = os.path.join(TMPDIR, "dsr_1s_last_sent.json")
MIN_CANDLES = 80  # Reduced minimum
LOOKBACK_BROKE_MA3 = 10  # Reduced lookback
PROXIMITY_THRESHOLD = 0.003  # Slightly higher for 1s volatility

# -------------------------
# 1-Second Indices Only
# -------------------------
SYMBOL_MAP_1S = {
    "V75(1s)": "1HZ75V",
    "V100(1s)": "1HZ100V", 
}

# -------------------------
# Persistence Functions
# -------------------------
def load_persist_1s():
    try:
        return json.load(open(ALERT_FILE))
    except Exception:
        return {}

def save_persist_1s(d):
    try:
        json.dump(d, open(ALERT_FILE,"w"))
    except Exception:
        pass

def already_sent_1s(shorthand, epoch, side):
    if TEST_MODE:
        return False
    rec = load_persist_1s().get(shorthand)
    return bool(rec and rec.get("epoch")==epoch and rec.get("side")==side)

def mark_sent_1s(shorthand, epoch, side):
    d = load_persist_1s()
    d[shorthand] = {"epoch": epoch, "side": side}
    save_persist_1s(d)

# -------------------------
# Moving Averages (Same as main bot)
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
# Rejection Pattern Detection (Same logic)
# -------------------------
def is_rejection_candle(candle, prev_candle=None):
    """Inclusive rejection detection for market reality"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range == 0 or total_range < 1e-9:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # DOJI FAMILY - expanded criteria
    if body_size <= total_range * 0.3:
        return True, "DOJI"
    
    # PINBAR FAMILY - relaxed criteria
    if upper_wick > 0 and upper_wick >= body_size * 0.8:
        return True, "PINBAR"
    
    if lower_wick > 0 and lower_wick >= body_size * 0.8:
        return True, "PINBAR"
    
    if upper_wick >= total_range * 0.4:
        return True, "PINBAR"
    
    if lower_wick >= total_range * 0.4:
        return True, "PINBAR"
    
    # ENGULFING FAMILY
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        prev_high, prev_low = prev_candle["high"], prev_candle["low"]
        
        if prev_c < prev_o and c > o:
            if o <= prev_c and c >= prev_o:
                return True, "BULL_ENGULF"
            elif c > prev_o * 0.7 + prev_c * 0.3:
                return True, "BULL_ENGULF"
            elif h >= prev_high * 0.9 and l <= prev_low * 1.1:
                return True, "BULL_ENGULF"
        
        if prev_c > prev_o and c < o:
            if o >= prev_c and c <= prev_o:
                return True, "BEAR_ENGULF"
            elif c < prev_o * 0.7 + prev_c * 0.3:
                return True, "BEAR_ENGULF"
            elif h >= prev_high * 0.9 and l <= prev_low * 1.1:
                return True, "BEAR_ENGULF"
    
    # TINY BODY FAMILY
    if body_size <= total_range * 0.25:
        return True, "TINY_BODY"
    
    # REJECTION PATTERNS
    if c <= h * 0.7 + l * 0.3:
        return True, "REJECTION"
    
    if c >= h * 0.3 + l * 0.7:
        return True, "REJECTION"
    
    if total_range > 0 and body_size <= total_range * 0.4:
        return True, "LONG_RANGE"
    
    return False, "NONE"

# -------------------------
# Analysis Functions (Same logic)
# -------------------------
def is_near_ma(price, ma_value, reference_price=None):
    """Check proximity to moving average"""
    if ma_value is None:
        return False
    
    if reference_price is None:
        reference_price = price
    
    threshold = reference_price * PROXIMITY_THRESHOLD
    return abs(price - ma_value) <= threshold

def analyze_trend(candles, ma1, ma2, ma3, current_idx):
    """Analyze current trend state"""
    if current_idx < 2 or not all(v is not None for v in [ma1[current_idx], ma2[current_idx], ma3[current_idx]]):
        return "UNDEFINED"
    
    current_ma1, current_ma2, current_ma3 = ma1[current_idx], ma2[current_idx], ma3[current_idx]
    current_price = candles[current_idx]["close"]
    
    if current_ma1 > current_ma2 > current_ma3 and current_price > current_ma3:
        return "UPTREND"
    
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
        
        if prev_close <= ma3[i-1] and curr_close > ma3[i]:
            return "BREAK_UP"
        
        if prev_close >= ma3[i-1] and curr_close < ma3[i]:
            return "BREAK_DOWN"
    
    return None

# -------------------------
# Optimized Data Fetching for 1s
# -------------------------
def fetch_candles_1s(sym, count=CANDLES_N):
    """Optimized candle fetching for 1-second timeframes"""
    for attempt in range(2):  # Reduced retry attempts
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=15)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                auth_resp = ws.recv()
            
            # Optimized request for 1s data
            request = {
                "ticks_history": sym,
                "style": "candles", 
                "granularity": 1,
                "count": count,
                "end": "latest"
            }
            
            ws.send(json.dumps(request))
            response = json.loads(ws.recv())
            ws.close()
            
            if DEBUG:
                print(f"[1S] Fetched {len(response.get('candles', []))} candles for {sym}")
            
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
                print(f"[1S] Attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(0.5)  # Shorter delay
    
    return []

# -------------------------
# Signal Detection (Same trend-aware logic)
# -------------------------
def detect_signal_1s(candles, shorthand):
    """Trend-aware signal detection for 1s indices"""
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    current_idx = n - 1
    current_candle = candles[current_idx]
    prev_candle = candles[current_idx - 1] if current_idx > 0 else None
    
    is_rejection, pattern_type = is_rejection_candle(current_candle, prev_candle)
    if not is_rejection:
        return None
    
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    current_high = current_candle["high"]
    current_low = current_candle["low"]
    current_close = current_candle["close"]
    
    trend = analyze_trend(candles, ma1, ma2, ma3, current_idx)
    ma3_break = check_ma3_break(candles, ma3, current_idx)
    
    signal_side = None
    reasons = []
    
    # Trend-aware logic (same as main bot)
    if trend == "UPTREND":
        if (is_near_ma(current_low, current_ma1, current_close) or 
            is_near_ma(current_low, current_ma2, current_close)):
            signal_side = "BUY"
            reasons.append(f"Uptrend continuation - {pattern_type} at MA support")
    
    elif trend == "DOWNTREND":
        if (is_near_ma(current_high, current_ma1, current_close) or 
            is_near_ma(current_high, current_ma2, current_close)):
            signal_side = "SELL"
            reasons.append(f"Downtrend continuation - {pattern_type} at MA resistance")
    
    elif trend == "CONSOLIDATION":
        if ma3_break == "BREAK_UP":
            if (is_near_ma(current_low, current_ma1, current_close) or 
                is_near_ma(current_low, current_ma2, current_close)):
                signal_side = "BUY"
                reasons.append(f"Bullish reversal - MA3 breakout + {pattern_type} retest")
        
        elif ma3_break == "BREAK_DOWN":
            if (is_near_ma(current_high, current_ma1, current_close) or 
                is_near_ma(current_high, current_ma2, current_close)):
                signal_side = "SELL"
                reasons.append(f"Bearish reversal - MA3 breakdown + {pattern_type} retest")
        
        else:
            if (is_near_ma(current_low, current_ma1, current_close) or 
                is_near_ma(current_low, current_ma2, current_close)):
                signal_side = "BUY"
                reasons.append(f"Consolidation - {pattern_type} rejection at MA support")
            
            elif (is_near_ma(current_high, current_ma1, current_close) or 
                  is_near_ma(current_high, current_ma2, current_close)):
                signal_side = "SELL"
                reasons.append(f"Consolidation - {pattern_type} rejection at MA resistance")
    
    # Additional reversal signals
    if signal_side is None:
        if (ma3_break == "BREAK_UP" and 
            (is_near_ma(current_low, current_ma1, current_close) or 
             is_near_ma(current_low, current_ma2, current_close))):
            signal_side = "BUY"
            reasons.append(f"Trend reversal - MA3 breakout + {pattern_type} support retest")
        
        elif (ma3_break == "BREAK_DOWN" and 
              (is_near_ma(current_high, current_ma1, current_close) or 
               is_near_ma(current_high, current_ma2, current_close))):
            signal_side = "SELL"
            reasons.append(f"Trend reversal - MA3 breakdown + {pattern_type} resistance retest")
    
    if signal_side and reasons:
        return {
            "symbol": shorthand,
            "side": signal_side,
            "reasons": reasons,
            "pattern": pattern_type,
            "trend": trend,
            "ma3_break": ma3_break,
            "price": current_close,
            "epoch": current_candle["epoch"]
        }
    
    return None

# -------------------------
# Text Alert Generation
# -------------------------
def create_text_alert(signal):
    """Create formatted text alert for 1s signals"""
    timestamp = datetime.fromtimestamp(signal["epoch"], timezone.utc).strftime("%H:%M:%S UTC")
    
    # Signal strength indicators
    strength_emoji = "🔥🔥🔥" if signal["ma3_break"] else "🔥🔥" if signal["trend"] != "CONSOLIDATION" else "🔥"
    
    # Direction emoji
    direction_emoji = "📈" if signal["side"] == "BUY" else "📉"
    
    alert_text = f"""⚡ HIGH-FREQUENCY SIGNAL ⚡

{direction_emoji} {signal['symbol']} - {signal['side']} {strength_emoji}

🎯 Pattern: {signal['pattern']}
📊 Trend: {signal['trend']}
💰 Price: {signal['price']}
⏰ Time: {timestamp}

📝 Analysis:
{chr(10).join(f"• {reason}" for reason in signal['reasons'])}

🔄 1-Second Timeframe Analysis
⚡ Text Alert (High-Freq Optimized)"""

    return alert_text

# -------------------------
# Main Execution Loop
# -------------------------
def run_1s_analysis():
    """Main analysis loop for 1s indices"""
    for shorthand, deriv_symbol in SYMBOL_MAP_1S.items():
        try:
            if DEBUG:
                print(f"[1S] Analyzing {shorthand} ({deriv_symbol})...")
            
            candles = fetch_candles_1s(deriv_symbol)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"[1S] Insufficient candles for {shorthand}: {len(candles)}")
                continue
            
            signal = detect_signal_1s(candles, shorthand)
            if not signal:
                continue
            
            # Check for duplicate alerts
            if already_sent_1s(shorthand, signal["epoch"], signal["side"]):
                if DEBUG:
                    print(f"[1S] Signal already sent for {shorthand}")
                continue
            
            # Create and send text alert
            alert_text = create_text_alert(signal)
            success, msg_id = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, alert_text)
            
            if success:
                mark_sent_1s(shorthand, signal["epoch"], signal["side"])
                if DEBUG:
                    print(f"[1S] Alert sent for {shorthand}: {signal['side']}")
                
        except Exception as e:
            if DEBUG:
                print(f"[1S] Error analyzing {shorthand}: {e}")

if __name__ == "__main__":
    try:
        if DEBUG:
            print(f"[1S] Starting high-frequency analysis at {datetime.now()}")
        run_1s_analysis()
    except Exception as e:
        print(f"[1S] Critical error: {e}")
        traceback.print_exc()

#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance (Trend Following Strategy)

STRATEGY LOGIC:
- MA3 break determines trend bias (above MA3 = buy bias, below MA3 = sell bias)  
- MA1 and MA2 act as dynamic support/resistance in trending markets
- Signals fire on rejection candlesticks at MA1/MA2 levels
- Immediate signal at candle close - no delays
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

CANDLES_N = 400
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

# Strategy parameters
MA_PROXIMITY_THRESHOLD = 0.008  # 0.8% proximity to MA levels
TREND_LOOKBACK = 10             # Periods to check trend alignment

# Default timeframe symbols only
SYMBOL_MAP = {
    "V10": "R_10",
    "V50": "R_50", 
    "V75": "R_75"
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

# Trend Analysis
def analyze_trend_bias(candles, ma1, ma2, ma3, current_idx):
    """Determine trend bias based on price position relative to MA3 and MA alignment"""
    if current_idx < TREND_LOOKBACK:
        return None, False
    
    current_price = candles[current_idx]["close"]
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx] 
    current_ma3 = ma3[current_idx]
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None, False
    
    # Determine bias based on price relative to MA3
    price_above_ma3 = current_price > current_ma3
    price_below_ma3 = current_price < current_ma3
    
    if not (price_above_ma3 or price_below_ma3):
        return None, False
    
    # Check MA alignment for trend strength
    uptrend_alignment = current_ma1 > current_ma2 > current_ma3
    downtrend_alignment = current_ma1 < current_ma2 < current_ma3
    
    # Validate trend consistency over recent periods
    trend_consistent = 0
    for i in range(max(0, current_idx - TREND_LOOKBACK), current_idx + 1):
        if all(v is not None for v in [ma1[i], ma2[i], ma3[i]]):
            if price_above_ma3 and ma1[i] >= ma2[i] >= ma3[i]:
                trend_consistent += 1
            elif price_below_ma3 and ma1[i] <= ma2[i] <= ma3[i]:
                trend_consistent += 1
    
    trend_strength = trend_consistent / (TREND_LOOKBACK + 1)
    
    if price_above_ma3 and uptrend_alignment and trend_strength > 0.6:
        return "BUY_BIAS", True
    elif price_below_ma3 and downtrend_alignment and trend_strength > 0.6:
        return "SELL_BIAS", True
    
    return None, False

# Rejection Pattern Detection
def is_rejection_pattern(candle, prev_candle=None):
    """Detect any rejection candlestick family"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # Doji family - any small body
    if body_size <= total_range * 0.35:
        return True, "DOJI"
    
    # Pinbar family - any significant wick
    if upper_wick >= total_range * 0.35:
        return True, "PINBAR"
    if lower_wick >= total_range * 0.35:
        return True, "PINBAR"
    
    # Engulfing patterns
    if prev_candle:
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        if prev_c < prev_o and c > o and o <= prev_c and c >= prev_o:
            return True, "BULL_ENGULF"
        if prev_c > prev_o and c < o and o >= prev_c and c <= prev_o:
            return True, "BEAR_ENGULF"
    
    # Tiny body candles
    if body_size <= total_range * 0.25:
        return True, "TINY_BODY"
    
    # Rejection closes
    if c <= h * 0.65 + l * 0.35:
        return True, "HIGH_REJECT"
    if c >= h * 0.35 + l * 0.65:
        return True, "LOW_REJECT"
    
    return False, "NONE"

# MA Level Interaction Check
def is_near_ma_levels(candle, ma1_val, ma2_val):
    """Check if candle interacts with MA1 or MA2 levels"""
    if ma1_val is None or ma2_val is None:
        return False
    
    high, low, close = candle["high"], candle["low"], candle["close"]
    
    # Check proximity to MA1
    ma1_distance = min(
        abs(high - ma1_val) / close,
        abs(low - ma1_val) / close,
        abs(close - ma1_val) / close
    )
    
    # Check proximity to MA2
    ma2_distance = min(
        abs(high - ma2_val) / close,
        abs(low - ma2_val) / close, 
        abs(close - ma2_val) / close
    )
    
    return ma1_distance <= MA_PROXIMITY_THRESHOLD or ma2_distance <= MA_PROXIMITY_THRESHOLD

# Data Fetching
def fetch_completed_candles(sym, tf):
    """Fetch completed candles only"""
    for attempt in range(3):
        try:
            ws = websocket.create_connection(DERIV_WS_URL, timeout=20)
            
            if DERIV_API_KEY:
                ws.send(json.dumps({"authorize": DERIV_API_KEY}))
                ws.recv()
            
            request = {
                "ticks_history": sym,
                "style": "candles",
                "granularity": tf,
                "count": CANDLES_N + 1,
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
                
                return candles_data[:-1]  # Remove forming candle
        
        except Exception as e:
            if DEBUG:
                print(f"Fetch attempt {attempt + 1} failed for {sym}: {e}")
            time.sleep(1)
    
    return []

# Main Signal Detection
def detect_trend_following_signal(candles, tf, shorthand):
    """Detect trend following signals based on updated strategy"""
    n = len(candles)
    if n < 80:
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
    
    # Step 1: Determine trend bias
    trend_bias, is_trending = analyze_trend_bias(candles, ma1, ma2, ma3, current_idx)
    if not is_trending or not trend_bias:
        return None
    
    # Step 2: Check if near MA1 or MA2 levels
    if not is_near_ma_levels(current_candle, current_ma1, current_ma2):
        return None
    
    # Step 3: Check for rejection pattern
    is_rejection, pattern_type = is_rejection_pattern(current_candle, prev_candle)
    if not is_rejection:
        return None
    
    # Step 4: Align signal with trend bias
    signal_side = None
    if trend_bias == "BUY_BIAS":
        # In uptrend, look for rejections at support (MA acting as support)
        signal_side = "BUY"
    elif trend_bias == "SELL_BIAS":
        # In downtrend, look for rejections at resistance (MA acting as resistance)
        signal_side = "SELL"
    
    if not signal_side:
        return None
    
    return {
        "symbol": shorthand,
        "tf": tf,
        "side": signal_side,
        "pattern": pattern_type,
        "trend_bias": trend_bias,
        "price": current_candle["close"],
        "idx": current_idx,
        "ma1": ma1,
        "ma2": ma2,
        "ma3": ma3,
        "candles": candles
    }

# Chart Generation
def create_trend_chart(signal_data):
    """Create chart for trend following signals"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1"], signal_data["ma2"], signal_data["ma3"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - 100)
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Plot candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        color = "#00FF00" if c >= o else "#FF0000"
        edge = "#00AA00" if c >= o else "#AA0000"
        
        ax.add_patch(Rectangle((i - 0.3/2, min(o, c)), 0.3, max(abs(c - o), 1e-9),
                              facecolor=color, edgecolor=edge, alpha=0.9, linewidth=1))
        ax.plot([i, i], [l, h], color=edge, linewidth=1, alpha=0.8)
    
    # Plot moving averages
    def plot_ma(ma_values, label, color):
        chart_ma = [ma_values[i] if i < len(ma_values) and ma_values[i] is not None else None 
                   for i in range(chart_start, n)]
        ax.plot(range(len(chart_candles)), chart_ma, color=color, linewidth=2, label=label)
    
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#FFFFFF")
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00BFFF")
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#FF6347")
    
    # Mark signal point
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_price = chart_candles[signal_chart_idx]["close"]
        color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker = "^" if signal_data["side"] == "BUY" else "v"
        ax.scatter([signal_chart_idx], [signal_price], color=color, marker=marker,
                  s=250, edgecolor="#FFFFFF", linewidth=2, zorder=10)
    
    tf_label = f"{signal_data['tf']}s" if signal_data['tf'] < 60 else f"{signal_data['tf']//60}m"
    ax.set_title(f"{signal_data['symbol']} {tf_label} - {signal_data['side']} TREND FOLLOWING", 
                fontsize=14, color='white', fontweight='bold')
    
    ax.legend(loc="upper left", facecolor='black', edgecolor='white')
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, dpi=150, bbox_inches="tight",
                facecolor='black', edgecolor='none')
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

# Main Execution
def run_trend_following_analysis():
    """Run trend following analysis for default symbols"""
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = TIMEFRAMES[0] if TIMEFRAMES else 300
            
            if DEBUG:
                print(f"Trend following analysis for {shorthand}...")
            
            candles = fetch_completed_candles(deriv_symbol, tf)
            if len(candles) < 80:
                continue
            
            signal = detect_trend_following_signal(candles, tf, shorthand)
            if not signal:
                continue
            
            # Check duplicates
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                continue
            
            # Create alert
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            
            caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']}\n"
                      f"📊 Pattern: {signal['pattern']}\n"
                      f"📈 Trend Bias: {signal['trend_bias']}\n"
                      f"💰 Price: {signal['price']}\n"
                      f"📝 Dynamic S/R Trend Following Signal")
            
            chart_path = create_trend_chart(signal)
            success, _ = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                if DEBUG:
                    print(f"✅ Trend following signal sent: {shorthand} {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")

if __name__ == "__main__":
    try:
        run_trend_following_analysis()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

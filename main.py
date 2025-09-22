#!/usr/bin/env python3
"""
main.py — Fixed Dynamic Support & Resistance Trading Bot

Enhanced DSR Strategy with immediate signal detection:
1. Break and Retest (Trend Reversal) - immediate signal at confirming candle close
2. Trend Following (Prevailing Trend) - immediate signal at confirming candle close

Signals fire IMMEDIATELY at the close of the confirmation candle as per documentation.
"""

import os, json, time, tempfile, traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
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
LOOKBACK_PERIOD = 25

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
    "Jump75": "JD75",
    "Jump100": "JD100"
}

SYMBOL_TF_MAP = {
    "V75(1s)": 1,
    "V100(1s)": 1,  
    "V150(1s)": 1,
    "V15(1s)": 1
}

# -------------------------
# Enums
# -------------------------
class ScenarioType(Enum):
    BREAK_AND_RETEST_BULLISH = "BREAK_AND_RETEST_BULLISH"
    BREAK_AND_RETEST_BEARISH = "BREAK_AND_RETEST_BEARISH"
    TREND_FOLLOWING_BULLISH = "TREND_FOLLOWING_BULLISH"
    TREND_FOLLOWING_BEARISH = "TREND_FOLLOWING_BEARISH"
    NO_CLEAR_SCENARIO = "NO_CLEAR_SCENARIO"

class CandlestickPattern(Enum):
    PIN_BAR_UP = "PIN_BAR_UP"
    PIN_BAR_DOWN = "PIN_BAR_DOWN"
    DOJI = "DOJI"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    NO_PATTERN = "NO_PATTERN"

@dataclass
class MovingAverages:
    ma1: float
    ma2: float
    ma3: float

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
    """Compute MAs: MA1=SMMA(HLC3,9), MA2=SMMA(Close,19), MA3=SMA(MA2,25)"""
    closes = [c["close"] for c in candles]
    hlc3 = [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    
    ma1 = smma_correct(hlc3, 9)
    ma2 = smma_correct(closes, 19)
    
    # MA3 = SMA of MA2 values
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

def create_ma_objects(ma1_list, ma2_list, ma3_list):
    """Convert MA arrays to MovingAverages objects"""
    mas = []
    min_len = min(len(ma1_list), len(ma2_list), len(ma3_list))
    
    for i in range(min_len):
        if all(v is not None for v in [ma1_list[i], ma2_list[i], ma3_list[i]]):
            mas.append(MovingAverages(ma1_list[i], ma2_list[i], ma3_list[i]))
        else:
            mas.append(None)
    
    return mas

# -------------------------
# Breakout Detection
# -------------------------
def detect_price_break_above_all_mas(candles, mas_objects, lookback=LOOKBACK_PERIOD):
    """Detect if price recently broke above ALL MAs (Factor 2 in documentation)"""
    if len(candles) < lookback or len(mas_objects) < lookback:
        return False, None
    
    # Look for break in recent history
    for i in range(-lookback, -1):
        if mas_objects[i] is None:
            continue
        
        price = candles[i]["close"]
        ma = mas_objects[i]
        
        # Check if price broke ABOVE all MAs
        if price > max(ma.ma1, ma.ma2, ma.ma3):
            return True, i
    
    return False, None

def detect_price_break_below_all_mas(candles, mas_objects, lookback=LOOKBACK_PERIOD):
    """Detect if price recently broke below ALL MAs (Factor 2 in documentation)"""
    if len(candles) < lookback or len(mas_objects) < lookback:
        return False, None
    
    # Look for break in recent history
    for i in range(-lookback, -1):
        if mas_objects[i] is None:
            continue
        
        price = candles[i]["close"]
        ma = mas_objects[i]
        
        # Check if price broke BELOW all MAs
        if price < min(ma.ma1, ma.ma2, ma.ma3):
            return True, i
    
    return False, None

def check_ma_rearrangement_bullish(mas_objects):
    """Factor 3: MAs rearranges - MA1 closest to price, then MA2, then MA3"""
    current_ma = mas_objects[-1]
    if current_ma is None:
        return False
    
    # After bullish break: MA1 > MA2 > MA3 (MA1 closest support)
    return current_ma.ma1 > current_ma.ma2 > current_ma.ma3

def check_ma_rearrangement_bearish(mas_objects):
    """Factor 3: MAs rearranges - MA1 closest to price, then MA2, then MA3"""
    current_ma = mas_objects[-1]
    if current_ma is None:
        return False
    
    # After bearish break: MA3 > MA2 > MA1 (MA1 closest resistance)
    return current_ma.ma3 > current_ma.ma2 > current_ma.ma1

def check_mas_smoothly_dispensed(mas_objects, lookback=10):
    """Factor 4: MAs starts to smoothly dispenses from each other"""
    if len(mas_objects) < lookback:
        return False
    
    recent_mas = mas_objects[-lookback:]
    valid_mas = [ma for ma in recent_mas if ma is not None]
    
    if len(valid_mas) < lookback * 0.6:
        return False
    
    # Check if MAs maintain good separation
    for ma in valid_mas[-3:]:  # Check last 3 periods
        sep1 = abs(ma.ma1 - ma.ma2)
        sep2 = abs(ma.ma2 - ma.ma3)
        if sep1 < 0.00001 or sep2 < 0.00001:
            return False
    
    return True

# -------------------------
# Scenario Detection
# -------------------------
def detect_scenario_type(candles, mas_objects):
    """Determine scenario based on documentation factors"""
    
    if len(mas_objects) < 10:
        return ScenarioType.NO_CLEAR_SCENARIO
    
    current_ma = mas_objects[-1]
    current_price = candles[-1]["close"]
    
    if current_ma is None:
        return ScenarioType.NO_CLEAR_SCENARIO
    
    # CHECK FOR BREAK AND RETEST SCENARIOS
    
    # Bullish Break and Retest Factors:
    # 1. Price was previously trending downwards (implied by break above)
    # 2. Price breaks above all MAs
    # 3. MAs rearranges - MA1 closest to price
    # 4. MAs smoothly dispenses
    
    broke_above, _ = detect_price_break_above_all_mas(candles, mas_objects)
    if broke_above:
        if check_ma_rearrangement_bullish(mas_objects):
            if check_mas_smoothly_dispensed(mas_objects):
                return ScenarioType.BREAK_AND_RETEST_BULLISH
    
    # Bearish Break and Retest Factors:
    # 1. Price was previously trending upwards (implied by break below)
    # 2. Price breaks below all MAs
    # 3. MAs rearranges - MA1 closest to price  
    # 4. MAs smoothly dispenses
    
    broke_below, _ = detect_price_break_below_all_mas(candles, mas_objects)
    if broke_below:
        if check_ma_rearrangement_bearish(mas_objects):
            if check_mas_smoothly_dispensed(mas_objects):
                return ScenarioType.BREAK_AND_RETEST_BEARISH
    
    # CHECK FOR TREND FOLLOWING SCENARIOS
    
    # Trend Following Bullish: Uptrend already established
    # Factor 1: MA1 close to price, followed by MA2 then MA3
    if (current_price > current_ma.ma1 > current_ma.ma2 > current_ma.ma3):
        return ScenarioType.TREND_FOLLOWING_BULLISH
    
    # Trend Following Bearish: Downtrend already established  
    # Factor 1: MA1 close to price, followed by MA2 then MA3
    if (current_ma.ma3 > current_ma.ma2 > current_ma.ma1 > current_price):
        return ScenarioType.TREND_FOLLOWING_BEARISH
    
    return ScenarioType.NO_CLEAR_SCENARIO

# -------------------------
# Pattern Detection
# -------------------------
def detect_candlestick_pattern(candle, recent_candles):
    """Detect rejection candlestick patterns"""
    if len(recent_candles) < 3:
        return CandlestickPattern.NO_PATTERN
    
    recent_bodies = [abs(c["close"] - c["open"]) for c in recent_candles[-5:]]
    avg_body = sum(recent_bodies) / len(recent_bodies) if recent_bodies else 0.001
    
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    if total_range <= 0:
        return CandlestickPattern.NO_PATTERN
    
    # Pin bar: wick must be at least 1.2 times body size
    min_wick_ratio = 1.2
    min_wick_size = max(body_size * min_wick_ratio, avg_body * 0.5)
    
    if lower_wick >= min_wick_size and upper_wick < body_size * 0.5:
        return CandlestickPattern.PIN_BAR_UP
    
    if upper_wick >= min_wick_size and lower_wick < body_size * 0.5:
        return CandlestickPattern.PIN_BAR_DOWN
    
    if body_size <= total_range * 0.15 and total_range > avg_body * 0.8:
        return CandlestickPattern.DOJI
    
    # Engulfing patterns
    if len(recent_candles) >= 2:
        prev_candle = recent_candles[-2]
        prev_o, prev_c = prev_candle["open"], prev_candle["close"]
        
        if (prev_c < prev_o and c > o and c > prev_o and o < prev_c and body_size > abs(prev_c - prev_o)):
            return CandlestickPattern.BULLISH_ENGULFING
        
        if (prev_c > prev_o and c < o and c < prev_o and o > prev_c and body_size > abs(prev_c - prev_o)):
            return CandlestickPattern.BEARISH_ENGULFING
    
    return CandlestickPattern.NO_PATTERN

def is_price_near_ma(price_level, ma_level, recent_candles, tolerance=0.8):
    """Check if price level is near MA level"""
    recent_ranges = [c["high"] - c["low"] for c in recent_candles[-10:]]
    avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0.001
    
    proximity_threshold = avg_range * tolerance
    return abs(price_level - ma_level) <= proximity_threshold

# -------------------------
# IMMEDIATE Signal Detection
# -------------------------
def detect_immediate_dsr_signal(candles, tf, shorthand):
    """
    Detect DSR signals IMMEDIATELY at candle close as per documentation.
    
    Key change: We analyze the CURRENT candle (just closed) for both:
    1. Rejection pattern formation
    2. Confirmation close
    
    Signal fires IMMEDIATELY when these conditions are met.
    """
    
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    # Compute MAs
    ma1_list, ma2_list, ma3_list = compute_mas(candles)
    mas_objects = create_ma_objects(ma1_list, ma2_list, ma3_list)
    
    if not mas_objects or mas_objects[-1] is None:
        return None
    
    # Determine scenario
    scenario = detect_scenario_type(candles, mas_objects)
    if scenario == ScenarioType.NO_CLEAR_SCENARIO:
        if DEBUG:
            print(f"{shorthand}: No clear scenario detected")
        return None
    
    current_candle = candles[-1]  # The candle that just closed
    current_ma = mas_objects[-1]
    
    # Look for rejection pattern in current candle AND confirmation close
    pattern = detect_candlestick_pattern(current_candle, candles)
    
    if pattern == CandlestickPattern.NO_PATTERN:
        return None
    
    # Check scenario-specific conditions
    signal_data = None
    
    if scenario == ScenarioType.BREAK_AND_RETEST_BULLISH:
        signal_data = check_bullish_retest_signal(current_candle, current_ma, pattern, candles)
        
    elif scenario == ScenarioType.BREAK_AND_RETEST_BEARISH:
        signal_data = check_bearish_retest_signal(current_candle, current_ma, pattern, candles)
        
    elif scenario == ScenarioType.TREND_FOLLOWING_BULLISH:
        signal_data = check_bullish_trend_signal(current_candle, current_ma, pattern, candles)
        
    elif scenario == ScenarioType.TREND_FOLLOWING_BEARISH:
        signal_data = check_bearish_trend_signal(current_candle, current_ma, pattern, candles)
    
    if not signal_data:
        return None
    
    # Build complete signal
    return {
        "symbol": shorthand,
        "tf": tf,
        "scenario": scenario.value,
        "side": signal_data["side"],
        "pattern": pattern.value,
        "ma_level": signal_data["ma_level"],
        "confirmation": signal_data["confirmation"],
        "price": current_candle["close"],
        "ma1": current_ma.ma1,
        "ma2": current_ma.ma2,
        "ma3": current_ma.ma3,
        "idx": n - 1,
        "candles": candles,
        "ma1_array": ma1_list,
        "ma2_array": ma2_list,
        "ma3_array": ma3_list
    }

def check_bullish_retest_signal(candle, ma, pattern, candles):
    """
    Break and Retest Bullish:
    - Factor 7: Price failed to break rearranged MAs (rejection at MA1/MA2)
    - Factor 8: Bullish confirmation close
    """
    
    valid_patterns = [CandlestickPattern.PIN_BAR_UP, CandlestickPattern.DOJI, 
                     CandlestickPattern.BULLISH_ENGULFING]
    
    if pattern not in valid_patterns:
        return None
    
    # Check if candle tested MA levels (low near MA1 or MA2)
    at_ma1 = is_price_near_ma(candle["low"], ma.ma1, candles)
    at_ma2 = is_price_near_ma(candle["low"], ma.ma2, candles)
    
    if not (at_ma1 or at_ma2):
        return None
    
    # Confirmation: bullish close (Factor 8)
    if candle["close"] <= candle["open"]:
        return None
    
    ma_level = "MA1" if at_ma1 else "MA2"
    return {
        "side": "BUY",
        "ma_level": ma_level,
        "confirmation": "BULLISH_CLOSE"
    }

def check_bearish_retest_signal(candle, ma, pattern, candles):
    """
    Break and Retest Bearish:
    - Factor 11: Price failed to break rearranged MAs (rejection at MA1/MA2)  
    - Factor 12: Bearish confirmation close
    """
    
    valid_patterns = [CandlestickPattern.PIN_BAR_DOWN, CandlestickPattern.DOJI,
                     CandlestickPattern.BEARISH_ENGULFING]
    
    if pattern not in valid_patterns:
        return None
    
    # Check if candle tested MA levels (high near MA1 or MA2)
    at_ma1 = is_price_near_ma(candle["high"], ma.ma1, candles)
    at_ma2 = is_price_near_ma(candle["high"], ma.ma2, candles)
    
    if not (at_ma1 or at_ma2):
        return None
    
    # Confirmation: bearish close (Factor 12)
    if candle["close"] >= candle["open"]:
        return None
    
    ma_level = "MA1" if at_ma1 else "MA2"
    return {
        "side": "SELL",
        "ma_level": ma_level,
        "confirmation": "BEARISH_CLOSE"
    }

def check_bullish_trend_signal(candle, ma, pattern, candles):
    """
    Trend Following Bullish:
    - Factor 3: Price failed to break MA1/MA2 (rejection pattern)
    - Factor 4: Bullish confirmation close
    """
    
    valid_patterns = [CandlestickPattern.PIN_BAR_UP, CandlestickPattern.DOJI,
                     CandlestickPattern.BULLISH_ENGULFING]
    
    if pattern not in valid_patterns:
        return None
    
    # Check if candle tested support levels (low near MA1 or MA2)
    at_ma1 = is_price_near_ma(candle["low"], ma.ma1, candles)
    at_ma2 = is_price_near_ma(candle["low"], ma.ma2, candles)
    
    if not (at_ma1 or at_ma2):
        return None
    
    # Confirmation: bullish close (Factor 4)
    if candle["close"] <= candle["open"]:
        return None
    
    ma_level = "MA1" if at_ma1 else "MA2"
    return {
        "side": "BUY",
        "ma_level": ma_level,
        "confirmation": "BULLISH_CLOSE"
    }

def check_bearish_trend_signal(candle, ma, pattern, candles):
    """
    Trend Following Bearish:
    - Factor 9: Price failed to break MA1/MA2 (rejection pattern)
    - Factor 6: Bearish confirmation close
    """
    
    valid_patterns = [CandlestickPattern.PIN_BAR_DOWN, CandlestickPattern.DOJI,
                     CandlestickPattern.BEARISH_ENGULFING]
    
    if pattern not in valid_patterns:
        return None
    
    # Check if candle tested resistance levels (high near MA1 or MA2)
    at_ma1 = is_price_near_ma(candle["high"], ma.ma1, candles)
    at_ma2 = is_price_near_ma(candle["high"], ma.ma2, candles)
    
    if not (at_ma1 or at_ma2):
        return None
    
    # Confirmation: bearish close (Factor 6)
    if candle["close"] >= candle["open"]:
        return None
    
    ma_level = "MA1" if at_ma1 else "MA2"
    return {
        "side": "SELL",
        "ma_level": ma_level,
        "confirmation": "BEARISH_CLOSE"
    }

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
# Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    """Create enhanced chart showing the immediate signal"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - LAST_N_CHART)
    chart_candles = candles[chart_start:]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Draw candlesticks
    for i, candle in enumerate(chart_candles):
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        
        if c >= o:
            body_color = "#00ff88"
            wick_color = "#00cc66"
        else:
            body_color = "#ff4444"
            wick_color = "#cc3333"
        
        ax.add_patch(Rectangle(
            (i - CANDLE_WIDTH/2, min(o, c)), 
            CANDLE_WIDTH, 
            max(abs(c - o), 1e-9),
            facecolor=body_color, 
            edgecolor=wick_color, 
            alpha=0.9,
            linewidth=1.2
        ))
        
        ax.plot([i, i], [l, h], color=wick_color, linewidth=1.5, alpha=0.8)
    
    # Plot moving averages
    def plot_ma(ma_values, label, color, linewidth=2.5):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        
        ax.plot(range(len(chart_candles)), chart_ma, 
                color=color, linewidth=linewidth, label=label,
                alpha=0.9, zorder=5)
    
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#ffffff")
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00bfff")  
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#ff6347")
    
    # Highlight signal candle
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_candle = chart_candles[signal_chart_idx]
        signal_price = signal_candle["close"]
        
        if signal_data["side"] == "BUY":
            marker_color = "#00ff88"
            marker_symbol = "^"
        else:
            marker_color = "#ff4444"
            marker_symbol = "v"
        
        ax.scatter([signal_chart_idx], [signal_price], 
                  color=marker_color, marker=marker_symbol, 
                  s=400, edgecolor="#ffffff", linewidth=3, zorder=15,
                  label=f'{signal_data["side"]} Signal')
        
        # Add signal annotation
       # Add signal annotation
        ax.annotate(f'SIGNAL\n{signal_data["confirmation"]}', 
                   xy=(signal_chart_idx, signal_price),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=marker_color, alpha=0.7))
    
    # Create title based on scenario
    scenario_emoji = {
        "BREAK_AND_RETEST_BULLISH": "🔄📈",
        "BREAK_AND_RETEST_BEARISH": "🔄📉", 
        "TREND_FOLLOWING_BULLISH": "📈🔥",
        "TREND_FOLLOWING_BEARISH": "📉🔥"
    }
    
    emoji = scenario_emoji.get(signal_data["scenario"], "📊")
    
    title = (f'{signal_data["symbol"]} - {signal_data["side"]} DSR SIGNAL {emoji}\n'
             f'Scenario: {signal_data["scenario"].replace("_", " ").title()}\n'
             f'Pattern: {signal_data["pattern"]} at {signal_data["ma_level"]} | '
             f'Confirmation: {signal_data["confirmation"]}')
    
    ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=20)
    
    # Legend
    legend = ax.legend(loc="upper left", frameon=True, facecolor='#1a1a1a', 
                      edgecolor='white', fontsize=10, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('white')
    
    # Grid and styling
    ax.grid(True, alpha=0.2, color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white', labelsize=9)
    
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.2)
    
    # Info box
    info_text = f"Scenario: {signal_data['scenario'].replace('_', ' ')}\nMA Level: {signal_data['ma_level']}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', alpha=0.8))
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, 
                dpi=200, 
                bbox_inches="tight", 
                facecolor='#0a0a0a',
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

def run_immediate_dsr_analysis():
    """Run DSR analysis with immediate signal detection"""
    signals_found = 0
    
    for shorthand, deriv_symbol in SYMBOL_MAP.items():
        try:
            tf = get_timeframe_for_symbol(shorthand)
            
            if DEBUG:
                tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
                print(f"Analyzing {shorthand} ({deriv_symbol}) on {tf_display}...")
            
            candles = fetch_candles(deriv_symbol, tf)
            if len(candles) < MIN_CANDLES:
                if DEBUG:
                    print(f"Insufficient candles for {shorthand}: {len(candles)}")
                continue
            
            # Detect immediate signal
            signal = detect_immediate_dsr_signal(candles, tf, shorthand)
            if not signal:
                continue
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            if already_sent(shorthand, tf, current_epoch, signal["side"]):
                if DEBUG:
                    print(f"Signal already sent for {shorthand}")
                continue
            
            tf_display = f"{tf}s" if tf < 60 else f"{tf//60}m"
            
            # Scenario descriptions
            scenario_desc = {
                "BREAK_AND_RETEST_BULLISH": "Break & Retest (Bullish Reversal)",
                "BREAK_AND_RETEST_BEARISH": "Break & Retest (Bearish Reversal)",
                "TREND_FOLLOWING_BULLISH": "Trend Following (Bullish Continuation)", 
                "TREND_FOLLOWING_BEARISH": "Trend Following (Bearish Continuation)"
            }
            
            scenario_text = scenario_desc.get(signal["scenario"], signal["scenario"])
            
            caption = (
                f"🎯 DSR SIGNAL DETECTED\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 {signal['symbol']} ({tf_display})\n"
                f"📈 {signal['side']} Signal\n"
                f"🎭 Scenario: {scenario_text}\n"
                f"🎨 Pattern: {signal['pattern']}\n"
                f"📍 Level: {signal['ma_level']} (Dynamic S/R)\n"
                f"✅ Confirmation: {signal['confirmation']}\n"
                f"💰 Entry Price: {signal['price']:.5f}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🔹 MA1: {signal['ma1']:.5f}\n"
                f"🔸 MA2: {signal['ma2']:.5f}\n"
                f"🔺 MA3: {signal['ma3']:.5f}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"⚡ IMMEDIATE SIGNAL\n"
                f"🎯 Fired at Candle Close"
            )
            
            chart_path = create_signal_chart(signal)
            
            success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
            
            if success:
                mark_sent(shorthand, tf, current_epoch, signal["side"])
                signals_found += 1
                if DEBUG:
                    print(f"IMMEDIATE DSR signal sent for {shorthand}: {signal['side']} - {scenario_text}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            if DEBUG:
                print(f"Error analyzing {shorthand}: {e}")
                traceback.print_exc()
    
    if DEBUG:
        print(f"Analysis complete. {signals_found} immediate DSR signals found.")
    
    return signals_found

def run_analysis():
    """Main wrapper function"""
    return run_immediate_dsr_analysis()

# -------------------------
# Testing Functions
# -------------------------
def validate_immediate_strategy():
    """Validate immediate signal detection"""
    print("Validating Immediate DSR Strategy...")
    print("=" * 50)
    
    # Create sample data with clear patterns
    sample_candles = []
    base_price = 100.0
    
    # Create uptrend with retest
    for i in range(80):
        if i < 30:  # Initial downtrend
            price_change = -0.05
        elif i < 35:  # Break above
            price_change = 0.3
        elif i < 50:  # MA rearrangement period
            price_change = 0.02
        elif i < 55:  # Pullback to test MAs
            price_change = -0.1
        else:  # Confirmation
            price_change = 0.05
        
        base_price += price_change
        
        # Create rejection pattern at test
        if i == 54:  # Rejection candle
            candle = {
                "epoch": 1640995200 + i * 300,
                "open": base_price - 0.02,
                "high": base_price + 0.01,
                "low": base_price - 0.15,  # Long lower wick (pin bar)
                "close": base_price + 0.03   # Bullish close
            }
        else:
            candle = {
                "epoch": 1640995200 + i * 300,
                "open": base_price - 0.02,
                "high": base_price + 0.05,
                "low": base_price - 0.05,
                "close": base_price
            }
        
        sample_candles.append(candle)
    
    # Test signal detection
    signal = detect_immediate_dsr_signal(sample_candles, 300, "TEST")
    
    if signal:
        print(f"✓ Signal detected: {signal['side']} - {signal['scenario']}")
        print(f"✓ Pattern: {signal['pattern']} at {signal['ma_level']}")
        print(f"✓ Confirmation: {signal['confirmation']}")
    else:
        print("✗ No signal detected in test data")
    
    print("Validation complete.")

def run_test_mode():
    """Run bot in test mode with sample data"""
    print("Running Immediate DSR Bot Test...")
    print("=" * 40)
    
    validate_immediate_strategy()
    
    print("\n" + "="*40)
    print("Test mode complete.")

if __name__ == "__main__":
    try:
        if DEBUG:
            print("🚀 Starting IMMEDIATE DSR Trading Bot")
            print("=" * 60)
            
            validate_immediate_strategy()
            print("\n" + "="*60 + "\n")
        
        if TEST_MODE:
            run_test_mode()
        else:
            run_analysis()
        
    except Exception as e:
        print(f"Critical error: {e}")
        if DEBUG:
            traceback.print_exc()
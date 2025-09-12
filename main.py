#!/usr/bin/env python3
"""
main.py — Dynamic Support & Resistance Trading Bot (Continuous Version 1)

Core DSR Strategy:
- MA crossovers and rearrangement confirm trend changes
- MA1 (SMMA HLC3-9) and MA2 (SMMA Close-19) = Dynamic S/R
- MA3 (SMA MA2-25) = Trend filter
- BUY BIAS: Only when MA1 > MA2 (bullish arrangement)
- SELL BIAS: Only when MA1 < MA2 (bearish arrangement)
- Rejection patterns at MA1/MA2 levels = Signals

Continuous 6-hour session with M5 and M10 timeframes
"""

import os, json, time, tempfile, traceback, threading, queue
from datetime import datetime, timezone, timedelta
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

DEBUG = os.getenv("DEBUG","0") == "1"
TEST_MODE = os.getenv("TEST_MODE","0") == "1"

# 6-hour runtime configuration
SESSION_DURATION = 6 * 60 * 60  # 6 hours in seconds
CANDLES_BUFFER_SIZE = 500  # Keep last 500 candles in memory
MIN_CANDLES = 50

CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

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
    "Jump50": "JD50", 
    "Jump100": "JD100",
}

# Timeframes: M5 (300s) and M10 (600s)
TIMEFRAMES = [300, 600]

# -------------------------
# Global State Management
# -------------------------
class ContinuousTradingBot:
    def __init__(self):
        self.ws = None
        self.running = False
        self.start_time = None
        self.symbol_data = {}  # Store candle data for each symbol and timeframe
        self.signal_queue = queue.Queue()
        self.last_signals = {}  # Cooldown tracking per symbol-timeframe
        
        # Initialize symbol data storage for both timeframes
        for symbol in SYMBOL_MAP.keys():
            self.symbol_data[symbol] = {}
            for tf in TIMEFRAMES:
                self.symbol_data[symbol][tf] = {
                    'candles': [],
                    'subscribed': False,
                    'last_candle_time': 0
                }
    
    def should_continue(self):
        """Check if we should continue running (within 6-hour limit)"""
        if not self.start_time:
            return True
        elapsed = time.time() - self.start_time
        return elapsed < SESSION_DURATION
    
    def add_candle(self, symbol, timeframe, candle_data):
        """Add new candle to symbol's data buffer"""
        if symbol not in self.symbol_data or timeframe not in self.symbol_data[symbol]:
            return
        
        candles = self.symbol_data[symbol][timeframe]['candles']
        
        # Convert to standard format
        candle = {
            "epoch": int(candle_data.get("epoch", 0)),
            "open": float(candle_data.get("open", 0)),
            "high": float(candle_data.get("high", 0)),
            "low": float(candle_data.get("low", 0)),
            "close": float(candle_data.get("close", 0))
        }
        
        # Avoid duplicates
        if candles and candles[-1]["epoch"] >= candle["epoch"]:
            return
        
        candles.append(candle)
        
        # Maintain buffer size
        if len(candles) > CANDLES_BUFFER_SIZE:
            candles.pop(0)
        
        self.symbol_data[symbol][timeframe]['last_candle_time'] = candle["epoch"]
        
        if DEBUG:
            tf_display = f"{timeframe//60}min" if timeframe >= 60 else f"{timeframe}s"
            print(f"Added candle for {symbol} {tf_display}: {candle['close']:.5f} at {candle['epoch']}")
        
        # Trigger signal analysis
        if len(candles) >= MIN_CANDLES:
            self.analyze_symbol(symbol, timeframe)
    
    def analyze_symbol(self, symbol, timeframe):
        """Analyze symbol for DSR signals"""
        candles = self.symbol_data[symbol][timeframe]['candles']
        if len(candles) < MIN_CANDLES:
            return
        
        signal = detect_signal(candles, timeframe, symbol)
        if not signal:
            return
        
        current_epoch = signal["candles"][signal["idx"]]["epoch"]
        
        # Cooldown check (30 minutes per symbol-timeframe combination)
        cooldown_key = f"{symbol}_{timeframe}_last_signal"
        last_time = self.last_signals.get(cooldown_key, 0)
        if current_epoch - last_time < 1800:  # 30 minutes
            return
        
        # Check if already sent (persistence)
        if already_sent(symbol, timeframe, current_epoch, signal["side"]):
            return
        
        self.last_signals[cooldown_key] = current_epoch
        self.signal_queue.put(signal)
    
    def process_signals(self):
        """Process queued signals in separate thread"""
        while self.running:
            try:
                signal = self.signal_queue.get(timeout=1)
                
                # Create and send signal
                tf_display = f"{signal['tf']//60}min" if signal['tf'] >= 60 else f"{signal['tf']}s"
                arrangement_emoji = "📈" if signal["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
                crossover_info = f" ({signal['crossover']})" if signal['crossover'] != "NONE" else ""
                
                caption = (f"🎯 {signal['symbol']} {tf_display} - {signal['side']} SIGNAL\n"
                          f"{arrangement_emoji} MA Setup: {signal['ma_arrangement'].replace('_', ' ')}{crossover_info}\n" 
                          f"🎨 Pattern: {signal['pattern']}\n"
                          f"📍 Level: {signal['ma_level']} Dynamic S/R\n"
                          f"💰 Price: {signal['price']:.5f}\n"
                          f"📊 Context: {signal['context']}")
                
                chart_path = create_signal_chart(signal)
                
                success, msg_id = send_telegram_photo(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, caption, chart_path)
                
                if success:
                    current_epoch = signal["candles"][signal["idx"]]["epoch"]
                    mark_sent(signal['symbol'], signal['tf'], current_epoch, signal["side"])
                    if DEBUG:
                        print(f"DSR signal sent for {signal['symbol']} {tf_display}: {signal['side']}")
                
                try:
                    os.unlink(chart_path)
                except:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"Error processing signal: {e}")
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get("msg_type", "")
            
            if msg_type == "candles":
                # Historical candles response
                symbol, timeframe = self.parse_request_id(data.get("req_id", ""))
                if symbol and timeframe:
                    candle_data = data.get("candles", [])
                    for candle in candle_data:
                        self.add_candle(symbol, timeframe, candle)
            
            elif msg_type == "ohlc":
                # Real-time candle update
                subscription = data.get("subscription", {})
                req_id = subscription.get("id", "")
                symbol, timeframe = self.parse_request_id(req_id)
                
                if symbol and timeframe:
                    ohlc = data.get("ohlc", {})
                    self.add_candle(symbol, timeframe, ohlc)
            
            elif data.get("error"):
                error_msg = data.get("error", {}).get("message", "Unknown error")
                if DEBUG:
                    print(f"WebSocket error: {error_msg}")
        
        except Exception as e:
            if DEBUG:
                print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        if DEBUG:
            print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        if DEBUG:
            print(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Attempt reconnection if still within session time
        if self.should_continue():
            if DEBUG:
                print("Attempting to reconnect...")
            time.sleep(5)  # Brief pause before reconnect
            self.connect()
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        if DEBUG:
            print("WebSocket connected successfully")
        
        # Authorize if API key provided
        if DERIV_API_KEY:
            auth_msg = {"authorize": DERIV_API_KEY}
            ws.send(json.dumps(auth_msg))
        
        # Subscribe to real-time data for all symbols and timeframes
        for shorthand, deriv_symbol in SYMBOL_MAP.items():
            for timeframe in TIMEFRAMES:
                # Get historical data first
                history_request = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": CANDLES_BUFFER_SIZE,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": timeframe,
                    "req_id": f"history_{shorthand}_{timeframe}"
                }
                ws.send(json.dumps(history_request))
                
                # Subscribe to real-time updates
                subscribe_request = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": 1,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": timeframe,
                    "subscribe": 1,
                    "req_id": f"subscribe_{shorthand}_{timeframe}"
                }
                ws.send(json.dumps(subscribe_request))
                
                self.symbol_data[shorthand][timeframe]['subscribed'] = True
                
                if DEBUG:
                    tf_display = f"{timeframe//60}min" if timeframe >= 60 else f"{timeframe}s"
                    print(f"Subscribed to {shorthand} {tf_display} ({deriv_symbol})")
    
    def parse_request_id(self, req_id):
        """Parse request ID to extract symbol and timeframe"""
        if not req_id:
            return None, None
        
        parts = req_id.split("_")
        if len(parts) >= 3:
            # Format: history_SYMBOL_TIMEFRAME or subscribe_SYMBOL_TIMEFRAME
            symbol = parts[1]
            try:
                timeframe = int(parts[2])
                if symbol in SYMBOL_MAP and timeframe in TIMEFRAMES:
                    return symbol, timeframe
            except ValueError:
                pass
        
        return None, None
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            self.ws = websocket.WebSocketApp(
                DERIV_WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run WebSocket connection
            self.ws.run_forever(
                ping_interval=30,  # Keep connection alive
                ping_timeout=10
            )
            
        except Exception as e:
            if DEBUG:
                print(f"Connection error: {e}")
            
            # Retry connection if within session time
            if self.should_continue():
                time.sleep(5)
                self.connect()
    
    def run(self):
        """Main run method - continuous 6-hour session"""
        self.running = True
        self.start_time = time.time()
        
        print(f"Starting 6-hour continuous trading session at {datetime.now()}")
        print(f"Monitoring symbols: {list(SYMBOL_MAP.keys())}")
        print(f"Timeframes: {[f'{tf//60}min' if tf >= 60 else f'{tf}s' for tf in TIMEFRAMES]}")
        
        # Start signal processing thread
        signal_thread = threading.Thread(target=self.process_signals, daemon=True)
        signal_thread.start()
        
        # Start WebSocket connection (blocks until connection ends)
        self.connect()
        
        # Session ended
        self.running = False
        print(f"6-hour session completed at {datetime.now()}")

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
# Rejection Detection
# -------------------------
def is_rejection_candle(candle):
    """Simple rejection detection - any candle showing rejection behavior"""
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # Any meaningful rejection signals
    has_upper_wick = upper_wick > 0
    has_lower_wick = lower_wick > 0
    has_small_body = body_size < total_range * 0.7  # Body less than 70% of range
    
    # Rejection criteria - very lenient
    if has_upper_wick and (upper_wick >= body_size * 0.5 or has_small_body):
        return True, "UPPER_REJECTION"
    
    if has_lower_wick and (lower_wick >= body_size * 0.5 or has_small_body):
        return True, "LOWER_REJECTION"
    
    if has_small_body and (has_upper_wick or has_lower_wick):
        return True, "SMALL_BODY_REJECTION"
    
    return False, "NONE"

# -------------------------
# MA Level Proximity
# -------------------------
def near_ma_levels(price, ma1_val, ma2_val):
    """Check if price is ACTUALLY near MA1 or MA2 - much stricter"""
    if ma1_val is None or ma2_val is None:
        return False, "NONE"
    
    # MUCH TIGHTER tolerance - 0.1% instead of 1%
    tolerance1 = abs(ma1_val) * 0.001  # 0.1% around MA1
    tolerance2 = abs(ma2_val) * 0.001  # 0.1% around MA2
    
    if abs(price - ma1_val) <= tolerance1:
        return True, "MA1"
    
    if abs(price - ma2_val) <= tolerance2:
        return True, "MA2"
    
    return False, "NONE"

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
# Core DSR Signal Detection
# -------------------------
def detect_signal(candles, tf, shorthand):
    """Complete DSR Strategy - All rules implemented correctly"""
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
    
    # DSR RULE 1: Determine bias from MA1/MA2 relationship
    if current_ma1 > current_ma2:
        bias = "BUY_BIAS"
    elif current_ma1 < current_ma2:
        bias = "SELL_BIAS"
    else:
        # MA1 = MA2, no clear bias
        return None
    
    # DSR RULE 2 & 3: Price position requirements
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
    
    # DSR RULE 6: No ranging markets
    is_ranging = check_ranging_market(candles, ma1, ma2, current_idx)
    if is_ranging:
        return None
    
    # Must have rejection pattern
    is_rejection, pattern_type = is_rejection_candle(current_candle)
    if not is_rejection:
        return None
    
    # DSR RULE 5: Price at or near MA1/MA2 (0.1% tolerance)
    ma1_tolerance = current_ma1 * 0.001  # 0.1%
    ma2_tolerance = current_ma2 * 0.001  # 0.1%
    
    # Check if any part of candle touched MA1 or MA2
    touched_ma1 = (abs(current_high - current_ma1) <= ma1_tolerance or 
                   abs(current_low - current_ma1) <= ma1_tolerance or 
                   abs(current_close - current_ma1) <= ma1_tolerance)
    
    touched_ma2 = (abs(current_high - current_ma2) <= ma2_tolerance or 
                   abs(current_low - current_ma2) <= ma2_tolerance or 
                   abs(current_close - current_ma2) <= ma2_tolerance)
    
    if not (touched_ma1 or touched_ma2):
        return None
    
    # Determine which MA level was touched
    if touched_ma1:
        ma_level = "MA1"
    else:
        ma_level = "MA2"
    
    # Generate signal based on bias
    if bias == "BUY_BIAS":
        signal_side = "BUY"
        context = "MA1 above MA2 - uptrend confirmed"
    else:
        signal_side = "SELL" 
        context = "MA1 below MA2 - downtrend confirmed"
    
    # Detect crossover for additional context
    has_crossover, crossover_type = detect_ma_crossover(ma1, ma2, current_idx)
    
    if DEBUG:
        tf_display = f"{tf//60}min" if tf >= 60 else f"{tf}s"
        print(f"VALID DSR: {signal_side} - {pattern_type} at {ma_level} - {shorthand} {tf_display}")
        print(f"  Price: {current_close:.5f}, MA1: {current_ma1:.5f}, MA2: {current_ma2:.5f}")
    
    return {
        "symbol": shorthand,
        "tf": tf,
        "side": signal_side,
        "pattern": pattern_type,
        "ma_level": ma_level,
        "ma_arrangement": "BULLISH_ARRANGEMENT" if bias == "BUY_BIAS" else "BEARISH_ARRANGEMENT",
        "crossover": crossover_type if has_crossover else "NONE",
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
# Chart Generation
# -------------------------
def create_signal_chart(signal_data):
    """Create chart for signal visualization"""
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
    signal_idx = signal_data["idx"]
    
    n = len(candles)
    chart_start = max(0, n - 180)  # Show last 180 candles
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
    
    # Title with MA arrangement info
    tf_display = f"{signal_data['tf']//60}min" if signal_data['tf'] >= 60 else f"{signal_data['tf']}s"
    arrangement_emoji = "📈" if signal_data["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
    ax.set_title(f"{signal_data['symbol']} {tf_display} - {signal_data['side']} DSR Signal {arrangement_emoji}", 
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
if __name__ == "__main__":
    try:
        bot = ContinuousTradingBot()
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

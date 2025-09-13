#!/usr/bin/env python3
"""
main.py — Continuous WebSocket DSR Trading Bot

Single persistent WebSocket connection for 6 hours with real-time analysis.
"""

import os, json, time, tempfile, traceback, threading, queue
from datetime import datetime
import websocket, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Telegram helpers
try:
    from bot import send_telegram_message, send_telegram_photo
except Exception:
    def send_telegram_message(token, chat_id, text): 
        print(f"[TEXT] {text}")
        return True, "local"
    def send_telegram_photo(token, chat_id, caption, photo): 
        print(f"[PHOTO] {caption}")
        return True, "local"

# Config
DERIV_API_KEY = os.getenv("DERIV_API_KEY", "").strip()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "1089").strip()
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DEBUG = os.getenv("DEBUG", "0") == "1"
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

SESSION_DURATION = 6 * 60 * 60
CANDLES_BUFFER_SIZE = 500
MIN_CANDLES = 50
CANDLE_WIDTH = 0.35
TMPDIR = tempfile.gettempdir()
ALERT_FILE = os.path.join(TMPDIR, "dsr_last_sent_main.json")

SYMBOL_MAP = {
    "V10": "R_10",
    "V25": "R_25", 
    "V50": "R_50",
    "V75": "R_75",
    "Jump10": "JD10",
    "Jump25": "JD25",
    "Jump50": "JD50",
    "Jump100": "JD100"
}

TIMEFRAMES = [300, 600]  # M5 and M10

class ContinuousBot:
    def __init__(self):
        self.ws = None
        self.running = False
        self.start_time = None
        self.symbol_data = {}
        self.signal_queue = queue.Queue()
        self.last_signals = {}
        self.candle_count = 0
        self.signal_count = 0
        
        for symbol in SYMBOL_MAP.keys():
            self.symbol_data[symbol] = {}
            for tf in TIMEFRAMES:
                self.symbol_data[symbol][tf] = {
                    'candles': [],
                    'last_update': 0
                }
    
    def should_continue(self):
        if not self.start_time:
            return True
        return (time.time() - self.start_time) < SESSION_DURATION
    
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        elapsed = ""
        if self.start_time:
            elapsed_sec = time.time() - self.start_time
            elapsed = f"[{elapsed_sec//3600:.0f}h{(elapsed_sec%3600)//60:.0f}m] "
        print(f"{timestamp} {elapsed}{level}: {message}")
    
    def add_candle(self, symbol, timeframe, candle_data):
        if symbol not in self.symbol_data or timeframe not in self.symbol_data[symbol]:
            return
        
        try:
            candle = {
                "epoch": int(candle_data.get("epoch", 0)),
                "open": float(candle_data.get("open", 0)),
                "high": float(candle_data.get("high", 0)),
                "low": float(candle_data.get("low", 0)),
                "close": float(candle_data.get("close", 0))
            }
            
            if candle["epoch"] <= 0 or candle["close"] <= 0:
                return
            
            candles = self.symbol_data[symbol][timeframe]['candles']
            
            if candles and candles[-1]["epoch"] >= candle["epoch"]:
                return
            
            candles.append(candle)
            self.candle_count += 1
            
            if len(candles) > CANDLES_BUFFER_SIZE:
                candles.pop(0)
            
            self.symbol_data[symbol][timeframe]['last_update'] = time.time()
            
            if DEBUG:
                tf_display = f"{timeframe//60}m"
                self.log(f"[{self.candle_count}] {symbol} {tf_display}: {candle['close']:.5f}")
            
            if len(candles) >= MIN_CANDLES:
                self.analyze_immediately(symbol, timeframe)
                
        except Exception as e:
            self.log(f"Error adding candle for {symbol} {timeframe}: {e}", "ERROR")
    
    def analyze_immediately(self, symbol, timeframe):
        try:
            candles = self.symbol_data[symbol][timeframe]['candles']
            signal = detect_signal(candles, timeframe, symbol)
            
            if not signal:
                return
            
            current_epoch = signal["candles"][signal["idx"]]["epoch"]
            cooldown_key = f"{symbol}_{timeframe}"
            last_time = self.last_signals.get(cooldown_key, 0)
            
            if current_epoch - last_time < 1800:  # 30 min cooldown
                return
            
            if already_sent(symbol, timeframe, current_epoch, signal["side"]):
                return
            
            self.last_signals[cooldown_key] = current_epoch
            self.signal_queue.put(signal)
            self.signal_count += 1
            
            tf_display = f"{timeframe//60}m"
            self.log(f"SIGNAL: {symbol} {tf_display} {signal['side']} - {signal['pattern']} at {signal['ma_level']}")
            
        except Exception as e:
            self.log(f"Analysis error for {symbol} {timeframe}: {e}", "ERROR")
    
    def signal_processor(self):
        while self.running:
            try:
                signal = self.signal_queue.get(timeout=2)
                self.send_signal(signal)
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Signal processor error: {e}", "ERROR")
    
    def send_signal(self, signal):
        try:
            tf_display = f"{signal['tf']//60}min"
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
                self.log(f"✅ SENT: {signal['symbol']} {tf_display} {signal['side']}")
            
            try:
                os.unlink(chart_path)
            except:
                pass
                
        except Exception as e:
            self.log(f"Error sending signal: {e}", "ERROR")
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("msg_type", "")
            
            if msg_type == "candles":
                req_id = data.get("req_id", "")
                if req_id.startswith("hist_"):
                    symbol, timeframe = self.parse_hist_req_id(req_id)
                    if symbol and timeframe:
                        candles_data = data.get("candles", [])
                        if DEBUG:
                            self.log(f"Loading {len(candles_data)} historical candles for {symbol} {timeframe//60}m")
                        
                        for candle_data in candles_data:
                            self.add_candle(symbol, timeframe, candle_data)
            
            elif msg_type == "ohlc":
                ohlc = data.get("ohlc", {})
                if ohlc:
                    # Add to all symbols/timeframes and let duplicate detection handle it
                    for symbol in SYMBOL_MAP.keys():
                        for tf in TIMEFRAMES:
                            self.add_candle(symbol, tf, ohlc)
            
            elif msg_type == "authorize":
                self.log("WebSocket authorized successfully")
            
            elif data.get("error"):
                error_info = data.get("error", {})
                self.log(f"API Error: {error_info}", "ERROR")
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.log(f"Message processing error: {e}", "ERROR")
    
    def parse_hist_req_id(self, req_id):
        try:
            parts = req_id.split("_")
            if len(parts) == 3:
                symbol = parts[1]
                timeframe = int(parts[2])
                if symbol in SYMBOL_MAP and timeframe in TIMEFRAMES:
                    return symbol, timeframe
        except:
            pass
        return None, None
    
    def on_error(self, ws, error):
        self.log(f"WebSocket error: {error}", "ERROR")
    
    def on_close(self, ws, close_status_code, close_msg):
        self.log(f"WebSocket closed: {close_status_code} - {close_msg}", "WARN")
        
        if self.should_continue():
            self.log("Reconnecting WebSocket...")
            time.sleep(5)
            self.connect()
    
    def on_open(self, ws):
        self.log("WebSocket connected - setting up subscriptions")
        
        if DERIV_API_KEY:
            ws.send(json.dumps({"authorize": DERIV_API_KEY}))
        
        for symbol_name, deriv_symbol in SYMBOL_MAP.items():
            for timeframe in TIMEFRAMES:
                
                hist_req = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": CANDLES_BUFFER_SIZE,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": timeframe,
                    "req_id": f"hist_{symbol_name}_{timeframe}"
                }
                ws.send(json.dumps(hist_req))
                
                live_req = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": 1,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": timeframe,
                    "subscribe": 1,
                    "req_id": f"live_{symbol_name}_{timeframe}"
                }
                ws.send(json.dumps(live_req))
                
                tf_display = f"{timeframe//60}m"
                self.log(f"Subscribed: {symbol_name} {tf_display} ({deriv_symbol})")
                
                time.sleep(0.2)
    
    def connect(self):
        if not self.should_continue():
            return
        
        try:
            self.ws = websocket.WebSocketApp(
                DERIV_WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
            
        except Exception as e:
            self.log(f"Connection error: {e}", "ERROR")
            if self.should_continue():
                time.sleep(10)
                self.connect()
    
    def status_monitor(self):
        while self.running and self.should_continue():
            try:
                elapsed = time.time() - self.start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                
                self.log(f"STATUS: {hours:02d}h{minutes:02d}m | Candles: {self.candle_count} | Signals: {self.signal_count}")
                time.sleep(300)  # Status every 5 minutes
                
            except Exception as e:
                self.log(f"Status monitor error: {e}", "ERROR")
                time.sleep(60)
    
    def run(self):
        self.running = True
        self.start_time = time.time()
        
        self.log("=" * 60)
        self.log("🚀 CONTINUOUS WEBSOCKET DSR BOT - 6 HOUR SESSION")
        self.log(f"Started: {datetime.now()}")
        self.log(f"Symbols: {list(SYMBOL_MAP.keys())}")
        self.log(f"Timeframes: {[f'{tf//60}min' for tf in TIMEFRAMES]}")
        self.log("=" * 60)
        
        try:
            signal_thread = threading.Thread(target=self.signal_processor, daemon=True)
            status_thread = threading.Thread(target=self.status_monitor, daemon=True)
            
            signal_thread.start()
            status_thread.start()
            
            self.log("Started background threads")
            self.connect()
            
        except KeyboardInterrupt:
            self.log("Bot stopped by user")
        except Exception as e:
            self.log(f"Critical error: {e}", "ERROR")
            traceback.print_exc()
        finally:
            self.running = False
            elapsed = time.time() - self.start_time
            
            self.log("=" * 60)
            self.log("✅ SESSION COMPLETE")
            self.log(f"Runtime: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m")
            self.log(f"Total Candles: {self.candle_count}")
            self.log(f"Total Signals: {self.signal_count}")
            self.log("=" * 60)

# Persistence functions
def load_persist():
    try:
        return json.load(open(ALERT_FILE))
    except Exception:
        return {}

def save_persist(d):
    try:
        json.dump(d, open(ALERT_FILE, "w"))
    except Exception:
        pass

def already_sent(shorthand, tf, epoch, side):
    if TEST_MODE:
        return False
    rec = load_persist().get(f"{shorthand}|{tf}")
    return bool(rec and rec.get("epoch") == epoch and rec.get("side") == side)

def mark_sent(shorthand, tf, epoch, side):
    d = load_persist()
    d[f"{shorthand}|{tf}"] = {"epoch": epoch, "side": side}
    save_persist(d)

# Moving averages
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

def detect_ma_crossover(ma1, ma2, current_idx, lookback=3):
    if current_idx < lookback:
        return False, "NONE"
    
    current_ma1 = ma1[current_idx]
    current_ma2 = ma2[current_idx]
    
    if current_ma1 is None or current_ma2 is None:
        return False, "NONE"
    
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i > 0 and i < len(ma1) and i < len(ma2):
            prev_ma1 = ma1[i-1]
            prev_ma2 = ma2[i-1]
            curr_ma1 = ma1[i]
            curr_ma2 = ma2[i]
            
            if all(v is not None for v in [prev_ma1, prev_ma2, curr_ma1, curr_ma2]):
                if prev_ma1 <= prev_ma2 and curr_ma1 > curr_ma2:
                    return True, "BULLISH_CROSSOVER"
                if prev_ma1 >= prev_ma2 and curr_ma1 < curr_ma2:
                    return True, "BEARISH_CROSSOVER"
    
    return False, "NONE"

def is_rejection_candle(candle):
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body_size = abs(c - o)
    total_range = h - l
    
    if total_range <= 0:
        return False, "NONE"
    
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    has_upper_wick = upper_wick > 0
    has_lower_wick = lower_wick > 0
    has_small_body = body_size < total_range * 0.7
    
    if has_upper_wick and (upper_wick >= body_size * 0.5 or has_small_body):
        return True, "UPPER_REJECTION"
    
    if has_lower_wick and (lower_wick >= body_size * 0.5 or has_small_body):
        return True, "LOWER_REJECTION"
    
    if has_small_body and (has_upper_wick or has_lower_wick):
        return True, "SMALL_BODY_REJECTION"
    
    return False, "NONE"

def check_ranging_market(candles, ma1, ma2, current_idx, lookback=10):
    if current_idx < lookback:
        return False
    
    ma2_touches = 0
    for i in range(current_idx - lookback + 1, current_idx + 1):
        if i < len(candles) and i < len(ma2) and ma2[i] is not None:
            candle = candles[i]
            ma2_val = ma2[i]
            if candle["low"] <= ma2_val <= candle["high"]:
                ma2_touches += 1
    
    return ma2_touches > 2

def detect_signal(candles, tf, shorthand):
    n = len(candles)
    if n < MIN_CANDLES:
        return None
    
    current_idx = n - 1
    current_candle = candles[current_idx]
    ma1, ma2, ma3 = compute_mas(candles)
    
    current_ma1 = ma1[current_idx] if current_idx < len(ma1) else None
    current_ma2 = ma2[current_idx] if current_idx < len(ma2) else None
    current_ma3 = ma3[current_idx] if current_idx < len(ma3) else None
    
    if not all(v is not None for v in [current_ma1, current_ma2, current_ma3]):
        return None
    
    current_close = current_candle["close"]
    current_high = current_candle["high"]
    current_low = current_candle["low"]
    
    if current_ma1 > current_ma2:
        bias = "BUY_BIAS"
    elif current_ma1 < current_ma2:
        bias = "SELL_BIAS"
    else:
        return None
    
    if bias == "BUY_BIAS" and current_close <= current_ma1:
        return None
    if bias == "SELL_BIAS" and current_close >= current_ma1:
        return None
    
    if current_ma1 > current_ma2:
        if current_ma2 < current_close < current_ma1:
            return None
    else:
        if current_ma1 < current_close < current_ma2:
            return None
    
    if check_ranging_market(candles, ma1, ma2, current_idx):
        return None
    
    is_rejection, pattern_type = is_rejection_candle(current_candle)
    if not is_rejection:
        return None
    
    ma1_tolerance = current_ma1 * 0.001
    ma2_tolerance = current_ma2 * 0.001
    
    touched_ma1 = (abs(current_high - current_ma1) <= ma1_tolerance or 
                   abs(current_low - current_ma1) <= ma1_tolerance or 
                   abs(current_close - current_ma1) <= ma1_tolerance)
    
    touched_ma2 = (abs(current_high - current_ma2) <= ma2_tolerance or 
                   abs(current_low - current_ma2) <= ma2_tolerance or 
                   abs(current_close - current_ma2) <= ma2_tolerance)
    
    if not (touched_ma1 or touched_ma2):
        return None
    
    ma_level = "MA1" if touched_ma1 else "MA2"
    
    if bias == "BUY_BIAS":
        signal_side = "BUY"
        context = "MA1 above MA2 - uptrend confirmed"
    else:
        signal_side = "SELL"
        context = "MA1 below MA2 - downtrend confirmed"
    
    has_crossover, crossover_type = detect_ma_crossover(ma1, ma2, current_idx)
    
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

def create_signal_chart(signal_data):
    candles = signal_data["candles"]
    ma1, ma2, ma3 = signal_data["ma1_array"], signal_data["ma2_array"], signal_data["ma3_array"]
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
        
        if c >= o:
            body_color, edge_color = "#00FF00", "#00AA00"
        else:
            body_color, edge_color = "#FF0000", "#AA0000"
        
        ax.add_patch(Rectangle((i - CANDLE_WIDTH/2, min(o, c)), CANDLE_WIDTH, 
                              max(abs(c - o), 1e-9), facecolor=body_color, 
                              edgecolor=edge_color, alpha=0.9, linewidth=1))
        ax.plot([i, i], [l, h], color=edge_color, linewidth=1.2, alpha=0.8)
    
    # MAs
    def plot_ma(ma_values, label, color, linewidth=2):
        chart_ma = []
        for i in range(chart_start, n):
            if i < len(ma_values) and ma_values[i] is not None:
                chart_ma.append(ma_values[i])
            else:
                chart_ma.append(None)
        ax.plot(range(len(chart_candles)), chart_ma, color=color, linewidth=linewidth, label=label, alpha=0.9)
    
    plot_ma(ma1, "MA1 (SMMA HLC3-9)", "#FFFFFF", 2)
    plot_ma(ma2, "MA2 (SMMA Close-19)", "#00BFFF", 2)
    plot_ma(ma3, "MA3 (SMA MA2-25)", "#FF6347", 2)
    
    # Signal marker
    signal_chart_idx = signal_idx - chart_start
    if 0 <= signal_chart_idx < len(chart_candles):
        signal_price = chart_candles[signal_chart_idx]["close"]
        marker_color = "#00FF00" if signal_data["side"] == "BUY" else "#FF0000"
        marker_symbol = "^" if signal_data["side"] == "BUY" else "v"
        ax.scatter([signal_chart_idx], [signal_price], color=marker_color, marker=marker_symbol, 
                  s=300, edgecolor="#FFFFFF", linewidth=3, zorder=10)
    
    # Title and styling
    tf_display = f"{signal_data['tf']//60}min"
    arrangement_emoji = "📈" if signal_data["ma_arrangement"] == "BULLISH_ARRANGEMENT" else "📉"
    ax.set_title(f"{signal_data['symbol']} {tf_display} - {signal_data['side']} DSR Signal {arrangement_emoji}", 
                fontsize=16, color='white', fontweight='bold', pad=20)
    
    legend = ax.legend(loc="upper left", frameon=True, facecolor='black', edgecolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.8)
    
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='white', labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    
    chart_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(chart_file.name, dpi=150, bbox_inches="tight", facecolor='black', edgecolor='none', pad_inches=0.1)
    plt.close()
    plt.style.use('default')
    
    return chart_file.name

if __name__ == "__main__":
    try:
        bot = ContinuousBot()
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()

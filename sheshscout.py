import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pro Stock Scanner - Chart Patterns + IPO Base", page_icon="📊", layout="wide")

# Enhanced CSS with Dark Mode Support
st.markdown("""<style>
.main-header{font-size:2.8rem;font-weight:700;background:linear-gradient(90deg,#00d4ff,#0066ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.6rem;font-weight:600;color:#fff;margin:1rem 0}
.metric-card{background:rgba(255,255,255,0.05);padding:1rem;border-radius:12px;border-left:4px solid #0066ff;margin:0.5rem 0;backdrop-filter:blur(10px)}
.pattern-badge{display:inline-block;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.85rem;font-weight:600;margin:0.2rem}
.pattern-bullish{background:#00ff88;color:#000}
.pattern-bearish{background:#ff4444;color:#fff}
.pattern-neutral{background:#ffaa00;color:#000}

/* Dark mode table styling */
div[data-testid="stDataFrame"] {background:rgba(0,0,0,0.3);border-radius:10px;padding:10px}
div[data-testid="stDataFrame"] table {color:#fff !important}
div[data-testid="stDataFrame"] th {background:rgba(0,102,255,0.3) !important;color:#fff !important;font-weight:600 !important}
div[data-testid="stDataFrame"] td {background:rgba(255,255,255,0.05) !important;color:#fff !important;border-color:rgba(255,255,255,0.1) !important}
div[data-testid="stDataFrame"] tr:hover td {background:rgba(0,102,255,0.2) !important}

/* Enhanced metrics */
div[data-testid="stMetricValue"] {font-size:1.8rem !important;font-weight:700 !important;color:#00d4ff !important}
div[data-testid="stMetricLabel"] {font-size:0.95rem !important;color:#aaa !important}

/* Better buttons */
.stButton>button {border-radius:10px;font-weight:600;border:2px solid #0066ff;background:linear-gradient(135deg,#0066ff,#00d4ff);color:#fff;transition:all 0.3s}
.stButton>button:hover {transform:translateY(-2px);box-shadow:0 5px 15px rgba(0,102,255,0.4)}

/* Sidebar styling */
section[data-testid="stSidebar"] {background:rgba(0,0,0,0.5);backdrop-filter:blur(15px)}
section[data-testid="stSidebar"] * {color:#fff !important}
</style>""", unsafe_allow_html=True)

# Comprehensive NSE Stock List (1000+ stocks from provided list)
NSE_STOCKS = [
    # From provided list
    'JSWHL', 'CREATIVE', 'KINGFA', 'NGLFINE', 'HITECHGEAR', 'BOSCHLTD', 'ABBOTINDIA', 'BANARISUG',
    'ESABINDIA', 'SEJALLTD', 'TASTYBITE', 'TEAMLEASE', 'ASTRAZEN', 'SUNDRMBRAK', 'RPGLIFE', 'SUMMITSEC',
    'JKCEMENT', 'FINEORG', 'YASHO', 'KICL', 'PILANIINVS', 'ELECTHERM', 'ZENITHEXPO', 'CHEVIOT',
    'WELINV', 'ACCELYA', 'AUTOAXLES', 'POLYCAB', 'RISHABH', 'ALICON', 'VOLTAMP', 'INNOVACAP',
    'BELLACASA', 'TVSSRICHAK', 'KIRLOSIND', 'KIRLPNU', 'RML', 'UNIVAFOODS', 'SANDESH', 'ORISSAMINE',
    'GLOSTERLTD', 'SUNCLAY', 'SABTNL', 'UNICHEMLAB', 'AKZOINDIA', 'HPIL', 'DIXON', 'RANEHOLDIN',
    'TVSHLTD', 'ORIENTBELL', 'LINCOLN', 'LTIM', 'SEAMECLTD', 'PERSISTENT', 'SALONA', 'ETHOSLTD',
    'PFIZER', 'PGHH', 'DHANUKA', 'EIMCOELECO', 'GALAXYSURF', 'CRISIL', 'MTARTECH', 'UEL',
    'MANORG', 'BFUTILITIE', 'POCL', 'WHEELS', 'RATNAMANI', 'GRINDWELL', 'NUVAMA', 'JINDRILL',
    'SAFARI', 'STEL', 'NUCLEUS', 'XPROINDIA', 'TATVA', 'PIXTRANS', 'MAPMYINDIA', 'INDIGOPNTS',
    # Nifty 50 + Popular Stocks
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH',
    'BAJFINANCE', 'WIPRO', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'ONGC',
    'TATAMOTORS', 'NTPC', 'POWERGRID', 'JSWSTEEL', 'M&M', 'TECHM', 'ADANIENT', 'ADANIPORTS',
    'COALINDIA', 'HINDALCO', 'TATASTEEL', 'BAJAJFINSV', 'DIVISLAB', 'DRREDDY', 'GRASIM',
    'CIPLA', 'BRITANNIA', 'EICHERMOT', 'HEROMOTOCO', 'APOLLOHOSP', 'INDUSINDBK', 'UPL',
    'BPCL', 'SBILIFE', 'HDFCLIFE', 'BAJAJ-AUTO', 'VEDL', 'TATACONSUM',
    'ADANIGREEN', 'GODREJCP', 'DABUR', 'PIDILITIND', 'HAVELLS', 'BERGEPAINT', 'SIEMENS',
    'AMBUJACEM', 'DLF', 'INDIGO', 'BANDHANBNK', 'CHOLAFIN', 'GAIL', 'TATAPOWER',
    'MUTHOOTFIN', 'MARICO', 'SAIL', 'AUROPHARMA', 'NMDC', 'LUPIN', 'ZYDUSLIFE',
    'PETRONET', 'PNB', 'BANKBARODA', 'RECLTD', 'CANBK', 'IRCTC', 'BEL', 'HAL',
    'HFCL', 'ZEEL', 'INDUSTOWER', 'YESBANK', 'TRENT', 'DMART', 'NAUKRI', 'ZOMATO',
    'PAYTM', 'PFC', 'MOTHERSON', 'ESCORTS', 'ASHOKLEY', 'TVSMOTOR', 'BALKRISIND', 'MRF',
    'APOLLOTYRE', 'CEAT', 'JUBLFOOD', 'PAGEIND', 'IGL', 'MGL', 'TORNTPOWER', 'TORNTPHARM',
    'LICI', 'ABFRL', 'VOLTAS', 'COFORGE', 'MPHASIS', 'OFSS', 'IRFC', 'RVNL', 'TIINDIA',
    'TATAELXSI', 'LTTS', 'BIOCON', 'ALKEM', 'CUMMINSIND', 'ABB', 'THERMAX',
    # Additional high-volume stocks
    'SYNGENE', 'APOLLOPIPE', 'RAILTEL', 'ROUTE', 'MACPOWER', 'CLEAN', 'VARROC', 'KAYNES'
]

# Add more stocks from provided list
ADDITIONAL_STOCKS = [
    '20MICRONS',

]

NSE_STOCKS.extend(ADDITIONAL_STOCKS)
NSE_STOCKS = list(set(NSE_STOCKS))  # Remove duplicates

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'DIXON': 'Electronics', 'POLYCAB': 'Cables', 'PERSISTENT': 'IT', 'COFORGE': 'IT', 'LTIM': 'IT',
    'ZOMATO': 'Tech', 'PAYTM': 'FinTech', 'NAUKRI': 'Tech', 'IRCTC': 'Travel', 'DMART': 'Retail'
}

@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    """Fetch comprehensive stock data"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="1y", interval="1d")
        
        if hist.empty or len(hist) < 50:
            return None
        
        info = ticker.info
        
        return {
            'symbol': symbol,
            'hist': hist,
            'closes': hist['Close'].values,
            'highs': hist['High'].values,
            'lows': hist['Low'].values,
            'volumes': hist['Volume'].values,
            'opens': hist['Open'].values,
            'info': info,
            'dates': hist.index
        }
    except Exception as e:
        return None

# ============ CHART PATTERN DETECTION ============

def detect_cup_and_handle(closes, highs, lows, lookback=60):
    """Detect Cup and Handle pattern"""
    if len(closes) < lookback:
        return False, 0
    
    recent = closes[-lookback:]
    recent_highs = highs[-lookback:]
    
    # Find the cup (U-shaped)
    max_idx = np.argmax(recent[:lookback//2])
    min_idx = np.argmax(recent[max_idx:max_idx+lookback//2]) + max_idx
    
    cup_depth = (recent[max_idx] - recent[min_idx]) / recent[max_idx]
    
    # Cup should be 12-33% deep
    if 0.12 <= cup_depth <= 0.33:
        # Check for handle (slight pullback)
        handle = recent[min_idx:]
        if len(handle) > 5:
            handle_high = np.max(handle[:len(handle)//2])
            handle_low = np.min(handle)
            handle_depth = (handle_high - handle_low) / handle_high
            
            if 0.05 <= handle_depth <= 0.15:
                confidence = 70 + (cup_depth * 50)
                return True, min(95, confidence)
    
    return False, 0

def detect_double_bottom(closes, lows, lookback=90):
    """Detect Double Bottom pattern"""
    if len(closes) < lookback:
        return False, 0
    
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]
    
    # Find two significant lows
    sorted_indices = np.argsort(recent_lows)
    
    if len(sorted_indices) < 2:
        return False, 0
    
    first_low_idx = sorted_indices[0]
    second_low_idx = sorted_indices[1]
    
    # They should be separated
    if abs(first_low_idx - second_low_idx) < 10:
        return False, 0
    
    # Lows should be similar (within 3%)
    low1 = recent_lows[first_low_idx]
    low2 = recent_lows[second_low_idx]
    
    if abs(low1 - low2) / low1 <= 0.03:
        # Check for neckline breakout
        peak_between = np.max(recent_closes[min(first_low_idx, second_low_idx):max(first_low_idx, second_low_idx)])
        current = recent_closes[-1]
        
        if current >= peak_between * 0.98:
            confidence = 75 + ((current - peak_between) / peak_between * 200)
            return True, min(95, confidence)
    
    return False, 0

def detect_ascending_triangle(highs, lows, lookback=60):
    """Detect Ascending Triangle"""
    if len(highs) < lookback:
        return False, 0
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Flat resistance (highs should be similar)
    high_std = np.std(recent_highs[-20:])
    high_mean = np.mean(recent_highs[-20:])
    
    if high_std / high_mean < 0.02:  # Flat top
        # Rising support (lows trending up)
        first_half_lows = np.mean(recent_lows[:lookback//2])
        second_half_lows = np.mean(recent_lows[lookback//2:])
        
        if second_half_lows > first_half_lows * 1.05:
            confidence = 70 + ((second_half_lows - first_half_lows) / first_half_lows * 100)
            return True, min(90, confidence)
    
    return False, 0

def detect_breakout(closes, highs, lookback=52):
    """Detect 52-week breakout"""
    if len(closes) < lookback:
        return False, 0
    
    current = closes[-1]
    year_high = np.max(highs[-lookback:-5])  # Exclude last 5 days
    
    if current >= year_high * 0.995:  # Within 0.5% of 52-week high
        confidence = 80 + ((current - year_high) / year_high * 200)
        return True, min(95, confidence)
    
    return False, 0

def detect_ipo_base(closes, dates, lookback_days=90):
    """Detect IPO Base pattern"""
    if len(closes) < 30:
        return False, 0, None
    
    # Check if stock is relatively new (within 2 years)
    days_since_first = (dates[-1] - dates[0]).days
    
    if days_since_first > 730:  # More than 2 years
        return False, 0, None
    
    # IPO base characteristics:
    # 1. Consolidation after listing
    # 2. Low volatility
    # 3. Tight price action
    
    recent = closes[-min(lookback_days, len(closes)):]
    
    # Calculate consolidation metrics
    high = np.max(recent)
    low = np.min(recent)
    range_pct = (high - low) / low * 100
    
    # Tight consolidation (15-25% range)
    if 15 <= range_pct <= 25:
        # Check for low volatility
        volatility = np.std(recent) / np.mean(recent)
        
        if volatility < 0.15:  # Low volatility
            # Check if near highs
            current = closes[-1]
            if current >= high * 0.95:
                confidence = 75 + (25 * (1 - volatility / 0.15))
                ipo_age_days = days_since_first
                return True, min(95, confidence), ipo_age_days
    
    return False, 0, None

def detect_vcp(closes, highs, lows, volumes, lookback=90):
    """Detect Volatility Contraction Pattern (Mark Minervini's VCP)"""
    if len(closes) < lookback:
        return False, 0
    
    recent_closes = closes[-lookback:]
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Divide into 3-4 contractions
    segment_size = lookback // 4
    contractions = []
    
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment = recent_closes[start:end]
        
        if len(segment) > 0:
            seg_high = np.max(segment)
            seg_low = np.min(segment)
            contraction = (seg_high - seg_low) / seg_high * 100
            contractions.append(contraction)
    
    # Check if contractions are getting tighter
    if len(contractions) >= 3:
        if contractions[-1] < contractions[-2] < contractions[-3]:
            # Contracting volatility
            avg_contraction = np.mean(contractions)
            if avg_contraction < 15:  # Tight pattern
                confidence = 70 + (30 * (1 - avg_contraction / 15))
                return True, min(90, confidence)
    
    return False, 0

def detect_flat_base(closes, lookback=30):
    """Detect Flat Base (3-5 weeks of tight consolidation)"""
    if len(closes) < lookback:
        return False, 0
    
    recent = closes[-lookback:]
    
    # Calculate range
    high = np.max(recent)
    low = np.min(recent)
    range_pct = (high - low) / high * 100
    
    # Flat base: 10-15% range
    if 8 <= range_pct <= 15:
        # Check if near highs
        current = closes[-1]
        if current >= high * 0.95:
            confidence = 75 + (range_pct / 15 * 20)
            return True, min(90, confidence)
    
    return False, 0

def detect_high_tight_flag(closes, lookback=60):
    """Detect High Tight Flag (strong uptrend + tight consolidation)"""
    if len(closes) < lookback:
        return False, 0
    
    # Strong uptrend (>100% gain in 4-8 weeks)
    older = closes[-lookback:-40] if len(closes) > 60 else closes[:-40]
    recent = closes[-40:]
    
    if len(older) > 0:
        gain = (recent[0] - np.mean(older)) / np.mean(older) * 100
        
        if gain >= 80:  # Strong prior move
            # Tight consolidation (3-5 weeks)
            consolidation = recent[-20:]
            con_high = np.max(consolidation)
            con_low = np.min(consolidation)
            con_range = (con_high - con_low) / con_high * 100
            
            if con_range < 20:  # Tight flag
                confidence = 80 + (gain / 100 * 15)
                return True, min(95, confidence)
    
    return False, 0

def detect_all_patterns(data):
    """Detect all chart patterns"""
    patterns = []
    
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    volumes = data['volumes']
    dates = data['dates']
    
    # Cup and Handle
    is_cup, conf = detect_cup_and_handle(closes, highs, lows)
    if is_cup:
        patterns.append({'name': '☕ Cup & Handle', 'confidence': conf, 'type': 'bullish'})
    
    # Double Bottom
    is_db, conf = detect_double_bottom(closes, lows)
    if is_db:
        patterns.append({'name': '⬇️⬇️ Double Bottom', 'confidence': conf, 'type': 'bullish'})
    
    # Ascending Triangle
    is_at, conf = detect_ascending_triangle(highs, lows)
    if is_at:
        patterns.append({'name': '📐 Ascending Triangle', 'confidence': conf, 'type': 'bullish'})
    
    # 52-Week Breakout
    is_bo, conf = detect_breakout(closes, highs)
    if is_bo:
        patterns.append({'name': '🚀 52W Breakout', 'confidence': conf, 'type': 'bullish'})
    
    # IPO Base
    is_ipo, conf, age = detect_ipo_base(closes, dates)
    if is_ipo:
        patterns.append({'name': f'🆕 IPO Base ({age}d)', 'confidence': conf, 'type': 'bullish'})
    
    # VCP
    is_vcp, conf = detect_vcp(closes, highs, lows, volumes)
    if is_vcp:
        patterns.append({'name': '📉 VCP', 'confidence': conf, 'type': 'bullish'})
    
    # Flat Base
    is_fb, conf = detect_flat_base(closes)
    if is_fb:
        patterns.append({'name': '➡️ Flat Base', 'confidence': conf, 'type': 'bullish'})
    
    # High Tight Flag
    is_htf, conf = detect_high_tight_flag(closes)
    if is_htf:
        patterns.append({'name': '🚩 High Tight Flag', 'confidence': conf, 'type': 'bullish'})
    
    return patterns

# ============ TECHNICAL INDICATORS ============

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:
        return 0
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return ema12 - ema26

def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = np.mean(prices[:period])
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_bb_position(prices, period=20):
    if len(prices) < period:
        return 50
    recent = prices[-period:]
    sma = np.mean(recent)
    std = np.std(recent)
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    if upper == lower:
        return 50
    return max(0, min(100, ((current - lower) / (upper - lower)) * 100))

def calculate_volume_multiple(volumes):
    if len(volumes) < 20:
        return 1.0
    current = volumes[-1]
    avg20 = np.mean(volumes[-20:])
    return current / avg20 if avg20 > 0 else 1.0

def detect_trend(prices):
    if len(prices) < 5:
        return 'Sideways'
    recent = prices[-5:]
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    if ups >= 4:
        return 'Strong Uptrend'
    elif ups >= 3:
        return 'Uptrend'
    elif ups <= 1:
        return 'Downtrend'
    return 'Sideways'

def calculate_relative_strength(closes, lookback=20):
    """Calculate price strength vs moving average"""
    if len(closes) < lookback:
        return 0
    ma = np.mean(closes[-lookback:])
    current = closes[-1]
    return ((current - ma) / ma) * 100

# ============ MAIN ANALYSIS FUNCTION ============

def analyze_stock(data):
    """Comprehensive stock analysis with patterns"""
    if not data:
        return None
    
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    volumes = data['volumes']
    
    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    change = ((price - prev_close) / prev_close) * 100
    
    # Technical indicators
    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    bb = calculate_bb_position(closes)
    vol = calculate_volume_multiple(volumes)
    trend = detect_trend(closes)
    rel_strength = calculate_relative_strength(closes)
    
    # Detect patterns
    patterns = detect_all_patterns(data)
    
    # Scoring
    score = 0
    criteria = []
    
    # 1. Chart Patterns (35 pts) - MOST IMPORTANT
    if len(patterns) > 0:
        pattern_score = min(35, len(patterns) * 12)
        score += pattern_score
        pattern_names = ", ".join([p['name'] for p in patterns[:3]])
        criteria.append(f'✅ Patterns: {pattern_names} [{pattern_score} pts]')
    else:
        criteria.append(f'❌ Patterns: None [0 pts]')
    
    # 2. RSI (15 pts)
    if 58 <= rsi <= 65:
        score += 15
        criteria.append(f'✅ RSI: Perfect ({rsi:.0f}) [15 pts]')
    elif 52 <= rsi <= 68:
        score += 12
        criteria.append(f'✅ RSI: Strong ({rsi:.0f}) [12 pts]')
    elif 35 <= rsi <= 42:
        score += 13
        criteria.append(f'✅ RSI: Oversold bounce ({rsi:.0f}) [13 pts]')
    else:
        criteria.append(f'❌ RSI: {rsi:.0f} [0 pts]')
    
    # 3. MACD (15 pts)
    if macd > 10:
        score += 15
        criteria.append(f'✅ MACD: Very Strong ({macd:.1f}) [15 pts]')
    elif macd > 5:
        score += 12
        criteria.append(f'✅ MACD: Strong ({macd:.1f}) [12 pts]')
    elif macd > 0:
        score += 8
        criteria.append(f'⚠ MACD: Bullish ({macd:.1f}) [8 pts]')
    else:
        criteria.append(f'❌ MACD: Bearish ({macd:.1f}) [0 pts]')
    
    # 4. Volume (15 pts)
    if vol >= 3.0:
        score += 15
        criteria.append(f'✅ Volume: Massive ({vol:.1f}x) [15 pts]')
    elif vol >= 2.0:
        score += 12
        criteria.append(f'✅ Volume: High ({vol:.1f}x) [12 pts]')
    elif vol >= 1.5:
        score += 8
        criteria.append(f'⚠ Volume: Above Avg ({vol:.1f}x) [8 pts]')
    else:
        criteria.append(f'❌ Volume: Low ({vol:.1f}x) [0 pts]')
    
    # 5. Trend (10 pts)
    if trend == 'Strong Uptrend':
        score += 10
        criteria.append(f'✅ Trend: Strong Uptrend [10 pts]')
    elif trend == 'Uptrend':
        score += 7
        criteria.append(f'✅ Trend: Uptrend [7 pts]')
    else:
        criteria.append(f'❌ Trend: {trend} [0 pts]')
    
    # 6. Daily Change (10 pts)
    if change >= 5:
        score += 10
        criteria.append(f'✅ Daily: Exceptional ({change:+.1f}%) [10 pts]')
    elif change >= 3:
        score += 8
        criteria.append(f'✅ Daily: Strong ({change:+.1f}%) [8 pts]')
    elif change >= 2:
        score += 5
        criteria.append(f'⚠ Daily: Good ({change:+.1f}%) [5 pts]')
    else:
        criteria.append(f'❌ Daily: {change:+.1f}% [0 pts]')
    
    # Rating
    if score >= 85:
        status = '🌟 EXCELLENT'
        rating = 'Excellent'
    elif score >= 75:
        status = '💎 VERY GOOD'
        rating = 'Very Good'
    elif score >= 65:
        status = '✅ GOOD'
        rating = 'Good'
    elif score >= 55:
        status = '👍 FAIR'
        rating = 'Fair'
    else:
        status = '⚠ WATCHLIST'
        rating = 'Watchlist'
    
    qualified = score >= 65
    met_count = len([c for c in criteria if '✅' in c])
    
    # Calculate potential
    potential_rs = max(20, price * 0.06)
    potential_pct = (potential_rs / price) * 100
    
    return {
        'symbol': data['symbol'],
        'price': price,
        'change': change,
        'potential_rs': potential_rs,
        'potential_pct': potential_pct,
        'rsi': rsi,
        'macd': macd,
        'bb': bb,
        'vol': vol,
        'trend': trend,
        'rel_strength': rel_strength,
        'score': score,
        'qualified': qualified,
        'status': status,
        'rating': rating,
        'criteria': criteria,
        'met_count': met_count,
        'sector': SECTOR_MAP.get(data['symbol'], 'Other'),
        'patterns': patterns,
        'pattern_count': len(patterns)
    }

# ============ STREAMLIT APP ============

st.markdown('<p class="main-header">📊 Pro Stock Scanner - Chart Patterns + IPO Base</p>', unsafe_allow_html=True)
st.markdown("**Advanced pattern recognition including Cup & Handle, Double Bottom, IPO Base, VCP, and more**")

# Sidebar
st.sidebar.header("⚙️ Scanner Configuration")

scan_mode = st.sidebar.radio("📊 Scan Mode", 
    ["Quick Scan (100 stocks)", "Medium Scan (200 stocks)", "Full Scan (500+ stocks)", "Custom List"])

if scan_mode == "Quick Scan (100 stocks)":
    stocks_to_scan = NSE_STOCKS[:100]
elif scan_mode == "Medium Scan (200 stocks)":
    stocks_to_scan = NSE_STOCKS[:200]
elif scan_mode == "Full Scan (500+ stocks)":
    stocks_to_scan = NSE_STOCKS[:500]
else:
    custom_input = st.sidebar.text_area("Enter NSE symbols (one per line)", 
        "RELIANCE\nTCS\nINFY\nHDFCBANK\nDIXON", height=150)
    stocks_to_scan = [s.strip().upper() for s in custom_input.split('\n') if s.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pattern Filters")

pattern_filter = st.sidebar.multiselect(
    "Show only stocks with patterns:",
    ["Cup & Handle", "Double Bottom", "Ascending Triangle", "52W Breakout", 
     "IPO Base", "VCP", "Flat Base", "High Tight Flag", "Any Pattern"],
    default=[]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Chart Patterns Detected")
st.sidebar.info("""
**Bullish Patterns:**
- ☕ **Cup & Handle**: Classic continuation
- ⬇️⬇️ **Double Bottom**: Reversal pattern
- 📐 **Ascending Triangle**: Breakout setup
- 🚀 **52W Breakout**: New highs
- 🆕 **IPO Base**: Post-listing base
- 📉 **VCP**: Volatility contraction
- ➡️ **Flat Base**: Tight consolidation
- 🚩 **High Tight Flag**: Strong momentum

**Scoring:**
- Patterns: 35 pts (most important!)
- RSI: 15 pts
- MACD: 15 pts
- Volume: 15 pts
- Trend: 10 pts
- Daily Change: 10 pts
""")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 START PATTERN SCAN", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning for Chart Patterns...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    results = []
    total = len(stocks_to_scan)
    failed = 0
    patterns_found = 0
    
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.info(f"📊 Analyzing **{symbol}**... ({idx+1}/{total})")
        
        data = fetch_stock_data(symbol)
        if data:
            analysis = analyze_stock(data)
            if analysis:
                results.append(analysis)
                if analysis['pattern_count'] > 0:
                    patterns_found += 1
        else:
            failed += 1
        
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        
        if (idx + 1) % 20 == 0 or idx == total - 1:
            qualified_count = len([r for r in results if r['qualified']])
            stats_placeholder.info(f"✅ Analyzed: {len(results)} | Qualified: {qualified_count} | Patterns Found: {patterns_found} | Failed: {failed}")
        
        time.sleep(0.1)
    
    st.session_state.scan_results = results
    st.session_state.scan_timestamp = datetime.now()
    
    status_text.success(f"✅ Scan complete! Found {patterns_found} stocks with patterns!")
    time.sleep(1)
    status_text.empty()
    stats_placeholder.empty()
    progress_bar.empty()
    
    st.rerun()

# Display Results
if st.session_state.scan_results:
    results = st.session_state.scan_results
    scan_time = st.session_state.scan_timestamp
    
    st.markdown("---")
    st.subheader(f"📈 Scan Results - Pattern Recognition")
    st.caption(f"Scanned at: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert to DataFrame
    df_data = []
    for r in results:
        pattern_str = ", ".join([p['name'] for p in r['patterns'][:2]]) if r['patterns'] else "None"
        
        df_data.append({
            'Symbol': r['symbol'],
            'Price': r['price'],
            'Change %': r['change'],
            'Patterns': pattern_str,
            'Pattern Count': r['pattern_count'],
            'RSI': r['rsi'],
            'MACD': 'Bull' if r['macd'] > 0 else 'Bear',
            'Vol': f"{r['vol']:.1f}x",
            'Trend': r['trend'],
            'Score': r['score'],
            'Rating': r['rating'],
            'Status': r['status'],
            'Sector': r['sector']
        })
    
    df = pd.DataFrame(df_data)
    
    # Statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_patterns = sum([r['pattern_count'] for r in results])
    stocks_with_patterns = len([r for r in results if r['pattern_count'] > 0])
    excellent = len([r for r in results if r['score'] >= 85])
    very_good = len([r for r in results if 75 <= r['score'] < 85])
    good = len([r for r in results if 65 <= r['score'] < 75])
    
    col1.metric("Total Scanned", len(df))
    col2.metric("🎯 With Patterns", stocks_with_patterns)
    col3.metric("📊 Total Patterns", total_patterns)
    col4.metric("🌟 Excellent", excellent)
    col5.metric("💎 Very Good", very_good)
    col6.metric("✅ Good", good)
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Excellent", "Very Good", "Good", "Fair", "Watchlist"])
    
    with filter_col2:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col3:
        pattern_count_filter = st.selectbox("Pattern Count", 
            ["All", "3+", "2+", "1+", "0"])
    
    with filter_col4:
        min_score_filter = st.number_input("Min Score", 0, 100, 0, 5)
    
    # Apply filters
    filtered_df = df.copy()
    
    if rating_filter != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == rating_filter]
    
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
    
    if pattern_count_filter == "3+":
        filtered_df = filtered_df[filtered_df['Pattern Count'] >= 3]
    elif pattern_count_filter == "2+":
        filtered_df = filtered_df[filtered_df['Pattern Count'] >= 2]
    elif pattern_count_filter == "1+":
        filtered_df = filtered_df[filtered_df['Pattern Count'] >= 1]
    elif pattern_count_filter == "0":
        filtered_df = filtered_df[filtered_df['Pattern Count'] == 0]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]
    
    # Apply pattern filter
    if pattern_filter:
        if "Any Pattern" in pattern_filter:
            filtered_df = filtered_df[filtered_df['Pattern Count'] > 0]
        else:
            # Filter by specific patterns
            filtered_results = []
            for r in results:
                for p in r['patterns']:
                    for pf in pattern_filter:
                        if pf in p['name']:
                            if r['symbol'] in filtered_df['Symbol'].values:
                                filtered_results.append(r['symbol'])
                                break
            if filtered_results:
                filtered_df = filtered_df[filtered_df['Symbol'].isin(filtered_results)]
    
    st.info(f"📊 Showing **{len(filtered_df)}** stocks (filtered from {len(df)} total)")
    
    # Sort by score
    filtered_df = filtered_df.sort_values('Score', ascending=False)
    
    # Display table
    st.subheader("📋 Stock Analysis Table")
    
    # Format and display
    display_df = filtered_df.copy()
    display_df['Price'] = display_df['Price'].apply(lambda x: f'₹{x:.2f}')
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f'{x:+.2f}%')
    display_df['RSI'] = display_df['RSI'].apply(lambda x: f'{x:.1f}')
    
    st.dataframe(display_df, use_container_width=True, height=600)
    
    # Pattern Summary
    st.markdown("---")
    st.subheader("📊 Pattern Distribution")
    
    pattern_stats = {}
    for r in results:
        for p in r['patterns']:
            pattern_name = p['name']
            if pattern_name not in pattern_stats:
                pattern_stats[pattern_name] = 0
            pattern_stats[pattern_name] += 1
    
    if pattern_stats:
        pattern_df = pd.DataFrame([
            {'Pattern': k, 'Count': v} 
            for k, v in sorted(pattern_stats.items(), key=lambda x: x[1], reverse=True)
        ])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(pattern_df, use_container_width=True)
        
        with col2:
            st.markdown("### Pattern Legend")
            st.markdown("""
            - ☕ Cup & Handle
            - ⬇️⬇️ Double Bottom
            - 📐 Ascending Triangle
            - 🚀 52W Breakout
            - 🆕 IPO Base
            - 📉 VCP (Volatility Contraction)
            - ➡️ Flat Base
            - 🚩 High Tight Flag
            """)
    
    # Detailed view
    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")
    
    if len(filtered_df) > 0:
        selected_symbol = st.selectbox("Select stock for details", filtered_df['Symbol'].tolist())
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)
        
        if selected_result:
            st.markdown(f"### {selected_symbol} - {selected_result['status']}")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Change", f"{selected_result['change']:+.2f}%")
            col4.metric("RSI", f"{selected_result['rsi']:.0f}")
            col5.metric("Volume", f"{selected_result['vol']:.1f}x")
            col6.metric("Patterns", selected_result['pattern_count'])
            
            if selected_result['patterns']:
                st.markdown("#### 🎯 Detected Patterns")
                pattern_cols = st.columns(min(3, len(selected_result['patterns'])))
                for idx, pattern in enumerate(selected_result['patterns'][:3]):
                    with pattern_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{pattern['name']}</h4>
                            <p>Confidence: <strong>{pattern['confidence']:.0f}%</strong></p>
                            <span class="pattern-badge pattern-{pattern['type']}">{pattern['type'].upper()}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Scoring Breakdown")
            for criterion in selected_result['criteria']:
                if '✅' in criterion:
                    st.success(criterion)
                elif '⚠' in criterion:
                    st.warning(criterion)
                else:
                    st.error(criterion)
    
    # Download
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered CSV",
            csv,
            f"pattern_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"pattern_scan_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure scanner and click 'START PATTERN SCAN' to begin")
    
    st.markdown("---")
    st.subheader("📚 Chart Pattern Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Continuation Patterns
        
        **☕ Cup and Handle**
        - U-shaped consolidation (12-33% deep)
        - Handle forms in upper half
        - Breakout above handle = buy signal
        - Best when 7-65 weeks in formation
        
        **📉 VCP (Volatility Contraction)**
        - Series of tightening pullbacks
        - Each correction smaller than last
        - 3-4 contractions ideal
        - Breakout on volume
        
        **➡️ Flat Base**
        - 5-8 weeks of tight consolidation
        - 8-15% price range
        - Near 52-week highs
        - Low volatility
        
        **🚩 High Tight Flag**
        - 100%+ gain in 4-8 weeks
        - Tight 3-5 week consolidation
        - <25% pullback
        - Rare but powerful
        """)
    
    with col2:
        st.markdown("""
        ### 🔄 Reversal Patterns
        
        **⬇️⬇️ Double Bottom**
        - Two similar lows (within 3%)
        - Separated by 4+ weeks
        - Neckline breakout = buy
        - Volume increases on breakout
        
        **📐 Ascending Triangle**
        - Flat resistance line
        - Rising support line
        - Breakout above resistance
        - Bullish pattern
        
        **🚀 52-Week Breakout**
        - New 52-week highs
        - Strong momentum
        - Institutional interest
        - Volume confirmation
        
        **🆕 IPO Base**
        - First consolidation after IPO
        - 15-25% range ideal
        - Low volatility
        - 8-12 weeks optimal
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Trading Tips
    
    1. **Best Patterns**: Cup & Handle, VCP, IPO Base (when properly formed)
    2. **Volume**: Always confirm breakouts with 40-50% above average volume
    3. **Risk Management**: Set stop loss 5-8% below entry
    4. **Position Sizing**: Risk only 1-2% of capital per trade
    5. **Market Condition**: Patterns work best in uptrending markets
    6. **Multiple Patterns**: Stocks with 2-3 patterns have higher success rate
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#aaa;'>
<p><strong>Pro Stock Scanner v2.0</strong> | Advanced Pattern Recognition</p>
<p style='font-size:0.85rem;'>⚠️ Educational purposes only. Patterns are probabilistic, not guaranteed. Always do your own research.</p>
</div>
""", unsafe_allow_html=True)

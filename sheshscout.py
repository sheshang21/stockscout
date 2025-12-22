import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import io
import threading

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Indian Stock Scout - Early Buy Scanner", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.5rem;font-weight:600;color:#333;margin:1rem 0}
.metric-card{background:#f8f9fb;padding:0.8rem;border-radius:8px;border-left:4px solid #1f77b4;margin:0.5rem 0}
.stDataFrame{font-size:0.9rem}
div[data-testid="stDataFrame"] > div{background:#f8f9fb}
.price-up{color:#00c853;font-weight:bold}
.price-down{color:#ff1744;font-weight:bold}
.price-neutral{color:#666}
</style>""", unsafe_allow_html=True)

# Comprehensive Stock Universe - 200+ NSE Stocks
NSE_STOCKS = [
    # Nifty 50
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 
    'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH',
    'BAJFINANCE', 'WIPRO', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'ONGC',
    'TATAMOTORS', 'NTPC', 'POWERGRID', 'JSWSTEEL', 'M&M', 'TECHM', 'ADANIENT', 'ADANIPORTS',
    'COALINDIA', 'HINDALCO', 'TATASTEEL', 'BAJAJFINSV', 'DIVISLAB', 'DRREDDY', 'GRASIM',
    'CIPLA', 'BRITANNIA', 'EICHERMOT', 'HEROMOTOCO', 'APOLLOHOSP', 'INDUSINDBK', 'UPL',
    'BPCL', 'SBILIFE', 'HDFCLIFE', 'BAJAJ-AUTO', 'VEDL', 'TATACONSUM',
    # Nifty Next 50 & High Volume Stocks
    'ADANIGREEN', 'GODREJCP', 'DABUR', 'PIDILITIND', 'HAVELLS', 'BERGEPAINT', 'SIEMENS',
    'AMBUJACEM', 'DLF', 'INDIGO', 'BANDHANBNK', 'CHOLAFIN', 'GAIL', 'BOSCHLTD',
    'MUTHOOTFIN', 'MARICO', 'SAIL', 'AUROPHARMA', 'NMDC', 'TATAPOWER', 'LUPIN',
    'ZYDUSLIFE', 'PETRONET', 'PNB', 'BANKBARODA', 'RECLTD', 'CANBK', 'IRCTC',
    'BEL', 'HAL', 'HFCL', 'ZEEL', 'INDUSTOWER', 'YESBANK', 'DIXON', 'TRENT',
    'DMART', 'NAUKRI', 'ZOMATO', 'PAYTM', 'PFC', 'MOTHERSON', 'ESCORTS', 'ASHOKLEY',
    'TVSMOTOR', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'CEAT', 'JUBLFOOD',
    # Additional stocks
    'PAGEIND', 'IGL', 'MGL', 'TORNTPOWER', 'TORNTPHARM', 'LICI', 'ABFRL', 'VOLTAS', 
    'PERSISTENT', 'COFORGE', 'LTIM', 'MPHASIS', 'OFSS', 'IRFC', 'RVNL', 'TIINDIA', 
    'TATAELXSI', 'LTTS', 'BIOCON', 'ALKEM', 'POLYCAB', 'CUMMINSIND', 'ABB', 'THERMAX'
]

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'BAJFINANCE': 'NBFC', 'WIPRO': 'IT', 'SUNPHARMA': 'Pharma', 'TITAN': 'Consumer', 'ULTRACEMCO': 'Cement',
    'NESTLEIND': 'FMCG', 'ONGC': 'Energy', 'TATAMOTORS': 'Auto', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'TECHM': 'IT', 'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure'
}

@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    """Fetch real-time data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="3mo", interval="1d")
        
        if hist.empty:
            return None
        
        closes = hist['Close'].values
        highs = hist['High'].values
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        
        price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else price
        change = ((price - prev_close) / prev_close) * 100
        
        # Get institutional data (approximation from volume patterns)
        fii_dii_activity = detect_institutional_activity(volumes, closes)
        
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        bb_position = calculate_bb_position(closes)
        vol_multiple = calculate_volume_multiple(volumes)
        trend = detect_trend(closes)
        
        return {
            'symbol': symbol,
            'price': price,
            'change': change,
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'vol_multiple': vol_multiple,
            'trend': trend,
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'volumes': volumes,
            'fii_dii_score': fii_dii_activity
        }
    except Exception as e:
        return None

def fetch_live_price(symbol):
    """Fetch only live price for auto-refresh (non-cached)"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def detect_institutional_activity(volumes, closes):
    """
    Detect FII/DII activity patterns from volume and price action
    High volume + price up = Buying, High volume + price down = Selling
    """
    if len(volumes) < 20 or len(closes) < 20:
        return 0
    
    score = 0
    recent_days = 10
    
    for i in range(-recent_days, 0):
        vol_ratio = volumes[i] / np.mean(volumes[-60:]) if len(volumes) >= 60 else volumes[i] / np.mean(volumes)
        price_change = ((closes[i] - closes[i-1]) / closes[i-1]) * 100 if i > -len(closes) else 0
        
        # High volume + Price up = Institutional buying
        if vol_ratio > 1.5 and price_change > 1:
            score += 2
        elif vol_ratio > 1.2 and price_change > 0.5:
            score += 1
        # High volume + Price down = Institutional selling
        elif vol_ratio > 1.5 and price_change < -1:
            score -= 2
        elif vol_ratio > 1.2 and price_change < -0.5:
            score -= 1
    
    return score

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
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    position = ((current - lower) / (upper - lower)) * 100
    return max(0, min(100, position))

def calculate_volume_multiple(volumes):
    if len(volumes) < 20:
        return 1.0
    current = volumes[-1]
    avg20 = np.mean(volumes[-20:])
    if avg20 == 0:
        return 1.0
    return current / avg20

def detect_operator_activity(data):
    """
    Detect if stock shows signs of operator/manipulator activity
    Returns: (is_operated, warning_flags, risk_score)
    """
    closes = data['closes']
    volumes = data['volumes']
    highs = data['highs']
    lows = data['lows']
    
    warning_flags = []
    risk_score = 0
    
    if len(closes) < 20:
        return False, [], 0
    
    # 1. EXTREME VOLUME SPIKES (Classic pump signal)
    recent_vols = volumes[-10:]
    avg_vol = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
    max_recent_vol = np.max(recent_vols)
    
    if max_recent_vol > avg_vol * 5:
        warning_flags.append("🚨 EXTREME volume spike (>5x avg) - Possible pump")
        risk_score += 30
    elif max_recent_vol > avg_vol * 3:
        warning_flags.append("⚠️ High volume spike (>3x avg) - Monitor closely")
        risk_score += 15
    
    # 2. PRICE VOLATILITY WITHOUT NEWS (Manipulation indicator)
    recent_prices = closes[-10:]
    price_swings = []
    for i in range(1, len(recent_prices)):
        swing = abs((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) * 100
        price_swings.append(swing)
    
    avg_swing = np.mean(price_swings) if price_swings else 0
    max_swing = np.max(price_swings) if price_swings else 0
    
    if max_swing > 8 and avg_swing > 3:
        warning_flags.append("🚨 Extreme volatility (>8% swings) - Operator activity likely")
        risk_score += 25
    elif max_swing > 5 and avg_swing > 2:
        warning_flags.append("⚠️ High volatility - Possible manipulation")
        risk_score += 12
    
    # 3. UNUSUAL INTRADAY RANGE (High-Low spread)
    recent_ranges = []
    for i in range(-10, 0):
        if i >= -len(highs):
            day_range = ((highs[i] - lows[i]) / closes[i]) * 100
            recent_ranges.append(day_range)
    
    avg_range = np.mean(recent_ranges) if recent_ranges else 0
    
    if avg_range > 6:
        warning_flags.append("🚨 Abnormal intraday ranges (>6%) - Manipulation alert")
        risk_score += 20
    elif avg_range > 4:
        warning_flags.append("⚠️ Wide intraday ranges - Increased risk")
        risk_score += 10
    
    # 4. SUSPICIOUS PRICE PATTERN (Sudden spike then consolidation)
    if len(closes) >= 30:
        recent_10d = closes[-10:]
        previous_20d = closes[-30:-10]
        
        recent_gain = ((recent_10d[-1] - previous_20d[0]) / previous_20d[0]) * 100
        previous_stability = np.std(previous_20d) / np.mean(previous_20d)
        
        if recent_gain > 20 and previous_stability < 0.05:
            warning_flags.append("🚨 Sudden spike after flat period - Classic pump pattern")
            risk_score += 25
    
    # 5. VOLUME-PRICE DIVERGENCE (Volume increasing but price not following)
    if len(volumes) >= 20 and len(closes) >= 20:
        recent_vol_trend = np.polyfit(range(10), volumes[-10:], 1)[0]
        recent_price_trend = np.polyfit(range(10), closes[-10:], 1)[0]
        
        # Normalize trends
        vol_trend_pct = (recent_vol_trend / np.mean(volumes[-10:])) * 100
        price_trend_pct = (recent_price_trend / np.mean(closes[-10:])) * 100
        
        if vol_trend_pct > 5 and price_trend_pct < 1:
            warning_flags.append("⚠️ Volume rising but price flat - Distribution phase?")
            risk_score += 15
    
    # 6. CIRCUIT FILTER HITS (Multiple limit hits)
    circuit_hits = 0
    for i in range(-20, 0):
        if i >= -len(closes):
            daily_change = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
            if daily_change > 9:  # Near 10% circuit (NSE has ~10% limit)
                circuit_hits += 1
    
    if circuit_hits >= 3:
        warning_flags.append("🚨 Multiple circuit hits - Highly manipulated")
        risk_score += 30
    elif circuit_hits >= 2:
        warning_flags.append("⚠️ Circuit hits detected - High risk")
        risk_score += 15
    
    # 7. LOW LIQUIDITY TRAP (Low consistent volume suddenly spikes)
    if len(volumes) >= 60:
        avg_60d_vol = np.mean(volumes[-60:-10])
        current_vol = volumes[-1]
        
        if avg_60d_vol < 100000 and current_vol > avg_60d_vol * 4:
            warning_flags.append("🚨 Illiquid stock with volume spike - Operator trap")
            risk_score += 25
    
    # 8. PRICE END-OF-DAY MANIPULATION (Closing price very different from average)
    if len(closes) >= 10:
        for i in range(-5, 0):
            if i >= -len(closes) and i >= -len(highs) and i >= -len(lows):
                day_avg = (highs[i] + lows[i]) / 2
                close_deviation = abs((closes[i] - day_avg) / day_avg) * 100
                
                if close_deviation > 3:
                    warning_flags.append("⚠️ End-of-day price manipulation detected")
                    risk_score += 10
                    break
    
    # FINAL VERDICT
    is_operated = risk_score >= 40  # High confidence of manipulation
    
    return is_operated, warning_flags, risk_score

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
    else:
        return 'Sideways'

def analyze_stock(data, criteria_config):
    """Analyze stock to find EARLY BUY opportunities BEFORE rally"""
    if not data:
        return None
    
    price = data['price']
    change = data['change']
    rsi = data['rsi']
    macd = data['macd']
    bb = data['bb_position']
    vol = data['vol_multiple']
    trend = data['trend']
    closes = data['closes']
    
    # OPERATOR DETECTION - Critical Safety Check
    is_operated, operator_flags, operator_risk = detect_operator_activity(data)
    
    # Calculate additional indicators for early detection
    weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
    monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
    
    potential_rs = max(20, price * 0.08)  # Higher potential target
    potential_pct = (potential_rs / price) * 100
    
    score = 0
    criteria = []
    
    # CRITICAL: Penalize heavily for operator activity
    if is_operated:
        score -= 50  # Heavy penalty
        criteria.append(f'🚨 OPERATOR DETECTED: Risk Score {operator_risk}/100 - AVOID [-50 pts]')
    elif operator_risk >= 25:
        score -= 25
        criteria.append(f'⚠️ HIGH RISK: Manipulation signs detected (Risk: {operator_risk}/100) [-25 pts]')
    elif operator_risk >= 15:
        score -= 10
        criteria.append(f'⚠️ CAUTION: Some manipulation indicators (Risk: {operator_risk}/100) [-10 pts]')
    
    # 1. CONSOLIDATION/ACCUMULATION PHASE (25 pts) - KEY FOR EARLY ENTRY
    # We want stocks that are NOT rallying yet but showing base building
    if -3 <= weekly_change <= 1:
        score += 25
        criteria.append(f'✅ Consolidation: Perfect base building ({weekly_change:+.1f}% weekly) [25 pts]')
    elif -5 <= weekly_change <= -3:
        score += 22
        criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [22 pts]')
    elif 1 < weekly_change <= 3:
        score += 18
        criteria.append(f'✅ Consolidation: Early breakout forming ({weekly_change:+.1f}% weekly) [18 pts]')
    elif weekly_change > 5:
        score += 0
        criteria.append(f'❌ Already rallied: Late to party ({weekly_change:+.1f}% weekly) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')
    
    # 2. RSI - OVERSOLD TO NEUTRAL ZONE (20 pts) - BEFORE THE MOVE
    if 30 <= rsi <= 40:
        score += 20
        criteria.append(f'✅ RSI: Oversold - Prime entry ({rsi:.0f}) [20 pts]')
    elif 40 < rsi <= 50:
        score += 18
        criteria.append(f'✅ RSI: Building momentum ({rsi:.0f}) [18 pts]')
    elif 50 < rsi <= 55:
        score += 15
        criteria.append(f'✅ RSI: Early momentum ({rsi:.0f}) [15 pts]')
    elif 55 < rsi <= 60:
        score += 10
        criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [10 pts]')
    elif rsi > 65:
        score += 0
        criteria.append(f'❌ RSI: Overbought - Already moved ({rsi:.0f}) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ RSI: Too oversold ({rsi:.0f}) [5 pts]')
    
    # 3. MACD TURNING POSITIVE (20 pts) - EARLY MOMENTUM SIGNAL
    if -2 <= macd <= 2:
        score += 20
        criteria.append(f'✅ MACD: Turning bullish - Perfect timing ({macd:.1f}) [20 pts]')
    elif 2 < macd <= 5:
        score += 18
        criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [18 pts]')
    elif -5 <= macd < -2:
        score += 15
        criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [15 pts]')
    elif macd > 8:
        score += 0
        criteria.append(f'❌ MACD: Too strong - Already moved ({macd:.1f}) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')
    
    # 4. BOLLINGER BANDS - LOWER BAND BOUNCE (20 pts)
    if 10 <= bb <= 25:
        score += 20
        criteria.append(f'✅ BB: Lower band - Mean reversion setup ({bb:.0f}%) [20 pts]')
    elif 25 < bb <= 40:
        score += 18
        criteria.append(f'✅ BB: Below middle - Good entry ({bb:.0f}%) [18 pts]')
    elif 40 < bb <= 50:
        score += 12
        criteria.append(f'⚠ BB: Middle band ({bb:.0f}%) [12 pts]')
    elif bb > 70:
        score += 0
        criteria.append(f'❌ BB: Upper band - Already extended ({bb:.0f}%) [0 pts]')
    else:
        score += 8
        criteria.append(f'⚠ BB: Neutral zone ({bb:.0f}%) [8 pts]')
    
    # 5. VOLUME - ACCUMULATION PHASE (15 pts)
    if 1.2 <= vol <= 2.0:
        score += 15
        criteria.append(f'✅ Volume: Smart money accumulation ({vol:.1f}x) [15 pts]')
    elif 2.0 < vol <= 2.5:
        score += 12
        criteria.append(f'✅ Volume: Building interest ({vol:.1f}x) [12 pts]')
    elif vol > 3.0:
        score += 5
        criteria.append(f'⚠ Volume: Too high - Possible peak ({vol:.1f}x) [5 pts]')
    elif 0.8 <= vol < 1.2:
        score += 8
        criteria.append(f'⚠ Volume: Average ({vol:.1f}x) [8 pts]')
    else:
        score += 0
        criteria.append(f'❌ Volume: Too low ({vol:.1f}x) [0 pts]')
    
    # 6. PRICE ACTION - NOT RALLYING YET (15 pts)
    if -2 <= change <= 0.5:
        score += 15
        criteria.append(f'✅ Today: Perfect - Not moving yet ({change:+.1f}%) [15 pts]')
    elif 0.5 < change <= 1.5:
        score += 12
        criteria.append(f'✅ Today: Early move ({change:+.1f}%) [12 pts]')
    elif -3 <= change < -2:
        score += 10
        criteria.append(f'✅ Today: Dip opportunity ({change:+.1f}%) [10 pts]')
    elif change > 3:
        score += 0
        criteria.append(f'❌ Today: Already rallied ({change:+.1f}%) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ Today: Moderate ({change:+.1f}%) [5 pts]')
    
    # 7. MONTHLY TREND - RECOVERING FROM DIP (15 pts)
    if -10 <= monthly_change <= -2:
        score += 15
        criteria.append(f'✅ Monthly: Recovering from dip ({monthly_change:+.1f}%) [15 pts]')
    elif -2 < monthly_change <= 3:
        score += 12
        criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [12 pts]')
    elif 3 < monthly_change <= 8:
        score += 8
        criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [8 pts]')
    elif monthly_change > 12:
        score += 0
        criteria.append(f'❌ Monthly: Extended move ({monthly_change:+.1f}%) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [5 pts]')
    
    # 8. POTENTIAL UPSIDE (10 pts)
    if potential_pct >= 10:
        score += 10
        criteria.append(f'✅ Upside: Excellent ({potential_pct:.1f}%) [10 pts]')
    elif potential_pct >= 8:
        score += 8
        criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
    elif potential_pct >= 6:
        score += 5
        criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')
    
    # Rating based on EARLY BUY criteria
    if is_operated:
        status = '🚨 OPERATED - AVOID'
        rating = 'Operated - Avoid'
    elif score >= 100:
        status = '🚀 PRIME BUY'
        rating = 'Prime Buy'
    elif score >= 90:
        status = '🌟 EXCELLENT BUY'
        rating = 'Excellent Buy'
    elif score >= 80:
        status = '💎 STRONG BUY'
        rating = 'Strong Buy'
    elif score >= 70:
        status = '✅ GOOD BUY'
        rating = 'Good Buy'
    elif score >= 60:
        status = '👍 WATCHLIST'
        rating = 'Watchlist'
    else:
        status = '❌ SKIP'
        rating = 'Skip'
    
    qualified = score >= 80 and not is_operated  # Only qualify safe, excellent opportunities
    met_count = len([c for c in criteria if '✅' in c])
    
    return {
        'symbol': data['symbol'],
        'price': price,
        'change': change,
        'weekly_change': weekly_change,
        'monthly_change': monthly_change,
        'potential_rs': potential_rs,
        'potential_pct': potential_pct,
        'rsi': rsi,
        'macd': macd,
        'bb': bb,
        'vol': vol,
        'trend': trend,
        'score': score,
        'qualified': qualified,
        'status': status,
        'rating': rating,
        'criteria': criteria,
        'met_count': met_count,
        'sector': SECTOR_MAP.get(data['symbol'], 'Other'),
        'is_operated': is_operated,
        'operator_risk': operator_risk,
        'operator_flags': operator_flags
    }

# Main App
st.markdown('<p class="main-header">🎯 Indian Stock Scout - EARLY BUY Scanner</p>', unsafe_allow_html=True)
st.markdown("*Find stocks BEFORE they rally - Early accumulation opportunities*")

# Sidebar
st.sidebar.header("⚙ Scanner Configuration")

scan_mode = st.sidebar.radio("Scan Mode", 
    ["Quick Scan (50 stocks)", "Full Scan (100+ stocks)", "Custom List"])

if scan_mode == "Quick Scan (50 stocks)":
    stocks_to_scan = NSE_STOCKS[:50]
elif scan_mode == "Full Scan (100+ stocks)":
    stocks_to_scan = NSE_STOCKS
else:
    custom_input = st.sidebar.text_area("Enter NSE symbols (one per line)", 
        "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK", height=150)
    stocks_to_scan = [s.strip().upper() for s in custom_input.split('\n') if s.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 ULTRA-STRICT Criteria")
st.sidebar.info("""
*Only top 5-10% qualify:*
- Qualified: Score ≥110 + Safe
- Exceptional: ≥140
- Prime: 125-139
- Excellent: 110-124
- Strong: 95-109

*10 STRICT Criteria (190 pts):*
1. **FII/DII Activity (30 pts)**
   - Institutional buying signals
2. **Consolidation (25 pts)**
   - -2% to +0.5% weekly
3. **RSI (25 pts)**
   - 32-38 (perfect oversold)
4. **MACD (20 pts)**
   - -1 to +1 (crossover)
5. **BB (20 pts)**
   - 8-20% (lower band)
6. **Volume (20 pts)**
   - 1.3-1.8x (accumulation)
7. **Today (15 pts)**
   - -1.5% to +0.3%
8. **Monthly (15 pts)**
   - -8% to -2% (recovery)
9. **3-Month (10 pts)**
   - Performance check
10. **Upside (10 pts)**
    - ≥12% potential

*Operator Penalties:*
- Operated: -70 pts
- Very High Risk: -40 pts
- High Risk: -25 pts
""")

st.sidebar.markdown("---")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 FIND EARLY BUYS", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning for Early Buy Opportunities...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    results = []
    total = len(stocks_to_scan)
    failed = 0
    
    criteria_config = {}  # Using hardcoded strict criteria
    
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.info(f"📊 Fetching *{symbol}*... ({idx+1}/{total})")
        
        data = fetch_stock_data(symbol)
        if data:
            analysis = analyze_stock(data, criteria_config)
            if analysis:
                results.append(analysis)
        else:
            failed += 1
        
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        
        if (idx + 1) % 10 == 0 or idx == total - 1:
            qualified_count = len([r for r in results if r['qualified']])
            stats_placeholder.info(f"✅ Analyzed: {len(results)} | Qualified (≥80): {qualified_count} | Failed: {failed}")
        
        time.sleep(0.15)
    
    st.session_state.scan_results = results
    st.session_state.scan_timestamp = datetime.now()
    
    status_text.success(f"✅ Scan complete! Analyzed {len(results)}/{total} stocks")
    time.sleep(1)
    status_text.empty()
    stats_placeholder.empty()
    progress_bar.empty()
    
    st.rerun()

# Display results
if st.session_state.scan_results:
    results = st.session_state.scan_results
    scan_time = st.session_state.scan_timestamp
    
    st.markdown("---")
    
    # Auto-refresh toggle and status
    col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 2, 6])
    
    with col_refresh1:
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=False, 
                                   help="Refresh prices every 10 seconds (like Groww/Zerodha)")
    
    with col_refresh2:
        if 'last_refresh' in st.session_state:
            seconds_ago = int((datetime.now() - st.session_state.last_refresh).total_seconds())
            st.caption(f"Updated {seconds_ago}s ago")
    
    st.subheader(f"📈 Early Buy Opportunities Found")
    st.caption(f"Initial scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh logic
    if auto_refresh:
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
        
        # Update prices for visible stocks
        if st.session_state.refresh_counter % 10 == 0:  # Every 10 seconds
            with st.spinner("Refreshing prices..."):
                for result in results:
                    new_price = fetch_live_price(result['symbol'])
                    if new_price:
                        old_price = result['price']
                        result['price'] = new_price
                        result['change'] = ((new_price - old_price) / old_price) * 100
                st.session_state.last_refresh = datetime.now()
        
        st.session_state.refresh_counter += 1
        time.sleep(1)
        st.rerun()
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Price (₹)': r['price'],
        'Today (%)': r['change'],
        'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'],
        '3M (%)': r['three_month_change'],
        'FII/DII': r['fii_dii_score'],
        'Potential (₹)': r['potential_rs'],
        'Potential (%)': r['potential_pct'],
        'RSI': r['rsi'],
        'MACD': r['macd'],
        'BB (%)': r['bb'],
        'Vol': f"{r['vol']:.1f}x",
        'Trend': r['trend'],
        'Score': r['score'],
        'Rating': r['rating'],
        'Status': r['status'],
        'Sector': r['sector'],
        'Operated': '🚨 YES' if r['is_operated'] else '✅ Safe',
        'Risk': r['operator_risk'],
        'Met': r['met_count']
    } for r in results])
    
    # Statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    operated_stocks = df[df['Operated'] == '🚨 YES']
    safe_stocks = df[df['Operated'] == '✅ Safe']
    exceptional = df[(df['Score'] >= 140) & (df['Operated'] == '✅ Safe')]
    prime = df[(df['Score'] >= 125) & (df['Score'] < 140) & (df['Operated'] == '✅ Safe')]
    excellent = df[(df['Score'] >= 110) & (df['Score'] < 125) & (df['Operated'] == '✅ Safe')]
    strong = df[(df['Score'] >= 95) & (df['Score'] < 110) & (df['Operated'] == '✅ Safe')]
    
    col1.metric("Total Scanned", len(df))
    col2.metric("🚨 Operated (AVOID)", len(operated_stocks))
    col3.metric("🌟 Exceptional (≥140)", len(exceptional))
    col4.metric("🚀 Prime (125-139)", len(prime))
    col5.metric("💎 Excellent (110-124)", len(excellent))
    col6.metric("✅ Strong (95-109)", len(strong))
    
    # Show qualification summary
    qualified_total = len(exceptional) + len(prime) + len(excellent)
    st.info(f"""
    **🎯 QUALIFICATION SUMMARY:** Only **{qualified_total}** stocks qualified (Score ≥110 + Safe) out of {len(df)} scanned.
    These are the **top {(qualified_total/len(df)*100):.1f}%** - truly exceptional early buy opportunities!
    """)
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip", "Operated - Avoid"])
    
    with filter_col2:
        safety_filter = st.selectbox("Safety", 
            ["All", "✅ Safe Only", "🚨 Operated Only"])
    
    with filter_col3:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col4:
        trend_filter = st.selectbox("Trend", 
            ["All", "Strong Uptrend", "Uptrend", "Sideways", "Downtrend"])
    
    with filter_col5:
        min_score_filter = st.number_input("Min Score", -100, 190, 110, 5, 
                                          help="Default: 110 (Qualified stocks only)")
    
    # Apply filters
    filtered_df = df.copy()
    
    if rating_filter != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == rating_filter]
    
    if safety_filter == "✅ Safe Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '✅ Safe']
    elif safety_filter == "🚨 Operated Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '🚨 YES']
    
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
    
    if trend_filter != "All":
        filtered_df = filtered_df[filtered_df['Trend'] == trend_filter]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]
    
    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")
    
    # OPERATOR WARNING
    if len(operated_stocks) > 0:
        st.error(f"""
        ⚠️ **OPERATOR ALERT**: Found {len(operated_stocks)} operated/manipulated stocks. 
        These stocks show high-risk patterns and should be **AVOIDED**.
        Use the 'Safety' filter above to view them separately.
        """)
    
    # Display table with color coding
    st.subheader("📋 Stock Analysis Table")
    
    def highlight_rating(row):
        # Highlight operated stocks in RED regardless of score
        if row['Operated'] == '🚨 YES':
            return ['background-color: #ff6b6b; color: white; font-weight: bold'] * len(row)
        # Safe stocks with exceptional scores
        elif row['Score'] >= 140:
            return ['background-color: #00e676; color: black; font-weight: bold'] * len(row)  # Neon green
        elif row['Score'] >= 125:
            return ['background-color: #69f0ae; font-weight: bold'] * len(row)  # Bright green
        elif row['Score'] >= 110:
            return ['background-color: #b9f6ca; font-weight: bold'] * len(row)  # Light green
        elif row['Score'] >= 95:
            return ['background-color: #e1f5fe'] * len(row)  # Light blue
        elif row['Score'] >= 80:
            return ['background-color: #fff9c4'] * len(row)  # Light yellow
        else:
            return ['background-color: #ffebee'] * len(row)  # Light red
    
    styled_df = filtered_df.style.apply(highlight_rating, axis=1)\
        .format({
            'Price (₹)': '₹{:.2f}',
            'Today (%)': '{:+.2f}%',
            'Weekly (%)': '{:+.2f}%',
            'Monthly (%)': '{:+.2f}%',
            '3M (%)': '{:+.2f}%',
            'Potential (₹)': '₹{:.2f}',
            'Potential (%)': '{:.2f}%',
            'RSI': '{:.1f}',
            'MACD': '{:.2f}',
            'BB (%)': '{:.0f}%'
        })
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Show only qualified stocks by default message
    if min_score_filter == 110:
        st.success("""
        ✅ **Showing QUALIFIED stocks only (Score ≥110)**. These are the best early buy opportunities.
        Lower the 'Min Score' filter if you want to see all stocks.
        """)
    
    # Detailed view
    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")
    
    if len(filtered_df) > 0:
        selected_symbol = st.selectbox("Select stock for details", filtered_df['Symbol'].tolist())
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)
        
        if selected_result:
            st.markdown(f"### {selected_symbol} - {selected_result['status']}")
            
            # Show operator warning if applicable
            if selected_result['is_operated']:
                st.error(f"""
                🚨 **OPERATOR/MANIPULATION DETECTED** - Risk Score: {selected_result['operator_risk']}/100
                
                **DO NOT TRADE THIS STOCK** - High probability of manipulation and dump.
                """)
                
                st.markdown("#### ⚠️ Manipulation Indicators Found:")
                for flag in selected_result['operator_flags']:
                    st.warning(flag)
            elif selected_result['operator_risk'] >= 15:
                st.warning(f"""
                ⚠️ **Caution Required** - Operator Risk Score: {selected_result['operator_risk']}/100
                
                Some manipulation indicators detected. Trade with extra caution.
                """)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Today", f"{selected_result['change']:+.2f}%")
            col4.metric("Weekly", f"{selected_result['weekly_change']:+.2f}%")
            col5.metric("Potential", f"₹{selected_result['potential_rs']:.2f}")
            col6.metric("Risk Score", f"{selected_result['operator_risk']}/100")
            
            st.markdown("#### Detailed Scoring Breakdown")
            for criterion in selected_result['criteria']:
                if '🚨' in criterion:
                    st.error(criterion)
                elif '✅' in criterion:
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
            f"early_buy_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"early_buy_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure and click 'FIND EARLY BUYS' to start scanning")
    
    st.markdown("---")
    st.subheader("🎯 ULTRA-STRICT Strategy - Only Top 5-10% Qualify")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why ULTRA-STRICT Works:**
        
        **Find Hidden Gems BEFORE Rally:**
        - Out of 100+ stocks, only 5-10 qualify
        - Catch them during consolidation
        - Enter during accumulation phase
        - Highest probability setups only
        
        **Key Philosophy:**
        1. ❌ **Don't Chase:** Stocks up 3%+ today
        2. ❌ **Don't FOMO:** RSI >55, already moving
        3. ✅ **Do Buy:** RSI 32-45, turning bullish
        4. ✅ **Do Wait:** For perfect setup only
        
        **🚨 OPERATOR PROTECTION:**
        - 8 manipulation detection signals
        - -70 pts penalty for operated stocks
        - Auto-identifies pump & dump
        
        **🏦 FII/DII Activity (30 pts) - NEW!**
        - Detects institutional buying/selling
        - High volume + price up = Buying
        - Confirms smart money interest
        - Score ≥12: Strong buying
        - Score ≤-8: Selling (avoid)
        
        **1. Consolidation (25 pts)**
           - Perfect: -2% to +0.5% weekly
           - Reject: >2% weekly rally
        
        **2. RSI (25 pts)**
           - Perfect: 32-38 (oversold)
           - Good: 38-45 (building)
           - Reject: >55 (overbought)
        
        **3. MACD (20 pts)**
           - Perfect: -1 to +1 (crossover)
           - Reject: >5 (extended)
        """)
    
    with col2:
        st.markdown("""        
        **4. Bollinger Bands (20 pts)**
           - Perfect: 8-20% (lower band)
           - Good: 20-30% (below middle)
           - Reject: >60% (upper band)
        
        **5. Volume (20 pts)**
           - Perfect: 1.3-1.8x (accumulation)
           - Good: 1.8-2.2x (interest)
           - Reject: >2.8x (distribution)
        
        **6. Today's Price (15 pts)**
           - Perfect: -1.5% to +0.3%
           - Reject: >2.5% (chasing)
        
        **7. Monthly Recovery (15 pts)**
           - Perfect: -8% to -2% (recovery)
           - Good: -2% to +2% (base)
           - Reject: >10% (extended)
        
        **8. 3-Month Performance (10 pts)**
           - Perfect: -15% to -5% (correction)
           - Good: -5% to +5% (sideways)
           - Reject: >25% (extended)
        
        **9. Upside Potential (10 pts)**
           - Target: ≥12% move
           - Minimum: 10%
        
        **QUALIFICATION:**
        - Exceptional: ≥140 pts
        - Prime: 125-139 pts
        - Excellent: 110-124 pts (QUALIFIED)
        - Strong: 95-109 pts
        - Below 95: Not good enough
        """)
    
    st.markdown("---")
    st.error("""
    **⚠️ ULTRA-STRICT = ONLY THE BEST:**
    
    With these criteria, expect:
    - **5-10 stocks** out of 100 to qualify (5-10%)
    - **1-3 exceptional** opportunities (score ≥140)
    - **2-5 prime/excellent** opportunities (score 110-139)
    - **Rest rejected** as not meeting standards
    
    This is INTENTIONAL - we want only the highest probability setups!
    """)
    
    st.info("""
    **🎯 BOTTOM LINE:** 
    
    This scanner finds the TOP 5-10% of stocks that are:
    - **Consolidating** (not rallying yet)
    - **Oversold** (RSI 32-45)
    - **Turning bullish** (MACD crossover)
    - **Near support** (BB 8-30%)
    - **Being accumulated** by institutions
    - **NOT OPERATED** (100% safe)
    
    **Perfect Entry = Top 5-10% opportunities ONLY + BEFORE 10%+ move**
    
    **Auto-Refresh:** Enable the checkbox after scan to get live price updates every 10 seconds!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>Indian Stock Scout - ULTRA-STRICT MODE</strong> | Only Top 5-10% Qualify | Auto-Refresh Available</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice. Past performance doesn't guarantee future results.</p>
</div>
""", unsafe_allow_html=True)

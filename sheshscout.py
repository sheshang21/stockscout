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

st.set_page_config(page_title="Indian Stock Scout - Elite Early Buy Scanner", page_icon="🎯", layout="wide")

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
    """Fetch real-time data from Yahoo Finance with fundamentals"""
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
        
        # Fetch FUNDAMENTAL DATA
        try:
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            
            # Revenue growth (TTM vs previous year)
            revenue_growth = info.get('revenueGrowth', None)  # YoY growth
            if revenue_growth is not None:
                revenue_growth = revenue_growth * 100  # Convert to percentage
            
            # Profit margin
            profit_margin = info.get('profitMargins', None)
            if profit_margin is not None:
                profit_margin = profit_margin * 100
            
            # Earnings growth
            earnings_growth = info.get('earningsGrowth', None)
            if earnings_growth is not None:
                earnings_growth = earnings_growth * 100
            
            # Operating margin
            operating_margin = info.get('operatingMargins', None)
            if operating_margin is not None:
                operating_margin = operating_margin * 100
            
            # ROE (Return on Equity)
            roe = info.get('returnOnEquity', None)
            if roe is not None:
                roe = roe * 100
            
            # Debt to Equity
            debt_to_equity = info.get('debtToEquity', None)
            
        except:
            market_cap = 0
            revenue_growth = None
            profit_margin = None
            earnings_growth = None
            operating_margin = None
            roe = None
            debt_to_equity = None
        
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
            'fii_dii_score': fii_dii_activity,
            'market_cap': market_cap,
            'revenue_growth': revenue_growth,
            'profit_margin': profit_margin,
            'earnings_growth': earnings_growth,
            'operating_margin': operating_margin,
            'roe': roe,
            'debt_to_equity': debt_to_equity
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
        score += 5
        criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')
    
    # 3. RSI - STRICTER (25 pts)
    if 32 <= rsi <= 38:
        score += 25
        criteria.append(f'✅ RSI: PERFECT oversold ({rsi:.0f}) [25 pts]')
    elif 38 < rsi <= 42:
        score += 22
        criteria.append(f'✅ RSI: Excellent oversold ({rsi:.0f}) [22 pts]')
    elif 42 < rsi <= 45:
        score += 18
        criteria.append(f'✅ RSI: Good entry ({rsi:.0f}) [18 pts]')
    elif 45 < rsi <= 50:
        score += 12
        criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [12 pts]')
    elif 50 < rsi <= 55:
        score += 5
        criteria.append(f'⚠ RSI: Starting to move ({rsi:.0f}) [5 pts]')
    elif rsi > 60:
        score += 0
        criteria.append(f'❌ RSI: Overbought - Too late ({rsi:.0f}) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ RSI: Below range ({rsi:.0f}) [3 pts]')
    
    # 4. MACD - STRICTER (20 pts)
    if -1 <= macd <= 1:
        score += 20
        criteria.append(f'✅ MACD: PERFECT crossover ({macd:.1f}) [20 pts]')
    elif 1 < macd <= 2:
        score += 18
        criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [18 pts]')
    elif -2 <= macd < -1:
        score += 15
        criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [15 pts]')
    elif 2 < macd <= 4:
        score += 10
        criteria.append(f'⚠ MACD: Building ({macd:.1f}) [10 pts]')
    elif macd > 6:
        score += 0
        criteria.append(f'❌ MACD: Extended - Too late ({macd:.1f}) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')
    
    # 5. BOLLINGER BANDS - STRICTER (20 pts)
    if 8 <= bb <= 20:
        score += 20
        criteria.append(f'✅ BB: PERFECT lower band ({bb:.0f}%) [20 pts]')
    elif 20 < bb <= 30:
        score += 18
        criteria.append(f'✅ BB: Excellent entry ({bb:.0f}%) [18 pts]')
    elif 30 < bb <= 40:
        score += 12
        criteria.append(f'✅ BB: Good entry ({bb:.0f}%) [12 pts]')
    elif 40 < bb <= 50:
        score += 8
        criteria.append(f'⚠ BB: Middle band ({bb:.0f}%) [8 pts]')
    elif bb > 65:
        score += 0
        criteria.append(f'❌ BB: Upper band - Extended ({bb:.0f}%) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ BB: Above middle ({bb:.0f}%) [5 pts]')
    
    # 6. VOLUME - STRICTER (20 pts)
    if 1.3 <= vol <= 1.8:
        score += 20
        criteria.append(f'✅ Volume: PERFECT accumulation ({vol:.1f}x) [20 pts]')
    elif 1.8 < vol <= 2.2:
        score += 18
        criteria.append(f'✅ Volume: Strong interest ({vol:.1f}x) [18 pts]')
    elif 1.1 <= vol < 1.3:
        score += 12
        criteria.append(f'✅ Volume: Building ({vol:.1f}x) [12 pts]')
    elif 2.2 < vol <= 2.8:
        score += 8
        criteria.append(f'⚠ Volume: High ({vol:.1f}x) [8 pts]')
    elif vol > 3.0:
        score += 0
        criteria.append(f'❌ Volume: Too high - Distribution? ({vol:.1f}x) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ Volume: Low ({vol:.1f}x) [3 pts]')
    
    # 7. TODAY'S PRICE - STRICTER (15 pts)
    if -1.5 <= change <= 0.3:
        score += 15
        criteria.append(f'✅ Today: PERFECT timing ({change:+.1f}%) [15 pts]')
    elif 0.3 < change <= 1:
        score += 12
        criteria.append(f'✅ Today: Early move ({change:+.1f}%) [12 pts]')
    elif -2.5 <= change < -1.5:
        score += 10
        criteria.append(f'✅ Today: Dip buy ({change:+.1f}%) [10 pts]')
    elif 1 < change <= 2:
        score += 5
        criteria.append(f'⚠ Today: Moving ({change:+.1f}%) [5 pts]')
    elif change > 2.5:
        score += 0
        criteria.append(f'❌ Today: Already rallied ({change:+.1f}%) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ Today: Weak ({change:+.1f}%) [3 pts]')
    
    # 8. MONTHLY RECOVERY - STRICTER (15 pts)
    if -8 <= monthly_change <= -2:
        score += 15
        criteria.append(f'✅ Monthly: PERFECT recovery ({monthly_change:+.1f}%) [15 pts]')
    elif -2 < monthly_change <= 2:
        score += 12
        criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [12 pts]')
    elif -12 <= monthly_change < -8:
        score += 10
        criteria.append(f'✅ Monthly: Deep correction ({monthly_change:+.1f}%) [10 pts]')
    elif 2 < monthly_change <= 5:
        score += 5
        criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [5 pts]')
    elif monthly_change > 10:
        score += 0
        criteria.append(f'❌ Monthly: Extended ({monthly_change:+.1f}%) [0 pts]')
    else:
        score += 3
        criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [3 pts]')
    
    # 9. 3-MONTH TREND - NEW STRICTER CRITERIA (10 pts)
    if -15 <= three_month_change <= -5:
        score += 10
        criteria.append(f'✅ 3-Month: Perfect correction ({three_month_change:+.1f}%) [10 pts]')
    elif -5 < three_month_change <= 5:
        score += 8
        criteria.append(f'✅ 3-Month: Sideways ({three_month_change:+.1f}%) [8 pts]')
    elif -25 <= three_month_change < -15:
        score += 5
        criteria.append(f'⚠ 3-Month: Deep fall ({three_month_change:+.1f}%) [5 pts]')
    elif 5 < three_month_change <= 15:
        score += 3
        criteria.append(f'⚠ 3-Month: Some gain ({three_month_change:+.1f}%) [3 pts]')
    elif three_month_change > 25:
        score += 0
        criteria.append(f'❌ 3-Month: Extended rally ({three_month_change:+.1f}%) [0 pts]')
    else:
        score += 2
        criteria.append(f'⚠ 3-Month: Very weak ({three_month_change:+.1f}%) [2 pts]')
    
    # 10. UPSIDE POTENTIAL - STRICTER (10 pts)
    if potential_pct >= 12:
        score += 10
        criteria.append(f'✅ Upside: EXCELLENT ({potential_pct:.1f}%) [10 pts]')
    elif potential_pct >= 10:
        score += 8
        criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
    elif potential_pct >= 8:
        score += 5
        criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')
    
    # FINAL RATING - ULTRA STRICT
    if is_operated or not fundamental_pass:
        status = '🚨 REJECTED'
        rating = 'Rejected'
    elif score >= 140:
        status = '🌟 EXCEPTIONAL BUY'
        rating = 'Exceptional Buy'
    elif score >= 125:
        status = '🚀 PRIME BUY'
        rating = 'Prime Buy'
    elif score >= 110:
        status = '💎 EXCELLENT BUY'
        rating = 'Excellent Buy'
    elif score >= 95:
        status = '✅ STRONG BUY'
        rating = 'Strong Buy'
    elif score >= 80:
        status = '👍 GOOD BUY'
        rating = 'Good Buy'
    elif score >= 60:
        status = '⚠️ WATCHLIST'
        rating = 'Watchlist'
    else:
        status = '❌ SKIP'
        rating = 'Skip'
    
    qualified = score >= 110 and not is_operated and fundamental_pass
    met_count = len([c for c in criteria if '✅' in c])
    
    return {
        'symbol': data['symbol'],
        'price': price,
        'change': change,
        'weekly_change': weekly_change,
        'monthly_change': monthly_change,
        'three_month_change': three_month_change,
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
        'operator_flags': operator_flags,
        'market_cap': market_cap,
        'revenue_growth': revenue_growth,
        'profit_margin': profit_margin,
        'earnings_growth': earnings_growth
    }

# Main App
st.markdown('<p class="main-header">🎯 Indian Stock Scout - ELITE Early Buy Scanner</p>', unsafe_allow_html=True)
st.markdown("*Find top 2-5% quality stocks BEFORE they rally - Fundamental + Technical perfection*")

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
st.sidebar.subheader("🎯 ULTRA-STRICT Criteria + Fundamentals")
st.sidebar.info("""
*FUNDAMENTAL FILTERS (MUST PASS):*
1. **Market Cap ≥ ₹1,000 Cr**
   - Avoid penny stocks
2. **Revenue Growth ≥ 10%**
   - YoY growth requirement
3. **Profit Margin ≥ 5%**
   - Must be profitable
4. **Earnings Growth ≥ 15%**
   - Strong earnings momentum
5. **ROE ≥ 12%**
   - Capital efficiency

*TECHNICAL CRITERIA (190 pts):*
1. **FII/DII (30 pts)**
   - Institutional buying ≥12
2. **Consolidation (25 pts)**
   - -2% to +0.5% weekly
3. **RSI (25 pts)**
   - 32-38 (oversold)
4. **MACD (20 pts)**
   - -1 to +1 (crossover)
5. **BB (20 pts)**
   - 8-20% (lower band)
6. **Volume (20 pts)**
   - 1.3-1.8x (accumulation)
7. **Today (15 pts)**
   - -1.5% to +0.3%
8. **Monthly (15 pts)**
   - -8% to -2%
9. **3-Month (10 pts)**
   - -15% to +5%
10. **Upside (10 pts)**
    - ≥12% potential

*QUALIFICATION:*
- Exceptional: ≥140 + Fundamentals
- Prime: 125-139 + Fundamentals
- Excellent: 110-124 + Fundamentals

*Expect only 2-5% to qualify!*
""")

st.sidebar.markdown("---")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 FIND ELITE BUYS", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning for Elite Early Buy Opportunities...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    results = []
    total = len(stocks_to_scan)
    failed = 0
    
    criteria_config = {}  # Using hardcoded strict criteria
    
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.info(f"📊 Fetching {symbol}... ({idx+1}/{total})")
        
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
            stats_placeholder.info(f"✅ Analyzed: {len(results)} | Qualified (≥110): {qualified_count} | Failed: {failed}")
        
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
    
    st.subheader(f"📈 Elite Early Buy Opportunities Found")
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
        'MCap (Cr)': r['market_cap'] / 1e7 if r.get('market_cap', 0) > 0 else 0,
        'Rev Growth (%)': r.get('revenue_growth', 0) if r.get('revenue_growth') is not None else 0,
        'Profit Margin (%)': r.get('profit_margin', 0) if r.get('profit_margin') is not None else 0,
        'Earnings Growth (%)': r.get('earnings_growth', 0) if r.get('earnings_growth') is not None else 0,
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
    **🎯 QUALIFICATION SUMMARY:** Only **{qualified_total}** stocks qualified (Score ≥110 + Safe + Fundamentals) out of {len(df)} scanned.
    These are the **top {(qualified_total/len(df)*100):.1f}%** - truly exceptional early buy opportunities with strong fundamentals!
    """)
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip", "Rejected"])
    
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
    
    st.info(f"📊 Showing {len(filtered_df)} stocks (filtered from {len(df)} total)")
    
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
            'MCap (Cr)': '₹{:.0f}',
            'Rev Growth (%)': '{:.1f}%',
            'Profit Margin (%)': '{:.1f}%',
            'Earnings Growth (%)': '{:.1f}%',
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
        ✅ **Showing QUALIFIED stocks only (Score ≥110 + Fundamentals)**. These are the best early buy opportunities.
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
            col3.metric("MCap", f"₹{selected_result.get('market_cap', 0) / 1e7:.0f} Cr")
            col4.metric("Rev Growth", f"{selected_result.get('revenue_growth', 0):.1f}%" if selected_result.get('revenue_growth') else "N/A")
            col5.metric("Earnings", f"{selected_result.get('earnings_growth', 0):.1f}%" if selected_result.get('earnings_growth') else "N/A")
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
            f"elite_buy_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"elite_buy_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure and click 'FIND ELITE BUYS' to start scanning")
    
    st.markdown("---")
    st.subheader("🎯 ULTRA-STRICT Strategy + FUNDAMENTALS - Only Top 2-5% Qualify")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏦 NEW: FUNDAMENTAL FILTERS (MUST PASS):**
        
        **Why Fundamentals Matter:**
        - Separate quality from junk
        - Avoid manipulation-prone stocks
        - Focus on growing businesses
        - Reduce downside risk
        
        **5 MANDATORY FILTERS:**
        
        **1. Market Cap ≥ ₹1,000 Crores**
        - Avoid penny stocks
        - Better liquidity
        - Lower manipulation risk
        - More institutional interest
        
        **2. Revenue Growth ≥ 10% YoY**
        - Growing businesses only
        - Sustainable competitive advantage
        - Market share gains
        - Reject declining companies
        
        **3. Profit Margin ≥ 5%**
        - Must be profitable
        - Operating efficiency
        - Pricing power
        - Sustainable business model
        
        **4. Earnings Growth ≥ 15% YoY**
        - Strong earnings momentum
        - Better than revenue growth
        - Operational leverage
        - Quality of growth
        
        **5. ROE ≥ 12%**
        - Capital efficiency
        - Management quality
        - Competitive moat
        - Value creation
        
        **⚠️ If ANY fundamental filter fails:**
        - **INSTANT REJECTION** (-100 score)
        - No technical analysis needed
        - Move to next stock
        
        **🚨 OPERATOR PROTECTION (Enhanced):**
        - 8 manipulation detection signals
        - -70 pts for confirmed operators
        - -40 pts for very high risk
        - -25 pts for high risk
        - Auto-identifies pump & dump
        """)
    
    with col2:
        st.markdown("""        
        **📊 TECHNICAL CRITERIA (190 pts):**
        
        *Now applied ONLY after fundamentals pass*
        
        **1. FII/DII Activity (30 pts) - STRICTER**
           - Perfect: ≥15 pts (strong buying)
           - Good: 12-14 pts
           - Moderate: 8-11 pts
           - Reject: Selling (≤-8)
        
        **2. Consolidation (25 pts) - STRICTER**
           - Perfect: -2% to +0.5% weekly
           - Good: -3% to -2% OR +0.5% to +1%
           - Reject: >+3% (already rallied)
        
        **3. RSI (25 pts) - STRICTER**
           - Perfect: 32-38 (deep oversold)
           - Good: 38-45
           - Reject: >60 (overbought)
        
        **4. MACD (20 pts) - STRICTER**
           - Perfect: -1 to +1 (crossover zone)
           - Good: -2 to +2
           - Reject: >6 (extended)
        
        **5. Bollinger Bands (20 pts) - STRICTER**
           - Perfect: 8-20% (deep lower band)
           - Good: 20-30%
           - Reject: >65% (upper band)
        
        **6. Volume (20 pts) - STRICTER**
           - Perfect: 1.3-1.8x (smart accumulation)
           - Good: 1.8-2.2x
           - Reject: >3.0x (distribution)
        
        **7. Today's Price (15 pts) - STRICTER**
           - Perfect: -1.5% to +0.3%
           - Good: +0.3% to +1%
           - Reject: >2.5% (chasing)
        
        **8. Monthly Recovery (15 pts) - STRICTER**
           - Perfect: -8% to -2% (recovery phase)
           - Good: -2% to +2%
           - Reject: >10% (extended)
        
        **9. 3-Month Trend (10 pts) - NEW**
           - Perfect: -15% to -5% (correction)
           - Good: -5% to +5% (sideways)
           - Reject: >25% (too extended)
        
        **10. Upside Potential (10 pts)**
           - Perfect: ≥12%
           - Good: 10-12%
           - Reject: <8%
        
        **🎯 QUALIFICATION THRESHOLDS:**
        - **Exceptional:** ≥140 pts + All fundamentals
        - **Prime:** 125-139 pts + All fundamentals
        - **Excellent:** 110-124 pts + All fundamentals (QUALIFIED)
        - **Strong:** 95-109 pts + All fundamentals
        - **Below 95 or failed fundamentals:** REJECTED
        """)
    
    st.markdown("---")
    st.error("""
    **⚠️ EXPECT ULTRA-LOW QUALIFICATION RATE:**
    
    With fundamental + technical filters:
    - **2-5 stocks** out of 100 qualify (2-5%)
    - **0-2 exceptional** opportunities (≥140)
    - **1-3 prime/excellent** opportunities (110-139)
    - **95-98% REJECTED** as not meeting standards
    
    **This is INTENTIONAL!** We're finding:
    - Top-tier businesses (fundamentals)
    - Perfect technical setup (technicals)
    - Safe from manipulation (operator detection)
    - BEFORE the 10%+ rally (early entry)
    """)
    
    st.info("""
    **🎯 THE ULTIMATE EDGE:** 
    
    This scanner finds the TOP 2-5% of stocks that have:
    
    **✅ FUNDAMENTAL QUALITY:**
    - ≥₹1,000 Cr market cap (institutional grade)
    - ≥10% revenue growth (growing business)
    - ≥5% profit margin (profitable)
    - ≥15% earnings growth (strong momentum)
    - ≥12% ROE (efficient capital use)
    
    **✅ TECHNICAL SETUP:**
    - Consolidating (not rallying yet)
    - Oversold (RSI 32-45)
    - Turning bullish (MACD crossover)
    - Near support (BB 8-30%)
    - Being accumulated (Volume 1.3-1.8x)
    - Institutional interest (FII/DII ≥12)
    
    **✅ SAFETY:**
    - NOT OPERATED (operator detection)
    - Clean price action
    - Sustainable growth
    
    **Perfect Entry = Top 2-5% Quality Businesses at Perfect Technical Timing**
    
    **Auto-Refresh:** Enable after scan for live updates every 10 seconds!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>Indian Stock Scout - ELITE MODE</strong> | Only Top 2-5% Qualify | Fundamentals + Technicals | Auto-Refresh Available</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice. Past performance doesn't guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
return 'Sideways'

def analyze_stock(data, criteria_config):
    """Analyze stock to find EARLY BUY opportunities BEFORE rally - ULTRA STRICT MODE"""
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
    
    # Fundamental data
    market_cap = data.get('market_cap', 0)
    revenue_growth = data.get('revenue_growth', None)
    profit_margin = data.get('profit_margin', None)
    earnings_growth = data.get('earnings_growth', None)
    operating_margin = data.get('operating_margin', None)
    roe = data.get('roe', None)
    debt_to_equity = data.get('debt_to_equity', None)
    
    # OPERATOR DETECTION - Critical Safety Check
    is_operated, operator_flags, operator_risk = detect_operator_activity(data)
    
    # Calculate additional indicators for early detection
    weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
    monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
    three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 60 else 0
    
    potential_rs = max(20, price * 0.08)
    potential_pct = (potential_rs / price) * 100
    
    score = 0
    criteria = []
    
    # ===== FUNDAMENTAL FILTERS (MUST PASS) =====
    fundamental_pass = True
    
    # 1. MARKET CAP FILTER - Avoid penny stocks and manipulation-prone small caps
    market_cap_cr = market_cap / 1e7 if market_cap > 0 else 0  # Convert to Crores
    
    if market_cap < 1000_00_00_000:  # Less than 1000 Cr
        criteria.append(f'❌ REJECTED: Market Cap too low (₹{market_cap_cr:.0f} Cr) - High manipulation risk')
        fundamental_pass = False
    elif market_cap < 5000_00_00_000:  # 1000-5000 Cr
        criteria.append(f'⚠️ Small Cap: ₹{market_cap_cr:.0f} Cr - Higher risk')
    elif market_cap < 20000_00_00_000:  # 5000-20000 Cr
        criteria.append(f'✅ Mid Cap: ₹{market_cap_cr:.0f} Cr - Good')
    else:  # >20000 Cr
        criteria.append(f'✅ Large Cap: ₹{market_cap_cr:.0f} Cr - Excellent')
    
    # 2. REVENUE GROWTH FILTER (Critical for growth stocks)
    if revenue_growth is not None:
        if revenue_growth < 10:
            criteria.append(f'❌ REJECTED: Revenue Growth too low ({revenue_growth:.1f}%) - Need ≥10%')
            fundamental_pass = False
        elif revenue_growth < 15:
            criteria.append(f'⚠️ Revenue Growth: Moderate ({revenue_growth:.1f}%) - Borderline')
        elif revenue_growth < 25:
            criteria.append(f'✅ Revenue Growth: Good ({revenue_growth:.1f}%)')
        else:
            criteria.append(f'✅ Revenue Growth: EXCELLENT ({revenue_growth:.1f}%)')
    else:
        criteria.append(f'⚠️ Revenue Growth: Data N/A')
    
    # 3. PROFIT MARGIN FILTER (Must be profitable)
    if profit_margin is not None:
        if profit_margin < 5:
            criteria.append(f'❌ REJECTED: Profit Margin too low ({profit_margin:.1f}%) - Need ≥5%')
            fundamental_pass = False
        elif profit_margin < 10:
            criteria.append(f'⚠️ Profit Margin: Low ({profit_margin:.1f}%)')
        elif profit_margin < 15:
            criteria.append(f'✅ Profit Margin: Good ({profit_margin:.1f}%)')
        else:
            criteria.append(f'✅ Profit Margin: EXCELLENT ({profit_margin:.1f}%)')
    else:
        criteria.append(f'⚠️ Profit Margin: Data N/A')
    
    # 4. EARNINGS GROWTH FILTER
    if earnings_growth is not None:
        if earnings_growth < 15:
            criteria.append(f'❌ REJECTED: Earnings Growth too low ({earnings_growth:.1f}%) - Need ≥15%')
            fundamental_pass = False
        elif earnings_growth < 20:
            criteria.append(f'⚠️ Earnings Growth: Moderate ({earnings_growth:.1f}%)')
        elif earnings_growth < 30:
            criteria.append(f'✅ Earnings Growth: Good ({earnings_growth:.1f}%)')
        else:
            criteria.append(f'✅ Earnings Growth: EXCELLENT ({earnings_growth:.1f}%)')
    else:
        criteria.append(f'⚠️ Earnings Growth: Data N/A')
    
    # 5. ROE FILTER (Return on Equity)
    if roe is not None:
        if roe < 12:
            criteria.append(f'❌ REJECTED: ROE too low ({roe:.1f}%) - Need ≥12%')
            fundamental_pass = False
        elif roe < 15:
            criteria.append(f'⚠️ ROE: Moderate ({roe:.1f}%)')
        elif roe < 20:
            criteria.append(f'✅ ROE: Good ({roe:.1f}%)')
        else:
            criteria.append(f'✅ ROE: EXCELLENT ({roe:.1f}%)')
    else:
        criteria.append(f'⚠️ ROE: Data N/A')
    
    # If fundamentals don't pass, reject immediately
    if not fundamental_pass:
        return {
            'symbol': data['symbol'],
            'price': price,
            'change': change,
            'weekly_change': weekly_change,
            'monthly_change': monthly_change,
            'three_month_change': three_month_change,
            'potential_rs': potential_rs,
            'potential_pct': potential_pct,
            'rsi': rsi,
            'macd': macd,
            'bb': bb,
            'vol': vol,
            'trend': trend,
            'score': -100,
            'qualified': False,
            'status': '❌ REJECTED - Fundamentals',
            'rating': 'Rejected',
            'criteria': criteria,
            'met_count': 0,
            'sector': SECTOR_MAP.get(data['symbol'], 'Other'),
            'is_operated': is_operated,
            'operator_risk': operator_risk,
            'operator_flags': operator_flags,
            'market_cap': market_cap,
            'revenue_growth': revenue_growth,
            'profit_margin': profit_margin,
            'earnings_growth': earnings_growth
        }
    
    # ===== TECHNICAL SCORING (ULTRA STRICT) =====
    
    # CRITICAL: Penalize heavily for operator activity
    if is_operated:
        score -= 70
        criteria.append(f'🚨 OPERATOR DETECTED: Risk Score {operator_risk}/100 - AVOID [-70 pts]')
    elif operator_risk >= 35:
        score -= 40
        criteria.append(f'⚠️ VERY HIGH RISK: Manipulation signs (Risk: {operator_risk}/100) [-40 pts]')
    elif operator_risk >= 20:
        score -= 25
        criteria.append(f'⚠️ HIGH RISK: Manipulation indicators (Risk: {operator_risk}/100) [-25 pts]')
    elif operator_risk >= 10:
        score -= 10
        criteria.append(f'⚠️ CAUTION: Some risk signals (Risk: {operator_risk}/100) [-10 pts]')
    
    # 1. FII/DII INSTITUTIONAL ACTIVITY (30 pts) - STRICTER
    fii_score = data['fii_dii_score']
    if fii_score >= 15:
        score += 30
        criteria.append(f'✅ FII/DII: STRONG buying ({fii_score} pts) [30 pts]')
    elif fii_score >= 12:
        score += 25
        criteria.append(f'✅ FII/DII: Good buying ({fii_score} pts) [25 pts]')
    elif fii_score >= 8:
        score += 18
        criteria.append(f'✅ FII/DII: Moderate buying ({fii_score} pts) [18 pts]')
    elif fii_score >= 4:
        score += 10
        criteria.append(f'⚠ FII/DII: Weak activity ({fii_score} pts) [10 pts]')
    elif fii_score <= -8:
        score += 0
        criteria.append(f'❌ FII/DII: Selling detected ({fii_score} pts) [0 pts]')
    else:
        score += 5
        criteria.append(f'⚠ FII/DII: Neutral ({fii_score} pts) [5 pts]')
    
    # 2. CONSOLIDATION PHASE (25 pts) - STRICTER
    if -2 <= weekly_change <= 0.5:
        score += 25
        criteria.append(f'✅ Consolidation: Perfect base ({weekly_change:+.1f}% weekly) [25 pts]')
    elif 0.5 < weekly_change <= 1:
        score += 20
        criteria.append(f'✅ Consolidation: Early breakout ({weekly_change:+.1f}% weekly) [20 pts]')
    elif -3 <= weekly_change < -2:
        score += 18
        criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [18 pts]')
    elif 1 < weekly_change <= 2:
        score += 10
        criteria.append(f'⚠ Consolidation: Moving up ({weekly_change:+.1f}% weekly) [10 pts]')
    elif weekly_change > 3:
        score += 0
        criteria.append(f'❌ Already rallied: Too late ({weekly_change:+.1f}% weekly) [0 pts]')
    else:

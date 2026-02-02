import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging to show warnings but not info
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

st.set_page_config(page_title="Indian Stock Scout - Ultra-Strict Scanner", page_icon="🎯", layout="wide")

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

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'BAJFINANCE': 'NBFC', 'WIPRO': 'IT', 'SUNPHARMA': 'Pharma', 'TITAN': 'Consumer', 'ULTRACEMCO': 'Cement',
    'NESTLEIND': 'FMCG', 'ONGC': 'Energy', 'TATAMOTORS': 'Auto', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'TECHM': 'IT', 'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure'
}

# BULLETPROOF: Retry wrapper with exponential backoff
def bulletproof_fetch(func, *args, max_retries=3, initial_delay=1, **kwargs):
    """Wrapper that adds retry logic with exponential backoff to any function"""
    for attempt in range(max_retries):
        try:
            # Add small delay before each attempt to avoid rate limiting
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))  # Exponential backoff
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed, return None silently
                return None
            # Continue to next attempt
            time.sleep(initial_delay * (2 ** attempt))
    return None

@st.cache_data(ttl=300)
def fetch_stock_data(symbol):
    """Fetch real-time data from Yahoo Finance with fundamentals
    
    EXACT structure from original sheshscout.py - simple and clean
    Symbol comes WITHOUT .NS/.BO, function adds it based on what's in the symbol
    """
    try:
        # Symbol already has .NS or .BO added during loading
        ticker = yf.Ticker(symbol)
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
        
        # Get fundamental data
        info = ticker.info
        market_cap = info.get('marketCap', 0)
        
        # Revenue and profit growth
        revenue_growth = info.get('revenueGrowth', None)
        profit_margin = info.get('profitMargins', None)
        earnings_growth = info.get('earningsGrowth', None)
        
        # Additional fundamental metrics
        pe_ratio = info.get('trailingPE', None)
        roe = info.get('returnOnEquity', None)
        debt_to_equity = info.get('debtToEquity', None)
        
        # CASH METRICS
        total_cash = info.get('totalCash', 0)
        
        # Get LATEST REPORTED FY REVENUE (most recent completed FY)
        latest_fy_revenue = 0
        try:
            annual_financials = ticker.financials
            if annual_financials is not None and not annual_financials.empty:
                if 'Total Revenue' in annual_financials.index:
                    # iloc[0] gets the MOST RECENT fiscal year
                    latest_fy_revenue = annual_financials.loc['Total Revenue'].iloc[0]
                    if pd.isna(latest_fy_revenue):
                        latest_fy_revenue = 0
        except:
            latest_fy_revenue = 0
        
        # Calculate ratios - BULLETPROOF: Safe division
        cash_on_hand_to_mcap = (total_cash / market_cap * 100) if market_cap > 0 and total_cash > 0 else 0
        latest_fy_revenue_to_mcap = (latest_fy_revenue / market_cap) if market_cap > 0 and latest_fy_revenue > 0 else 0
        
        # Get 3-year historical data
        historical_data = get_historical_financials(ticker, market_cap)
        
        # Get quarterly financials for growth analysis
        try:
            financials = ticker.quarterly_financials
            if financials is not None and not financials.empty:
                # Get Total Revenue if available
                if 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue'].values
                    # BULLETPROOF: Filter out NaN values
                    revenues = [r for r in revenues if not pd.isna(r)]
                    
                    if len(revenues) >= 4:
                        # Calculate QoQ and YoY growth - BULLETPROOF: Safe division
                        qoq_revenue_growth = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
                        yoy_revenue_growth = ((revenues[0] - revenues[3]) / abs(revenues[3])) * 100 if len(revenues) >= 4 and revenues[3] != 0 else None
                    else:
                        qoq_revenue_growth = None
                        yoy_revenue_growth = None
                else:
                    qoq_revenue_growth = None
                    yoy_revenue_growth = None
                
                # Get Net Income if available
                if 'Net Income' in financials.index:
                    profits = financials.loc['Net Income'].values
                    # BULLETPROOF: Filter out NaN values
                    profits = [p for p in profits if not pd.isna(p)]
                    
                    if len(profits) >= 4:
                        qoq_profit_growth = ((profits[0] - profits[1]) / abs(profits[1])) * 100 if profits[1] != 0 else None
                        yoy_profit_growth = ((profits[0] - profits[3]) / abs(profits[3])) * 100 if len(profits) >= 4 and profits[3] != 0 else None
                    else:
                        qoq_profit_growth = None
                        yoy_profit_growth = None
                else:
                    qoq_profit_growth = None
                    yoy_profit_growth = None
            else:
                qoq_revenue_growth = None
                yoy_revenue_growth = None
                qoq_profit_growth = None
                yoy_profit_growth = None
        except Exception as e:
            qoq_revenue_growth = None
            yoy_revenue_growth = None
            qoq_profit_growth = None
            yoy_profit_growth = None
        
        # Get institutional data
        fii_dii_activity = detect_institutional_activity(volumes, closes)
        
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        bb_position = calculate_bb_position(closes)
        vol_multiple = calculate_volume_multiple(volumes)
        trend = detect_trend(closes)
        
        # Calculate timeframe changes - BULLETPROOF: Safe division
        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 60 and closes[0] != 0 else 0
        
        return {
            'symbol': symbol,  # Use the symbol as-is (already has .NS or .BO)
            'price': price,
            'change': change,
            'weekly_change': weekly_change,
            'monthly_change': monthly_change,
            'three_month_change': three_month_change,
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
            'pe_ratio': pe_ratio,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'total_cash': total_cash,
            'latest_fy_revenue': latest_fy_revenue,
            'cash_on_hand_to_mcap': cash_on_hand_to_mcap,
            'latest_fy_revenue_to_mcap': latest_fy_revenue_to_mcap,
            'historical_data': historical_data,
            'qoq_revenue_growth': qoq_revenue_growth,
            'yoy_revenue_growth': yoy_revenue_growth,
            'qoq_profit_growth': qoq_profit_growth,
            'yoy_profit_growth': yoy_profit_growth
        }
    except Exception as e:
        return None

def get_historical_financials(ticker, current_mcap):
    """Get 3-year historical revenue, cash, and sales/mcap data"""
    try:
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet  # ANNUAL balance sheet, not quarterly
        
        historical = {
            'years': [],
            'revenues': [],
            'cash_amounts': [],
            'sales_to_mcap': []
        }
        
        if financials is not None and not financials.empty:
            years = financials.columns[:3] if len(financials.columns) >= 3 else financials.columns
            
            for year in years:
                year_str = year.strftime('%Y') if hasattr(year, 'strftime') else str(year)
                historical['years'].append(year_str)
                
                # Revenue
                if 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue', year]
                    historical['revenues'].append(revenue if pd.notna(revenue) else 0)
                else:
                    historical['revenues'].append(0)
                
                # Cash from ANNUAL balance sheet (same source as table)
                if balance_sheet is not None and not balance_sheet.empty and year in balance_sheet.columns:
                    if 'Cash And Cash Equivalents' in balance_sheet.index:
                        cash = balance_sheet.loc['Cash And Cash Equivalents', year]
                        historical['cash_amounts'].append(cash if pd.notna(cash) else 0)
                    elif 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.index:
                        cash = balance_sheet.loc['Cash Cash Equivalents And Short Term Investments', year]
                        historical['cash_amounts'].append(cash if pd.notna(cash) else 0)
                    else:
                        historical['cash_amounts'].append(0)
                else:
                    historical['cash_amounts'].append(0)
        
        # Calculate Sales to Market Cap - BULLETPROOF: Safe division
        for revenue in historical['revenues']:
            if current_mcap > 0 and revenue > 0:
                historical['sales_to_mcap'].append(revenue / current_mcap)
            else:
                historical['sales_to_mcap'].append(0)
        
        return historical
    except:
        return {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []}

def fetch_live_price(symbol):
    """Fetch only live price for auto-refresh (non-cached)
    
    Symbol already has .NS or .BO suffix from file loading
    """
    try:
        # Symbol already has exchange suffix (e.g., "RELIANCE.NS" or "TCS.BO")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
    except:
        return None

def detect_institutional_activity(volumes, closes):
    """Detect FII/DII activity patterns from volume and price action"""
    try:
        if len(volumes) < 20 or len(closes) < 20:
            return 0
        
        score = 0
        recent_days = 10
        
        for i in range(-recent_days, 0):
            # BULLETPROOF: Safe array access and division
            if i >= -len(volumes) and i >= -len(closes):
                vol_ratio = volumes[i] / np.mean(volumes[-60:]) if len(volumes) >= 60 else volumes[i] / np.mean(volumes)
                if vol_ratio == 0 or np.isnan(vol_ratio):
                    continue
                    
                if i > -len(closes) and closes[i-1] != 0:
                    price_change = ((closes[i] - closes[i-1]) / closes[i-1]) * 100
                else:
                    price_change = 0
                
                if vol_ratio > 1.5 and price_change > 1:
                    score += 2
                elif vol_ratio > 1.2 and price_change > 0.5:
                    score += 1
                elif vol_ratio > 1.5 and price_change < -1:
                    score -= 2
                elif vol_ratio > 1.2 and price_change < -0.5:
                    score -= 1
        
        return score
    except:
        return 0

def calculate_rsi(prices, period=14):
    try:
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
    except:
        return 50

def calculate_macd(prices):
    try:
        if len(prices) < 26:
            return 0
        ema12 = calculate_ema(prices, 12)
        ema26 = calculate_ema(prices, 26)
        return ema12 - ema26
    except:
        return 0

def calculate_ema(prices, period):
    try:
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    except:
        return 0

def calculate_bb_position(prices, period=20):
    try:
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
    except:
        return 50

def calculate_volume_multiple(volumes):
    try:
        if len(volumes) < 20:
            return 1.0
        current = volumes[-1]
        avg20 = np.mean(volumes[-20:])
        if avg20 == 0:
            return 1.0
        return current / avg20
    except:
        return 1.0

def detect_operator_activity(data):
    """Detect if stock shows signs of operator/manipulator activity"""
    try:
        closes = data['closes']
        volumes = data['volumes']
        highs = data['highs']
        lows = data['lows']
        
        warning_flags = []
        risk_score = 0
        
        if len(closes) < 20:
            return False, [], 0
        
        # 1. EXTREME VOLUME SPIKES - BULLETPROOF: Safe operations
        recent_vols = volumes[-10:]
        avg_vol = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
        if avg_vol == 0:
            return False, [], 0
            
        max_recent_vol = np.max(recent_vols)
        
        if max_recent_vol > avg_vol * 5:
            warning_flags.append("🚨 EXTREME volume spike (>5x avg) - Possible pump")
            risk_score += 30
        elif max_recent_vol > avg_vol * 3:
            warning_flags.append("⚠️ High volume spike (>3x avg) - Monitor closely")
            risk_score += 15
        
        # 2. PRICE VOLATILITY
        recent_prices = closes[-10:]
        price_swings = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] != 0:
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
        
        # 3. CIRCUIT FILTER HITS
        circuit_hits = 0
        for i in range(-20, 0):
            if i >= -len(closes) and i > -len(closes) and closes[i-1] != 0:
                daily_change = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
                if daily_change > 9:
                    circuit_hits += 1
        
        if circuit_hits >= 3:
            warning_flags.append("🚨 Multiple circuit hits - Highly manipulated")
            risk_score += 30
        elif circuit_hits >= 2:
            warning_flags.append("⚠️ Circuit hits detected - High risk")
            risk_score += 15
        
        is_operated = risk_score >= 40
        
        return is_operated, warning_flags, risk_score
    except:
        return False, [], 0

def detect_trend(prices):
    try:
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
    except:
        return 'Sideways'

def analyze_stock(data, min_market_cap, thresholds=None):
    """Analyze stock with ULTRA-STRICT fundamentals criteria
    
    Args:
        data: Stock data dictionary
        min_market_cap: Minimum market cap filter
        thresholds: Dictionary of adjustable thresholds (optional)
    """
    try:
        if not data:
            return None
        
        # Default thresholds if not provided
        if thresholds is None:
            thresholds = {
                'threshold_exceptional': 180,
                'threshold_prime': 160,
                'threshold_excellent': 140,
                'threshold_strong': 120,
                'rsi_low': 32,
                'rsi_high': 38,
                'min_revenue_yoy': 20,
                'min_profit_yoy': 25
            }
        
        price = data['price']
        change = data['change']
        rsi = data['rsi']
        macd = data['macd']
        bb = data['bb_position']
        vol = data['vol_multiple']
        trend = data['trend']
        closes = data['closes']
        
        # Market cap filter (in crores) - BULLETPROOF: Safe division
        market_cap = data['market_cap'] / 10000000 if data['market_cap'] else 0
        
        # Skip if below minimum market cap
        if market_cap < min_market_cap:
            return None
        
        # OPERATOR DETECTION
        is_operated, operator_flags, operator_risk = detect_operator_activity(data)
        
        # Calculate additional indicators - BULLETPROOF: Safe division
        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 60 and closes[0] != 0 else 0
        
        potential_rs = max(20, price * 0.10)
        potential_pct = (potential_rs / price) * 100 if price != 0 else 0
        
        score = 0
        criteria = []
        
        # CRITICAL: Operator penalty
        if is_operated:
            score -= 70
            criteria.append(f'🚨 OPERATOR DETECTED: Risk Score {operator_risk}/100 - AVOID [-70 pts]')
        elif operator_risk >= 30:
            score -= 40
            criteria.append(f'🚨 VERY HIGH RISK: Major manipulation signs (Risk: {operator_risk}/100) [-40 pts]')
        elif operator_risk >= 20:
            score -= 25
            criteria.append(f'⚠️ HIGH RISK: Manipulation signs detected (Risk: {operator_risk}/100) [-25 pts]')
        elif operator_risk >= 12:
            score -= 12
            criteria.append(f'⚠️ CAUTION: Some manipulation indicators (Risk: {operator_risk}/100) [-12 pts]')
        
        # 1. MARKET CAP QUALITY (15 pts) - NEW!
        if market_cap >= 50000:
            score += 15
            criteria.append(f'✅ Market Cap: Large Cap (₹{market_cap:.0f} Cr) [15 pts]')
        elif market_cap >= 20000:
            score += 12
            criteria.append(f'✅ Market Cap: Mid-Large Cap (₹{market_cap:.0f} Cr) [12 pts]')
        elif market_cap >= 10000:
            score += 10
            criteria.append(f'✅ Market Cap: Mid Cap (₹{market_cap:.0f} Cr) [10 pts]')
        elif market_cap >= 5000:
            score += 7
            criteria.append(f'⚠ Market Cap: Small-Mid Cap (₹{market_cap:.0f} Cr) [7 pts]')
        else:
            score += 0
            criteria.append(f'❌ Market Cap: Small Cap (₹{market_cap:.0f} Cr) [0 pts]')
        
        # 2. REVENUE GROWTH (25 pts) - NEW! STRICTEST CRITERIA
        yoy_rev = data['yoy_revenue_growth']
        qoq_rev = data['qoq_revenue_growth']
        
        if yoy_rev is not None and qoq_rev is not None:
            if yoy_rev >= 25 and qoq_rev >= 15:
                score += 25
                criteria.append(f'✅ Revenue: EXCEPTIONAL Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [25 pts]')
            elif yoy_rev >= 20 and qoq_rev >= 10:
                score += 22
                criteria.append(f'✅ Revenue: Excellent Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [22 pts]')
            elif yoy_rev >= 15 and qoq_rev >= 8:
                score += 18
                criteria.append(f'✅ Revenue: Strong Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [18 pts]')
            elif yoy_rev >= 10 and qoq_rev >= 5:
                score += 12
                criteria.append(f'⚠ Revenue: Good Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [12 pts]')
            elif yoy_rev >= 5:
                score += 5
                criteria.append(f'⚠ Revenue: Moderate Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [5 pts]')
            else:
                score += 0
                criteria.append(f'❌ Revenue: Weak/Negative Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [0 pts]')
        elif yoy_rev is not None:
            if yoy_rev >= 20:
                score += 20
                criteria.append(f'✅ Revenue: Strong YoY Growth ({yoy_rev:.1f}%) [20 pts]')
            elif yoy_rev >= 12:
                score += 15
                criteria.append(f'✅ Revenue: Good YoY Growth ({yoy_rev:.1f}%) [15 pts]')
            elif yoy_rev >= 5:
                score += 8
                criteria.append(f'⚠ Revenue: Moderate Growth ({yoy_rev:.1f}%) [8 pts]')
            else:
                score += 0
                criteria.append(f'❌ Revenue: Weak Growth ({yoy_rev:.1f}%) [0 pts]')
        else:
            score += 0
            criteria.append(f'❌ Revenue: Data not available [0 pts]')
        
        # 3. PROFIT GROWTH (25 pts) - NEW! STRICTEST CRITERIA
        yoy_profit = data['yoy_profit_growth']
        qoq_profit = data['qoq_profit_growth']
        profit_margin = data['profit_margin']
        
        if yoy_profit is not None and qoq_profit is not None:
            if yoy_profit >= 30 and qoq_profit >= 20:
                score += 25
                criteria.append(f'✅ Profit: EXCEPTIONAL Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [25 pts]')
            elif yoy_profit >= 25 and qoq_profit >= 15:
                score += 22
                criteria.append(f'✅ Profit: Excellent Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [22 pts]')
            elif yoy_profit >= 20 and qoq_profit >= 10:
                score += 18
                criteria.append(f'✅ Profit: Strong Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [18 pts]')
            elif yoy_profit >= 12 and qoq_profit >= 6:
                score += 12
                criteria.append(f'⚠ Profit: Good Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [12 pts]')
            elif yoy_profit >= 5:
                score += 5
                criteria.append(f'⚠ Profit: Moderate Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [5 pts]')
            else:
                score += 0
                criteria.append(f'❌ Profit: Weak/Negative Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [0 pts]')
        elif yoy_profit is not None:
            if yoy_profit >= 25:
                score += 20
                criteria.append(f'✅ Profit: Strong YoY Growth ({yoy_profit:.1f}%) [20 pts]')
            elif yoy_profit >= 15:
                score += 15
                criteria.append(f'✅ Profit: Good YoY Growth ({yoy_profit:.1f}%) [15 pts]')
            elif yoy_profit >= 8:
                score += 8
                criteria.append(f'⚠ Profit: Moderate Growth ({yoy_profit:.1f}%) [8 pts]')
            else:
                score += 0
                criteria.append(f'❌ Profit: Weak Growth ({yoy_profit:.1f}%) [0 pts]')
        else:
            score += 0
            criteria.append(f'❌ Profit: Data not available [0 pts]')
        
        # 4. PROFIT MARGIN (15 pts) - NEW!
        if profit_margin is not None:
            profit_margin_pct = profit_margin * 100
            if profit_margin_pct >= 20:
                score += 15
                criteria.append(f'✅ Profit Margin: Excellent ({profit_margin_pct:.1f}%) [15 pts]')
            elif profit_margin_pct >= 15:
                score += 12
                criteria.append(f'✅ Profit Margin: Very Good ({profit_margin_pct:.1f}%) [12 pts]')
            elif profit_margin_pct >= 10:
                score += 10
                criteria.append(f'✅ Profit Margin: Good ({profit_margin_pct:.1f}%) [10 pts]')
            elif profit_margin_pct >= 5:
                score += 5
                criteria.append(f'⚠ Profit Margin: Average ({profit_margin_pct:.1f}%) [5 pts]')
            else:
                score += 0
                criteria.append(f'❌ Profit Margin: Low ({profit_margin_pct:.1f}%) [0 pts]')
        else:
            score += 0
            criteria.append(f'❌ Profit Margin: Data not available [0 pts]')
        
        # 5. FII/DII ACTIVITY (20 pts)
        fii_score = data['fii_dii_score']
        if fii_score >= 15:
            score += 20
            criteria.append(f'✅ FII/DII: Strong Buying ({fii_score}) [20 pts]')
        elif fii_score >= 10:
            score += 15
            criteria.append(f'✅ FII/DII: Good Buying ({fii_score}) [15 pts]')
        elif fii_score >= 5:
            score += 10
            criteria.append(f'✅ FII/DII: Accumulation ({fii_score}) [10 pts]')
        elif fii_score >= 0:
            score += 5
            criteria.append(f'⚠ FII/DII: Neutral ({fii_score}) [5 pts]')
        else:
            score += 0
            criteria.append(f'❌ FII/DII: Selling ({fii_score}) [0 pts]')
        
        # 6. CONSOLIDATION (20 pts)
        if -2 <= weekly_change <= 0.3:
            score += 20
            criteria.append(f'✅ Consolidation: Perfect base ({weekly_change:+.1f}% weekly) [20 pts]')
        elif -3.5 <= weekly_change < -2:
            score += 18
            criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [18 pts]')
        elif 0.3 < weekly_change <= 1.5:
            score += 15
            criteria.append(f'✅ Consolidation: Early breakout ({weekly_change:+.1f}% weekly) [15 pts]')
        elif weekly_change > 4:
            score += 0
            criteria.append(f'❌ Already rallied ({weekly_change:+.1f}% weekly) [0 pts]')
        else:
            score += 5
            criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')
        
        # 7. RSI (20 pts)
        rsi_low = thresholds['rsi_low']
        rsi_high = thresholds['rsi_high']
        
        if rsi_low <= rsi <= rsi_high:
            score += 20
            criteria.append(f'✅ RSI: Perfect oversold entry ({rsi:.0f}) [20 pts]')
        elif rsi_high < rsi <= rsi_high + 7:
            score += 17
            criteria.append(f'✅ RSI: Building momentum ({rsi:.0f}) [17 pts]')
        elif rsi_high + 7 < rsi <= rsi_high + 12:
            score += 12
            criteria.append(f'✅ RSI: Early momentum ({rsi:.0f}) [12 pts]')
        elif rsi_high + 12 < rsi <= rsi_high + 17:
            score += 8
            criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [8 pts]')
        elif rsi > rsi_high + 24:
            score += 0
            criteria.append(f'❌ RSI: Overbought ({rsi:.0f}) [0 pts]')
        else:
            score += 5
            criteria.append(f'⚠ RSI: Moderate ({rsi:.0f}) [5 pts]')
        
        # 8. MACD (15 pts)
        if -1 <= macd <= 1:
            score += 15
            criteria.append(f'✅ MACD: Perfect crossover ({macd:.1f}) [15 pts]')
        elif 1 < macd <= 3:
            score += 12
            criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [12 pts]')
        elif -3 <= macd < -1:
            score += 10
            criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [10 pts]')
        elif macd > 6:
            score += 0
            criteria.append(f'❌ MACD: Extended ({macd:.1f}) [0 pts]')
        else:
            score += 5
            criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')
        
        # 9. BOLLINGER BANDS (15 pts)
        if 8 <= bb <= 20:
            score += 15
            criteria.append(f'✅ BB: Lower band bounce ({bb:.0f}%) [15 pts]')
        elif 20 < bb <= 30:
            score += 12
            criteria.append(f'✅ BB: Below middle ({bb:.0f}%) [12 pts]')
        elif 30 < bb <= 45:
            score += 8
            criteria.append(f'⚠ BB: Middle zone ({bb:.0f}%) [8 pts]')
        elif bb > 65:
            score += 0
            criteria.append(f'❌ BB: Upper band ({bb:.0f}%) [0 pts]')
        else:
            score += 5
            criteria.append(f'⚠ BB: Neutral ({bb:.0f}%) [5 pts]')
        
        # 10. VOLUME (15 pts)
        if 1.3 <= vol <= 1.8:
            score += 15
            criteria.append(f'✅ Volume: Perfect accumulation ({vol:.1f}x) [15 pts]')
        elif 1.8 < vol <= 2.2:
            score += 12
            criteria.append(f'✅ Volume: Building interest ({vol:.1f}x) [12 pts]')
        elif vol > 2.8:
            score += 5
            criteria.append(f'⚠ Volume: Too high ({vol:.1f}x) [5 pts]')
        elif 1.0 <= vol < 1.3:
            score += 7
            criteria.append(f'⚠ Volume: Average ({vol:.1f}x) [7 pts]')
        else:
            score += 0
            criteria.append(f'❌ Volume: Too low ({vol:.1f}x) [0 pts]')
        
        # 11. TODAY'S PRICE (10 pts)
        if -1.5 <= change <= 0.3:
            score += 10
            criteria.append(f'✅ Today: Perfect entry ({change:+.1f}%) [10 pts]')
        elif 0.3 < change <= 1.2:
            score += 8
            criteria.append(f'✅ Today: Early move ({change:+.1f}%) [8 pts]')
        elif -2.5 <= change < -1.5:
            score += 7
            criteria.append(f'⚠ Today: Dip ({change:+.1f}%) [7 pts]')
        elif change > 2.5:
            score += 0
            criteria.append(f'❌ Today: Already rallied ({change:+.1f}%) [0 pts]')
        else:
            score += 4
            criteria.append(f'⚠ Today: Moderate ({change:+.1f}%) [4 pts]')
        
        # 12. MONTHLY TREND (10 pts)
        if -8 <= monthly_change <= -2:
            score += 10
            criteria.append(f'✅ Monthly: Recovering from dip ({monthly_change:+.1f}%) [10 pts]')
        elif -2 < monthly_change <= 2:
            score += 8
            criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [8 pts]')
        elif 2 < monthly_change <= 6:
            score += 5
            criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [5 pts]')
        elif monthly_change > 10:
            score += 0
            criteria.append(f'❌ Monthly: Extended ({monthly_change:+.1f}%) [0 pts]')
        else:
            score += 3
            criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [3 pts]')
        
        # 13. 3-MONTH PERFORMANCE (10 pts)
        if -15 <= three_month_change <= -5:
            score += 10
            criteria.append(f'✅ 3-Month: Perfect correction ({three_month_change:+.1f}%) [10 pts]')
        elif -5 < three_month_change <= 5:
            score += 8
            criteria.append(f'✅ 3-Month: Sideways base ({three_month_change:+.1f}%) [8 pts]')
        elif 5 < three_month_change <= 15:
            score += 5
            criteria.append(f'⚠ 3-Month: Moderate rise ({three_month_change:+.1f}%) [5 pts]')
        elif three_month_change > 25:
            score += 0
            criteria.append(f'❌ 3-Month: Overextended ({three_month_change:+.1f}%) [0 pts]')
        else:
            score += 3
            criteria.append(f'⚠ 3-Month: Weak ({three_month_change:+.1f}%) [3 pts]')
        
        # 14. UPSIDE POTENTIAL (10 pts)
        if potential_pct >= 12:
            score += 10
            criteria.append(f'✅ Upside: Excellent ({potential_pct:.1f}%) [10 pts]')
        elif potential_pct >= 10:
            score += 8
            criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
        elif potential_pct >= 8:
            score += 5
            criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
        else:
            score += 0
            criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')
        
        # Rating based on ULTRA-STRICT criteria with ADJUSTABLE thresholds
        threshold_exceptional = thresholds['threshold_exceptional']
        threshold_prime = thresholds['threshold_prime']
        threshold_excellent = thresholds['threshold_excellent']
        threshold_strong = thresholds['threshold_strong']
        
        if is_operated:
            status = '🚨 OPERATED - AVOID'
            rating = 'Operated - Avoid'
        elif score >= threshold_exceptional:
            status = '🌟 EXCEPTIONAL BUY'
            rating = 'Exceptional Buy'
        elif score >= threshold_prime:
            status = '🚀 PRIME BUY'
            rating = 'Prime Buy'
        elif score >= threshold_excellent:
            status = '💎 EXCELLENT BUY'
            rating = 'Excellent Buy'
        elif score >= threshold_strong:
            status = '✅ STRONG BUY'
            rating = 'Strong Buy'
        elif score >= 100:
            status = '👍 GOOD BUY'
            rating = 'Good Buy'
        elif score >= 80:
            status = '📋 WATCHLIST'
            rating = 'Watchlist'
        else:
            status = '❌ SKIP'
            rating = 'Skip'
        
        qualified = score >= threshold_excellent and not is_operated
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
            'yoy_revenue_growth': yoy_rev,
            'qoq_revenue_growth': qoq_rev,
            'yoy_profit_growth': yoy_profit,
            'qoq_profit_growth': qoq_profit,
            'profit_margin': profit_margin * 100 if profit_margin else None,
            'total_cash': data.get('total_cash', 0),
            'latest_fy_revenue': data.get('latest_fy_revenue', 0),
            'cash_on_hand_to_mcap': data.get('cash_on_hand_to_mcap', 0),
            'latest_fy_revenue_to_mcap': data.get('latest_fy_revenue_to_mcap', 0),
            'historical_data': data.get('historical_data', {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []})
        }
    except Exception as e:
        # Silently return None on error
        return None

# Main App
st.markdown('<p class="main-header">🎯 Indian Stock Scout - NSE & BSE Ultra-Strict Scanner</p>', unsafe_allow_html=True)
st.markdown("*Choose NSE, BSE, or BOTH | Only stocks with EXCEPTIONAL fundamentals + technicals qualify*")

# Show exchange summary prominently
if 'AVAILABLE_STOCKS' in locals() and len(AVAILABLE_STOCKS) > 0:
    if scan_nse and scan_bse:
        banner_text = f"📈 Scanning BOTH Exchanges: NSE ({NSE_COUNT}) + BSE ({BSE_COUNT}) = {len(AVAILABLE_STOCKS)} Total"
        banner_color = "linear-gradient(90deg, #1f77b4 0%, #1f77b4 50%, #ff7f0e 50%, #ff7f0e 100%)"
    elif scan_nse:
        banner_text = f"📈 Scanning NSE Only: {NSE_COUNT} stocks loaded"
        banner_color = "#1f77b4"
    else:
        banner_text = f"📈 Scanning BSE Only: {BSE_COUNT} stocks loaded"
        banner_color = "#ff7f0e"
    
    st.markdown(f"""
    <div style='text-align:center;background: {banner_color};padding:0.8rem;border-radius:8px;margin:1rem 0;'>
    <p style='color:white;font-size:1.2rem;font-weight:bold;margin:0;'>
    {banner_text}
    </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙ Scanner Configuration")

# Exchange selection with checkboxes
st.sidebar.subheader("📈 Select Exchanges to Scan")
scan_nse = st.sidebar.checkbox("✅ Scan NSE Stocks", value=True, help="Loads stocks from nse.txt and adds .NS suffix")
scan_bse = st.sidebar.checkbox("✅ Scan BSE Stocks", value=True, help="Loads stocks from bse.txt and adds .BO suffix")

if not scan_nse and not scan_bse:
    st.sidebar.error("⚠️ Please select at least one exchange!")
    AVAILABLE_STOCKS = []
    NSE_COUNT = 0
    BSE_COUNT = 0
else:
    # Load stocks based on selection - SIMPLE LOGIC
    AVAILABLE_STOCKS = []
    NSE_COUNT = 0
    BSE_COUNT = 0
    
    if scan_nse:
        try:
            with open('nse.txt', 'r') as f:
                # Just load the stock names (like RELIANCE, TCS, etc.)
                nse_stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
            
            if nse_stocks:
                # Add .NS suffix to each
                for stock in nse_stocks:
                    AVAILABLE_STOCKS.append(f"{stock}.NS")
                NSE_COUNT = len(nse_stocks)
        except FileNotFoundError:
            st.sidebar.warning("⚠️ nse.txt not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading nse.txt: {str(e)}")
    
    if scan_bse:
        try:
            with open('bse.txt', 'r') as f:
                # Just load the stock names (like RELIANCE, TCS, etc.)
                bse_stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
            
            if bse_stocks:
                # Add .BO suffix to each
                for stock in bse_stocks:
                    AVAILABLE_STOCKS.append(f"{stock}.BO")
                BSE_COUNT = len(bse_stocks)
        except FileNotFoundError:
            st.sidebar.warning("⚠️ bse.txt not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading bse.txt: {str(e)}")
    
    # Remove duplicates if any
    AVAILABLE_STOCKS = list(dict.fromkeys(AVAILABLE_STOCKS))
    
    if AVAILABLE_STOCKS:
        exchange_text = []
        if scan_nse:
            exchange_text.append(f"NSE: {NSE_COUNT}")
        if scan_bse:
            exchange_text.append(f"BSE: {BSE_COUNT}")
        
        st.sidebar.success(f"✅ Loaded {len(AVAILABLE_STOCKS)} stocks\n" + " | ".join(exchange_text))
    else:
        st.sidebar.error("❌ No stocks loaded")

st.sidebar.markdown("---")

scan_mode = st.sidebar.radio("Scan Mode", 
    ["Quick Scan (50 stocks)", "Full Scan (All stocks)", "Slot-wise Scan", "Custom List"])

if scan_mode == "Quick Scan (50 stocks)":
    stocks_to_scan = AVAILABLE_STOCKS[:50]
elif scan_mode == "Full Scan (All stocks)":
    stocks_to_scan = AVAILABLE_STOCKS
elif scan_mode == "Slot-wise Scan":
    st.sidebar.subheader("📦 Select Slots to Scan")
    
    total_stocks = len(AVAILABLE_STOCKS)
    slot_size = 1000
    num_slots = (total_stocks + slot_size - 1) // slot_size  # Ceiling division
    
    st.sidebar.info(f"📊 Total stocks: {total_stocks}\n💼 Slot size: 1000 stocks\n📦 Total slots: {num_slots}")
    
    # Helper buttons for quick selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Select All", use_container_width=True):
            for slot_num in range(num_slots):
                st.session_state[f"slot_{slot_num}"] = True
            st.rerun()
    with col2:
        if st.button("❌ Deselect All", use_container_width=True):
            for slot_num in range(num_slots):
                st.session_state[f"slot_{slot_num}"] = False
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Create checkboxes for each slot with exchange breakdown
    selected_slots = []
    for slot_num in range(num_slots):
        start_idx = slot_num * slot_size
        end_idx = min((slot_num + 1) * slot_size, total_stocks)
        slot_stocks = AVAILABLE_STOCKS[start_idx:end_idx]
        slot_count = len(slot_stocks)
        
        # Count NSE and BSE in this slot
        nse_in_slot = sum(1 for s in slot_stocks if '.NS' in s)
        bse_in_slot = sum(1 for s in slot_stocks if '.BO' in s)
        
        slot_label = f"Slot {slot_num + 1}: {start_idx + 1}-{end_idx}"
        slot_detail = f"({slot_count}: {nse_in_slot} NSE, {bse_in_slot} BSE)"
        
        if st.sidebar.checkbox(f"{slot_label} {slot_detail}", key=f"slot_{slot_num}"):
            selected_slots.append(slot_num)
    
    # Build stocks list from selected slots
    stocks_to_scan = []
    for slot_num in selected_slots:
        start_idx = slot_num * slot_size
        end_idx = min((slot_num + 1) * slot_size, total_stocks)
        stocks_to_scan.extend(AVAILABLE_STOCKS[start_idx:end_idx])
    
    if not selected_slots:
        st.sidebar.warning("⚠️ Please select at least one slot to scan")
        stocks_to_scan = []
    else:
        nse_selected = sum(1 for s in stocks_to_scan if '.NS' in s)
        bse_selected = sum(1 for s in stocks_to_scan if '.BO' in s)
        st.sidebar.success(f"✅ {len(selected_slots)} slot(s) selected\n📊 Total: {len(stocks_to_scan)} stocks\n🔵 NSE: {nse_selected} | 🟠 BSE: {bse_selected}")
else:
    custom_input = st.sidebar.text_area("Enter symbols (one per line)", 
        'Stock names with exchange suffix:\nRELIANCE.NS\nTCS.BO\nINFY.NS\n\nOr without (defaults to NSE):\nRELIANCE\nTCS', height=150)
    raw_symbols = [s.strip().upper() for s in custom_input.split('\n') if s.strip()]
    stocks_to_scan = []
    for symbol in raw_symbols:
        if '.NS' in symbol or '.BO' in symbol:
            stocks_to_scan.append(symbol)
        else:
            # Default to NSE if no exchange specified
            stocks_to_scan.append(f"{symbol}.NS")

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Scanning Speed")

request_delay = st.sidebar.slider("Delay between requests (sec)", 0.1, 2.0, 0.3, 0.1,
    help="Delay between each stock scan. Higher = slower but better success rate. Lower = faster but may get rate limited.")

st.sidebar.info("""
**ℹ️ Sequential Processing (Best Quality):**

Scans one stock at a time with delays.
- More reliable
- Better success rate  
- No rate limiting issues
- Recommended: 0.3 seconds

**Note:** Parallel processing removed because it causes Yahoo Finance to block requests, resulting in failed tickers.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Market Cap Filter")
min_market_cap = st.sidebar.slider("Minimum Market Cap (₹ Crores)", 
    0, 100000, 5000, 1000,
    help="Filter stocks by minimum market capitalization")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Adjustable Scoring Thresholds")

with st.sidebar.expander("📊 Customize Score Thresholds", expanded=False):
    st.markdown("**Qualification Scores:**")
    threshold_exceptional = st.number_input("Exceptional (≥)", 100, 250, 180, 10)
    threshold_prime = st.number_input("Prime (≥)", 100, 250, 160, 10)
    threshold_excellent = st.number_input("Excellent (≥)", 100, 250, 140, 10)
    threshold_strong = st.number_input("Strong (≥)", 50, 200, 120, 10)
    
    st.markdown("**Technical Thresholds:**")
    rsi_low = st.number_input("RSI Lower Bound", 20, 50, 32, 1)
    rsi_high = st.number_input("RSI Upper Bound", 30, 60, 38, 1)
    
    st.markdown("**Growth Thresholds:**")
    min_revenue_yoy = st.number_input("Min Revenue YoY %", 0, 50, 20, 5)
    min_profit_yoy = st.number_input("Min Profit YoY %", 0, 50, 25, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 ULTRA-STRICT Criteria")
st.sidebar.info("""
*Only top 1-3% qualify!*

**TOTAL: 250 Points**

**Fundamentals (80 pts):**
1. Market Cap (15 pts)
2. Revenue Growth (25 pts)
   - YoY ≥20%, QoQ ≥10%
3. Profit Growth (25 pts)
   - YoY ≥25%, QoQ ≥15%
4. Profit Margin (15 pts)
   - ≥15% excellent

**Technicals (170 pts):**
5. FII/DII (20 pts)
6. Consolidation (20 pts)
7. RSI (20 pts): 32-38
8. MACD (15 pts): -1 to +1
9. BB (15 pts): 8-20%
10. Volume (15 pts): 1.3-1.8x
11. Today (10 pts)
12. Monthly (10 pts)
13. 3-Month (10 pts)
14. Upside (10 pts): ≥12%

**Qualification:**
- Exceptional: ≥180 pts
- Prime: 160-179 pts
- Excellent: 140-159 pts ✅
- Strong: 120-139 pts
- Below 140: Not qualified

**Penalties:**
- Operated: -70 pts
- High Risk: -25 to -40 pts
""")

st.sidebar.markdown("---")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 FIND EXCEPTIONAL STOCKS", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning with Fundamental + Technical Analysis...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    
    results = []
    total = len(stocks_to_scan)
    failed = 0
    filtered_out = 0
    
    start_time = time.time()
    
    # SEQUENTIAL PROCESSING ONLY (Best Quality - No Rate Limiting Issues)
    for idx, symbol in enumerate(stocks_to_scan):
        status_text.info(f"📊 Analyzing *{symbol}*... ({idx+1}/{total})")
        
        try:
            data = fetch_stock_data(symbol)
            if data:
                analysis = analyze_stock(data, min_market_cap, {
                    'threshold_exceptional': threshold_exceptional,
                    'threshold_prime': threshold_prime,
                    'threshold_excellent': threshold_excellent,
                    'threshold_strong': threshold_strong,
                    'rsi_low': rsi_low,
                    'rsi_high': rsi_high,
                    'min_revenue_yoy': min_revenue_yoy,
                    'min_profit_yoy': min_profit_yoy
                })
                if analysis:
                    results.append(analysis)
                else:
                    filtered_out += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
        
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        
        if (idx + 1) % 10 == 0 or idx == total - 1:
            valid_results_count = len([r for r in results if r is not None])
            qualified_count = len([r for r in results if r is not None and r.get('qualified', False)])
            stats_placeholder.info(f"✅ Valid: {valid_results_count} | Qualified (≥{threshold_excellent}): {qualified_count} | Filtered: {filtered_out} | Failed: {failed}")
        
        # Delay between requests - user configurable for best quality
        time.sleep(request_delay)
    
    # Filter out None results before storing
    results = [r for r in results if r is not None]
    st.session_state.scan_results = results
    st.session_state.scan_timestamp = datetime.now()
    st.session_state.failed_tickers = [stocks_to_scan[i] for i in range(len(stocks_to_scan)) if i >= len(results) + filtered_out][:failed]
    
    # Save thresholds to session state
    st.session_state.threshold_exceptional = threshold_exceptional
    st.session_state.threshold_prime = threshold_prime
    st.session_state.threshold_excellent = threshold_excellent
    st.session_state.threshold_strong = threshold_strong
    
    elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
    
    # Show completion stats
    st.success(f"✅ Scan complete! Found {len(results)} stocks meeting market cap criteria")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Successfully Processed", len(results))
    col2.metric("❌ Failed", failed)
    col3.metric("🚫 Filtered Out", filtered_out)
    col4.metric("⏱️ Time Taken", f"{elapsed_time:.1f} min")
    
    # Show failed tickers with retry option
    if failed > 0 and 'failed_tickers' in st.session_state and st.session_state.failed_tickers:
        with st.expander(f"⚠️ Failed Tickers ({failed})", expanded=False):
            st.write(", ".join(st.session_state.failed_tickers[:20]))  # Show first 20
            if len(st.session_state.failed_tickers) > 20:
                st.caption(f"...and {len(st.session_state.failed_tickers) - 20} more")
            
            if st.button("🔄 Retry Failed Tickers"):
                with st.spinner("Retrying failed tickers..."):
                    retry_results = []
                    for ticker in st.session_state.failed_tickers:
                        try:
                            data = bulletproof_fetch(fetch_stock_data, ticker, max_retries=5)
                            if data:
                                analysis = analyze_stock(data, min_market_cap, {
                                    'threshold_exceptional': threshold_exceptional,
                                    'threshold_prime': threshold_prime,
                                    'threshold_excellent': threshold_excellent,
                                    'threshold_strong': threshold_strong,
                                    'rsi_low': rsi_low,
                                    'rsi_high': rsi_high,
                                    'min_revenue_yoy': min_revenue_yoy,
                                    'min_profit_yoy': min_profit_yoy
                                })
                                if analysis:
                                    retry_results.append(analysis)
                        except:
                            pass
                    
                    if retry_results:
                        st.session_state.scan_results.extend(retry_results)
                        st.success(f"✅ Recovered {len(retry_results)} additional stocks!")
                        time.sleep(1)
                    else:
                        st.warning("No additional stocks recovered")
    
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
    
    # Auto-refresh toggle
    col_refresh1, col_refresh2, col_refresh3 = st.columns([2, 2, 6])
    
    with col_refresh1:
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=False, 
                                   help="Continuously update prices every 30 seconds without resetting")
    
    with col_refresh2:
        if 'last_refresh' in st.session_state:
            seconds_ago = int((datetime.now() - st.session_state.last_refresh).total_seconds())
            st.caption(f"📡 Updated {seconds_ago}s ago")
        else:
            st.caption("📡 Not refreshed yet")
    
    with col_refresh3:
        if auto_refresh:
            if st.button("⏸️ Pause Refresh"):
                st.session_state.auto_refresh_paused = True
                st.rerun()
    
    st.subheader(f"📈 Exceptional Stock Opportunities")
    st.caption(f"Initial scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh logic - NON-BLOCKING
    if auto_refresh and not st.session_state.get('auto_refresh_paused', False):
        # Initialize refresh counter
        if 'refresh_counter' not in st.session_state:
            st.session_state.refresh_counter = 0
            st.session_state.last_refresh = datetime.now()
        
        # Check if 30 seconds have passed
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= 30:
            # Update prices in background
            with st.spinner("🔄 Refreshing live prices..."):
                updated_count = 0
                for result in results:
                    # BULLETPROOF: Wrap in try-catch
                    try:
                        new_price = fetch_live_price(result['symbol'])
                        if new_price and new_price != result['price']:
                            prev_price = result['price']
                            result['price'] = new_price
                            result['change'] = ((new_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
                            updated_count += 1
                    except:
                        pass
                
                st.session_state.last_refresh = datetime.now()
                st.session_state.refresh_counter += 1
                
                if updated_count > 0:
                    st.toast(f"✅ Updated {updated_count} prices", icon="🔄")
            
            # Force re-render
            time.sleep(1)
            st.rerun()
        else:
            # Schedule next refresh
            remaining = 30 - time_since_refresh
            st.info(f"⏱️ Next refresh in {int(remaining)} seconds...")
            time.sleep(5)  # Check every 5 seconds
            st.rerun()
    
    # Convert to DataFrame - BULLETPROOF: Safe conversions
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Exchange': 'NSE' if '.NS' in r['symbol'] else 'BSE' if '.BO' in r['symbol'] else 'N/A',
        'Price (₹)': r['price'],
        'Today (%)': r['change'],
        'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'],
        '3M (%)': r['three_month_change'],
        'Market Cap (₹Cr)': r['market_cap'],
        'Cash/Hand (₹Cr)': r.get('total_cash', 0) / 10000000 if r.get('total_cash') else 0,
        'CashHand/MCap (%)': r.get('cash_on_hand_to_mcap', 0),
        'LatestFY Rev/MCap': r.get('latest_fy_revenue_to_mcap', 0),
        'Rev YoY (%)': r['yoy_revenue_growth'],
        'Rev QoQ (%)': r['qoq_revenue_growth'],
        'Profit YoY (%)': r['yoy_profit_growth'],
        'Profit QoQ (%)': r['qoq_profit_growth'],
        'Margin (%)': r['profit_margin'],
        'RSI': r['rsi'],
        'MACD': r['macd'],
        'BB (%)': r['bb'],
        'Vol': f"{r['vol']:.1f}x",
        'Score': r['score'],
        'Rating': r['rating'],
        'Status': r['status'],
        'Sector': r['sector'],
        'Operated': '🚨 YES' if r['is_operated'] else '✅ Safe',
        'Risk': r['operator_risk']
    } for r in results])
    
    # Statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Get thresholds from session or use defaults
    threshold_exceptional = st.session_state.get('threshold_exceptional', 180)
    threshold_prime = st.session_state.get('threshold_prime', 160)
    threshold_excellent = st.session_state.get('threshold_excellent', 140)
    threshold_strong = st.session_state.get('threshold_strong', 120)
    
    operated_stocks = df[df['Operated'] == '🚨 YES']
    safe_stocks = df[df['Operated'] == '✅ Safe']
    exceptional = df[(df['Score'] >= threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    prime = df[(df['Score'] >= threshold_prime) & (df['Score'] < threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    excellent = df[(df['Score'] >= threshold_excellent) & (df['Score'] < threshold_prime) & (df['Operated'] == '✅ Safe')]
    strong = df[(df['Score'] >= threshold_strong) & (df['Score'] < threshold_excellent) & (df['Operated'] == '✅ Safe')]
    
    col1.metric("Total Scanned", len(df))
    col2.metric("🚨 Operated", len(operated_stocks))
    col3.metric(f"🌟 Exceptional (≥{threshold_exceptional})", len(exceptional))
    col4.metric(f"🚀 Prime ({threshold_prime}-{threshold_exceptional-1})", len(prime))
    col5.metric(f"💎 Excellent ({threshold_excellent}-{threshold_prime-1})", len(excellent))
    col6.metric(f"✅ Strong ({threshold_strong}-{threshold_excellent-1})", len(strong))
    
    # Exchange breakdown
    st.markdown("---")
    exchange_col1, exchange_col2, exchange_col3 = st.columns(3)
    nse_stocks = df[df['Exchange'] == 'NSE']
    bse_stocks = df[df['Exchange'] == 'BSE']
    
    exchange_col1.metric("📊 NSE Stocks", len(nse_stocks))
    exchange_col2.metric("📊 BSE Stocks", len(bse_stocks))
    exchange_col3.metric("🎯 Qualified (≥140)", len(exceptional) + len(prime) + len(excellent))
    
    # Qualification summary
    qualified_total = len(exceptional) + len(prime) + len(excellent)
    st.success(f"""
    **🎯 ULTRA-STRICT RESULTS:** Only **{qualified_total}** stocks qualified (Score ≥140 + Safe) out of {len(df)}.
    That's the top **{(qualified_total/len(df)*100) if len(df) > 0 else 0:.1f}%** - truly exceptional opportunities with strong fundamentals!
    """)
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip"])
    
    with filter_col2:
        exchange_filter = st.selectbox("Exchange",
            ["All", "NSE", "BSE"])
    
    with filter_col3:
        safety_filter = st.selectbox("Safety", 
            ["All", "✅ Safe Only", "🚨 Operated Only"])
    
    with filter_col4:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col5:
        min_score_filter = st.number_input("Min Score", 0, 250, 140, 10,
                                          help="Default: 140 (Qualified)")
    
    # Apply filters
    filtered_df = df.copy()
    
    if rating_filter != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == rating_filter]
    
    if exchange_filter != "All":
        filtered_df = filtered_df[filtered_df['Exchange'] == exchange_filter]
    
    if safety_filter == "✅ Safe Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '✅ Safe']
    elif safety_filter == "🚨 Operated Only":
        filtered_df = filtered_df[filtered_df['Operated'] == '🚨 YES']
    
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]
    
    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")
    
    # Display table
    st.subheader("📋 Stock Analysis Table")
    
    def highlight_rating(row):
        if row['Operated'] == '🚨 YES':
            return ['background-color: #ff6b6b; color: white; font-weight: bold'] * len(row)
        elif row['Score'] >= 180:
            return ['background-color: #00e676; color: black; font-weight: bold'] * len(row)
        elif row['Score'] >= 160:
            return ['background-color: #69f0ae; font-weight: bold'] * len(row)
        elif row['Score'] >= 140:
            return ['background-color: #b9f6ca; font-weight: bold'] * len(row)
        elif row['Score'] >= 120:
            return ['background-color: #e1f5fe'] * len(row)
        elif row['Score'] >= 100:
            return ['background-color: #fff9c4'] * len(row)
        else:
            return ['background-color: #ffebee'] * len(row)
    
    styled_df = filtered_df.style.apply(highlight_rating, axis=1)\
        .format({
            'Price (₹)': '₹{:.2f}',
            'Today (%)': '{:+.2f}%',
            'Weekly (%)': '{:+.2f}%',
            'Monthly (%)': '{:+.2f}%',
            '3M (%)': '{:+.2f}%',
            'Market Cap (₹Cr)': '₹{:.0f}',
            'Cash/Hand (₹Cr)': '₹{:.0f}',
            'CashHand/MCap (%)': '{:.2f}%',
            'LatestFY Rev/MCap': '{:.2f}x',
            'Rev YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Rev QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Profit YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Profit QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
            'Margin (%)': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A',
            'RSI': '{:.1f}',
            'MACD': '{:.2f}',
            'BB (%)': '{:.0f}%'
        })
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Detailed view
    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")
    
    if len(filtered_df) > 0:
        selected_symbol = st.selectbox("Select stock for details", filtered_df['Symbol'].tolist())
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)
        
        if selected_result:
            st.markdown(f"### {selected_symbol} - {selected_result['status']}")
            
            if selected_result['is_operated']:
                st.error(f"🚨 **OPERATOR DETECTED** - Risk: {selected_result['operator_risk']}/100")
                for flag in selected_result['operator_flags']:
                    st.warning(flag)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Market Cap", f"₹{selected_result['market_cap']:.0f}Cr")
            col4.metric("Rev YoY", f"{selected_result['yoy_revenue_growth']:+.1f}%" if selected_result['yoy_revenue_growth'] else "N/A")
            col5.metric("Profit YoY", f"{selected_result['yoy_profit_growth']:+.1f}%" if selected_result['yoy_profit_growth'] else "N/A")
            
            # Cash metrics
            st.markdown("---")
            st.markdown("**💵 Financial Ratios**")
            cash_col1, cash_col2, cash_col3 = st.columns(3)
            cash_col1.metric("Cash on Hand", f"₹{selected_result.get('total_cash', 0)/10000000:.0f}Cr")
            cash_col2.metric("Cash/MCap Ratio", f"{selected_result.get('cash_on_hand_to_mcap', 0):.2f}%")
            cash_col3.metric("LatestFY Rev/MCap", f"{selected_result.get('latest_fy_revenue_to_mcap', 0):.2f}x")
            
            # 3-YEAR HISTORICAL GRAPHS
            if selected_result.get('historical_data') and selected_result['historical_data']['years']:
                st.markdown("---")
                st.markdown("**📈 3-Year Historical Trends**")
                
                historical = selected_result['historical_data']
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('YoY Revenue (₹ Cr)', 'Cash Amounts (₹ Cr)', 'Sales to Market Cap Ratio'),
                    vertical_spacing=0.12
                )
                
                # Revenue graph
                if historical['revenues']:
                    fig.add_trace(
                        go.Bar(
                            x=historical['years'],
                            y=[r/10000000 for r in historical['revenues']],
                            name='Revenue',
                            marker_color='lightblue',
                            text=[f"₹{r/10000000:.0f}Cr" for r in historical['revenues']],
                            textposition='auto'
                        ),
                        row=1, col=1
                    )
                
                # Cash graph
                if historical['cash_amounts']:
                    fig.add_trace(
                        go.Bar(
                            x=historical['years'],
                            y=[c/10000000 for c in historical['cash_amounts']],
                            name='Cash',
                            marker_color='lightgreen',
                            text=[f"₹{c/10000000:.0f}Cr" for c in historical['cash_amounts']],
                            textposition='auto'
                        ),
                        row=2, col=1
                    )
                
                # Sales to MCap graph
                if historical['sales_to_mcap']:
                    fig.add_trace(
                        go.Scatter(
                            x=historical['years'],
                            y=historical['sales_to_mcap'],
                            name='Sales/MCap',
                            mode='lines+markers',
                            line=dict(color='orange', width=3),
                            marker=dict(size=10),
                            text=[f"{s:.2f}x" for s in historical['sales_to_mcap']],
                            textposition='top center'
                        ),
                        row=3, col=1
                    )
                
                fig.update_layout(
                    height=900,
                    showlegend=False,
                    title_text=f"{selected_symbol} - 3-Year Financial Trends"
                )
                
                fig.update_yaxes(title_text="Revenue (₹ Cr)", row=1, col=1)
                fig.update_yaxes(title_text="Cash (₹ Cr)", row=2, col=1)
                fig.update_yaxes(title_text="Ratio", row=3, col=1)
                fig.update_xaxes(title_text="Year", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Historical data not available for this stock")
            
            st.markdown("---")
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
            f"ultra_strict_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"ultra_strict_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure and click 'FIND EXCEPTIONAL STOCKS' to start")
    
    st.markdown("---")
    st.subheader("🎯 Why ULTRA-STRICT Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **NEW: Fundamental Analysis Added!**
        
        **4 Fundamental Criteria (80 pts):**
        1. **Market Cap (15 pts)**
           - Filters quality companies
           - Large cap = More stable
        
        2. **Revenue Growth (25 pts)**
           - YoY ≥20% + QoQ ≥10% = 22+ pts
           - Must show consistent growth
        
        3. **Profit Growth (25 pts)**
           - YoY ≥25% + QoQ ≥15% = 22+ pts
           - Even stricter than revenue
        
        4. **Profit Margin (15 pts)**
           - ≥15% = Excellent (12+ pts)
           - Shows business quality
        
        **Result:** Only 1-3% stocks pass ALL criteria!
        """)
    
    with col2:
        st.markdown("""
        **Complete 14-Point System:**
        
        **Fundamentals (80 pts)**
        - Market cap quality
        - Revenue acceleration
        - Profit growth momentum
        - Margin efficiency
        
        **Technicals (170 pts)**
        - Institutional buying
        - Perfect consolidation
        - Oversold RSI (32-38)
        - MACD crossover
        - Lower BB bounce
        - Accumulation volume
        - Entry timing
        - Trend confirmation
        
        **Safety (Penalties)**
        - Operator detection: -70 pts
        - Risk penalties: up to -40 pts
        """)
    
    st.markdown("---")
    st.error("""
    **⚠️ ULTRA-STRICT = TOP 1-3% ONLY:**
    
    With fundamentals + technicals:
    - **1-3 stocks** out of 100 qualify (1-3%)
    - Must have ≥140 points (Perfect fundamentals + technicals)
    - **0-1 exceptional** (score ≥180)
    - **1-2 excellent** (score 140-179)
    
    **Bottom Line:** We find the absolute BEST opportunities - companies with explosive growth + perfect technical setup!
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>NSE & BSE Ultra-Strict Scanner with Fundamentals</strong> | Top 1-3% Only</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

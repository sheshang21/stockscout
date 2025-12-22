import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import io

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Indian Stock Scout - Early Buy Scanner", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.5rem;font-weight:600;color:#333;margin:1rem 0}
.metric-card{background:#f8f9fb;padding:0.8rem;border-radius:8px;border-left:4px solid #1f77b4;margin:0.5rem 0}
.stDataFrame{font-size:0.9rem}
div[data-testid="stDataFrame"] > div{background:#f8f9fb}
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
            'volumes': volumes
        }
    except Exception as e:
        return None

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
    
    # Calculate additional indicators for early detection
    weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
    monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0
    
    potential_rs = max(20, price * 0.08)  # Higher potential target
    potential_pct = (potential_rs / price) * 100
    
    score = 0
    criteria = []
    
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
    if score >= 100:
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
    
    qualified = score >= 80  # Only qualify truly excellent early opportunities
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
        'sector': SECTOR_MAP.get(data['symbol'], 'Other')
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
st.sidebar.subheader("🎯 EARLY BUY Criteria")
st.sidebar.info("""
*Find stocks BEFORE rally:*
- Qualified: Score ≥80
- Prime Buy: ≥100
- Excellent: 90-99
- Strong: 80-89
- Good: 70-79
- Watchlist: 60-69

*Key Signals:*
- RSI: 30-50 (oversold to neutral)
- MACD: -2 to +5 (turning bullish)
- Weekly: -3% to +1% (consolidating)
- Volume: 1.2-2.0x (accumulation)
- BB: 10-40% (lower zone)
- Daily: -2% to +1.5% (not rallying)
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
    st.subheader(f"📈 Early Buy Opportunities Found")
    st.caption(f"Scanned at: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Price (₹)': r['price'],
        'Today (%)': r['change'],
        'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'],
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
        'Met': r['met_count']
    } for r in results])
    
    # Statistics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    prime = df[df['Score'] >= 100]
    excellent = df[(df['Score'] >= 90) & (df['Score'] < 100)]
    strong = df[(df['Score'] >= 80) & (df['Score'] < 90)]
    good = df[(df['Score'] >= 70) & (df['Score'] < 80)]
    watchlist = df[(df['Score'] >= 60) & (df['Score'] < 70)]
    skip = df[df['Score'] < 60]
    
    col1.metric("Total", len(df))
    col2.metric("🚀 Prime Buy (≥100)", len(prime))
    col3.metric("🌟 Excellent (90-99)", len(excellent))
    col4.metric("💎 Strong (80-89)", len(strong))
    col5.metric("✅ Good (70-79)", len(good))
    col6.metric("👍 Watch (60-69)", len(watchlist))
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip"])
    
    with filter_col2:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col3:
        trend_filter = st.selectbox("Trend", 
            ["All", "Strong Uptrend", "Uptrend", "Sideways", "Downtrend"])
    
    with filter_col4:
        min_score_filter = st.number_input("Min Score", 0, 140, 0, 5)
    
    # Apply filters
    filtered_df = df.copy()
    
    if rating_filter != "All":
        filtered_df = filtered_df[filtered_df['Rating'] == rating_filter]
    
    if sector_filter != "All":
        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
    
    if trend_filter != "All":
        filtered_df = filtered_df[filtered_df['Trend'] == trend_filter]
    
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]
    
    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")
    
    # Display table with color coding
    st.subheader("📋 Stock Analysis Table")
    
    def highlight_rating(row):
        if row['Score'] >= 100:
            return ['background-color: #c3f7c3; font-weight: bold'] * len(row)  # Bright green
        elif row['Score'] >= 90:
            return ['background-color: #d4edda; font-weight: bold'] * len(row)  # Green
        elif row['Score'] >= 80:
            return ['background-color: #cfe2ff'] * len(row)  # Blue
        elif row['Score'] >= 70:
            return ['background-color: #d1ecf1'] * len(row)  # Light blue
        elif row['Score'] >= 60:
            return ['background-color: #fff3cd'] * len(row)  # Yellow
        else:
            return ['background-color: #f8d7da'] * len(row)  # Red
    
    styled_df = filtered_df.style.apply(highlight_rating, axis=1)\
        .format({
            'Price (₹)': '₹{:.2f}',
            'Today (%)': '{:+.2f}%',
            'Weekly (%)': '{:+.2f}%',
            'Monthly (%)': '{:+.2f}%',
            'Potential (₹)': '₹{:.2f}',
            'Potential (%)': '{:.2f}%',
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
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Today", f"{selected_result['change']:+.2f}%")
            col4.metric("Weekly", f"{selected_result['weekly_change']:+.2f}%")
            col5.metric("Potential", f"₹{selected_result['potential_rs']:.2f}")
            col6.metric("Rating", selected_result['rating'])
            
            st.markdown("#### Detailed Scoring Breakdown")
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
    st.subheader("🎯 Early Buy Strategy Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why This Works Better:**
        
        **We Find Stocks BEFORE Rally:**
        - Not after they've moved 5-10%
        - Catch them during consolidation
        - Enter during accumulation phase
        - Lower risk, higher reward
        
        **Key Philosophy:**
        1. ❌ **Don't Chase:** Stocks up 5%+ today
        2. ❌ **Don't FOMO:** RSI >65, already extended
        3. ✅ **Do Buy:** RSI 30-50, turning bullish
        4. ✅ **Do Wait:** For setup, not rally
        
        **1. Consolidation (25 pts)**
           - Perfect: -3% to +1% weekly
           - Healthy: -5% to -3% pullback
           - Avoid: >5% weekly rally
        
        **2. RSI Positioning (20 pts)**
           - Prime: 30-40 (oversold)
           - Good: 40-50 (building)
           - Avoid: >65 (overbought)
        """)
    
    with col2:
        st.markdown("""
        **3. MACD Turning (20 pts)**
           - Perfect: -2 to +2 (crossover zone)
           - Early: +2 to +5
           - Avoid: >8 (extended)
        
        **4. Bollinger Bands (20 pts)**
           - Prime: 10-25% (lower band)
           - Good: 25-40% (below middle)
           - Avoid: >70% (upper band)
        
        **5. Smart Money Volume (15 pts)**
           - Accumulation: 1.2-2.0x
           - Building: 2.0-2.5x
           - Avoid: >3x (distribution)
        
        **6. Today's Price (15 pts)**
           - Perfect: -2% to +0.5%
           - Early: +0.5% to +1.5%
           - Avoid: >3% (late)
        
        **7. Monthly Recovery (15 pts)**
           - Prime: -10% to -2% (recovery)
           - Base: -2% to +3%
           - Avoid: >12% (extended)
        
        **8. Upside Potential (10 pts)**
           - Target: 8-10%+ move
        """)
    
    st.markdown("---")
    st.info("""
    **🎯 BOTTOM LINE:** 
    
    This scanner finds stocks that are:
    - **Consolidating** (not rallying)
    - **Oversold/Neutral** (RSI 30-50)
    - **Turning bullish** (MACD crossing)
    - **Near support** (BB lower zone)
    - **Being accumulated** (smart money buying)
    
    **Perfect Entry = BEFORE the 5%+ move, not AFTER**
    """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>Indian Stock Scout - EARLY BUY MODE</strong> | Find stocks BEFORE they rally</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

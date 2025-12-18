import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import io

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Indian Stock Scout - LIVE Scanner", page_icon="🎯", layout="wide")

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
    """Analyze stock with VERY HARD criteria"""
    if not data:
        return None
    
    price = data['price']
    change = data['change']
    rsi = data['rsi']
    macd = data['macd']
    bb = data['bb_position']
    vol = data['vol_multiple']
    trend = data['trend']
    
    potential_rs = max(20, price * 0.05)
    potential_pct = (potential_rs / price) * 100
    
    score = 0
    criteria = []
    
    # 1. Price Potential (20 pts) - STRICT
    if potential_rs >= 30 and potential_pct >= 7:
        score += 20
        criteria.append(f'✅ Potential: Excellent (₹{potential_rs:.0f}, {potential_pct:.1f}%) [20 pts]')
    elif potential_rs >= 25 and potential_pct >= 6:
        score += 15
        criteria.append(f'✅ Potential: Very Good (₹{potential_rs:.0f}, {potential_pct:.1f}%) [15 pts]')
    elif potential_rs >= 20 and potential_pct >= 5:
        score += 10
        criteria.append(f'⚠ Potential: Good (₹{potential_rs:.0f}, {potential_pct:.1f}%) [10 pts]')
    else:
        criteria.append(f'❌ Potential: Low (₹{potential_rs:.0f}, {potential_pct:.1f}%) [0 pts]')
    
    # 2. RSI (25 pts) - VERY STRICT
    if 58 <= rsi <= 65:
        score += 25
        criteria.append(f'✅ RSI: Perfect zone ({rsi:.0f}) [25 pts]')
    elif 52 <= rsi <= 68:
        score += 18
        criteria.append(f'✅ RSI: Strong ({rsi:.0f}) [18 pts]')
    elif 35 <= rsi <= 42:
        score += 20
        criteria.append(f'✅ RSI: Oversold bounce zone ({rsi:.0f}) [20 pts]')
    elif 45 <= rsi <= 55:
        score += 12
        criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [12 pts]')
    elif rsi > 68:
        score += 5
        criteria.append(f'❌ RSI: Overbought ({rsi:.0f}) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ RSI: Weak ({rsi:.0f}) [0 pts]')
    
    # 3. MACD (20 pts) - STRICT
    if macd > 10:
        score += 20
        criteria.append(f'✅ MACD: Very Strong ({macd:.1f}) [20 pts]')
    elif macd > 5:
        score += 15
        criteria.append(f'✅ MACD: Strong ({macd:.1f}) [15 pts]')
    elif macd > 0:
        score += 10
        criteria.append(f'⚠ MACD: Slightly Bullish ({macd:.1f}) [10 pts]')
    elif macd > -3:
        score += 5
        criteria.append(f'⚠ MACD: Turning ({macd:.1f}) [5 pts]')
    else:
        score += 0
        criteria.append(f'❌ MACD: Bearish ({macd:.1f}) [0 pts]')
    
    # 4. Bollinger Bands (15 pts) - STRICT
    if 15 <= bb <= 30:
        score += 15
        criteria.append(f'✅ BB: Lower band - Oversold ({bb:.0f}%) [15 pts]')
    elif 60 <= bb <= 75:
        score += 12
        criteria.append(f'✅ BB: Upper band - Strong ({bb:.0f}%) [12 pts]')
    elif 40 <= bb <= 55:
        score += 8
        criteria.append(f'⚠ BB: Middle ({bb:.0f}%) [8 pts]')
    else:
        score += 0
        criteria.append(f'❌ BB: Extreme ({bb:.0f}%) [0 pts]')
    
    # 5. Volume (20 pts) - VERY STRICT
    if vol >= 3.0:
        score += 20
        criteria.append(f'✅ Volume: Massive ({vol:.1f}x) [20 pts]')
    elif vol >= 2.5:
        score += 15
        criteria.append(f'✅ Volume: Very High ({vol:.1f}x) [15 pts]')
    elif vol >= 2.0:
        score += 12
        criteria.append(f'✅ Volume: High ({vol:.1f}x) [12 pts]')
    elif vol >= 1.5:
        score += 8
        criteria.append(f'⚠ Volume: Above Average ({vol:.1f}x) [8 pts]')
    else:
        score += 0
        criteria.append(f'❌ Volume: Low ({vol:.1f}x) [0 pts]')
    
    # 6. Trend (15 pts) - STRICT
    if trend == 'Strong Uptrend':
        score += 15
        criteria.append(f'✅ Trend: Strong Uptrend [15 pts]')
    elif trend == 'Uptrend':
        score += 10
        criteria.append(f'✅ Trend: Uptrend [10 pts]')
    elif trend == 'Sideways':
        score += 3
        criteria.append(f'⚠ Trend: Sideways [3 pts]')
    else:
        score += 0
        criteria.append(f'❌ Trend: Downtrend [0 pts]')
    
    # 7. Daily Change (10 pts) - STRICT  
    if change >= 5:
        score += 10
        criteria.append(f'✅ Daily: Exceptional ({change:+.1f}%) [10 pts]')
    elif change >= 3:
        score += 8
        criteria.append(f'✅ Daily: Strong ({change:+.1f}%) [8 pts]')
    elif change >= 2:
        score += 5
        criteria.append(f'✅ Daily: Good ({change:+.1f}%) [5 pts]')
    elif change >= 0:
        score += 2
        criteria.append(f'⚠ Daily: Slight gain ({change:+.1f}%) [2 pts]')
    else:
        score += 0
        criteria.append(f'❌ Daily: Negative ({change:+.1f}%) [0 pts]')
    
    # Rating based on STRICT thresholds
    if score >= 90:
        status = '🌟 EXCELLENT'
        rating = 'Excellent'
    elif score >= 80:
        status = '💎 VERY GOOD'
        rating = 'Very Good'
    elif score >= 70:
        status = '✅ GOOD'
        rating = 'Good'
    elif score >= 65:
        status = '👍 FAIR'
        rating = 'Fair'
    elif score >= 55:
        status = '⚠ WATCHLIST'
        rating = 'Watchlist'
    else:
        status = '❌ POOR'
        rating = 'Poor'
    
    qualified = score >= 70  # MUCH STRICTER - was 60
    met_count = len([c for c in criteria if '✅' in c])
    
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
        'score': score,
        'qualified': qualified,
        'status': status,
        'rating': rating,
        'criteria': criteria,
        'met_count': met_count,
        'sector': SECTOR_MAP.get(data['symbol'], 'Other')
    }

# Main App
st.markdown('<p class="main-header">🎯 Indian Stock Scout - STRICT CRITERIA Scanner</p>', unsafe_allow_html=True)
st.markdown("*VERY HARD qualification criteria - Only the best stocks qualify*")

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
st.sidebar.subheader("🎯 STRICT Criteria Overview")
st.sidebar.info("""
*Much Harder Standards:*
- Qualified: Score ≥70 (was 60)
- Excellent: ≥90
- Very Good: 80-89
- Good: 70-79
- Fair: 65-69
- Watchlist: 55-64
- Poor: <55

*Tougher Requirements:*
- RSI: 58-65 (perfect)
- Volume: ≥2.5x (high)
- MACD: >5 (strong)
- Potential: ≥₹25 + ≥6%
""")

st.sidebar.markdown("---")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

if st.sidebar.button("🚀 START STRICT SCAN", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Scanning with STRICT Criteria...")
    
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
            stats_placeholder.info(f"✅ Analyzed: {len(results)} | Qualified (≥70): {qualified_count} | Failed: {failed}")
        
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
    st.subheader(f"📈 Scan Results - STRICT CRITERIA")
    st.caption(f"Scanned at: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Symbol': r['symbol'],
        'Price (₹)': r['price'],
        'Change (%)': r['change'],
        'Potential (₹)': r['potential_rs'],
        'Potential (%)': r['potential_pct'],
        'RSI': r['rsi'],
        'MACD': 'Bullish' if r['macd'] > 0 else 'Bearish',
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
    
    excellent = df[df['Score'] >= 90]
    very_good = df[(df['Score'] >= 80) & (df['Score'] < 90)]
    good = df[(df['Score'] >= 70) & (df['Score'] < 80)]
    fair = df[(df['Score'] >= 65) & (df['Score'] < 70)]
    watchlist = df[(df['Score'] >= 55) & (df['Score'] < 65)]
    poor = df[df['Score'] < 55]
    
    col1.metric("Total", len(df))
    col2.metric("🌟 Excellent (≥90)", len(excellent))
    col3.metric("💎 Very Good (80-89)", len(very_good))
    col4.metric("✅ Good (70-79)", len(good))
    col5.metric("👍 Fair (65-69)", len(fair))
    col6.metric("⚠ Watchlist (55-64)", len(watchlist))
    
    st.markdown("---")
    
    # Filtering
    st.subheader("🔍 Filter Results")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        rating_filter = st.selectbox("Rating", 
            ["All", "Excellent", "Very Good", "Good", "Fair", "Watchlist", "Poor"])
    
    with filter_col2:
        sector_filter = st.selectbox("Sector", 
            ["All"] + sorted(df['Sector'].unique().tolist()))
    
    with filter_col3:
        trend_filter = st.selectbox("Trend", 
            ["All", "Strong Uptrend", "Uptrend", "Sideways", "Downtrend"])
    
    with filter_col4:
        min_score_filter = st.number_input("Min Score", 0, 100, 0, 5)
    
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
        if row['Score'] >= 90:
            return ['background-color: #d4edda; font-weight: bold'] * len(row)
        elif row['Score'] >= 80:
            return ['background-color: #cfe2ff'] * len(row)
        elif row['Score'] >= 70:
            return ['background-color: #d1ecf1'] * len(row)
        elif row['Score'] >= 65:
            return ['background-color: #fff3cd'] * len(row)
        elif row['Score'] >= 55:
            return ['background-color: #f8d7da'] * len(row)
        else:
            return ['background-color: #f5c6cb'] * len(row)
    
    styled_df = filtered_df.style.apply(highlight_rating, axis=1)\
        .format({
            'Price (₹)': '₹{:.2f}',
            'Change (%)': '{:+.2f}%',
            'Potential (₹)': '₹{:.2f}',
            'Potential (%)': '{:.2f}%',
            'RSI': '{:.1f}',
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
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Score", selected_result['score'])
            col2.metric("Price", f"₹{selected_result['price']:.2f}")
            col3.metric("Change", f"{selected_result['change']:+.2f}%")
            col4.metric("Potential", f"₹{selected_result['potential_rs']:.2f}")
            col5.metric("Rating", selected_result['rating'])
            
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
            f"strict_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        all_csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Results CSV",
            all_csv,
            f"strict_scan_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )

else:
    st.info("👈 Configure and click 'START STRICT SCAN' to begin")
    
    st.markdown("---")
    st.subheader("🎯 STRICT Criteria Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        *MUCH HARDER Standards:*
        
        1. *Price Potential (20 pts)*
           - Excellent: ≥₹30 AND ≥7%
           - Very Good: ≥₹25 AND ≥6%
           - Good: ≥₹20 AND ≥5%
        
        2. *RSI (25 pts)*
           - Perfect: 58-65 (sweet spot)
           - Strong: 52-68
           - Oversold: 35-42 (bounce zone)
        
        3. *MACD (20 pts)*
           - Very Strong: >10
           - Strong: >5
           - Slightly Bullish: >0
        
        4. *Bollinger Bands (15 pts)*
           - Lower: 15-30% (oversold)
           - Upper: 60-75% (momentum)
        """)
    
    with col2:
        st.markdown("""
        5. *Volume (20 pts)*
           - Massive: ≥3.0x
           - Very High: ≥2.5x
           - High: ≥2.0x
           - Above Avg: ≥1.5x
        
        6. *Trend (15 pts)*
           - Strong Uptrend: 4/5 up days
           - Uptrend: 3/5 up days
        
        7. *Daily Change (10 pts)*
           - Exceptional: ≥5%
           - Strong: ≥3%
           - Good: ≥2%
        
        *Qualification:*
        - EXCELLENT: ≥90 points
        - VERY GOOD: 80-89 points
        - GOOD: 70-79 points (QUALIFIED)
        - FAIR: 65-69 points
        - WATCHLIST: 55-64 points
        - POOR: <55 points
        """)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;'>
<p><strong>Indian Stock Scout - STRICT MODE</strong> | Only top performers qualify</p>
<p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

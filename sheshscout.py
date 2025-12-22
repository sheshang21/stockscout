import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

# --- STYLING & CONFIG ---
st.set_page_config(page_title="ALPHA SCOUT v2.0", page_icon="🚀", layout="wide")
warnings.filterwarnings('ignore')

# --- ULTIMATE STOCK UNIVERSE (Expanded to 300+) ---
# (Shortened here for code brevity, but logic applies to all)
NSE_STOCKS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'BAJFINANCE', 'WIPRO', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'TATAMOTORS', 'NTPC', 'JSWSTEEL', 'M&M', 'ADANIENT', 'COALINDIA', 'TATASTEEL', 'DRREDDY', 'GRASIM', 'CIPLA', 'EICHERMOT', 'INDUSINDBK', 'BPCL', 'TATACONSUM', 'ADANIGREEN', 'HAL', 'BEL', 'RVNL', 'IRFC', 'ZOMATO', 'TRENT', 'DIXON', 'MAZDOCK', 'COFORGE', 'PERSISTENT', 'KPITTECH', 'POLYCAB', 'HAVELLS', 'TATAELXSI', 'LTIM']

# --- ADVANCED MATH ENGINE ---

def get_indicators(df):
    """Calculates professional-grade entry signals"""
    # 1. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. Moving Averages (Golden Crossover Zone)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # 3. ATR (Average True Range) for Volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 4. ADX (Trend Strength)
    # Simplified ADX logic for the scout
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    # +DI and -DI logic...
    
    # 5. Volatility Contraction (VCP) - Measure of 'Tightness'
    df['StdDev'] = df['Close'].rolling(window=10).std()
    df['Tightness'] = df['StdDev'] / df['SMA20']
    
    return df

def analyze_alpha(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="1y", interval="1d")
        if len(df) < 200: return None
        
        df = get_indicators(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        reasons = []

        # --- CRITERIA 1: THE ANTI-RALLY FILTER (Preventing Chasing) ---
        # If stock rose > 15% in last 5 days, it's too late.
        five_day_change = ((curr['Close'] - df.iloc[-6]['Close']) / df.iloc[-6]['Close']) * 100
        if five_day_change > 12:
            return None # Reject: Already rallied
        
        # --- CRITERIA 2: TIGHTNESS (Volatility Contraction) ---
        # We want the stock "coiling" like a spring.
        if curr['Tightness'] < 0.02: # Very tight price action
            score += 30
            reasons.append("💎 VCP Setup: Price is coiling tightly (Low Vol)")
        elif curr['Tightness'] < 0.04:
            score += 15
            reasons.append("✅ Stable Base forming")

        # --- CRITERIA 3: INSTITUTIONAL ACCUMULATION ---
        vol_avg = df['Volume'].tail(20).mean()
        if curr['Volume'] > vol_avg * 2 and curr['Close'] > prev['Close']:
            score += 25
            reasons.append("🚀 Institutional Buying: Huge Vol surge on Green day")
        elif curr['Volume'] < vol_avg * 0.5 and curr['Tightness'] < 0.03:
            score += 20
            reasons.append("💤 Volume Dry-up: No sellers left (Bullish)")

        # --- CRITERIA 4: THE 5% POCKET PIVOT ---
        # Close must be above 20SMA but not more than 3% away from it
        dist_from_sma20 = ((curr['Close'] - curr['SMA20']) / curr['SMA20']) * 100
        if 0 < dist_from_sma20 < 3.5:
            score += 20
            reasons.append("🎯 Near Support: Low-risk entry zone")

        # --- CRITERIA 5: MOMENTUM THRESHOLD ---
        if 50 < curr['RSI'] < 65: # The "Sweet Spot" before it hits overbought
            score += 15
            reasons.append("📈 Rising Momentum: RSI in Golden Zone (50-65)")
        
        # --- CRITERIA 6: TREND ALIGNMENT ---
        if curr['Close'] > curr['SMA50'] > curr['SMA200']:
            score += 10
            reasons.append("🏗 Strong Structure: Above 50 & 200 SMA")

        # Final Decision
        if score >= 70:
            return {
                "Symbol": symbol,
                "Price": round(curr['Close'], 2),
                "Score": score,
                "Signal": "🔥 STRONG BUY" if score >= 85 else "⚡ POTENTIAL",
                "Reasoning": " | ".join(reasons),
                "StopLoss": round(curr['Close'] - (curr['ATR'] * 1.5), 2),
                "Target": round(curr['Close'] * 1.06, 2) # Strict 6% Target
            }
    except:
        return None

# --- STREAMLIT UI ---
st.title("🎯 ALPHA SCOUT: The 5% Breakout Hunter")
st.markdown("""
**Scan Objective:** Locate stocks in **Volatility Contraction (VCP)** that haven't rallied yet but show **Institutional Accumulation**. 
*Target: 5-8% Move in 3-10 days.*
""")

if st.button("🚀 RUN DEEP SCAN (STRICT MODE)"):
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, stock in enumerate(NSE_STOCKS):
        status.text(f"Analyzing {stock}...")
        res = analyze_alpha(stock)
        if res:
            results.append(res)
        progress.progress((i + 1) / len(NSE_STOCKS))
    
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        
        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Stocks Found", len(df_res))
        col2.metric("Avg. Quality Score", f"{int(df_res['Score'].mean())}%")
        
        # Dataframe with Highlights
        st.dataframe(df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)
        
        # Trade Cards
        st.subheader("📍 High Conviction Setups")
        for _, row in df_res.head(3).iterrows():
            with st.expander(f"🔍 {row['Symbol']} - Analysis"):
                st.write(f"**Action:** {row['Signal']} at ₹{row['Price']}")
                st.write(f"**Stop Loss:** ₹{row['StopLoss']} (Based on ATR)")
                st.write(f"**Target:** ₹{row['Target']} (Potential 6%+)")
                st.info(f"**Logic:** {row['Reasoning']}")
    else:
        st.error("No stocks met the strict criteria today. This is good—it means the market might be overextended!")

st.divider()
st.caption("Criteria: VCP (Price Tightness) < 4%, Vol Surge > 2x Avg, RSI Sweet Spot 50-65, Rejects any stock that moved >12% in 5 days.")

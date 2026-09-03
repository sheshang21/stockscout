"""
mode_positional.py — Positional / Ultra-Strict long-term value scanner.

This is the CORE LOGIC that makes this mode distinct from the two intraday
modes: fetch_stock_data() (3-month daily history + annual/quarterly
financials) and analyze_stock() (the 14-criterion, 250-point fundamentals +
technicals scoring system). Everything else — exchange/scan-mode selection,
rate limiting, checkpointing, the results table shell, filters, CSV export —
comes from scanner_common so all 3 modes share it exactly.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import bhavcopy
import indicators
import scanner_common as sc
from scanner_common import sskey, get_state, set_state, yf

MODE_KEY = sc.MODE_POSITIONAL

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'BAJFINANCE': 'NBFC', 'WIPRO': 'IT', 'SUNPHARMA': 'Pharma', 'TITAN': 'Consumer', 'ULTRACEMCO': 'Cement',
    'NESTLEIND': 'FMCG', 'ONGC': 'Energy', 'TATAMOTORS': 'Auto', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'TECHM': 'IT', 'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure',
}

_CACHE_TTL_S = 300  # fundamentals barely move intraday; long TTL is fine and keeps repeat scans fast


# ── CORE LOGIC: fetch ────────────────────────────────────────────────────
def fetch_stock_data(yf_symbol: str):
    """History for technicals comes from NSE/BSE's own bhavcopy first (see
    bhavcopy.py) — it's free EOD data with no Yahoo rate limit attached, so
    using it removes this call from Yahoo's load entirely instead of just
    retrying it more politely. Falls back to yfinance's ticker.history() if
    bhavcopy has no usable data for this symbol (unrecognized exchange
    suffix, a fetch failure, or too little history). Financials/balance
    sheet still come from yfinance — bhavcopy has no such data. ticker.info
    is intentionally never called — it's the most throttled Yahoo endpoint
    and everything here is derivable from the financial statements +
    fast_info instead."""
    if sc.is_known_dead(yf_symbol):
        return None

    cache_key = f"{MODE_KEY}:{yf_symbol}"
    cached = sc.cache_get(cache_key, _CACHE_TTL_S)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(yf_symbol)

        bhav = bhavcopy.get_daily_series(yf_symbol, trading_days=65)
        if bhav is not None:
            closes = bhav['close'].values
            highs = bhav['high'].values
            lows = bhav['low'].values
            volumes = bhav['volume'].values
        else:
            hist = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                sc.mark_dead_symbol(yf_symbol)
                return None
            closes = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            volumes = hist['Volume'].values

        price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else price
        change = ((price - prev_close) / prev_close) * 100

        fi = ticker.fast_info
        market_cap = getattr(fi, 'market_cap', None) or 0

        annual_inc = None
        try:
            annual_inc = ticker.income_stmt if hasattr(ticker, 'income_stmt') else ticker.financials
        except Exception:
            pass

        annual_bs = None
        try:
            annual_bs = ticker.balance_sheet
        except Exception:
            pass

        q_inc = None
        try:
            q_inc = ticker.quarterly_income_stmt if hasattr(ticker, 'quarterly_income_stmt') else ticker.quarterly_financials
        except Exception:
            pass

        latest_fy_revenue = 0
        if annual_inc is not None and not annual_inc.empty and 'Total Revenue' in annual_inc.index:
            v = annual_inc.loc['Total Revenue'].iloc[0]
            latest_fy_revenue = 0 if pd.isna(v) else v

        total_cash = 0
        if annual_bs is not None and not annual_bs.empty:
            for cash_key in ('Cash And Cash Equivalents',
                              'Cash Cash Equivalents And Short Term Investments',
                              'Cash And Short Term Investments'):
                if cash_key in annual_bs.index:
                    v = annual_bs.loc[cash_key].iloc[0]
                    total_cash = 0 if pd.isna(v) else v
                    break

        profit_margin = None
        if annual_inc is not None and not annual_inc.empty:
            try:
                rev = annual_inc.loc['Total Revenue'].iloc[0] if 'Total Revenue' in annual_inc.index else None
                net = annual_inc.loc['Net Income'].iloc[0] if 'Net Income' in annual_inc.index else None
                if rev and net and not pd.isna(rev) and not pd.isna(net) and rev != 0:
                    profit_margin = net / rev
            except Exception:
                pass

        pe_ratio = getattr(fi, 'p_e_ratio', None)

        qoq_revenue_growth = yoy_revenue_growth = None
        qoq_profit_growth = yoy_profit_growth = None

        if q_inc is not None and not q_inc.empty:
            if 'Total Revenue' in q_inc.index:
                revenues = [r for r in q_inc.loc['Total Revenue'].values if not pd.isna(r)]
                if len(revenues) >= 2:
                    qoq_revenue_growth = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
                if len(revenues) >= 4:
                    yoy_revenue_growth = ((revenues[0] - revenues[3]) / abs(revenues[3])) * 100 if revenues[3] != 0 else None
            if 'Net Income' in q_inc.index:
                profits = [p for p in q_inc.loc['Net Income'].values if not pd.isna(p)]
                if len(profits) >= 2:
                    qoq_profit_growth = ((profits[0] - profits[1]) / abs(profits[1])) * 100 if profits[1] != 0 else None
                if len(profits) >= 4:
                    yoy_profit_growth = ((profits[0] - profits[3]) / abs(profits[3])) * 100 if profits[3] != 0 else None

        cash_on_hand_to_mcap = (total_cash / market_cap * 100) if market_cap > 0 and total_cash > 0 else 0
        latest_fy_revenue_to_mcap = (latest_fy_revenue / market_cap) if market_cap > 0 and latest_fy_revenue > 0 else 0

        historical_data = get_historical_financials_from_data(annual_inc, annual_bs, market_cap)

        fii_dii_activity = indicators.detect_institutional_activity(volumes, closes)
        rsi = indicators.rsi(closes)
        macd = indicators.macd(closes)
        bb_position = indicators.bollinger_position(closes)
        vol_multiple = indicators.volume_multiple(volumes)
        trend = indicators.detect_trend(closes)

        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 5 and closes[0] != 0 else 0

        result = {
            'symbol': yf_symbol, 'price': price, 'change': change,
            'weekly_change': weekly_change, 'monthly_change': monthly_change,
            'three_month_change': three_month_change, 'rsi': rsi, 'macd': macd,
            'bb_position': bb_position, 'vol_multiple': vol_multiple, 'trend': trend,
            'closes': closes, 'highs': highs, 'lows': lows, 'volumes': volumes,
            'fii_dii_score': fii_dii_activity, 'market_cap': market_cap,
            'profit_margin': profit_margin, 'pe_ratio': pe_ratio,
            'total_cash': total_cash, 'latest_fy_revenue': latest_fy_revenue,
            'cash_on_hand_to_mcap': cash_on_hand_to_mcap,
            'latest_fy_revenue_to_mcap': latest_fy_revenue_to_mcap,
            'historical_data': historical_data,
            'qoq_revenue_growth': qoq_revenue_growth, 'yoy_revenue_growth': yoy_revenue_growth,
            'qoq_profit_growth': qoq_profit_growth, 'yoy_profit_growth': yoy_profit_growth,
        }
        sc.cache_set(cache_key, result)
        return result

    except Exception as e:
        if any(kw in str(e).lower() for kw in ("delisted", "not found", "no data found")):
            sc.mark_dead_symbol(yf_symbol)
        return None


def get_historical_financials_from_data(annual_inc, annual_bs, current_mcap):
    historical = {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []}
    try:
        if annual_inc is None or annual_inc.empty:
            return historical
        years = list(annual_inc.columns[:3]) if len(annual_inc.columns) >= 3 else list(annual_inc.columns)
        for year in years:
            year_str = year.strftime('%Y') if hasattr(year, 'strftime') else str(year)
            historical['years'].append(year_str)
            if 'Total Revenue' in annual_inc.index:
                v = annual_inc.loc['Total Revenue', year]
                historical['revenues'].append(0 if pd.isna(v) else v)
            else:
                historical['revenues'].append(0)
            cash = 0
            if annual_bs is not None and not annual_bs.empty and year in annual_bs.columns:
                for cash_key in ('Cash And Cash Equivalents',
                                  'Cash Cash Equivalents And Short Term Investments',
                                  'Cash And Short Term Investments'):
                    if cash_key in annual_bs.index:
                        v = annual_bs.loc[cash_key, year]
                        cash = 0 if pd.isna(v) else v
                        break
            historical['cash_amounts'].append(cash)
        for revenue in historical['revenues']:
            historical['sales_to_mcap'].append(revenue / current_mcap if current_mcap > 0 and revenue > 0 else 0)
    except Exception:
        pass
    return historical


def fetch_live_price(yf_symbol: str):
    try:
        data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception:
        return None


# ── CORE LOGIC: score ───────────────────────────────────────────────────
def analyze_stock(data, min_market_cap, thresholds):
    try:
        if not data:
            return None

        price = data['price']; change = data['change']; rsi = data['rsi']; macd = data['macd']
        bb = data['bb_position']; vol = data['vol_multiple']; trend = data['trend']; closes = data['closes']

        market_cap = data['market_cap'] / 10000000 if data['market_cap'] else 0
        if market_cap < min_market_cap:
            return None

        is_operated, operator_flags, operator_risk = indicators.detect_operator_activity(closes, data['volumes'])

        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 5 and closes[0] != 0 else 0

        potential_rs = max(20, price * 0.10)
        potential_pct = (potential_rs / price) * 100 if price != 0 else 0

        score = 0
        criteria = []

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
            criteria.append(f'⚠️ MODERATE RISK: Some volatility flags (Risk: {operator_risk}/100) [-12 pts]')

        # 1. MARKET CAP (15 pts)
        if market_cap >= 20000:
            score += 15
            criteria.append(f'✅ Market Cap: Large Cap (₹{market_cap:.0f} Cr) [15 pts]')
        elif market_cap >= 5000:
            score += 7
            criteria.append(f'⚠ Market Cap: Small-Mid Cap (₹{market_cap:.0f} Cr) [7 pts]')
        else:
            criteria.append(f'❌ Market Cap: Small Cap (₹{market_cap:.0f} Cr) [0 pts]')

        # 2. REVENUE GROWTH (25 pts)
        yoy_rev = data['yoy_revenue_growth']
        qoq_rev = data['qoq_revenue_growth']
        if yoy_rev is not None and qoq_rev is not None:
            if yoy_rev >= 25 and qoq_rev >= 15:
                score += 25; criteria.append(f'✅ Revenue: EXCEPTIONAL Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [25 pts]')
            elif yoy_rev >= 20 and qoq_rev >= 10:
                score += 22; criteria.append(f'✅ Revenue: Excellent Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [22 pts]')
            elif yoy_rev >= 15 and qoq_rev >= 8:
                score += 18; criteria.append(f'✅ Revenue: Strong Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [18 pts]')
            elif yoy_rev >= 10 and qoq_rev >= 5:
                score += 12; criteria.append(f'⚠ Revenue: Good Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [12 pts]')
            elif yoy_rev >= 5:
                score += 5; criteria.append(f'⚠ Revenue: Moderate Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Revenue: Weak/Negative Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [0 pts]')
        elif yoy_rev is not None:
            if yoy_rev >= 20:
                score += 20; criteria.append(f'✅ Revenue: Strong YoY Growth ({yoy_rev:.1f}%) [20 pts]')
            elif yoy_rev >= 12:
                score += 15; criteria.append(f'✅ Revenue: Good YoY Growth ({yoy_rev:.1f}%) [15 pts]')
            elif yoy_rev >= 5:
                score += 8; criteria.append(f'⚠ Revenue: Moderate Growth ({yoy_rev:.1f}%) [8 pts]')
            else:
                criteria.append(f'❌ Revenue: Weak Growth ({yoy_rev:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Revenue: Data not available [0 pts]')

        # 3. PROFIT GROWTH (25 pts)
        yoy_profit = data['yoy_profit_growth']
        qoq_profit = data['qoq_profit_growth']
        profit_margin = data['profit_margin']
        if yoy_profit is not None and qoq_profit is not None:
            if yoy_profit >= 30 and qoq_profit >= 20:
                score += 25; criteria.append(f'✅ Profit: EXCEPTIONAL Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [25 pts]')
            elif yoy_profit >= 25 and qoq_profit >= 15:
                score += 22; criteria.append(f'✅ Profit: Excellent Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [22 pts]')
            elif yoy_profit >= 20 and qoq_profit >= 10:
                score += 18; criteria.append(f'✅ Profit: Strong Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [18 pts]')
            elif yoy_profit >= 12 and qoq_profit >= 6:
                score += 12; criteria.append(f'⚠ Profit: Good Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [12 pts]')
            elif yoy_profit >= 5:
                score += 5; criteria.append(f'⚠ Profit: Moderate Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Profit: Weak/Negative Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [0 pts]')
        elif yoy_profit is not None:
            if yoy_profit >= 25:
                score += 20; criteria.append(f'✅ Profit: Strong YoY Growth ({yoy_profit:.1f}%) [20 pts]')
            elif yoy_profit >= 15:
                score += 15; criteria.append(f'✅ Profit: Good YoY Growth ({yoy_profit:.1f}%) [15 pts]')
            elif yoy_profit >= 8:
                score += 8; criteria.append(f'⚠ Profit: Moderate Growth ({yoy_profit:.1f}%) [8 pts]')
            else:
                criteria.append(f'❌ Profit: Weak Growth ({yoy_profit:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Profit: Data not available [0 pts]')

        # 4. PROFIT MARGIN (15 pts)
        if profit_margin is not None:
            pm = profit_margin * 100
            if pm >= 20:
                score += 15; criteria.append(f'✅ Profit Margin: Excellent ({pm:.1f}%) [15 pts]')
            elif pm >= 15:
                score += 12; criteria.append(f'✅ Profit Margin: Very Good ({pm:.1f}%) [12 pts]')
            elif pm >= 10:
                score += 10; criteria.append(f'✅ Profit Margin: Good ({pm:.1f}%) [10 pts]')
            elif pm >= 5:
                score += 5; criteria.append(f'⚠ Profit Margin: Average ({pm:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Profit Margin: Low ({pm:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Profit Margin: Data not available [0 pts]')

        # 5. FII/DII ACTIVITY (20 pts)
        fii_score = data['fii_dii_score']
        if fii_score >= 15:
            score += 20; criteria.append(f'✅ FII/DII: Strong Buying ({fii_score}) [20 pts]')
        elif fii_score >= 10:
            score += 15; criteria.append(f'✅ FII/DII: Good Buying ({fii_score}) [15 pts]')
        elif fii_score >= 5:
            score += 10; criteria.append(f'✅ FII/DII: Accumulation ({fii_score}) [10 pts]')
        elif fii_score >= 0:
            score += 5; criteria.append(f'⚠ FII/DII: Neutral ({fii_score}) [5 pts]')
        else:
            criteria.append(f'❌ FII/DII: Selling ({fii_score}) [0 pts]')

        # 6. CONSOLIDATION (20 pts)
        if -2 <= weekly_change <= 0.3:
            score += 20; criteria.append(f'✅ Consolidation: Perfect base ({weekly_change:+.1f}% weekly) [20 pts]')
        elif -3.5 <= weekly_change < -2:
            score += 18; criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [18 pts]')
        elif 0.3 < weekly_change <= 1.5:
            score += 15; criteria.append(f'✅ Consolidation: Early breakout ({weekly_change:+.1f}% weekly) [15 pts]')
        elif weekly_change > 4:
            criteria.append(f'❌ Already rallied ({weekly_change:+.1f}% weekly) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')

        # 7. RSI (20 pts)
        rsi_low = thresholds['rsi_low']; rsi_high = thresholds['rsi_high']
        if rsi_low <= rsi <= rsi_high:
            score += 20; criteria.append(f'✅ RSI: Perfect oversold entry ({rsi:.0f}) [20 pts]')
        elif rsi_high < rsi <= rsi_high + 7:
            score += 17; criteria.append(f'✅ RSI: Building momentum ({rsi:.0f}) [17 pts]')
        elif rsi_high + 7 < rsi <= rsi_high + 12:
            score += 12; criteria.append(f'✅ RSI: Early momentum ({rsi:.0f}) [12 pts]')
        elif rsi_high + 12 < rsi <= rsi_high + 17:
            score += 8; criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [8 pts]')
        elif rsi > rsi_high + 24:
            criteria.append(f'❌ RSI: Overbought ({rsi:.0f}) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ RSI: Moderate ({rsi:.0f}) [5 pts]')

        # 8. MACD (15 pts)
        if -1 <= macd <= 1:
            score += 15; criteria.append(f'✅ MACD: Perfect crossover ({macd:.1f}) [15 pts]')
        elif 1 < macd <= 3:
            score += 12; criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [12 pts]')
        elif -3 <= macd < -1:
            score += 10; criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [10 pts]')
        elif macd > 6:
            criteria.append(f'❌ MACD: Extended ({macd:.1f}) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')

        # 9. BOLLINGER BANDS (15 pts)
        if 8 <= bb <= 20:
            score += 15; criteria.append(f'✅ BB: Lower band bounce ({bb:.0f}%) [15 pts]')
        elif 20 < bb <= 30:
            score += 12; criteria.append(f'✅ BB: Below middle ({bb:.0f}%) [12 pts]')
        elif 30 < bb <= 45:
            score += 8; criteria.append(f'⚠ BB: Middle zone ({bb:.0f}%) [8 pts]')
        elif bb > 65:
            criteria.append(f'❌ BB: Upper band ({bb:.0f}%) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ BB: Neutral ({bb:.0f}%) [5 pts]')

        # 10. VOLUME (15 pts)
        if 1.3 <= vol <= 1.8:
            score += 15; criteria.append(f'✅ Volume: Perfect accumulation ({vol:.1f}x) [15 pts]')
        elif 1.8 < vol <= 2.2:
            score += 12; criteria.append(f'✅ Volume: Building interest ({vol:.1f}x) [12 pts]')
        elif vol > 2.8:
            score += 5; criteria.append(f'⚠ Volume: Too high ({vol:.1f}x) [5 pts]')
        elif 1.0 <= vol < 1.3:
            score += 7; criteria.append(f'⚠ Volume: Average ({vol:.1f}x) [7 pts]')
        else:
            criteria.append(f'❌ Volume: Too low ({vol:.1f}x) [0 pts]')

        # 11. TODAY'S PRICE (10 pts)
        if -1.5 <= change <= 0.3:
            score += 10; criteria.append(f"✅ Today: Perfect entry ({change:+.1f}%) [10 pts]")
        elif 0.3 < change <= 1.2:
            score += 8; criteria.append(f"✅ Today: Early move ({change:+.1f}%) [8 pts]")
        elif -2.5 <= change < -1.5:
            score += 7; criteria.append(f"⚠ Today: Dip ({change:+.1f}%) [7 pts]")
        elif change > 2.5:
            criteria.append(f"❌ Today: Already rallied ({change:+.1f}%) [0 pts]")
        else:
            score += 4; criteria.append(f"⚠ Today: Moderate ({change:+.1f}%) [4 pts]")

        # 12. MONTHLY TREND (10 pts)
        if -8 <= monthly_change <= -2:
            score += 10; criteria.append(f'✅ Monthly: Recovering from dip ({monthly_change:+.1f}%) [10 pts]')
        elif -2 < monthly_change <= 2:
            score += 8; criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [8 pts]')
        elif 2 < monthly_change <= 6:
            score += 5; criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [5 pts]')
        elif monthly_change > 10:
            criteria.append(f'❌ Monthly: Extended ({monthly_change:+.1f}%) [0 pts]')
        else:
            score += 3; criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [3 pts]')

        # 13. 3-MONTH PERFORMANCE (10 pts)
        if -15 <= three_month_change <= -5:
            score += 10; criteria.append(f'✅ 3-Month: Perfect correction ({three_month_change:+.1f}%) [10 pts]')
        elif -5 < three_month_change <= 5:
            score += 8; criteria.append(f'✅ 3-Month: Sideways base ({three_month_change:+.1f}%) [8 pts]')
        elif 5 < three_month_change <= 15:
            score += 5; criteria.append(f'⚠ 3-Month: Moderate rise ({three_month_change:+.1f}%) [5 pts]')
        elif three_month_change > 25:
            criteria.append(f'❌ 3-Month: Overextended ({three_month_change:+.1f}%) [0 pts]')
        else:
            score += 3; criteria.append(f'⚠ 3-Month: Weak ({three_month_change:+.1f}%) [3 pts]')

        # 14. UPSIDE POTENTIAL (10 pts)
        if potential_pct >= 12:
            score += 10; criteria.append(f'✅ Upside: Excellent ({potential_pct:.1f}%) [10 pts]')
        elif potential_pct >= 10:
            score += 8; criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
        elif potential_pct >= 8:
            score += 5; criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
        else:
            criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')

        threshold_exceptional = thresholds['threshold_exceptional']
        threshold_prime = thresholds['threshold_prime']
        threshold_excellent = thresholds['threshold_excellent']
        threshold_strong = thresholds['threshold_strong']

        if is_operated:
            status = '🚨 OPERATED - AVOID'; rating = 'Operated - Avoid'
        elif score >= threshold_exceptional:
            status = '🌟 EXCEPTIONAL BUY'; rating = 'Exceptional Buy'
        elif score >= threshold_prime:
            status = '🚀 PRIME BUY'; rating = 'Prime Buy'
        elif score >= threshold_excellent:
            status = '💎 EXCELLENT BUY'; rating = 'Excellent Buy'
        elif score >= threshold_strong:
            status = '✅ STRONG BUY'; rating = 'Strong Buy'
        elif score >= 100:
            status = '👍 GOOD BUY'; rating = 'Good Buy'
        elif score >= 80:
            status = '📋 WATCHLIST'; rating = 'Watchlist'
        else:
            status = '❌ SKIP'; rating = 'Skip'

        qualified = score >= threshold_excellent and not is_operated
        bare_symbol = data['symbol'].replace('.NS', '').replace('.BO', '')

        return {
            'symbol': data['symbol'], 'price': price, 'change': change,
            'weekly_change': weekly_change, 'monthly_change': monthly_change,
            'three_month_change': three_month_change, 'potential_rs': potential_rs,
            'potential_pct': potential_pct, 'rsi': rsi, 'macd': macd, 'bb': bb, 'vol': vol,
            'trend': trend, 'score': score, 'qualified': qualified, 'status': status,
            'rating': rating, 'criteria': criteria,
            'met_count': len([c for c in criteria if '✅' in c]),
            'sector': SECTOR_MAP.get(bare_symbol, 'Other'),
            'is_operated': is_operated, 'operator_risk': operator_risk, 'operator_flags': operator_flags,
            'market_cap': market_cap, 'yoy_revenue_growth': yoy_rev, 'qoq_revenue_growth': qoq_rev,
            'yoy_profit_growth': yoy_profit, 'qoq_profit_growth': qoq_profit,
            'profit_margin': profit_margin * 100 if profit_margin else None,
            'total_cash': data.get('total_cash', 0), 'latest_fy_revenue': data.get('latest_fy_revenue', 0),
            'cash_on_hand_to_mcap': data.get('cash_on_hand_to_mcap', 0),
            'latest_fy_revenue_to_mcap': data.get('latest_fy_revenue_to_mcap', 0),
            'historical_data': data.get('historical_data', {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []}),
        }
    except Exception:
        return None


# ── UI ───────────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown('<p class="main-header">🎯 Positional Scanner — NSE & BSE Ultra-Strict</p>', unsafe_allow_html=True)
    st.markdown("*Choose NSE, BSE, or BOTH | Only stocks with EXCEPTIONAL fundamentals + technicals qualify*")

    scan_nse, scan_bse, universe = sc.render_exchange_selector(MODE_KEY)
    stocks_to_scan = sc.render_scan_mode_selector(MODE_KEY, universe)
    rate_cfg = sc.render_rate_limit_controls(MODE_KEY)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Market Cap Filter")
    min_market_cap = st.sidebar.slider("Minimum Market Cap (₹ Crores)", 0, 100000, 5000, 1000,
                                        help="Filter stocks by minimum market capitalization",
                                        key=sskey(MODE_KEY, "min_mcap"))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Adjustable Scoring Thresholds")
    with st.sidebar.expander("📊 Customize Score Thresholds", expanded=False):
        st.markdown("**Qualification Scores:**")
        threshold_exceptional = st.number_input("Exceptional (≥)", 100, 250, 180, 10, key=sskey(MODE_KEY, "th_exc"))
        threshold_prime = st.number_input("Prime (≥)", 100, 250, 160, 10, key=sskey(MODE_KEY, "th_prime"))
        threshold_excellent = st.number_input("Excellent (≥)", 100, 250, 140, 10, key=sskey(MODE_KEY, "th_excel"))
        threshold_strong = st.number_input("Strong (≥)", 50, 200, 120, 10, key=sskey(MODE_KEY, "th_strong"))
        st.markdown("**Technical Thresholds:**")
        rsi_low = st.number_input("RSI Lower Bound", 20, 50, 32, 1, key=sskey(MODE_KEY, "rsi_low"))
        rsi_high = st.number_input("RSI Upper Bound", 30, 60, 38, 1, key=sskey(MODE_KEY, "rsi_high"))

    thresholds = {
        'threshold_exceptional': threshold_exceptional, 'threshold_prime': threshold_prime,
        'threshold_excellent': threshold_excellent, 'threshold_strong': threshold_strong,
        'rsi_low': rsi_low, 'rsi_high': rsi_high,
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 ULTRA-STRICT Criteria")
    st.sidebar.info("""*Only top 1-3% qualify!* **TOTAL: 250 Points**

**Fundamentals (80):** Market Cap 15 · Revenue Growth 25 · Profit Growth 25 · Profit Margin 15
**Technicals (170):** FII/DII 20 · Consolidation 20 · RSI 20 · MACD 15 · BB 15 · Volume 15 · Today 10 · Monthly 10 · 3-Month 10 · Upside 10

**Qualification:** Exceptional ≥180 · Prime 160-179 · Excellent 140-159 ✅ · Strong 120-139
**Penalties:** Operated -70 · High Risk -25 to -40
""")

    def fetch_and_analyze(rec):
        data = sc.bulletproof_fetch(fetch_stock_data, rec["yf_symbol"])
        if data is None:
            return 'failed', None
        analysis = analyze_stock(data, min_market_cap, thresholds)
        if analysis is None:
            return 'filtered', None
        analysis['name'] = rec['name']
        return 'ok', analysis

    do_scan, resume_scan, checkpoint, _sig = sc.render_scan_trigger(
        MODE_KEY, stocks_to_scan, "🚀 FIND EXCEPTIONAL STOCKS")

    if do_scan:
        sc.run_scan(MODE_KEY, stocks_to_scan, fetch_and_analyze, rate_cfg, resume_scan, checkpoint)

    _render_results(thresholds)
    sc.footer("<strong>NSE & BSE Positional Scanner with Fundamentals</strong> | Top 1-3% Only")


def _render_results(thresholds) -> None:
    results = get_state(MODE_KEY, "results")
    if not results:
        st.info("👈 Configure and click 'FIND EXCEPTIONAL STOCKS' to start")
        return

    scan_time = get_state(MODE_KEY, "timestamp")
    st.markdown("---")

    col_r1, col_r2, col_r3 = st.columns([2, 2, 6])
    with col_r1:
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=False,
                                    help="Continuously update prices every 30 seconds without resetting",
                                    key=sskey(MODE_KEY, "auto_refresh"))
    with col_r2:
        last_refresh = get_state(MODE_KEY, "last_refresh")
        if last_refresh:
            st.caption(f"📡 Updated {int((pd.Timestamp.now() - pd.Timestamp(last_refresh)).total_seconds())}s ago")
        else:
            st.caption("📡 Not refreshed yet")
    with col_r3:
        if auto_refresh:
            if st.button("⏸️ Pause Refresh", key=sskey(MODE_KEY, "pause_refresh")):
                set_state(MODE_KEY, "auto_refresh_paused", True)
                st.rerun()

    st.subheader("📈 Exceptional Stock Opportunities")
    if scan_time:
        st.caption(f"Initial scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh and not get_state(MODE_KEY, "auto_refresh_paused", False):
        last_refresh = get_state(MODE_KEY, "last_refresh")
        if last_refresh is None:
            set_state(MODE_KEY, "last_refresh", pd.Timestamp.now())
            last_refresh = get_state(MODE_KEY, "last_refresh")
        elapsed = (pd.Timestamp.now() - pd.Timestamp(last_refresh)).total_seconds()
        if elapsed >= 30:
            with st.spinner("🔄 Refreshing live prices..."):
                updated = 0
                for r in results:
                    try:
                        new_price = fetch_live_price(r['symbol'])
                        if new_price and new_price != r['price']:
                            prev = r['price']
                            r['price'] = new_price
                            r['change'] = ((new_price - prev) / prev) * 100 if prev != 0 else 0
                            updated += 1
                    except Exception:
                        pass
                set_state(MODE_KEY, "last_refresh", pd.Timestamp.now())
                if updated > 0:
                    st.toast(f"✅ Updated {updated} prices", icon="🔄")
            st.rerun()
        else:
            st.caption(f"⏱️ Next price refresh in {int(30 - elapsed)}s")

    df = pd.DataFrame([{
        'Symbol': r['symbol'], 'Name': r.get('name', ''),
        'Exchange': 'NSE' if '.NS' in r['symbol'] else 'BSE' if '.BO' in r['symbol'] else 'N/A',
        'Price (₹)': r['price'], 'Today (%)': r['change'], 'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'], '3M (%)': r['three_month_change'],
        'Market Cap (₹Cr)': r['market_cap'],
        'Cash/Hand (₹Cr)': r.get('total_cash', 0) / 10000000 if r.get('total_cash') else 0,
        'CashHand/MCap (%)': r.get('cash_on_hand_to_mcap', 0),
        'LatestFY Rev/MCap': r.get('latest_fy_revenue_to_mcap', 0),
        'Rev YoY (%)': r['yoy_revenue_growth'], 'Rev QoQ (%)': r['qoq_revenue_growth'],
        'Profit YoY (%)': r['yoy_profit_growth'], 'Profit QoQ (%)': r['qoq_profit_growth'],
        'Margin (%)': r['profit_margin'], 'RSI': r['rsi'], 'MACD': r['macd'], 'BB (%)': r['bb'],
        'Vol': f"{r['vol']:.1f}x", 'Score': r['score'], 'Rating': r['rating'], 'Status': r['status'],
        'Sector': r['sector'], 'Operated': '🚨 YES' if r['is_operated'] else '✅ Safe', 'Risk': r['operator_risk'],
    } for r in results])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    threshold_exceptional = thresholds['threshold_exceptional']
    threshold_prime = thresholds['threshold_prime']
    threshold_excellent = thresholds['threshold_excellent']
    threshold_strong = thresholds['threshold_strong']

    exceptional = df[(df['Score'] >= threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    prime = df[(df['Score'] >= threshold_prime) & (df['Score'] < threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    excellent = df[(df['Score'] >= threshold_excellent) & (df['Score'] < threshold_prime) & (df['Operated'] == '✅ Safe')]
    strong = df[(df['Score'] >= threshold_strong) & (df['Score'] < threshold_excellent) & (df['Operated'] == '✅ Safe')]
    operated_stocks = df[df['Operated'] == '🚨 YES']

    c1.metric("Total Scanned", len(df))
    c2.metric("🚨 Operated", len(operated_stocks))
    c3.metric(f"🌟 Exceptional (≥{threshold_exceptional})", len(exceptional))
    c4.metric(f"🚀 Prime ({threshold_prime}-{threshold_exceptional - 1})", len(prime))
    c5.metric(f"💎 Excellent ({threshold_excellent}-{threshold_prime - 1})", len(excellent))
    c6.metric(f"✅ Strong ({threshold_strong}-{threshold_excellent - 1})", len(strong))

    st.markdown("---")
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("📊 NSE Stocks", len(df[df['Exchange'] == 'NSE']))
    ec2.metric("📊 BSE Stocks", len(df[df['Exchange'] == 'BSE']))
    qualified_total = len(exceptional) + len(prime) + len(excellent)
    ec3.metric(f"🎯 Qualified (≥{threshold_excellent})", qualified_total)

    st.success(f"""
    **🎯 ULTRA-STRICT RESULTS:** Only **{qualified_total}** stocks qualified (Score ≥{threshold_excellent} + Safe) out of {len(df)}.
    That's the top **{(qualified_total/len(df)*100) if len(df) > 0 else 0:.1f}%** - truly exceptional opportunities with strong fundamentals!
    """)

    st.markdown("---")
    st.subheader("🔍 Filter Results")
    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        st.markdown("**📊 Rating**")
        rating_opts = ["Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip"]
        rating_filter = [r for r in rating_opts if st.checkbox(r, value=True, key=sskey(MODE_KEY, f"flt_rating_{r}"))]
    with f2:
        st.markdown("**📈 Exchange**")
        exchange_filter = []
        if st.checkbox("NSE", value=True, key=sskey(MODE_KEY, "flt_exc_nse")):
            exchange_filter.append("NSE")
        if st.checkbox("BSE", value=True, key=sskey(MODE_KEY, "flt_exc_bse")):
            exchange_filter.append("BSE")
    with f3:
        st.markdown("**🛡️ Safety**")
        safety_vals = []
        if st.checkbox("✅ Safe", value=True, key=sskey(MODE_KEY, "flt_safe")):
            safety_vals.append("✅ Safe")
        if st.checkbox("🚨 Operated", value=False, key=sskey(MODE_KEY, "flt_oper")):
            safety_vals.append("🚨 YES")
    with f4:
        st.markdown("**🏭 Sector**")
        all_sectors = sorted(df['Sector'].unique().tolist())
        sector_filter = [s for s in all_sectors if st.checkbox(s, value=True, key=sskey(MODE_KEY, f"flt_sector_{s}"))]
    with f5:
        min_score_filter = st.number_input("Min Score", 0, 250, threshold_excellent, 10,
                                            key=sskey(MODE_KEY, "flt_min_score"))

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['Rating'].isin(rating_filter)] if rating_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Exchange'].isin(exchange_filter)] if exchange_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Operated'].isin(safety_vals)] if safety_vals else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Sector'].isin(sector_filter)] if sector_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]

    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")

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
        return ['background-color: #ffebee'] * len(row)

    styled = filtered_df.style.apply(highlight_rating, axis=1).format({
        'Price (₹)': '₹{:.2f}', 'Today (%)': '{:+.2f}%', 'Weekly (%)': '{:+.2f}%',
        'Monthly (%)': '{:+.2f}%', '3M (%)': '{:+.2f}%', 'Market Cap (₹Cr)': '₹{:.0f}',
        'Cash/Hand (₹Cr)': '₹{:.0f}', 'CashHand/MCap (%)': '{:.2f}%', 'LatestFY Rev/MCap': '{:.2f}x',
        'Rev YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Rev QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Profit YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Profit QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Margin (%)': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A',
        'RSI': '{:.1f}', 'MACD': '{:.2f}', 'BB (%)': '{:.0f}%',
    })
    st.dataframe(styled, use_container_width=True, height=600)

    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")

    if len(filtered_df) > 0:
        options = filtered_df.apply(lambda r: f"{r['Symbol']} — {r['Name']}" if r['Name'] else r['Symbol'], axis=1).tolist()
        symbol_by_option = dict(zip(options, filtered_df['Symbol'].tolist()))
        selected_option = st.selectbox("Select stock for details", options, key=sskey(MODE_KEY, "detail_select"))
        selected_symbol = symbol_by_option[selected_option]
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)

        if selected_result:
            st.markdown(f"### {selected_symbol} — {selected_result.get('name', '')} · {selected_result['status']}")

            if selected_result['is_operated']:
                st.error(f"🚨 **OPERATOR DETECTED** - Risk: {selected_result['operator_risk']}/100")
                for flag in selected_result['operator_flags']:
                    st.warning(flag)

            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Score", selected_result['score'])
            d2.metric("Price", f"₹{selected_result['price']:.2f}")
            d3.metric("Market Cap", f"₹{selected_result['market_cap']:.0f}Cr")
            d4.metric("Rev YoY", f"{selected_result['yoy_revenue_growth']:+.1f}%" if selected_result['yoy_revenue_growth'] else "N/A")
            d5.metric("Profit YoY", f"{selected_result['yoy_profit_growth']:+.1f}%" if selected_result['yoy_profit_growth'] else "N/A")

            st.markdown("---")
            st.markdown("**💵 Financial Ratios**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Cash on Hand", f"₹{selected_result.get('total_cash', 0)/10000000:.0f}Cr")
            cc2.metric("Cash/MCap Ratio", f"{selected_result.get('cash_on_hand_to_mcap', 0):.2f}%")
            cc3.metric("LatestFY Rev/MCap", f"{selected_result.get('latest_fy_revenue_to_mcap', 0):.2f}x")

            if selected_result.get('historical_data') and selected_result['historical_data']['years']:
                st.markdown("---")
                st.markdown("**📈 3-Year Historical Trends**")
                historical = selected_result['historical_data']
                fig = make_subplots(rows=3, cols=1,
                                     subplot_titles=('YoY Revenue (₹ Cr)', 'Cash Amounts (₹ Cr)', 'Sales to Market Cap Ratio'),
                                     vertical_spacing=0.12)
                if historical['revenues']:
                    fig.add_trace(go.Bar(x=historical['years'], y=[r/10000000 for r in historical['revenues']],
                                          name='Revenue', marker_color='lightblue',
                                          text=[f"₹{r/10000000:.0f}Cr" for r in historical['revenues']],
                                          textposition='auto'), row=1, col=1)
                if historical['cash_amounts']:
                    fig.add_trace(go.Bar(x=historical['years'], y=[c/10000000 for c in historical['cash_amounts']],
                                          name='Cash', marker_color='lightgreen',
                                          text=[f"₹{c/10000000:.0f}Cr" for c in historical['cash_amounts']],
                                          textposition='auto'), row=2, col=1)
                if historical['sales_to_mcap']:
                    fig.add_trace(go.Scatter(x=historical['years'], y=historical['sales_to_mcap'],
                                              name='Sales/MCap', mode='lines+markers',
                                              line=dict(color='orange', width=3), marker=dict(size=10),
                                              text=[f"{s:.2f}x" for s in historical['sales_to_mcap']],
                                              textposition='top center'), row=3, col=1)
                fig.update_layout(height=900, showlegend=False, title_text=f"{selected_symbol} - 3-Year Financial Trends")
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

    sc.download_buttons(MODE_KEY, filtered_df, df, "positional_scan")

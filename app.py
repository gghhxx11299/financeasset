import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime
import yfinance as yf
import math
import numpy as np
from scipy.stats import norm, percentileofscore
import plotly.graph_objs as go
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
from plotly.subplots import make_subplots

# --- Extensive Custom CSS for styling ---
# --- EXTREME CSS STYLING ---
# --- ULTRA CYBERPUNK CSS STYLING ---
# --- Enhanced Cyberpunk CSS with Better Layout ---
# --- ULTRA CYBERPUNK CSS WITH WIDER CHARTS ---
# --- ENHANCED CYBERPUNK CSS WITH BALANCED LAYOUT ---
st.markdown("""
<style>
    /* === CYBERPUNK BASE === */
    :root {
        --neon-blue: #0ff;
        --neon-green: #0f0;
        --neon-pink: #f0f;
        --dark-bg: #0a0a14;
        --darker-bg: #050510;
        --glow-blue: 0 0 10px rgba(0, 255, 255, 0.7);
        --glow-green: 0 0 10px rgba(0, 255, 0, 0.7);
    }
    
    /* === MAIN CONTAINER === */
    .main {
        background: linear-gradient(135deg, var(--dark-bg) 0%, var(--darker-bg) 100%) !important;
        border: 1px solid var(--neon-blue) !important;
        box-shadow: var(--glow-blue), inset 0 0 20px rgba(0, 255, 255, 0.2) !important;
        border-radius: 0 !important;
        padding: 2rem !important;
        max-width: 98vw !important;
        margin: 0 auto !important;
    }

    /* === THREE-COLUMN LAYOUT === */
    .stContainer {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 1.5rem !important;
        margin-bottom: 2rem !important;
    }

    .stColumn {
        flex: 1 !important;
        min-width: 300px !important;
        background: rgba(15, 15, 30, 0.6) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        padding: 1.5rem !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2) !important;
    }

    /* === WIDE ELEMENTS === */
    div[data-testid="stExpander"] > div {
        width: 100% !important;
        background: rgba(10, 15, 25, 0.9) !important;
        border: 1px solid var(--neon-green) !important;
        box-shadow: var(--glow-green) !important;
    }

    .stPlotlyChart {
        width: 100% !important;
        min-width: 100% !important;
    }

    /* === METRIC CARDS === */
    .stMetric {
        background: rgba(10, 10, 20, 0.8) !important;
        border: 1px solid var(--neon-blue) !important;
        box-shadow: var(--glow-blue), inset 0 0 10px rgba(0, 255, 255, 0.1) !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* === DATA TABLES === */
    .stDataFrame {
        width: 100% !important;
        background: rgba(5, 5, 15, 0.7) !important;
        border: 1px solid var(--neon-blue) !important;
        box-shadow: var(--glow-blue) !important;
    }

    /* === FOOTER === */
    .footer {
        background: rgba(0, 5, 10, 0.9) !important;
        border-top: 1px solid var(--neon-blue) !important;
        padding: 1.5rem !important;
        margin-top: 2rem !important;
        text-align: center !important;
    }

    .footer a {
        color: var(--neon-green) !important;
        margin: 0 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- FOOTER WITH CREDITS ---
st.markdown("""
<div class="footer">
    <div>
        <span style="color: #0ff;">Made with ❤️ by </span>
        <span style="color: #0f0; font-weight: bold;">Geabral Mulugeta</span>
    </div>
    <div style="margin-top: 1rem;">
        <a href="https://github.com/gghhxx11299" target="_blank">GitHub</a>
        <span style="color: #0ff;"> | </span>
        <a href="https://www.linkedin.com/in/geabral-mulugeta-334358327/" target="_blank">LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)
SECTOR_MAP = {
    "technology": ["XLK", "VGT", "QTEC"],
    "financial": ["XLF", "VFH", "IYF"],
    "energy": ["XLE", "VDE", "IXC"],
    "healthcare": ["XLV", "IBB", "VHT"],
    "consumer discretionary": ["XLY", "VCR", "FDIS"],
    "consumer staples": ["XLP", "VDC", "FSTA"],
    "industrial": ["XLI", "VIS", "VMI"],
    "utilities": ["XLU", "VPU"],
    "real estate": ["XLRE", "VNQ", "SCHH"],
    "materials": ["XLB", "VAW"],
    "agriculture": ["DBA", "COW", "MOO"],
    "gold": ["GLD", "IAU", "SGOL"],
    "oil": ["USO", "OIH", "XOP"],
    "cryptocurrency": ["BTC-USD", "ETH-USD", "GBTC"],
    "bonds": ["AGG", "BND", "LQD"],
    "semiconductors": ["SMH", "SOXX"],
    "retail": ["XRT", "RTH"],
    "telecommunications": ["XTL", "VOX"],
    "transportation": ["IYT", "XTN"],
}

# --- Pricing Models ---
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes option price"""
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree_price(S, K, T, r, sigma, option_type="call", steps=100):
    """Calculate option price using binomial tree model"""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if option_type == "call":
        values = [max(0, price - K) for price in prices]
    else:
        values = [max(0, K - price) for price in prices]

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            values[j] = (p * values[j + 1] + (1 - p) * values[j]) * math.exp(-r * dt)

    return values[0]

def monte_carlo_price(S, K, T, r, sigma, option_type="call", simulations=10000):
    """Calculate option price using Monte Carlo simulation"""
    np.random.seed(42)
    dt = T
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(simulations))
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return float(price)

# --- Greeks Calculations ---
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes Greeks"""
    if T <= 0 or sigma == 0:
        return dict(Delta=0, Gamma=0, Vega=0, Theta=0, Rho=0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_call if option_type == "call" else theta_put
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    rho = rho_call if option_type == "call" else rho_put

    return dict(
        Delta=float(delta),
        Gamma=float(gamma),
        Vega=float(vega),
        Theta=float(theta),
        Rho=float(rho)
    )

# --- Implied Volatility ---
def implied_volatility(option_market_price, S, K, T, r, option_type="call", tol=1e-5, max_iter=100):
    """Calculate implied volatility using bisection method"""
    sigma_low, sigma_high = 0.0001, 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = black_scholes_price(S, K, T, r, sigma_mid, option_type)
        if abs(price - option_market_price) < tol:
            return sigma_mid
        if price > option_market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return None

# --- Market Data Functions ---
def get_option_market_price(ticker, option_type, strike, expiry_date):
    """Fetch current market price for given option"""
    stock = yf.Ticker(ticker)
    try:
        if expiry_date not in stock.options:
            return None
        opt_chain = stock.option_chain(expiry_date)
        options = opt_chain.calls if option_type == "call" else opt_chain.puts
        row = options[options['strike'] == strike]
        return None if row.empty else float(row.iloc[0]['lastPrice'])
    except:
        return None

def get_us_10yr_treasury_yield():
    """Fetch current 10-year Treasury yield"""
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/Datasets/yield.csv"
    fallback_yield = 0.025  # fallback 2.5%

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        df = df.dropna(subset=["10 Yr"])
        if df.empty:
            return fallback_yield

        latest_yield_str = df["10 Yr"].iloc[-1]
        return float(latest_yield_str) / 100
    except Exception:
        return fallback_yield

# --- Volatility Analysis ---
def calculate_iv_percentile(ticker, current_iv, lookback_days=365):
    """Calculate how current IV compares to historical levels"""
    try:
        hist = yf.download(ticker, period=f"{lookback_days}d")["Close"]
        daily_returns = hist.pct_change().dropna()
        realized_vol = daily_returns.std() * np.sqrt(252)
        
        return float(percentileofscore([realized_vol], current_iv))
    except Exception as e:
        st.warning(f"Could not calculate IV percentile: {e}")
        return None
def plot_stock_volume(ticker, days_to_expiry):
    """Plot stock/ETF trading volume - handles both simple and MultiIndex DataFrames"""
    try:
        # 1. Fetch data with multiple fallback attempts
        stock_data = None
        fetch_attempts = [
            {'auto_adjust': True, 'actions': False},
            {'auto_adjust': False, 'actions': False},
            {'auto_adjust': True, 'actions': True},
            {'auto_adjust': False, 'actions': True}
        ]
        
        for attempt in fetch_attempts:
            try:
                stock_data = yf.download(
                    ticker,
                    period=f"{min(days_to_expiry, 365)}d",
                    progress=False,
                    **attempt
                )
                if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
                    break
            except:
                continue

        # 2. Validate data structure
        if not isinstance(stock_data, pd.DataFrame) or stock_data.empty:
            st.warning(f"⚠️ No market data available for {ticker}")
            return None

        # 3. Find volume column (handles both regular and MultiIndex columns)
        volume_col = None
        for col in stock_data.columns:
            # Case 1: Simple string column name (e.g., 'Volume')
            if isinstance(col, str) and 'volume' in col.lower():
                volume_col = col
                break
            # Case 2: MultiIndex tuple column (e.g., ('Volume', 'SPY'))
            elif isinstance(col, tuple) and any('volume' in str(s).lower() for s in col):
                volume_col = col
                break

        if not volume_col:
            st.warning(f"📊 Volume data missing for {ticker} (Available columns: {stock_data.columns.tolist()})")
            return None

        # 4. Clean data
        clean_data = stock_data[[volume_col]].dropna()
        if clean_data.empty:
            st.warning(f"🧹 No valid volume data after cleaning for {ticker}")
            return None

        # 5. Create plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clean_data.index,
            y=clean_data[volume_col],
            marker_color='#1f77b4',
            hovertemplate="<b>Date</b>: %{x|%b %d}<br><b>Volume</b>: %{y:,}<extra></extra>"
        ))
        
        # Add average line if we have valid data
        try:
            avg_volume = clean_data[volume_col].mean()
            fig.add_shape(
                type="line",
                x0=clean_data.index[0],
                x1=clean_data.index[-1],
                y0=avg_volume,
                y1=avg_volume,
                line=dict(color='#ff7f0e', dash='dot')
            )
            fig.add_annotation(
                x=clean_data.index[-1],
                y=avg_volume,
                text=f"Avg: {avg_volume:,.0f}",
                showarrow=False,
                xanchor='right',
                yanchor='bottom',
                font=dict(color="#ff7f0e")
            )
        except:
            pass
        
        fig.update_layout(
            title=f"<b>{ticker} Volume</b> | Last {len(clean_data)} Trading Days",
            yaxis_title="Shares Traded",
            template="plotly_white",
            hovermode="x unified"
        )
        
        return fig

    except Exception as e:
        st.error(f"❌ Error processing {ticker}: {str(e)}")
        return None

def plot_black_scholes_sensitivities(S, K, T, r, sigma, option_type):
    """Create enhanced interactive sensitivity plot for Black-Scholes model"""
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=("Price vs Underlying Asset", 
                                      "Price vs Days to Expiry", 
                                      "Price vs Volatility"))
    
    S_range = np.linspace(0.5*S, 1.5*S, 100)
    prices_S = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]
    fig.add_trace(go.Scatter(
        x=S_range, 
        y=prices_S, 
        name='Price vs Underlying',
        line=dict(color='#636EFA'),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.1)',
        hovertemplate="<b>Stock Price</b>: $%{x:.2f}<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    T_range = np.linspace(0.01, T*2, 100)
    prices_T = [black_scholes_price(S, K, t, r, sigma, option_type) for t in T_range]
    fig.add_trace(go.Scatter(
        x=T_range*365, 
        y=prices_T, 
        name='Price vs Days to Expiry',
        line=dict(color='#EF553B'),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.1)',
        hovertemplate="<b>Days to Expiry</b>: %{x:.0f}<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=2, col=1)
    
    sigma_range = np.linspace(0.01, 2*sigma, 100)
    prices_sigma = [black_scholes_price(S, K, T, r, s, option_type) for s in sigma_range]
    fig.add_trace(go.Scatter(
        x=sigma_range*100, 
        y=prices_sigma, 
        name='Price vs Volatility',
        line=dict(color='#00CC96'),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)',
        hovertemplate="<b>Volatility</b>: %{x:.2f}%<br><b>Option Price</b>: $%{y:.2f}<extra></extra>"
    ), row=3, col=1)
    
    fig.add_vline(x=S, row=1, col=1, line=dict(color='#636EFA', dash='dash'), 
                annotation_text=f'Current Price: ${S:.2f}',
                annotation_position="top right")
    
    fig.add_vline(x=T*365, row=2, col=1, line=dict(color='#EF553B', dash='dash'), 
                annotation_text=f'Current DTE: {T*365:.0f} days',
                annotation_position="top right")
    
    fig.add_vline(x=sigma*100, row=3, col=1, line=dict(color='#00CC96', dash='dash'), 
                annotation_text=f'Current IV: {sigma*100:.2f}%',
                annotation_position="top right")
    
    fig.update_layout(
        title=f'<b>Black-Scholes Sensitivities ({option_type.capitalize()} Option)</b>',
        height=900,
        showlegend=False,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=50, r=50, b=50, t=100),
        title_font=dict(size=20, color='#2c3e50'),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_yaxes(title_text="Option Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Option Price ($)", row=3, col=1)
    
    fig.update_xaxes(title_text="Underlying Asset Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Days to Expiration", row=2, col=1)
    fig.update_xaxes(title_text="Implied Volatility (%)", row=3, col=1)
    
    return fig

# --- Reporting ---
def prepare_export_csv(greeks_df, summary_df, trading_advice):
    greeks_export = greeks_df.rename(columns={"Greek": "Metric"})
    summary_export = summary_df
    advice_export = trading_advice.rename(columns={"Advice": "Metric", "Reason": "Value"})
    
    export_df = pd.concat([greeks_export, summary_export, advice_export], ignore_index=True)
    return export_df.to_csv(index=False).encode('utf-8')

def generate_pdf_report(input_data, greeks_df, summary_df, trading_advice):
    """Generate PDF report using FPDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, "Options Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Input Parameters", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Greeks", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in greeks_df.iterrows():
        pdf.cell(200, 10, f"{row['Greek']}: {row['Value']}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in summary_df.iterrows():
        pdf.cell(200, 10, f"{row['Metric']}: {row['Value']}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Trading Advice", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in trading_advice.iterrows():
        # Handle special characters in advice
        text = f"{row['Advice']}: {row['Reason']}"
        try:
            pdf.multi_cell(200, 10, text.encode('latin-1', 'replace').decode('latin-1'))
        except:
            pdf.multi_cell(200, 10, "Trading advice (special characters omitted)")

    pdf.ln(10)
    pdf.set_font("Arial", 'I', size=10)
    pdf.cell(200, 10, "Note: Interactive plots are available in the web interface", ln=True)

    # Save to bytes with error handling
    try:
        return pdf.output(dest='S').encode('latin-1', 'replace')
    except:
        try:
            return pdf.output(dest='S').encode('utf-8')
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            return None

def generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital):
    """Generate personalized trading advice based on analysis"""
    advice = []
    reasons = []
    
    high_iv_divergence = any(d > 0.1 for d in iv_divergences.values())
    if high_iv_divergence:
        max_divergence = max(iv_divergences.values())
        advice.append("Reduce position size")
        reasons.append(f"High IV divergence ({max_divergence:.2f} > 0.1) suggests overpriced options")
    
    extreme_z = abs(latest_z) > 2
    if extreme_z:
        advice.append("Exercise caution")
        reasons.append(f"Extreme price movement (Z-score: {latest_z:.2f}) indicates potential mean reversion")
    
    low_correlation = correlation < 0.5
    if low_correlation:
        advice.append("Consider hedging")
        reasons.append(f"Low sector correlation ({correlation:.2f}) reduces hedging effectiveness")
    
    capital_ratio = capital / comfortable_capital
    if capital_ratio < 0.7:
        advice.append("Reduce trade size significantly")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount")
    elif capital_ratio < 0.9:
        advice.append("Reduce trade size moderately")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount")
    
    if not advice:
        advice.append("Normal trading conditions")
        reasons.append("All metrics within normal ranges - standard position sizing appropriate")
    
    return pd.DataFrame({
        "Advice": advice,
        "Reason": reasons
    })

# --- Streamlit UI ---
def main():
    st.title("Options Profit & Capital Advisor")

    # Initialize session state variables
    if "calculation_done" not in st.session_state:
        st.session_state.calculation_done = False
    if "export_csv" not in st.session_state:
        st.session_state.export_csv = None
    if "export_pdf" not in st.session_state:
        st.session_state.export_pdf = None
    if "greeks_df" not in st.session_state:
        st.session_state.greeks_df = None
    if "summary_info" not in st.session_state:
        st.session_state.summary_info = None
    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    if "trading_advice" not in st.session_state:
        st.session_state.trading_advice = None
    if "bs_sensitivities_fig" not in st.session_state:
        st.session_state.bs_sensitivities_fig = None
    if "iv_percentile" not in st.session_state:
        st.session_state.iv_percentile = None
    if "volume_fig" not in st.session_state:
        st.session_state.volume_fig = None

    # Input widgets
    st.markdown("### Input Parameters")
    with st.expander("Configure your option trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
            option_type = st.selectbox("Option Type", ["call", "put"])
            strike_price = st.number_input("Strike Price", min_value=0.0, value=150.0)
            days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
            risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.025)
            sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
            
        with col2:
            return_type = st.selectbox("Return Type", ["Simple", "Log"])
            comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
            max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
            min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
            pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

    # Calculation button
    st.markdown("---")
    calculate_clicked = st.button("Calculate Profit & Advice", key="calculate")

    # When Calculate button is pressed
    if calculate_clicked:
        with st.spinner("Calculating option values and generating advice..."):
            try:
                # Store input data for PDF report
                st.session_state.input_data = {
                    "Stock Ticker": ticker,
                    "Option Type": option_type,
                    "Strike Price": strike_price,
                    "Days to Expiry": days_to_expiry,
                    "Risk-Free Rate": risk_free_rate,
                    "Sector": sector,
                    "Return Type": return_type,
                    "Comfortable Capital": comfortable_capital,
                    "Max Capital": max_capital,
                    "Min Capital": min_capital,
                    "Pricing Model": pricing_model
                }

                # Fetch live treasury yield
                live_rate = get_us_10yr_treasury_yield()
                if live_rate is not None:
                    risk_free_rate = live_rate

                T = days_to_expiry / 365
                stock_data = yf.Ticker(ticker).history(period="1d")
                if stock_data.empty:
                    st.error("Could not fetch stock data. Please check the ticker symbol.")
                    st.session_state.calculation_done = False
                    return
                
                S = float(stock_data["Close"].iloc[-1])

                # Find closest expiry date
                options_expiries = yf.Ticker(ticker).options
                expiry_date = None
                for date in options_expiries:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    diff_days = abs((dt - datetime.now()).days - days_to_expiry)
                    if diff_days <= 5:
                        expiry_date = date
                        break

                if expiry_date is None:
                    st.error("No matching expiry date found near the specified days to expiry.")
                    st.session_state.calculation_done = False
                    return

                # Get market price and implied volatility
                price_market = get_option_market_price(ticker, option_type, strike_price, expiry_date)
                if price_market is None:
                    st.error("Failed to fetch option market price. Try a closer-to-the-money strike.")
                    st.session_state.calculation_done = False
                    return

                iv = implied_volatility(price_market, S, strike_price, T, risk_free_rate, option_type)
                if iv is None:
                    st.error("Could not compute implied volatility. Try a closer-to-the-money strike.")
                    st.session_state.calculation_done = False
                    return

                # Calculate Greeks
                greeks = black_scholes_greeks(S, strike_price, T, risk_free_rate, iv, option_type)
                greeks_df = pd.DataFrame({
                    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
                    "Value": [
                        greeks['Delta'],
                        greeks['Gamma'],
                        greeks['Vega'],
                        greeks['Theta'],
                        greeks['Rho']
                    ]
                })
                st.session_state.greeks_df = greeks_df

                # Calculate option price using selected model
                start = time.time()
                if pricing_model == "Black-Scholes":
                    price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
                elif pricing_model == "Binomial Tree":
                    price = binomial_tree_price(S, strike_price, T, risk_free_rate, iv, option_type)
                elif pricing_model == "Monte Carlo":
                    price = monte_carlo_price(S, strike_price, T, risk_free_rate, iv, option_type)
                else:
                    price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
                end = time.time()
                calc_time = end - start

                # Sector analysis
                etfs = SECTOR_MAP.get(sector, [])
                symbols = [ticker] + etfs
                df = yf.download(symbols, period="1mo", interval="1d")["Close"].dropna(axis=1, how="any")

                if return_type == "Log":
                    returns = (df / df.shift(1)).apply(np.log).dropna()
                else:
                    returns = df.pct_change().dropna()

                # Z-score calculation
                window = 20
                zscore = ((df[ticker] - df[ticker].rolling(window).mean()) / df[ticker].rolling(window).std()).dropna()
                latest_z = float(zscore.iloc[-1]) if not zscore.empty else 0

                # Correlation analysis
                correlation = float(returns.corr().loc[ticker].drop(ticker).mean())
                iv_divergences = {etf: iv - 0.2 for etf in df.columns if etf != ticker}

                # Capital adjustment logic
                capital = comfortable_capital
                if any(d > 0.1 for d in iv_divergences.values()):
                    capital *= 0.6
                if abs(latest_z) > 2:
                    capital *= 0.7
                if correlation < 0.5:
                    capital *= 0.8

                capital = max(min_capital, min(max_capital, capital))

                # IV percentile analysis
                iv_percentile = calculate_iv_percentile(ticker, iv)
                st.session_state.iv_percentile = iv_percentile
                
                # Generate stock volume chart
                volume_fig = plot_stock_volume(ticker, days_to_expiry)
                st.session_state.volume_fig = volume_fig

                # Generate trading advice
                trading_advice = generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital)
                
                # Add warning if IV is extreme
                if iv_percentile and iv_percentile > 90:
                    trading_advice = pd.concat([
                        trading_advice,
                        pd.DataFrame({
                            "Advice": ["Market Stress Warning"],
                            "Reason": [f"IV is in top {100-iv_percentile:.0f}% of historical levels - possible crisis ahead"]
                        })
                    ])
                
                st.session_state.trading_advice = trading_advice

                # Prepare summary DataFrame
                summary_df = pd.DataFrame({
                    "Metric": ["Market Price", f"Model Price ({pricing_model})", "Implied Volatility (IV)", "Suggested Capital", "Calculation Time"],
                    "Value": [
                        f"${price_market:.2f}",
                        f"${float(price):.2f}",
                        f"{iv*100:.2f}%",
                        f"${float(capital):.2f}",
                        f"{float(calc_time):.4f} seconds"
                    ]
                })
                st.session_state.summary_info = summary_df

                # Prepare CSV export
                csv = prepare_export_csv(greeks_df, summary_df, trading_advice)
                st.session_state.export_csv = csv

                # Profit vs capital plot
                capitals = list(range(int(min_capital), int(max_capital) + 1, 100))
                profits = []
                profits_ci_lower = []
                profits_ci_upper = []

                if pricing_model == "Monte Carlo":
                    simulations = 10000
                    np.random.seed(42)
                    dt = T
                    ST = S * np.exp((risk_free_rate - 0.5 * iv**2) * dt + iv * np.sqrt(dt) * np.random.randn(simulations))
                    if option_type == "call":
                        payoffs = np.maximum(ST - strike_price, 0)
                    else:
                        payoffs = np.maximum(strike_price - ST, 0)
                    discounted_payoffs = np.exp(-risk_free_rate * T) * payoffs
                    price_samples = discounted_payoffs

                    for cap in capitals:
                        contracts = int(cap / (price * 100)) if price > 0 else 0
                        profits_samples = contracts * 100 * (price_samples * 1.05 - price_samples)
                        mean_profit = float(profits_samples.mean())
                        std_profit = float(profits_samples.std())
                        ci_lower = mean_profit - 1.96 * std_profit / np.sqrt(simulations)
                        ci_upper = mean_profit + 1.96 * std_profit / np.sqrt(simulations)
                        profits.append(mean_profit)
                        profits_ci_lower.append(ci_lower)
                        profits_ci_upper.append(ci_upper)
                else:
                    for cap in capitals:
                        contracts = int(cap / (price * 100)) if price > 0 else 0
                        profit = contracts * 100 * (price * 1.05 - price)
                        profits.append(float(profit))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=capitals,
                    y=profits,
                    mode='lines+markers',
                    name='Expected Profit',
                    line=dict(color='#4CAF50', width=2),
                    marker=dict(size=8, color='#4CAF50'),
                    hovertemplate='<b>Capital</b>: $%{x:,.0f}<br><b>Profit</b>: $%{y:,.2f}<extra></extra>',
                ))

                if pricing_model == "Monte Carlo":
                    fig.add_trace(go.Scatter(
                        x=capitals + capitals[::-1],
                        y=profits_ci_upper + profits_ci_lower[::-1],
                        fill='toself',
                        fillcolor='rgba(76, 175, 80, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Interval",
                    ))

                fig.update_layout(
                    title=f"<b>Expected Profit vs Capital for {ticker} {option_type.capitalize()} Option</b>",
                    xaxis_title="Capital Invested ($)",
                    yaxis_title="Expected Profit ($)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=500,
                    margin=dict(l=50, r=50, b=50, t=80),
                    title_font=dict(size=18, color="#2c3e50"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.session_state.plot_fig = fig

                # Generate Black-Scholes sensitivities plot if using BS model
                if pricing_model == "Black-Scholes":
                    bs_sensitivities_fig = plot_black_scholes_sensitivities(S, strike_price, T, risk_free_rate, iv, option_type)
                    st.session_state.bs_sensitivities_fig = bs_sensitivities_fig

                # Generate PDF report
                try:
                    pdf = generate_pdf_report(st.session_state.input_data, greeks_df, summary_df, trading_advice)
                    if pdf is not None:
                        st.session_state.export_pdf = pdf
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                    st.session_state.export_pdf = None
                
                st.session_state.calculation_done = True
                st.success("Calculation complete!")

            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                st.session_state.calculation_done = False

    # Display results if calculation is done
    if st.session_state.calculation_done:
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        # Metrics in cards
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Option Greeks")
                if st.session_state.greeks_df is not None:
                    st.dataframe(st.session_state.greeks_df, use_container_width=True)
            
            with col2:
                st.markdown("### Summary Metrics")
                if st.session_state.summary_info is not None:
                    st.dataframe(st.session_state.summary_info, use_container_width=True)
            
            with col3:
                if st.session_state.iv_percentile is not None:
                    st.markdown("### Volatility Context")
                    st.metric(
                        label="Implied Volatility Percentile",
                        value=f"{st.session_state.iv_percentile:.0f}th percentile",
                        help="How current IV compares to 1-year history (higher = more extreme)"
                    )
        
        # Trading Advice
        st.markdown("### Trading Advice")
        with st.expander("View detailed trading recommendations"):
            if st.session_state.trading_advice is not None:
                st.dataframe(st.session_state.trading_advice, use_container_width=True)
        
        # Plots
        if st.session_state.plot_fig is not None:
            st.plotly_chart(st.session_state.plot_fig, use_container_width=True)
        
        if st.session_state.volume_fig is not None:
            st.plotly_chart(st.session_state.volume_fig, use_container_width=True)
        
        if st.session_state.bs_sensitivities_fig is not None:
            st.plotly_chart(st.session_state.bs_sensitivities_fig, use_container_width=True)
        
        # Export buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.export_csv is not None:
                st.download_button(
                    label="Download CSV Report",
                    data=st.session_state.export_csv,
                    file_name="options_analysis_report.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.session_state.export_pdf is not None:
                st.download_button(
                    label="Download PDF Report",
                    data=st.session_state.export_pdf,
                    file_name="options_analysis_report.pdf",
                    mime="application/pdf"
                )

# Add this function near the other data functions
def get_company_financials(ticker):
    """Fetch and display key financial metrics for a stock"""
    try:
        company = yf.Ticker(ticker)
        
        # Check if this is actually a stock (has financials)
        if not company.info:
            return False, None
        
        # Get key financial data
        financials = {
            'Company Name': company.info.get('longName', 'N/A'),
            'Sector': company.info.get('sector', 'N/A'),
            'Industry': company.info.get('industry', 'N/A'),
            'Market Cap': f"${company.info.get('marketCap', 0)/1e9:.2f}B" if company.info.get('marketCap') else 'N/A',
            'P/E Ratio': company.info.get('trailingPE', 'N/A'),
            'EPS': company.info.get('trailingEps', 'N/A'),
            'Dividend Yield': f"{company.info.get('dividendYield', 0)*100:.2f}%" if company.info.get('dividendYield') else '0%',
            '52 Week High': f"${company.info.get('fiftyTwoWeekHigh', 'N/A')}",
            '52 Week Low': f"${company.info.get('fiftyTwoWeekLow', 'N/A')}",
            'Beta': company.info.get('beta', 'N/A')
        }
        
        return True, pd.DataFrame.from_dict(financials, orient='index', columns=['Value'])
    
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return False, None

# Modify the main function to include financials display
def main():
    st.title("Options Profit & Capital Advisor")

    # Initialize session state variables
    if "calculation_done" not in st.session_state:
        st.session_state.calculation_done = False
    if "export_csv" not in st.session_state:
        st.session_state.export_csv = None
    if "export_pdf" not in st.session_state:
        st.session_state.export_pdf = None
    if "greeks_df" not in st.session_state:
        st.session_state.greeks_df = None
    if "summary_info" not in st.session_state:
        st.session_state.summary_info = None
    if "plot_fig" not in st.session_state:
        st.session_state.plot_fig = None
    if "input_data" not in st.session_state:
        st.session_state.input_data = None
    if "trading_advice" not in st.session_state:
        st.session_state.trading_advice = None
    if "bs_sensitivities_fig" not in st.session_state:
        st.session_state.bs_sensitivities_fig = None
    if "iv_percentile" not in st.session_state:
        st.session_state.iv_percentile = None
    if "volume_fig" not in st.session_state:
        st.session_state.volume_fig = None
    if "is_stock" not in st.session_state:
        st.session_state.is_stock = None
    if "financials_df" not in st.session_state:
        st.session_state.financials_df = None

    # Input widgets
    st.markdown("### Input Parameters")
    with st.expander("Configure your option trade"):
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
            option_type = st.selectbox("Option Type", ["call", "put"])
            strike_price = st.number_input("Strike Price", min_value=0.0, value=150.0)
            days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
            risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.025)
            sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
            
        with col2:
            return_type = st.selectbox("Return Type", ["Simple", "Log"])
            comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
            max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
            min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
            pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

    # Check if ticker is a stock and get financials
    if ticker:
        is_stock, financials_df = get_company_financials(ticker)
        st.session_state.is_stock = is_stock
        st.session_state.financials_df = financials_df
        
        if not is_stock:
            st.warning(f"⚠️ {ticker} does not appear to be a stock. This tool works best with individual stocks, not ETFs, indices, or cryptocurrencies.")
        else:
            with st.expander("View Company Financials", expanded=True):
                st.dataframe(financials_df, use_container_width=True)

    # Calculation button (only show if it's a stock)
    st.markdown("---")
    if st.session_state.is_stock is not False:
        calculate_clicked = st.button("Calculate Profit & Advice", key="calculate")
    else:
        st.error("Please enter a valid stock ticker to proceed with calculations")
        calculate_clicked = False

    # Rest of your existing main() function continues here...
    # [Keep all the existing calculation and display code]

if __name__ == "__main__":
    main()

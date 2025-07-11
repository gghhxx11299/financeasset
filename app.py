import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from datetime import datetime
import yfinance as yf
import math
import numpy as np
from scipy.stats import norm
import plotly.graph_objs as go
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO


def prepare_export_csv(greeks_df, summary_df, trading_advice):
    # Combine all data for CSV export
    greeks_export = greeks_df.rename(columns={"Greek": "Metric"})
    summary_export = summary_df
    advice_export = trading_advice.rename(columns={"Advice": "Metric", "Reason": "Value"})
    
    export_df = pd.concat([greeks_export, summary_export, advice_export], ignore_index=True)
    return export_df.to_csv(index=False).encode('utf-8')


def generate_pdf_report(input_data, greeks_df, summary_df, trading_advice):
    """Generate PDF report using FPDF without image dependencies"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, "Options Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Inputs section
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Input Parameters", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    # Greeks section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Greeks", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in greeks_df.iterrows():
        pdf.cell(200, 10, f"{row['Greek']}: {row['Value']}", ln=True)

    # Summary section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Summary", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in summary_df.iterrows():
        pdf.cell(200, 10, f"{row['Metric']}: {row['Value']}", ln=True)

    # Trading Advice section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Trading Advice", ln=True)
    pdf.set_font("Arial", size=12)
    for _, row in trading_advice.iterrows():
        pdf.multi_cell(200, 10, f"{row['Advice']}: {row['Reason']}")

    # Note about plots
    pdf.ln(10)
    pdf.set_font("Arial", 'I', size=10)
    pdf.cell(200, 10, "Note: Interactive plots are available in the web interface", ln=True)

    return pdf


def plot_black_scholes_sensitivities(S, K, T, r, sigma, option_type):
    """Create interactive sensitivity plot for Black-Scholes model"""
    fig = go.Figure()
    
    # Price vs Underlying (S)
    S_range = np.linspace(0.5*S, 1.5*S, 50)
    prices_S = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]
    fig.add_trace(go.Scatter(
        x=S_range, 
        y=prices_S, 
        name='Price vs Underlying',
        line=dict(color='royalblue')
    ))
    
    # Price vs Time (T)
    T_range = np.linspace(0.01, T*2, 50)
    prices_T = [black_scholes_price(S, K, t, r, sigma, option_type) for t in T_range]
    fig.add_trace(go.Scatter(
        x=T_range*365, 
        y=prices_T, 
        name='Price vs Days to Expiry',
        line=dict(color='firebrick')
    ))
    
    # Price vs Volatility (σ)
    sigma_range = np.linspace(0.01, 2*sigma, 50)
    prices_sigma = [black_scholes_price(S, K, T, r, s, option_type) for s in sigma_range]
    fig.add_trace(go.Scatter(
        x=sigma_range, 
        y=prices_sigma, 
        name='Price vs Volatility',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title=f'Black-Scholes Sensitivities ({option_type.capitalize()} Option)',
        xaxis_title='Parameter Value',
        yaxis_title='Option Price',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    # Add vertical lines for current values
    fig.add_vline(x=S, line=dict(color='royalblue', dash='dash'), 
                annotation_text=f'Current S={S:.2f}')
    fig.add_vline(x=T*365, line=dict(color='firebrick', dash='dash'), 
                annotation_text=f'Current T={T*365:.0f} days')
    fig.add_vline(x=sigma, line=dict(color='green', dash='dash'), 
                annotation_text=f'Current σ={sigma:.2f}')
    
    return fig


def generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital):
    """Generate personalized trading advice based on analysis"""
    advice = []
    reasons = []
    
    # Analyze IV divergence
    high_iv_divergence = any(d > 0.1 for d in iv_divergences.values())
    if high_iv_divergence:
        max_divergence = max(iv_divergences.values())
        advice.append("Reduce position size")
        reasons.append(f"High IV divergence ({max_divergence:.2f} > 0.1) suggests overpriced options relative to sector peers")
    
    # Analyze Z-score
    extreme_z = abs(latest_z) > 2
    if extreme_z:
        advice.append("Exercise caution")
        reasons.append(f"Extreme price movement (Z-score: {latest_z:.2f}) indicates potential mean reversion")
    
    # Analyze correlation
    low_correlation = correlation < 0.5
    if low_correlation:
        advice.append("Consider hedging")
        reasons.append(f"Low sector correlation ({correlation:.2f}) reduces hedging effectiveness")
    
    # Capital adjustment analysis
    capital_ratio = capital / comfortable_capital
    if capital_ratio < 0.7:
        advice.append("Reduce trade size significantly")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount due to multiple risk factors")
    elif capital_ratio < 0.9:
        advice.append("Reduce trade size moderately")
        reasons.append(f"Suggested capital ${capital:.0f} is {capital_ratio*100:.0f}% of comfortable amount")
    
    # Default advice if no warnings
    if not advice:
        advice.append("Normal trading conditions")
        reasons.append("All metrics within normal ranges - standard position sizing appropriate")
    
    return pd.DataFrame({
        "Advice": advice,
        "Reason": reasons
    })


# --- Sector ETFs ---
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

# Pricing and Greeks functions
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
    return price

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """Calculate Black-Scholes Greeks"""
    if T <= 0 or sigma == 0:
        return dict(Delta=0, Gamma=0, Vega=0, Theta=0, Rho=0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                  - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = theta_call if option_type == "call" else theta_put
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    rho = rho_call if option_type == "call" else rho_put

    return dict(Delta=delta, Gamma=gamma, Vega=vega, Theta=theta, Rho=rho)

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

def get_option_market_price(ticker, option_type, strike, expiry_date):
    """Fetch current market price for given option"""
    stock = yf.Ticker(ticker)
    try:
        if expiry_date not in stock.options:
            return None
        opt_chain = stock.option_chain(expiry_date)
        options = opt_chain.calls if option_type == "call" else opt_chain.puts
        row = options[options['strike'] == strike]
        return None if row.empty else row.iloc[0]['lastPrice']
    except:
        return None

def get_us_10yr_treasury_yield():
    """Fetch current 10-year Treasury yield"""
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/Datasets/yield.csv"
    fallback_yield = 0.025  # fallback 2.5%

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Read CSV from text response
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        # The CSV has a 'Date' column and columns for maturities like '10 Yr'
        # The most recent date is at the bottom, get the last valid 10 Yr yield
        df = df.dropna(subset=["10 Yr"])
        if df.empty:
            return fallback_yield

        latest_yield_str = df["10 Yr"].iloc[-1]
        latest_yield = float(latest_yield_str) / 100  # convert from percent to decimal

        return latest_yield

    except Exception:
        return fallback_yield

# --- Streamlit UI ---
st.title("Options Profit & Capital Advisor")

# Input widgets
ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
strike_price = st.number_input("Desired Strike Price", min_value=0.0, value=150.0)
days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
default_risk_free_rate = 0.025
risk_free_rate = st.number_input("Risk-Free Rate (e.g. 0.025)", min_value=0.0, max_value=1.0, value=default_risk_free_rate)
sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
return_type = st.selectbox("Return Type", ["Simple", "Log"])
comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

# --- Buttons ---
calc_col, export_csv_col, export_pdf_col = st.columns([2, 1, 1])

with calc_col:
    calculate_clicked = st.button("Calculate Profit & Advice")

# Initialize session state variables for storing results
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

# When Calculate button is pressed, run calculations and save results in session_state
if calculate_clicked:
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

        # Fetch live treasury yield on calculation click
        live_rate = get_us_10yr_treasury_yield()
        if live_rate is not None:
            risk_free_rate = live_rate

        T = days_to_expiry / 365
        S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

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
            st.stop()

        price_market = get_option_market_price(ticker, option_type, strike_price, expiry_date)
        if price_market is None:
            st.error("Failed to fetch option market price. Try a closer-to-the-money strike.")
            st.session_state.calculation_done = False
            st.stop()

        iv = implied_volatility(price_market, S, strike_price, T, risk_free_rate, option_type)
        if iv is None:
            st.error("Could not compute implied volatility. Try a closer-to-the-money strike.")
            st.session_state.calculation_done = False
            st.stop()

        greeks = black_scholes_greeks(S, strike_price, T, risk_free_rate, iv, option_type)
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [greeks["Delta"], greeks["Gamma"], greeks["Vega"], greeks["Theta"], greeks["Rho"]]
        })
        greeks_df["Value"] = greeks_df["Value"].map(lambda x: f"{x:.4f}")

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

        etfs = SECTOR_MAP.get(sector, [])
        symbols = [ticker] + etfs
        df = yf.download(symbols, period="1mo", interval="1d")["Close"].dropna(axis=1, how="any")

        if return_type == "Log":
            returns = (df / df.shift(1)).apply(np.log).dropna()
        else:
            returns = df.pct_change().dropna()

        window = 20
        zscore = ((df[ticker] - df[ticker].rolling(window).mean()) / df[ticker].rolling(window).std()).dropna()

        latest_z = zscore.iloc[-1] if not zscore.empty else 0

        correlation = returns.corr().loc[ticker].drop(ticker).mean()
        iv_divergences = {etf: iv - 0.2 for etf in df.columns if etf != ticker}

        capital = comfortable_capital
        explanation = []
        if any(d > 0.1 for d in iv_divergences.values()):
            explanation.append("High IV divergence → reduce capital")
            capital *= 0.6
        if abs(latest_z) > 2:
            explanation.append(f"Z-score extreme ({latest_z:.2f}) → reduce capital")
            capital *= 0.7
        if correlation < 0.5:
            explanation.append(f"Weak correlation ({correlation:.2f}) → reduce capital")
            capital *= 0.8

        capital = max(min_capital, min(max_capital, capital))

        # Generate trading advice
        trading_advice = generate_trading_advice(iv_divergences, latest_z, correlation, capital, comfortable_capital)
        st.session_state.trading_advice = trading_advice

        # Prepare summary DataFrame for export
        summary_df = pd.DataFrame({
            "Metric": ["Market Price", f"Model Price ({pricing_model})", "Implied Volatility (IV)", "Suggested Capital", "Calculation Time (seconds)"],
            "Value": [f"{price_market:.2f}", f"{price:.2f}", f"{iv*100:.2f}%", f"{capital:.2f}", f"{calc_time:.4f}"]
        })

        export_df = pd.concat([greeks_df.rename(columns={"Greek": "Metric"}), summary_df], ignore_index=True)
        csv = prepare_export_csv(greeks_df, summary_df, trading_advice)
        st.session_state.export_csv = csv

        # Calculate profit vs capital plot
        capitals = list(range(int(min_capital), int(max_capital) + 1, 100))
        profits = []
        profits_ci_lower = []
        profits_ci_upper = []

# Add this function to calculate IV percentiles
def calculate_iv_percentile(ticker, current_iv, lookback_days=365):
    """Calculate how current IV compares to historical levels"""
    try:
        # Get historical volatility (using realized vol as proxy for IV)
        hist = yf.download(ticker, period=f"{lookback_days}d")["Close"]
        daily_returns = hist.pct_change().dropna()
        realized_vol = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate percentile (current IV vs historical realized vol)
        percentile = percentileofscore([realized_vol], current_iv)
        return percentile
    except Exception as e:
        st.warning(f"Could not calculate IV percentile: {e}")
        return None

# Add this function to plot IV crisis signals
def plot_iv_crisis_signal(ticker, current_iv):
    """Show historical IV spikes and current position"""
    try:
        # Get VIX data for market-wide volatility context
        vix = yf.download("^VIX", period="1y")["Close"]
        
        # Create plot
        fig = go.Figure()
        
        # Add VIX
        fig.add_trace(go.Scatter(
            x=vix.index,
            y=vix,
            name="VIX (Market Volatility)",
            line=dict(color="purple")
        ))
        
        # Add current IV level
        fig.add_hline(
            y=current_iv*100,  # Convert to VIX scale (VIX is %)
            line=dict(color="red", dash="dash"),
            annotation_text=f"Current IV: {current_iv*100:.1f}%",
            annotation_position="bottom right"
        )
        
        # Add crisis periods
        crisis_periods = {
            "COVID Crash (Mar 2020)": "2020-03-01",
            "Dec 2018 Selloff": "2018-12-01",
            "Feb 2018 Volmageddon": "2018-02-01"
        }
        
        for name, date in crisis_periods.items():
            fig.add_vline(
                x=pd.to_datetime(date),
                line=dict(color="orange", width=1),
                annotation_text=name,
                annotation_position="top left"
            )
        
        fig.update_layout(
            title=f"Volatility Context for {ticker}",
            yaxis_title="Volatility (%)",
            hovermode="x",
            template="plotly_white"
        )
        return fig
    except Exception as e:
        st.warning(f"Could not generate volatility plot: {e}")
        return None

# In your calculation section (after computing IV), add:
if calculate_clicked:
    try:
        # ... (your existing code)
        
        # Add IV percentile analysis
        iv_percentile = calculate_iv_percentile(ticker, iv)
        st.session_state.iv_percentile = iv_percentile
        
        # Add crisis signal plot
        iv_crisis_fig = plot_iv_crisis_signal(ticker, iv)
        st.session_state.iv_crisis_fig = iv_crisis_fig
        
        # Add warning to trading advice if IV is extreme
        if iv_percentile and iv_percentile > 90:
            st.session_state.trading_advice = pd.concat([
                st.session_state.trading_advice,
                pd.DataFrame({
                    "Advice": ["⚠️ Market Stress Warning"],
                    "Reason": [f"IV is in top {100-iv_percentile:.0f}% of historical levels - possible crisis ahead"]
                })
            ])
            
    except Exception as e:
        st.error(f"IV analysis failed: {e}")

# In your results display section, add this after the trading advice:
if st.session_state.calculation_done:
    # ... (your existing display code)
    
    if hasattr(st.session_state, 'iv_percentile') and st.session_state.iv_percentile:
        st.metric(
            label="Implied Volatility Percentile",
            value=f"{st.session_state.iv_percentile:.0f}th percentile",
            help="How current IV compares to 1-year history (higher = more extreme)"
        )
        
    if hasattr(st.session_state, 'iv_crisis_fig') and st.session_state.iv_crisis_fig:
        st.subheader("Volatility Context Analysis")
        st.plotly_chart(st.session_state.iv_crisis_fig)
        st.caption("Compare current IV (red line) to historical crisis periods")

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
                mean_profit = profits_samples.mean()
                std_profit = profits_samples.std()
                ci_lower = mean_profit - 1.96 * std_profit / np.sqrt(simulations)
                ci_upper = mean_profit + 1.96 * std_profit / np.sqrt(simulations)
                profits.append(mean_profit)
                profits_ci_lower.append(ci_lower)
                profits_ci_upper.append(ci_upper)
        else:
            for cap in capitals:
                contracts = int(cap / (price * 100)) if price > 0 else 0
                profit = contracts * 100 * (price * 1.05 - price)
                profits.append(profit)
                profits_ci_lower.append(None)
                profits_ci_upper.append(None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=capitals,
            y=profits,
            mode='lines+markers',
            name='Expected Profit',
            line=dict(color='blue'),
            hovertemplate='Capital: $%{x}<br>Profit: $%{y:.2f}<extra></extra>',
        ))

        if pricing_model == "Monte Carlo":
            fig.add_trace(go.Scatter(
                x=capitals + capitals[::-1],
                y=profits_ci_upper + profits_ci_lower[::-1],
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            ))

        fig.update_layout(
            title=f"Expected Profit vs Capital for {ticker} ({option_type} option)",
            xaxis_title="Capital ($)",
            yaxis_title="Expected Profit ($)",
            hovermode="x unified",
            template="plotly_white"
        )
        st.session_state.plot_fig = fig

        # Generate Black-Scholes sensitivities plot if using BS model
        if pricing_model == "Black-Scholes":
            bs_sensitivities_fig = plot_black_scholes_sensitivities(S, strike_price, T, risk_free_rate, iv, option_type)
            st.session_state.bs_sensitivities_fig = bs_sensitivities_fig

        # Generate PDF report
        try:
            pdf = generate_pdf_report(st.session_state.input_data, greeks_df, summary_df, trading_advice)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.session_state.export_pdf = pdf_bytes
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")
            st.session_state.export_pdf = None
        
        st.session_state.greeks_df = greeks_df
        st.session_state.summary_info = summary_df
        st.session_state.calculation_done = True
        st.success("Calculation done!")

    except Exception as e:
        st.error(f"Calculation failed: {e}")
        st.session_state.calculation_done = False


# Display results and export buttons
if st.session_state.calculation_done:
    st.subheader("Option Greeks")
    st.dataframe(st.session_state.greeks_df)

    st.subheader("Summary")
    st.dataframe(st.session_state.summary_info)

    st.subheader("Trading Advice")
    st.dataframe(st.session_state.trading_advice)
    
    with st.expander("Understanding the Advice"):
        st.markdown("""
        **How we determine trading advice:**
        
        - **IV Divergence**: Measures how different this option's implied volatility is from sector peers.  
          *> 0.1 suggests overpriced options*
          
        - **Z-Score**: Shows how far the stock price is from its recent average.  
          *> |2| suggests overextended move*
          
        - **Sector Correlation**: Measures how well sector ETFs hedge this stock.  
          *< 0.5 suggests poor hedging*
          
        - **Capital Adjustment**: Compares suggested capital to your comfortable amount.  
          *< 70% suggests high risk environment*
        """)

    st.plotly_chart(st.session_state.plot_fig)

    if pricing_model == "Black-Scholes" and st.session_state.bs_sensitivities_fig is not None:
        st.subheader("Black-Scholes Sensitivities")
        st.plotly_chart(st.session_state.bs_sensitivities_fig)
        
        with st.expander("Understanding the Sensitivities"):
            st.markdown("""
            **What these curves show:**
            
            - **Price vs Underlying**: How the option price changes as the stock price moves  
              *(Shows your option's leverage and moneyness)*
              
            - **Price vs Days to Expiry**: How time decay affects your option  
              *(Theta effect - note the accelerating decay as expiration approaches)*
              
            - **Price vs Volatility**: How implied volatility changes affect price  
              *(Vega effect - important for volatility trading strategies)*
            
            The dashed lines show your current position in each relationship.
            """)

    with export_csv_col:
        st.download_button(
            label="Download Data (CSV)",
            data=st.session_state.export_csv,
            file_name=f"{ticker}_option_data.csv",
            mime="text/csv"
        )

    with export_pdf_col:
        if st.session_state.export_pdf:
            st.download_button(
                label="Download Report (PDF)",
                data=st.session_state.export_pdf,
                file_name=f"{ticker}_option_report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("PDF report not available")

else:
    st.info("Click 'Calculate Profit & Advice' to generate data and export options.")

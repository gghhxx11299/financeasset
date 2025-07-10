import streamlit as st
from datetime import datetime
import yfinance as yf
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import requests
from bs4 import BeautifulSoup

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

# Pricing Models and Helpers
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
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
    return dict(Delta=delta, Gamma=gamma, Vega=vega, Theta=theta, Rho=rho)

def implied_volatility(option_market_price, S, K, T, r, option_type="call", tol=1e-5, max_iter=100):
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

def get_option_market_price(ticker, option_type, strike):
    stock = yf.Ticker(ticker)
    try:
        expiry_dates = stock.options
        for expiry in expiry_dates:
            chain = stock.option_chain(expiry)
            options = chain.calls if option_type == "call" else chain.puts
            options = options.dropna(subset=["lastPrice"])
            closest = options.iloc[(options['strike'] - strike).abs().argsort()[:1]]
            if not closest.empty and closest.iloc[0]['lastPrice'] > 0:
                return closest.iloc[0]['lastPrice'], expiry
    except:
        pass
    return None, None

def get_us_10yr_treasury_yield():
    try:
        url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yield"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 't-chart'})
        rows = table.find_all('tr')
        latest_row = rows[-1].find_all('td')
        yield_10yr = latest_row[5].text.strip()
        return float(yield_10yr) / 100
    except:
        return 0.03

# --- Streamlit App ---
st.title("Options Profit & Capital Advisor")

# Inputs
ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
strike = st.number_input("Desired Strike Price", min_value=0.0, value=150.0)
days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
risk_free = st.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=get_us_10yr_treasury_yield())
sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
return_type = st.selectbox("Return Type", ["Simple", "Log"])
comfortable = st.number_input("Comfortable Capital", min_value=0.0, value=1000.0)
max_cap = st.number_input("Max Capital", min_value=0.0, value=5000.0)
min_cap = st.number_input("Min Capital", min_value=0.0, value=500.0)

if st.button("Calculate"):
    try:
        S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
        T = days_to_expiry / 365

        market_price, expiry_date = get_option_market_price(ticker, option_type, strike)
        if market_price is None:
            st.error("Failed to get options data.")
            st.stop()

        iv = implied_volatility(market_price, S, strike, T, risk_free, option_type)
        if iv is None or iv <= 0:
            st.error("Could not compute implied volatility. Try a different strike.")
            st.stop()

        greeks = black_scholes_greeks(S, strike, T, risk_free, iv, option_type)

        etfs = SECTOR_MAP.get(sector, [])
        df = yf.download([ticker] + etfs, period="1mo", interval="1d")["Close"].dropna(axis=1, how="any")

        returns = (df / df.shift(1)).apply(np.log).dropna() if return_type == "Log" else df.pct_change().dropna()
        zscore = ((df[ticker] - df[ticker].rolling(3).mean()) / df[ticker].rolling(3).std()).dropna()
        latest_z = zscore.iloc[-1] if not zscore.empty else 0
        correlation = returns.corr().loc[ticker].drop(ticker).mean()
        iv_div = {etf: iv - 0.2 for etf in df.columns if etf != ticker}

        cap = comfortable
        reasons = []
        if any(v > 0.1 for v in iv_div.values()):
            reasons.append("High IV divergence")
            cap *= 0.6
        if abs(latest_z) > 2:
            reasons.append(f"Z-score = {latest_z:.2f}")
            cap *= 0.7
        if correlation < 0.5:
            reasons.append(f"Corr = {correlation:.2f}")
            cap *= 0.8

        cap = max(min_cap, min(max_cap, cap))

        st.markdown(f"**Market Price:** ${market_price:.2f}")
        st.markdown(f"**IV:** {iv*100:.2f}%")
        st.markdown(f"**Suggested Capital:** ${cap:.2f}")
        st.markdown("**Greeks:**")
        for k, v in greeks.items():
            st.write(f"- {k}: {v:.4f}")

        if reasons:
            st.markdown("**Adjustments due to:**")
            for r in reasons:
                st.write(f"- {r}")

        caps = list(range(int(min_cap), int(max_cap)+1, 100))
        profits = [int(c / (market_price * 100)) * 100 * (market_price - market_price * 0.95) for c in caps]

        fig, ax = plt.subplots()
        ax.plot(caps, profits)
        ax.set_title("Profit vs Capital")
        ax.set_xlabel("Capital ($)")
        ax.set_ylabel("Profit ($)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")


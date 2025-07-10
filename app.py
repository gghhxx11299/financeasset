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

# Pricing Models
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
    try:
        url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yield"
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 't-chart'})
        rows = table.find_all('tr')
        latest_row = rows[-1].find_all('td')
        yield_10yr = latest_row[5].text.strip()

        return float(yield_10yr) / 100
    except Exception:
        return 0.025  # fallback 2.5%

st.title("Options Profit & Capital Advisor")

# Inputs
ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
strike_price = st.number_input("Strike Price", min_value=0.0, value=150.0)
expiry_date = st.text_input("Option Expiry (YYYY-MM-DD)", value="")
days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
risk_free_rate = st.number_input("Risk-Free Rate (e.g. 0.025)", min_value=0.0, max_value=1.0, value=get_us_10yr_treasury_yield())
sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
return_type = st.selectbox("Return Type", ["Simple", "Log"])
comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
pricing_model = st.selectbox("Pricing Model", ["Black-Scholes"])

if st.button("Calculate Profit & Advice"):
    if not expiry_date:
        st.error("Please enter the option expiry date (YYYY-MM-DD).")
    else:
        try:
            T = days_to_expiry / 365
            S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

            # Get option price
            price = get_option_market_price(ticker, option_type, strike_price, expiry_date)
            if price is None:
                st.error("Failed to fetch option market price.")
                st.stop()

            iv = implied_volatility(price, S, strike_price, T, risk_free_rate, option_type)
            if iv is None:
                st.error("Could not calculate implied volatility.")
                st.stop()

            greeks = black_scholes_greeks(S, strike_price, T, risk_free_rate, iv, option_type)
            greeks_text = (
                f"Delta: {greeks['Delta']:.4f} | "
                f"Gamma: {greeks['Gamma']:.4f} | "
                f"Vega: {greeks['Vega']:.4f} | "
                f"Theta: {greeks['Theta']:.4f} | "
                f"Rho: {greeks['Rho']:.4f}"
            )

            # Sector ETFs data
            etfs = SECTOR_MAP.get(sector, [])
            symbols = [ticker] + etfs
            df = yf.download(symbols, period="1mo", interval="1d")["Close"].dropna(axis=1, how="any")

            # Calculate returns
            if return_type == "Log":
                returns = (df / df.shift(1)).apply(np.log).dropna()
            else:
                returns = df.pct_change(fill_method=None).dropna()

            zscore = ((df[ticker] - df[ticker].rolling(3).mean()) / df[ticker].rolling(3).std()).dropna()
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

            st.write(f"### Market Price: ${price:.2f}")
            st.write(f"### Implied Volatility: {iv*100:.2f}%")
            st.write(f"### Greeks: {greeks_text}")
            st.write(f"### Suggested Capital: ${capital:.2f}")
            if explanation:
                st.write("### Explanation:")
                for line in explanation:
                    st.write(f"- {line}")

            # Profit vs Capital plot
            capitals = list(range(int(min_capital), int(max_capital) + 1, 100))
            profits = []
            for cap in capitals:
                contracts = int(cap / (price * 100))
                profit = contracts * 100 * (price - price * 0.95)  # example profit calc
                profits.append(profit)

            fig, ax = plt.subplots()
            ax.plot(capitals, profits, label="Profit")
            ax.set_xlabel("Capital ($)")
            ax.set_ylabel("Profit ($)")
            ax.set_title("Profit vs Capital")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")

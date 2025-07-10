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

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree_price(S, K, T, r, sigma, steps, option_type="call"):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if option_type == "call":
        values = [max(0, p - K) for p in prices]
    else:
        values = [max(0, K - p) for p in prices]

    for i in range(steps - 1, -1, -1):
        values = [math.exp(-r * dt) * (q * values[j + 1] + (1 - q) * values[j]) for j in range(i + 1)]

    return values[0]

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
        return 0.025

st.title("Options Profit & Capital Advisor")

# Inputs
ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
strike_price = st.number_input("Desired Strike Price", min_value=0.0, value=150.0)
days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
risk_free_rate = st.number_input("Risk-Free Rate (e.g. 0.025)", min_value=0.0, max_value=1.0, value=get_us_10yr_treasury_yield())
sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
return_type = st.selectbox("Return Type", ["Simple", "Log"])
comfortable_capital = st.number_input("Comfortable Capital ($)", min_value=0.0, value=1000.0)
max_capital = st.number_input("Max Capital ($)", min_value=0.0, value=5000.0)
min_capital = st.number_input("Min Capital ($)", min_value=0.0, value=500.0)
pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree"])

if st.button("Calculate Profit & Advice"):
    try:
        T = days_to_expiry / 365
        stock_data = yf.Ticker(ticker).history(period="1d")
        if stock_data.empty:
            st.error("Could not fetch stock price.")
            st.stop()
        S = stock_data["Close"].iloc[-1]

        expiry_date = None
        for date in yf.Ticker(ticker).options:
            dt = datetime.strptime(date, "%Y-%m-%d")
            if abs((dt - datetime.now()).days - days_to_expiry) <= 5:
                expiry_date = date
                break
        if expiry_date is None:
            st.error("No matching expiry date found.")
            st.stop()

        price = get_option_market_price(ticker, option_type, strike_price, expiry_date)
        if price is None:
            st.error("Failed to fetch option market price.")
            st.stop()

        if pricing_model == "Black-Scholes":
            iv = implied_volatility(price, S, strike_price, T, risk_free_rate, option_type)
            if iv is None:
                st.error("Could not compute implied volatility.")
                st.stop()
            model_price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
        else:
            iv = 0.25  # assume initial volatility
            model_price = binomial_tree_price(S, strike_price, T, risk_free_rate, iv, option_type=option_type, steps=100)

        st.write(f"### Market Price: ${price:.2f}")
        st.write(f"### Implied Volatility (IV): {iv*100:.2f}%" if pricing_model == "Black-Scholes" else f"### Binomial Tree Price: ${model_price:.2f}")

        # Sector data
        symbols = [ticker] + SECTOR_MAP.get(sector, [])
        df = yf.download(symbols, period="1mo", interval="1d")["Close"].dropna(axis=1)

        if return_type == "Log":
            returns = np.log(df / df.shift(1)).dropna()
        else:
            returns = df.pct_change().dropna()

        correlation = returns.corr().loc[ticker].drop(ticker).mean()
        zscore = ((df[ticker] - df[ticker].rolling(3).mean()) / df[ticker].rolling(3).std()).dropna()
        latest_z = zscore.iloc[-1] if not zscore.empty else 0
        iv_divergence = abs(iv - 0.2)

        capital = comfortable_capital
        advice = []
        if iv_divergence > 0.1:
            capital *= 0.7
            advice.append("High IV divergence → reduce capital")
        if abs(latest_z) > 2:
            capital *= 0.7
            advice.append(f"Z-score extreme ({latest_z:.2f}) → reduce capital")
        if correlation < 0.5:
            capital *= 0.8
            advice.append(f"Weak correlation ({correlation:.2f}) → reduce capital")

        capital = max(min_capital, min(max_capital, capital))

        st.write(f"### Suggested Capital: ${capital:.2f}")
        st.write("### Advice:")
        for a in advice or ["- No significant adjustment needed."]:
            st.write(f"- {a}")

        # Plot
        capitals = list(range(int(min_capital), int(max_capital)+1, 100))
        profits = [int(c / (price * 100)) * 100 * (price - price * 0.95) for c in capitals]
        fig, ax = plt.subplots()
        ax.plot(capitals, profits)
        ax.set_title("Profit vs Capital")
        ax.set_xlabel("Capital ($)")
        ax.set_ylabel("Profit ($)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

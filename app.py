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

# --- Pricing Models ---
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def binomial_tree_price(S, K, T, r, sigma, option_type="call", steps=100):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    prices = np.zeros((steps + 1, steps + 1))
    prices[0, 0] = S
    for i in range(1, steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)

    option = np.zeros((steps + 1))
    for j in range(steps + 1):
        option[j] = max(0, prices[j, steps] - K) if option_type == "call" else max(0, K - prices[j, steps])

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option[j] = math.exp(-r * dt) * (p * option[j] + (1 - p) * option[j + 1])

    return option[0]

def monte_carlo_price(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    np.random.seed(0)
    dt = T
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z)
    payoffs = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoffs)

def implied_volatility(market_price, S, K, T, r, option_type="call", tol=1e-5, max_iter=100):
    low, high = 0.0001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return None

def get_option_market_price(ticker, option_type, strike, expiry_date):
    try:
        opt = yf.Ticker(ticker)
        if expiry_date not in opt.options:
            return None
        chain = opt.option_chain(expiry_date)
        data = chain.calls if option_type == "call" else chain.puts
        row = data[data['strike'] == strike]
        return None if row.empty else row.iloc[0]['lastPrice']
    except:
        return None

def get_us_10yr_treasury_yield():
    try:
        url = "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yield"
        soup = BeautifulSoup(requests.get(url, timeout=5).text, 'html.parser')
        yield_10yr = soup.find_all('table', {'class': 't-chart'})[0].find_all('tr')[-1].find_all('td')[5].text.strip()
        return float(yield_10yr) / 100
    except:
        return 0.025

# --- Streamlit App ---
st.title("Options Profit & Capital Advisor")

ticker = st.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])
strike_price = st.number_input("Desired Strike Price", min_value=0.0, value=150.0)
days_to_expiry = st.number_input("Days to Expiry", min_value=1, max_value=365, value=30)
risk_free_rate = st.number_input("Risk-Free Rate (e.g. 0.025)", min_value=0.0, max_value=1.0, value=get_us_10yr_treasury_yield())
sector = st.selectbox("Sector", list(SECTOR_MAP.keys()))
return_type = st.selectbox("Return Type", ["Simple", "Log"])
comfortable_capital = st.number_input("Comfortable Capital ($)", value=1000.0)
max_capital = st.number_input("Max Capital ($)", value=5000.0)
min_capital = st.number_input("Min Capital ($)", value=500.0)
pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

if st.button("Calculate Profit & Advice"):
    try:
        T = days_to_expiry / 365
        S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

        expiries = yf.Ticker(ticker).options
        expiry_date = next((date for date in expiries if abs((datetime.strptime(date, "%Y-%m-%d") - datetime.now()).days - days_to_expiry) <= 5), None)
        if not expiry_date:
            st.error("No matching expiry date found.")
            st.stop()

        market_price = get_option_market_price(ticker, option_type, strike_price, expiry_date)
        if not market_price:
            st.error("Failed to fetch option market price.")
            st.stop()

        iv = implied_volatility(market_price, S, strike_price, T, risk_free_rate, option_type)
        if not iv:
            st.error("Could not compute implied volatility.")
            st.stop()

        if pricing_model == "Black-Scholes":
            model_price = black_scholes_price(S, strike_price, T, risk_free_rate, iv, option_type)
        elif pricing_model == "Binomial Tree":
            model_price = binomial_tree_price(S, strike_price, T, risk_free_rate, iv, option_type)
        else:
            model_price = monte_carlo_price(S, strike_price, T, risk_free_rate, iv, option_type)

        df = yf.download([ticker] + SECTOR_MAP[sector], period="1mo", interval="1d")["Close"].dropna()
        returns = np.log(df / df.shift(1)).dropna() if return_type == "Log" else df.pct_change().dropna()
        zscore = ((df[ticker] - df[ticker].rolling(3).mean()) / df[ticker].rolling(3).std()).dropna()
        latest_z = zscore.iloc[-1] if not zscore.empty else 0
        correlation = returns.corr().loc[ticker].drop(ticker).mean()

        capital = comfortable_capital
        explanation = []
        if iv > 0.3:
            explanation.append("High IV → reduce capital")
            capital *= 0.6
        if abs(latest_z) > 2:
            explanation.append(f"Z-score extreme ({latest_z:.2f}) → reduce capital")
            capital *= 0.7
        if correlation < 0.5:
            explanation.append(f"Weak correlation ({correlation:.2f}) → reduce capital")
            capital *= 0.8
        capital = max(min_capital, min(max_capital, capital))

        st.write(f"### Model Price: ${model_price:.2f}")
        st.write(f"### Implied Volatility: {iv*100:.2f}%")
        st.write(f"### Suggested Capital: ${capital:.2f}")
        st.write("### Advice:")
        for line in explanation:
            st.write(f"- {line}")

        capitals = list(range(int(min_capital), int(max_capital) + 1, 100))
        profits = [int(c / (model_price * 100)) * 100 * (model_price * 0.05) for c in capitals]

        fig, ax = plt.subplots()
        ax.plot(capitals, profits)
        ax.set_xlabel("Capital ($)")
        ax.set_ylabel("Profit ($)")
        ax.set_title("Profit vs Capital")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")


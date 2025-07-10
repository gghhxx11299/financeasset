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

def binomial_tree_price(S, K, T, r, sigma, steps=100, option_type="call"):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)

    prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    if option_type == "call":
        option_values = [max(price - K, 0) for price in prices]
    else:
        option_values = [max(K - price, 0) for price in prices]

    for i in range(steps - 1, -1, -1):
        option_values = [
            math.exp(-r * dt) * (q * option_values[j + 1] + (1 - q) * option_values[j])
            for j in range(i + 1)
        ]
    return option_values[0]

def monte_carlo_price(S, K, T, r, sigma, simulations=10000, option_type="call"):
    np.random.seed(0)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(simulations))
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    return math.exp(-r * T) * np.mean(payoffs)

# --- UI ---
st.title("Options Pricing Model Comparison")

ticker = st.text_input("Ticker", value="AAPL")
option_type = st.selectbox("Option Type", ["call", "put"])
K = st.number_input("Strike Price", value=150.0)
T_days = st.number_input("Days to Expiry", value=30)
r = st.number_input("Risk-Free Rate", value=0.03)
sigma = st.number_input("Implied Volatility (%)", value=20.0) / 100

T = T_days / 365.0
S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

if st.button("Compare All Models"):
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
    bt_price = binomial_tree_price(S, K, T, r, sigma, option_type=option_type)
    mc_price = monte_carlo_price(S, K, T, r, sigma, option_type=option_type)

    st.subheader("Pricing Results")
    st.write(f"Black-Scholes Price: ${bs_price:.2f}")
    st.write(f"Binomial Tree Price: ${bt_price:.2f}")
    st.write(f"Monte Carlo Price: ${mc_price:.2f}")

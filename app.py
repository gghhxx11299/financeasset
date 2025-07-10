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

def binomial_tree_price(S, K, T, r, sigma, steps=100, option_type="call"):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    q = (math.exp(r * dt) - d) / (u - d)
    
    prices = [S * u**j * d**(steps - j) for j in range(steps + 1)]
    if option_type == "call":
        values = [max(p - K, 0) for p in prices]
    else:
        values = [max(K - p, 0) for p in prices]

    for i in range(steps - 1, -1, -1):
        values = [math.exp(-r * dt) * (q * values[j + 1] + (1 - q) * values[j]) for j in range(i + 1)]
    return values[0]

def monte_carlo_price(S, K, T, r, sigma, option_type="call", simulations=10000):
    np.random.seed(42)
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    return math.exp(-r * T) * np.mean(payoffs)

# --- UI ---
st.title("Options Pricing - Compare Models")

S = st.number_input("Stock Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Expiry in Years (T)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.03)
sigma = st.number_input("Volatility (sigma)", value=0.2)
option_type = st.selectbox("Option Type", ["call", "put"])

model_choice = st.selectbox("Select Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo", "Compare All"])

if st.button("Calculate Option Price"):
    try:
        if model_choice == "Black-Scholes":
            price = black_scholes_price(S, K, T, r, sigma, option_type)
            st.success(f"Black-Scholes Price: ${price:.4f}")

        elif model_choice == "Binomial Tree":
            price = binomial_tree_price(S, K, T, r, sigma, option_type=option_type)
            st.success(f"Binomial Tree Price: ${price:.4f}")

        elif model_choice == "Monte Carlo":
            price = monte_carlo_price(S, K, T, r, sigma, option_type=option_type)
            st.success(f"Monte Carlo Price: ${price:.4f}")

        elif model_choice == "Compare All":
            bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
            bt_price = binomial_tree_price(S, K, T, r, sigma, option_type=option_type)
            mc_price = monte_carlo_price(S, K, T, r, sigma, option_type=option_type)
            
            st.subheader("Pricing Model Comparison")
            st.write(f"- Black-Scholes: ${bs_price:.4f}")
            st.write(f"- Binomial Tree: ${bt_price:.4f}")
            st.write(f"- Monte Carlo: ${mc_price:.4f}")

    except Exception as e:
        st.error(f"Error in calculation: {e}")

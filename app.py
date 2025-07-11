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
import time
import io
import base64
import plotly.graph_objs as go

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

def binomial_tree_price(S, K, T, r, sigma, option_type="call", steps=100):
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
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 't-chart'})
        if table is None:
            st.warning("Could not find Treasury yield table on the page.")
            return 0.025

        rows = table.find_all('tr')
        latest_row = rows[-1].find_all('td')
        yield_10yr = latest_row[5].text.strip()

        return float(yield_10yr) / 100
    except Exception as e:
        st.warning(f"Error fetching Treasury yield: {e}")
        return 0.025  # fallback 2.5%

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
pricing_model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])

if st.button("Calculate Profit & Advice"):
    try:
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
            st.stop()

        price_market = get_option_market_price(ticker, option_type, strike_price, expiry_date)
        if price_market is None:
            st.error("Failed to fetch option market price. Try a closer-to-the-money strike.")
            st.stop()

        iv = implied_volatility(price_market, S, strike_price, T, risk_free_rate, option_type)
        if iv is None:
            st.error("Could not compute implied volatility. Try a closer-to-the-money strike.")
            st.stop()

        greeks = black_scholes_greeks(S, strike_price, T, risk_free_rate, iv, option_type)
        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [greeks["Delta"], greeks["Gamma"], greeks["Vega"], greeks["Theta"], greeks["Rho"]]
                      })
        greeks_df["Value"] = greeks_df["Value"].map(lambda x: f"{x:.4f}")

        # Display chosen pricing model
        st.markdown(f"## Pricing Model Selected: **{pricing_model}**")

        # Greeks table
        st.write("### Greeks")
        st.table(greeks_df)

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

        # Pricing summary output with better formatting
        st.markdown(
            f"""
            ### Pricing Summary

            - **Market Price:** `${price_market:.2f}`
            - **Model Price ({pricing_model}):** `${price:.2f}`
            - **Implied Volatility (IV):** `{iv*100:.2f}%`
            - **Suggested Capital:** `${capital:.2f}`
            - ⏱️ **Calculation Time:** `{calc_time:.4f} seconds`
            """
        )

        # Advice section
        st.write("### Advice")
        if explanation:
            for line in explanation:
                st.write(f"- {line}")
        else:
            st.write("- No significant adjustments. Capital allocation looks good.")

        # Reason section explaining suggested capital adjustments
        st.write("### Reason for Suggested Capital")
        if explanation:
            reasons = "\n".join(explanation)
            st.markdown(f"""
            The suggested capital is adjusted due to the following factors observed in the market data and analysis:
            {reasons}
            """)
        else:
            st.markdown("The suggested capital is based on your comfortable capital without any adjustments because market indicators show stable conditions.")

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
                fillcolor='rgba(173,216,230,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            ))

        fig.update_layout(
            title='Profit vs Capital',
            xaxis_title='Capital ($)',
            yaxis_title='Profit ($)',
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(showgrid=True, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridcolor='LightGray'),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Export Greeks & Summary CSV ---
        summary_df = pd.DataFrame({
            "Metric": ["Market Price", f"Model Price ({pricing_model})", "Implied Volatility (IV)", "Suggested Capital", "Calculation Time (seconds)"],
            "Value": [f"{price_market:.2f}", f"{price:.2f}", f"{iv*100:.2f}%", f"{capital:.2f}", f"{calc_time:.4f}"]
        })
        export_df = pd.concat([greeks_df.rename(columns={"Greek": "Metric"}), summary_df], ignore_index=True)
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Greeks & Summary CSV",
            data=csv,
            file_name=f"{ticker}_option_analysis.csv",
            mime="text/csv"
        )

        # --- Export Profit vs Capital Plot PNG ---
        png_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download Profit vs Capital Plot (PNG)",
            data=png_bytes,
            file_name=f"{ticker}_profit_vs_capital.png",
            mime="image/png"
        )

        # --- Implied Volatility Surface Plot ---
        ticker_obj = yf.Ticker(ticker)
        expiries = ticker_obj.options[:3]  # next 3 expiry dates

        iv_surface_data = []
        strikes_set = set()
        for exp in expiries:
            try:
                opt_chain = ticker_obj.option_chain(exp)
                options_df = opt_chain.calls if option_type == "call" else opt_chain.puts
                strikes = options_df['strike'].values
                market_prices = options_df['lastPrice'].values
                T_exp = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365
                for K_opt, price_opt in zip(strikes, market_prices):
                    if price_opt > 0 and T_exp > 0:
                        iv_opt = implied_volatility(price_opt, S, K_opt, T_exp, risk_free_rate, option_type)
                        if iv_opt is not None:
                            iv_surface_data.append((exp, K_opt, iv_opt))
                            strikes_set.add(K_opt)
            except Exception:
                continue

        if iv_surface_data:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig3d = plt.figure(figsize=(10, 6))
            ax = fig3d.add_subplot(111, projection='3d')

            expiries_unique = sorted(set(x[0] for x in iv_surface_data))
            expiry_nums = list(range(len(expiries_unique)))
            expiry_map = dict(zip(expiries_unique, expiry_nums))

            xs = [expiry_map[x[0]] for x in iv_surface_data]
            ys = [x[1] for x in iv_surface_data]
            zs = [x[2] for x in iv_surface_data]

            sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis')

            ax.set_xticks(expiry_nums)
            ax.set_xticklabels([x[0][5:] for x in expiries_unique], rotation=20)
            ax.set_xlabel('Expiry Date (YY-MM)')
            ax.set_ylabel('Strike Price')
            ax.set_zlabel('Implied Volatility')
            plt.colorbar(sc, label='IV')
            plt.title(f"Implied Volatility Surface for {ticker}")

            st.pyplot(fig3d)

            buf = io.BytesIO()
            fig3d.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Implied Volatility Surface Plot (PNG)",
                data=buf,
                file_name=f"{ticker}_iv_surface.png",
                mime="image/png"
            )

        st.write(
            "_Disclaimer: This tool provides estimates and is not financial advice. "
            "Always perform your own due diligence before investing._"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")

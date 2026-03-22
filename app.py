import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from data_fetcher import get_stock_data, get_nifty_data, compute_returns, compute_annual_stats
from optimizer import monte_carlo_simulation, max_sharpe_portfolio, min_volatility_portfolio, efficient_frontier_points
from risk_metrics import compute_all_metrics, cumulative_return_series, portfolio_daily_returns
from charts import (
    efficient_frontier_chart, weights_pie_chart,
    correlation_heatmap, covariance_heatmap,
    cumulative_return_chart, rolling_sharpe_chart,
    drawdown_chart, var_histogram,
)
from report_exporter import build_excel_report

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Portfolio Optimizer")

    st.markdown("### 🔍 Enter Stocks")
    user_input = st.text_input(
        "Enter tickers (comma separated)",
        placeholder="infy, icicibank, reliance"
    )

    st.markdown("### ⚙️ Settings")
    period = st.select_slider(
        "Historical period",
        options=["1y", "2y", "3y", "5y"],
        value="3y",
    )

    num_portfolios = st.select_slider(
        "Monte Carlo portfolios",
        options=[1000, 5000, 10000],
        value=5000,
    )

    run_btn = st.button("🚀 Run", use_container_width=True)

# ─── Convert user input → Yahoo tickers ─────────────────────
def parse_tickers(user_input):
    tickers = []
    for t in user_input.split(","):
        t = t.strip().upper()
        if t:
            if not t.endswith(".NS"):
                t = t + ".NS"
            tickers.append(t)
    return tickers

# ─── Main ─────────────────────────────────────────────
st.title("Portfolio Optimizer")

if not user_input:
    st.info("Enter stock tickers in sidebar to begin.")
    st.stop()

tickers = parse_tickers(user_input)

if len(tickers) < 2:
    st.warning("Enter at least 2 stocks.")
    st.stop()

if run_btn:
    with st.spinner("Fetching data..."):
        prices = get_stock_data(tickers, period)
        nifty = get_nifty_data(period)

    if prices.empty:
        st.error("Invalid tickers or no data found.")
        st.stop()

    prices = prices.dropna(axis=1, how="all")
    tickers = list(prices.columns)

    returns = compute_returns(prices)
    mean_ret = returns.mean()
    cov = returns.cov()
    nifty_ret = compute_returns(nifty.to_frame()).iloc[:, 0]

   
    allow_short = False  #no short selling

    with st.spinner("Optimizing portfolio..."):
        sim_df = monte_carlo_simulation(mean_ret, cov, num_portfolios)
        ms_w, ms_r, ms_v, ms_s = max_sharpe_portfolio(mean_ret, cov, allow_short)
        mv_w, mv_r, mv_v, mv_s = min_volatility_portfolio(mean_ret, cov, allow_short)
        frontier_df = efficient_frontier_points(mean_ret, cov)

    # ─── Metrics ─────────────────────────
    ms_metrics = compute_all_metrics(returns, ms_w, nifty_ret)
    ms_cum = cumulative_return_series(returns, ms_w)
    nifty_cum = (1 + nifty_ret.reindex(ms_cum.index).fillna(0)).cumprod()
    ms_daily = portfolio_daily_returns(returns, ms_w)

    # ─── Output ─────────────────────────
    st.subheader("📊 Efficient Frontier")

    plot_sim = sim_df
    plot_frontier = frontier_df

    st.plotly_chart(
        efficient_frontier_chart(
            plot_sim,
            plot_frontier,
            {"ret": ms_r, "vol": ms_v, "sharpe": ms_s},
            {"ret": mv_r, "vol": mv_v, "sharpe": mv_s},
        ),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🥧 Portfolio Weights")
    st.plotly_chart(
        weights_pie_chart(ms_w, tickers, "Max Sharpe Portfolio"),
        use_container_width=True
    )

    st.subheader("📈 Performance")
    st.plotly_chart(
        cumulative_return_chart({
            "Portfolio": ms_cum,
            "NIFTY 50": nifty_cum,
        }),
        use_container_width=True
    )

    st.subheader("📋 Max Sharpe Portfolio Details")

    weights_df = pd.DataFrame({
        "Stock": tickers,
        "Weight": ms_w
    })

    weights_df["Weight"] = weights_df["Weight"].round(4)

    summary_df = pd.DataFrame({
        "Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
        "Value": [ms_r, ms_v, ms_s]
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Weights")
        st.dataframe(weights_df)

    with col2:
        st.write("### Summary")
        st.dataframe(summary_df)

    st.subheader("⚠️ Risk Metrics")
    st.dataframe(pd.DataFrame(ms_metrics.items(), columns=["Metric", "Value"]))

    st.subheader("📥 Export")

    excel_file = build_excel_report(
        tickers, prices, returns,
        ms_w, mv_w,
        ms_metrics, ms_metrics,
        sim_df
    )

    st.download_button(
        label="Download Excel",
        data=excel_file,
        file_name="portfolio_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

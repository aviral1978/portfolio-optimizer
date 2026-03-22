# 📈 Portfolio Optimizer

A Python-based portfolio optimization tool that enables users to construct, analyze, and optimize equity portfolios using quantitative finance techniques.

---

## 🚀 Features

- **Custom Stock Selection**  
  Input your own stock tickers (e.g., `INFY`, `ICICIBANK`, `RELIANCE`)

- **Monte Carlo Portfolio Simulation**  
  Generates thousands of random portfolio combinations to explore risk-return tradeoffs

- **Efficient Frontier Visualization**  
  Plots the set of optimal portfolios for different risk levels

- **Max Sharpe Ratio Portfolio**  
  Identifies the portfolio with the highest risk-adjusted return

- **Risk Metrics Calculation**  
  - Expected Return  
  - Volatility (Standard Deviation)  
  - Sharpe Ratio  

- **Interactive Dashboard**  
  Built using Streamlit for easy exploration and visualization

- **Excel Export**  
  Download portfolio weights, returns, and risk metrics as a structured Excel report

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Plotly  
- yfinance  

---

## ⚙️ Methodology

The optimizer is based on **Modern Portfolio Theory (MPT)**:

- Random portfolio weights are generated using Monte Carlo simulation  
- Portfolio return and volatility are computed for each simulation  
- The Efficient Frontier is derived from simulated portfolios  
- The optimal portfolio is selected using the **maximum Sharpe ratio**

---

## 📌 Key Assumptions

- **Risk-Free Rate:**  
  Fixed at 6.5% (approx. Indian government bond yield; configurable in code)

- **Return Calculation:**  
  - Based on historical daily returns  
  - Annualized using ~252 trading days  

- **Volatility Estimation:**  
  - Standard deviation of returns  
  - Assumes normally distributed returns  

- **Market Assumptions:**  
  - Historical data is a proxy for future performance  
  - No structural breaks or regime shifts considered  

- **Transaction Costs & Taxes:**  
  Ignored for simplicity

- **Liquidity:**  
  All assets assumed to be perfectly liquid  

- **Short Selling:**  
  Not allowed (weights constrained between 0 and 1)  

- **Portfolio Constraint:**  
  Fully invested portfolio (weights sum to 1)  

---

## 📊 Output

- Efficient Frontier plot  
- Risk-return scatter of simulated portfolios  
- Maximum Sharpe ratio portfolio allocation  
- Downloadable Excel report including:
  - Portfolio weights  
  - Expected returns  
  - Volatility  
  - Sharpe ratios  

---

## 📂 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

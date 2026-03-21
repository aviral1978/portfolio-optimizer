# 📈 Portfolio Optimizer — NSE India

A professional Streamlit web app for optimising Indian stock portfolios using
**Modern Portfolio Theory**, Monte Carlo simulation, and advanced risk analytics.

---

## 🗂 Project Structure

```
portfolio_optimizer/
├── app.py                # Main Streamlit application
├── data_fetcher.py       # Yahoo Finance data download & return computation
├── optimizer.py          # Monte Carlo, Max-Sharpe, Min-Volatility, Efficient Frontier
├── risk_metrics.py       # VaR, CVaR, Sortino, Jensen's Alpha, Beta, Drawdown
├── charts.py             # All Plotly & Matplotlib/Seaborn chart builders
├── report_exporter.py    # Styled multi-sheet Excel report generator
├── requirements.txt      # Python dependencies
└── .vscode/
    ├── launch.json       # One-click Run in VS Code
    └── settings.json     # Linting, formatting, interpreter settings
```

---

## 🚀 Quick Start

### 1 — Clone / open in VS Code
Open the `portfolio_optimizer/` folder in VS Code.

### 2 — Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
streamlit run app.py
```
The app opens automatically at **http://localhost:8501**

> **VS Code shortcut:** Press `F5` (or use *Run > Start Debugging*) to launch directly
> via the pre-configured `.vscode/launch.json`.

---

## 🎛 Features

| Tab | What you get |
|---|---|
| **Efficient Frontier** | Monte Carlo scatter + frontier curve, Max-Sharpe ⭐ & Min-Vol ◆ markers |
| **Weights** | Donut pie charts + weight comparison table |
| **Correlation & Covariance** | Interactive seaborn heatmaps (with annualisation) |
| **Performance** | Cumulative return vs NIFTY 50, rolling Sharpe, drawdown chart |
| **Risk Metrics** | Return distribution, VaR histogram, full metrics table |
| **Data** | Raw price & return tables, per-stock annual stats |
| **Export** | Download a styled 6-sheet Excel workbook |

### Risk metrics computed
- Value at Risk (VaR) — 95% and 99% — Historical method
- Conditional VaR / Expected Shortfall — 95%
- Sortino Ratio
- Jensen's Alpha (vs NIFTY 50)
- Maximum Drawdown
- Portfolio Beta (vs NIFTY 50)
- Sharpe Ratio (risk-free rate: 6.5%)

---

## 📦 Key Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `yfinance` | Market data (NSE via Yahoo Finance) |
| `scipy` | Portfolio optimisation (SLSQP) |
| `plotly` | Interactive charts |
| `seaborn / matplotlib` | Heatmaps |
| `openpyxl` | Styled Excel export |

---

## ⚙️ Configuration

All user-facing settings live in the **sidebar**:

| Setting | Default | Notes |
|---|---|---|
| Stocks | 5 defaults | Choose 3–15 NSE stocks |
| Historical period | 3 years | 1y / 2y / 3y / 5y |
| Monte Carlo portfolios | 10,000 | More = slower but smoother frontier |
| Allow short selling | Off | Enables negative weights |

---

## 🛠 Extending the Project

- **Add stocks:** Edit the `NSE_STOCKS` dictionary in `data_fetcher.py`
- **Change risk-free rate:** Update `RISK_FREE_RATE` in `optimizer.py` and `risk_metrics.py`
- **Add factor models (Fama-French):** Extend `risk_metrics.py`
- **Add sector constraints:** Extend the `_optimize()` bounds in `optimizer.py`

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. Past performance is not indicative
of future results. This is not financial advice. Always consult a SEBI-registered
investment advisor before making investment decisions.

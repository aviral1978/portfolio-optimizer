"""
data_fetcher.py
Handles downloading stock data from Yahoo Finance for NSE-listed stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np


# Popular NSE stocks (ticker format: SYMBOL.NS)
NSE_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "SBI": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Wipro": "WIPRO.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Axis Bank": "AXISBANK.NS",
    "Nestle India": "NESTLEIND.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Larsen & Toubro": "LT.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Shree Cement": "SHREECEM.NS",
    "Grasim Industries": "GRASIM.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Britannia": "BRITANNIA.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "Hindalco": "HINDALCO.NS",
}

NIFTY50_TICKER = "^NSEI"


def get_stock_data(tickers: list, period: str = "3y") -> pd.DataFrame:
    """
    Download adjusted close prices for given tickers.

    Args:
        tickers: List of Yahoo Finance ticker symbols (e.g., ["RELIANCE.NS", "TCS.NS"])
        period: Data period string (e.g., "1y", "2y", "3y", "5y")

    Returns:
        DataFrame of adjusted close prices, columns = tickers
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if len(tickers) == 1:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    else:
        prices = raw["Close"].copy()

    prices.dropna(how="all", inplace=True)
    # Forward-fill minor gaps (weekends / holidays)
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    return prices


def get_nifty_data(period: str = "3y") -> pd.Series:
    """Download NIFTY 50 index data as a Series."""
    raw = yf.download(NIFTY50_TICKER, period=period, auto_adjust=True, progress=False)
    nifty = raw["Close"].squeeze()
    nifty.ffill(inplace=True)
    nifty.dropna(inplace=True)
    return nifty


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price DataFrame."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_annual_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised mean return and volatility for each asset.

    Returns:
        DataFrame with columns ['Annual Return', 'Annual Volatility']
    """
    trading_days = 252
    ann_return = returns.mean() * trading_days
    ann_vol = returns.std() * np.sqrt(trading_days)
    stats = pd.DataFrame({"Annual Return": ann_return, "Annual Volatility": ann_vol})
    stats.index.name = "Ticker"
    return stats

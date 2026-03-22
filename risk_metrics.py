import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065  # annualised


def portfolio_daily_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Weighted daily return series for the portfolio."""
    return (returns * weights).sum(axis=1)


def value_at_risk(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at given confidence level (as a positive loss figure).
    E.g. VaR95 = 0.02 means 'worst 5% of days lose at least 2%'.
    """
    return float(-np.percentile(port_returns, (1 - confidence) * 100))


def conditional_var(port_returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall / CVaR (average loss beyond VaR)."""
    var = value_at_risk(port_returns, confidence)
    tail = port_returns[port_returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


def sortino_ratio(port_returns: pd.Series) -> float:
    """
    Annualised Sortino Ratio using downside deviation relative to 0.
    """
    daily_rf = RISK_FREE_RATE / TRADING_DAYS
    excess = port_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)
    ann_excess = excess.mean() * TRADING_DAYS
    return ann_excess / downside_std if downside_std > 0 else np.nan


def maximum_drawdown(port_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the cumulative return series."""
    cum = (1 + port_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return float(drawdown.min())


def portfolio_beta(
    port_returns: pd.Series, market_returns: pd.Series
) -> float:
    """OLS beta of portfolio vs market."""
    aligned = pd.concat([port_returns, market_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 20:
        return np.nan
    p = aligned.iloc[:, 0].values
    m = aligned.iloc[:, 1].values
    cov = np.cov(p, m)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan


def jensens_alpha(
    port_returns: pd.Series, market_returns: pd.Series, beta: float
) -> float:
    """
    Jensen's Alpha = Ann.PortfolioReturn - [Rf + beta*(Ann.MarketReturn - Rf)]
    """
    ann_port = port_returns.mean() * TRADING_DAYS
    aligned = pd.concat([port_returns, market_returns], axis=1, join="inner").dropna()
    ann_mkt = aligned.iloc[:, 1].mean() * TRADING_DAYS
    return ann_port - (RISK_FREE_RATE + beta * (ann_mkt - RISK_FREE_RATE))


def compute_all_metrics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    market_returns: pd.Series,
) -> Dict[str, float]:
    """
    Compute the full suite of risk metrics for a portfolio.

    Returns:
        Dictionary with metric names and values.
    """
    port_ret = portfolio_daily_returns(returns, weights)
    beta = portfolio_beta(port_ret, market_returns)
    alpha = jensens_alpha(port_ret, market_returns, beta)

    metrics = {
        "Annualised Return (%)": port_ret.mean() * TRADING_DAYS * 100,
        "Annualised Volatility (%)": port_ret.std() * np.sqrt(TRADING_DAYS) * 100,
        "Sharpe Ratio": (
            (port_ret.mean() * TRADING_DAYS - RISK_FREE_RATE)
            / (port_ret.std() * np.sqrt(TRADING_DAYS))
        ),
        "Sortino Ratio": sortino_ratio(port_ret),
        "Max Drawdown (%)": maximum_drawdown(port_ret) * 100,
        "VaR 95% (daily %)": value_at_risk(port_ret, 0.95) * 100,
        "VaR 99% (daily %)": value_at_risk(port_ret, 0.99) * 100,
        "CVaR 95% (daily %)": conditional_var(port_ret, 0.95) * 100,
        "Portfolio Beta": beta,
        "Jensen's Alpha (%)": alpha * 100,
    }
    return metrics


def cumulative_return_series(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Cumulative (growth of ₹1) return series for the portfolio."""
    port_ret = portfolio_daily_returns(returns, weights)
    return (1 + port_ret).cumprod()

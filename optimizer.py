"""
optimizer.py
Core portfolio optimisation logic: Monte Carlo simulation, Efficient Frontier,
and scipy-based Max-Sharpe / Min-Volatility optimisation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065  # ~6.5% (approx Indian 91-day T-bill yield)


# ─── Portfolio Statistics ────────────────────────────────────────────────────

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute annualised return, volatility, and Sharpe ratio for a weight vector.

    Returns:
        (annual_return, annual_volatility, sharpe_ratio)
    """
    ret = np.dot(weights, mean_returns) * TRADING_DAYS
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


# ─── Monte Carlo Simulation ──────────────────────────────────────────────────

def monte_carlo_simulation(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_portfolios: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate random portfolios and return their statistics.

    Returns:
        DataFrame with columns: weights columns + ['Return','Volatility','Sharpe']
    """
    np.random.seed(random_seed)
    n = len(mean_returns)
    results = []

    for _ in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n))
        ret, vol, sharpe = portfolio_performance(w, mean_returns.values, cov_matrix.values)
        row = list(w) + [ret, vol, sharpe]
        results.append(row)

    cols = list(mean_returns.index) + ["Return", "Volatility", "Sharpe"]
    return pd.DataFrame(results, columns=cols)


# ─── Optimisation ────────────────────────────────────────────────────────────

def _neg_sharpe(weights, mean_returns, cov_matrix):
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    return -sharpe


def _portfolio_vol(weights, mean_returns, cov_matrix):
    _, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    return vol


def _optimize(
    objective,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    allow_short: bool = False,
) -> np.ndarray:
    n = len(mean_returns)
    bounds = ((-1.0, 1.0) if allow_short else (0.0, 1.0),) * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x if result.success else x0


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False,
) -> Tuple[np.ndarray, float, float, float]:
    """Return (weights, return, volatility, sharpe) for Max-Sharpe portfolio."""
    w = _optimize(_neg_sharpe, mean_returns.values, cov_matrix.values, allow_short)
    ret, vol, sharpe = portfolio_performance(w, mean_returns.values, cov_matrix.values)
    return w, ret, vol, sharpe


def min_volatility_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    allow_short: bool = False,
) -> Tuple[np.ndarray, float, float, float]:
    """Return (weights, return, volatility, sharpe) for Min-Volatility portfolio."""
    w = _optimize(_portfolio_vol, mean_returns.values, cov_matrix.values, allow_short)
    ret, vol, sharpe = portfolio_performance(w, mean_returns.values, cov_matrix.values)
    return w, ret, vol, sharpe


# ─── Efficient Frontier ──────────────────────────────────────────────────────

def efficient_frontier_points(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by minimising volatility for a range of
    target returns.

    Returns:
        DataFrame with columns ['Return', 'Volatility']
    """
    n = len(mean_returns)
    min_ret = mean_returns.min() * TRADING_DAYS
    max_ret = mean_returns.max() * TRADING_DAYS
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier = []
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w, t=target: np.dot(w, mean_returns.values) * TRADING_DAYS - t,
            },
        ]
        result = minimize(
            _portfolio_vol,
            np.ones(n) / n,
            args=(mean_returns.values, cov_matrix.values),
            method="SLSQP",
            bounds=((0.0, 1.0),) * n,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.success:
            vol = result.fun
            frontier.append({"Return": target, "Volatility": vol})

    return pd.DataFrame(frontier)

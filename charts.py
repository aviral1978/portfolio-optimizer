"""
charts.py
All Plotly and Matplotlib/Seaborn chart builders for the Streamlit app.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# ─── Colour palette ─────────────────────────────────────────────────────────
TEAL     = "#00C9A7"
AMBER    = "#FFB347"
CRIMSON  = "#FF6B6B"
NAVY     = "#1A1A2E"
CARD_BG  = "#16213E"
TEXT     = "#E0E0E0"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=NAVY,
        font=dict(color=TEXT, family="IBM Plex Mono, monospace"),
        xaxis=dict(gridcolor="#2A2A4A", zerolinecolor="#2A2A4A"),
        yaxis=dict(gridcolor="#2A2A4A", zerolinecolor="#2A2A4A"),
        colorway=[TEAL, AMBER, CRIMSON, "#A78BFA", "#34D399", "#FB923C"],
    )
)


def efficient_frontier_chart(
    sim_df: pd.DataFrame,
    frontier_df: Optional[pd.DataFrame],
    max_sharpe: dict,
    min_vol: dict,
) -> go.Figure:
    """
    Scatter of simulated portfolios coloured by Sharpe ratio, overlaid with
    the efficient frontier curve and the two optimal portfolios.
    """
    fig = go.Figure()

    # Simulated portfolios
    fig.add_trace(
        go.Scatter(
            x=sim_df["Volatility"] * 100,
            y=sim_df["Return"] * 100,
            mode="markers",
            marker=dict(
                color=sim_df["Sharpe"],
                colorscale="Viridis",
                size=3,
                opacity=0.6,
                colorbar=dict(title="Sharpe", tickfont=dict(color=TEXT)),
            ),
            name="Simulated Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
        )
    )

    # Efficient frontier line
    if frontier_df is not None and not frontier_df.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_df["Volatility"] * 100,
                y=frontier_df["Return"] * 100,
                mode="lines",
                line=dict(color=TEAL, width=2.5),
                name="Efficient Frontier",
            )
        )

    # Max-Sharpe star
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["vol"] * 100],
            y=[max_sharpe["ret"] * 100],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color=AMBER),
            text=["★ Max Sharpe"],
            textposition="top right",
            textfont=dict(color=AMBER, size=11),
            name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
        )
    )

    # Min-Volatility diamond
    fig.add_trace(
        go.Scatter(
            x=[min_vol["vol"] * 100],
            y=[min_vol["ret"] * 100],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=CRIMSON),
            text=["◆ Min Vol"],
            textposition="top right",
            textfont=dict(color=CRIMSON, size=11),
            name=f"Min Volatility ({min_vol['vol']*100:.1f}%)",
        )
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Efficient Frontier & Monte Carlo Simulation",
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        height=520,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


def weights_pie_chart(weights: np.ndarray, labels: list, title: str) -> go.Figure:
    """Donut chart for portfolio weights."""
    # Only show positions above 0.5%
    mask = weights > 0.005
    w = weights[mask]
    l = [labels[i] for i in range(len(labels)) if mask[i]]

    fig = go.Figure(
        go.Pie(
            labels=l,
            values=w,
            hole=0.45,
            textinfo="label+percent",
            marker=dict(
                colors=px.colors.sequential.Viridis[:: max(1, len(px.colors.sequential.Viridis) // len(l))][: len(l)],
                line=dict(color=NAVY, width=2),
            ),
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=title,
        height=420,
        showlegend=False,
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> plt.Figure:
    """Seaborn correlation heatmap (matplotlib figure)."""
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.9), max(5, len(corr) * 0.8)))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(NAVY)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor=NAVY,
        annot_kws={"size": 8, "color": "white"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix", color=TEXT, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def covariance_heatmap(cov: pd.DataFrame) -> plt.Figure:
    """Seaborn covariance heatmap (matplotlib figure)."""
    fig, ax = plt.subplots(figsize=(max(6, len(cov) * 0.9), max(5, len(cov) * 0.8)))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(NAVY)
    sns.heatmap(
        cov * 252,  # annualise
        ax=ax,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor=NAVY,
        annot_kws={"size": 7, "color": "black"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Annualised Covariance Matrix", color=TEXT, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def cumulative_return_chart(
    cum_returns: dict,  # {label: pd.Series}
) -> go.Figure:
    """Overlaid cumulative return curves."""
    colours = [TEAL, AMBER, CRIMSON, "#A78BFA"]
    fig = go.Figure()
    for i, (label, series) in enumerate(cum_returns.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line=dict(width=2, color=colours[i % len(colours)]),
                hovertemplate="%{y:.3f}<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Portfolio Growth (₹1 Invested)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def rolling_sharpe_chart(port_returns: pd.Series, window: int = 63) -> go.Figure:
    """Rolling Sharpe ratio (default ~1-quarter window)."""
    rf_daily = 0.065 / 252
    excess = port_returns - rf_daily
    roll_sharpe = (
        excess.rolling(window).mean() / excess.rolling(window).std()
    ) * np.sqrt(252)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roll_sharpe.index,
            y=roll_sharpe.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0,201,167,0.15)",
            line=dict(color=TEAL, width=1.5),
            name=f"Rolling Sharpe ({window}d)",
        )
    )
    fig.add_hline(y=0, line=dict(color=CRIMSON, dash="dash", width=1))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=f"Rolling Sharpe Ratio ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=350,
    )
    return fig


def drawdown_chart(port_returns: pd.Series) -> go.Figure:
    """Underwater (drawdown) chart."""
    cum = (1 + port_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.25)",
            line=dict(color=CRIMSON, width=1.5),
            name="Drawdown",
            hovertemplate="%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=320,
    )
    return fig


def var_histogram(port_returns: pd.Series) -> go.Figure:
    """Daily-return distribution with VaR lines."""
    var95 = float(-np.percentile(port_returns, 5))
    var99 = float(-np.percentile(port_returns, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=port_returns * 100,
            nbinsx=80,
            marker=dict(color=TEAL, opacity=0.75, line=dict(color=NAVY, width=0.3)),
            name="Daily Returns",
        )
    )
    for val, label, col in [
        (var95 * 100, "VaR 95%", AMBER),
        (var99 * 100, "VaR 99%", CRIMSON),
    ]:
        fig.add_vline(
            x=-val,
            line=dict(color=col, dash="dash", width=1.5),
            annotation_text=f"{label}: {val:.2f}%",
            annotation_font_color=col,
            annotation_font_size=10,
        )
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Return Distribution & Value at Risk",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=360,
        bargap=0.02,
    )
    return fig

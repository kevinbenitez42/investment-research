"""Idiosyncratic-risk factor-analysis views."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_idiosyncratic_risk_view(
    rolling_results,
    ticker_label,
    *,
    column="idio_vol_annualized",
    template="plotly_white",
):
    """Compose rolling idiosyncratic-risk figure from regression results."""
    if column not in rolling_results.columns:
        raise ValueError(f"Column '{column}' not found in rolling_results.")

    window_used = rolling_results.attrs.get("window_used")
    aligned_start = rolling_results.attrs.get("aligned_start")
    if window_used is not None and aligned_start is not None:
        title = f"{ticker_label}: Rolling Idiosyncratic Risk ({window_used}-day window, start {aligned_start})"
    else:
        title = f"{ticker_label}: Rolling Idiosyncratic Risk"

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=rolling_results.index,
            y=rolling_results[column],
            mode="lines",
            name=column,
            line=dict(color="firebrick", width=2),
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility",
        template=template,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

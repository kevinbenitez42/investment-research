"""Backtesting equity-curve comparison views."""

from __future__ import annotations

import plotly.graph_objects as go


def _equity_trace(frame, column, *, name, color, width):
    return go.Scatter(
        x=frame.index,
        y=frame[column],
        name=name,
        line={"color": color, "width": width},
    )


def plot_backtest_equity_curves_view(price_frame, *, ticker_label="Asset", template="plotly_dark"):
    """Compose buy-and-hold and strategy equity curves."""
    fig = go.Figure()
    fig.add_trace(
        _equity_trace(
            price_frame,
            "buy_hold_index",
            name="Buy and hold",
            color="#94a3b8",
            width=2,
        )
    )
    fig.add_trace(
        _equity_trace(
            price_frame,
            "strategy_index",
            name="MA crossover strategy",
            color="#22c55e",
            width=2.5,
        )
    )
    fig.add_trace(
        _equity_trace(
            price_frame,
            "inverse_vol_index",
            name="Inverse volatility strategy",
            color="#38bdf8",
            width=2.5,
        )
    )
    fig.update_layout(
        title=f"{ticker_label} backtest equity curves",
        template=template,
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=500,
    )
    return fig

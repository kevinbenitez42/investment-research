"""Portfolio allocation visualization views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_rolling_portfolio_allocation_stacked(results: pd.DataFrame, smooth_window: int = 5) -> go.Figure | None:
    if results.empty:
        print("No rolling allocations were produced. Check window size or input data.")
        return None

    smoothed = results.rolling(window=smooth_window, min_periods=1).mean().dropna(how="all")
    fig = go.Figure()
    for col in smoothed.columns:
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed[col],
                stackgroup="weights",
                mode="lines",
                name=col,
            )
        )
    fig.update_layout(
        title="Smoothed Rolling Portfolio Weights",
        xaxis_title="Date",
        yaxis_title="Weight",
        template="plotly_dark",
        hovermode="x unified",
        height=600,
    )
    return fig


def plot_rolling_portfolio_allocation(results: pd.DataFrame) -> go.Figure | None:
    if results.empty:
        print("No rolling allocations were produced. Check window size or input data.")
        return None

    fig = go.Figure()
    for col in results.columns:
        fig.add_trace(go.Scatter(x=results.index, y=results[col], mode="lines", name=col))

    latest_date = results.index[-1]
    for col in results.columns:
        latest_weight = results[col].iloc[-1]
        fig.add_annotation(
            x=latest_date,
            y=latest_weight,
            text=f"{latest_weight:.2f}",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(color=fig.data[results.columns.get_loc(col)].line.color),
            bgcolor="rgba(255,255,255,0.5)",
        )

    fig.update_layout(
        title="Rolling Optimal Portfolio Allocation",
        xaxis_title="Date",
        yaxis_title="Weight",
        template="plotly_dark",
        height=600,
        margin=dict(r=150),
    )
    return fig

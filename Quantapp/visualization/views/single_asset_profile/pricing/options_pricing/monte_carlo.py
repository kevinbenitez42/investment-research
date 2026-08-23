"""Monte Carlo projection views for options-pricing workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _future_dates(dates, projection_length):
    last_date = pd.Timestamp(pd.Index(dates)[-1])
    return [last_date + pd.Timedelta(days=i + 1) for i in range(projection_length)]


def plot_gbm_paths_view(
    dates,
    prices,
    price_paths,
    median_path,
    p5,
    p95,
    *,
    ticker_label="Ticker",
    template="plotly_white",
):
    """Compose historical price plus geometric Brownian motion simulation paths."""
    prices = np.asarray(prices, dtype=float)
    price_paths = np.asarray(price_paths, dtype=float)
    median_path = np.asarray(median_path, dtype=float)
    p5 = np.asarray(p5, dtype=float)
    p95 = np.asarray(p95, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            name="Historical Price",
            line=dict(color="black", width=2),
        )
    )

    projection_length, path_count = price_paths.shape
    future_dates = _future_dates(dates, projection_length)
    for path_idx in range(path_count):
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=price_paths[:, path_idx],
                mode="lines",
                line=dict(color="blue", width=1),
                opacity=0.05,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,255,0.25)",
            line=dict(color="rgba(0,0,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="5-95% Range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=median_path,
            mode="lines",
            line=dict(color="red", width=2),
            name="Median Path",
        )
    )

    terminal_prices = np.percentile(price_paths[-1, :], [5, 25, 50, 75, 95])
    terminal_names = ["5%", "25%", "50% Median", "75%", "95%"]
    for price, name in zip(terminal_prices, terminal_names):
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(pd.Index(dates)[-1]), future_dates[-1]],
                y=[price, price],
                mode="lines",
                line=dict(dash="dash", width=2),
                name=f"Terminal {name}",
            )
        )
        fig.add_annotation(
            x=future_dates[-1],
            y=price,
            xanchor="left",
            text=f"{name}: {price:.2f}",
            showarrow=False,
            font=dict(color="black"),
        )

    fig.update_layout(
        title=f"Historical Price + 30-Day Monte Carlo Simulation ({ticker_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        template=template,
        width=1200,
        height=800,
    )
    return fig

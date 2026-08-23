"""DCF fair value views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ..._shared import apply_price_history_layout


def plot_dcf_snapshot_vs_price(
    price_history: pd.DataFrame,
    dcf_history: pd.DataFrame,
    *,
    symbol: str,
    chart_label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_history["date"],
            y=price_history["price"],
            name=f"{symbol} close",
            line={"color": "#e5eefc", "width": 2.5},
        )
    )

    dcf_mode = "lines+markers" if len(dcf_history) > 1 else "markers+text"
    dcf_text = None if len(dcf_history) > 1 else [f"DCF: {dcf_history['dcf'].iloc[0]:.2f}"]
    fig.add_trace(
        go.Scatter(
            x=dcf_history["date"],
            y=dcf_history["dcf"],
            name="FMP DCF",
            mode=dcf_mode,
            line={"color": "#f472b6", "width": 2.5},
            marker={"color": "#f472b6", "size": 10},
            text=dcf_text,
            textposition="top center",
        )
    )

    if "Stock Price" in dcf_history.columns and dcf_history["Stock Price"].notna().any():
        priced_dcf = dcf_history.loc[dcf_history["Stock Price"].notna()]
        fig.add_trace(
            go.Scatter(
                x=priced_dcf["date"],
                y=priced_dcf["Stock Price"],
                name="FMP stock price at DCF snapshot",
                mode="markers",
                marker={"color": "#38bdf8", "size": 8, "symbol": "diamond"},
            )
        )

    apply_price_history_layout(fig, f"{chart_label} DCF vs price")
    return fig


def plot_backfilled_dcf_vs_price(
    price_history: pd.DataFrame,
    dcf_series: pd.DataFrame,
    *,
    symbol: str,
    chart_label: str,
    dcf_column: str,
    dcf_label: str,
    title_suffix: str,
    color: str = "#f472b6",
    marker_size: int = 9,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_history["date"],
            y=price_history["price"],
            name=f"{symbol} close",
            line={"color": "#e5eefc", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dcf_series["date"],
            y=dcf_series[dcf_column],
            name=dcf_label,
            mode="lines+markers",
            line={"color": color, "width": 2.5},
            marker={"color": color, "size": marker_size},
        )
    )
    apply_price_history_layout(fig, f"{chart_label} {title_suffix}")
    return fig


def plot_annual_vs_quarterly_dcf(
    price_history: pd.DataFrame,
    annual_dcf_series: pd.DataFrame,
    quarterly_dcf_series: pd.DataFrame,
    *,
    symbol: str,
    chart_label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_history["date"],
            y=price_history["price"],
            name=f"{symbol} close",
            line={"color": "#e5eefc", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=annual_dcf_series["date"],
            y=annual_dcf_series["historicalDcfPerShare"],
            name="Annual backfilled DCF",
            mode="lines+markers",
            line={"color": "#f472b6", "width": 2.5, "dash": "dash"},
            marker={"color": "#f472b6", "size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quarterly_dcf_series["date"],
            y=quarterly_dcf_series["quarterlyBackfilledDcfPerShare"],
            name="Quarterly TTM backfilled DCF",
            mode="lines+markers",
            line={"color": "#22d3ee", "width": 2.5},
            marker={"color": "#22d3ee", "size": 6},
        )
    )
    apply_price_history_layout(fig, f"{chart_label} annual vs quarterly backfilled DCF")
    return fig

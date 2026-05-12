"""Momentum-window diagnostics grid view."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.traces.line import build_line_trace
from ._shared import add_reference_vlines, coerce_momentum_diagnostics_context, finalize_dark_figure


def _build_optimal_window_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Optimal Window Size",
        visible=visible,
    )


def _build_optimal_window_histogram_trace(values, *, nbins):
    return go.Histogram(
        x=values,
        nbinsx=nbins,
        name="Optimal Window Size Distribution",
        showlegend=False,
    )


def _build_mean_sharpe_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Mean Sharpe Ratio",
        mode="lines+markers",
        visible=visible,
    )


def _build_median_sharpe_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Median Sharpe Ratio",
        mode="lines+markers",
        visible=visible,
    )


def _build_mean_volatility_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Mean Volatility",
        mode="lines+markers",
        visible=visible,
    )


def _build_median_volatility_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Median Volatility",
        mode="lines+markers",
        visible=visible,
    )


def plot_momentum_window_diagnostics_grid_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_dark",
):
    """Compose the primary momentum-window diagnostics in a 2x2 grid."""
    context = coerce_momentum_diagnostics_context(diagnostics_context)
    sharpe_table = context["sharpe_table"]
    optimal_windows_int = context["optimal_windows_int"]
    window_sizes = context["window_sizes"]
    highlight_windows = context["highlight_windows"]
    mean_sharpe = context["mean_sharpe"]
    median_sharpe = context["median_sharpe"]
    mean_volatility = context["mean_volatility"]
    median_volatility = context["median_volatility"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Rolling Optimal Momentum Window",
            "Optimal Window Distribution",
            "Mean vs Median Sharpe by Window",
            "Mean vs Median Volatility by Window",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.13,
    )

    fig.add_trace(
        _build_optimal_window_trace(
            sharpe_table.index,
            sharpe_table["Optimal_Window"],
            visible=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _build_optimal_window_histogram_trace(
            optimal_windows_int,
            nbins=len(window_sizes),
        ),
        row=1,
        col=2,
    )
    mean_sharpe_trace = _build_mean_sharpe_trace(
        mean_sharpe.index,
        mean_sharpe.values,
        visible=True,
    )
    mean_sharpe_trace.showlegend = True
    fig.add_trace(
        mean_sharpe_trace,
        row=2,
        col=1,
    )
    median_sharpe_trace = _build_median_sharpe_trace(
        median_sharpe.index,
        median_sharpe.values,
        visible=True,
    )
    median_sharpe_trace.showlegend = True
    fig.add_trace(
        median_sharpe_trace,
        row=2,
        col=1,
    )
    mean_volatility_trace = _build_mean_volatility_trace(
        mean_volatility.index,
        mean_volatility.values,
        visible=True,
    )
    mean_volatility_trace.showlegend = True
    fig.add_trace(
        mean_volatility_trace,
        row=2,
        col=2,
    )
    median_volatility_trace = _build_median_volatility_trace(
        median_volatility.index,
        median_volatility.values,
        visible=True,
    )
    median_volatility_trace.showlegend = True
    fig.add_trace(
        median_volatility_trace,
        row=2,
        col=2,
    )

    for row, col in ((1, 2), (2, 1), (2, 2)):
        add_reference_vlines(fig, highlight_windows, row=row, col=col)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Optimal Window Size (Days)", row=1, col=2)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=2, col=1)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=2, col=2)

    fig.update_yaxes(title_text="Window Size (Days)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=2, col=2)

    fig.update_layout(
        title=f"{ticker_label} Momentum Window Diagnostics",
        template=template,
        height=900,
        bargap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return finalize_dark_figure(fig)

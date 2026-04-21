"""Composed momentum diagnostic figures built from plot-type traces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..traces.histogram import build_optimal_window_histogram_trace
from ..traces.line import (
    build_mean_sharpe_trace,
    build_mean_volatility_trace,
    build_median_sharpe_trace,
    build_median_volatility_trace,
    build_optimal_window_trace,
)
from ..traces.surface import build_sharpe_surface_trace, build_vertical_plane_trace

REFERENCE_WINDOW_STYLE_MAP = {
    7: ("orange", "7 Days"),
    21: ("red", "21 Days"),
    50: ("blue", "50 Days"),
    200: ("green", "200 Days"),
}


def _coerce_momentum_diagnostics_context(diagnostics_context):
    required_keys = {
        "window_sizes",
        "highlight_windows",
        "sharpe_table",
        "optimal_windows_int",
        "mean_sharpe",
        "median_sharpe",
        "mean_volatility",
        "median_volatility",
        "sharpe_surface",
        "surface_years",
    }
    if not isinstance(diagnostics_context, dict):
        try:
            diagnostics_context = dict(diagnostics_context)
        except Exception as exc:
            raise TypeError("diagnostics_context must be a mapping.") from exc

    missing = [key for key in required_keys if key not in diagnostics_context]
    if missing:
        raise ValueError(f"diagnostics_context missing required keys: {missing}")

    return diagnostics_context


def _add_reference_vlines(fig: go.Figure, highlight_windows) -> None:
    for window in highlight_windows:
        color, label = REFERENCE_WINDOW_STYLE_MAP.get(window, ("gray", f"{window} Days"))
        fig.add_vline(
            x=window,
            line_color=color,
            line_dash="dash",
            annotation_text=label,
            annotation_position="top left",
        )


def plot_optimal_window_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_white",
):
    """Compose the rolling optimal momentum window figure."""
    context = _coerce_momentum_diagnostics_context(diagnostics_context)
    sharpe_table = context["sharpe_table"]

    fig = go.Figure()
    fig.add_trace(
        build_optimal_window_trace(
            sharpe_table.index,
            sharpe_table["Optimal_Window"],
        )
    )
    fig.update_layout(
        title=f"{ticker_label} Rolling Optimal Momentum Window",
        xaxis_title="Date",
        yaxis_title="Window Size (Days)",
        template=template,
        showlegend=False,
    )
    return fig


def plot_optimal_window_histogram_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_white",
):
    """Compose the optimal-window histogram figure."""
    context = _coerce_momentum_diagnostics_context(diagnostics_context)
    optimal_windows_int = context["optimal_windows_int"]
    window_sizes = context["window_sizes"]
    highlight_windows = context["highlight_windows"]

    fig = go.Figure()
    fig.add_trace(
        build_optimal_window_histogram_trace(
            optimal_windows_int,
            nbins=len(window_sizes),
        )
    )
    fig.update_layout(
        title=f"{ticker_label} Distribution of Optimal Sharpe Momentum Windows Over Time",
        xaxis_title="Optimal Window Size (Days)",
        yaxis_title="Frequency",
        bargap=0.1,
        template=template,
        showlegend=False,
    )
    _add_reference_vlines(fig, highlight_windows)
    return fig


def plot_sharpe_mean_median_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_white",
):
    """Compose the mean-versus-median Sharpe figure."""
    context = _coerce_momentum_diagnostics_context(diagnostics_context)
    mean_sharpe = context["mean_sharpe"]
    median_sharpe = context["median_sharpe"]
    highlight_windows = context["highlight_windows"]

    fig = go.Figure()
    fig.add_trace(build_mean_sharpe_trace(mean_sharpe.index, mean_sharpe.values))
    fig.add_trace(build_median_sharpe_trace(median_sharpe.index, median_sharpe.values))
    fig.update_layout(
        title=f"{ticker_label} Mean and Median Sharpe Ratios Across All Dates by Momentum Window",
        xaxis_title="Momentum Window Size (Days)",
        yaxis_title="Sharpe Ratio",
        template=template,
    )
    _add_reference_vlines(fig, highlight_windows)
    return fig


def plot_volatility_mean_median_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_white",
):
    """Compose the mean-versus-median volatility figure."""
    context = _coerce_momentum_diagnostics_context(diagnostics_context)
    mean_volatility = context["mean_volatility"]
    median_volatility = context["median_volatility"]
    highlight_windows = context["highlight_windows"]

    fig = go.Figure()
    fig.add_trace(build_mean_volatility_trace(mean_volatility.index, mean_volatility.values))
    fig.add_trace(build_median_volatility_trace(median_volatility.index, median_volatility.values))
    fig.update_layout(
        title=f"{ticker_label} Mean and Median Volatility Across All Dates by Momentum Window",
        xaxis_title="Momentum Window Size (Days)",
        yaxis_title="Annualized Volatility (Rolling Std of Excess Returns)",
        template=template,
    )
    _add_reference_vlines(fig, highlight_windows)
    return fig


def plot_sharpe_surface_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_white",
):
    """Compose the 3D Sharpe surface figure."""
    context = _coerce_momentum_diagnostics_context(diagnostics_context)
    sharpe_surface = context["sharpe_surface"]
    surface_years = context["surface_years"]
    highlight_windows = context["highlight_windows"]

    if sharpe_surface.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker_label} 3D Surface of Sharpe Ratios by Window and Date (Last {surface_years} Years)",
            template=template,
        )
        fig.add_annotation(
            text="No Sharpe surface data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    z_values = sharpe_surface.values
    finite_values = z_values[np.isfinite(z_values)]
    if finite_values.size == 0:
        z_min, z_max = 0.0, 1.0
    else:
        z_min, z_max = float(finite_values.min()), float(finite_values.max())
        if z_min == z_max:
            z_max = z_min + 1.0

    date_labels = sharpe_surface.index.strftime("%Y-%m-%d")
    y_vals = np.arange(len(sharpe_surface))

    traces = [build_sharpe_surface_trace(sharpe_surface)]
    plane_color_map = {7: "orange", 21: "red", 50: "blue", 200: "green"}
    for window in highlight_windows:
        traces.append(
            build_vertical_plane_trace(
                x_val=window,
                y_vals=y_vals,
                z_min=z_min,
                z_max=z_max,
                opacity=0.3,
                color=plane_color_map.get(window, "gray"),
            )
        )

    tick_step = max(1, len(date_labels) // 10)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{ticker_label} 3D Surface of Sharpe Ratios by Window and Date (Last {surface_years} Years) with Highlighted Windows",
        scene=dict(
            xaxis_title="Momentum Window Size (Days)",
            yaxis_title="Date",
            yaxis=dict(
                tickmode="array",
                tickvals=np.arange(0, len(date_labels), step=tick_step),
                ticktext=date_labels[::tick_step],
            ),
            zaxis_title="Sharpe Ratio",
        ),
        height=900,
        template=template,
    )
    return fig

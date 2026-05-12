"""Sharpe surface momentum diagnostic view."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ._shared import coerce_momentum_diagnostics_context, finalize_dark_figure


def _build_sharpe_surface_trace(surface_frame):
    return go.Surface(
        z=surface_frame.values,
        x=surface_frame.columns,
        y=np.arange(len(surface_frame)),
        colorscale="Viridis",
        colorbar=dict(title="Sharpe Ratio"),
        name="Sharpe Surface",
    )


def _build_vertical_plane_trace(
    *,
    x_val,
    y_vals,
    z_min,
    z_max,
    opacity=0.3,
    color="red",
):
    y_grid, z_grid = np.meshgrid(y_vals, np.linspace(z_min, z_max, 2))
    x_grid = np.full_like(y_grid, x_val)
    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        showscale=False,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
    )


def plot_sharpe_surface_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_dark",
):
    """Compose the 3D Sharpe surface figure."""
    context = coerce_momentum_diagnostics_context(diagnostics_context)
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
        return finalize_dark_figure(fig)

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

    traces = [_build_sharpe_surface_trace(sharpe_surface)]
    plane_color_map = {7: "orange", 21: "red", 50: "blue", 200: "green"}
    for window in highlight_windows:
        traces.append(
            _build_vertical_plane_trace(
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
    return finalize_dark_figure(fig)

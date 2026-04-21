"""3D surface trace builders used by composed visualization views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_sharpe_surface_trace(surface_frame):
    """Build the main Sharpe-ratio surface trace."""
    return go.Surface(
        z=surface_frame.values,
        x=surface_frame.columns,
        y=np.arange(len(surface_frame)),
        colorscale="Viridis",
        colorbar=dict(title="Sharpe Ratio"),
        name="Sharpe Surface",
    )


def build_vertical_plane_trace(
    *,
    x_val,
    y_vals,
    z_min,
    z_max,
    opacity=0.3,
    color="red",
):
    """Build a vertical reference plane for a highlighted window."""
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

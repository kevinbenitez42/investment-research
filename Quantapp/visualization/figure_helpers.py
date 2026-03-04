"""Reusable Plotly figure helpers."""

from __future__ import annotations

import plotly.graph_objects as go


def add_sigma_reference_lines(
    fig: go.Figure,
    row: int,
    x_ref,
    levels=(1, 2, 3),
    sigma: float = 1.0,
    center: float = 0.0,
    col: int = 1,
    line_color: str = "rgba(220, 220, 220, 0.55)",
    line_dash: str = "dot",
    line_width: int = 1,
    visible: bool = True,
) -> None:
    """Add symmetric horizontal sigma reference lines to a figure."""
    if x_ref is None or len(x_ref) == 0:
        return

    for level in levels:
        for direction in (1, -1):
            value = center + (direction * level * sigma)
            fig.add_trace(
                go.Scatter(
                    x=x_ref,
                    y=[value] * len(x_ref),
                    mode="lines",
                    line=dict(color=line_color, dash=line_dash, width=line_width),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=visible,
                ),
                row=row,
                col=col,
            )

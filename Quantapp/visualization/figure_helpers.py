"""Reusable Plotly figure helpers."""

from __future__ import annotations

import pandas as pd
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


def add_mean_reference_line(
    fig: go.Figure,
    row: int,
    x_ref,
    col: int = 1,
    line_color: str = "rgba(255, 255, 255, 0.80)",
    line_width: int = 1,
    visible: bool = True,
) -> None:
    """Add a horizontal center line at y=0 using a trace."""
    if x_ref is None or len(x_ref) == 0:
        return
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=[0] * len(x_ref),
            mode="lines",
            line=dict(color=line_color, width=line_width),
            showlegend=False,
            hoverinfo="skip",
            visible=visible,
        ),
        row=row,
        col=col,
    )


def add_std_annotations(
    fig: go.Figure,
    row: int,
    levels=(0.5, 1, 1.5, 2),
    col: int = 1,
    visible: bool = True,
) -> None:
    """Add +/− standard deviation labels anchored to the subplot x-domain."""
    labels = [(0, "Mean")]
    for level in levels:
        labels.append((level, f"+{level} SD"))
        labels.append((-level, f"-{level} SD"))
    for y_value, label in labels:
        fig.add_annotation(
            x=0.985,
            y=y_value,
            xref="x domain",
            yref="y",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(color="rgba(220, 220, 220, 0.90)", size=11),
            align="left",
            visible=visible,
            row=row,
            col=col,
        )


def add_zone_annotation(
    fig: go.Figure,
    row: int,
    y0: float,
    y1: float,
    text: str,
    font_color: str,
    col: int = 1,
    visible: bool = True,
) -> None:
    """Annotate a horizontal zone band in the middle of its y-range."""
    fig.add_annotation(
        x=0.5,
        y=(y0 + y1) / 2,
        xref="x domain",
        yref="y",
        text=text,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(color=font_color, size=14),
        visible=visible,
        row=row,
        col=col,
    )


def add_horizontal_zone(
    fig: go.Figure,
    row: int,
    y0: float,
    y1: float,
    fillcolor: str,
    opacity: float = 0.45,
    col: int = 1,
    line_color: str = "green",
    line_width: int = 1,
) -> None:
    """Add an hrect zone for a subplot row."""
    fig.add_hrect(
        y0=y0,
        y1=y1,
        fillcolor=fillcolor,
        opacity=opacity,
        line_color=line_color,
        line_width=line_width,
        layer="below",
        row=row,
        col=col,
    )


def add_horizontal_zone_trace(
    fig: go.Figure,
    row: int,
    x_ref,
    y0: float,
    y1: float,
    fillcolor: str,
    col: int = 1,
    visible: bool = True,
) -> None:
    """Add a rectangular horizontal zone using a closed polygon trace."""
    if x_ref is None or len(x_ref) == 0:
        return
    x0 = x_ref[0]
    x1 = x_ref[-1]
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor=fillcolor,
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=col,
    )


def build_time_range_buttons(global_start, global_end, axis_count: int = 1):
    """Create relayout buttons for standard lookback windows."""

    def make_range(years=None):
        if years is None:
            start = global_start
        else:
            start = max(global_start, global_end - pd.DateOffset(years=years))
        layout = {}
        for axis_idx in range(1, axis_count + 1):
            axis_name = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            layout[f"{axis_name}.range"] = [start, global_end]
        return layout

    return [
        dict(label="10 Years", method="relayout", args=[make_range(10)]),
        dict(label="5 Years", method="relayout", args=[make_range(5)]),
        dict(label="3 Years", method="relayout", args=[make_range(3)]),
        dict(label="1 Year", method="relayout", args=[make_range(1)]),
    ]


def build_detail_visibility_mask(
    dynamic_trace_count: int,
    total_traces: int,
    view_index: int,
    traces_per_view: int,
):
    """Visibility mask for multi-view detail figures with constant trailing traces."""
    visibility = [False] * dynamic_trace_count + [True] * (total_traces - dynamic_trace_count)
    start = view_index * traces_per_view
    for trace_idx in range(start, min(start + traces_per_view, dynamic_trace_count)):
        visibility[trace_idx] = True
    return visibility


def build_visibility_mask(
    total_traces: int,
    active_window_index: int,
    traces_per_window: int,
    constant_trace_indices,
):
    """Visibility mask for windowed dropdown figures."""
    visibility = [False] * total_traces
    start = active_window_index * traces_per_window
    for trace_idx in range(start, start + traces_per_window):
        visibility[trace_idx] = True
    for trace_idx in constant_trace_indices:
        visibility[trace_idx] = True
    return visibility

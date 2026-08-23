"""Line trace builders used by composed visualization views."""

from __future__ import annotations

import plotly.graph_objects as go


def build_line_trace(
    *,
    x,
    y,
    name,
    color=None,
    width=2,
    dash=None,
    visible=True,
    showlegend=False,
    hovertemplate=None,
    mode="lines",
    fill=None,
    fillcolor=None,
    hoverinfo=None,
    opacity=None,
    marker=None,
    customdata=None,
    legendgroup=None,
):
    """Build a Plotly line trace."""
    line = dict(width=width)
    if color is not None:
        line["color"] = color
    if dash is not None:
        line["dash"] = dash

    trace = go.Scatter(
        x=x,
        y=y,
        mode=mode,
        name=name,
        line=line,
        visible=visible,
        showlegend=showlegend,
    )
    if hovertemplate is not None:
        trace.hovertemplate = hovertemplate
    if fill is not None:
        trace.fill = fill
    if fillcolor is not None:
        trace.fillcolor = fillcolor
    if hoverinfo is not None:
        trace.hoverinfo = hoverinfo
    if opacity is not None:
        trace.opacity = opacity
    if marker is not None:
        trace.marker = marker
    if customdata is not None:
        trace.customdata = customdata
    if legendgroup is not None:
        trace.legendgroup = legendgroup
    return trace


def build_horizontal_level_trace(
    x,
    *,
    y_value,
    name,
    color="rgba(220, 220, 220, 0.55)",
    width=1,
    dash=None,
    visible=True,
    showlegend=False,
    hoverinfo="skip",
):
    """Build a horizontal reference line using the supplied x-axis reference."""
    return build_line_trace(
        x=x,
        y=[y_value] * len(x),
        name=name,
        color=color,
        width=width,
        dash=dash,
        visible=visible,
        showlegend=showlegend,
        hoverinfo=hoverinfo,
    )

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
    return trace


def build_underwater_trace(window, x, y, *, visible):
    """Build the underwater drawdown line."""
    return build_line_trace(
        x=x,
        y=y,
        name=f"{window}-Day Underwater",
        color="#0ea5e9",
        width=2,
        visible=visible,
    )


def build_textbook_drawdown_trace(window, x, y, *, visible):
    """Build the textbook rolling max drawdown line."""
    return build_line_trace(
        x=x,
        y=y,
        name=f"{window}-Day Textbook Max Drawdown",
        color="#f97316",
        width=2,
        dash="dot",
        visible=visible,
    )


def build_recovery_time_trace(window, x, y, *, visible):
    """Build the rolling recovery-time line."""
    return build_line_trace(
        x=x,
        y=y,
        name=f"{window}-Day Recovery Time",
        color="#22c55e",
        width=2,
        visible=visible,
    )


def build_percentile_reference_trace(
    *,
    x,
    y_value,
    name,
    color,
    visible,
    hovertemplate,
    width=1.5,
    dash="dash",
):
    """Build a horizontal percentile reference line as a scatter trace."""
    return build_line_trace(
        x=x,
        y=[y_value] * len(x),
        name=name,
        color=color,
        width=width,
        dash=dash,
        visible=visible,
        hovertemplate=hovertemplate,
    )


def build_optimal_window_trace(x, y, *, visible=True):
    """Build the rolling optimal-window line."""
    return build_line_trace(
        x=x,
        y=y,
        name="Optimal Window Size",
        visible=visible,
    )


def build_mean_sharpe_trace(x, y, *, visible=True):
    """Build the mean Sharpe line."""
    return build_line_trace(
        x=x,
        y=y,
        name="Mean Sharpe Ratio",
        mode="lines+markers",
        visible=visible,
    )


def build_median_sharpe_trace(x, y, *, visible=True):
    """Build the median Sharpe line."""
    return build_line_trace(
        x=x,
        y=y,
        name="Median Sharpe Ratio",
        mode="lines+markers",
        visible=visible,
    )


def build_mean_volatility_trace(x, y, *, visible=True):
    """Build the mean volatility line."""
    return build_line_trace(
        x=x,
        y=y,
        name="Mean Volatility",
        mode="lines+markers",
        visible=visible,
    )


def build_median_volatility_trace(x, y, *, visible=True):
    """Build the median volatility line."""
    return build_line_trace(
        x=x,
        y=y,
        name="Median Volatility",
        mode="lines+markers",
        visible=visible,
    )

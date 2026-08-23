"""Momentum-window diagnostics grid view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import add_zone_annotation
from Quantapp.visualization.traces.line import build_line_trace
from ._shared import add_reference_vlines, coerce_momentum_diagnostics_context, finalize_dark_figure


HORIZON_SEGMENT_REGIONS = (
    ("Very Short", None, 21, "#64748b"),
    ("Short", 21, 63, "#94a3b8"),
    ("Medium", 63, 126, "#64748b"),
    ("Long", 126, 252, "#94a3b8"),
    ("LEAPS", 252, 504, "#64748b"),
    ("Extended", 504, None, "#94a3b8"),
)

HORIZON_SEGMENT_LINE_STYLES = (
    ("#f8fafc", "solid"),
    ("#e2e8f0", "dash"),
    ("#cbd5e1", "dot"),
    ("#f1f5f9", "dashdot"),
    ("#d1d5db", "longdash"),
    ("#94a3b8", "longdashdot"),
)


def _coerce_ratio_label(ratio_label):
    label = str(ratio_label or "Sharpe").strip()
    return label or "Sharpe"


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


def _build_mean_sharpe_trace(x, y, *, ratio_label="Sharpe", visible=True):
    ratio_label = _coerce_ratio_label(ratio_label)
    return build_line_trace(
        x=x,
        y=y,
        name=f"Mean {ratio_label} Ratio",
        mode="lines+markers",
        visible=visible,
    )


def _build_median_sharpe_trace(x, y, *, ratio_label="Sharpe", visible=True):
    ratio_label = _coerce_ratio_label(ratio_label)
    return build_line_trace(
        x=x,
        y=y,
        name=f"Median {ratio_label} Ratio",
        mode="lines+markers",
        visible=visible,
    )


def _build_current_sharpe_trace(x, y, *, ratio_label="Sharpe", visible=True):
    ratio_label = _coerce_ratio_label(ratio_label)
    return build_line_trace(
        x=x,
        y=y,
        name=f"Current {ratio_label} Ratio",
        mode="lines+markers",
        visible=visible,
        width=3,
        marker=dict(size=4),
        hovertemplate=(
            f"Window: %{{x}} day(s)<br>Current {ratio_label}: "
            "%{y:.2f}<extra></extra>"
        ),
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


def _build_current_sharpe_zscore_trace(
    x, y, *, ratio_label="Sharpe", visible=True
):
    ratio_label = _coerce_ratio_label(ratio_label)
    return build_line_trace(
        x=x,
        y=y,
        name=f"Current {ratio_label} Z-Score",
        mode="lines+markers",
        visible=visible,
        marker=dict(size=4),
        hovertemplate=(
            f"Window: %{{x}} day(s)<br>Current {ratio_label} Z-Score: "
            "%{y:.2f}<extra></extra>"
        ),
    )


def _build_cross_window_sharpe_zscore_trace(x, y, *, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name="Cross-Window Relative Z-Score",
        mode="lines+markers",
        visible=visible,
        width=2,
        dash="dot",
        marker=dict(size=4),
        hovertemplate=(
            "Window: %{x} day(s)<br>"
            "Cross-Window Relative Z-Score: %{y:.2f}<extra></extra>"
        ),
    )


def _horizon_derivative_series(x, y, *, order):
    """Return finite-difference horizon derivatives scaled to 20 trading days."""
    numeric = pd.DataFrame(
        {
            "x": pd.to_numeric(pd.Series(x), errors="coerce"),
            "y": pd.to_numeric(pd.Series(y), errors="coerce"),
        }
    ).dropna()
    numeric = numeric.drop_duplicates("x").sort_values("x")
    if len(numeric) < max(3, order + 1):
        return pd.Series(dtype=float)
    x_values = numeric["x"].to_numpy(dtype=float)
    derivative = numeric["y"].to_numpy(dtype=float)
    edge_order = 2 if len(numeric) >= 3 else 1
    for _ in range(order):
        derivative = np.gradient(derivative, x_values, edge_order=edge_order)
    return pd.Series(derivative * (20.0 ** order), index=x_values)


def _build_horizon_derivative_trace(x, y, *, base_name, order, visible=True):
    derivative = _horizon_derivative_series(x, y, order=order)
    derivative_label = "First Derivative" if order == 1 else "Second Derivative"
    units = "Z-score / 20 horizon days" if order == 1 else "Z-score / (20 horizon days)^2"
    return build_line_trace(
        x=derivative.index,
        y=derivative.values,
        name=f"{base_name} — {derivative_label}",
        visible=visible,
        width=2,
        hovertemplate=(
            "Lookback horizon: %{x} trading days<br>"
            f"{derivative_label}: %{{y:.3f}} {units}<extra></extra>"
        ),
    )


def _build_zscore_mean_reference_trace(x, y, *, name, visible=True):
    return build_line_trace(
        x=x,
        y=y,
        name=name,
        mode="lines",
        visible=visible,
        width=2,
        dash="dash",
        hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
    )


def _build_zscore_reference_trace(
    x,
    y,
    *,
    name,
    color,
    dash="dot",
    width=1,
    visible=True,
):
    return build_line_trace(
        x=x,
        y=y,
        name=name,
        mode="lines",
        color=color,
        width=width,
        dash=dash,
        visible=visible,
        hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
    )


def _build_zscore_std_reference_traces(x, *, mean_by_window, std_by_window, label_prefix, visible=True):
    mean_series = pd.Series(mean_by_window).reindex(x)
    std_series = pd.Series(std_by_window).reindex(x)
    if mean_series.dropna().empty or std_series.dropna().empty:
        return []

    traces = []
    level_styles = {
        1: "rgba(255, 210, 80, 0.70)",
        2: "rgba(255, 145, 80, 0.60)",
    }
    for level, color in level_styles.items():
        offset = std_series * level
        for sign, label in ((1, "+"), (-1, "-")):
            traces.append(
                _build_zscore_reference_trace(
                    x,
                    mean_series + sign * offset,
                    name=f"{label_prefix} {label}{level} Std Dev",
                    color=color,
                    visible=visible,
                )
            )
    return traces


def _axis_range_for_series(series_collection, *, padding=0.12, min_span=1.0):
    finite_chunks = []
    for values in series_collection:
        if values is None:
            continue
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.size > 0:
            finite_chunks.append(finite_values)

    if not finite_chunks:
        return None

    values = np.concatenate(finite_chunks)
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span <= 0:
        midpoint = (lower + upper) / 2.0
        span = min_span
        lower = midpoint - span / 2.0
        upper = midpoint + span / 2.0

    axis_padding = max(span * padding, min_span * 0.05)
    return [lower - axis_padding, upper + axis_padding]


def _std_reference_series(mean_by_window, std_by_window, level, sign):
    return pd.Series(mean_by_window) + sign * pd.Series(std_by_window) * level


def _add_zscore_zone(fig, row, col, y0, y1, fillcolor):
    fig.add_hrect(
        y0=y0,
        y1=y1,
        fillcolor=fillcolor,
        opacity=1.0,
        line_width=0,
        layer="below",
        row=row,
        col=col,
    )


def _add_current_sharpe_zscore_zones(fig, row, col):
    _add_zscore_zone(fig, row, col, -1, 1, "rgba(211, 211, 211, 0.18)")
    _add_zscore_zone(fig, row, col, -2, -1, "rgba(0, 128, 0, 0.30)")
    _add_zscore_zone(fig, row, col, 1, 2, "rgba(180, 0, 0, 0.30)")
    add_zone_annotation(fig, row, -1, 1, "Neutral", "rgba(235, 235, 235, 0.95)", col=col)
    add_zone_annotation(fig, row, -2, -1, "Accumulate", "rgba(235, 255, 235, 0.95)", col=col)
    add_zone_annotation(fig, row, 1, 2, "Liquidate", "rgba(255, 235, 235, 0.95)", col=col)


def _add_relative_zscore_zones(fig, row, col):
    _add_zscore_zone(fig, row, col, -2, -1.5, "rgba(180, 0, 0, 0.30)")
    _add_zscore_zone(fig, row, col, 1.5, 2, "rgba(0, 128, 0, 0.36)")
    add_zone_annotation(fig, row, -2, -1.5, "Liquidate", "rgba(255, 235, 235, 0.95)", col=col)
    add_zone_annotation(fig, row, 1.5, 2, "Accumulate", "rgba(235, 255, 235, 0.95)", col=col)


def plot_momentum_window_diagnostics_grid_view(
    diagnostics_context,
    *,
    ticker_label="Asset",
    template="plotly_dark",
    ratio_label="Sharpe",
):
    """Compose the primary momentum-window diagnostics grid for one ratio."""
    ratio_label = _coerce_ratio_label(ratio_label)
    context = coerce_momentum_diagnostics_context(diagnostics_context)
    optimal_windows = context["optimal_windows"]
    optimal_windows_int = context["optimal_windows_int"]
    window_sizes = context["window_sizes"]
    highlight_windows = context["highlight_windows"]
    current_sharpe = context["current_sharpe"]
    current_sharpe_zscore = context["current_sharpe_zscore"]
    current_sharpe_zscore_date = context["current_sharpe_zscore_date"]
    sharpe_zscore_mean_by_window = context["sharpe_zscore_mean_by_window"]
    sharpe_zscore_std_by_window = context["sharpe_zscore_std_by_window"]
    mean_sharpe = context["mean_sharpe"]
    median_sharpe = context["median_sharpe"]
    mean_volatility = context["mean_volatility"]
    median_volatility = context["median_volatility"]
    current_sharpe_zscore_title = f"Current {ratio_label} Z-Score by Window"
    if current_sharpe_zscore_date is not None:
        if hasattr(current_sharpe_zscore_date, "strftime"):
            current_sharpe_zscore_date = current_sharpe_zscore_date.strftime("%Y-%m-%d")
        current_sharpe_zscore_title = (
            f"Current {ratio_label} Z-Score by Window "
            f"({current_sharpe_zscore_date})"
        )
    fig = make_subplots(
        rows=5,
        cols=2,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        subplot_titles=(
            "Rolling Optimal Momentum Window",
            "Optimal Window Distribution",
            f"Current vs Mean/Median {ratio_label} by Window",
            "Mean vs Median Volatility by Window",
            current_sharpe_zscore_title,
            f"First Horizon Derivative of Current {ratio_label} Z-Score",
            f"Second Horizon Derivative of Current {ratio_label} Z-Score",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.035,
        row_heights=[0.16, 0.16, 0.34, 0.17, 0.17],
    )

    fig.add_trace(
        _build_optimal_window_trace(
            optimal_windows.index,
            optimal_windows,
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
        ratio_label=ratio_label,
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
        ratio_label=ratio_label,
        visible=True,
    )
    median_sharpe_trace.showlegend = True
    fig.add_trace(
        median_sharpe_trace,
        row=2,
        col=1,
    )
    current_sharpe_trace = _build_current_sharpe_trace(
        current_sharpe.index,
        current_sharpe.values,
        ratio_label=ratio_label,
        visible=True,
    )
    current_sharpe_trace.showlegend = True
    fig.add_trace(
        current_sharpe_trace,
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
    current_sharpe_zscore_trace = _build_current_sharpe_zscore_trace(
        current_sharpe_zscore.index,
        current_sharpe_zscore.values,
        ratio_label=ratio_label,
        visible=True,
    )
    current_sharpe_zscore_trace.showlegend = True
    fig.add_trace(
        current_sharpe_zscore_trace,
        row=3,
        col=1,
    )
    fig.add_trace(
        _build_horizon_derivative_trace(
            current_sharpe_zscore.index,
            current_sharpe_zscore.values,
            base_name=f"Current {ratio_label} Z-Score",
            order=1,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        _build_horizon_derivative_trace(
            current_sharpe_zscore.index,
            current_sharpe_zscore.values,
            base_name=f"Current {ratio_label} Z-Score",
            order=2,
        ),
        row=5,
        col=1,
    )
    if not sharpe_zscore_mean_by_window.dropna().empty:
        sharpe_zscore_mean_trace = _build_zscore_mean_reference_trace(
            current_sharpe_zscore.index,
            sharpe_zscore_mean_by_window,
            name=f"Historical Mean {ratio_label} Z-Score",
            visible=True,
        )
        sharpe_zscore_mean_trace.showlegend = True
        fig.add_trace(
            sharpe_zscore_mean_trace,
            row=3,
            col=1,
        )
    for reference_trace in _build_zscore_std_reference_traces(
        current_sharpe_zscore.index,
        mean_by_window=sharpe_zscore_mean_by_window,
        std_by_window=sharpe_zscore_std_by_window,
        label_prefix=f"Historical {ratio_label} Z-Score",
    ):
        fig.add_trace(
            reference_trace,
            row=3,
            col=1,
        )
    _add_current_sharpe_zscore_zones(fig, 3, 1)
    for row, col in ((1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (5, 1)):
        add_reference_vlines(fig, highlight_windows, row=row, col=col)
    minimum_horizon = min(window_sizes)
    maximum_horizon = max(window_sizes)
    for segment_index, (
        label, region_start, region_end, color
    ) in enumerate(HORIZON_SEGMENT_REGIONS):
        left = max(minimum_horizon, region_start if region_start is not None else minimum_horizon)
        right = min(maximum_horizon, region_end if region_end is not None else maximum_horizon)
        if right <= left:
            continue
        for profile_row in (3, 4, 5):
            region_kwargs = dict(
                x0=left,
                x1=right,
                row=profile_row,
                col=1,
                fillcolor=color,
                opacity=0.06 if segment_index % 2 == 0 else 0.0,
                line_width=0,
                layer="below",
                name=f"quantapp-horizon-region|{left}|{right}",
            )
            if profile_row == 3:
                region_kwargs.update(
                    annotation_text=f"<b>{label}</b>",
                    annotation_position="inside",
                    annotation_y=0.75,
                    annotation_yanchor="middle",
                    annotation_name=(
                        f"quantapp-horizon-region-label|{left}|{right}"
                    ),
                    annotation_font_size=13,
                    annotation_font_color="#e2e8f0",
                    annotation_bgcolor="rgba(15, 23, 42, 0.72)",
                    annotation_bordercolor="#64748b",
                    annotation_borderwidth=1,
                    annotation_borderpad=3,
                )
            fig.add_vrect(**region_kwargs)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Optimal Window Size (Days)", row=1, col=2)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=2, col=1)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=2, col=2)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=3, col=1)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=4, col=1)
    fig.update_xaxes(title_text="Momentum Window Size (Days)", row=5, col=1)
    for linked_row in (4, 5):
        fig.update_xaxes(matches="x5", row=linked_row, col=1)

    row3_range = _axis_range_for_series(
        [
            current_sharpe_zscore,
            sharpe_zscore_mean_by_window,
            *[
                _std_reference_series(sharpe_zscore_mean_by_window, sharpe_zscore_std_by_window, level, sign)
                for level in (1, 2)
                for sign in (1, -1)
            ],
            pd.Series([-2.0, 2.0]),
        ],
        min_span=1.0,
    )
    fig.update_yaxes(title_text="Window Size (Days)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text=f"{ratio_label} Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=2, col=2)
    fig.update_yaxes(
        title_text=f"{ratio_label} Z-Score", row=3, col=1, range=row3_range
    )
    fig.update_yaxes(title_text="Slope (Z / 20d)", row=4, col=1)
    fig.update_yaxes(title_text="Curvature (Z / 20d²)", row=5, col=1)
    for derivative_row in (4, 5):
        fig.add_hline(
            y=0,
            row=derivative_row,
            col=1,
            line_color="rgba(226, 232, 240, 0.45)",
            line_width=1,
            line_dash="dot",
        )

    fig.update_layout(
        title=f"{ticker_label} Momentum Window Diagnostics",
        template=template,
        height=1900,
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

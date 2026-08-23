"""Distribution-oriented views composed from plot-type trace modules."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import add_horizontal_zone, add_zone_annotation, build_visibility_mask
from Quantapp.visualization.traces.line import build_horizontal_level_trace, build_line_trace

HEADER_TOP_MARGIN = 150
HEADER_TITLE_Y = 0.97
HEADER_MENU_Y = 1.08


def _header_margin(top=None):
    return dict(t=HEADER_TOP_MARGIN if top is None else int(top))


def _header_title(text):
    return dict(
        text=str(text),
        x=0.5,
        xanchor="center",
        y=HEADER_TITLE_Y,
        yanchor="top",
    )


def _dropdown_menu(
    *,
    buttons,
    x,
    active=None,
    y=None,
    direction="down",
    showactive=True,
    xanchor="left",
    yanchor="top",
    **overrides,
):
    menu = dict(
        type="dropdown",
        buttons=buttons,
        direction=direction,
        showactive=showactive,
        x=x,
        xanchor=xanchor,
        y=HEADER_MENU_Y if y is None else y,
        yanchor=yanchor,
    )
    if active is not None:
        menu["active"] = active
    menu.update(overrides)
    return menu


def _coerce_window_options(metrics_by_window, window_options):
    if window_options is None:
        return list(metrics_by_window.keys())
    try:
        normalized = [int(window) for window in window_options]
    except Exception as exc:
        raise ValueError("window_options must be iterable integers.") from exc
    return [window for window in normalized if window in metrics_by_window]


def _preferred_numeric_window(options, preferred=200):
    normalized = []
    seen = set()
    for option in options:
        try:
            coerced = int(option)
        except (TypeError, ValueError):
            continue
        if coerced <= 0 or coerced in seen:
            continue
        normalized.append(coerced)
        seen.add(coerced)
    if not normalized:
        return None
    if preferred in seen:
        return preferred
    return max(normalized)


def _add_distribution_reference_lines(fig, row, x_ref, levels, mean_color):
    if len(x_ref) == 0:
        return
    fig.add_trace(
        build_horizontal_level_trace(
            x_ref,
            y_value=0.0,
            name="Mean",
            color=mean_color,
            width=1,
        ),
        row=row,
        col=1,
    )
    for level in levels:
        fig.add_trace(
            build_horizontal_level_trace(
                x_ref,
                y_value=float(level),
                name=f"+{level} Sigma",
                color="rgba(220, 220, 220, 0.55)",
                width=1,
                dash="dot",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            build_horizontal_level_trace(
                x_ref,
                y_value=float(-level),
                name=f"-{level} Sigma",
                color="rgba(220, 220, 220, 0.55)",
                width=1,
                dash="dot",
            ),
            row=row,
            col=1,
        )


def _build_distribution_return_bar_trace(*, x, y, name, marker_color, visible, opacity=0.75):
    trace = go.Bar(
        x=x,
        y=y,
        name=name,
        visible=visible,
        showlegend=False,
    )
    trace.marker_color = marker_color
    trace.opacity = opacity
    return trace


def _build_distribution_band_trace(
    *,
    x,
    y,
    name,
    visible,
    fill=None,
    fillcolor=None,
):
    return build_line_trace(
        x=x,
        y=y,
        name=name,
        color="rgba(148, 163, 184, 0.0)",
        width=1,
        visible=visible,
        fill=fill,
        fillcolor=fillcolor,
        hoverinfo="skip",
    )


def _build_distribution_median_trace(*, x, y, name, color, visible):
    return build_line_trace(
        x=x,
        y=y,
        name=name,
        color=color,
        width=1.6,
        dash="dash",
        visible=visible,
    )


def _build_distribution_metric_trace(*, x, y, name, color, visible):
    return build_line_trace(
        x=x,
        y=y,
        name=name,
        color=color,
        visible=visible,
    )


def _add_distribution_return_panel_traces(
    fig,
    *,
    row,
    window,
    visible,
    daily_return_series,
    return_q10,
    return_q25,
    return_median,
    return_q75,
    return_q90,
    metric_colors,
):
    return_colors = np.where(
        daily_return_series >= 0,
        "rgba(34, 197, 94, 0.45)",
        "rgba(239, 68, 68, 0.45)",
    ).tolist()

    fig.add_trace(
        _build_distribution_return_bar_trace(
            x=daily_return_series.index,
            y=daily_return_series,
            name=f"{window}-Day 1D Returns",
            marker_color=return_colors,
            visible=visible,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        _build_distribution_band_trace(
            x=return_q90.index,
            y=return_q90,
            name=f"{window}-Day 90th Percentile",
            visible=visible,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        _build_distribution_band_trace(
            x=return_q10.index,
            y=return_q10,
            name=f"{window}-Day 10th Percentile",
            visible=visible,
            fill="tonexty",
            fillcolor=metric_colors["outer_band"],
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        _build_distribution_band_trace(
            x=return_q75.index,
            y=return_q75,
            name=f"{window}-Day 75th Percentile",
            visible=visible,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        _build_distribution_band_trace(
            x=return_q25.index,
            y=return_q25,
            name=f"{window}-Day 25th Percentile",
            visible=visible,
            fill="tonexty",
            fillcolor=metric_colors["inner_band"],
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        _build_distribution_median_trace(
            x=return_median.index,
            y=return_median,
            name=f"{window}-Day Rolling Median",
            color=metric_colors["return_median"],
            visible=visible,
        ),
        row=row,
        col=1,
    )


def _add_distribution_zscore_traces(
    fig,
    *,
    row_map,
    window,
    visible,
    skew_series,
    kurtosis_series,
    gini_series,
    metric_colors,
):
    fig.add_trace(
        _build_distribution_metric_trace(
            x=skew_series.index,
            y=skew_series,
            name=f"{window}-Day Skew Z-Score",
            color=metric_colors["skew"],
            visible=visible,
        ),
        row=row_map["skew"],
        col=1,
    )
    fig.add_trace(
        _build_distribution_metric_trace(
            x=kurtosis_series.index,
            y=kurtosis_series,
            name=f"{window}-Day Excess Kurtosis Z-Score",
            color=metric_colors["kurtosis"],
            visible=visible,
        ),
        row=row_map["kurtosis"],
        col=1,
    )
    fig.add_trace(
        _build_distribution_metric_trace(
            x=gini_series.index,
            y=gini_series,
            name=f"{window}-Day Gini Coefficient Z-Score",
            color=metric_colors["gini"],
            visible=visible,
        ),
        row=row_map["gini"],
        col=1,
    )


def plot_distribution_shape_zscores_view(
    metrics_by_window,
    *,
    window_options=None,
    default_window=None,
    ticker_label="Asset",
    template="plotly_dark",
    include_return_panel=True,
):
    """Compose the rolling distribution z-score view."""
    if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
        raise ValueError("metrics_by_window must be a non-empty mapping of window -> metric map.")

    window_options = _coerce_window_options(metrics_by_window, window_options)
    if not window_options:
        raise ValueError("No valid window options available for plotting.")

    if default_window not in window_options:
        default_window = _preferred_numeric_window(window_options) or window_options[0]

    if include_return_panel:
        subplot_titles = (
            "Daily Returns with Rolling Quantile Bands",
            "Rolling Skew Z-Score",
            "Rolling Excess Kurtosis Z-Score",
            "Rolling Gini Coefficient Z-Score",
        )
        row_count = 4
        row_map = {"returns": 1, "skew": 2, "kurtosis": 3, "gini": 4}
        figure_height = 1450
        figure_title = f"{ticker_label} Daily Returns & Distribution Z-Scores ({default_window}-Day Window)"
    else:
        subplot_titles = (
            "Rolling Skew Z-Score",
            "Rolling Excess Kurtosis Z-Score",
            "Rolling Gini Coefficient Z-Score",
        )
        row_count = 3
        row_map = {"skew": 1, "kurtosis": 2, "gini": 3}
        figure_height = 1150
        figure_title = f"{ticker_label} Distribution Z-Scores ({default_window}-Day Window)"

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
    )

    metric_colors = {
        "return_median": "#f8fafc",
        "outer_band": "rgba(148, 163, 184, 0.18)",
        "inner_band": "rgba(148, 163, 184, 0.32)",
        "skew": "#ff7f0e",
        "kurtosis": "#2ca02c",
        "gini": "#d62728",
    }
    traces_per_window = None
    zscore_index_candidates = []
    return_index_candidates = []

    for window in window_options:
        visible = window == default_window
        metric_set = metrics_by_window.get(window, {})
        daily_return_series = metric_set.get("daily_returns", pd.Series(dtype=float)).dropna()
        return_q10 = metric_set.get("return_q10", pd.Series(dtype=float)).dropna()
        return_q25 = metric_set.get("return_q25", pd.Series(dtype=float)).dropna()
        return_median = metric_set.get("return_median", pd.Series(dtype=float)).dropna()
        return_q75 = metric_set.get("return_q75", pd.Series(dtype=float)).dropna()
        return_q90 = metric_set.get("return_q90", pd.Series(dtype=float)).dropna()
        skew_series = metric_set.get("skew_z", pd.Series(dtype=float)).dropna()
        kurtosis_series = metric_set.get("kurtosis_z", pd.Series(dtype=float)).dropna()
        gini_series = metric_set.get("gini_z", pd.Series(dtype=float)).dropna()

        trace_start = len(fig.data)

        if include_return_panel:
            if not daily_return_series.empty:
                return_index_candidates.append(daily_return_series.index)

            _add_distribution_return_panel_traces(
                fig,
                row=row_map["returns"],
                window=window,
                visible=visible,
                daily_return_series=daily_return_series,
                return_q10=return_q10,
                return_q25=return_q25,
                return_median=return_median,
                return_q75=return_q75,
                return_q90=return_q90,
                metric_colors=metric_colors,
            )
        _add_distribution_zscore_traces(
            fig,
            row_map=row_map,
            window=window,
            visible=visible,
            skew_series=skew_series,
            kurtosis_series=kurtosis_series,
            gini_series=gini_series,
            metric_colors=metric_colors,
        )

        added_traces = len(fig.data) - trace_start
        if traces_per_window is None:
            traces_per_window = added_traces

        for series in (skew_series, kurtosis_series, gini_series):
            if not series.empty:
                zscore_index_candidates.append(series.index)

    if traces_per_window is None or traces_per_window <= 0:
        traces_per_window = 9 if include_return_panel else 3

    x_ref = max(zscore_index_candidates, key=len, default=pd.Index([]))
    return_x_ref = max(return_index_candidates, key=len, default=pd.Index([]))

    if len(x_ref) > 0:
        _add_distribution_reference_lines(
            fig,
            row=row_map["skew"],
            x_ref=x_ref,
            levels=(1, 1.5, 2, 3),
            mean_color="rgba(255, 255, 255, 0.80)",
        )
        _add_distribution_reference_lines(
            fig,
            row=row_map["gini"],
            x_ref=x_ref,
            levels=(1, 2, 3),
            mean_color="rgba(255, 255, 255, 0.80)",
        )
        fig.add_trace(
            build_horizontal_level_trace(
                x_ref,
                y_value=0.0,
                name="Mean",
                color="rgba(255, 255, 255, 0.80)",
                width=1,
            ),
            row=row_map["kurtosis"],
            col=1,
        )
        for value in (-1.5, -1, -0.5, 1, 2, 3):
            fig.add_trace(
                build_horizontal_level_trace(
                    x_ref,
                    y_value=float(value),
                    name=f"Kurtosis Level {value}",
                    color="rgba(220, 220, 220, 0.55)",
                    width=1,
                    dash="dot",
                ),
                row=row_map["kurtosis"],
                col=1,
            )

    if include_return_panel and len(return_x_ref) > 0:
        fig.add_trace(
            build_horizontal_level_trace(
                return_x_ref,
                y_value=0.0,
                name="Return Zero",
                color="rgba(248, 250, 252, 0.45)",
                width=1,
                dash="dash",
            ),
            row=row_map["returns"],
            col=1,
        )

    total_traces = len(fig.data)
    first_constant_trace = traces_per_window * len(window_options)
    constant_trace_indices = list(range(first_constant_trace, total_traces))
    buttons = []
    for idx, window in enumerate(window_options):
        visibility = build_visibility_mask(
            total_traces=total_traces,
            active_window_index=idx,
            traces_per_window=traces_per_window,
            constant_trace_indices=constant_trace_indices,
        )
        buttons.append(
            dict(
                label=str(window),
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": _header_title(f"{ticker_label} Rolling Distribution Z-Scores ({window}-Day Window)")},
                ],
            )
        )

    if include_return_panel:
        fig.update_yaxes(title_text="1D Return", tickformat=".1%", row=row_map["returns"], col=1)
    fig.update_yaxes(title_text="Skew Z", row=row_map["skew"], col=1)
    fig.update_yaxes(title_text="Excess Kurtosis Z", row=row_map["kurtosis"], col=1)
    fig.update_yaxes(title_text="Gini Z", row=row_map["gini"], col=1)

    add_horizontal_zone(
        fig,
        row=row_map["skew"],
        col=1,
        y0=-3,
        y1=-1.5,
        fillcolor="rgba(180, 0, 0, 0.30)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )
    add_horizontal_zone(
        fig,
        row=row_map["skew"],
        col=1,
        y0=-1.5,
        y1=1.5,
        fillcolor="rgba(211, 211, 211, 0.18)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )
    add_horizontal_zone(
        fig,
        row=row_map["skew"],
        col=1,
        y0=-1,
        y1=1,
        fillcolor="rgba(169, 169, 169, 0.24)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )
    add_horizontal_zone(
        fig,
        row=row_map["skew"],
        col=1,
        y0=1.5,
        y1=3,
        fillcolor="rgba(180, 0, 0, 0.30)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )
    add_zone_annotation(
        fig,
        row=row_map["skew"],
        col=1,
        y0=-1.5,
        y1=-1,
        text="More Likely to Have Larger Upside Moves",
        font_color="rgba(235, 235, 235, 0.92)",
    )
    add_zone_annotation(
        fig,
        row=row_map["skew"],
        col=1,
        y0=1,
        y1=1.5,
        text="More Likely to Have Larger Downside Moves",
        font_color="rgba(235, 235, 235, 0.92)",
    )
    add_horizontal_zone(
        fig,
        row=row_map["kurtosis"],
        col=1,
        y0=-1.5,
        y1=-1,
        fillcolor="rgba(180, 0, 0, 0.24)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )
    add_horizontal_zone(
        fig,
        row=row_map["kurtosis"],
        col=1,
        y0=-1,
        y1=-0.5,
        fillcolor="rgba(56, 189, 248, 0.16)",
        opacity=1.0,
        line_color="rgba(0, 0, 0, 0)",
        line_width=0,
    )

    fig.update_layout(
        updatemenus=[_dropdown_menu(buttons=buttons, x=0.0)],
        title=_header_title(figure_title),
        height=figure_height,
        margin=_header_margin(),
        template=template,
        showlegend=False,
    )
    return fig

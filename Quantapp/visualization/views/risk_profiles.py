"""High-level risk profile views composed from plot-type trace modules."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..traces.candlestick import (
    build_candlestick_trace_bundle,
    build_candlestick_y_range,
    build_numeric_axis_range,
    build_time_range,
    slice_series_to_range,
)
from ..traces.line import (
    build_percentile_reference_trace,
    build_recovery_time_trace,
    build_textbook_drawdown_trace,
    build_underwater_trace,
)

HEADER_TITLE_Y = 0.97
HEADER_MENU_Y = 1.08


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


def _annotation_payload(annotation):
    if hasattr(annotation, "to_plotly_json"):
        return copy.deepcopy(annotation).to_plotly_json()
    return copy.deepcopy(dict(annotation))


def _percentile_rank(series, value):
    cleaned = pd.Series(series).dropna()
    if cleaned.empty or pd.isna(value):
        return np.nan
    return float(cleaned.le(value).mean() * 100)


def _format_percentile(value):
    return "n/a" if pd.isna(value) else f"{value:.0f}th pctile"


def _line_annotation(*, x, y, text, color, xref, yref, yshift=0):
    if x is None or pd.isna(y):
        return None
    return dict(
        x=x,
        y=float(y),
        xref=xref,
        yref=yref,
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=8,
        yshift=yshift,
        font=dict(size=10, color=color),
        bgcolor="rgba(15, 23, 42, 0.70)",
        bordercolor=color,
        borderwidth=1,
    )


def _constant_axis_series(value):
    if pd.isna(value):
        return None
    return pd.Series([float(value)])


def _rolling_recovery_time(close_series, window):
    def recovery_window(window_values):
        values = np.asarray(window_values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.nan

        peaks = np.maximum.accumulate(values)
        drawdowns = values / peaks - 1.0
        trough_idx = int(np.nanargmin(drawdowns))
        if trough_idx >= values.size - 1:
            return np.nan

        peak_at_trough = peaks[trough_idx]
        recovery_candidates = np.flatnonzero(values[trough_idx + 1:] >= peak_at_trough)
        if recovery_candidates.size == 0:
            return np.nan

        recovery_idx = trough_idx + 1 + int(recovery_candidates[0])
        return float(recovery_idx - trough_idx)

    return close_series.rolling(window=window).apply(recovery_window, raw=True).dropna()


def plot_candlestick_drawdown_recovery_view(
    price_frame,
    metrics_by_window,
    window_options=None,
    default_window=None,
    show_window_menu=True,
    ticker_label="Asset",
    title=None,
    candlestick_period=None,
    candlestick_bollinger_window=50,
    candlestick_mapped_drawdown_windows=None,
    timeframe_options=None,
    default_timeframe_label="Full",
    template="plotly_dark",
):
    """Plot a stacked candlestick, drawdown, and recovery profile with linked zoom."""
    if "Close" not in price_frame.columns:
        raise ValueError("price_frame must contain a 'Close' column.")
    if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
        raise ValueError("metrics_by_window must be a non-empty mapping of window -> metric map.")

    plot_data = price_frame.copy()
    if not isinstance(plot_data.index, pd.DatetimeIndex):
        plot_data.index = pd.to_datetime(plot_data.index)
    plot_data = plot_data.sort_index().dropna(subset=["Close"])
    if plot_data.empty:
        raise ValueError("No non-null close data available for plotting.")

    if window_options is None:
        window_options = list(metrics_by_window.keys())
    else:
        try:
            window_options = [int(window) for window in window_options]
        except Exception as exc:
            raise ValueError("window_options must be iterable integers.") from exc
        window_options = [window for window in window_options if window in metrics_by_window]

    if not window_options:
        raise ValueError("No valid window options available for plotting.")

    if default_window not in window_options:
        default_window = _preferred_numeric_window(window_options) or window_options[0]

    if candlestick_mapped_drawdown_windows is None:
        candlestick_mapped_drawdown_windows = list(window_options)
    else:
        candlestick_mapped_drawdown_windows = [int(window) for window in candlestick_mapped_drawdown_windows]

    if timeframe_options is None:
        timeframe_options = [
            ("Full", None),
            ("10 Years", pd.DateOffset(years=10)),
            ("5 Years", pd.DateOffset(years=5)),
            ("3 Years", pd.DateOffset(years=3)),
            ("2 Years", pd.DateOffset(years=2)),
            ("1 Year", pd.DateOffset(years=1)),
            ("6 Months", pd.DateOffset(months=6)),
            ("3 Months", pd.DateOffset(months=3)),
        ]

    timeframe_labels = [str(label) for label, _ in timeframe_options]
    try:
        default_timeframe_index = next(
            index
            for index, label in enumerate(timeframe_labels)
            if label.lower() == str(default_timeframe_label).lower()
        )
    except StopIteration:
        default_timeframe_index = 0

    combined_title = str(title or f"{ticker_label} Candlestick, Drawdown, and Recovery Profile")

    def window_title(window):
        return f"{combined_title} ({window}-Day Window)"

    candlestick_bundle = build_candlestick_trace_bundle(
        ticker_data=plot_data,
        period=candlestick_period,
        bollinger_window=candlestick_bollinger_window,
        max_drawdown_price_windows=candlestick_mapped_drawdown_windows,
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.30, 0.20],
        subplot_titles=(
            f"Candlestick Price with {candlestick_bollinger_window}-Period Bollinger or Mapped MDD Overlay",
            "Underwater vs Textbook Rolling Max Drawdown",
            "Rolling Recovery Time (Trough to Recovered High)",
        ),
    )
    subplot_title_annotations = [_annotation_payload(annotation) for annotation in fig.layout.annotations]

    for trace in candlestick_bundle["traces"]:
        fig.add_trace(copy.deepcopy(trace), row=1, col=1)

    candlestick_overlay_groups = {
        overlay_name: list(trace_indices)
        for overlay_name, trace_indices in candlestick_bundle["overlay_trace_groups"].items()
    }
    overlay_trace_indices = (
        candlestick_overlay_groups["bollinger"] + candlestick_overlay_groups["mapped_mdd"]
    )
    default_overlay = (
        "mapped_mdd" if candlestick_overlay_groups["mapped_mdd"] else "bollinger"
    )
    for trace_idx in candlestick_overlay_groups["bollinger"]:
        fig.data[trace_idx].visible = default_overlay == "bollinger"
    for trace_idx in candlestick_overlay_groups["mapped_mdd"]:
        fig.data[trace_idx].visible = default_overlay == "mapped_mdd"

    static_annotations = list(subplot_title_annotations)
    if candlestick_bundle["latest_close_annotation"] is not None:
        top_annotation = _annotation_payload(candlestick_bundle["latest_close_annotation"])
        top_annotation["xref"] = "x"
        top_annotation["yref"] = "y"
        static_annotations.append(top_annotation)

    timeframe_menu_x = 0.18 if show_window_menu else 0.0
    overlay_menu_x = 0.38 if show_window_menu else 0.20
    menu_specs = []
    if show_window_menu:
        menu_specs.append(("Damage window", 0.00))
    menu_specs.append(("View timeframe", timeframe_menu_x))
    if overlay_trace_indices:
        menu_specs.append(("Overlay mode", overlay_menu_x))
    for label, x_pos in menu_specs:
        static_annotations.append(
            dict(
                text=label,
                x=x_pos,
                xref="paper",
                y=1.115,
                yref="paper",
                showarrow=False,
                xanchor="left",
            )
        )

    trace_state_map = {}
    annotation_map = {}
    underwater_series_map = {}
    drawdown_series_map = {}
    recovery_series_map = {}
    underwater_percentile_map = {}
    drawdown_percentile_map = {}
    recovery_percentile_map = {}
    lower_panel_trace_indices = []

    for window in window_options:
        rolling_peak = plot_data["Close"].rolling(window=window, min_periods=1).max()
        underwater_series = plot_data["Close"].div(rolling_peak).sub(1.0).dropna()
        drawdown_series = (
            metrics_by_window.get(window, {}).get("max_drawdown", pd.Series(dtype=float)).dropna()
        )
        recovery_series = _rolling_recovery_time(plot_data["Close"], window)

        shared_index = underwater_series.index.intersection(drawdown_series.index)
        visible_underwater = underwater_series.reindex(shared_index).dropna()
        visible_drawdown = drawdown_series.reindex(shared_index).dropna()
        visible_recovery = recovery_series

        underwater_series_map[window] = visible_underwater
        drawdown_series_map[window] = visible_drawdown
        recovery_series_map[window] = visible_recovery

        underwater_p05 = underwater_series.quantile(0.05) if not underwater_series.empty else np.nan
        drawdown_p05 = drawdown_series.quantile(0.05) if not drawdown_series.empty else np.nan
        recovery_p95 = recovery_series.quantile(0.95) if not recovery_series.empty else np.nan

        underwater_percentile_map[window] = underwater_p05
        drawdown_percentile_map[window] = drawdown_p05
        recovery_percentile_map[window] = recovery_p95

        underwater_current = underwater_series.iloc[-1] if not underwater_series.empty else np.nan
        drawdown_current = drawdown_series.iloc[-1] if not drawdown_series.empty else np.nan
        recovery_current = recovery_series.iloc[-1] if not recovery_series.empty else np.nan

        underwater_rank = _percentile_rank(underwater_series, underwater_current)
        drawdown_rank = _percentile_rank(drawdown_series, drawdown_current)
        recovery_rank = _percentile_rank(recovery_series, recovery_current)

        drawdown_x_ref = visible_underwater.index if not visible_underwater.empty else visible_drawdown.index
        recovery_x_ref = visible_recovery.index
        drawdown_last_x = drawdown_x_ref[-1] if len(drawdown_x_ref) > 0 else None
        recovery_last_x = recovery_x_ref[-1] if len(recovery_x_ref) > 0 else None

        trace_positions = []
        window_visible = window == default_window

        fig.add_trace(
            build_underwater_trace(window, visible_underwater.index, visible_underwater, visible=window_visible),
            row=2,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        fig.add_trace(
            build_textbook_drawdown_trace(window, visible_drawdown.index, visible_drawdown, visible=window_visible),
            row=2,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        fig.add_trace(
            build_percentile_reference_trace(
                x=drawdown_x_ref,
                y_value=underwater_p05,
                name="Underwater 5th Percentile",
                color="rgba(14, 165, 233, 0.55)",
                hovertemplate="Underwater 5th percentile: %{y:.1%}<extra></extra>",
                visible=window_visible,
            ),
            row=2,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        fig.add_trace(
            build_percentile_reference_trace(
                x=drawdown_x_ref,
                y_value=drawdown_p05,
                name="Textbook 5th Percentile",
                color="rgba(249, 115, 22, 0.55)",
                hovertemplate="Textbook 5th percentile: %{y:.1%}<extra></extra>",
                visible=window_visible,
            ),
            row=2,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        fig.add_trace(
            build_recovery_time_trace(window, visible_recovery.index, visible_recovery, visible=window_visible),
            row=3,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        fig.add_trace(
            build_percentile_reference_trace(
                x=recovery_x_ref,
                y_value=recovery_p95,
                name="Recovery 95th Percentile",
                color="rgba(34, 197, 94, 0.60)",
                hovertemplate="Recovery 95th percentile: %{y:.0f} sessions<extra></extra>",
                visible=window_visible,
            ),
            row=3,
            col=1,
        )
        lower_panel_trace_indices.append(len(fig.data) - 1)
        trace_positions.append((len(lower_panel_trace_indices) - 1, True))

        line_annotations = [
            _line_annotation(
                x=visible_underwater.index[-1] if not visible_underwater.empty else None,
                y=visible_underwater.iloc[-1] if not visible_underwater.empty else np.nan,
                text="Underwater",
                color="#0ea5e9",
                xref="x2",
                yref="y2",
                yshift=12,
            ),
            _line_annotation(
                x=visible_drawdown.index[-1] if not visible_drawdown.empty else None,
                y=visible_drawdown.iloc[-1] if not visible_drawdown.empty else np.nan,
                text="Textbook Max DD",
                color="#f97316",
                xref="x2",
                yref="y2",
                yshift=-12,
            ),
            _line_annotation(
                x=drawdown_last_x,
                y=underwater_p05,
                text="Underwater 5th pct",
                color="rgba(14, 165, 233, 0.90)",
                xref="x2",
                yref="y2",
                yshift=-12,
            ),
            _line_annotation(
                x=drawdown_last_x,
                y=drawdown_p05,
                text="Textbook 5th pct",
                color="rgba(249, 115, 22, 0.90)",
                xref="x2",
                yref="y2",
                yshift=12,
            ),
            _line_annotation(
                x=visible_recovery.index[-1] if not visible_recovery.empty else None,
                y=visible_recovery.iloc[-1] if not visible_recovery.empty else np.nan,
                text="Recovery Time",
                color="#22c55e",
                xref="x3",
                yref="y3",
                yshift=12,
            ),
            _line_annotation(
                x=recovery_last_x,
                y=recovery_p95,
                text="Recovery 95th pct",
                color="rgba(34, 197, 94, 0.90)",
                xref="x3",
                yref="y3",
                yshift=-12,
            ),
        ]

        trace_state_map[window] = trace_positions
        annotation_map[window] = static_annotations + [
            annotation for annotation in line_annotations if annotation is not None
        ] + [
            dict(
                x=0.995,
                y=0.95,
                xref="paper",
                yref="y2 domain",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="right",
                font=dict(size=11),
                bgcolor="rgba(15, 23, 42, 0.72)",
                bordercolor="rgba(148, 163, 184, 0.35)",
                borderwidth=1,
                text=(
                    f"Latest underwater: {underwater_current:.1%} ({_format_percentile(underwater_rank)})"
                    "<br>"
                    f"Latest textbook: {drawdown_current:.1%} ({_format_percentile(drawdown_rank)})"
                ),
            ),
            dict(
                x=0.995,
                y=0.95,
                xref="paper",
                yref="y3 domain",
                xanchor="right",
                yanchor="top",
                showarrow=False,
                align="right",
                font=dict(size=11),
                bgcolor="rgba(15, 23, 42, 0.72)",
                bordercolor="rgba(148, 163, 184, 0.35)",
                borderwidth=1,
                text=(
                    f"Latest completed recovery: {recovery_current:.0f} sessions ({_format_percentile(recovery_rank)})"
                    if not pd.isna(recovery_current)
                    else "Latest completed recovery: n/a"
                ),
            ),
        ]

    fig.update_xaxes(
        title_text="Date",
        type="date",
        rangeslider=dict(visible=False),
        showgrid=True,
        zeroline=False,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Price", autorange=True, fixedrange=False, row=1, col=1)
    fig.update_xaxes(
        title_text="Date",
        type="date",
        showgrid=True,
        zeroline=False,
        rangebreaks=[dict(bounds=["sat", "mon"])],
        matches="x",
        showticklabels=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(
        title_text="Date",
        type="date",
        rangebreaks=[dict(bounds=["sat", "mon"])],
        matches="x",
        showticklabels=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Sessions", rangemode="tozero", row=3, col=1)
    fig.add_hline(
        y=0,
        row=2,
        col=1,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.45)",
        line_width=1,
    )

    window_buttons = []
    for window in window_options:
        visibility = [False] * len(lower_panel_trace_indices)
        for trace_position, state in trace_state_map[window]:
            visibility[trace_position] = state
        window_buttons.append(
            dict(
                label=f"{window}-Day",
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": _header_title(window_title(window)),
                        "annotations": annotation_map[window],
                    },
                    lower_panel_trace_indices,
                ],
            )
        )

    global_start_candidates = []
    global_end_candidates = []
    if candlestick_bundle["period_start"] is not None and candlestick_bundle["period_end"] is not None:
        global_start_candidates.append(candlestick_bundle["period_start"])
        global_end_candidates.append(candlestick_bundle["period_end"])
    available_indexes = [series.index for series in drawdown_series_map.values() if not series.empty]
    recovery_indexes = [series.index for series in recovery_series_map.values() if not series.empty]
    if available_indexes:
        global_start_candidates.append(min(index[0] for index in available_indexes))
        global_end_candidates.append(max(index[-1] for index in available_indexes))
    if recovery_indexes:
        global_start_candidates.append(min(index[0] for index in recovery_indexes))
        global_end_candidates.append(max(index[-1] for index in recovery_indexes))

    updatemenus = []
    if show_window_menu:
        updatemenus.append(
            _dropdown_menu(
                buttons=window_buttons,
                x=0.0,
                active=window_options.index(default_window),
            )
        )

    if global_start_candidates and global_end_candidates:
        global_start = min(global_start_candidates)
        global_end = max(global_end_candidates)

        def visible_drawdown_axis_series(start=None, end=None):
            axis_series = []
            for window in window_options:
                underwater_slice = slice_series_to_range(underwater_series_map.get(window), start=start, end=end)
                drawdown_slice = slice_series_to_range(drawdown_series_map.get(window), start=start, end=end)
                if not underwater_slice.empty:
                    axis_series.append(underwater_slice)
                    underwater_percentile = _constant_axis_series(underwater_percentile_map.get(window))
                    if underwater_percentile is not None:
                        axis_series.append(underwater_percentile)
                if not drawdown_slice.empty:
                    axis_series.append(drawdown_slice)
                    drawdown_percentile = _constant_axis_series(drawdown_percentile_map.get(window))
                    if drawdown_percentile is not None:
                        axis_series.append(drawdown_percentile)
            return axis_series

        def visible_recovery_axis_series(start=None, end=None):
            axis_series = []
            for window in window_options:
                recovery_slice = slice_series_to_range(recovery_series_map.get(window), start=start, end=end)
                if recovery_slice.empty:
                    continue
                axis_series.append(recovery_slice)
                recovery_percentile = _constant_axis_series(recovery_percentile_map.get(window))
                if recovery_percentile is not None:
                    axis_series.append(recovery_percentile)
            return axis_series

        def combined_range(offset=None):
            visible_start = global_start if offset is None else max(global_start, global_end - offset)
            new_range = build_time_range(global_start, global_end, offset)
            layout_updates = {
                "xaxis.range": new_range,
                "xaxis2.range": new_range,
                "xaxis3.range": new_range,
            }
            price_range = build_candlestick_y_range(
                candlestick_bundle,
                start=visible_start,
                end=global_end,
            )
            if price_range is not None:
                layout_updates["yaxis.range"] = price_range

            drawdown_range = build_numeric_axis_range(
                visible_drawdown_axis_series(start=visible_start, end=global_end),
                include_zero=True,
                padding_ratio=0.08,
            )
            if drawdown_range is not None:
                layout_updates["yaxis2.range"] = drawdown_range

            recovery_range = build_numeric_axis_range(
                visible_recovery_axis_series(start=visible_start, end=global_end),
                include_zero=True,
                padding_ratio=0.08,
            )
            if recovery_range is not None:
                layout_updates["yaxis3.range"] = recovery_range

            return layout_updates

        updatemenus.append(
            _dropdown_menu(
                buttons=[
                    dict(label=label, method="relayout", args=[combined_range(offset)])
                    for label, offset in timeframe_options
                ],
                x=timeframe_menu_x,
                active=default_timeframe_index,
            )
        )

        initial_range = build_time_range(
            global_start,
            global_end,
            timeframe_options[default_timeframe_index][1],
        )
        fig.update_xaxes(range=initial_range, row=1, col=1)
        fig.update_xaxes(range=initial_range, row=2, col=1)
        fig.update_xaxes(range=initial_range, row=3, col=1)
        initial_layout = combined_range(timeframe_options[default_timeframe_index][1])
        if "yaxis.range" in initial_layout:
            fig.update_yaxes(range=initial_layout["yaxis.range"], row=1, col=1)
        if "yaxis2.range" in initial_layout:
            fig.update_yaxes(range=initial_layout["yaxis2.range"], row=2, col=1)
        if "yaxis3.range" in initial_layout:
            fig.update_yaxes(range=initial_layout["yaxis3.range"], row=3, col=1)

    if overlay_trace_indices:
        updatemenus.append(
            _dropdown_menu(
                buttons=[
                    dict(
                        label="Bollinger",
                        method="update",
                        args=[
                            {
                                "visible": ([True] * len(candlestick_overlay_groups["bollinger"]))
                                + ([False] * len(candlestick_overlay_groups["mapped_mdd"]))
                            },
                            {},
                            overlay_trace_indices,
                        ],
                    ),
                    dict(
                        label="Mapped MDD",
                        method="update",
                        args=[
                            {
                                "visible": ([False] * len(candlestick_overlay_groups["bollinger"]))
                                + ([True] * len(candlestick_overlay_groups["mapped_mdd"]))
                            },
                            {},
                            overlay_trace_indices,
                        ],
                    ),
                ],
                x=overlay_menu_x,
                active=0 if default_overlay == "bollinger" else 1,
            )
        )

    fig.update_layout(
        updatemenus=updatemenus,
        title=_header_title(window_title(default_window)),
        height=1650,
        margin=dict(t=220, r=max(candlestick_bundle["right_margin"], 240)),
        template=template,
        showlegend=False,
        hovermode="x unified",
        annotations=annotation_map[default_window],
    )
    return fig

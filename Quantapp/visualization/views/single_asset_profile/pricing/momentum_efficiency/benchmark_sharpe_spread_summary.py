"""Benchmark Sharpe-spread summary view."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import (
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    add_std_annotations,
    add_zone_annotation,
    build_time_range_buttons,
)
from ._shared import dropdown_menu, finalize_dark_figure, header_margin, header_title, preferred_term_key

def plot_multi_benchmark_sharpe_spread_summary(
    summary_zscore_map,
    time_frame_map,
    ticker_label="Asset",
    default_term=None,
    template="plotly_dark",
):
    """
    Plot benchmark Sharpe-spread z-score summary with optional term/time controls.
    """
    if not isinstance(summary_zscore_map, Mapping) or not summary_zscore_map:
        raise ValueError("summary_zscore_map must be a non-empty mapping.")
    if not isinstance(time_frame_map, Mapping) or not time_frame_map:
        raise ValueError("time_frame_map must be a non-empty mapping.")

    term_order = [term for term in time_frame_map.keys() if term in summary_zscore_map]
    if not term_order:
        raise ValueError("No overlapping term keys between summary_zscore_map and time_frame_map.")

    if default_term not in term_order:
        default_term = preferred_term_key(time_frame_map, term_order) or term_order[0]
    benchmark_order = []
    for term in term_order:
        for symbol in summary_zscore_map.get(term, {}).keys():
            if symbol not in benchmark_order:
                benchmark_order.append(symbol)

    fig = make_subplots(rows=1, cols=1)
    benchmark_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    term_default_ranges = {}
    term_full_ranges = {}
    term_trace_bounds = {}

    for term in term_order:
        term_series_map = summary_zscore_map.get(term, {})
        visible = term == default_term
        non_empty_series = []
        x_ref = pd.Index([])

        term_trace_start = len(fig.data)
        for idx, symbol in enumerate(benchmark_order):
            zscore_series = term_series_map.get(symbol, pd.Series(dtype=float)).dropna()
            if not zscore_series.empty:
                non_empty_series.append(zscore_series)
                if len(zscore_series.index) > len(x_ref):
                    x_ref = zscore_series.index

        add_horizontal_zone_trace(fig, 1, x_ref, -2, -1.5, "rgba(180, 0, 0, 0.40)", visible=visible)
        add_horizontal_zone_trace(fig, 1, x_ref, 1.5, 2, "rgba(0, 128, 0, 0.55)", visible=visible)

        for idx, symbol in enumerate(benchmark_order):
            zscore_series = term_series_map.get(symbol, pd.Series(dtype=float)).dropna()
            fig.add_trace(
                go.Scatter(
                    x=zscore_series.index,
                    y=zscore_series,
                    mode="lines",
                    name=symbol,
                    legendgroup=symbol,
                    showlegend=visible,
                    line=dict(color=benchmark_palette[idx % len(benchmark_palette)]),
                    visible=visible,
                ),
                row=1,
                col=1,
            )

        add_mean_reference_line(fig, 1, x_ref, visible=visible)
        add_sigma_reference_lines(fig, 1, x_ref, levels=(0.5, 1, 1.5, 2), visible=visible)

        term_trace_bounds[term] = (term_trace_start, len(fig.data))
        if non_empty_series:
            max_index = max(series.index.max() for series in non_empty_series)
            min_index = min(series.index.min() for series in non_empty_series)
            term_full_ranges[term] = [min_index, max_index]
            term_default_ranges[term] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
        else:
            term_full_ranges[term] = None
            term_default_ranges[term] = None

    add_zone_annotation(fig, 1, -2, -1.5, "Liquidate", "rgba(255, 235, 235, 0.95)")
    add_zone_annotation(fig, 1, 1.5, 2, "Accumulate", "rgba(235, 255, 235, 0.95)")
    add_std_annotations(fig, 1, levels=(0.5, 1, 1.5, 2))

    timeframe_buttons = []
    total_traces = len(fig.data)
    for term in term_order:
        visibility = [False] * total_traces
        start, end = term_trace_bounds[term]
        for trace_idx in range(start, end):
            visibility[trace_idx] = True

        layout_updates = {
            "title": header_title(
                f"{ticker_label} {term.title()} Sharpe Spread Z-Scores vs Benchmarks ({time_frame_map[term]}-Day)"
            ),
            "yaxis": {"title": "Sharpe Spread Z-Score"},
        }
        if term_default_ranges.get(term) is not None:
            layout_updates["xaxis"] = {"range": term_default_ranges[term]}

        timeframe_buttons.append(
            dict(
                label=f"{term.title()} ({time_frame_map[term]})",
                method="update",
                args=[{"visible": visibility}, layout_updates],
            )
        )

    available_ranges = [date_range for date_range in term_full_ranges.values() if date_range is not None]
    if available_ranges:
        global_start = min(date_range[0] for date_range in available_ranges)
        global_end = max(date_range[1] for date_range in available_ranges)
        fig.update_xaxes(range=term_default_ranges[default_term] or [global_start, global_end])
        time_range_menu = dropdown_menu(
            buttons=build_time_range_buttons(global_start, global_end),
            x=0.22 if len(term_order) > 1 else 0.0,
        )
    else:
        time_range_menu = None

    updatemenus = []
    if len(term_order) > 1:
        updatemenus.append(
            dropdown_menu(
                buttons=timeframe_buttons,
                x=0.0,
            )
        )
    if time_range_menu is not None:
        updatemenus.append(time_range_menu)

    fig.update_yaxes(title_text="Sharpe Spread Z-Score", row=1, col=1)
    fig.update_layout(
        title=header_title(
            f"{ticker_label} {default_term.title()} Sharpe Spread Z-Scores vs Benchmarks ({time_frame_map[default_term]}-Day)"
        ),
        height=650,
        margin=header_margin(),
        template=template,
        showlegend=True,
        updatemenus=updatemenus,
    )
    return finalize_dark_figure(fig)


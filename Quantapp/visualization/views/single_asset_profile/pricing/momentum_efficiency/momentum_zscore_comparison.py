"""Momentum z-score comparison view."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import (
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    build_time_range_buttons,
    build_visibility_mask,
)
from ._shared import dropdown_menu, finalize_dark_figure, header_margin, header_title, preferred_window_label

def plot_momentum_zscore_comparison(
    zscore_data,
    ticker_label="Asset",
    default_label=None,
    default_time_label="3 Years",
    sigma_levels=(0.5, 1.0, 1.5),
    show_zones=True,
):
    """
    Plot interactive momentum z-score comparisons with window and time dropdowns.
    """
    if not isinstance(zscore_data, Mapping) or not zscore_data:
        raise ValueError("zscore_data must be a non-empty mapping of label -> pandas Series.")

    prepared_data = {}
    for label, series in zscore_data.items():
        if not isinstance(series, pd.Series):
            raise TypeError(f"zscore_data['{label}'] must be a pandas Series.")
        cleaned = series.dropna().sort_index()
        if not cleaned.empty:
            prepared_data[str(label)] = cleaned

    if not prepared_data:
        raise ValueError("No non-empty momentum z-score series found in zscore_data.")

    labels = list(prepared_data.keys())
    if default_label not in prepared_data:
        default_label = preferred_window_label(prepared_data) or labels[0]

    fig = make_subplots(rows=1, cols=1)
    for label in labels:
        series = prepared_data[label]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                visible=(label == default_label),
            ),
            row=1,
            col=1,
        )

    global_start = min(series.index.min() for series in prepared_data.values())
    global_end = max(series.index.max() for series in prepared_data.values())
    x_ref = prepared_data[default_label].index

    overlay_start_idx = len(fig.data)
    add_sigma_reference_lines(
        fig,
        row=1,
        col=1,
        x_ref=x_ref,
        levels=sigma_levels,
        sigma=1.0,
        center=0.0,
        line_color="rgba(160, 160, 160, 0.50)",
        line_dash="dash",
    )
    add_mean_reference_line(
        fig,
        row=1,
        col=1,
        x_ref=x_ref,
        line_color="rgba(0, 0, 0, 0.85)",
    )

    if show_zones:
        add_horizontal_zone_trace(
            fig,
            row=1,
            col=1,
            x_ref=x_ref,
            y0=-1.5,
            y1=-1.0,
            fillcolor="rgba(0, 170, 0, 0.15)",
        )
        add_horizontal_zone_trace(
            fig,
            row=1,
            col=1,
            x_ref=x_ref,
            y0=1.0,
            y1=1.5,
            fillcolor="rgba(220, 0, 0, 0.15)",
        )

    constant_trace_indices = list(range(overlay_start_idx, len(fig.data)))
    total_traces = len(fig.data)
    buttons_window = []
    for idx, label in enumerate(labels):
        visibility = build_visibility_mask(
            total_traces=total_traces,
            active_window_index=idx,
            traces_per_window=1,
            constant_trace_indices=constant_trace_indices,
        )
        buttons_window.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": header_title(f"{ticker_label} Momentum Z-Score {label}")},
                ],
            )
        )

    buttons_time = build_time_range_buttons(global_start, global_end, axis_count=1)
    year_map = {"10 Years": 10, "5 Years": 5, "3 Years": 3, "1 Year": 1}
    default_years = year_map.get(default_time_label, 3)
    default_start = max(global_start, global_end - pd.DateOffset(years=default_years))

    fig.update_layout(
        updatemenus=[
            dropdown_menu(
                buttons=buttons_window,
                x=0.10,
                direction="down",
            ),
            dropdown_menu(
                buttons=buttons_time,
                x=0.33,
                direction="down",
            ),
        ],
        height=600,
        margin=header_margin(),
        title=header_title(f"{ticker_label} Momentum Z-Score {default_label}"),
        template="plotly_dark",
        yaxis_title="Z-Score",
        xaxis=dict(range=[default_start, global_end]),
    )
    return finalize_dark_figure(fig)


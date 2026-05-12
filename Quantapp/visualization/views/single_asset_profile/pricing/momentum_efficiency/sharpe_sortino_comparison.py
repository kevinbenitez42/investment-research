"""Sharpe and Sortino comparison view."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.analytics.series_utils import calculate_zscore
from Quantapp.visualization.figure_helpers import (
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    build_time_range_buttons,
)
from ._shared import dropdown_menu, finalize_dark_figure, header_margin, header_title, preferred_window_label


def plot_sharpe_sortino_comparison(term_config_map, ticker_label="Asset", default_label=None):
    """
    Plot Sharpe/Sortino z-scores, raw ratios, and Sortino-minus-Sharpe spread z-scores.
    """
    if not isinstance(term_config_map, Mapping) or not term_config_map:
        raise ValueError("term_config_map must be a non-empty mapping.")

    term_labels = list(term_config_map.keys())
    if default_label not in term_config_map:
        default_label = preferred_window_label(term_config_map) or term_labels[0]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.30, 0.38, 0.32],
        subplot_titles=(
            "Risk-Adjusted Return Z-Score Comparison",
            "Rolling Sharpe and Sortino Ratios",
            "Sortino-Sharpe Spread (z-score)",
        ),
    )
    fig.update_xaxes(matches="x3", row=1, col=1)
    fig.update_xaxes(matches="x3", row=2, col=1)

    term_trace_map = {}
    term_default_ranges = {}
    term_full_ranges = {}
    for term_label in term_labels:
        cfg = term_config_map[term_label]
        sharpe = cfg["sharpe"]
        sortino = cfg["sortino"]
        spread = cfg.get("spread", sortino - sharpe)
        sharpe_zscore = cfg.get("sharpe_zscore")
        sortino_zscore = cfg.get("sortino_zscore")
        spread_zscore = cfg.get("spread_zscore")
        time_frame = cfg.get("time_frame", term_label)

        sharpe_clean = sharpe.dropna()
        sortino_clean = sortino.dropna()
        spread_clean = spread.dropna()
        sharpe_zscore = (
            sharpe_zscore
            if isinstance(sharpe_zscore, pd.Series)
            else calculate_zscore(sharpe_clean).dropna()
            if not sharpe_clean.empty
            else pd.Series(dtype=float)
        )
        sortino_zscore = (
            sortino_zscore
            if isinstance(sortino_zscore, pd.Series)
            else calculate_zscore(sortino_clean).dropna()
            if not sortino_clean.empty
            else pd.Series(dtype=float)
        )
        spread_zscore = (
            spread_zscore
            if isinstance(spread_zscore, pd.Series)
            else calculate_zscore(spread_clean).dropna()
            if not spread_clean.empty
            else pd.Series(dtype=float)
        )
        mean_sharpe = sharpe_clean.mean()
        mean_sortino = sortino_clean.mean()

        zscore_x_ref = max(
            [series.index for series in (sharpe_zscore, sortino_zscore, spread_zscore) if not series.empty],
            key=len,
            default=pd.Index([]),
        )
        non_empty_series = [
            series
            for series in (sharpe_clean, sortino_clean, sharpe_zscore, sortino_zscore, spread_zscore)
            if not series.empty
        ]

        visible = term_label == default_label
        trace_indices = []

        fig.add_trace(
            go.Scatter(
                x=sharpe_zscore.index,
                y=sharpe_zscore,
                mode="lines",
                name=f"Sharpe Z-Score ({time_frame}-day)",
                line=dict(color="blue"),
                visible=visible,
                legendgroup="Sharpe Z",
            ),
            row=1,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=sortino_zscore.index,
                y=sortino_zscore,
                mode="lines",
                name=f"Sortino Z-Score ({time_frame}-day)",
                line=dict(color="orange"),
                visible=visible,
                legendgroup="Sortino Z",
            ),
            row=1,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        if len(zscore_x_ref) > 0:
            add_mean_reference_line(
                fig,
                1,
                zscore_x_ref,
                line_color="rgba(0, 0, 0, 0.70)",
                visible=visible,
            )
            trace_indices.append(len(fig.data) - 1)
            add_sigma_reference_lines(
                fig,
                1,
                zscore_x_ref,
                levels=(1, 2),
                line_color="rgba(110, 110, 110, 0.45)",
                visible=visible,
            )
            trace_indices.extend(range(len(fig.data) - 4, len(fig.data)))

        fig.add_trace(
            go.Scatter(
                x=sharpe.index,
                y=sharpe,
                mode="lines",
                name=f"Sharpe Ratio ({time_frame}-day)",
                line=dict(color="blue"),
                visible=visible,
                legendgroup="Sharpe",
            ),
            row=2,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=sortino.index,
                y=sortino,
                mode="lines",
                name=f"Sortino Ratio ({time_frame}-day)",
                line=dict(color="orange"),
                visible=visible,
                legendgroup="Sortino",
            ),
            row=2,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=sharpe.index,
                y=np.full(len(sharpe.index), mean_sharpe),
                mode="lines",
                name=f"Mean Sharpe ({mean_sharpe:.3f})",
                line=dict(color="blue", dash="dash"),
                opacity=0.7,
                visible=visible,
                legendgroup="Sharpe Mean",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=sortino.index,
                y=np.full(len(sortino.index), mean_sortino),
                mode="lines",
                name=f"Mean Sortino ({mean_sortino:.3f})",
                line=dict(color="orange", dash="dash"),
                opacity=0.7,
                visible=visible,
                legendgroup="Sortino Mean",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=spread_zscore.index,
                y=spread_zscore,
                mode="lines",
                name=f"Sortino-Sharpe Spread ({time_frame}-day)",
                line=dict(color="green"),
                visible=visible,
                legendgroup="Spread",
            ),
            row=3,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

        if len(zscore_x_ref) > 0:
            add_horizontal_zone_trace(
                fig,
                3,
                zscore_x_ref,
                2,
                3,
                "rgba(180, 0, 0, 0.20)",
                visible=visible,
            )
            trace_indices.append(len(fig.data) - 1)
            add_horizontal_zone_trace(
                fig,
                3,
                zscore_x_ref,
                -1,
                -0.5,
                "rgba(0, 128, 0, 0.18)",
                visible=visible,
            )
            trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scatter(
                    x=zscore_x_ref,
                    y=np.full(len(zscore_x_ref), 3.0),
                    mode="lines",
                    line=dict(color="rgba(128, 0, 128, 0.75)", dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    visible=visible,
                ),
                row=3,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)
            add_mean_reference_line(
                fig,
                3,
                zscore_x_ref,
                line_color="rgba(0, 0, 0, 0.70)",
                visible=visible,
            )
            trace_indices.append(len(fig.data) - 1)
            for level in (-1, -0.5, 1, 2):
                fig.add_trace(
                    go.Scatter(
                        x=zscore_x_ref,
                        y=np.full(len(zscore_x_ref), float(level)),
                        mode="lines",
                        line=dict(color="rgba(110, 110, 110, 0.45)", dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )
                trace_indices.append(len(fig.data) - 1)

        if not spread_zscore.empty:
            fig.add_trace(
                go.Scatter(
                    x=[spread_zscore.index[-1]],
                    y=[spread_zscore.iloc[-1]],
                    mode="markers+text",
                    text=[f"Latest z: {spread_zscore.iloc[-1]:.2f}"],
                    textposition="middle right",
                    marker=dict(color="purple", size=6),
                    showlegend=False,
                    visible=visible,
                ),
                row=3,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

        term_trace_map[term_label] = trace_indices
        if non_empty_series:
            max_index = max(series.index.max() for series in non_empty_series)
            min_index = min(series.index.min() for series in non_empty_series)
            term_full_ranges[term_label] = [min_index, max_index]
            term_default_ranges[term_label] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
        else:
            term_full_ranges[term_label] = None
            term_default_ranges[term_label] = None

    total_traces = len(fig.data)
    buttons = []
    for term_label in term_labels:
        visibility = [False] * total_traces
        for trace_idx in term_trace_map[term_label]:
            visibility[trace_idx] = True

        layout_updates = {
            "title": header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({term_label})")
        }
        if term_default_ranges.get(term_label) is not None:
            layout_updates["xaxis"] = {"range": term_default_ranges[term_label]}
            layout_updates["xaxis2"] = {"range": term_default_ranges[term_label]}
            layout_updates["xaxis3"] = {"range": term_default_ranges[term_label]}

        buttons.append(
            dict(
                label=term_label,
                method="update",
                args=[
                    {"visible": visibility},
                    layout_updates,
                ],
            )
        )

    available_ranges = [date_range for date_range in term_full_ranges.values() if date_range is not None]
    if available_ranges:
        global_start = min(date_range[0] for date_range in available_ranges)
        global_end = max(date_range[1] for date_range in available_ranges)
        fig.update_xaxes(range=term_default_ranges[default_label] or [global_start, global_end])
        time_range_menu = dropdown_menu(
            buttons=build_time_range_buttons(global_start, global_end, axis_count=3),
            x=0.22 if len(term_labels) > 1 else 0.01,
        )
    else:
        time_range_menu = None

    updatemenus = []
    if len(term_labels) > 1:
        updatemenus.append(
            dropdown_menu(
                buttons=buttons,
                x=0.01,
                active=term_labels.index(default_label),
            )
        )
    if time_range_menu is not None:
        updatemenus.append(time_range_menu)

    fig.update_layout(
        template="plotly_dark",
        height=1100,
        margin=header_margin(),
        legend=dict(x=0.01, y=0.99),
        xaxis3_title="Date",
        yaxis_title="Z-Score",
        yaxis2_title="Ratio Value",
        yaxis3_title="Z-Score",
        title=header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({default_label})"),
        updatemenus=updatemenus,
    )
    return finalize_dark_figure(fig)


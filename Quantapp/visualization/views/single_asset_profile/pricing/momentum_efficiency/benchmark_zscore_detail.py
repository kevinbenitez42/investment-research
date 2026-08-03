"""Benchmark risk-adjusted z-score detail view."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import (
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    add_zone_annotation,
    build_detail_visibility_mask,
    build_time_range_buttons,
)
from ._shared import dropdown_menu, finalize_dark_figure, header_margin, header_title, preferred_term_key, trace_datetime_bounds

def plot_benchmark_zscore_detail(
    detail_zscore_map,
    benchmark_order,
    time_frame_map,
    ticker_label="Asset",
    default_benchmark=None,
    default_term=None,
    template="plotly_dark",
):
    """
    Plot benchmark detail panel: z-score, spread z-score, Sharpe, excess return, and volatility decomposition.
    """
    if not benchmark_order:
        raise ValueError("benchmark_order is empty.")
    if not isinstance(detail_zscore_map, Mapping) or not detail_zscore_map:
        raise ValueError("detail_zscore_map must be a non-empty mapping.")
    if not isinstance(time_frame_map, Mapping) or not time_frame_map:
        raise ValueError("time_frame_map must be a non-empty mapping.")

    term_order = list(time_frame_map.keys())
    if default_term not in term_order:
        default_term = preferred_term_key(time_frame_map, term_order) or term_order[0]
    default_benchmark = default_benchmark if default_benchmark in benchmark_order else benchmark_order[0]

    detail_fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.26, 0.20, 0.18, 0.18, 0.18],
        subplot_titles=(
            "Risk-Adjusted Return Z-Score Comparison",
            "Sharpe Spread Z-Score",
            "Rolling Sharpe Ratio",
            "Annualized Excess Return",
            "Annualized Volatility",
        ),
    )

    term_styles = {
        term_order[0]: {"color": "#1f77b4"},
        term_order[1] if len(term_order) > 1 else term_order[0]: {"color": "#ff7f0e"},
        term_order[2] if len(term_order) > 2 else term_order[0]: {"color": "#2ca02c"},
    }
    for term in term_order:
        term_styles.setdefault(term, {"color": "#7f7f7f"})

    detail_view_order = [(symbol, term) for symbol in benchmark_order for term in term_order]
    default_detail_view = (default_benchmark, default_term)
    traces_per_view = None

    for symbol, term in detail_view_order:
        visible = (symbol, term) == default_detail_view
        metric_set = detail_zscore_map.get(symbol, {}).get(term, {})
        style = term_styles.get(term, {"color": "#7f7f7f"})
        trace_start = len(detail_fig.data)

        asset_zscore = metric_set.get("asset", pd.Series(dtype=float)).dropna()
        benchmark_zscore = metric_set.get("benchmark", pd.Series(dtype=float)).dropna()
        asset_sharpe = metric_set.get("asset_sharpe", pd.Series(dtype=float)).dropna()
        benchmark_sharpe = metric_set.get("benchmark_sharpe", pd.Series(dtype=float)).dropna()
        asset_excess_return = metric_set.get("asset_excess_return", pd.Series(dtype=float)).dropna()
        benchmark_excess_return = metric_set.get("benchmark_excess_return", pd.Series(dtype=float)).dropna()
        asset_volatility = metric_set.get("asset_volatility", pd.Series(dtype=float)).dropna()
        benchmark_volatility = metric_set.get("benchmark_volatility", pd.Series(dtype=float)).dropna()
        sharpe_spread = metric_set.get("sharpe_spread", pd.Series(dtype=float)).dropna()

        detail_fig.add_trace(
            go.Scatter(
                x=asset_zscore.index,
                y=asset_zscore,
                mode="lines",
                name=f"{ticker_label} {term.title()} Sharpe Z-Score",
                legendgroup=f"asset-{term}",
                line=dict(color=style["color"], dash="solid", width=2),
                visible=visible,
                showlegend=visible,
            ),
            row=1,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=benchmark_zscore.index,
                y=benchmark_zscore,
                mode="lines",
                name=f"{symbol} {term.title()} Sharpe Z-Score",
                legendgroup=f"{symbol}-{term}",
                line=dict(color=style["color"], dash="dot", width=2),
                visible=visible,
                showlegend=visible,
            ),
            row=1,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=sharpe_spread.index,
                y=sharpe_spread,
                mode="lines",
                name=f"{symbol} - {ticker_label} {term.title()} Sharpe Spread Z-Score",
                legendgroup=f"{symbol}-{term}",
                line=dict(color=style["color"], dash="dash", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=asset_sharpe.index,
                y=asset_sharpe,
                mode="lines",
                name=f"{ticker_label} {term.title()} Sharpe",
                legendgroup=f"asset-{term}",
                line=dict(color=style["color"], dash="solid", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=benchmark_sharpe.index,
                y=benchmark_sharpe,
                mode="lines",
                name=f"{symbol} {term.title()} Sharpe",
                legendgroup=f"{symbol}-{term}",
                line=dict(color=style["color"], dash="dot", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=asset_excess_return.index,
                y=asset_excess_return,
                mode="lines",
                name=f"{ticker_label} {term.title()} Excess Return",
                legendgroup=f"asset-{term}",
                line=dict(color=style["color"], dash="solid", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=benchmark_excess_return.index,
                y=benchmark_excess_return,
                mode="lines",
                name=f"{symbol} {term.title()} Excess Return",
                legendgroup=f"{symbol}-{term}",
                line=dict(color=style["color"], dash="dot", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=asset_volatility.index,
                y=asset_volatility,
                mode="lines",
                name=f"{ticker_label} {term.title()} Volatility",
                legendgroup=f"asset-{term}",
                line=dict(color=style["color"], dash="solid", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=5,
            col=1,
        )
        detail_fig.add_trace(
            go.Scatter(
                x=benchmark_volatility.index,
                y=benchmark_volatility,
                mode="lines",
                name=f"{symbol} {term.title()} Volatility",
                legendgroup=f"{symbol}-{term}",
                line=dict(color=style["color"], dash="dot", width=2),
                visible=visible,
                showlegend=False,
            ),
            row=5,
            col=1,
        )

        added_traces = len(detail_fig.data) - trace_start
        if traces_per_view is None:
            traces_per_view = added_traces

    dynamic_trace_count = len(detail_fig.data)
    detail_x_ref = max((trace.x for trace in detail_fig.data if len(trace.x) > 0), key=len, default=None)
    if detail_x_ref is not None:
        add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, -1, 1, "rgba(211, 211, 211, 0.18)")
        add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, -2, -1, "rgba(0, 128, 0, 0.30)")
        add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, 1, 2, "rgba(180, 0, 0, 0.30)")
        add_zone_annotation(detail_fig, 1, -1, 1, "Neutral", "rgba(235, 235, 235, 0.95)")
        add_zone_annotation(detail_fig, 1, -0.85, -0.25, "Bullish Neutral (on the way up)", "rgba(235, 235, 235, 0.90)")
        add_zone_annotation(detail_fig, 1, 0.25, 0.85, "Bearish Neutral (on the way down)", "rgba(235, 235, 235, 0.90)")
        add_zone_annotation(detail_fig, 1, -2, -1, "Accumulate", "rgba(235, 255, 235, 0.95)")
        add_zone_annotation(detail_fig, 1, 1, 2, "Liquidate", "rgba(255, 235, 235, 0.95)")
        add_sigma_reference_lines(detail_fig, 1, detail_x_ref)
        add_horizontal_zone_trace(detail_fig, 2, detail_x_ref, -2, -1.5, "rgba(180, 0, 0, 0.30)")
        add_horizontal_zone_trace(detail_fig, 2, detail_x_ref, 1.5, 2, "rgba(0, 128, 0, 0.36)")
        add_zone_annotation(detail_fig, 2, -2, -1.5, "Liquidate", "rgba(255, 235, 235, 0.95)")
        add_zone_annotation(detail_fig, 2, 1.5, 2, "Accumulate", "rgba(235, 255, 235, 0.95)")
        add_sigma_reference_lines(detail_fig, 2, detail_x_ref, levels=(0.5, 1, 1.5, 2))
        add_mean_reference_line(detail_fig, 3, detail_x_ref)
        add_mean_reference_line(detail_fig, 4, detail_x_ref)
        add_mean_reference_line(detail_fig, 5, detail_x_ref)

    total_traces = len(detail_fig.data)
    buttons = []
    single_term_view = len(term_order) == 1
    for idx, (symbol, term) in enumerate(detail_view_order):
        visibility = build_detail_visibility_mask(dynamic_trace_count, total_traces, idx, traces_per_view)
        buttons.append(
            dict(
                label=symbol if single_term_view else f"{symbol} | {term.title()} ({time_frame_map[term]})",
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": header_title(
                            f"{ticker_label} vs {symbol} Risk-Adjusted Return Decomposition [{term.title()} {time_frame_map[term]}-Day]"
                        )
                    },
                ],
            )
        )

    detail_fig.update_yaxes(title_text="Sharpe Z-Score", row=1, col=1)
    detail_fig.update_yaxes(title_text="Spread Z-Score", row=2, col=1)
    detail_fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
    detail_fig.update_yaxes(title_text="Excess Return", tickformat=".1%", row=4, col=1)
    detail_fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=5, col=1)

    detail_start, detail_end = trace_datetime_bounds(detail_fig.data)
    if detail_start is not None and detail_end is not None:
        detail_default_start = max(detail_start, detail_end - pd.DateOffset(years=3))
        detail_fig.update_xaxes(range=[detail_default_start, detail_end])
        time_range_menu = dropdown_menu(
            buttons=build_time_range_buttons(detail_start, detail_end, axis_count=5),
            x=0.18 if len(detail_view_order) > 1 else 0.0,
        )
    else:
        time_range_menu = None

    updatemenus = []
    if len(detail_view_order) > 1:
        updatemenus.append(
            dropdown_menu(
                buttons=buttons,
                x=0.0,
            )
        )
    if time_range_menu is not None:
        updatemenus.append(time_range_menu)

    detail_fig.update_layout(
        title=header_title(
            f"{ticker_label} vs {default_benchmark} Risk-Adjusted Return Decomposition [{default_term.title()} {time_frame_map[default_term]}-Day]"
        ),
        height=1550,
        margin=header_margin(),
        template=template,
        updatemenus=updatemenus,
    )
    return finalize_dark_figure(detail_fig)


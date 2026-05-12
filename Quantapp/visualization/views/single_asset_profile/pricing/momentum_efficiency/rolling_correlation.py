"""Rolling benchmark correlation momentum diagnostic view."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import build_time_range_buttons
from Quantapp.visualization.traces.line import build_line_trace
from ._shared import finalize_dark_figure


def _coerce_series(series) -> pd.Series:
    cleaned = pd.Series(series).dropna().sort_index()
    return cleaned.astype(float) if not cleaned.empty else pd.Series(dtype=float)


def _coerce_correlation_map(rolling_correlation_map):
    if not rolling_correlation_map:
        return {}

    coerced_map = {}
    for term, symbol_map in dict(rolling_correlation_map).items():
        term_map = {}
        for symbol, series in dict(symbol_map or {}).items():
            cleaned = _coerce_series(series)
            if not cleaned.empty:
                term_map[str(symbol)] = cleaned
        if term_map:
            coerced_map[str(term)] = term_map
    return coerced_map


def _ordered_terms(correlation_map, term_order):
    if term_order is None:
        return list(correlation_map.keys())
    return [str(term) for term in term_order if str(term) in correlation_map]


def _ordered_symbols(correlation_map, benchmark_order):
    available = []
    for symbol_map in correlation_map.values():
        for symbol in symbol_map:
            if symbol not in available:
                available.append(symbol)

    if benchmark_order is None:
        return available
    ordered = [str(symbol) for symbol in benchmark_order if str(symbol) in available]
    ordered.extend(symbol for symbol in available if symbol not in ordered)
    return ordered


def _benchmark_colors(symbols):
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    return {
        symbol: palette[idx % len(palette)]
        for idx, symbol in enumerate(symbols)
    }


def _term_window_label(term, time_frame_map):
    if time_frame_map and term in time_frame_map:
        return f"{int(time_frame_map[term])}-Day Rolling Correlation"
    return f"{term} Rolling Correlation"


def _series_date_ranges(correlation_map):
    ranges = []
    for symbol_map in correlation_map.values():
        for series in symbol_map.values():
            if not series.empty:
                ranges.append((series.index.min(), series.index.max()))
    return ranges


def _row_axis_range(row_values):
    if not row_values:
        return None

    row_frame = pd.concat(row_values, axis=1)
    row_min = min(row_frame.min().min(), 0)
    row_max = max(row_frame.max().max(), 0)
    row_span = row_max - row_min
    row_padding = max(row_span * 0.08, 0.03) if pd.notna(row_span) else 0.03
    return [row_min - row_padding, row_max + row_padding]


def plot_rolling_correlation_view(
    rolling_correlation_map,
    *,
    time_frame_map=None,
    term_order=None,
    benchmark_order=None,
    ticker_label="Asset",
    template="plotly_dark",
    title=None,
):
    """Compose stacked rolling-correlation panels for an asset versus benchmarks."""
    correlation_map = _coerce_correlation_map(rolling_correlation_map)
    correlation_rows = _ordered_terms(correlation_map, term_order)
    ordered_symbols = _ordered_symbols(correlation_map, benchmark_order)

    if not correlation_rows:
        fig = go.Figure()
        fig.update_layout(
            title=title or f"{ticker_label} Rolling Correlation vs Benchmarks",
            template=template,
            height=420,
            margin=dict(l=50, r=40, t=90, b=40),
        )
        fig.add_annotation(
            text="No rolling correlation data available for benchmark comparison.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="rgba(220, 220, 220, 0.85)"),
        )
        return finalize_dark_figure(fig)

    benchmark_colors = _benchmark_colors(ordered_symbols)
    fig = make_subplots(
        rows=len(correlation_rows),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            _term_window_label(term, time_frame_map)
            for term in correlation_rows
        ],
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=11, color="rgba(220, 220, 220, 0.90)")

    for row_idx, term in enumerate(correlation_rows, start=1):
        row_series_map = correlation_map[term]
        row_values = []
        visible_symbols = [
            symbol for symbol in ordered_symbols if symbol in row_series_map
        ]

        for symbol_idx, symbol in enumerate(visible_symbols):
            correlation_series = row_series_map[symbol]
            row_values.append(correlation_series)
            fig.add_trace(
                build_line_trace(
                    x=correlation_series.index,
                    y=correlation_series,
                    name=symbol,
                    color=benchmark_colors[symbol],
                    width=2,
                    hovertemplate=(
                        f"Benchmark: {symbol}<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Rolling Correlation: %{y:.2f}<extra></extra>"
                    ),
                    showlegend=row_idx == 1,
                    legendgroup=symbol,
                ),
                row=row_idx,
                col=1,
            )

            correlation_mean = float(correlation_series.mean())
            fig.add_hline(
                y=correlation_mean,
                line_dash="dot",
                line_color=benchmark_colors[symbol],
                line_width=1,
                opacity=0.7,
                row=row_idx,
                col=1,
            )
            fig.add_annotation(
                x=max(0.18, 0.985 - (symbol_idx * 0.18)),
                y=correlation_mean,
                xref="x domain",
                yref="y",
                text=f"{symbol} Mean {correlation_mean:.2f}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10, color=benchmark_colors[symbol]),
                row=row_idx,
                col=1,
            )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(180, 180, 180, 0.65)",
            line_width=1,
            opacity=0.8,
            row=row_idx,
            col=1,
        )
        fig.add_annotation(
            x=0.985,
            y=0,
            xref="x domain",
            yref="y",
            text="Zero",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font=dict(size=10, color="rgba(200, 200, 200, 0.85)"),
            row=row_idx,
            col=1,
        )

        row_axis_range = _row_axis_range(row_values)
        fig.update_yaxes(
            title_text="Correlation",
            range=row_axis_range,
            row=row_idx,
            col=1,
        )

    date_ranges = _series_date_ranges(correlation_map)
    time_range_menu = None
    if date_ranges:
        global_start = min(start for start, _ in date_ranges)
        global_end = max(end for _, end in date_ranges)
        default_start = max(global_start, global_end - pd.DateOffset(years=3))
        for axis_idx in range(1, len(correlation_rows) + 1):
            axis_name = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            fig.layout[axis_name].update(range=[default_start, global_end])
        time_range_menu = dict(
            buttons=build_time_range_buttons(
                global_start,
                global_end,
                axis_count=len(correlation_rows),
            ),
            direction="down",
            showactive=True,
            x=0.01,
            y=1.12,
            xanchor="left",
            yanchor="top",
            active=2,
        )

    fig.update_layout(
        title=title or f"{ticker_label} Rolling Correlation vs Benchmarks",
        template=template,
        height=max(760, 240 * len(correlation_rows) + 140),
        margin=dict(l=50, r=40, t=90, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        updatemenus=[time_range_menu] if time_range_menu is not None else [],
    )
    fig.update_xaxes(title_text="Date", row=len(correlation_rows), col=1)
    return finalize_dark_figure(fig)

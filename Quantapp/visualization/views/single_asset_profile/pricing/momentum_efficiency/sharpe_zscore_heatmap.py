"""Rolling Sharpe z-score heatmap momentum diagnostic view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.traces.line import build_horizontal_level_trace, build_line_trace
from ._shared import finalize_dark_figure


HEATMAP_COLORSCALE = [
    (0.0, "#b2182b"),
    (0.5, "#f7f7f7"),
    (1.0, "#1a9850"),
]


def _coerce_heatmap_frame(matrix) -> pd.DataFrame:
    if matrix is None:
        return pd.DataFrame()
    frame = pd.DataFrame(matrix).copy()
    if frame.empty:
        return frame
    return frame.sort_index(axis=0).sort_index(axis=1)


def _cross_window_mean(matrix: pd.DataFrame) -> pd.Series:
    if matrix.empty:
        return pd.Series(dtype=float)
    return matrix.mean(axis=0, skipna=True).dropna().sort_index()


def _latest_window_series(matrix: pd.DataFrame) -> pd.Series:
    if matrix.empty:
        return pd.Series(dtype=float)

    latest_by_window = {}
    for window, row in matrix.iterrows():
        valid_row = row.dropna()
        latest_by_window[window] = valid_row.iloc[-1] if not valid_row.empty else np.nan
    return pd.Series(latest_by_window, dtype=float).sort_index()


def _collect_heatmap_abs_values(matrices) -> np.ndarray:
    finite_abs_chunks = []
    for matrix in matrices:
        if matrix is None or matrix.empty:
            continue
        values = matrix.to_numpy(dtype=float)
        finite_abs_values = np.abs(values[np.isfinite(values)])
        if finite_abs_values.size > 0:
            finite_abs_chunks.append(finite_abs_values)
    if finite_abs_chunks:
        return np.concatenate(finite_abs_chunks)
    return np.array([], dtype=float)


def _color_scale_stats(matrices):
    abs_values = _collect_heatmap_abs_values(matrices)
    if abs_values.size == 0:
        return 1.0, 1.0
    observed_max = float(abs_values.max())
    color_scale_cap = max(float(np.nanpercentile(abs_values, 97.5)), 1.0)
    return observed_max, color_scale_cap


def _numeric_axis_range(series_collection):
    finite_chunks = []
    for series in series_collection:
        if series is None or len(series) == 0:
            continue
        values = np.asarray(series, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            finite_chunks.append(finite_values)
    if not finite_chunks:
        return [-1.0, 1.0]

    combined_values = np.concatenate(finite_chunks)
    lower_bound = min(float(combined_values.min()), 0.0)
    upper_bound = max(float(combined_values.max()), 0.0)
    span = upper_bound - lower_bound
    padding = max(span * 0.08, 0.25)
    return [lower_bound - padding, upper_bound + padding]


def _heatmap_time_range_buttons(global_start, global_end, axis_count):
    def _time_range(*, months=None, years=None):
        if months is not None:
            start = max(global_start, global_end - pd.DateOffset(months=months))
        elif years is not None:
            start = max(global_start, global_end - pd.DateOffset(years=years))
        else:
            start = global_start

        layout_updates = {}
        for axis_idx in range(1, axis_count + 1):
            axis_name = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
            layout_updates[f"{axis_name}.range"] = [start, global_end]
        return dict(method="relayout", args=[layout_updates])

    return [
        dict(label="10 Years", **_time_range(years=10)),
        dict(label="5 Years", **_time_range(years=5)),
        dict(label="3 Years", **_time_range(years=3)),
        dict(label="1 Year", **_time_range(years=1)),
        dict(label="6 Months", **_time_range(months=6)),
        dict(label="3 Months", **_time_range(months=3)),
        dict(label="1 Month", **_time_range(months=1)),
    ], max(global_start, global_end - pd.DateOffset(months=6))


def _date_ranges(matrices):
    ranges = []
    for matrix in matrices:
        if matrix is not None and not matrix.empty and matrix.columns.size > 0:
            ranges.append((matrix.columns.min(), matrix.columns.max()))
    return ranges


def _window_order(asset_matrix, benchmark_matrices, heatmap_windows):
    if heatmap_windows is not None:
        return list(heatmap_windows)

    windows = []
    for matrix in [asset_matrix, *benchmark_matrices]:
        if matrix is None or matrix.empty:
            continue
        windows.extend(matrix.index.tolist())
    return sorted(dict.fromkeys(windows))


def _benchmark_order(benchmark_heatmap_matrices, benchmark_order):
    available_symbols = [
        symbol
        for symbol, matrix in benchmark_heatmap_matrices.items()
        if matrix is not None and matrix.columns.size > 0
    ]
    if benchmark_order is None:
        return available_symbols
    return [symbol for symbol in benchmark_order if symbol in available_symbols]


def _default_benchmark(benchmark_order, default_benchmark):
    if default_benchmark in benchmark_order:
        return default_benchmark
    return benchmark_order[0] if benchmark_order else None


def _build_heatmap_trace(
    *,
    matrix,
    name,
    colorbar_title,
    colorbar_y,
    color_scale_cap,
    hovertemplate,
    visible=True,
):
    return go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=HEATMAP_COLORSCALE,
        zmid=0,
        zmin=-color_scale_cap,
        zmax=color_scale_cap,
        colorbar=dict(title=colorbar_title, x=1.11, y=colorbar_y, len=0.22),
        hovertemplate=hovertemplate,
        visible=visible,
        showscale=True,
        name=name,
    )


def plot_sharpe_zscore_heatmap_view(
    heatmap_matrix,
    benchmark_heatmap_matrices=None,
    *,
    heatmap_windows=None,
    benchmark_order=None,
    default_benchmark=None,
    ticker_label="Asset",
    template="plotly_dark",
    title=None,
):
    """Compose rolling Sharpe z-score heatmaps with separate mean-summary panels."""
    asset_matrix = _coerce_heatmap_frame(heatmap_matrix)
    benchmark_heatmap_matrices = {
        symbol: _coerce_heatmap_frame(matrix)
        for symbol, matrix in (benchmark_heatmap_matrices or {}).items()
    }

    benchmark_order = _benchmark_order(benchmark_heatmap_matrices, benchmark_order)
    default_benchmark = _default_benchmark(benchmark_order, default_benchmark)
    benchmark_matrices = [benchmark_heatmap_matrices[symbol] for symbol in benchmark_order]
    heatmap_windows = _window_order(asset_matrix, benchmark_matrices, heatmap_windows)

    asset_daily_mean = _cross_window_mean(asset_matrix)
    benchmark_daily_mean = {
        symbol: _cross_window_mean(benchmark_heatmap_matrices[symbol])
        for symbol in benchmark_order
    }
    asset_current_by_window = _latest_window_series(asset_matrix)
    benchmark_current_by_window = {
        symbol: _latest_window_series(benchmark_heatmap_matrices[symbol])
        for symbol in benchmark_order
    }

    asset_observed_max_abs_zscore, asset_color_scale_cap = _color_scale_stats([asset_matrix])
    benchmark_observed_max_abs_zscore, benchmark_color_scale_cap = _color_scale_stats(benchmark_matrices)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.045,
        row_heights=[0.30, 0.12, 0.30, 0.12, 0.16],
        subplot_titles=(
            "Asset Sharpe Z-Score by Rolling Window",
            "Asset Cross-Window Mean Sharpe Z-Score",
            "Benchmark Sharpe Spread Z-Score by Rolling Window",
            "Benchmark Cross-Window Mean Sharpe Spread Z-Score",
            "Current Asset Sharpe Z-Score vs Benchmark Sharpe Spread by Rolling Window",
        ),
    )

    constant_trace_indices = []

    fig.add_trace(
        _build_heatmap_trace(
            matrix=asset_matrix,
            name="Asset Heatmap",
            colorbar_title="Sharpe Z-Score",
            colorbar_y=0.88,
            color_scale_cap=asset_color_scale_cap,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Window: %{y} day(s)<br>"
                "Sharpe Z-Score: %{z:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    constant_trace_indices.append(len(fig.data) - 1)

    asset_mean_x = asset_daily_mean.index if not asset_daily_mean.empty else asset_matrix.columns
    if not asset_daily_mean.empty:
        fig.add_trace(
            build_line_trace(
                x=asset_daily_mean.index,
                y=asset_daily_mean,
                name="Asset Cross-Window Mean",
                color="#38bdf8",
                width=2.2,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Mean Sharpe Z-Score: %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=2,
            col=1,
        )
        constant_trace_indices.append(len(fig.data) - 1)

    if len(asset_mean_x) > 0:
        fig.add_trace(
            build_horizontal_level_trace(
                asset_mean_x,
                y_value=0,
                name="Asset Mean Zero Line",
                color="rgba(226, 232, 240, 0.60)",
                width=1.3,
            ),
            row=2,
            col=1,
        )
        constant_trace_indices.append(len(fig.data) - 1)

    if not asset_current_by_window.empty:
        fig.add_trace(
            build_line_trace(
                x=asset_current_by_window.index,
                y=asset_current_by_window,
                name="Current Asset Sharpe Z-Score",
                color="#22c55e",
                width=2.4,
                mode="lines+markers",
                marker=dict(color="#22c55e", size=6),
                hovertemplate="Window: %{x} day(s)<br>Current Sharpe Z-Score: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=5,
            col=1,
        )
        constant_trace_indices.append(len(fig.data) - 1)

    if heatmap_windows:
        fig.add_trace(
            build_horizontal_level_trace(
                heatmap_windows,
                y_value=0,
                name="Current Z-Score Zero Line",
                color="rgba(226, 232, 240, 0.60)",
                width=1.3,
            ),
            row=5,
            col=1,
        )
        constant_trace_indices.append(len(fig.data) - 1)

    benchmark_trace_bounds = {}
    for symbol in benchmark_order:
        symbol_matrix = benchmark_heatmap_matrices[symbol]
        visible = symbol == default_benchmark
        fig.add_trace(
            _build_heatmap_trace(
                matrix=symbol_matrix,
                name=f"{symbol} Heatmap",
                colorbar_title="Sharpe Spread Z-Score",
                colorbar_y=0.45,
                color_scale_cap=benchmark_color_scale_cap,
                hovertemplate=(
                    "Benchmark: " + symbol + "<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Window: %{y} day(s)<br>"
                    "Sharpe Spread Z-Score: %{z:.2f}<extra></extra>"
                ),
                visible=visible,
            ),
            row=3,
            col=1,
        )
        heatmap_trace_idx = len(fig.data) - 1

        symbol_mean = benchmark_daily_mean.get(symbol, pd.Series(dtype=float))
        fig.add_trace(
            build_line_trace(
                x=symbol_mean.index,
                y=symbol_mean,
                name=f"{symbol} Cross-Window Mean",
                color="#38bdf8",
                width=2.2,
                hovertemplate=(
                    "Benchmark: " + symbol + "<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Mean Sharpe Spread Z-Score: %{y:.2f}<extra></extra>"
                ),
                visible=visible,
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        mean_trace_idx = len(fig.data) - 1

        symbol_current = benchmark_current_by_window.get(symbol, pd.Series(dtype=float))
        fig.add_trace(
            build_line_trace(
                x=symbol_current.index,
                y=symbol_current,
                name=f"Current {symbol} Sharpe Spread Z-Score",
                color="#f59e0b",
                width=2.4,
                mode="lines+markers",
                marker=dict(color="#f59e0b", size=6),
                hovertemplate=(
                    "Benchmark: " + symbol + "<br>"
                    "Window: %{x} day(s)<br>"
                    "Current Sharpe Spread Z-Score: %{y:.2f}<extra></extra>"
                ),
                visible=visible,
                showlegend=False,
            ),
            row=5,
            col=1,
        )
        current_trace_idx = len(fig.data) - 1
        benchmark_trace_bounds[symbol] = (heatmap_trace_idx, mean_trace_idx, current_trace_idx)

    benchmark_mean_x = max(
        [matrix.columns for matrix in benchmark_matrices if matrix.columns.size > 0],
        key=len,
        default=pd.Index([]),
    )
    if len(benchmark_mean_x) > 0:
        fig.add_trace(
            build_horizontal_level_trace(
                benchmark_mean_x,
                y_value=0,
                name="Benchmark Mean Zero Line",
                color="rgba(226, 232, 240, 0.60)",
                width=1.3,
            ),
            row=4,
            col=1,
        )
        constant_trace_indices.append(len(fig.data) - 1)

    base_title = title or f"{ticker_label} Rolling Sharpe Z-Score Heatmaps (1-200 Day Windows)"
    figure_title = base_title
    if default_benchmark is not None:
        figure_title += f" | Benchmark: {default_benchmark}"

    fig.update_layout(
        title=figure_title,
        template=template,
        height=1520,
        margin=dict(l=70, r=150, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )

    for row in (1, 2, 3):
        fig.update_xaxes(matches="x4", showticklabels=False, row=row, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    fig.update_yaxes(title_text="Rolling Window (Days)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(
        title_text="Mean Sharpe Z-Score",
        range=_numeric_axis_range([asset_daily_mean]),
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Rolling Window (Days)", autorange="reversed", row=3, col=1)
    fig.update_yaxes(
        title_text="Mean Spread Z-Score",
        range=_numeric_axis_range(benchmark_daily_mean.values()),
        zeroline=False,
        row=4,
        col=1,
    )

    if heatmap_windows:
        fig.update_xaxes(
            title_text="Rolling Window (Days)",
            range=[min(heatmap_windows), max(heatmap_windows)],
            dtick=20,
            row=5,
            col=1,
        )
    else:
        fig.update_xaxes(title_text="Rolling Window (Days)", row=5, col=1)
    fig.update_yaxes(
        title_text="Z-Score Value",
        range=_numeric_axis_range([asset_current_by_window, *benchmark_current_by_window.values()]),
        zeroline=False,
        row=5,
        col=1,
    )

    updatemenus = []
    if benchmark_order:
        total_traces = len(fig.data)
        benchmark_buttons = []
        for symbol in benchmark_order:
            visibility = [False] * total_traces
            for trace_idx in constant_trace_indices:
                visibility[trace_idx] = True
            heatmap_trace_idx, mean_trace_idx, current_trace_idx = benchmark_trace_bounds[symbol]
            visibility[heatmap_trace_idx] = True
            visibility[mean_trace_idx] = True
            visibility[current_trace_idx] = True
            benchmark_buttons.append(
                dict(
                    label=symbol,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"{base_title} | Benchmark: {symbol}"},
                    ],
                )
            )
        updatemenus.append(
            dict(
                buttons=benchmark_buttons,
                direction="down",
                showactive=True,
                x=0.01,
                y=1.11,
                xanchor="left",
                yanchor="top",
                active=benchmark_order.index(default_benchmark),
            )
        )

    date_ranges = _date_ranges([asset_matrix, *benchmark_matrices])
    if date_ranges:
        global_start = min(start for start, _ in date_ranges)
        global_end = max(end for _, end in date_ranges)
        heatmap_time_range_buttons, default_start = _heatmap_time_range_buttons(
            global_start,
            global_end,
            axis_count=4,
        )
        for row in (1, 2, 3, 4):
            fig.update_xaxes(range=[default_start, global_end], row=row, col=1)
        updatemenus.append(
            dict(
                buttons=heatmap_time_range_buttons,
                direction="down",
                showactive=True,
                x=0.18,
                y=1.11,
                xanchor="left",
                yanchor="top",
                active=4,
            )
        )

    if updatemenus:
        fig.update_layout(updatemenus=updatemenus)

    if asset_observed_max_abs_zscore > asset_color_scale_cap:
        fig.add_annotation(
            text=(
                f"Asset heatmap color scale capped at +/-{asset_color_scale_cap:.2f}; observed max is "
                f"{asset_observed_max_abs_zscore:.2f}."
            ),
            xref="paper",
            yref="paper",
            x=1,
            y=1.14,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=11, color="rgba(220, 220, 220, 0.85)"),
        )

    if 1 in asset_matrix.index and asset_matrix.loc[1].isna().all():
        fig.add_annotation(
            text="Asset 1-day window is blank because rolling Sharpe requires at least two return observations.",
            xref="paper",
            yref="paper",
            x=1,
            y=1.10,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=11, color="rgba(220, 220, 220, 0.85)"),
        )

    if benchmark_order and benchmark_observed_max_abs_zscore > benchmark_color_scale_cap:
        fig.add_annotation(
            text=(
                f"Benchmark heatmap color scale capped at +/-{benchmark_color_scale_cap:.2f}; observed max is "
                f"{benchmark_observed_max_abs_zscore:.2f}."
            ),
            xref="paper",
            yref="paper",
            x=1,
            y=0.62,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=11, color="rgba(220, 220, 220, 0.85)"),
        )

    if benchmark_order:
        default_matrix = benchmark_heatmap_matrices[default_benchmark]
        if 1 in default_matrix.index and default_matrix.loc[1].isna().all():
            fig.add_annotation(
                text="Benchmark 1-day window is blank because rolling Sharpe requires at least two return observations.",
                xref="paper",
                yref="paper",
                x=1,
                y=0.58,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=11, color="rgba(220, 220, 220, 0.85)"),
            )
    else:
        fig.add_annotation(
            text="No benchmark Sharpe spread z-score heatmap data available.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.41,
            showarrow=False,
            font=dict(size=12, color="rgba(220, 220, 220, 0.85)"),
        )

    return finalize_dark_figure(fig)

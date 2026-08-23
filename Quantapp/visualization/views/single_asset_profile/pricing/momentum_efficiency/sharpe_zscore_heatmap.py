"""Rolling Sharpe z-score heatmap momentum diagnostic view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.traces.line import build_horizontal_level_trace, build_line_trace
from ._shared import finalize_dark_figure


MIN_HEATMAP_COLOR_SCALE_CAP = 2.0
HEATMAP_ZONE_COLORS = {
    "green": "#008000",
    "white": "#f7f7f7",
    "red": "#b40000",
}
HEATMAP_COLORBAR_TICK_VALUES = [-1, 0, 1]
HEATMAP_COLORBAR_TICK_TEXT = ["<= -1", "-1 to +1", ">= +1"]
MEAN_PANEL_AXIS_RANGE_ANCHOR = (-2.0, 2.0)
MEAN_PANEL_ZONE_FILLS = {
    "green": "rgba(0, 128, 0, 0.22)",
    "neutral": "rgba(211, 211, 211, 0.10)",
    "red": "rgba(180, 0, 0, 0.22)",
}
MEAN_PANEL_REFERENCE_LEVELS = (-2, -1, 1, 2)
MEAN_PANEL_REFERENCE_LINE_COLOR = "rgba(226, 232, 240, 0.48)"
SIGNAL_ZSCORE_THRESHOLD = 1.0
OPTIMAL_BUY_WINDOW_LINE_COLOR = "#111827"
OPTIMAL_SELL_WINDOW_LINE_COLOR = "#4c1d95"
OPTIMAL_HOLD_WINDOW_LINE_COLOR = "#0f766e"
OPTIMAL_BUY_WINDOW_HALO_COLOR = "rgba(255, 255, 255, 0.88)"
OPTIMAL_BUY_WINDOW_LINE_WIDTH = 2.8
OPTIMAL_BUY_WINDOW_HALO_WIDTH = 6.0


def _colorscale_position(value, color_scale_cap):
    cap = max(abs(float(color_scale_cap)), MIN_HEATMAP_COLOR_SCALE_CAP)
    return float(np.clip((float(value) + cap) / (2.0 * cap), 0.0, 1.0))


def _discrete_heatmap_colorscale(color_scale_cap, *, negative_color, positive_color):
    """Build a three-state scale with hard -1/+1 cutoffs."""
    minus_one = _colorscale_position(-1.0, color_scale_cap)
    plus_one = _colorscale_position(1.0, color_scale_cap)
    return [
        (0.0, negative_color),
        (minus_one, negative_color),
        (minus_one, HEATMAP_ZONE_COLORS["white"]),
        (plus_one, HEATMAP_ZONE_COLORS["white"]),
        (plus_one, positive_color),
        (1.0, positive_color),
    ]


def _coerce_heatmap_frame(matrix) -> pd.DataFrame:
    if matrix is None:
        return pd.DataFrame()
    frame = pd.DataFrame(matrix).copy()
    if frame.empty:
        return frame
    return frame.sort_index(axis=0).sort_index(axis=1)


def _window_from_term_key(term_key):
    coerced = pd.to_numeric(term_key, errors="coerce")
    if pd.notna(coerced):
        return int(coerced)

    digits = "".join(ch if ch.isdigit() else " " for ch in str(term_key)).split()
    return int(digits[0]) if digits else None


def _window_order_from_time_frame_map(time_frame_map):
    if not time_frame_map:
        return None
    return [int(window) for window in time_frame_map.values()]


def _asset_heatmap_matrix_from_context(sharpe_heatmap_context, heatmap_windows=None):
    if not sharpe_heatmap_context:
        return pd.DataFrame(), heatmap_windows

    term_config_map = sharpe_heatmap_context.get("term_config_map", {})
    if heatmap_windows is None:
        heatmap_windows = _window_order_from_time_frame_map(sharpe_heatmap_context.get("time_frame_map"))

    series_by_window = {}
    for term_key, config in term_config_map.items():
        window = config.get("time_frame") or _window_from_term_key(term_key)
        if window is None:
            continue
        series_by_window[int(window)] = config.get("sharpe_zscore", pd.Series(dtype=float))

    matrix = pd.DataFrame(series_by_window).sort_index().T
    if heatmap_windows is not None:
        matrix = matrix.reindex(heatmap_windows)
    return matrix, heatmap_windows


def _heatmap_matrix_from_zscore_frame(zscore_frame, heatmap_windows=None):
    frame = pd.DataFrame(zscore_frame).copy()
    if frame.empty:
        return frame, heatmap_windows

    matrix = frame.sort_index().T
    if heatmap_windows is not None:
        matrix = matrix.reindex(heatmap_windows)
    return matrix, heatmap_windows


def _benchmark_heatmap_matrices_from_zscore_frames(benchmark_zscore_frames, heatmap_windows=None):
    matrices = {}
    for symbol, zscore_frame in (benchmark_zscore_frames or {}).items():
        matrix, heatmap_windows = _heatmap_matrix_from_zscore_frame(
            zscore_frame,
            heatmap_windows=heatmap_windows,
        )
        if matrix.columns.size > 0:
            matrices[symbol] = matrix
    return matrices, heatmap_windows


def _benchmark_heatmap_matrices_from_payload(benchmark_plot_payload, heatmap_windows=None):
    if not benchmark_plot_payload:
        return {}, heatmap_windows

    time_frame_map = benchmark_plot_payload.get("time_frame_map", {})
    if heatmap_windows is None:
        heatmap_windows = _window_order_from_time_frame_map(time_frame_map)

    matrices = {}
    benchmark_order = benchmark_plot_payload.get("benchmark_order", [])
    summary_zscore_map = benchmark_plot_payload.get("summary_zscore_map", {})

    for symbol in benchmark_order:
        series_by_window = {}
        for term_key, term_zscores in summary_zscore_map.items():
            window = time_frame_map.get(term_key) or _window_from_term_key(term_key)
            if window is None:
                continue
            series_by_window[int(window)] = term_zscores.get(symbol, pd.Series(dtype=float))

        symbol_matrix = pd.DataFrame(series_by_window).sort_index().T
        if heatmap_windows is not None:
            symbol_matrix = symbol_matrix.reindex(heatmap_windows)
        if symbol_matrix.columns.size > 0:
            matrices[symbol] = symbol_matrix

    return matrices, heatmap_windows


def _cross_window_mean(matrix: pd.DataFrame) -> pd.Series:
    if matrix.empty:
        return pd.Series(dtype=float)
    return matrix.mean(axis=0, skipna=True).dropna().sort_index()


def _average_signal_window_summary(matrix: pd.DataFrame, *, signal_when: str):
    if matrix.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    window_values = pd.to_numeric(pd.Series(matrix.index), errors="coerce").to_numpy(dtype=float)
    values = matrix.to_numpy(dtype=float)
    finite_mask = np.isfinite(values) & np.isfinite(window_values[:, None])
    if signal_when == "low":
        signal_mask = finite_mask & (values <= -SIGNAL_ZSCORE_THRESHOLD)
    elif signal_when == "high":
        signal_mask = finite_mask & (values >= SIGNAL_ZSCORE_THRESHOLD)
    elif signal_when == "neutral":
        signal_mask = finite_mask & (values > -SIGNAL_ZSCORE_THRESHOLD) & (values < SIGNAL_ZSCORE_THRESHOLD)
    else:
        raise ValueError("signal_when must be 'low', 'high', or 'neutral'.")

    signal_counts = signal_mask.sum(axis=0)
    weighted_windows = np.where(signal_mask, window_values[:, None], np.nan)
    average_windows = np.divide(
        np.nansum(weighted_windows, axis=0),
        signal_counts,
        out=np.full(signal_counts.shape, np.nan, dtype=float),
        where=signal_counts > 0,
    )
    return (
        pd.Series(average_windows, index=matrix.columns, dtype=float),
        pd.Series(signal_counts, index=matrix.columns, dtype=float),
    )


def _optimal_window_traces(
    window_average,
    window_count,
    *,
    name,
    hover_label,
    count_label,
    line_color,
    dash=None,
    visible=True,
):
    clean_average = pd.Series(window_average, dtype=float)
    clean_count = pd.Series(window_count, dtype=float).reindex(clean_average.index).fillna(0)
    hovertemplate = (
        "Date: %{x|%Y-%m-%d}<br>"
        + hover_label
        + ": %{y:.1f} day(s)<br>"
        + count_label
        + ": %{customdata}<extra></extra>"
    )
    halo_trace = build_line_trace(
        x=clean_average.index,
        y=clean_average,
        name=f"{name} Halo",
        color=OPTIMAL_BUY_WINDOW_HALO_COLOR,
        width=OPTIMAL_BUY_WINDOW_HALO_WIDTH,
        dash=dash,
        visible=visible,
        showlegend=False,
        hoverinfo="skip",
    )
    line_trace = build_line_trace(
        x=clean_average.index,
        y=clean_average,
        name=name,
        color=line_color,
        width=OPTIMAL_BUY_WINDOW_LINE_WIDTH,
        dash=dash,
        visible=visible,
        showlegend=True,
        customdata=clean_count.to_numpy(dtype=int),
        hovertemplate=hovertemplate,
    )
    return [halo_trace, line_trace]


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
        return 1.0, MIN_HEATMAP_COLOR_SCALE_CAP
    observed_max = float(abs_values.max())
    color_scale_cap = max(float(np.nanpercentile(abs_values, 97.5)), MIN_HEATMAP_COLOR_SCALE_CAP)
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


def _mean_panel_zone_traces(x_ref, *, name_prefix, negative_fill, positive_fill, visible=True):
    if x_ref is None or len(x_ref) == 0:
        return []

    x0 = x_ref[0]
    x1 = x_ref[-1]
    zone_specs = [
        (-1, 1, MEAN_PANEL_ZONE_FILLS["neutral"], "Neutral"),
        (-2, -1, negative_fill, "Low"),
        (1, 2, positive_fill, "High"),
    ]
    return [
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor=fillcolor,
            hoverinfo="skip",
            showlegend=False,
            visible=visible,
            name=f"{name_prefix} {zone_name} Zone",
        )
        for y0, y1, fillcolor, zone_name in zone_specs
    ]


def _mean_panel_reference_line_traces(x_ref, *, name_prefix, visible=True):
    if x_ref is None or len(x_ref) == 0:
        return []

    return [
        build_horizontal_level_trace(
            x_ref,
            y_value=level,
            name=f"{name_prefix} {level:+g} Z-Score Line",
            color=MEAN_PANEL_REFERENCE_LINE_COLOR,
            width=1,
            dash="dot",
            visible=visible,
        )
        for level in MEAN_PANEL_REFERENCE_LEVELS
    ]


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
    ], max(global_start, global_end - pd.DateOffset(years=3))


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
    negative_color,
    positive_color,
    hovertemplate,
    visible=True,
):
    return go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=_discrete_heatmap_colorscale(
            color_scale_cap,
            negative_color=negative_color,
            positive_color=positive_color,
        ),
        zmid=0,
        zmin=-color_scale_cap,
        zmax=color_scale_cap,
        colorbar=dict(
            title=colorbar_title,
            x=1.11,
            y=colorbar_y,
            len=0.22,
            tickmode="array",
            tickvals=HEATMAP_COLORBAR_TICK_VALUES,
            ticktext=HEATMAP_COLORBAR_TICK_TEXT,
        ),
        hovertemplate=hovertemplate,
        visible=visible,
        showscale=True,
        name=name,
    )


def plot_sharpe_zscore_heatmap_view(
    heatmap_matrix=None,
    benchmark_heatmap_matrices=None,
    *,
    asset_sharpe_zscore_frame=None,
    benchmark_sharpe_zscore_frames=None,
    benchmark_spread_zscore_frames=None,
    sharpe_heatmap_context=None,
    benchmark_plot_payload=None,
    heatmap_windows=None,
    benchmark_order=None,
    default_benchmark=None,
    ticker_label="Asset",
    template="plotly_dark",
    title=None,
):
    """Compose rolling Sharpe z-score heatmaps with separate mean-summary panels."""
    if asset_sharpe_zscore_frame is not None:
        heatmap_matrix, heatmap_windows = _heatmap_matrix_from_zscore_frame(
            asset_sharpe_zscore_frame,
            heatmap_windows=heatmap_windows,
        )
    elif sharpe_heatmap_context is not None:
        heatmap_matrix, heatmap_windows = _asset_heatmap_matrix_from_context(
            sharpe_heatmap_context,
            heatmap_windows=heatmap_windows,
        )

    if benchmark_spread_zscore_frames is not None:
        benchmark_heatmap_matrices, heatmap_windows = _benchmark_heatmap_matrices_from_zscore_frames(
            benchmark_spread_zscore_frames,
            heatmap_windows=heatmap_windows,
        )
        if benchmark_order is None:
            benchmark_order = list(benchmark_spread_zscore_frames)
    elif benchmark_plot_payload is not None:
        benchmark_heatmap_matrices, heatmap_windows = _benchmark_heatmap_matrices_from_payload(
            benchmark_plot_payload,
            heatmap_windows=heatmap_windows,
        )
        if benchmark_order is None:
            benchmark_order = benchmark_plot_payload.get("benchmark_order")
        if default_benchmark is None:
            default_benchmark = benchmark_plot_payload.get("default_benchmark")

    benchmark_sharpe_zscore_matrices = {}
    if benchmark_sharpe_zscore_frames is not None:
        benchmark_sharpe_zscore_matrices, heatmap_windows = _benchmark_heatmap_matrices_from_zscore_frames(
            benchmark_sharpe_zscore_frames,
            heatmap_windows=heatmap_windows,
        )

    asset_matrix = _coerce_heatmap_frame(heatmap_matrix)
    benchmark_heatmap_matrices = {
        symbol: _coerce_heatmap_frame(matrix)
        for symbol, matrix in (benchmark_heatmap_matrices or {}).items()
    }
    benchmark_sharpe_zscore_matrices = {
        symbol: _coerce_heatmap_frame(matrix)
        for symbol, matrix in benchmark_sharpe_zscore_matrices.items()
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
    benchmark_sharpe_daily_mean = {
        symbol: _cross_window_mean(benchmark_sharpe_zscore_matrices.get(symbol, pd.DataFrame()))
        for symbol in benchmark_order
    }
    asset_green_window_average, asset_green_window_count = _average_signal_window_summary(
        asset_matrix,
        signal_when="low",
    )
    asset_red_window_average, asset_red_window_count = _average_signal_window_summary(
        asset_matrix,
        signal_when="high",
    )
    asset_white_window_average, asset_white_window_count = _average_signal_window_summary(
        asset_matrix,
        signal_when="neutral",
    )
    benchmark_green_window_summary = {
        symbol: _average_signal_window_summary(benchmark_heatmap_matrices[symbol], signal_when="high")
        for symbol in benchmark_order
    }
    benchmark_red_window_summary = {
        symbol: _average_signal_window_summary(benchmark_heatmap_matrices[symbol], signal_when="low")
        for symbol in benchmark_order
    }
    benchmark_white_window_summary = {
        symbol: _average_signal_window_summary(benchmark_heatmap_matrices[symbol], signal_when="neutral")
        for symbol in benchmark_order
    }

    asset_observed_max_abs_zscore, asset_color_scale_cap = _color_scale_stats([asset_matrix])
    benchmark_observed_max_abs_zscore, benchmark_color_scale_cap = _color_scale_stats(benchmark_matrices)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.045,
        row_heights=[0.31, 0.19, 0.31, 0.19],
        subplot_titles=(
            "Asset Sharpe Z-Score by Rolling Window",
            "Asset and Benchmark Cross-Window Mean Sharpe Z-Score",
            "Benchmark Sharpe Spread Z-Score by Rolling Window",
            "Benchmark Cross-Window Mean Sharpe Spread Z-Score",
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
            negative_color=HEATMAP_ZONE_COLORS["green"],
            positive_color=HEATMAP_ZONE_COLORS["red"],
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

    for trace in _optimal_window_traces(
        asset_green_window_average,
        asset_green_window_count,
        name="Asset Optimal Buy Window",
        hover_label="Optimal Buy Window",
        count_label="Green windows",
        line_color=OPTIMAL_BUY_WINDOW_LINE_COLOR,
    ):
        fig.add_trace(trace, row=1, col=1)
        constant_trace_indices.append(len(fig.data) - 1)

    for trace in _optimal_window_traces(
        asset_red_window_average,
        asset_red_window_count,
        name="Asset Optimal Sell Window",
        hover_label="Optimal Sell Window",
        count_label="Red windows",
        line_color=OPTIMAL_SELL_WINDOW_LINE_COLOR,
        dash="dash",
    ):
        fig.add_trace(trace, row=1, col=1)
        constant_trace_indices.append(len(fig.data) - 1)

    for trace in _optimal_window_traces(
        asset_white_window_average,
        asset_white_window_count,
        name="Asset Optimal Hold Window",
        hover_label="Optimal Hold Window",
        count_label="White windows",
        line_color=OPTIMAL_HOLD_WINDOW_LINE_COLOR,
        dash="dot",
    ):
        fig.add_trace(trace, row=1, col=1)
        constant_trace_indices.append(len(fig.data) - 1)

    asset_mean_x = asset_daily_mean.index if not asset_daily_mean.empty else asset_matrix.columns
    for trace in _mean_panel_zone_traces(
        asset_mean_x,
        name_prefix="Asset Mean",
        negative_fill=MEAN_PANEL_ZONE_FILLS["green"],
        positive_fill=MEAN_PANEL_ZONE_FILLS["red"],
    ):
        fig.add_trace(trace, row=2, col=1)
        constant_trace_indices.append(len(fig.data) - 1)

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
        for trace in _mean_panel_reference_line_traces(asset_mean_x, name_prefix="Asset Mean"):
            fig.add_trace(trace, row=2, col=1)
            constant_trace_indices.append(len(fig.data) - 1)

    benchmark_mean_x = max(
        [matrix.columns for matrix in benchmark_matrices if matrix.columns.size > 0],
        key=len,
        default=pd.Index([]),
    )
    for trace in _mean_panel_zone_traces(
        benchmark_mean_x,
        name_prefix="Benchmark Mean",
        negative_fill=MEAN_PANEL_ZONE_FILLS["red"],
        positive_fill=MEAN_PANEL_ZONE_FILLS["green"],
    ):
        fig.add_trace(trace, row=4, col=1)
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
                negative_color=HEATMAP_ZONE_COLORS["red"],
                positive_color=HEATMAP_ZONE_COLORS["green"],
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

        signal_window_trace_indices = []
        symbol_green_window_average, symbol_green_window_count = benchmark_green_window_summary[symbol]
        for trace in _optimal_window_traces(
            symbol_green_window_average,
            symbol_green_window_count,
            name=f"{symbol} Optimal Buy Window",
            hover_label="Optimal Buy Window",
            count_label="Green windows",
            line_color=OPTIMAL_BUY_WINDOW_LINE_COLOR,
            visible=visible,
        ):
            fig.add_trace(trace, row=3, col=1)
            signal_window_trace_indices.append(len(fig.data) - 1)

        symbol_red_window_average, symbol_red_window_count = benchmark_red_window_summary[symbol]
        for trace in _optimal_window_traces(
            symbol_red_window_average,
            symbol_red_window_count,
            name=f"{symbol} Optimal Sell Window",
            hover_label="Optimal Sell Window",
            count_label="Red windows",
            line_color=OPTIMAL_SELL_WINDOW_LINE_COLOR,
            dash="dash",
            visible=visible,
        ):
            fig.add_trace(trace, row=3, col=1)
            signal_window_trace_indices.append(len(fig.data) - 1)

        symbol_white_window_average, symbol_white_window_count = benchmark_white_window_summary[symbol]
        for trace in _optimal_window_traces(
            symbol_white_window_average,
            symbol_white_window_count,
            name=f"{symbol} Optimal Hold Window",
            hover_label="Optimal Hold Window",
            count_label="White windows",
            line_color=OPTIMAL_HOLD_WINDOW_LINE_COLOR,
            dash="dot",
            visible=visible,
        ):
            fig.add_trace(trace, row=3, col=1)
            signal_window_trace_indices.append(len(fig.data) - 1)

        mean_trace_indices = []
        symbol_sharpe_mean = benchmark_sharpe_daily_mean.get(symbol, pd.Series(dtype=float))
        if not symbol_sharpe_mean.empty:
            fig.add_trace(
                build_line_trace(
                    x=symbol_sharpe_mean.index,
                    y=symbol_sharpe_mean,
                    name=f"{symbol} Cross-Window Mean Sharpe Z-Score",
                    color="#f97316",
                    width=2.2,
                    dash="dash",
                    hovertemplate=(
                        "Benchmark: " + symbol + "<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Mean Sharpe Z-Score: %{y:.2f}<extra></extra>"
                    ),
                    visible=visible,
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            mean_trace_indices.append(len(fig.data) - 1)

        symbol_mean = benchmark_daily_mean.get(symbol, pd.Series(dtype=float))
        fig.add_trace(
            build_line_trace(
                x=symbol_mean.index,
                y=symbol_mean,
                name=f"{symbol} Cross-Window Mean Sharpe Spread Z-Score",
                color="#38bdf8",
                width=2.2,
                hovertemplate=(
                    "Benchmark: " + symbol + "<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Mean Sharpe Spread Z-Score: %{y:.2f}<extra></extra>"
                ),
                visible=visible,
                showlegend=True,
            ),
            row=4,
            col=1,
        )
        mean_trace_indices.append(len(fig.data) - 1)

        benchmark_trace_bounds[symbol] = (heatmap_trace_idx, signal_window_trace_indices, mean_trace_indices)

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
        for trace in _mean_panel_reference_line_traces(benchmark_mean_x, name_prefix="Benchmark Mean"):
            fig.add_trace(trace, row=4, col=1)
            constant_trace_indices.append(len(fig.data) - 1)

    base_title = title or f"{ticker_label} Rolling Sharpe Z-Score Heatmaps (1-200 Day Windows)"
    figure_title = base_title
    if default_benchmark is not None:
        figure_title += f" | Benchmark: {default_benchmark}"

    fig.update_layout(
        title=figure_title,
        template=template,
        height=1420,
        margin=dict(l=70, r=150, t=110, b=135),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.055,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(11, 15, 20, 0.88)",
            bordercolor="rgba(100, 116, 139, 0.45)",
            borderwidth=1,
        ),
    )

    for row in (1, 2, 3):
        fig.update_xaxes(matches="x4", showticklabels=False, row=row, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    fig.update_yaxes(title_text="Rolling Window (Days)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(
        title_text="Mean Sharpe Z-Score",
        range=_numeric_axis_range(
            [
                asset_daily_mean,
                *benchmark_sharpe_daily_mean.values(),
                MEAN_PANEL_AXIS_RANGE_ANCHOR,
            ]
        ),
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Rolling Window (Days)", autorange="reversed", row=3, col=1)
    fig.update_yaxes(
        title_text="Mean Spread Z-Score",
        range=_numeric_axis_range([*benchmark_daily_mean.values(), MEAN_PANEL_AXIS_RANGE_ANCHOR]),
        zeroline=False,
        row=4,
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
            heatmap_trace_idx, signal_window_trace_indices, mean_trace_indices = benchmark_trace_bounds[symbol]
            visibility[heatmap_trace_idx] = True
            for trace_idx in signal_window_trace_indices:
                visibility[trace_idx] = True
            for trace_idx in mean_trace_indices:
                visibility[trace_idx] = True
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
                active=2,
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

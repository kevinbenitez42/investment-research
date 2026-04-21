"""Reusable trace builders for visualization views."""

from .candlestick import (
    build_candlestick_trace_bundle,
    build_candlestick_y_range,
    build_numeric_axis_range,
    build_time_range,
    slice_series_to_range,
)
from .histogram import build_optimal_window_histogram_trace
from .line import (
    build_line_trace,
    build_mean_sharpe_trace,
    build_mean_volatility_trace,
    build_median_sharpe_trace,
    build_median_volatility_trace,
    build_optimal_window_trace,
    build_percentile_reference_trace,
    build_recovery_time_trace,
    build_textbook_drawdown_trace,
    build_underwater_trace,
)
from .surface import build_sharpe_surface_trace, build_vertical_plane_trace

__all__ = [
    "build_candlestick_trace_bundle",
    "build_candlestick_y_range",
    "build_numeric_axis_range",
    "build_time_range",
    "slice_series_to_range",
    "build_optimal_window_histogram_trace",
    "build_line_trace",
    "build_optimal_window_trace",
    "build_mean_sharpe_trace",
    "build_median_sharpe_trace",
    "build_mean_volatility_trace",
    "build_median_volatility_trace",
    "build_percentile_reference_trace",
    "build_recovery_time_trace",
    "build_textbook_drawdown_trace",
    "build_underwater_trace",
    "build_sharpe_surface_trace",
    "build_vertical_plane_trace",
]

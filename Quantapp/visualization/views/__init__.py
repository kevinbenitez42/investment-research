"""High-level visualization views grouped by analysis domain."""

from .momentum_diagnostics import (
    plot_optimal_window_histogram_view,
    plot_optimal_window_view,
    plot_sharpe_mean_median_view,
    plot_sharpe_surface_view,
    plot_volatility_mean_median_view,
)
from .risk_profiles import plot_candlestick_drawdown_recovery_view
from .volatility import plot_vix_fix_bands

__all__ = [
    "plot_candlestick_drawdown_recovery_view",
    "plot_optimal_window_view",
    "plot_optimal_window_histogram_view",
    "plot_sharpe_mean_median_view",
    "plot_volatility_mean_median_view",
    "plot_sharpe_surface_view",
    "plot_vix_fix_bands",
]

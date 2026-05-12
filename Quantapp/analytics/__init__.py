"""Analytics and computation helpers."""

try:
    from .metric import Algorithm, Metric
except ModuleNotFoundError:  # Optional dependency path, e.g. investpy not installed.
    Algorithm = None
    Metric = None
from .close_analytics import CloseAnalytics
from .helper import Helper
from .momentum_analytics import MomentumAnalytics
from .risk_relative_analytics import RiskRelativeAnalytics
from .risk_distribution_analytics import RiskDistributionAnalytics
from .ohlc_analytics import OHLCAnalytics
from .rolling import Rolling, TimeSeriesAnalytics
from .series_transforms import SeriesTransforms
from .series_utils import (
    calculate_historical_var_metrics,
    calculate_max_drawdown,
    calculate_textbook_rolling_max_drawdown,
    calculate_window_metrics,
    calculate_zscore,
    gini_coefficient,
    zscore,
)

__all__ = [
    "Helper",
    "CloseAnalytics",
    "OHLCAnalytics",
    "Rolling",
    "TimeSeriesAnalytics",
    "MomentumAnalytics",
    "RiskRelativeAnalytics",
    "RiskDistributionAnalytics",
    "SeriesTransforms",
    "Algorithm",
    "Metric",
    "calculate_zscore",
    "zscore",
    "calculate_historical_var_metrics",
    "calculate_max_drawdown",
    "calculate_textbook_rolling_max_drawdown",
    "gini_coefficient",
    "calculate_window_metrics",
]

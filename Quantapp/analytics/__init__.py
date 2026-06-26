"""Analytics and computation helpers."""

try:
    from .metric import Algorithm, Metric
except ModuleNotFoundError:  # Optional dependency path, e.g. investpy not installed.
    Algorithm = None
    Metric = None
from .helper import Helper
from .rolling import Rolling, TimeSeriesAnalytics
from .series_transforms import SeriesTransforms
from .series_utils import (
    calculate_historical_var_metrics,
    calculate_max_drawdown,
    calculate_rolling_recovery_time,
    calculate_textbook_rolling_max_drawdown,
    calculate_zscore,
    gini_coefficient,
    zscore,
)

__all__ = [
    "Helper",
    "Rolling",
    "TimeSeriesAnalytics",
    "SeriesTransforms",
    "Algorithm",
    "Metric",
    "calculate_zscore",
    "zscore",
    "calculate_historical_var_metrics",
    "calculate_max_drawdown",
    "calculate_rolling_recovery_time",
    "calculate_textbook_rolling_max_drawdown",
    "gini_coefficient",
]

"""Compatibility facade for close-based and OHLC rolling analytics."""

from __future__ import annotations

from .close_analytics import (
    CloseAnalytics,
    _calculate_excess_returns,
    _coerce_periodic_risk_free_rate,
    _coerce_price_frame,
    _normalize_windows,
    _rolling_downside_deviation,
    _rolling_series_statistic_frame,
    _rolling_sortino_ratio_frame,
)
from .ohlc_analytics import OHLCAnalytics


class TimeSeriesAnalytics(CloseAnalytics, OHLCAnalytics):
    """Backward-compatible facade combining close-based and OHLC rolling analytics."""


Rolling = TimeSeriesAnalytics

__all__ = [
    "CloseAnalytics",
    "OHLCAnalytics",
    "Rolling",
    "TimeSeriesAnalytics",
    "_calculate_excess_returns",
    "_coerce_periodic_risk_free_rate",
    "_coerce_price_frame",
    "_normalize_windows",
    "_rolling_downside_deviation",
    "_rolling_series_statistic_frame",
    "_rolling_sortino_ratio_frame",
]

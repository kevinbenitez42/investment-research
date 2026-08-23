"""Backward-compatible rolling analytics facade."""

from __future__ import annotations

from . import compute


class Rolling:
    """Compatibility wrapper around the current ``Quantapp.analytics.compute`` API."""

    latest = staticmethod(compute.latest)
    latest_frame = staticmethod(compute.latest_frame)
    latest_by_asset = staticmethod(compute.latest_by_asset)
    rolling = staticmethod(compute.rolling)
    rolling_windows = staticmethod(compute.rolling_windows)
    rolling_frame = staticmethod(compute.rolling_frame)
    rolling_by_asset = staticmethod(compute.rolling_by_asset)


TimeSeriesAnalytics = Rolling

__all__ = ["Rolling", "TimeSeriesAnalytics"]

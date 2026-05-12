"""Apply metric functions to pandas Series and DataFrames."""

from __future__ import annotations

from importlib import import_module

import pandas as pd


def _validate_metric(metric):
    if not callable(metric):
        raise TypeError("metric must be callable.")


def _validate_window(window: int) -> int:
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    return window


def _normalize_min_periods(min_periods: int | None, window: int) -> int:
    if min_periods is None:
        return window
    min_periods = int(min_periods)
    if min_periods <= 0:
        raise ValueError("min_periods must be a positive integer.")
    if min_periods > window:
        raise ValueError("min_periods cannot be greater than window.")
    return min_periods


def _resolve_asset_level(columns: pd.Index, asset_level: str | int) -> str | int:
    if not isinstance(columns, pd.MultiIndex):
        raise TypeError("data must have MultiIndex columns for by-asset calculations.")

    if isinstance(asset_level, str):
        if asset_level in columns.names:
            return asset_level
        if len(columns.names) > 1:
            return 1
        raise ValueError(f"asset_level {asset_level!r} was not found in column level names.")

    return asset_level


def latest(data, metric, *args, dropna: bool = True, **kwargs):
    """
    Apply a scalar metric to the full history.

    Returns a scalar for Series input and one scalar per column for DataFrame input.
    """
    _validate_metric(metric)

    if isinstance(data, pd.Series):
        series = data.dropna() if dropna else data
        return metric(series, *args, **kwargs)

    if isinstance(data, pd.DataFrame):
        return data.apply(
            lambda column: latest(
                column,
                metric,
                *args,
                dropna=dropna,
                **kwargs,
            )
        )

    raise TypeError("data must be a pandas Series or DataFrame.")


def latest_frame(data, metric, *args, dropna: bool = True, **kwargs):
    """
    Apply a scalar metric to a full DataFrame history.

    Use this for metrics that consume a whole frame, such as single-asset OHLCV metrics.
    """
    _validate_metric(metric)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    frame = data.dropna() if dropna else data.copy()
    if frame.empty:
        raise ValueError("data is empty after dropping NaNs.")
    return metric(frame, *args, **kwargs)


def rolling(data, metric, window: int, *args, min_periods: int | None = None, dropna: bool = True, **kwargs):
    """
    Apply a scalar metric over rolling windows.

    Returns a Series for Series input and a DataFrame for DataFrame input.
    """
    _validate_metric(metric)

    window = _validate_window(window)
    min_periods = _normalize_min_periods(min_periods, window)

    if isinstance(data, pd.Series):
        def apply_calculation(window_values):
            series = window_values.dropna() if dropna else window_values
            if series.empty:
                return float("nan")
            return metric(series, *args, **kwargs)

        return data.rolling(window=window, min_periods=min_periods).apply(
            apply_calculation,
            raw=False,
        )

    if isinstance(data, pd.DataFrame):
        return data.apply(
            lambda column: rolling(
                column,
                metric,
                window,
                *args,
                min_periods=min_periods,
                dropna=dropna,
                **kwargs,
            )
        )

    raise TypeError("data must be a pandas Series or DataFrame.")


def rolling_frame(data, metric, window: int, *args, min_periods: int | None = None, dropna: bool = True, **kwargs):
    """
    Apply a scalar metric over rolling DataFrame windows.

    Use this for metrics that consume a whole frame window, such as single-asset OHLCV metrics.
    """
    _validate_metric(metric)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    window = _validate_window(window)
    min_periods = _normalize_min_periods(min_periods, window)

    values = []
    for end_position in range(len(data)):
        start_position = max(0, end_position - window + 1)
        window_frame = data.iloc[start_position : end_position + 1]

        if len(window_frame) < min_periods:
            values.append(float("nan"))
            continue

        if dropna:
            window_frame = window_frame.dropna()
            if len(window_frame) < min_periods:
                values.append(float("nan"))
                continue

        values.append(metric(window_frame, *args, **kwargs))

    return pd.Series(values, index=data.index)


def latest_by_asset(data, metric, *args, asset_level: str | int = "symbol", dropna: bool = True, **kwargs):
    """
    Apply a scalar frame metric to each asset in a MultiIndex-column panel.

    The default expects columns shaped like (field, symbol), with a level named "symbol"
    or the asset symbols in column level 1.
    """
    _validate_metric(metric)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    resolved_asset_level = _resolve_asset_level(data.columns, asset_level)
    results = {}
    for asset in data.columns.get_level_values(resolved_asset_level).unique():
        asset_frame = data.xs(asset, axis=1, level=resolved_asset_level)
        results[asset] = latest_frame(
            asset_frame,
            metric,
            *args,
            dropna=dropna,
            **kwargs,
        )

    return pd.Series(results)


def rolling_by_asset(
    data,
    metric,
    window: int,
    *args,
    asset_level: str | int = "symbol",
    min_periods: int | None = None,
    dropna: bool = True,
    **kwargs,
):
    """
    Apply a scalar frame metric over rolling windows for each asset in a panel.

    The default expects columns shaped like (field, symbol), with a level named "symbol"
    or the asset symbols in column level 1.
    """
    _validate_metric(metric)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    resolved_asset_level = _resolve_asset_level(data.columns, asset_level)
    results = {}
    for asset in data.columns.get_level_values(resolved_asset_level).unique():
        asset_frame = data.xs(asset, axis=1, level=resolved_asset_level)
        results[asset] = rolling_frame(
            asset_frame,
            metric,
            window,
            *args,
            min_periods=min_periods,
            dropna=dropna,
            **kwargs,
        )

    return pd.DataFrame(results, index=data.index)


_LAZY_EXPORTS = {
    "Algorithm": ("Quantapp.analytics.metric", "Algorithm"),
    "Metric": ("Quantapp.analytics.metric", "Metric"),
    "CloseAnalytics": ("Quantapp.analytics.close_analytics", "CloseAnalytics"),
    "Helper": ("Quantapp.analytics.helper", "Helper"),
    "MomentumAnalytics": ("Quantapp.analytics.momentum_analytics", "MomentumAnalytics"),
    "RiskRelativeAnalytics": ("Quantapp.analytics.risk_relative_analytics", "RiskRelativeAnalytics"),
    "RiskDistributionAnalytics": ("Quantapp.analytics.risk_distribution_analytics", "RiskDistributionAnalytics"),
    "OHLCAnalytics": ("Quantapp.analytics.ohlc_analytics", "OHLCAnalytics"),
    "Rolling": ("Quantapp.analytics.rolling", "Rolling"),
    "TimeSeriesAnalytics": ("Quantapp.analytics.rolling", "TimeSeriesAnalytics"),
    "SeriesTransforms": ("Quantapp.analytics.series_transforms", "SeriesTransforms"),
}


def __getattr__(name: str):
    """Load legacy compatibility exports only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute


__all__ = [
    "latest",
    "latest_frame",
    "latest_by_asset",
    "rolling",
    "rolling_frame",
    "rolling_by_asset",
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
]

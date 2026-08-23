"""Vectorized implementations for multi-window rolling metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _metric_name(metric) -> str | None:
    name = getattr(metric, "__name__", None)
    return str(name).strip().lower() if name else None


def get_rolling_windows_optimization(metric):
    """Return an optimized multi-window implementation for a scalar metric."""
    return _ROLLING_WINDOWS_OPTIMIZATIONS.get(_metric_name(metric))


def rolling_sharpe(data, windows, risk_free_rate, *args, min_periods=None, dropna=True, **kwargs):
    """
    Vectorized multi-window implementation matching Metric.sharpe.

    Metric.sharpe currently accepts risk_free_rate but does not use it, so this
    helper preserves that behavior while avoiding one Python call per window row.
    """
    if args or kwargs or min_periods is not None or not dropna:
        return None
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return None

    results = {}
    log_returns = np.log(data / data.shift())
    for window in windows:
        if window <= 1:
            results[window] = data.astype(float) * np.nan
            continue

        first_value = data.shift(window - 1)
        portfolio_return = data.sub(first_value).div(first_value.abs())
        portfolio_std = log_returns.rolling(window=window - 1, min_periods=window - 1).std()
        sharpe = portfolio_return.div(portfolio_std).mul(np.sqrt(window / 252))
        results[window] = sharpe.replace([np.inf, -np.inf], np.nan)

    if isinstance(data, pd.Series):
        return pd.DataFrame(results, index=data.index)

    return pd.concat(results, axis=1, names=["window"])


_ROLLING_WINDOWS_OPTIMIZATIONS = {
    "sharpe": rolling_sharpe,
}

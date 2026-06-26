"""Reusable time-series statistics helpers for notebook and library workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import compute
from .metric import Metric

_metric = Metric()


def coerce_close_series(data, argument_name: str = "close_series") -> pd.Series:
    """Return a clean close-price Series from a Series or OHLC DataFrame."""
    if isinstance(data, pd.Series):
        close = data
    elif isinstance(data, pd.DataFrame):
        if "Close" not in data.columns:
            if argument_name == "close_series":
                raise ValueError("DataFrame input must contain a 'Close' column.")
            raise ValueError(f"{argument_name} DataFrame must contain a 'Close' column.")
        close = data["Close"]
    else:
        if argument_name == "close_series":
            raise TypeError("close_series must be a pandas Series or DataFrame with 'Close'.")
        raise TypeError(f"{argument_name} must be a pandas Series or DataFrame.")

    close = close.dropna()
    if close.empty:
        raise ValueError(f"{argument_name} is empty after dropping NaNs.")
    return close.sort_index()


def coerce_series(data, argument_name: str = "series", preferred_column: str | None = None) -> pd.Series:
    """Coerce Series-like input to a Series, allowing one-column DataFrames."""
    if isinstance(data, pd.Series):
        return data

    if isinstance(data, pd.DataFrame):
        if preferred_column is not None and preferred_column in data.columns:
            out = data[preferred_column]
            if isinstance(out, pd.DataFrame):
                return out.iloc[:, 0]
            return out

        if data.shape[1] == 1:
            return data.iloc[:, 0]

        raise TypeError(
            f"{argument_name} must be a Series or single-column DataFrame. "
            f"Received DataFrame with columns: {list(data.columns)}"
        )

    raise TypeError(f"{argument_name} must be a pandas Series.")


def coerce_datetime_index(data):
    """Return a copy with a naive, sorted DatetimeIndex."""
    out = data.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def calculate_zscore(series: pd.Series) -> pd.Series:
    """Calculate z-score with safe handling for zero/NaN standard deviation."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(index=series.index, data=np.nan)
    return (series - series.mean()) / std


def calculate_max_drawdown(price_series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling max drawdown over a window."""
    rolling_max = price_series.rolling(window=window).max()
    drawdown = price_series / rolling_max - 1
    return drawdown.rolling(window=window).min()


def calculate_textbook_rolling_max_drawdown(price_series: pd.Series, window: int = 21) -> pd.Series:
    """Textbook rolling max drawdown computed independently inside each trailing window."""
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    return compute.rolling(price_series, metric=_metric.textbook_window_drawdown, window=window, dropna=False)


def calculate_rolling_recovery_time(price_series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling sessions from trough to recovered high inside each trailing window."""
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    return compute.rolling(price_series, metric=_metric.window_recovery_time, window=window, dropna=False)


def gini_coefficient(array) -> float:
    """Gini coefficient on absolute values."""
    values = np.asarray(array, dtype=float)
    values = np.abs(values)
    if values.size == 0:
        return np.nan
    sorted_array = np.sort(values)
    n = values.size
    cumvals = np.cumsum(sorted_array)
    if cumvals[-1] == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n


def calculate_historical_var_metrics(daily_returns: pd.Series, window: int, alpha: float):
    """Compute rolling historical VaR / CVaR (Expected Shortfall) metrics for a return series."""
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    returns = pd.Series(daily_returns).dropna().sort_index()
    if returns.empty:
        return {
            "daily_returns": returns,
            "var_threshold": pd.Series(dtype=float),
            "expected_shortfall_threshold": pd.Series(dtype=float),
            "var": pd.Series(dtype=float),
            "expected_shortfall": pd.Series(dtype=float),
            "breaches": pd.Series(dtype=float),
            "rolling_breach_rate": pd.Series(dtype=float),
            "expected_breach_rate": pd.Series(dtype=float),
        }

    def tail_mean(window_values):
        values = np.asarray(window_values, dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            return np.nan

        cutoff = np.quantile(values, alpha)
        tail_values = values[values <= cutoff]
        if tail_values.size == 0:
            return cutoff
        return tail_values.mean()

    var_threshold = returns.rolling(window).quantile(alpha).dropna()
    expected_shortfall_threshold = returns.rolling(window).apply(tail_mean, raw=True).dropna()

    aligned_returns = returns.reindex(var_threshold.index)
    breaches = aligned_returns.lt(var_threshold).astype(float)
    rolling_breach_rate = breaches.rolling(window).mean().dropna()

    expected_breach_rate = pd.Series(
        data=np.full(len(rolling_breach_rate.index), alpha, dtype=float),
        index=rolling_breach_rate.index,
    )

    var_loss = (-var_threshold).clip(lower=0)
    expected_shortfall_loss = (-expected_shortfall_threshold).clip(lower=0)

    return {
        "daily_returns": returns.copy(),
        "var_threshold": var_threshold,
        "expected_shortfall_threshold": expected_shortfall_threshold,
        "var": var_loss,
        "expected_shortfall": expected_shortfall_loss,
        "breaches": breaches,
        "rolling_breach_rate": rolling_breach_rate,
        "expected_breach_rate": expected_breach_rate,
    }


def zscore(series: pd.Series) -> pd.Series:
    """Alias for compatibility with older notebook code."""
    return calculate_zscore(series)

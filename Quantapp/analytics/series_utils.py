"""Reusable time-series statistics helpers for notebook and library workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


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


def calculate_window_metrics(daily_returns: pd.Series, close_series: pd.Series, window: int):
    """Compute rolling drawdown and z-scored skew/kurtosis/gini metrics."""
    max_drawdown_series = calculate_max_drawdown(close_series, window=window).dropna()
    rolling_skew = daily_returns.rolling(window).apply(lambda x: skew(x, bias=False), raw=False).dropna()
    rolling_kurtosis = daily_returns.rolling(window).apply(
        lambda x: kurtosis(x, fisher=True, bias=False), raw=False
    ).dropna()
    rolling_gini = daily_returns.rolling(window).apply(lambda x: gini_coefficient(x), raw=False).dropna()
    return {
        "max_drawdown": max_drawdown_series,
        "skew_z": calculate_zscore(rolling_skew),
        "kurtosis_z": calculate_zscore(rolling_kurtosis),
        "gini_z": calculate_zscore(rolling_gini),
    }


def zscore(series: pd.Series) -> pd.Series:
    """Alias for compatibility with older notebook code."""
    return calculate_zscore(series)

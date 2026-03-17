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


def calculate_textbook_rolling_max_drawdown(price_series: pd.Series, window: int = 21) -> pd.Series:
    """Textbook rolling max drawdown computed independently inside each trailing window."""
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    def textbook_window_drawdown(window_values):
        values = np.asarray(window_values, dtype=float)
        if values.size == 0 or np.isnan(values).all():
            return np.nan

        peaks = np.maximum.accumulate(values)
        drawdowns = values / peaks - 1
        return np.nanmin(drawdowns)

    return price_series.rolling(window=window).apply(textbook_window_drawdown, raw=True)


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
    """Compute rolling return bands, textbook max drawdown, and z-scored distribution metrics."""
    max_drawdown_series = calculate_textbook_rolling_max_drawdown(close_series, window=window).dropna()
    rolling_return_q10 = daily_returns.rolling(window).quantile(0.10).dropna()
    rolling_return_q25 = daily_returns.rolling(window).quantile(0.25).dropna()
    rolling_return_median = daily_returns.rolling(window).median().dropna()
    rolling_return_q75 = daily_returns.rolling(window).quantile(0.75).dropna()
    rolling_return_q90 = daily_returns.rolling(window).quantile(0.90).dropna()
    rolling_skew = daily_returns.rolling(window).apply(lambda x: skew(x, bias=False), raw=False).dropna()
    rolling_kurtosis = daily_returns.rolling(window).apply(
        lambda x: kurtosis(x, fisher=True, bias=False), raw=False
    ).dropna()
    rolling_gini = daily_returns.rolling(window).apply(lambda x: gini_coefficient(x), raw=False).dropna()
    return {
        "daily_returns": daily_returns.copy(),
        "return_q10": rolling_return_q10,
        "return_q25": rolling_return_q25,
        "return_median": rolling_return_median,
        "return_q75": rolling_return_q75,
        "return_q90": rolling_return_q90,
        "max_drawdown": max_drawdown_series,
        "skew_z": calculate_zscore(rolling_skew),
        "kurtosis_z": calculate_zscore(rolling_kurtosis),
        "gini_z": calculate_zscore(rolling_gini),
    }


def zscore(series: pd.Series) -> pd.Series:
    """Alias for compatibility with older notebook code."""
    return calculate_zscore(series)

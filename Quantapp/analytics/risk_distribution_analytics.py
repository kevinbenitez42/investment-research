"""Risk-distribution metrics context builders for plotting workflows."""

from __future__ import annotations

import pandas as pd

from .series_utils import calculate_window_metrics


class RiskDistributionAnalytics:
    """Prepare rolling drawdown/skew/kurtosis/gini metric sets for visualization."""

    @staticmethod
    def _coerce_close_series(data) -> pd.Series:
        if isinstance(data, pd.Series):
            close = data
        elif isinstance(data, pd.DataFrame):
            if "Close" not in data.columns:
                raise ValueError("DataFrame input must contain a 'Close' column.")
            close = data["Close"]
        else:
            raise TypeError("close_series must be a pandas Series or DataFrame with 'Close'.")

        close = close.dropna().sort_index()
        if close.empty:
            raise ValueError("close_series is empty after dropping NaN values.")
        return close

    @staticmethod
    def _normalize_windows(windows):
        if windows is None:
            raise ValueError("windows must be provided.")
        try:
            windows_iter = list(windows)
        except TypeError:
            windows_iter = [windows]

        normalized = []
        for window in windows_iter:
            try:
                w = int(window)
            except (TypeError, ValueError):
                continue
            if w > 0:
                normalized.append(w)

        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("No valid windows supplied.")
        return normalized

    def build_risk_distribution_context(self, close_series, windows, default_window=None):
        """
        Build drawdown/skew/kurtosis/gini rolling metrics for each window.

        Returns
        -------
        dict
            {
                "close_series": pd.Series,
                "daily_returns": pd.Series,
                "windows": list[int],
                "default_window": int,
                "metrics_by_window": dict[int, dict[str, pd.Series]],
            }
        """
        close = self._coerce_close_series(close_series)
        window_options = self._normalize_windows(windows)
        if default_window not in window_options:
            default_window = window_options[0]

        daily_returns = close.pct_change().dropna()
        metrics_by_window = {
            window: calculate_window_metrics(daily_returns, close, window) for window in window_options
        }

        return {
            "close_series": close,
            "daily_returns": daily_returns,
            "windows": window_options,
            "default_window": default_window,
            "metrics_by_window": metrics_by_window,
        }

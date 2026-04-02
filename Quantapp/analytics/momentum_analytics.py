"""Momentum-specific return and z-score utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .close_analytics import _calculate_excess_returns


class MomentumAnalytics:
    """Helpers to compute momentum spreads and momentum z-score series."""

    @staticmethod
    def _coerce_close_series(close_series) -> pd.Series:
        if isinstance(close_series, pd.Series):
            series = close_series
        elif isinstance(close_series, pd.DataFrame):
            if "Close" not in close_series.columns:
                raise ValueError("DataFrame input must contain a 'Close' column.")
            series = close_series["Close"]
        else:
            raise TypeError("close_series must be a pandas Series or DataFrame with 'Close'.")

        series = series.dropna()
        if series.empty:
            raise ValueError("close_series is empty after dropping NaNs.")
        return series.sort_index()

    def average_return(self, close_series, window: int, percent: bool = True) -> pd.Series:
        """Alias for average-return computation without the legacy `compute_` prefix."""
        return self.compute_average_return(close_series, window=window, percent=percent)

    def compute_average_return(self, close_series, window: int, percent: bool = True) -> pd.Series:
        """Compute rolling average return over `window` observations."""
        close = self._coerce_close_series(close_series)
        returns = close.pct_change()
        avg_return = returns.rolling(window=int(window)).mean()
        if percent:
            avg_return = avg_return * 100.0
        return avg_return

    def momentum_diff(self, close_series, short_window: int, long_window: int) -> pd.Series:
        """Alias for momentum spread computation without the legacy `compute_` prefix."""
        return self.compute_momentum_diff(close_series, short_window=short_window, long_window=long_window)

    def compute_momentum_diff(self, close_series, short_window: int, long_window: int) -> pd.Series:
        """Compute momentum spread as short minus long rolling average returns."""
        avg_short = self.compute_average_return(close_series, short_window, percent=True)
        avg_long = self.compute_average_return(close_series, long_window, percent=True)
        return avg_short - avg_long

    def momentum_zscore(
        self,
        close_series,
        short_window: int,
        long_window: int,
        normalizer_window: int | None = None,
        ddof: int = 0,
    ) -> pd.Series:
        """Alias for momentum z-score computation without the legacy `compute_` prefix."""
        return self.compute_momentum_zscore(
            close_series=close_series,
            short_window=short_window,
            long_window=long_window,
            normalizer_window=normalizer_window,
            ddof=ddof,
        )

    def compute_momentum_zscore(
        self,
        close_series,
        short_window: int,
        long_window: int,
        normalizer_window: int | None = None,
        ddof: int = 0,
    ) -> pd.Series:
        """
        Compute momentum z-score from short/long return spread.

        If `normalizer_window` is set, mean/std are rolling (no full-sample lookahead).
        """
        momentum_diff = self.compute_momentum_diff(close_series, short_window, long_window)

        if normalizer_window is None:
            mean = momentum_diff.mean()
            std = momentum_diff.std(ddof=ddof)
        else:
            normalizer_window = int(normalizer_window)
            if normalizer_window <= 0:
                raise ValueError("normalizer_window must be a positive integer.")
            mean = momentum_diff.rolling(window=normalizer_window, min_periods=normalizer_window).mean()
            std = momentum_diff.rolling(window=normalizer_window, min_periods=normalizer_window).std(ddof=ddof)

        if np.isscalar(std):
            std = np.nan if std == 0 else std
        else:
            std = std.replace(0, np.nan)

        return (momentum_diff - mean) / std

    def momentum_zscore_map(
        self,
        close_series,
        window_pairs: Mapping[str, tuple[int, int]],
        normalizer_window: int | None = None,
        ddof: int = 0,
    ) -> dict[str, pd.Series]:
        """Alias for momentum z-score map computation without the legacy `compute_` prefix."""
        return self.compute_momentum_zscore_map(
            close_series=close_series,
            window_pairs=window_pairs,
            normalizer_window=normalizer_window,
            ddof=ddof,
        )

    def compute_momentum_zscore_map(
        self,
        close_series,
        window_pairs: Mapping[str, tuple[int, int]],
        normalizer_window: int | None = None,
        ddof: int = 0,
    ) -> dict[str, pd.Series]:
        """Compute momentum z-score series for multiple short/long window pairs."""
        if not isinstance(window_pairs, Mapping) or not window_pairs:
            raise ValueError("window_pairs must be a non-empty mapping of label -> (short, long).")

        zscore_data = {}
        for label, pair in window_pairs.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"Invalid window pair for '{label}'. Expected (short_window, long_window).")
            short_window, long_window = int(pair[0]), int(pair[1])
            zscore_data[str(label)] = self.compute_momentum_zscore(
                close_series=close_series,
                short_window=short_window,
                long_window=long_window,
                normalizer_window=normalizer_window,
                ddof=ddof,
            )
        return zscore_data

    def optimal_momentum_window(
        self,
        close_series,
        windows,
        risk_free_rate=0.0,
        annualization_factor: int = 252,
    ) -> pd.DataFrame:
        """Compute rolling Sharpe values for each window and the optimal window by date."""
        close = self._coerce_close_series(close_series)
        returns = close.pct_change()
        excess_returns = _calculate_excess_returns(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )
        sharpe_by_window = {}

        for window in windows:
            window = int(window)
            mean_excess_return = excess_returns.rolling(window=window).mean()
            std_excess_return = excess_returns.rolling(window=window).std()
            sharpe_ratio = np.sqrt(annualization_factor) * mean_excess_return / std_excess_return
            sharpe_by_window[window] = sharpe_ratio.where(std_excess_return > 0).replace([np.inf, -np.inf], np.nan)

        sharpe_df = pd.DataFrame(sharpe_by_window, index=close.index)

        sharpe_df = sharpe_df.dropna(how="all")
        sharpe_df["Optimal_Window"] = sharpe_df.idxmax(axis=1).astype(float)
        return sharpe_df

    @staticmethod
    def _normalize_windows(window_sizes):
        if window_sizes is None:
            raise ValueError("window_sizes must be provided.")
        try:
            windows_iter = list(window_sizes)
        except TypeError:
            windows_iter = [window_sizes]

        normalized = []
        for window in windows_iter:
            try:
                win = int(window)
            except (TypeError, ValueError):
                continue
            if win > 0:
                normalized.append(win)

        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("No valid window sizes supplied.")
        return normalized

    def build_momentum_window_diagnostics_context(
        self,
        close_series,
        window_sizes,
        highlight_windows=(7, 21, 50, 200),
        surface_years: int = 10,
        analytics=None,
        risk_free_rate=0.0,
        annualization_factor: int = 252,
    ):
        """
        Build diagnostics context for rolling Sharpe windows and volatility.

        Returns
        -------
        dict
            {
                "window_sizes": list[int],
                "highlight_windows": list[int],
                "sharpe_table": pd.DataFrame,
                "sharpe_only": pd.DataFrame,
                "optimal_windows": pd.Series,
                "optimal_windows_int": pd.Series,
                "mean_sharpe": pd.Series,
                "median_sharpe": pd.Series,
                "volatility_df": pd.DataFrame,
                "mean_volatility": pd.Series,
                "median_volatility": pd.Series,
                "sharpe_surface": pd.DataFrame,
                "surface_years": int,
            }
        """
        close = self._coerce_close_series(close_series)
        window_sizes = self._normalize_windows(window_sizes)
        highlight_windows = self._normalize_windows(highlight_windows)
        surface_years = int(surface_years)
        if surface_years <= 0:
            raise ValueError("surface_years must be a positive integer.")

        sharpe_source = analytics if analytics is not None else self
        sharpe_table = sharpe_source.optimal_momentum_window(
            close,
            window_sizes,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )
        if sharpe_table.empty:
            raise ValueError("Unable to compute rolling Sharpe table for the provided inputs.")

        sharpe_only = sharpe_table.drop(columns="Optimal_Window", errors="ignore")
        optimal_windows = sharpe_table.get("Optimal_Window", pd.Series(dtype=float)).dropna()
        optimal_windows_int = optimal_windows.astype(int) if not optimal_windows.empty else pd.Series(dtype=int)

        mean_sharpe = sharpe_only.mean()
        median_sharpe = sharpe_only.median()

        returns = close.pct_change()
        excess_returns = _calculate_excess_returns(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )
        volatility_by_window = {
            int(window): np.sqrt(annualization_factor) * excess_returns.rolling(window=int(window)).std()
            for window in window_sizes
        }
        volatility_df = pd.DataFrame(volatility_by_window, index=close.index)
        volatility_df = volatility_df.reindex(sorted(volatility_by_window), axis=1)
        mean_volatility = volatility_df.mean()
        median_volatility = volatility_df.median()

        if sharpe_only.empty:
            sharpe_surface = sharpe_only
        else:
            last_date = sharpe_only.index[-1]
            start_date = last_date - pd.Timedelta(days=365 * surface_years)
            sharpe_surface = sharpe_only.loc[start_date:last_date]

        return {
            "window_sizes": window_sizes,
            "highlight_windows": highlight_windows,
            "sharpe_table": sharpe_table,
            "sharpe_only": sharpe_only,
            "optimal_windows": optimal_windows,
            "optimal_windows_int": optimal_windows_int,
            "mean_sharpe": mean_sharpe,
            "median_sharpe": median_sharpe,
            "volatility_df": volatility_df,
            "mean_volatility": mean_volatility,
            "median_volatility": median_volatility,
            "sharpe_surface": sharpe_surface,
            "surface_years": surface_years,
        }

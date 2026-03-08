"""Risk-adjusted and relative benchmark analytics helpers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .rolling import _rolling_sortino_ratio_frame
from .series_utils import calculate_zscore


class RiskRelativeAnalytics:
    """Compute Sharpe/Sortino maps, spreads, and benchmark-relative metrics."""

    @staticmethod
    def _coerce_close_series(data, argument_name: str = "asset_close") -> pd.Series:
        if isinstance(data, pd.Series):
            close = data
        elif isinstance(data, pd.DataFrame):
            if "Close" not in data.columns:
                raise ValueError(f"{argument_name} DataFrame must contain a 'Close' column.")
            close = data["Close"]
        else:
            raise TypeError(f"{argument_name} must be a pandas Series or DataFrame.")

        close = close.dropna()
        if close.empty:
            raise ValueError(f"{argument_name} is empty after dropping NaNs.")
        return close.sort_index()

    @staticmethod
    def _coerce_time_frame_map(time_frame_map: Mapping[str, int]) -> dict[str, int]:
        if not isinstance(time_frame_map, Mapping) or not time_frame_map:
            raise ValueError("time_frame_map must be a non-empty mapping of term -> window.")

        normalized = {}
        for term, window in time_frame_map.items():
            win = int(window)
            if win <= 0:
                raise ValueError(f"Invalid window '{window}' for term '{term}'.")
            normalized[str(term)] = win
        return normalized

    @staticmethod
    def _extract_ratio_series(ratio_df: pd.DataFrame, ratio_type: str, window: int) -> pd.Series:
        expected_col = f"{ratio_type}_ratio_{window}"
        if expected_col in ratio_df.columns:
            return ratio_df[expected_col]
        if ratio_df.empty:
            return pd.Series(dtype=float)
        return ratio_df.iloc[:, 0]

    @staticmethod
    def _coerce_series_or_frame(data, argument_name: str):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if data.empty:
                raise ValueError(f"{argument_name} is empty.")
            return data.sort_index()
        raise TypeError(f"{argument_name} must be a pandas Series or DataFrame.")

    @staticmethod
    def _coerce_benchmark_series(benchmark_series) -> pd.Series:
        if isinstance(benchmark_series, pd.Series):
            series = benchmark_series
        elif isinstance(benchmark_series, pd.DataFrame):
            if benchmark_series.shape[1] != 1 and "Close" not in benchmark_series.columns:
                raise ValueError("benchmark_series DataFrame must contain exactly one column or a 'Close' column.")
            series = benchmark_series["Close"] if "Close" in benchmark_series.columns else benchmark_series.iloc[:, 0]
        else:
            raise TypeError("benchmark_series must be a pandas Series or DataFrame.")

        series = series.dropna()
        if series.empty:
            raise ValueError("benchmark_series is empty after dropping NaNs.")
        return series.sort_index()

    @staticmethod
    def _rolling_sortino_ratio(data, window: int, risk_free_rate: float = 0.0) -> pd.DataFrame:
        return _rolling_sortino_ratio_frame(
            data,
            window=window,
            risk_free_rate=risk_free_rate,
        )

    def asset_ratio_map(self, analytics, asset_close, time_frame_map, ratio_type: str) -> dict[str, pd.Series]:
        """Alias for ratio-map construction without the legacy `compute_` prefix."""
        return self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type=ratio_type,
        )

    def compute_asset_ratio_map(self, analytics, asset_close, time_frame_map, ratio_type: str) -> dict[str, pd.Series]:
        """Compute one risk-adjusted ratio series per term/window."""
        close = self._coerce_close_series(asset_close, argument_name="asset_close")
        tf_map = self._coerce_time_frame_map(time_frame_map)

        ratio_map = {}
        for term, window in tf_map.items():
            ratio_df = analytics.risk_adjusted_returns(
                close,
                ratio_type=ratio_type,
                windows=[window],
            )
            ratio_map[term] = self._extract_ratio_series(ratio_df, ratio_type=ratio_type, window=window)
        return ratio_map

    def asset_ratio_maps(self, analytics, asset_close, time_frame_map):
        """Alias for Sharpe/Sortino map construction without the legacy `compute_` prefix."""
        return self.compute_asset_ratio_maps(analytics=analytics, asset_close=asset_close, time_frame_map=time_frame_map)

    def compute_asset_ratio_maps(self, analytics, asset_close, time_frame_map):
        """Compute Sharpe and Sortino maps keyed by term."""
        asset_sharpe_map = self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type="sharpe",
        )
        asset_sortino_map = self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type="sortino",
        )
        return asset_sharpe_map, asset_sortino_map

    @staticmethod
    def ratio_spread_map(left_map: Mapping[str, pd.Series], right_map: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
        """Alias for spread-map construction without the legacy `compute_` prefix."""
        return RiskRelativeAnalytics.compute_ratio_spread_map(left_map, right_map)

    @staticmethod
    def compute_ratio_spread_map(left_map: Mapping[str, pd.Series], right_map: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
        """Compute left-right spread map using shared term keys."""
        return {term: left_map[term] - right_map[term] for term in left_map.keys() if term in right_map}

    def benchmark_metrics(self, analytics, benchmark_data, asset_close, asset_sharpe_map, time_frame_map):
        """Alias for benchmark-metric construction without the legacy `compute_` prefix."""
        return self.compute_benchmark_metrics(
            analytics=analytics,
            benchmark_data=benchmark_data,
            asset_close=asset_close,
            asset_sharpe_map=asset_sharpe_map,
            time_frame_map=time_frame_map,
        )

    def compute_benchmark_metrics(self, analytics, benchmark_data, asset_close, asset_sharpe_map, time_frame_map):
        """Compute benchmark Sharpe, relative returns spread, and Sharpe spread by term."""
        close = self._coerce_close_series(asset_close, argument_name="asset_close")
        tf_map = self._coerce_time_frame_map(time_frame_map)

        if not benchmark_data:
            return {}

        metrics = {}
        for symbol, benchmark_frame in benchmark_data.items():
            benchmark_close = self._coerce_close_series(benchmark_frame, argument_name=f"benchmark_data[{symbol}]")
            metrics[symbol] = {}

            for term, window in tf_map.items():
                benchmark_sharpe_df = analytics.risk_adjusted_returns(
                    benchmark_close,
                    ratio_type="sharpe",
                    windows=[window],
                )
                benchmark_sharpe = self._extract_ratio_series(
                    benchmark_sharpe_df,
                    ratio_type="sharpe",
                    window=window,
                )
                relative_spread = benchmark_close.pct_change(window) - close.pct_change(window)
                metrics[symbol][term] = {
                    "spread": relative_spread,
                    "sharpe_ratio": benchmark_sharpe,
                    "sharpe_spread": benchmark_sharpe - asset_sharpe_map[term],
                }

        return metrics

    def spread(self, asset_series, benchmark_series, time_frame, mode: str = "standard", risk_free_rate: float = 0.0):
        """Compute benchmark-minus-asset spread for raw returns or rolling Sortino ratios."""
        time_frame = int(time_frame)
        if time_frame <= 0:
            raise ValueError("time_frame must be a positive integer.")

        asset_data = self._coerce_series_or_frame(asset_series, argument_name="asset_series")
        benchmark = self._coerce_benchmark_series(benchmark_series)

        if mode == "standard":
            asset_metric = asset_data.pct_change(time_frame)
            benchmark_metric = benchmark.pct_change(time_frame)
        elif mode == "sortino":
            asset_metric = self._rolling_sortino_ratio(asset_data, window=time_frame, risk_free_rate=risk_free_rate)
            benchmark_metric = self._rolling_sortino_ratio(
                benchmark,
                window=time_frame,
                risk_free_rate=risk_free_rate,
            ).iloc[:, 0]
        else:
            raise ValueError("Invalid mode. Use 'standard' or 'sortino'.")

        if isinstance(asset_metric, pd.Series):
            asset_metric = asset_metric.to_frame(name=asset_metric.name or "asset")

        spread_df = pd.DataFrame(index=asset_metric.index)
        benchmark_metric = benchmark_metric.reindex(asset_metric.index)
        for col in asset_metric.columns:
            spread_df[f"Benchmark_minus_{col}"] = benchmark_metric - asset_metric[col]

        return spread_df

    def build_term_config_map(self, asset_sharpe_map, asset_sortino_map, time_frame_map):
        """Build plotting config map keyed as '<window>-day'."""
        tf_map = self._coerce_time_frame_map(time_frame_map)
        term_config_map = {}
        for term, window in tf_map.items():
            label = f"{window}-day"
            term_config_map[label] = {
                "sharpe": asset_sharpe_map[term],
                "sortino": asset_sortino_map[term],
                "time_frame": window,
                "term_key": term,
            }
        return term_config_map

    def build_sharpe_sortino_context(self, analytics, asset_close, time_frame_map, benchmark_data=None):
        """Return all Sharpe/Sortino maps and benchmark-relative artifacts in one payload."""
        tf_map = self._coerce_time_frame_map(time_frame_map)
        asset_sharpe_map, asset_sortino_map = self.compute_asset_ratio_maps(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=tf_map,
        )
        sharpe_sortino_spread_map = self.compute_ratio_spread_map(asset_sharpe_map, asset_sortino_map)
        benchmark_metrics = self.compute_benchmark_metrics(
            analytics=analytics,
            benchmark_data=benchmark_data,
            asset_close=asset_close,
            asset_sharpe_map=asset_sharpe_map,
            time_frame_map=tf_map,
        )
        benchmark_order = list(benchmark_metrics.keys())
        spread_plot_data = {
            term: {symbol: benchmark_metrics[symbol][term]["sharpe_spread"] for symbol in benchmark_order}
            for term in tf_map
        }

        return {
            "asset_sharpe_map": asset_sharpe_map,
            "asset_sortino_map": asset_sortino_map,
            "asset_sharpe_sortino_spread_map": sharpe_sortino_spread_map,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_order": benchmark_order,
            "default_benchmark": benchmark_order[0] if benchmark_order else None,
            "spread_plot_data": spread_plot_data,
            "term_config_map": self.build_term_config_map(asset_sharpe_map, asset_sortino_map, tf_map),
        }

    def build_benchmark_plot_payload(self, asset_sharpe_map, benchmark_metrics, spread_plot_data, time_frame_map):
        """
        Prepare z-scored benchmark comparison payload for plotting.

        Returns
        -------
        dict
            {
                "term_order": [...],
                "benchmark_order": [...],
                "default_benchmark": str | None,
                "asset_zscore_map": {term: Series},
                "summary_zscore_map": {term: {symbol: Series}},
                "detail_zscore_map": {symbol: {term: {"asset","benchmark","sharpe_spread","relative_spread"}}},
            }
        """
        tf_map = self._coerce_time_frame_map(time_frame_map)
        term_order = list(tf_map.keys())
        benchmark_order = list(benchmark_metrics.keys())

        asset_zscore_map = {
            term: calculate_zscore(asset_sharpe_map[term].dropna()).dropna()
            for term in term_order
            if term in asset_sharpe_map
        }

        summary_zscore_map = {}
        for term in term_order:
            term_series_map = spread_plot_data.get(term, {})
            summary_zscore_map[term] = {}
            for symbol in benchmark_order:
                series = term_series_map.get(symbol, pd.Series(dtype=float)).dropna()
                zscore_series = calculate_zscore(series).dropna() if not series.empty else pd.Series(dtype=float)
                summary_zscore_map[term][symbol] = zscore_series

        detail_zscore_map = {}
        for symbol in benchmark_order:
            detail_zscore_map[symbol] = {}
            symbol_metrics = benchmark_metrics.get(symbol, {})
            for term in term_order:
                term_metrics = symbol_metrics.get(term, {})
                benchmark_sharpe = calculate_zscore(
                    term_metrics.get("sharpe_ratio", pd.Series(dtype=float)).dropna()
                ).dropna()
                sharpe_spread = calculate_zscore(
                    term_metrics.get("sharpe_spread", pd.Series(dtype=float)).dropna()
                ).dropna()
                relative_spread = calculate_zscore(
                    term_metrics.get("spread", pd.Series(dtype=float)).dropna()
                ).dropna()

                detail_zscore_map[symbol][term] = {
                    "asset": asset_zscore_map.get(term, pd.Series(dtype=float)),
                    "benchmark": benchmark_sharpe,
                    "sharpe_spread": sharpe_spread,
                    "relative_spread": relative_spread,
                }

        return {
            "term_order": term_order,
            "benchmark_order": benchmark_order,
            "default_benchmark": benchmark_order[0] if benchmark_order else None,
            "asset_zscore_map": asset_zscore_map,
            "summary_zscore_map": summary_zscore_map,
            "detail_zscore_map": detail_zscore_map,
        }

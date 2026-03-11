"""Risk-adjusted and relative benchmark analytics helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .close_analytics import _calculate_excess_returns
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
    def _coerce_close_frame(data, argument_name: str = "asset_close") -> pd.DataFrame:
        if isinstance(data, pd.Series):
            name = data.name or "asset"
            frame = data.to_frame(name=name)
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            raise TypeError(f"{argument_name} must be a pandas Series or DataFrame.")

        frame = frame.dropna(how="all")
        if frame.empty:
            raise ValueError(f"{argument_name} is empty after dropping NaNs.")
        return frame.sort_index()

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
    def _rolling_sharpe_ratio_from_returns(returns, window: int, annualization_factor: int = 252):
        """Compute rolling Sharpe ratios from precomputed returns."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        ratio = np.sqrt(annualization_factor) * rolling_mean / rolling_std
        return ratio.where(rolling_std > 0).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _rolling_risk_components(
        close: pd.Series,
        window: int,
        risk_free_rate=0.0,
        annualization_factor: int = 252,
    ) -> dict[str, pd.Series]:
        """Compute rolling excess-return, volatility, and Sharpe components for one close series."""
        returns = close.pct_change()
        excess_returns = _calculate_excess_returns(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        annualized_excess_return = annualization_factor * rolling_mean
        annualized_volatility = np.sqrt(annualization_factor) * rolling_std
        sharpe_ratio = np.sqrt(annualization_factor) * rolling_mean / rolling_std
        sharpe_ratio = sharpe_ratio.where(rolling_std > 0).replace([np.inf, -np.inf], np.nan)
        return {
            "annualized_excess_return": annualized_excess_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
        }

    @staticmethod
    def _latest_zscore_snapshot(metric_frame: pd.DataFrame) -> pd.Series:
        """Return the latest cross-sectional snapshot from time-series z-scored metrics."""
        if metric_frame.empty:
            return pd.Series(dtype=float)
        zscore_frame = metric_frame.apply(calculate_zscore)
        if zscore_frame.empty:
            return pd.Series(dtype=float)
        return zscore_frame.iloc[-1].dropna().sort_values(ascending=False)

    @classmethod
    def _latest_spread_zscore_snapshot(cls, asset_metric_frame: pd.DataFrame, benchmark_metric: pd.Series) -> pd.Series:
        """Return the latest z-score snapshot of the benchmark-minus-asset spread series."""
        if asset_metric_frame.empty:
            return pd.Series(dtype=float)
        benchmark_aligned = benchmark_metric.reindex(asset_metric_frame.index)
        spread_frame = asset_metric_frame.apply(lambda column: benchmark_aligned - column, axis=0)
        return cls._latest_zscore_snapshot(spread_frame)

    @staticmethod
    def _rolling_sortino_ratio(data, window: int, risk_free_rate: float = 0.0) -> pd.DataFrame:
        return _rolling_sortino_ratio_frame(
            data,
            window=window,
            risk_free_rate=risk_free_rate,
        )

    def compute_asset_component_map(
        self,
        asset_close,
        time_frame_map,
        risk_free_rate=0.0,
        annualization_factor: int = 252,
    ) -> dict[str, dict[str, pd.Series]]:
        """Compute rolling excess-return, volatility, and Sharpe components keyed by term."""
        close = self._coerce_close_series(asset_close, argument_name="asset_close")
        tf_map = self._coerce_time_frame_map(time_frame_map)
        return {
            term: self._rolling_risk_components(
                close,
                window=window,
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
            )
            for term, window in tf_map.items()
        }

    def asset_ratio_map(self, analytics, asset_close, time_frame_map, ratio_type: str, risk_free_rate=0.0) -> dict[str, pd.Series]:
        """Alias for ratio-map construction without the legacy `compute_` prefix."""
        return self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type=ratio_type,
            risk_free_rate=risk_free_rate,
        )

    def compute_asset_ratio_map(
        self,
        analytics,
        asset_close,
        time_frame_map,
        ratio_type: str,
        risk_free_rate=0.0,
    ) -> dict[str, pd.Series]:
        """Compute one risk-adjusted ratio series per term/window."""
        close = self._coerce_close_series(asset_close, argument_name="asset_close")
        tf_map = self._coerce_time_frame_map(time_frame_map)

        ratio_map = {}
        for term, window in tf_map.items():
            ratio_df = analytics.risk_adjusted_returns(
                close,
                ratio_type=ratio_type,
                windows=[window],
                risk_free_rate=risk_free_rate,
            )
            ratio_map[term] = self._extract_ratio_series(ratio_df, ratio_type=ratio_type, window=window)
        return ratio_map

    def asset_ratio_maps(self, analytics, asset_close, time_frame_map, risk_free_rate=0.0):
        """Alias for Sharpe/Sortino map construction without the legacy `compute_` prefix."""
        return self.compute_asset_ratio_maps(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            risk_free_rate=risk_free_rate,
        )

    def compute_asset_ratio_maps(self, analytics, asset_close, time_frame_map, risk_free_rate=0.0):
        """Compute Sharpe and Sortino maps keyed by term."""
        asset_sharpe_map = self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type="sharpe",
            risk_free_rate=risk_free_rate,
        )
        asset_sortino_map = self.compute_asset_ratio_map(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            ratio_type="sortino",
            risk_free_rate=risk_free_rate,
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

    def benchmark_metrics(
        self,
        analytics,
        benchmark_data,
        asset_close,
        asset_sharpe_map,
        time_frame_map,
        risk_free_rate=0.0,
    ):
        """Alias for benchmark-metric construction without the legacy `compute_` prefix."""
        return self.compute_benchmark_metrics(
            analytics=analytics,
            benchmark_data=benchmark_data,
            asset_close=asset_close,
            asset_sharpe_map=asset_sharpe_map,
            time_frame_map=time_frame_map,
            risk_free_rate=risk_free_rate,
        )

    def compute_benchmark_metrics(
        self,
        analytics,
        benchmark_data,
        asset_close,
        asset_sharpe_map,
        time_frame_map,
        risk_free_rate=0.0,
    ):
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
                benchmark_components = self._rolling_risk_components(
                    benchmark_close,
                    window=window,
                    risk_free_rate=risk_free_rate,
                )
                benchmark_sharpe = benchmark_components["sharpe_ratio"]
                relative_spread = benchmark_close.pct_change(window) - close.pct_change(window)
                metrics[symbol][term] = {
                    "spread": relative_spread,
                    "annualized_excess_return": benchmark_components["annualized_excess_return"],
                    "annualized_volatility": benchmark_components["annualized_volatility"],
                    "sharpe_ratio": benchmark_sharpe,
                    "sharpe_spread": benchmark_sharpe - asset_sharpe_map[term],
                }

        return metrics

    def build_multi_asset_benchmark_snapshot(
        self,
        analytics,
        asset_close,
        benchmark_close,
        time_frame_map,
        sign_map=None,
        annualization_factor: int = 252,
    ):
        """
        Build latest cross-sectional Sharpe and benchmark-spread z-score snapshots for a multi-asset frame.

        This mirrors the single-asset benchmark spread methodology used elsewhere:
        compute rolling Sharpe series first, then z-score the benchmark-minus-asset spread series.
        """
        asset_frame = self._coerce_close_frame(asset_close, argument_name="asset_close")
        benchmark_series = self._coerce_benchmark_series(benchmark_close)
        tf_map = self._coerce_time_frame_map(time_frame_map)

        common_index = asset_frame.index.intersection(benchmark_series.index)
        asset_frame = asset_frame.reindex(common_index).dropna(how="all")
        benchmark_series = benchmark_series.reindex(common_index).dropna()
        if asset_frame.empty or benchmark_series.empty:
            raise ValueError("asset_close and benchmark_close do not share a non-empty common index.")

        sign_series = pd.Series(1.0, index=asset_frame.columns, dtype=float)
        if sign_map is not None:
            sign_series = pd.Series(sign_map, dtype=float).reindex(asset_frame.columns).fillna(1.0)

        signed_returns = asset_frame.pct_change().mul(sign_series, axis=1)

        payload = {
            "unsigned_asset_latest_zscores": {},
            "signed_asset_latest_zscores": {},
            "unsigned_spread_latest_zscores": {},
            "signed_spread_latest_zscores": {},
            "signed_return_latest_zscores": {},
        }

        for term, window in tf_map.items():
            unsigned_sharpe = analytics.risk_adjusted_returns(
                asset_frame,
                windows=[window],
                ratio_type="sharpe",
            )
            if unsigned_sharpe.shape[1] == asset_frame.shape[1]:
                unsigned_sharpe.columns = asset_frame.columns

            benchmark_sharpe = analytics.risk_adjusted_returns(
                benchmark_series,
                windows=[window],
                ratio_type="sharpe",
            ).iloc[:, 0]

            signed_sharpe = self._rolling_sharpe_ratio_from_returns(
                signed_returns,
                window=window,
                annualization_factor=annualization_factor,
            )
            signed_return_frame = asset_frame.pct_change(window).mul(sign_series, axis=1)

            payload["unsigned_asset_latest_zscores"][term] = self._latest_zscore_snapshot(unsigned_sharpe)
            payload["signed_asset_latest_zscores"][term] = self._latest_zscore_snapshot(signed_sharpe)
            payload["unsigned_spread_latest_zscores"][term] = self._latest_spread_zscore_snapshot(
                unsigned_sharpe,
                benchmark_sharpe,
            )
            payload["signed_spread_latest_zscores"][term] = self._latest_spread_zscore_snapshot(
                signed_sharpe,
                benchmark_sharpe,
            )
            payload["signed_return_latest_zscores"][term] = self._latest_zscore_snapshot(signed_return_frame)

        return payload

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

    def build_sharpe_sortino_context(
        self,
        analytics,
        asset_close,
        time_frame_map,
        benchmark_data=None,
        risk_free_rate=0.0,
    ):
        """Return all Sharpe/Sortino maps and benchmark-relative artifacts in one payload."""
        tf_map = self._coerce_time_frame_map(time_frame_map)
        asset_sharpe_map, asset_sortino_map = self.compute_asset_ratio_maps(
            analytics=analytics,
            asset_close=asset_close,
            time_frame_map=tf_map,
            risk_free_rate=risk_free_rate,
        )
        asset_component_map = self.compute_asset_component_map(
            asset_close=asset_close,
            time_frame_map=tf_map,
            risk_free_rate=risk_free_rate,
        )
        sharpe_sortino_spread_map = self.compute_ratio_spread_map(asset_sharpe_map, asset_sortino_map)
        benchmark_metrics = self.compute_benchmark_metrics(
            analytics=analytics,
            benchmark_data=benchmark_data,
            asset_close=asset_close,
            asset_sharpe_map=asset_sharpe_map,
            time_frame_map=tf_map,
            risk_free_rate=risk_free_rate,
        )
        benchmark_order = list(benchmark_metrics.keys())
        spread_plot_data = {
            term: {symbol: benchmark_metrics[symbol][term]["sharpe_spread"] for symbol in benchmark_order}
            for term in tf_map
        }

        return {
            "asset_sharpe_map": asset_sharpe_map,
            "asset_component_map": asset_component_map,
            "asset_sortino_map": asset_sortino_map,
            "asset_sharpe_sortino_spread_map": sharpe_sortino_spread_map,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_order": benchmark_order,
            "default_benchmark": benchmark_order[0] if benchmark_order else None,
            "spread_plot_data": spread_plot_data,
            "term_config_map": self.build_term_config_map(asset_sharpe_map, asset_sortino_map, tf_map),
        }

    def build_benchmark_plot_payload(
        self,
        asset_sharpe_map,
        benchmark_metrics,
        spread_plot_data,
        time_frame_map,
        asset_component_map=None,
    ):
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
                "detail_zscore_map": {symbol: {term: {"asset","benchmark","asset_sharpe","benchmark_sharpe", ...}}},
            }
        """
        tf_map = self._coerce_time_frame_map(time_frame_map)
        term_order = list(tf_map.keys())
        benchmark_order = list(benchmark_metrics.keys())
        asset_component_map = asset_component_map or {}

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
                asset_components = asset_component_map.get(term, {})
                asset_sharpe = asset_sharpe_map.get(term, pd.Series(dtype=float)).dropna()
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
                    "asset_sharpe": asset_sharpe,
                    "benchmark_sharpe": term_metrics.get("sharpe_ratio", pd.Series(dtype=float)).dropna(),
                    "asset_excess_return": asset_components.get("annualized_excess_return", pd.Series(dtype=float)).dropna(),
                    "benchmark_excess_return": term_metrics.get("annualized_excess_return", pd.Series(dtype=float)).dropna(),
                    "asset_volatility": asset_components.get("annualized_volatility", pd.Series(dtype=float)).dropna(),
                    "benchmark_volatility": term_metrics.get("annualized_volatility", pd.Series(dtype=float)).dropna(),
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

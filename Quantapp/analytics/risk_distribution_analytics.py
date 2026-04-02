"""Risk-distribution metrics context builders for plotting workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .series_utils import calculate_historical_var_metrics, calculate_window_metrics


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
    def _coerce_ohlc_frame(data) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("price_frame must be a pandas DataFrame with 'Open' and 'Close' columns.")

        required_columns = {"Open", "Close"}
        missing_columns = sorted(required_columns.difference(data.columns))
        if missing_columns:
            raise ValueError(
                f"price_frame is missing required columns: {missing_columns}"
            )

        frame = data.loc[:, ["Open", "Close"]].dropna().sort_index()
        if frame.empty:
            raise ValueError("price_frame is empty after dropping NaN values.")
        return frame

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

    @staticmethod
    def _select_default_window(window_options, default_window=None):
        if default_window in window_options:
            return default_window
        if 200 in window_options:
            return 200
        return max(window_options)

    @staticmethod
    def _normalize_confidence_levels(confidence_levels):
        if confidence_levels is None:
            confidence_levels = (0.95, 0.99)

        try:
            level_iter = list(confidence_levels)
        except TypeError:
            level_iter = [confidence_levels]

        normalized = []
        for level in level_iter:
            try:
                confidence = float(level)
            except (TypeError, ValueError):
                continue

            if confidence > 1:
                confidence = confidence / 100.0
            if 0 < confidence < 1:
                normalized.append(confidence)

        normalized = list(dict.fromkeys(normalized))
        if not normalized:
            raise ValueError("No valid confidence levels supplied.")
        return sorted(normalized)

    @staticmethod
    def _select_default_confidence(confidence_levels, default_confidence=None):
        if default_confidence is not None:
            try:
                confidence = float(default_confidence)
            except (TypeError, ValueError):
                confidence = None
            else:
                if confidence > 1:
                    confidence = confidence / 100.0
                if confidence in confidence_levels:
                    return confidence

        if 0.95 in confidence_levels:
            return 0.95
        return confidence_levels[0]

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
        default_window = self._select_default_window(window_options, default_window=default_window)

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

    def build_value_at_risk_context(
        self,
        close_series,
        windows,
        confidence_levels=(0.95, 0.99),
        default_window=None,
        default_confidence=None,
        position_value=None,
    ):
        """
        Build rolling historical VaR / CVaR (Expected Shortfall) metrics for each window and confidence level.

        Returns
        -------
        dict
            {
                "close_series": pd.Series,
                "daily_returns": pd.Series,
                "windows": list[int],
                "default_window": int,
                "confidence_levels": list[float],
                "default_confidence": float,
                "metrics_by_window": dict[int, dict[float, dict[str, pd.Series]]],
                "summary_table": pd.DataFrame,
                "position_value": float | None,
            }
        """
        close = self._coerce_close_series(close_series)
        window_options = self._normalize_windows(windows)
        default_window = self._select_default_window(window_options, default_window=default_window)
        confidence_levels = self._normalize_confidence_levels(confidence_levels)
        default_confidence = self._select_default_confidence(
            confidence_levels,
            default_confidence=default_confidence,
        )

        daily_returns = close.pct_change(fill_method=None).dropna()
        metrics_by_window = {}
        summary_rows = []

        for window in window_options:
            metrics_by_confidence = {}
            for confidence in confidence_levels:
                alpha = 1 - confidence
                metric_set = calculate_historical_var_metrics(daily_returns, window=window, alpha=alpha)

                if position_value is not None:
                    metric_set["var_dollar"] = metric_set["var"] * float(position_value)
                    metric_set["expected_shortfall_dollar"] = (
                        metric_set["expected_shortfall"] * float(position_value)
                    )

                metrics_by_confidence[confidence] = metric_set

                var_series = metric_set["var"].dropna()
                es_series = metric_set["expected_shortfall"].dropna()
                breach_series = metric_set["breaches"].dropna()
                rolling_breach_rate = metric_set["rolling_breach_rate"].dropna()

                summary_row = {
                    "Window": window,
                    "Confidence": f"{confidence:.0%}",
                    "Latest VaR": var_series.iloc[-1] if not var_series.empty else np.nan,
                    "Latest CVaR": es_series.iloc[-1] if not es_series.empty else np.nan,
                    "Observed Breach Rate": breach_series.mean() if not breach_series.empty else np.nan,
                    "Expected Breach Rate": alpha,
                    "Latest Rolling Breach Rate": (
                        rolling_breach_rate.iloc[-1] if not rolling_breach_rate.empty else np.nan
                    ),
                }

                if position_value is not None:
                    var_dollar = metric_set["var_dollar"].dropna()
                    es_dollar = metric_set["expected_shortfall_dollar"].dropna()
                    summary_row["Latest VaR Dollar"] = var_dollar.iloc[-1] if not var_dollar.empty else np.nan
                    summary_row["Latest CVaR Dollar"] = es_dollar.iloc[-1] if not es_dollar.empty else np.nan

                summary_rows.append(summary_row)

            metrics_by_window[window] = metrics_by_confidence

        summary_table = pd.DataFrame(summary_rows)
        if not summary_table.empty:
            summary_table = summary_table.sort_values(["Window", "Confidence"]).reset_index(drop=True)

        return {
            "close_series": close,
            "daily_returns": daily_returns,
            "windows": window_options,
            "default_window": default_window,
            "confidence_levels": confidence_levels,
            "default_confidence": default_confidence,
            "metrics_by_window": metrics_by_window,
            "summary_table": summary_table,
            "position_value": position_value,
        }

    def build_session_probability_cone_context(
        self,
        price_frame,
        window=200,
        interval_confidence_levels=(0.50, 0.80, 0.90, 0.95),
        var_confidence_levels=(0.95, 0.99),
        anchor_price=None,
    ):
        """
        Build an open-anchored probability cone for the latest session using trailing
        open-to-close returns from completed sessions.

        Returns
        -------
        dict
            {
                "session_date": pd.Timestamp,
                "window": int,
                "effective_window": int,
                "anchor_price": float,
                "latest_price": float,
                "sample_returns": pd.Series,
                "interval_confidence_levels": list[float],
                "var_confidence_levels": list[float],
                "intervals": dict[float, dict[str, float]],
                "var_levels": dict[float, dict[str, float]],
                "median_return": float,
                "median_price": float,
                "summary_table": pd.DataFrame,
            }
        """
        frame = self._coerce_ohlc_frame(price_frame)
        try:
            window = int(window)
        except (TypeError, ValueError) as exc:
            raise ValueError("window must be a positive integer.") from exc
        if window <= 0:
            raise ValueError("window must be a positive integer.")

        interval_confidence_levels = self._normalize_confidence_levels(interval_confidence_levels)
        var_confidence_levels = self._normalize_confidence_levels(var_confidence_levels)

        session_returns = frame["Close"].div(frame["Open"]).sub(1.0).dropna()
        if len(session_returns) < 2:
            raise ValueError(
                "At least two sessions with valid open and close prices are required "
                "to build a current-session probability cone."
            )

        historical_returns = session_returns.iloc[:-1].dropna()
        if historical_returns.empty:
            raise ValueError("No completed session returns are available for the cone sample.")

        effective_window = min(window, len(historical_returns))
        sample_returns = historical_returns.tail(effective_window)
        session_date = frame.index[-1]
        latest_price = float(frame["Close"].iloc[-1])

        if anchor_price is None:
            anchor_price = float(frame["Open"].iloc[-1])
        else:
            anchor_price = float(anchor_price)

        if not np.isfinite(anchor_price) or anchor_price <= 0:
            raise ValueError("anchor_price must be a positive finite value.")

        median_return = float(sample_returns.median())
        median_price = anchor_price * (1.0 + median_return)

        interval_map = {}
        summary_rows = []
        for confidence in interval_confidence_levels:
            tail_probability = (1.0 - confidence) / 2.0
            lower_return = float(sample_returns.quantile(tail_probability))
            upper_return = float(sample_returns.quantile(1.0 - tail_probability))
            lower_price = anchor_price * (1.0 + lower_return)
            upper_price = anchor_price * (1.0 + upper_return)
            interval_map[confidence] = {
                "lower_return": lower_return,
                "upper_return": upper_return,
                "lower_price": lower_price,
                "upper_price": upper_price,
            }
            summary_rows.append(
                {
                    "Metric": f"{confidence:.0%} Central Range",
                    "Lower Return": lower_return,
                    "Upper Return": upper_return,
                    "Lower Price": lower_price,
                    "Upper Price": upper_price,
                }
            )

        var_level_map = {}
        for confidence in var_confidence_levels:
            alpha = 1.0 - confidence
            metric_set = calculate_historical_var_metrics(
                sample_returns,
                window=effective_window,
                alpha=alpha,
            )
            var_series = metric_set["var"].dropna()
            expected_shortfall_series = metric_set["expected_shortfall"].dropna()

            var_loss = float(var_series.iloc[-1]) if not var_series.empty else np.nan
            expected_shortfall_loss = (
                float(expected_shortfall_series.iloc[-1])
                if not expected_shortfall_series.empty
                else np.nan
            )
            if pd.isna(var_loss):
                var_loss = float(max(-float(sample_returns.quantile(alpha)), 0.0))
            if pd.isna(expected_shortfall_loss):
                tail_values = sample_returns[sample_returns <= sample_returns.quantile(alpha)]
                tail_mean = float(tail_values.mean()) if not tail_values.empty else -var_loss
                expected_shortfall_loss = float(max(-tail_mean, var_loss))

            var_return = -var_loss
            expected_shortfall_return = -expected_shortfall_loss
            var_price = anchor_price * (1.0 + var_return)
            expected_shortfall_price = anchor_price * (1.0 + expected_shortfall_return)

            var_level_map[confidence] = {
                "var_loss": var_loss,
                "var_return": var_return,
                "var_price": var_price,
                "expected_shortfall_loss": expected_shortfall_loss,
                "expected_shortfall_return": expected_shortfall_return,
                "expected_shortfall_price": expected_shortfall_price,
            }
            summary_rows.extend(
                [
                    {
                        "Metric": f"{confidence:.0%} VaR Floor",
                        "Lower Return": var_return,
                        "Upper Return": np.nan,
                        "Lower Price": var_price,
                        "Upper Price": np.nan,
                    },
                    {
                        "Metric": f"{confidence:.0%} CVaR Floor",
                        "Lower Return": expected_shortfall_return,
                        "Upper Return": np.nan,
                        "Lower Price": expected_shortfall_price,
                        "Upper Price": np.nan,
                    },
                ]
            )

        summary_rows.insert(
            0,
            {
                "Metric": "Session Open",
                "Lower Return": 0.0,
                "Upper Return": 0.0,
                "Lower Price": anchor_price,
                "Upper Price": anchor_price,
            },
        )
        summary_rows.insert(
            1,
            {
                "Metric": "Latest Session Price",
                "Lower Return": (latest_price / anchor_price) - 1.0 if anchor_price else np.nan,
                "Upper Return": np.nan,
                "Lower Price": latest_price,
                "Upper Price": np.nan,
            },
        )

        summary_table = pd.DataFrame(summary_rows)

        return {
            "session_date": session_date,
            "window": window,
            "effective_window": effective_window,
            "anchor_price": anchor_price,
            "latest_price": latest_price,
            "sample_returns": sample_returns,
            "interval_confidence_levels": interval_confidence_levels,
            "var_confidence_levels": var_confidence_levels,
            "intervals": interval_map,
            "var_levels": var_level_map,
            "median_return": median_return,
            "median_price": median_price,
            "summary_table": summary_table,
        }

    def build_trade_range_probability_context(
        self,
        price_frame,
        window=200,
        interval_confidence_levels=(0.95, 0.99),
        tail_confidence_levels=(0.95, 0.99),
        anchor_price=None,
    ):
        """
        Build a two-sided open-anchored session probability context for long and short
        decision support using trailing open-to-close returns from completed sessions.
        """
        frame = self._coerce_ohlc_frame(price_frame)
        try:
            window = int(window)
        except (TypeError, ValueError) as exc:
            raise ValueError("window must be a positive integer.") from exc
        if window <= 0:
            raise ValueError("window must be a positive integer.")

        interval_confidence_levels = self._normalize_confidence_levels(interval_confidence_levels)
        tail_confidence_levels = self._normalize_confidence_levels(tail_confidence_levels)

        session_returns = frame["Close"].div(frame["Open"]).sub(1.0).dropna()
        if len(session_returns) < 2:
            raise ValueError(
                "At least two sessions with valid open and close prices are required "
                "to build a current-session probability range."
            )

        historical_returns = session_returns.iloc[:-1].dropna()
        if historical_returns.empty:
            raise ValueError("No completed session returns are available for the trade range sample.")

        effective_window = min(window, len(historical_returns))
        sample_returns = historical_returns.tail(effective_window)
        session_date = frame.index[-1]
        latest_price = float(frame["Close"].iloc[-1])

        if anchor_price is None:
            anchor_price = float(frame["Open"].iloc[-1])
        else:
            anchor_price = float(anchor_price)

        if not np.isfinite(anchor_price) or anchor_price <= 0:
            raise ValueError("anchor_price must be a positive finite value.")

        median_return = float(sample_returns.median())
        median_price = anchor_price * (1.0 + median_return)

        interval_map = {}
        range_rows = []
        for confidence in interval_confidence_levels:
            tail_probability = (1.0 - confidence) / 2.0
            lower_return = float(sample_returns.quantile(tail_probability))
            upper_return = float(sample_returns.quantile(1.0 - tail_probability))
            lower_price = anchor_price * (1.0 + lower_return)
            upper_price = anchor_price * (1.0 + upper_return)
            interval_map[confidence] = {
                "lower_return": lower_return,
                "upper_return": upper_return,
                "lower_price": lower_price,
                "upper_price": upper_price,
            }
            range_rows.append(
                {
                    "Confidence": f"{confidence:.0%}",
                    "Lower Return": lower_return,
                    "Upper Return": upper_return,
                    "Lower Close": lower_price,
                    "Upper Close": upper_price,
                }
            )

        long_tail_map = {}
        short_tail_map = {}
        tail_rows = []
        for confidence in tail_confidence_levels:
            alpha = 1.0 - confidence

            lower_cutoff = float(sample_returns.quantile(alpha))
            upper_cutoff = float(sample_returns.quantile(1.0 - alpha))

            lower_tail_values = sample_returns[sample_returns <= lower_cutoff]
            upper_tail_values = sample_returns[sample_returns >= upper_cutoff]

            long_cvar_return = (
                float(lower_tail_values.mean())
                if not lower_tail_values.empty
                else lower_cutoff
            )
            short_cvar_return = (
                float(upper_tail_values.mean())
                if not upper_tail_values.empty
                else upper_cutoff
            )

            long_tail_map[confidence] = {
                "var_return": lower_cutoff,
                "var_price": anchor_price * (1.0 + lower_cutoff),
                "expected_shortfall_return": long_cvar_return,
                "expected_shortfall_price": anchor_price * (1.0 + long_cvar_return),
            }
            short_tail_map[confidence] = {
                "var_return": upper_cutoff,
                "var_price": anchor_price * (1.0 + upper_cutoff),
                "expected_shortfall_return": short_cvar_return,
                "expected_shortfall_price": anchor_price * (1.0 + short_cvar_return),
            }

            tail_rows.append(
                {
                    "Confidence": f"{confidence:.0%}",
                    "Long VaR Floor": long_tail_map[confidence]["var_price"],
                    "Long CVaR Floor": long_tail_map[confidence]["expected_shortfall_price"],
                    "Short VaR Ceiling": short_tail_map[confidence]["var_price"],
                    "Short CVaR Ceiling": short_tail_map[confidence]["expected_shortfall_price"],
                }
            )

        return {
            "session_date": session_date,
            "window": window,
            "effective_window": effective_window,
            "anchor_price": anchor_price,
            "latest_price": latest_price,
            "sample_returns": sample_returns,
            "interval_confidence_levels": interval_confidence_levels,
            "tail_confidence_levels": tail_confidence_levels,
            "intervals": interval_map,
            "long_tail_levels": long_tail_map,
            "short_tail_levels": short_tail_map,
            "median_return": median_return,
            "median_price": median_price,
            "range_summary_table": pd.DataFrame(range_rows),
            "tail_summary_table": pd.DataFrame(tail_rows),
        }

    def build_trade_range_history_context(
        self,
        price_frame,
        window=200,
        windows=None,
        interval_confidence_levels=(0.95, 0.99),
        tail_confidence_levels=(0.95, 0.99),
        default_window=None,
    ):
        """
        Build ex-ante historical session trade-range metrics for one or more rolling
        windows using only information available before each session. Returns are
        open-to-close.
        """
        frame = self._coerce_ohlc_frame(price_frame)
        window_seed = windows if windows is not None else [window]
        window_options = self._normalize_windows(window_seed)
        default_window = self._select_default_window(
            window_options,
            default_window=default_window,
        )
        interval_confidence_levels = self._normalize_confidence_levels(interval_confidence_levels)
        tail_confidence_levels = self._normalize_confidence_levels(tail_confidence_levels)

        open_series = frame["Open"].astype(float)
        close_series = frame["Close"].astype(float)
        session_returns = close_series.div(open_series).sub(1.0).dropna()
        max_window = max(window_options)
        if len(session_returns) <= max_window:
            raise ValueError(
                f"Not enough completed sessions to build a {max_window}-session historical trade range context."
            )

        def lower_tail_mean(values, quantile_level):
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan
            cutoff = np.quantile(arr, quantile_level)
            tail_values = arr[arr <= cutoff]
            if tail_values.size == 0:
                return cutoff
            return tail_values.mean()

        def upper_tail_mean(values, quantile_level):
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan
            cutoff = np.quantile(arr, quantile_level)
            tail_values = arr[arr >= cutoff]
            if tail_values.size == 0:
                return cutoff
            return tail_values.mean()

        all_confidences = sorted(set(interval_confidence_levels).union(tail_confidence_levels))
        metrics_by_window = {}
        for rolling_window in window_options:
            metrics_by_confidence = {}
            for confidence in all_confidences:
                alpha = 1.0 - confidence
                interval_alpha = (1.0 - confidence) / 2.0

                lower_interval_return = (
                    session_returns.rolling(rolling_window).quantile(interval_alpha).shift(1).dropna()
                )
                upper_interval_return = (
                    session_returns.rolling(rolling_window).quantile(1.0 - interval_alpha).shift(1).dropna()
                )
                lower_var_return = session_returns.rolling(rolling_window).quantile(alpha).shift(1).dropna()
                upper_var_return = session_returns.rolling(rolling_window).quantile(1.0 - alpha).shift(1).dropna()
                lower_expected_shortfall_return = (
                    session_returns
                    .rolling(rolling_window)
                    .apply(lambda values, q=alpha: lower_tail_mean(values, q), raw=True)
                    .shift(1)
                    .dropna()
                )
                upper_expected_shortfall_return = (
                    session_returns
                    .rolling(rolling_window)
                    .apply(lambda values, q=1.0 - alpha: upper_tail_mean(values, q), raw=True)
                    .shift(1)
                    .dropna()
                )

                aligned_returns = session_returns.reindex(lower_var_return.index).dropna()
                lower_var_return = lower_var_return.reindex(aligned_returns.index)
                upper_var_return = upper_var_return.reindex(aligned_returns.index)
                lower_breaches = aligned_returns.lt(lower_var_return).astype(float)
                upper_breaches = aligned_returns.gt(upper_var_return).astype(float)
                lower_rolling_breach_rate = lower_breaches.rolling(rolling_window).mean().dropna()
                upper_rolling_breach_rate = upper_breaches.rolling(rolling_window).mean().dropna()

                lower_expected_breach_rate = pd.Series(
                    data=np.full(len(lower_rolling_breach_rate.index), alpha, dtype=float),
                    index=lower_rolling_breach_rate.index,
                )
                upper_expected_breach_rate = pd.Series(
                    data=np.full(len(upper_rolling_breach_rate.index), alpha, dtype=float),
                    index=upper_rolling_breach_rate.index,
                )

                open_for_interval = open_series.reindex(lower_interval_return.index)
                open_for_var = open_series.reindex(lower_var_return.index)
                open_for_es = open_series.reindex(lower_expected_shortfall_return.index)

                metrics_by_confidence[confidence] = {
                    "session_returns": session_returns,
                    "session_open": open_series,
                    "session_close": close_series,
                    "lower_interval_return": lower_interval_return,
                    "upper_interval_return": upper_interval_return,
                    "lower_interval_price": open_for_interval.mul(1.0 + lower_interval_return),
                    "upper_interval_price": open_for_interval.mul(1.0 + upper_interval_return),
                    "lower_var_return": lower_var_return,
                    "upper_var_return": upper_var_return,
                    "lower_var_price": open_for_var.mul(1.0 + lower_var_return),
                    "upper_var_price": open_for_var.mul(1.0 + upper_var_return),
                    "lower_expected_shortfall_return": lower_expected_shortfall_return,
                    "upper_expected_shortfall_return": upper_expected_shortfall_return,
                    "lower_expected_shortfall_price": open_for_es.mul(1.0 + lower_expected_shortfall_return),
                    "upper_expected_shortfall_price": open_for_es.mul(1.0 + upper_expected_shortfall_return),
                    "lower_breaches": lower_breaches,
                    "upper_breaches": upper_breaches,
                    "lower_rolling_breach_rate": lower_rolling_breach_rate,
                    "upper_rolling_breach_rate": upper_rolling_breach_rate,
                    "lower_expected_breach_rate": lower_expected_breach_rate,
                    "upper_expected_breach_rate": upper_expected_breach_rate,
                }

            metrics_by_window[rolling_window] = metrics_by_confidence

        return {
            "window": default_window,
            "windows": window_options,
            "default_window": default_window,
            "interval_confidence_levels": interval_confidence_levels,
            "tail_confidence_levels": tail_confidence_levels,
            "metrics_by_confidence": metrics_by_window[default_window],
            "metrics_by_window": metrics_by_window,
            "session_returns": session_returns,
            "session_open": open_series,
            "session_close": close_series,
        }

"""Close-based rolling analytics for pandas Series and DataFrames."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _coerce_price_frame(data, argument_name: str = "data") -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "price")
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError(f"{argument_name} must be a pandas Series or DataFrame.")


def _normalize_windows(windows) -> list[int]:
    if isinstance(windows, (int, np.integer)):
        normalized = [int(windows)]
    elif isinstance(windows, Iterable) and not isinstance(windows, (str, bytes)):
        normalized = [int(window) for window in windows]
    else:
        raise ValueError("windows must be an integer or an iterable of integers")

    if not normalized:
        raise ValueError("windows must contain at least one positive integer")
    if any(window <= 0 for window in normalized):
        raise ValueError("windows must contain positive integers")
    return normalized


def _coerce_periodic_risk_free_rate(risk_free_rate, index: pd.Index, annualization_factor: int) -> pd.Series:
    if isinstance(risk_free_rate, pd.Series):
        return risk_free_rate.astype(float).sort_index().reindex(index).ffill()
    if np.isscalar(risk_free_rate):
        annual_rate = float(risk_free_rate)
        if annual_rate <= -1.0:
            raise ValueError("risk_free_rate must be greater than -1.0")
        periodic_rate = (1.0 + annual_rate) ** (1.0 / annualization_factor) - 1.0
        return pd.Series(periodic_rate, index=index, dtype=float)
    raise TypeError("risk_free_rate must be a scalar annual rate or a pandas Series of aligned periodic rates.")


def _calculate_excess_returns(returns, risk_free_rate=0.0, annualization_factor: int = 252):
    periodic_rate = _coerce_periodic_risk_free_rate(
        risk_free_rate=risk_free_rate,
        index=returns.index,
        annualization_factor=annualization_factor,
    )
    if isinstance(returns, pd.DataFrame):
        return returns.sub(periodic_rate, axis=0)
    return returns - periodic_rate


def _rolling_downside_deviation(returns, window: int):
    downside_returns = returns.where(returns < 0, 0.0)
    return downside_returns.rolling(window=window).apply(lambda x: np.sqrt((x**2).mean()), raw=True)


def _rolling_sortino_ratio_frame(
    data,
    window: int,
    risk_free_rate=0.0,
    annualization_factor: int = 252,
    annualize: bool = True,
) -> pd.DataFrame:
    price_frame = _coerce_price_frame(data)
    excess_returns = _calculate_excess_returns(
        price_frame.pct_change(),
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )
    downside_deviation = _rolling_downside_deviation(excess_returns, window=window)
    ratio = excess_returns.rolling(window=window).mean() / downside_deviation
    if annualize:
        ratio = ratio * np.sqrt(annualization_factor)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _rolling_series_statistic_frame(series: pd.Series, windows, statistic: str, prefix: str) -> pd.DataFrame:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    normalized_windows = _normalize_windows(windows)
    statistic_frames = []
    for window in normalized_windows:
        rolling_series = getattr(series.rolling(window=window), statistic)()
        statistic_frames.append(rolling_series.rename(f"{prefix}_{window}"))
    return pd.concat(statistic_frames, axis=1)


class CloseAnalytics:
    """Compute rolling metrics derived from close-price data."""

    def moving_averages(self, series, windows=(21, 50, 200), ma_type="simple"):
        """Compute one or more moving-average variants for a price series."""
        windows = _normalize_windows(windows)
        moving_averages = pd.DataFrame(index=series.index)

        if ma_type == "simple":
            for window in windows:
                moving_averages[f"ma_{window}"] = series.rolling(window=window).mean()

        elif ma_type == "exponential":
            for window in windows:
                moving_averages[f"ema_{window}"] = series.ewm(span=window, adjust=False).mean()

        elif ma_type == "hull":
            for window in windows:
                half_window = window // 2
                sqrt_window = int(np.sqrt(window))
                wma1 = 2 * series.rolling(window=half_window).mean() - series.rolling(window=window).mean()
                moving_averages[f"hull_ma_{window}"] = wma1.rolling(window=sqrt_window).mean()

        elif ma_type == "tema":
            for window in windows:
                ema = series.ewm(span=window, adjust=False).mean()
                ema2 = ema.ewm(span=window, adjust=False).mean()
                ema3 = ema2.ewm(span=window, adjust=False).mean()
                moving_averages[f"tema_{window}"] = 3 * (ema - ema2) + ema3

        elif ma_type == "kama":
            for window in windows:
                change = series.diff(window - 1)
                volatility = series.diff().abs().rolling(window=window).sum()
                er = change / volatility
                sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) ** 2).fillna(0.0)
                kama = series.astype(float).copy()
                for i in range(window, len(series)):
                    kama.iat[i] = kama.iat[i - 1] + sc.iat[i] * (series.iat[i] - kama.iat[i - 1])
                moving_averages[f"kama_{window}"] = kama

        else:
            raise ValueError("Invalid moving average type. Use 'simple', 'exponential', 'hull', 'tema', or 'kama'.")

        moving_averages["original"] = series
        return moving_averages

    def rsi(self, series, windows=(21, 50, 200), indicator_type="RSI"):
        """Compute RSI or Rocket RS over multiple rolling windows."""
        windows = _normalize_windows(windows)
        if indicator_type not in ["RSI", "Rocket_RS"]:
            raise ValueError("Invalid indicator type. Use 'RSI' or 'Rocket_RS'.")

        indicators_df = pd.DataFrame(index=series.index)

        for window in windows:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean().replace(0, 1e-10)

            if indicator_type == "RSI":
                rs = gain / loss
                indicators_df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
            else:
                indicators_df[f"Rocket_RS_{window}"] = gain / loss

        return indicators_df

    def drawdowns(self, series, windows=(21, 50, 200)):
        """Compute rolling drawdowns over multiple windows."""
        windows = _normalize_windows(windows)
        drawdowns_df = pd.DataFrame(index=series.index)

        for window in windows:
            rolling_max = series.rolling(window=window).max()
            drawdowns_df[f"drawdown_{window}"] = series - rolling_max

        return drawdowns_df

    def skew(self, series, windows=range(1, 11)):
        """Compute rolling skew over multiple windows."""
        return _rolling_series_statistic_frame(series, windows=windows, statistic="skew", prefix="skew")

    def kurtosis(self, series, windows=range(1, 11)):
        """Compute rolling kurtosis over multiple windows."""
        return _rolling_series_statistic_frame(series, windows=windows, statistic="kurt", prefix="kurt")

    def std(self, series, windows=range(1, 11)):
        """Compute rolling standard deviation over multiple windows."""
        return _rolling_series_statistic_frame(series, windows=windows, statistic="std", prefix="std")

    def risk_adjusted_returns(
        self,
        data,
        windows,
        ratio_type="sharpe",
        risk_free_rate=0.0,
        threshold=0.0,
        annualization_factor=252,
    ):
        """Compute rolling Sharpe, Sortino, Omega, Calmar, or Sterling ratios."""

        windows = _normalize_windows(windows)
        data = _coerce_price_frame(data)
        single_window = len(windows) == 1
        single_series = data.shape[1] == 1

        returns = data.pct_change()
        excess_returns = _calculate_excess_returns(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )

        sortino_frames = {}
        if ratio_type == "sortino":
            for window in windows:
                sortino_frames[window] = _rolling_sortino_ratio_frame(
                    data,
                    window=window,
                    risk_free_rate=risk_free_rate,
                    annualization_factor=annualization_factor,
                    annualize=True,
                )
        output = []

        for col in returns.columns:
            r = returns[col]
            excess_r = excess_returns[col]

            for window in windows:
                if ratio_type == "sharpe":
                    mean_excess = excess_r.rolling(window).mean()
                    vol = excess_r.rolling(window).std()
                    ratio = np.sqrt(annualization_factor) * mean_excess / vol
                    ratio = ratio.where(vol > 0)

                elif ratio_type == "sortino":
                    ratio = sortino_frames[window][col]

                elif ratio_type == "omega":

                    def omega(x):
                        excess = x - threshold
                        gains = excess[excess > 0].sum()
                        losses = -excess[excess < 0].sum()
                        return np.nan if losses == 0 else gains / losses

                    ratio = r.rolling(window).apply(omega, raw=False)

                elif ratio_type == "calmar":

                    def calmar(x):
                        growth = (1 + x).cumprod()
                        peak = growth.cummax()
                        dd = growth / peak - 1
                        max_dd = dd.min()
                        if max_dd >= 0:
                            return np.nan
                        total_ret = growth.iloc[-1] - 1
                        ann_ret = (1 + total_ret) ** (annualization_factor / len(x)) - 1
                        return ann_ret / abs(max_dd)

                    ratio = r.rolling(window).apply(calmar, raw=False)

                elif ratio_type == "sterling":

                    def sterling(x):
                        growth = (1 + x).cumprod()
                        peak = growth.cummax()
                        dd = growth / peak - 1
                        worst = dd[dd < 0].nsmallest(3)
                        if worst.empty:
                            return np.nan
                        avg_dd = abs(worst).mean()
                        total_ret = growth.iloc[-1] - 1
                        ann_ret = (1 + total_ret) ** (annualization_factor / len(x)) - 1
                        return ann_ret / avg_dd

                    ratio = r.rolling(window).apply(sterling, raw=False)

                else:
                    raise ValueError("Invalid ratio_type")

                ratio = ratio.replace([np.inf, -np.inf], np.nan)
                ratio.name = (
                    f"{ratio_type}_ratio_{window}" if single_window and single_series else f"{col}_{ratio_type}_{window}"
                )
                output.append(ratio)

        return pd.concat(output, axis=1)

    def calculate_percentage_drop(self, data, windows=(14,)):
        """Calculate percentage drop from rolling highest closes over one or more windows."""
        windows = _normalize_windows(windows)
        if "Close" not in data.columns:
            raise ValueError("The DataFrame must contain a 'Close' column.")

        ticker_copy = data.copy()
        single_window = len(windows) == 1

        for window in windows:
            highest_high = ticker_copy["Close"].rolling(window=window, min_periods=1).max()
            highest_high_col = "HighestHigh" if single_window else f"HighestHigh_{window}"
            percentage_drop_col = "PercentageDrop" if single_window else f"PercentageDrop_{window}"

            ticker_copy[highest_high_col] = highest_high
            ticker_copy[percentage_drop_col] = -(
                (highest_high - ticker_copy["Close"]) / highest_high
            ) * 100

        return ticker_copy

    def vix_fix(self, data, windows=(22,)):
        """Compute the VIX Fix indicator from rolling highest closes."""
        windows = _normalize_windows(windows)

        if isinstance(data, pd.Series):
            close = data
        elif isinstance(data, pd.DataFrame):
            if "Close" not in data.columns:
                raise ValueError("The DataFrame must contain a 'Close' column.")
            close = data["Close"]
        else:
            raise TypeError("data must be a pandas Series or DataFrame with a 'Close' column.")

        if len(windows) == 1:
            window = windows[0]
            highest_close = close.rolling(window=window).max()
            return 100 * (highest_close - close) / highest_close

        output = {}
        for window in windows:
            highest_close = close.rolling(window=window).max()
            output[f"vix_fix_{window}"] = 100 * (highest_close - close) / highest_close

        return pd.DataFrame(output, index=close.index)

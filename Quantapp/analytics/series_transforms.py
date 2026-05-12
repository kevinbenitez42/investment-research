"""Reusable series resampling and alignment helpers."""

from __future__ import annotations

import pandas as pd


class SeriesTransforms:
    """Helpers for resampling time series and aligning feature sets."""

    @staticmethod
    def annualized_yield_to_periodic_rate(
        data,
        *,
        annualization_factor: int = 252,
        input_is_percent: bool = True,
        lag_periods: int = 0,
        reference_index=None,
        column: str = "Close",
    ) -> pd.Series:
        """
        Convert an annualized yield series into a periodic decimal rate.

        Data may be a Series or a DataFrame containing the selected yield column.
        Use ``lag_periods`` when the rate should be shifted before alignment, such
        as using the prior day's quoted yield for the current day's return.
        """
        if isinstance(data, pd.DataFrame):
            if column not in data.columns:
                raise ValueError(f"data must contain a {column!r} column.")
            annualized_yield = data[column]
        elif isinstance(data, pd.Series):
            annualized_yield = data
        else:
            raise TypeError("data must be a pandas Series or DataFrame.")

        annualization_factor = int(annualization_factor)
        if annualization_factor <= 0:
            raise ValueError("annualization_factor must be a positive integer.")

        lag_periods = int(lag_periods)
        if lag_periods < 0:
            raise ValueError("lag_periods must be non-negative.")

        annualized_yield = annualized_yield.astype(float).dropna().sort_index()
        if annualized_yield.empty:
            raise ValueError("data is empty after dropping NaNs.")

        annual_rate = annualized_yield.div(100.0) if input_is_percent else annualized_yield
        if (annual_rate <= -1.0).any():
            raise ValueError("annualized yields must be greater than -100%.")

        periodic_rate = (1.0 + annual_rate).pow(1.0 / annualization_factor).sub(1.0)
        if lag_periods:
            periodic_rate = periodic_rate.shift(lag_periods)
        if reference_index is not None:
            periodic_rate = periodic_rate.reindex(reference_index).ffill()

        return periodic_rate

    @staticmethod
    def resample(data, frequency: str = "monthly", how="last"):
        """
        Resample a Series or DataFrame at the requested frequency.

        Parameters
        ----------
        data
            Series or DataFrame with a DatetimeIndex-compatible index.
        frequency
            One of ``daily``, ``weekly``, ``monthly``, ``quarterly``, or ``yearly``.
        how
            Aggregation passed to pandas resample. Defaults to ``last``.
        """
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("data must be a pandas Series or DataFrame.")

        series_or_frame = data.copy()
        if not isinstance(series_or_frame.index, pd.DatetimeIndex):
            series_or_frame.index = pd.to_datetime(series_or_frame.index)
        series_or_frame = series_or_frame.sort_index()

        freq_map = {
            "weekly": "W-FRI",
            "monthly": "ME",
            "quarterly": "QE",
            "yearly": "YE",
        }
        normalized_frequency = str(frequency).lower()

        if normalized_frequency == "daily":
            return series_or_frame

        try:
            resample_freq = freq_map[normalized_frequency]
        except KeyError as exc:
            raise ValueError(
                "Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'."
            ) from exc

        resampler = series_or_frame.resample(resample_freq)
        if isinstance(how, str):
            if not hasattr(resampler, how):
                raise ValueError(f"Unsupported resample aggregation: {how!r}.")
            return getattr(resampler, how)()
        return resampler.agg(how)

    @staticmethod
    def returns(data, frequency: str = "monthly") -> pd.Series:
        """
        Calculate close-to-close returns at the requested sampling frequency.

        Parameters
        ----------
        data
            DataFrame with a DatetimeIndex and a ``Close`` column.
        frequency
            One of ``daily``, ``weekly``, ``monthly``, ``quarterly``, or ``yearly``.
        """
        sampled = SeriesTransforms.resample(data, frequency=frequency)
        if isinstance(sampled, pd.DataFrame):
            if "Close" not in sampled.columns:
                raise ValueError("data must contain a 'Close' column.")
            sampled = sampled["Close"]
        sampled_close = sampled
        return sampled_close.pct_change(fill_method=None).dropna()

    @staticmethod
    def align_features_to_index(features_dict, reference_index):
        """Reindex each feature frame to a shared index and forward-fill missing values."""
        aligned_features = {}
        for key, df in features_dict.items():
            aligned_features[key] = df.reindex(reference_index).ffill()
        return aligned_features

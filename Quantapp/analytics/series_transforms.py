"""Reusable series resampling and alignment helpers."""

from __future__ import annotations

import pandas as pd


class SeriesTransforms:
    """Helpers for resampling close-price series and aligning feature sets."""

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
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if "Close" not in data.columns:
            raise ValueError("data must contain a 'Close' column.")

        frame = data.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        close = frame["Close"]

        freq_map = {
            "weekly": "W-FRI",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "A",
        }
        normalized_frequency = str(frequency).lower()

        if normalized_frequency == "daily":
            return close.pct_change(fill_method=None).dropna()

        try:
            resample_freq = freq_map[normalized_frequency]
        except KeyError as exc:
            raise ValueError(
                "Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'."
            ) from exc

        resampled = close.resample(resample_freq).last()
        return resampled.pct_change(fill_method=None).dropna()

    @staticmethod
    def align_features_to_index(features_dict, reference_index):
        """Reindex each feature frame to a shared index and forward-fill missing values."""
        aligned_features = {}
        for key, df in features_dict.items():
            aligned_features[key] = df.reindex(reference_index).ffill()
        return aligned_features

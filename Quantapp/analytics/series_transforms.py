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

        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q",
            "yearly": "A",
        }
        try:
            resample_freq = freq_map[str(frequency).lower()]
        except KeyError as exc:
            raise ValueError(
                "Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'."
            ) from exc

        resampled = frame.resample(resample_freq).last()
        return resampled["Close"].pct_change().dropna()

    @staticmethod
    def align_features_to_index(features_dict, reference_index):
        """Reindex each feature frame to a shared index and forward-fill missing values."""
        aligned_features = {}
        for key, df in features_dict.items():
            aligned_features[key] = df.reindex(reference_index).ffill()
        return aligned_features

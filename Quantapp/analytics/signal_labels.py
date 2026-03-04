"""Signal labeling and screening helpers."""

from __future__ import annotations

import pandas as pd


class SignalLabels:
    """Threshold/categorical label generation for analytics outputs."""

    def z_score(self, time_series_data):
        """Z-score the latest row against each column's full-history mean/std."""
        mean = time_series_data.mean()
        std = time_series_data.std()
        latest_values = time_series_data.iloc[-1]
        return (latest_values - mean) / std

    def categorize_z_score(self, z):
        """Bucket z-score into integer bands."""
        if z > 3:
            return 3
        if z > 2:
            return 2
        if z > 1:
            return 1
        if z > -1:
            return 0
        if z > -2:
            return -1
        if z > -3:
            return -2
        return -3

    def create_sortino_negative_indicators(self, sortino_diff_50, sortino_diff_200):
        """Create latest 50/200-day sortino relative-performance indicator table."""
        latest_50 = sortino_diff_50.iloc[-1]
        latest_200 = sortino_diff_200.iloc[-1]

        indicators_df = pd.DataFrame(
            {
                "50_Day": latest_50 > 0,
                "200_Day": latest_200 > 0,
            }
        )

        indicators_df = indicators_df.reset_index()
        indicators_df.columns = [
            "Ticker",
            "Relative performance: 50 Day Sortino (Benchmark - asset)",
            "Relative performance: 200 Day Sortino (Benchmark - asset)",
        ]
        return indicators_df

    def create_sortino_std_deviation_table(self, rolling_sortino_ratio):
        """Categorize latest sortino values by z-score bucket."""
        z_scores = self.z_score(rolling_sortino_ratio)
        categories = z_scores.apply(self.categorize_z_score)

        return pd.DataFrame(
            {
                "Ticker": rolling_sortino_ratio.columns.tolist(),
                "Std Dev Direction": categories.tolist(),
            }
        )

    def create_price_std_deviation_table(self, price_data, window_sizes=(21, 50, 200)):
        """Categorize latest price-vs-rolling-mean by z-score bucket per window."""
        deviation_data = {"Ticker": price_data.columns.tolist()}

        for window in window_sizes:
            categories = []
            for ticker in price_data.columns:
                ticker_prices = price_data[ticker].dropna()
                if len(ticker_prices) >= window:
                    rolling_mean = ticker_prices.rolling(window=window).mean()
                    rolling_std = ticker_prices.rolling(window=window).std()

                    latest_price = ticker_prices.iloc[-1]
                    latest_mean = rolling_mean.iloc[-1]
                    latest_std = rolling_std.iloc[-1]

                    if latest_std == 0 or pd.isna(latest_std):
                        category = "Insufficient Data"
                    else:
                        z_score = (latest_price - latest_mean) / latest_std
                        category = self.categorize_z_score(z_score)
                else:
                    category = "Insufficient Data"
                categories.append(category)
            deviation_data[f"Std Dev Direction for {window}_Day Price"] = categories

        return pd.DataFrame(deviation_data)

    def filter_assets_by_positive_spread_std(self, asset_spreads):
        """True if latest spread is above mean+1σ of non-negative spread values."""
        positive_spreads = asset_spreads[asset_spreads >= 0]
        mean = positive_spreads.mean()
        std_dev = positive_spreads.std()
        latest_spread = asset_spreads.iloc[-1]
        threshold = mean + std_dev
        return latest_spread >= threshold

    def filter_assets_below_negative_std(self, asset_spreads):
        """Boolean mask where spread is below mean-0.75σ over negative observations."""
        if not isinstance(asset_spreads, pd.Series):
            raise TypeError("asset_spreads must be a pandas Series")

        negative_spreads = asset_spreads[asset_spreads < 0]
        if negative_spreads.empty:
            return pd.Series(dtype=bool)

        mean_negative = negative_spreads.mean()
        std_dev_negative = negative_spreads.std()
        threshold_negative = mean_negative - 0.75 * std_dev_negative
        return asset_spreads < threshold_negative

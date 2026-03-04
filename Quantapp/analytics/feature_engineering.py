"""Feature engineering transforms for time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineering:
    """Column-wise and pairwise feature transformation helpers."""

    def compute_pairwise(self, df, operations=("differences", "products", "sums", "ratios")):
        """
        Compute pairwise operations for each unique combination of columns.

        Returns a DataFrame indexed like `df`, with one engineered feature per
        requested operation and pair.
        """
        pairwise_df = pd.DataFrame(index=df.index)
        columns = df.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                for operation in operations:
                    if operation == "differences":
                        result = df[col1] - df[col2]
                        result_name = f"({col1} - {col2})"
                    elif operation == "products":
                        result = df[col1] * df[col2]
                        result_name = f"({col1} * {col2})"
                    elif operation == "sums":
                        result = df[col1] + df[col2]
                        result_name = f"({col1} + {col2})"
                    elif operation == "ratios":
                        col2_safe = df[col2].replace(0, 1e-10)
                        result = df[col1] / col2_safe
                        result_name = f"({col1} / {col2})"
                    else:
                        raise ValueError(
                            "Invalid operation type. Use 'differences', 'products', 'sums', or 'ratios'."
                        )

                    pairwise_df[result_name] = result

        return pairwise_df.dropna(axis=1, how="all")

    def compute_lags(self, df, lags=range(5, 200), steps=1):
        """Generate lagged features for each column and include the originals."""
        lagged_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            series = df[col]
            for i in range(0, len(lags), steps):
                lag = lags[i]
                lagged_df[f"{col}_lag_{lag}"] = series.shift(lag)

        return pd.concat([lagged_df, df], axis=1)

    def compute_non_linear(
        self,
        df,
        transformations=("polynomial",),
        degrees=range(2, 3),
        roots=(2, 3),
        logs=True,
        exponentials=True,
    ):
        """Generate nonlinear transforms for each input column."""
        transformed_features_list = []

        for col in df.columns:
            series = df[col]
            transformed_features = pd.DataFrame(index=series.index)

            if "polynomial" in transformations:
                for degree in degrees:
                    transformed_features[f"({col}^{degree})"] = series**degree

            if "exponential" in transformations and exponentials:
                transformed_features[f"{col}_exp"] = np.exp(series)

            if "root" in transformations:
                for root in roots:
                    transformed_features[f"({col}^(1/{root}))"] = series ** (1 / root)

            if "log" in transformations and logs:
                transformed_features[f"{col}_log"] = np.log(series + 1e-8)

            transformed_features_list.append(transformed_features)

        return pd.concat(transformed_features_list, axis=1, join="inner")

    def calculate_differences(self, df):
        """Compute all pairwise column differences, including self-pairs."""
        diff_ = pd.DataFrame(index=df.index)
        for column in df.columns:
            for column2 in df.columns:
                diff_[f"{column} min {column2}"] = df[column] - df[column2]
        return diff_

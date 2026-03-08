"""OHLC-based rolling analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .close_analytics import _normalize_windows


class OHLCAnalytics:
    """Compute rolling metrics that require OHLC market data."""

    def volatility(self, df, windows=(21, 50, 200), method="close-to-close"):
        """Compute rolling volatility using close-to-close or OHLC-based estimators."""
        windows = _normalize_windows(windows)
        volatility_df = pd.DataFrame(index=df.index)
        df_copy = df.copy()

        if method == "close-to-close":
            if "Close" not in df_copy.columns:
                raise ValueError("DataFrame must contain 'Close' column for close-to-close calculation.")

            returns = df_copy["Close"].pct_change()
            for window in windows:
                volatility_df[f"close_to_close_volatility_{window}"] = returns.rolling(window=window).std()

        elif method == "garman-klass":
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            df_copy["log_HL"] = np.log(df_copy["High"] / df_copy["Low"])
            df_copy["log_CO"] = np.log(df_copy["Close"] / df_copy["Open"])
            garman_klass_variance = 0.5 * df_copy["log_HL"] ** 2 - df_copy["log_CO"] ** 2
            for window in windows:
                rolling_variance = garman_klass_variance.rolling(window=window).mean()
                volatility_df[f"gk_volatility_{window}"] = np.sqrt(rolling_variance) * np.sqrt(252)

        elif method == "parkinson":
            if not all(col in df_copy.columns for col in ["High", "Low"]):
                raise ValueError("DataFrame must contain 'High' and 'Low' columns for Parkinson volatility calculation.")

            df_copy["log_HL"] = np.log(df_copy["High"] / df_copy["Low"])
            parkinson_variance = (1 / (4 * np.log(2))) * df_copy["log_HL"] ** 2
            for window in windows:
                rolling_variance = parkinson_variance.rolling(window=window).mean()
                volatility_df[f"parkinson_volatility_{window}"] = np.sqrt(rolling_variance) * np.sqrt(252)

        elif method == "rogers-satchell":
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            df_copy["log_HL"] = np.log(df_copy["High"] / df_copy["Low"])
            df_copy["log_CO"] = np.log(df_copy["Close"] / df_copy["Open"])
            rs_variance = (df_copy["log_HL"] ** 2 - df_copy["log_CO"] ** 2 + 2 * np.log(2) * (df_copy["log_CO"] ** 2)) / 2
            for window in windows:
                rolling_variance = rs_variance.rolling(window=window).mean()
                volatility_df[f"rs_volatility_{window}"] = np.sqrt(rolling_variance) * np.sqrt(252)

        elif method == "yang-zhang":
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            df_copy["log_HL"] = np.log(df_copy["High"] / df_copy["Low"])
            df_copy["log_CO"] = np.log(df_copy["Close"] / df_copy["Open"])
            yang_zhang_variance = 0.5 * (df_copy["log_HL"] ** 2 - df_copy["log_CO"] ** 2) + df_copy["Close"].pct_change() ** 2
            for window in windows:
                rolling_variance = yang_zhang_variance.rolling(window=window).mean()
                volatility_df[f"yz_volatility_{window}"] = np.sqrt(rolling_variance) * np.sqrt(252)

        elif method == "gk-yz":
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            df_copy["log_HL"] = np.log(df_copy["High"] / df_copy["Low"])
            df_copy["log_CO"] = np.log(df_copy["Close"] / df_copy["Open"])
            garman_klass_variance = 0.5 * df_copy["log_HL"] ** 2 - df_copy["log_CO"] ** 2
            yang_zhang_variance = 0.5 * (df_copy["log_HL"] ** 2 - df_copy["log_CO"] ** 2) + df_copy["Close"].pct_change() ** 2
            gk_yz_variance = (garman_klass_variance + yang_zhang_variance) / 2
            for window in windows:
                rolling_variance = gk_yz_variance.rolling(window=window).mean()
                volatility_df[f"gk_yz_volatility_{window}"] = np.sqrt(rolling_variance) * np.sqrt(252)

        else:
            raise ValueError(
                "Invalid method. Use 'close-to-close', 'garman-klass', 'parkinson', 'rogers-satchell', 'yang-zhang', or 'gk-yz'."
            )

        return volatility_df

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
        annualization = np.sqrt(252)

        if method == "close-to-close":
            if "Close" not in df_copy.columns:
                raise ValueError("DataFrame must contain 'Close' column for close-to-close calculation.")

            returns = df_copy["Close"].pct_change()
            for window in windows:
                volatility_df[f"close_to_close_volatility_{window}"] = returns.rolling(window=window).std()

        elif method in {"garman-klass", "parkinson", "rogers-satchell", "yang-zhang", "gk-yz"}:
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            log_hl = np.log(df_copy["High"] / df_copy["Low"])
            log_ho = np.log(df_copy["High"] / df_copy["Open"])
            log_lo = np.log(df_copy["Low"] / df_copy["Open"])
            log_co = np.log(df_copy["Close"] / df_copy["Open"])
            log_oc = np.log(df_copy["Open"] / df_copy["Close"].shift(1))
            log_hc = np.log(df_copy["High"] / df_copy["Close"])
            log_lc = np.log(df_copy["Low"] / df_copy["Close"])

            garman_klass_variance = 0.5 * (log_hl ** 2) - ((2 * np.log(2)) - 1) * (log_co ** 2)
            parkinson_variance = (log_hl ** 2) / (4 * np.log(2))
            rs_variance = (log_hc * log_ho) + (log_lc * log_lo)

            if method == "garman-klass":
                for window in windows:
                    rolling_variance = garman_klass_variance.rolling(window=window).mean().clip(lower=0)
                    volatility_df[f"gk_volatility_{window}"] = np.sqrt(rolling_variance) * annualization

            elif method == "parkinson":
                for window in windows:
                    rolling_variance = parkinson_variance.rolling(window=window).mean().clip(lower=0)
                    volatility_df[f"parkinson_volatility_{window}"] = np.sqrt(rolling_variance) * annualization

            elif method == "rogers-satchell":
                for window in windows:
                    rolling_variance = rs_variance.rolling(window=window).mean().clip(lower=0)
                    volatility_df[f"rs_volatility_{window}"] = np.sqrt(rolling_variance) * annualization

            elif method == "yang-zhang":
                for window in windows:
                    if window < 2:
                        volatility_df[f"yz_volatility_{window}"] = np.nan
                        continue

                    # Yang-Zhang combines overnight variance, open-to-close variance,
                    # and the Rogers-Satchell estimator with a window-dependent weight.
                    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
                    overnight_variance = log_oc.rolling(window=window).var()
                    open_to_close_variance = log_co.rolling(window=window).var()
                    rs_component = rs_variance.rolling(window=window).mean()
                    yz_variance = (
                        overnight_variance
                        + (k * open_to_close_variance)
                        + ((1 - k) * rs_component)
                    ).clip(lower=0)
                    volatility_df[f"yz_volatility_{window}"] = np.sqrt(yz_variance) * annualization

            elif method == "gk-yz":
                for window in windows:
                    if window < 2:
                        volatility_df[f"gk_yz_volatility_{window}"] = np.nan
                        continue

                    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
                    overnight_variance = log_oc.rolling(window=window).var()
                    open_to_close_variance = log_co.rolling(window=window).var()
                    rs_component = rs_variance.rolling(window=window).mean()
                    yz_variance = (
                        overnight_variance
                        + (k * open_to_close_variance)
                        + ((1 - k) * rs_component)
                    )
                    gk_yz_variance = (
                        garman_klass_variance.rolling(window=window).mean()
                        + yz_variance
                    ).div(2).clip(lower=0)
                    volatility_df[f"gk_yz_volatility_{window}"] = np.sqrt(gk_yz_variance) * annualization

        else:
            raise ValueError(
                "Invalid method. Use 'close-to-close', 'garman-klass', 'parkinson', 'rogers-satchell', 'yang-zhang', or 'gk-yz'."
            )

        return volatility_df

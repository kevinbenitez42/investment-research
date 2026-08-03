import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from Quantapp.analytics.series_utils import coerce_datetime_index, coerce_series


def _download_market_history(*args, **kwargs):
    """Download market history through the data package compatibility facade."""
    from Quantapp.data.yf import download

    return download(*args, **kwargs)


class FactorRegressionModel:
    def build_ff5_proxy_factor_returns(self, proxy_returns):
        """
        Build ETF-proxy FF-style factor returns from a returns panel.

        Required columns in proxy_returns:
        SPY, SIZE, VLUE, QUAL, USMV, MTUM, BIL
        """
        required = ["SPY", "SIZE", "VLUE", "QUAL", "USMV", "MTUM", "BIL"]
        missing = [c for c in required if c not in proxy_returns.columns]
        if missing:
            raise ValueError(f"Missing required proxy return columns: {missing}")

        factor_returns = pd.DataFrame(index=proxy_returns.index)
        factor_returns["Mkt-RF"] = proxy_returns["SPY"] - proxy_returns["BIL"]
        factor_returns["SMB"] = proxy_returns["SIZE"] - proxy_returns["SPY"]
        factor_returns["HML"] = proxy_returns["VLUE"] - proxy_returns["SPY"]
        factor_returns["RMW"] = proxy_returns["QUAL"] - proxy_returns["SPY"]
        factor_returns["CMA"] = proxy_returns["USMV"] - proxy_returns["MTUM"]

        return {
            "all": factor_returns,
            "capm": factor_returns[["Mkt-RF"]].copy(),
            "ff3": factor_returns[["Mkt-RF", "SMB", "HML"]].copy(),
            "ff5": factor_returns[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].copy(),
        }

    def rolling_factor_regression(
        self,
        stock_returns,
        rf_series,
        factor_returns,
        window,
        auto_window=True,
        verbose=False,
        annualization=252,
    ):
        """
        Rolling OLS regression on asset excess returns vs factor returns.

        Returns columns:
        alpha, <factor>_beta..., r_squared, adj_r_squared,
        idio_vol_daily, idio_vol_annualized.
        """
        stock_returns = coerce_series(stock_returns, "stock_returns")
        rf_series = coerce_series(rf_series, "rf_series", preferred_column="BIL")
        if not isinstance(factor_returns, pd.DataFrame):
            raise TypeError("factor_returns must be a pandas DataFrame.")

        stock = coerce_datetime_index(stock_returns)
        rf = coerce_datetime_index(rf_series)
        factors = coerce_datetime_index(factor_returns)
        factor_cols = list(factors.columns)

        aligned = pd.concat(
            [stock.rename("stock"), rf.rename("rf"), factors],
            axis=1,
            join="inner",
        ).dropna()

        if aligned.empty:
            raise ValueError(
                "No overlapping non-null dates between stock returns, rf series, and factor returns."
            )

        effective_window = int(window)
        if len(aligned) < effective_window:
            if auto_window:
                effective_window = len(aligned)
                if verbose:
                    print(
                        f"Requested window={window} but only {len(aligned)} aligned rows are available. "
                        f"Using window={effective_window}."
                    )
            else:
                raise ValueError(
                    f"Not enough aligned observations for window={window}. Got {len(aligned)}."
                )

        min_obs = len(factor_cols) + 2  # const + factors + at least 1 residual d.o.f.
        if effective_window < min_obs:
            raise ValueError(
                f"Aligned history is too short for regression. Need at least {min_obs} rows, got {effective_window}."
            )

        if verbose:
            print(
                f"Aligned sample: {aligned.index.min().date()} to {aligned.index.max().date()} "
                f"({len(aligned)} rows), window={effective_window}."
            )

        y = aligned["stock"] - aligned["rf"]
        X = sm.add_constant(aligned[factor_cols], has_constant="add")
        rolling_fit = RollingOLS(
            y,
            X,
            window=effective_window,
            min_nobs=effective_window,
            missing="drop",
        ).fit(params_only=False)

        valid_index = rolling_fit.params["const"].dropna().index
        rolling_results_df = pd.DataFrame(index=valid_index)
        rolling_results_df.index.name = "date"
        rolling_results_df["alpha"] = rolling_fit.params.loc[valid_index, "const"]
        rolling_results_df["r_squared"] = rolling_fit.rsquared.loc[valid_index]
        rolling_results_df["adj_r_squared"] = rolling_fit.rsquared_adj.loc[valid_index]

        # The previous implementation used pandas' residual std (ddof=1).
        # With an intercept, rolling residuals are zero-mean, so this is exactly
        # sqrt(SSR / (n - 1)) and does not require refitting every window.
        residual_denominator = rolling_fit.nobs.loc[valid_index] - 1
        rolling_results_df["idio_vol_daily"] = np.sqrt(
            rolling_fit.ssr.loc[valid_index] / residual_denominator
        )
        rolling_results_df["idio_vol_annualized"] = (
            rolling_results_df["idio_vol_daily"] * np.sqrt(annualization)
        )
        for factor in factor_cols:
            rolling_results_df[f"{factor}_beta"] = rolling_fit.params.loc[valid_index, factor]
        rolling_results_df.attrs["window_used"] = effective_window
        rolling_results_df.attrs["aligned_start"] = aligned.index.min().strftime("%Y-%m-%d")
        rolling_results_df.attrs["aligned_end"] = aligned.index.max().strftime("%Y-%m-%d")
        return rolling_results_df

    def run_ff5_proxy_analysis(
        self,
        asset_ticker,
        period="max",
        interval="1d",
        window=252,
        auto_window=True,
        verbose=False,
        proxy_tickers=None,
    ):
        """
        End-to-end ETF-proxy FF-style rolling analysis.

        Returns a dict with prices, returns, factor sets, and rolling regression output.
        """
        if proxy_tickers is None:
            proxy_tickers = ["SPY", "SIZE", "VLUE", "QUAL", "USMV", "MTUM", "BIL"]

        panel = _download_market_history(proxy_tickers, period=period, interval=interval, progress=False)
        if panel.empty:
            raise ValueError("Failed to download proxy ticker data.")
        if isinstance(panel.columns, pd.MultiIndex):
            if "Close" in panel.columns.get_level_values(0):
                proxy_prices = panel["Close"].copy()
            elif "Close" in panel.columns.get_level_values(1):
                proxy_prices = panel.xs("Close", axis=1, level=1).copy()
            else:
                raise ValueError("Unable to locate 'Close' level in proxy download columns.")
        else:
            if "Close" not in panel.columns:
                raise ValueError("Proxy download did not include a 'Close' column.")
            proxy_prices = pd.DataFrame({proxy_tickers[0]: panel["Close"]})

        proxy_returns = proxy_prices.pct_change().dropna()
        factor_sets = self.build_ff5_proxy_factor_returns(proxy_returns)

        asset_px = _download_market_history(asset_ticker, period=period, interval=interval, progress=False)
        if asset_px.empty:
            raise ValueError(f"Failed to download data for asset ticker '{asset_ticker}'.")
        if isinstance(asset_px.columns, pd.MultiIndex):
            if "Close" in asset_px.columns.get_level_values(0):
                asset_close = asset_px["Close"]
            elif "Close" in asset_px.columns.get_level_values(1):
                asset_close = asset_px.xs("Close", axis=1, level=1)
            else:
                raise ValueError(f"Failed to locate 'Close' data for asset ticker '{asset_ticker}'.")
        else:
            if "Close" not in asset_px.columns:
                raise ValueError(f"Failed to locate 'Close' data for asset ticker '{asset_ticker}'.")
            asset_close = asset_px["Close"]

        stock_returns = coerce_series(asset_close, "asset close prices").pct_change().dropna()

        rolling_results = self.rolling_factor_regression(
            stock_returns=stock_returns,
            rf_series=proxy_returns["BIL"],
            factor_returns=factor_sets["ff5"],
            window=window,
            auto_window=auto_window,
            verbose=verbose,
        )

        return {
            "proxy_prices": proxy_prices,
            "proxy_returns": proxy_returns,
            "factor_returns_all": factor_sets["all"],
            "factor_returns_capm": factor_sets["capm"],
            "factor_returns_ff3": factor_sets["ff3"],
            "factor_returns_ff5": factor_sets["ff5"],
            "asset_returns": stock_returns,
            "stock_returns": stock_returns,
            "rolling_results": rolling_results,
        }

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import holidays
from statsmodels.tsa.seasonal import STL
from scipy.stats import entropy as scipy_entropy
try:
    import investpy
except ModuleNotFoundError:  # Optional dependency for selected data workflows.
    investpy = None
import requests 
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

class Models:
    def __init__(self):
        pass
    
    def _coerce_series(self, obj, name, preferred_col=None):
        """Coerce Series-like input to pd.Series; allow one-column DataFrame."""
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, pd.DataFrame):
            if preferred_col is not None and preferred_col in obj.columns:
                out = obj[preferred_col]
                if isinstance(out, pd.DataFrame):
                    return out.iloc[:, 0]
                return out
            if obj.shape[1] == 1:
                return obj.iloc[:, 0]
            raise TypeError(
                f"{name} must be a Series or single-column DataFrame. "
                f"Received DataFrame with columns: {list(obj.columns)}"
            )
        raise TypeError(f"{name} must be a pandas Series.")

    def _normalize_datetime_index(self, obj):
        """Return a copy with a naive, sorted DatetimeIndex."""
        out = obj.copy()
        out.index = pd.to_datetime(out.index)
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
        return out.sort_index()

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
        stock_returns = self._coerce_series(stock_returns, "stock_returns")
        rf_series = self._coerce_series(rf_series, "rf_series", preferred_col="BIL")
        if not isinstance(factor_returns, pd.DataFrame):
            raise TypeError("factor_returns must be a pandas DataFrame.")

        stock = self._normalize_datetime_index(stock_returns)
        rf = self._normalize_datetime_index(rf_series)
        factors = self._normalize_datetime_index(factor_returns)
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

        results = []
        for end in range(effective_window, len(aligned) + 1):
            window_data = aligned.iloc[end - effective_window : end]
            y = window_data["stock"] - window_data["rf"]
            X = sm.add_constant(window_data[factor_cols], has_constant="add")
            model = sm.OLS(y, X).fit()

            idio_vol_daily = model.resid.std()
            regression_result = {
                "date": window_data.index[-1],
                "alpha": model.params["const"],
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "idio_vol_daily": idio_vol_daily,
                "idio_vol_annualized": idio_vol_daily * np.sqrt(annualization),
            }
            for factor in factor_cols:
                regression_result[f"{factor}_beta"] = model.params[factor]
            results.append(regression_result)

        rolling_results_df = pd.DataFrame(results)
        rolling_results_df.set_index("date", inplace=True)
        rolling_results_df.attrs["window_used"] = effective_window
        rolling_results_df.attrs["aligned_start"] = aligned.index.min().strftime("%Y-%m-%d")
        rolling_results_df.attrs["aligned_end"] = aligned.index.max().strftime("%Y-%m-%d")
        return rolling_results_df

    def rolling_regression(
        self,
        data,
        rf_series,
        factor_returns,
        window,
        auto_window=True,
        verbose=False,
        annualization=252,
    ):
        """
        Backward-compatible wrapper around rolling_factor_regression.

        `data` may be:
        - OHLCV DataFrame with 'Close' column
        - price Series
        - return Series (if `as_returns=True` is handled upstream)
        """
        if isinstance(data, pd.DataFrame):
            if "Close" not in data.columns:
                raise ValueError("DataFrame input must include a 'Close' column.")
            stock_returns = data["Close"].pct_change().dropna()
        elif isinstance(data, pd.Series):
            stock_returns = data.dropna()
        else:
            raise TypeError("data must be a pandas DataFrame (with 'Close') or a pandas Series.")

        return self.rolling_factor_regression(
            stock_returns=stock_returns,
            rf_series=rf_series,
            factor_returns=factor_returns,
            window=window,
            auto_window=auto_window,
            verbose=verbose,
            annualization=annualization,
        )

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

        panel = yf.download(proxy_tickers, period=period, interval=interval, progress=False)
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

        asset_px = yf.download(asset_ticker, period=period, interval=interval, progress=False)
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

        stock_returns = self._coerce_series(asset_close, "asset close prices").pct_change().dropna()

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

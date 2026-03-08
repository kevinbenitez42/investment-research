"""Market-data retrieval and preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


class MarketDataUtils:
    """Helpers for downloading and preparing market data for analytics workflows."""

    def load_and_prepare_data(
        self,
        tickers="SPY",
        period: str = "5y",
        gen_returns: bool = False,
        gen_log_returns: bool = False,
        gen_cumulative_returns: bool = False,
        train_percentage: float = 0.8,
    ):
        """Download Yahoo Finance price history and return full/train/test splits by ticker."""
        if isinstance(tickers, str):
            tickers = [tickers]

        data_dict = {}

        for ticker in tickers:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period)

            df = pd.DataFrame(data)
            df.index.name = "Date"
            df = df.asfreq("B").ffill()
            df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")).ffill()

            raw_data_df = df.copy()

            if gen_returns:
                pct_returns_df = raw_data_df.pct_change()
                pct_returns_df.columns = [col + "_Returns" for col in pct_returns_df.columns]
                df = pd.concat([df, pct_returns_df], axis=1)

            if gen_log_returns:
                log_returns_df = np.log1p(raw_data_df.pct_change())
                log_returns_df.columns = [col + "_Log_Returns" for col in log_returns_df.columns]
                df = pd.concat([df, log_returns_df], axis=1)

            if gen_cumulative_returns:
                pct_returns_raw_df = raw_data_df.pct_change()
                cumulative_returns_raw_df = (1 + pct_returns_raw_df).cumprod() - 1
                cumulative_returns_raw_df.columns = [col + "_Cumulative_Returns" for col in cumulative_returns_raw_df.columns]
                df = pd.concat([df, cumulative_returns_raw_df], axis=1)

            split_index = int(len(df) * train_percentage)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()

            data_dict[ticker] = {
                "full": df,
                "train": train_df,
                "test": test_df,
            }

        return data_dict

    def n_positive_days(self, ticker: str = "SPY", number_of_days: int = 21) -> pd.Series:
        """Rolling share of sessions in which close finished above open."""
        data = yf.Ticker(ticker).history(period="max", interval="1d")
        data = data[["Close", "Open"]]
        closed_higher = (data["Close"] >= data["Open"]).astype(int)
        return closed_higher.rolling(number_of_days).sum() / number_of_days

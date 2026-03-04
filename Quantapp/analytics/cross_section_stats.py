"""Cross-sectional relationship and pair statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


class CrossSectionStats:
    """Helpers for pairwise spreads/correlations/cointegration diagnostics."""

    def pairwise_spreads(self, etf_dataframes, window=20):
        """
        Create pairwise spreads of rolling returns between assets in each category.
        """
        pairwise_spreads = {}

        for category, df in etf_dataframes.items():
            if df.shape[1] <= 1:
                continue

            valid_tickers = [ticker for ticker in df.columns if not df[ticker].isna().all()]
            if len(valid_tickers) < 2:
                continue

            category_spreads = pd.DataFrame(index=df.index)

            for i in range(len(valid_tickers)):
                for j in range(i + 1, len(valid_tickers)):
                    ticker1, ticker2 = valid_tickers[i], valid_tickers[j]
                    valid_data = df[[ticker1, ticker2]].dropna()
                    if valid_data.empty:
                        continue

                    returns1 = df[ticker1].pct_change(window)
                    returns2 = df[ticker2].pct_change(window)
                    category_spreads[f"{ticker1}-{ticker2}"] = returns1 - returns2

            if not category_spreads.empty:
                pairwise_spreads[category] = category_spreads

        return pairwise_spreads

    def get_sorted_correlations(self, corr_matrix):
        """Return pair names and correlation values sorted ascending."""
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        corr_pairs = []

        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if mask[i, j]:
                    corr_pairs.append((f"{row}-{col}", corr_matrix.iloc[i, j]))

        corr_pairs.sort(key=lambda x: x[1])
        return zip(*corr_pairs)

    def get_cointegration_pvals(self, df, correlation_pairs):
        """Compute residual ADF cointegration p-values for listed column pairs."""
        p_values = []

        for pair_name in correlation_pairs:
            ticker1, ticker2 = pair_name.split("-")

            series1 = df[ticker1].dropna()
            series2 = df[ticker2].dropna()
            common_idx = series1.index.intersection(series2.index)

            if len(common_idx) < 10:
                p_values.append(1.0)
                continue

            s1 = series1.loc[common_idx]
            s2 = series2.loc[common_idx]

            try:
                x = np.array(s2).reshape(-1, 1)
                x = np.hstack((np.ones(x.shape[0]).reshape(-1, 1), x))
                y = np.array(s1).reshape(-1, 1)
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                residuals = y - x.dot(beta)
                adf_result = adfuller(residuals.flatten(), autolag="AIC")
                p_values.append(adf_result[1])
            except Exception:
                p_values.append(1.0)

        return correlation_pairs, p_values

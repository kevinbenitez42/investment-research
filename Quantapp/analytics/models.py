import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import holidays
from statsmodels.tsa.seasonal import STL
from scipy.stats import entropy as scipy_entropy
import investpy
import requests 
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

class Models:
    def __init__(self):
        pass
    
    def rolling_regression(self, data, rf_series, factor_returns, window):
        """
        Computes a rolling OLS regression on an asset's excess returns relative to the risk-free rate
        using provided factor returns over a specified rolling window.
        
        For each rolling window, it calculates:
        - alpha (intercept)
        - beta for each factor
        - r_squared
        - adjusted r_squared
        
        Excess returns = stock_returns - returns["BIL"]

        Parameters:
            stock_returns (pd.Series): Series of asset returns.
            returns (pd.DataFrame): DataFrame containing returns for various tickers.
                                Must include the risk-free rate under the column "BIL".
            factor_returns (pd.DataFrame): DataFrame containing factor returns.
            window (int): The number of periods in each rolling window.

        Returns:
            pd.DataFrame: A DataFrame indexed by the end date of each window with columns:
                        "alpha", "<factor>_beta" for each factor, "r_squared", "adj_r_squared".
        """

        asset_close_returns = data['Close'].pct_change().dropna()
    
        results = []
        # Loop over rolling window periods
        for end in range(window, len(asset_close_returns) + 1):
            # Define the window of dates for the current regression
            window_index = asset_close_returns.index[end - window:end]
            # Extract the window data
            window_asset_close_returns = asset_close_returns.loc[window_index]
            window_rf = rf_series.loc[window_index]
            window_excess = window_asset_close_returns - window_rf
            window_factors = factor_returns.loc[window_index]

            # Prepare independent variables with a constant
            X = sm.add_constant(window_factors)
            y = window_excess
            
            # Run the OLS regression
            model = sm.OLS(y, X).fit()
            
            # Extract regression parameters
            regression_result = {"date": window_index[-1],
                                "alpha": model.params["const"],
                                "r_squared": model.rsquared,
                                "adj_r_squared": model.rsquared_adj}
            
            for factor in window_factors.columns:
                regression_result[f"{factor}_beta"] = model.params[factor]
            
            results.append(regression_result)
        
        # Create a DataFrame from the list of results and set the index to the window end dates
        rolling_results_df = pd.DataFrame(results)
        rolling_results_df.set_index("date", inplace=True)
        
        return rolling_results_df

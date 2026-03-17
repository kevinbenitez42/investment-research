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

class Helper:
    def simplify_datetime_index(self,series):
        """
        Simplifies the DateTime index of a Series to contain only the date (YYYY-MM-DD),
        maintaining it as a DateTimeIndex without timezone information.
        
        Parameters:
            series (pd.Series): The input Series with a DateTimeIndex.
        
        Returns:
            pd.Series: The Series with the DateTime index simplified to YYYY-MM-DD.
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("The Series index must be a DateTimeIndex.")
        
        # Remove timezone information if present
        if series.index.tz is not None:
            series = series.copy()
            series.index = series.index.tz_convert('UTC').tz_localize(None)
        
        # Normalize the index to remove the time component
        series.index = series.index.normalize()
        
        return series
    
    def fill_missing_dates(self, data, freq='D', method='ffill'):
        """
        Fill missing dates in a Series or DataFrame, forward-filling missing values.

        Parameters:
            data (pd.Series or pd.DataFrame): Input data with a DateTimeIndex.
            freq (str): Frequency for the new date index (default 'D' for daily).
            method (str): Method for filling missing values (default 'ffill').

        Returns:
            pd.Series or pd.DataFrame: Data with missing dates filled and values forward-filled.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Input must have a DatetimeIndex.")

        date_index = pd.date_range(start=data.index[0], end=data.index[-1], freq=freq)
        if isinstance(data, pd.Series):
            filled = data.reindex(date_index)
            filled = filled.fillna(method=method)
            return filled
        elif isinstance(data, pd.DataFrame):
            filled = data.reindex(date_index)
            filled = filled.fillna(method=method)
            return filled
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
    
    def monthly_to_daily(self,data):
        dates = pd.date_range(data.index[0], data.index[-1], freq='D')
        s_daily = data.reindex(dates, method='ffill')
        return s_daily.fillna(0)
    
    def remove_weekends_and_holidays(df, country='US'):
        """
        Removes weekend and holiday rows from a DataFrame with a DateTime index.

        Parameters:
            df (pd.DataFrame): DataFrame with DateTime index.
            country (str): Country code for holidays. Default is 'US'.

        Returns:
            pd.DataFrame: DataFrame without weekend and holiday data.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DateTimeIndex")

        # Remove weekends
        df_weekdays = df[df.index.dayofweek < 5]

        # Get holidays
        country_holidays = holidays.CountryHoliday(country)

        # Remove holidays
        df_clean = df_weekdays[~df_weekdays.index.normalize().isin(country_holidays)]

        return df_clean
    
    def train_test_split(self, series, percent_split):
        X = series.values
        size = int(len(X) * percent_split)
        y_train,y_test =X [0: size], X[size:len(X)]

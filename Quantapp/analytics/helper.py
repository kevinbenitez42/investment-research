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

    @staticmethod
    def is_futures_ticker(ticker_str):
        symbol = str(ticker_str).strip().upper()
        return symbol.endswith("=F")

    def build_equity_like_trade_range_source(
        self,
        ticker_str,
        daily_frame,
        *,
        intraday_period="60d",
        intraday_interval="30m",
        session_timezone="America/New_York",
    ):
        """
        Build the Open/Close source used by trade-range style analytics.

        Equities keep the supplied daily bars. Futures keep their full daily history too,
        but we attach NY cash-session overrides for the latest session so the current cone
        can anchor off 09:30-16:00 ET data without collapsing the history panel to the
        recent intraday download window.
        """
        if not isinstance(daily_frame, pd.DataFrame):
            raise TypeError("daily_frame must be a pandas DataFrame.")
        if "Open" not in daily_frame.columns or "Close" not in daily_frame.columns:
            raise ValueError("daily_frame must contain 'Open' and 'Close' columns.")

        base_frame = daily_frame.copy()
        base_frame.attrs["source_ticker"] = str(ticker_str).upper()
        if not self.is_futures_ticker(ticker_str):
            base_frame.attrs["session_mode"] = "daily_bar"
            return base_frame

        base_frame.attrs["session_mode"] = "futures_daily_bar"

        try:
            intraday_frame = yf.Ticker(str(ticker_str)).history(
                period=intraday_period,
                interval=intraday_interval,
            )
        except Exception:
            intraday_frame = pd.DataFrame()

        if intraday_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_fallback"
            return base_frame

        intraday_frame = intraday_frame.copy()
        if not isinstance(intraday_frame.index, pd.DatetimeIndex):
            intraday_frame.index = pd.to_datetime(intraday_frame.index)

        if intraday_frame.index.tz is None:
            intraday_frame.index = intraday_frame.index.tz_localize(session_timezone)
        else:
            intraday_frame.index = intraday_frame.index.tz_convert(session_timezone)

        intraday_frame = intraday_frame.sort_index()
        intraday_frame = intraday_frame[intraday_frame.index.dayofweek < 5]
        intraday_frame = intraday_frame.between_time("09:30", "16:00", inclusive="both")

        required_columns = [column for column in ("Open", "High", "Low", "Close") if column in intraday_frame.columns]
        if "Open" not in required_columns or "Close" not in required_columns:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_missing_columns"
            return base_frame

        if intraday_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_empty_session"
            return base_frame

        aggregation_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in intraday_frame.columns:
            aggregation_map["Volume"] = "sum"

        session_frame = (
            intraday_frame.groupby(intraday_frame.index.normalize())
            .agg(aggregation_map)
            .dropna(subset=["Open", "Close"])
            .sort_index()
        )
        if session_frame.empty:
            base_frame.attrs["session_mode"] = "futures_daily_bar_intraday_no_sessions"
            return base_frame

        session_frame.index = pd.DatetimeIndex(session_frame.index).tz_localize(None)
        latest_session = session_frame.iloc[-1]

        base_frame.attrs["session_mode"] = "hybrid_daily_history_ny_anchor"
        base_frame.attrs["current_session_anchor_price"] = float(latest_session["Open"])
        base_frame.attrs["current_session_reference_price"] = float(base_frame["Close"].iloc[-1])
        base_frame.attrs["current_session_latest_price"] = float(latest_session["Close"])
        base_frame.attrs["current_session_date"] = pd.Timestamp(session_frame.index[-1])
        base_frame.attrs["trade_range_recent_session_count"] = int(len(session_frame))
        return base_frame
    
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

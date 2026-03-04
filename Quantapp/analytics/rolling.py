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

from .algorithm import Algorithm
from .cross_section_stats import CrossSectionStats
from .feature_engineering import FeatureEngineering
from .momentum_analytics import MomentumAnalytics
from .risk_distribution_analytics import RiskDistributionAnalytics
from .risk_relative_analytics import RiskRelativeAnalytics
from .signal_labels import SignalLabels
from .time_features import TimeFeatures

class TimeSeriesAnalytics:
    
    def __init__(self):
        self.algorithm = Algorithm()
        self.time_features = TimeFeatures()
        self.feature_engineering = FeatureEngineering()
        self.cross_section_stats = CrossSectionStats()
        self.momentum_analytics = MomentumAnalytics()
        self.risk_distribution_analytics = RiskDistributionAnalytics()
        self.risk_relative_analytics = RiskRelativeAnalytics()
        self.signal_labels = SignalLabels()
        
    def compute_holiday_features(self, df, country='US'):
        """
        Returns all dates within the date range of the input DataFrame or Series, labeling holidays,
        adding binary 'is_holiday', 'holiday_id', 'days_until_next_holiday', and countdown columns for each unique holiday.

        Args:
            df (pd.DataFrame or pd.Series): Input data with a datetime index.
            country (str): The country code for generating holidays (default is 'US').

        Returns:
            pd.DataFrame: A DataFrame with all dates in the index, binary 'is_holiday', unique 'holiday_id',
                        'days_until_next_holiday', and countdown columns for each holiday.
        """
        # Ensure the DataFrame or Series has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame or Series must have a DatetimeIndex")

        # Extract the range of years from the input data's index
        start_year = df.index.year.min()
        end_year = df.index.year.max()

        # Generate holiday dates and unique IDs for holidays within the date range
        holiday_names = {}
        holiday_id_map = {}  # Maps holiday name to a unique ID
        holiday_counter = 1  # Unique ID for each holiday name
        holiday_dates_map = {}  # Map for each holiday's dates

        for year in range(start_year, end_year + 1):
            country_holidays = holidays.CountryHoliday(country, years=[year])
            for date, name in country_holidays.items():
                timestamp_date = pd.Timestamp(date)
                holiday_names[timestamp_date] = name
                # Assign a unique ID to each holiday name, consistent across years
                if name not in holiday_id_map:
                    holiday_id_map[name] = holiday_counter
                    holiday_counter += 1
                # Store all dates for each unique holiday name
                if name not in holiday_dates_map:
                    holiday_dates_map[name] = []
                holiday_dates_map[name].append(timestamp_date)

        # Sort holiday dates within each holiday
        for name in holiday_dates_map:
            holiday_dates_map[name] = sorted(holiday_dates_map[name])

        # Create 'is_holiday' column: 1 for holidays, 0 for non-holidays
        is_holiday = df.index.map(lambda x: 1 if x in holiday_names else 0)

        # Create 'holiday_id' column: unique ID based on holiday name, 0 for non-holidays
        holiday_id = df.index.map(lambda x: holiday_id_map.get(holiday_names.get(x, ''), 0))

        # Create a countdown column for each holiday, initialized with None
        countdown_columns = {f'days_until_{name}': [] for name in holiday_dates_map}

        # Compute countdown for each holiday
        for date in df.index:
            for name, dates in holiday_dates_map.items():
                # Find the first future date for the specific holiday
                days_until = None
                for holiday_date in dates:
                    if holiday_date >= date:
                        days_until = (holiday_date - date).days
                        break
                if days_until is None:
                    # If no future holiday, reset countdown to 365 or 366 days
                    next_year_start = pd.Timestamp(year=date.year + 1, month=1, day=1)
                    days_until = (next_year_start - date).days
                countdown_columns[f'days_until_{name}'].append(days_until)

        # Create 'days_until_next_holiday' column
        holiday_dates_sorted = sorted(holiday_names.keys())
        days_until_next_holiday = []
        for x in df.index:
            future_dates = [(holiday_date - x).days for holiday_date in holiday_dates_sorted if holiday_date > x]
            if future_dates:
                days_until_next_holiday.append(min(future_dates))
            else:
                # If no future holiday, reset countdown to 365 or 366 days
                next_year_start = pd.Timestamp(year=x.year + 1, month=1, day=1)
                days_until_next_holiday.append((next_year_start - x).days)

        # Create the result DataFrame without the holiday name
        holidays_df = pd.DataFrame({
            'date': df.index,
            'is_holiday': is_holiday,
            'holiday_id': holiday_id,
            'days_until_next_holiday': days_until_next_holiday
        }).set_index('date')

        # Add countdown columns to the DataFrame
        for name, countdown in countdown_columns.items():
            holidays_df[f'days_until_{name}'] = countdown

        return holidays_df

    def compute_seasonal_decompositions(self, df, seasonal_periods=[5, 21, 63, 125, 253]):
        """
        Compute seasonal decompositions for a given DataFrame with multiple columns and seasonal periods.

        Parameters:
        df (pd.DataFrame): The input DataFrame with a datetime index and multiple columns.
        seasonal_periods (list of int): The seasonal periods for decomposition.

        Returns:
        pd.DataFrame: A DataFrame with seasonal decompositions for each specified period and each column.
        """
        decompositions = pd.DataFrame(index=df.index)

        for period in seasonal_periods:
            print(f"Computing decomposition for period: {period}")
            for col in df.columns:
                series = df[col]
                stl = STL(series, seasonal=period)
                result = stl.fit()
                decompositions[f'{col}_trend_{period}'] = result.trend
                decompositions[f'{col}_seasonal_{period}'] = result.seasonal
                decompositions[f'{col}_residual_{period}'] = result.resid
            
        return decompositions
    
    def compute_moving_averages(self, series, windows=[21,50,200] , ma_type='simple'):
        """
        Computes various types of moving averages for a given time series.

        Args:
            series (pd.Series): The input time series with a datetime index.
            windows (list of int): A list of window sizes for which to compute moving averages.
            ma_type (str): Type of moving average to compute ('simple', 'exponential', 'hull', 'tema', or 'kama').

        Returns:
            pd.DataFrame: A DataFrame with moving averages.
            
        Descriptions:
            - Simple Moving Average (SMA): Calculates the average of prices over a specified window size.
            - Exponential Moving Average (EMA): Weighs more recent prices more heavily, making it more responsive to new information.
            - Hull Moving Average (HMA): Aims to reduce lag and improve accuracy by combining weighted moving averages.
            - Triple Exponential Moving Average (TEMA): Reduces lag by combining multiple EMAs, enhancing trend visibility.
            - Kaufman Adaptive Moving Average (KAMA): Adjusts its sensitivity based on market volatility, providing a more adaptive trend-following indicator.
        """
        moving_averages = pd.DataFrame(index=series.index)

        if ma_type == 'simple':
            for window in windows:
                moving_averages[f'ma_{window}'] = series.rolling(window=window).mean()
        
        elif ma_type == 'exponential':
            for window in windows:
                moving_averages[f'ema_{window}'] = series.ewm(span=window, adjust=False).mean()
        
        elif ma_type == 'hull':
            for window in windows:
                half_window = window // 2
                sqrt_window = int(np.sqrt(window))
                wma1 = 2 * series.rolling(window=half_window).mean() - series.rolling(window=window).mean()
                wma2 = series.rolling(window=sqrt_window).mean()
                moving_averages[f'hull_ma_{window}'] = wma2.rolling(window=sqrt_window).mean()
        
        elif ma_type == 'tema':
            for window in windows:
                ema = series.ewm(span=window, adjust=False).mean()
                ema2 = ema.ewm(span=window, adjust=False).mean()
                ema3 = ema2.ewm(span=window, adjust=False).mean()
                tema = 3 * (ema - ema2) + ema3
                moving_averages[f'tema_{window}'] = tema
        
        elif ma_type == 'kama':
            for window in windows:
                change = series.diff(window - 1)
                volatility = series.diff().abs().rolling(window=window).sum()
                er = change / volatility
                sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) ** 2).fillna(0)
                kama = series.copy()
                for i in range(window, len(series)):
                    kama[i] = kama[i - 1] + sc[i] * (series[i] - kama[i - 1])
                moving_averages[f'kama_{window}'] = kama
            
        else:
            raise ValueError("Invalid moving average type. Use 'simple', 'exponential', 'hull', 'tema', or 'kama'.")
        
        moving_averages['original'] = series
        
        return moving_averages

    def compute_optimal_momentum_window(self, close_series, windows):
        """
        Compute rolling Sharpe values for each window and the optimal window by date.

        Parameters:
            close_series (pd.Series): Price series (typically close prices).
            windows (iterable): Window sizes to evaluate.

        Returns:
            pd.DataFrame: Rolling Sharpe columns keyed by window plus `Optimal_Window`.
        """
        if not isinstance(close_series, pd.Series):
            raise TypeError("close_series must be a pandas Series.")

        returns = close_series.pct_change()
        sharpe_df = pd.DataFrame(index=close_series.index)

        for w in windows:
            mean_return = returns.rolling(window=w).mean()
            std_return = returns.rolling(window=w).std()
            sharpe_df[w] = mean_return / (std_return + 1e-8)

        sharpe_df = sharpe_df.dropna(how="all")
        sharpe_df["Optimal_Window"] = sharpe_df.idxmax(axis=1).astype(float)
        return sharpe_df
    
    def compute_volatility(self, df, windows=[21, 50, 200], method='close-to-close'):
        """
        Computes volatility using different methods.

        Args:
            df (pd.DataFrame): DataFrame with columns for price data.
            windows (list of int): A list of window sizes for which to compute volatility.
            method (str): Method to compute volatility ('close-to-close', 'garman-klass', 'parkinson', 'rogers-satchell', 'yang-zhang', 'gk-yz').

        Returns:
            pd.DataFrame: A DataFrame with volatility values for each specified window.
        """
        volatility_df = pd.DataFrame(index=df.index)

        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        if method == 'close-to-close':
            # Calculate volatility as standard deviation of close-to-close returns
            if 'Close' not in df_copy.columns:
                raise ValueError("DataFrame must contain 'Close' column for close-to-close calculation.")
            
            series = df_copy['Close']
            returns = series.pct_change()
            for window in windows:
                volatility_df[f'close_to_close_volatility_{window}'] = returns.rolling(window=window).std()

        elif method == 'garman-klass':
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            # Calculate the components of the Garman-Klass estimator
            df_copy['log_HL'] = np.log(df_copy['High'] / df_copy['Low'])
            df_copy['log_CO'] = np.log(df_copy['Close'] / df_copy['Open'])
            
            term1 = 0.5 * df_copy['log_HL'] ** 2
            term2 = df_copy['log_CO'] ** 2
            
            garman_klass_variance = term1 - term2
            for window in windows:
                rolling_variance = garman_klass_variance.rolling(window=window).mean()
                annualized_volatility = np.sqrt(rolling_variance) * np.sqrt(252)
                volatility_df[f'gk_volatility_{window}'] = annualized_volatility

        elif method == 'parkinson':
            if not all(col in df_copy.columns for col in ['High', 'Low']):
                raise ValueError("DataFrame must contain 'High' and 'Low' columns for Parkinson volatility calculation.")
            
            # Calculate Parkinson volatility
            df_copy['log_HL'] = np.log(df_copy['High'] / df_copy['Low'])
            parkinson_variance = (1 / (4 * np.log(2))) * df_copy['log_HL'] ** 2
            for window in windows:
                rolling_variance = parkinson_variance.rolling(window=window).mean()
                annualized_volatility = np.sqrt(rolling_variance) * np.sqrt(252)
                volatility_df[f'parkinson_volatility_{window}'] = annualized_volatility

        elif method == 'rogers-satchell':
            if not all(col in df_copy.columns for col in ['High', 'Low', 'Close']):
                raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns for Rogers-Satchell volatility calculation.")
            
            # Calculate Rogers-Satchell volatility
            df_copy['log_HL'] = np.log(df_copy['High'] / df_copy['Low'])
            df_copy['log_CO'] = np.log(df_copy['Close'] / df_copy['Open'])
            rs_variance = (df_copy['log_HL'] ** 2 - df_copy['log_CO'] ** 2 + 2 * np.log(2) * (df_copy['log_CO'] ** 2)) / 2
            for window in windows:
                rolling_variance = rs_variance.rolling(window=window).mean()
                annualized_volatility = np.sqrt(rolling_variance) * np.sqrt(252)
                volatility_df[f'rs_volatility_{window}'] = annualized_volatility

        elif method == 'yang-zhang':
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")
            
            # Calculate Yang-Zhang volatility
            df_copy['log_HL'] = np.log(df_copy['High'] / df_copy['Low'])
            df_copy['log_CO'] = np.log(df_copy['Close'] / df_copy['Open'])
            
            term1 = df_copy['log_HL'] ** 2
            term2 = df_copy['log_CO'] ** 2
            term3 = df_copy['Close'].pct_change() ** 2
            
            yang_zhang_variance = (1/2) * (term1 - term2) + term3
            for window in windows:
                rolling_variance = yang_zhang_variance.rolling(window=window).mean()
                annualized_volatility = np.sqrt(rolling_variance) * np.sqrt(252)
                volatility_df[f'yz_volatility_{window}'] = annualized_volatility

        elif method == 'gk-yz':
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df_copy.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {', '.join(required_columns)}")

            # Calculate components for GK-YZ volatility
            df_copy['log_HL'] = np.log(df_copy['High'] / df_copy['Low'])
            df_copy['log_CO'] = np.log(df_copy['Close'] / df_copy['Open'])
            
            # Garman-Klass components
            term1 = 0.5 * df_copy['log_HL'] ** 2
            term2 = df_copy['log_CO'] ** 2
            garman_klass_variance = term1 - term2
            
            # Yang-Zhang components
            yang_zhang_term1 = df_copy['log_HL'] ** 2
            yang_zhang_term2 = df_copy['log_CO'] ** 2
            yang_zhang_term3 = df_copy['Close'].pct_change() ** 2
            yang_zhang_variance = (1/2) * (yang_zhang_term1 - yang_zhang_term2) + yang_zhang_term3

            # Combined GK-YZ variance
            gk_yz_variance = (garman_klass_variance + yang_zhang_variance) / 2
            for window in windows:
                rolling_variance = gk_yz_variance.rolling(window=window).mean()
                annualized_volatility = np.sqrt(rolling_variance) * np.sqrt(252)
                volatility_df[f'gk_yz_volatility_{window}'] = annualized_volatility

        else:
            raise ValueError("Invalid method. Use 'close-to-close', 'garman-klass', 'parkinson', 'rogers-satchell', 'yang-zhang', or 'gk-yz'.")

        return volatility_df

    def compute_rsi(self,series, windows=[21,50,200], indicator_type='RSI'):
        """
        Compute either the Relative Strength Index (RSI) or Rocket RS indicator for a given time series with multiple windows.

        Parameters:
        series (pd.Series): The input time series (e.g., closing prices).
        windows (list of int): The periods for calculation. Default is [14].
        indicator_type (str): Type of indicator to compute. Use 'RSI' or 'Rocket_RS'. Default is 'RSI'.

        Returns:
        pd.DataFrame: A DataFrame with computed indicator values for each specified window.
        """
        if indicator_type not in ['RSI', 'Rocket_RS']:
            raise ValueError("Invalid indicator type. Use 'RSI' or 'Rocket_RS'.")
        
        indicators_df = pd.DataFrame(index=series.index)

        for window in windows:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            loss = loss.replace(0, 1e-10)  # Avoid division by zero
            
            if indicator_type == 'RSI':
                rs = gain / loss
                indicators_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            elif indicator_type == 'Rocket_RS':
                indicators_df[f'Rocket_RS_{window}'] = gain / loss

        return indicators_df
        
    def compute_pairwise(self,df, operations=['differences', 'products', 'sums', 'ratios']):
        """
        Compute pairwise operations for each combination of columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame with the features to compute pairwise operations on.
        operations (list): List of operations to perform. Options include 'differences', 'products', 'sums', and 'ratios'.

        Returns:
        pd.DataFrame: DataFrame containing the results of the pairwise operations.
        """
        # Create an empty DataFrame to store pairwise results
        pairwise_df = pd.DataFrame(index=df.index)

        # Get the columns of the DataFrame
        columns = df.columns
        
        # Perform the specified pairwise operations
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                
                for operation in operations:
                    if operation == 'differences':
                        result = df[col1] - df[col2]
                        result_name = f'({col1} - {col2})'
                    elif operation == 'products':
                        result = df[col1] * df[col2]
                        result_name = f'({col1} * {col2})'
                    elif operation == 'sums':
                        result = df[col1] + df[col2]
                        result_name = f'({col1} + {col2})'
                    elif operation == 'ratios':
                        # Avoid division by zero by replacing 0 with a very small number
                        col2_safe = df[col2].replace(0, 1e-10)
                        result = df[col1] / col2_safe
                        result_name = f'({col1} / {col2})'
                    else:
                        raise ValueError("Invalid operation type. Use 'differences', 'products', 'sums', or 'ratios'.")
                    
                    pairwise_df[result_name] = result

        # Drop columns that are completely NaN (if any)
        pairwise_df = pairwise_df.dropna(axis=1, how='all')
        
        return pairwise_df

    def compute_lags(self, df, lags=range(5, 200), steps=1):
        """
        Computes specified lagged features for a given DataFrame with optional steps between lags.

        Args:
            df (pd.DataFrame): The input DataFrame with a datetime index and multiple columns.
            lags (list of int): A list of lags to generate.
            steps (int): The step size between successive lags (default is 1).

        Returns:
            pd.DataFrame: A DataFrame containing the specified lagged series and the original series.
        """
        # Initialize a DataFrame with the same index as the input DataFrame
        lagged_df = pd.DataFrame(index=df.index)

        # Generate lagged features for each column in the DataFrame
        for col in df.columns:
            series = df[col]
            # Generate lagged features for each lag in the list with specified step size
            for i in range(0, len(lags), steps):
                lag = lags[i]
                lagged_df[f'{col}_lag_{lag}'] = series.shift(lag)

        # Include the original columns in the DataFrame
        lagged_df = pd.concat([lagged_df, df], axis=1)

        return lagged_df
    
    def compute_drawdowns(self,series, windows=[21, 50, 200]):
        """
        Compute drawdowns for a given time series over multiple window sizes.

        Args:
            series (pd.Series): The input time series.
            windows (list of int): A list of window sizes for which to compute drawdowns.

        Returns:
            pd.DataFrame: A DataFrame with drawdown values for each specified window.
        """
        drawdowns_df = pd.DataFrame(index=series.index)

        for window in windows:
            # Compute the rolling maximum
            rolling_max = series.rolling(window=window).max()
            # Compute the drawdown as the difference between the series and the rolling maximum
            drawdown = series - rolling_max
            drawdowns_df[f'drawdown_{window}'] = drawdown

        return drawdowns_df
      
    def compute_non_linear(self,df, transformations=['polynomial'], degrees=range(2, 3), roots=[2, 3], logs=True, exponentials=True):
        """
        Compute non-linear transformations for each column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame with original data.
        - transformations (list of str): Types of transformations to compute ('polynomial', 'exponential', 'root', 'log').
        - degrees (iterable): Range or list of polynomial degrees to compute.
        - roots (list of int): List of root degrees to compute.
        - logs (bool): Whether to compute logarithmic transformations.
        - exponentials (bool): Whether to compute exponential transformations.

        Returns:
        - pd.DataFrame: DataFrame with original columns and their non-linear transformed features.
        """
        transformed_features_list = []
        
        for col in df.columns:
            series = df[col]
            transformed_features = pd.DataFrame(index=series.index)
            
            if 'polynomial' in transformations:
                for degree in degrees:
                    transformed_features[f'({col}^{degree})'] = series ** degree
            
            if 'exponential' in transformations and exponentials:
                transformed_features[f'{col}_exp'] = np.exp(series)
            
            if 'root' in transformations:
                for root in roots:
                    transformed_features[f'({col}^(1/{root}))'] = series ** (1 / root)
            
            if 'log' in transformations and logs:
                # Adding a small constant to avoid taking log of zero or negative values
                transformed_features[f'{col}_log'] = np.log(series + 1e-8)
            
            transformed_features_list.append(transformed_features)
        
        # Concatenate all features horizontally
        all_transformed_features_df = pd.concat(transformed_features_list, axis=1, join='inner')
        
        return all_transformed_features_df

    def compute_moments_ex_mean(self,series, windows=range(1, 11), moment='skew'):
        """
        Compute rolling moments (skewness, kurtosis, or standard deviation) for a given series.

        Parameters:
        series (pd.Series): The input time series.
        windows (iterable): Range or list of window sizes for rolling moments.
        moment (str): Moment to compute. Options are 'skew', 'kurt', or 'std'.

        Returns:
        pd.DataFrame: DataFrame with rolling moments features for the series.
        """
        if moment == 'skew':
            # Compute rolling skewness for each window size
            skewness_list = [series.rolling(window=w).skew() for w in windows]
            moments_df = pd.concat(skewness_list, axis=1)
            moments_df.columns = [f'skew_{w}' for w in windows]
        
        elif moment == 'kurt':
            # Compute rolling kurtosis for each window size
            kurtosis_list = [series.rolling(window=w).kurt() for w in windows]
            moments_df = pd.concat(kurtosis_list, axis=1)
            moments_df.columns = [f'kurt_{w}' for w in windows]

        elif moment == 'std':
            # Compute rolling standard deviation for each window size
            std_dev_list = [series.rolling(window=w).std() for w in windows]
            moments_df = pd.concat(std_dev_list, axis=1)
            moments_df.columns = [f'std_{w}' for w in windows]

        else:
            raise ValueError("Invalid moment type. Choose 'skew', 'kurt', or 'std'.")

        return moments_df
    
    def compute_rolling_sortino_ratios(self,df, n, risk_free_rate=0.0):
        """
        Computes the rolling "n" day Sortino ratios for a DataFrame of stock prices.

        Parameters:
            df (pd.DataFrame): DataFrame containing stock prices with ticker symbols as columns.
            n (int): The window size for the rolling calculation.
            risk_free_rate (float): The risk-free rate for the Sortino ratio calculation (default is 0.0).

        Returns:
            pd.DataFrame: DataFrame containing the rolling "n" day Sortino ratios for each ticker symbol.
        """
        # Calculate daily returns
        returns = df.pct_change()

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / 252

        # Calculate downside deviation
        def downside_deviation(x):
            negative_returns = x[x < 0]
            return np.sqrt((negative_returns ** 2).mean())

        rolling_downside_dev = excess_returns.rolling(window=n).apply(downside_deviation, raw=False)

        # Calculate rolling mean of excess returns
        rolling_mean_excess_returns = excess_returns.rolling(window=n).mean()
        
        # Calculate rolling Sortino ratio
        rolling_sortino_ratio = rolling_mean_excess_returns / rolling_downside_dev
        
        return rolling_sortino_ratio
    
    def compute_rolling_sortino_ratios_benchmark_minus_asset(self,df,benchmark_ticker, n, risk_free_rate=0.0):
        """
        Computes the rolling "n" day Sortino ratios for a DataFrame of stock prices.

        Parameters:
            df (pd.DataFrame): DataFrame containing stock prices with ticker symbols as columns.
            n (int): The window size for the rolling calculation.
            risk_free_rate (float): The risk-free rate for the Sortino ratio calculation (default is 0.0).

        Returns:
            pd.DataFrame: DataFrame containing the rolling "n" day Sortino ratios for each ticker symbol.
        """
        # Calculate daily returns
        returns = df.pct_change()

        # Calculate excess returns
        excess_returns = returns - risk_free_rate / 252

        # Calculate downside deviation
        def downside_deviation(x):
            negative_returns = x[x < 0]
            return np.sqrt((negative_returns ** 2).mean())

        rolling_downside_dev = excess_returns.rolling(window=n).apply(downside_deviation, raw=False)

        # Calculate rolling mean of excess returns
        rolling_mean_excess_returns = excess_returns.rolling(window=n).mean()
        
        # Calculate rolling Sortino ratio
        rolling_sortino_ratio = rolling_mean_excess_returns / rolling_downside_dev
        
        benchmark_sortino = rolling_sortino_ratio[benchmark_ticker]
        rolling_sortino_ratio = rolling_sortino_ratio.sub(benchmark_sortino, axis=0)
        benchmark_minus_asset = -rolling_sortino_ratio
        rolling_sortino_ratio = benchmark_minus_asset
        #rolling_sortino_ratio.columns = ['Benchmark_Minus_' + col for col in rolling_sortino_ratio.columns]

        return rolling_sortino_ratio
    
    def z_score(self, time_series_data):
        """
        Calculates the z-scores for the latest row of a time series DataFrame.

        Parameters:
            time_series_data (pd.DataFrame): DataFrame of time series data, with columns as different series.

        Returns:
            pd.Series: A Series of z-scores for the latest values.
        """
        mean = time_series_data.mean()
        std = time_series_data.std()
        latest_values = time_series_data.iloc[-1]
        z_scores = (latest_values - mean) / std
        return z_scores
    
    def categorize_z_score(self,z):
        """
        Categorizes a z-score into integer buckets.

        Integer categories (based on z-score):
        3  => z > 3
        2  => 2 < z <= 3
        1  => 1 < z <= 2
        0  => -1 < z <= 1
        -1  => -2 < z <= -1
        -2  => -3 < z <= -2
        -3  => z <= -3

        Parameters:
            z (float): The z-score to categorize.

        Returns:
            int: The integer category.
        """
        if z > 3:
            return 3
        elif z > 2:
            return 2
        elif z > 1:
            return 1
        elif z > -1:
            return 0
        elif z > -2:
            return -1
        elif z > -3:
            return -2
        else:
            return -3
        
    def risk_adjusted_returns(
        self,
        data,
        windows,
        ratio_type="sharpe",
        threshold=0.0,
        annualization_factor=252,
    ):
        if not isinstance(windows, list):
            raise ValueError("windows must be a list of integers")

        single_window = len(windows) == 1

        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name or "price")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.Series or pd.DataFrame")

        returns = data.pct_change()#.dropna()
        output = []

        for col in returns.columns:
            r = returns[col]

            for window in windows:
                mean_ret = r.rolling(window).mean()

                if ratio_type == "sharpe":
                    vol = r.rolling(window).std()
                    ratio = (mean_ret / vol) * np.sqrt(annualization_factor)

                elif ratio_type == "sortino":
                    downside_deviation = (
                        r.where(r < 0, 0)
                        .rolling(window)
                        .apply(lambda x: np.sqrt((x ** 2).mean()), raw=True)
                    )
                    ratio = mean_ret / downside_deviation
                elif ratio_type == "omega":
                    def omega(x):
                        excess = x - threshold
                        gains = excess[excess > 0].sum()
                        losses = -excess[excess < 0].sum()
                        return np.nan if losses == 0 else gains / losses

                    ratio = r.rolling(window).apply(omega, raw=False)

                elif ratio_type == "calmar":
                    def calmar(x):
                        growth = (1 + x).cumprod()
                        peak = growth.cummax()
                        dd = growth / peak - 1
                        max_dd = dd.min()
                        if max_dd >= 0:
                            return np.nan
                        total_ret = growth.iloc[-1] - 1
                        ann_ret = (1 + total_ret) ** (annualization_factor / len(x)) - 1
                        return ann_ret / abs(max_dd)

                    ratio = r.rolling(window).apply(calmar, raw=False)

                elif ratio_type == "sterling":
                    def sterling(x):
                        growth = (1 + x).cumprod()
                        peak = growth.cummax()
                        dd = growth / peak - 1
                        worst = dd[dd < 0].nsmallest(3)
                        if worst.empty:
                            return np.nan
                        avg_dd = abs(worst).mean()
                        total_ret = growth.iloc[-1] - 1
                        ann_ret = (1 + total_ret) ** (annualization_factor / len(x)) - 1
                        return ann_ret / avg_dd

                    ratio = r.rolling(window).apply(sterling, raw=False)

                else:
                    raise ValueError("Invalid ratio_type")

                ratio = ratio.replace([np.inf, -np.inf], np.nan)

                # 🔹 Column naming logic
                if single_window:
                    ratio.name = f"{ratio_type}_ratio_{window}"
                else:
                    ratio.name = f"{col}_{ratio_type}_{window}"

                output.append(ratio)

        return pd.concat(output, axis=1)

    def calculate_percentage_drop(self, ticker, n=14):
        """
        Calculate the percentage drop from the highest peak in a rolling window.

        Parameters:
        - ticker: pd.DataFrame with 'Close' prices indexed by date.
        - n: Number of days for the rolling window.

        Returns:
        - pd.DataFrame with 'Close', 'HighestHigh', and 'PercentageDrop' columns.
        """
        # Ensure ticker is a DataFrame and has a 'Close' column
        if 'Close' not in ticker.columns:
            raise ValueError("The DataFrame must contain a 'Close' column.")

        # Work on a copy to avoid modifying the original DataFrame
        ticker_copy = ticker.copy()

        # Calculate the highest peak in the last n days
        ticker_copy['HighestHigh'] = ticker_copy['Close'].rolling(window=n, min_periods=1).max()

        # Calculate the percentage drop from the highest peak
        ticker_copy['PercentageDrop'] = -((ticker_copy['HighestHigh'] - ticker_copy['Close']) / ticker_copy['HighestHigh']) * 100

        return ticker_copy
    
    def align_features_to_index(self,features_dict, reference_index):
        aligned_features = {}
        for key, df in features_dict.items():
            aligned_features[key] = df.reindex(reference_index).fillna(method='ffill')
        return aligned_features

    def n_positive_days(self,ticker="SPY", number_of_days=21 ):
        data = yf.Ticker(ticker).history(period="max", interval="1d")
        data = data[['Close', 'Open']]
        data['closed_higher'] = data['Close'] >= data['Open']
        closed_higher = data['closed_higher']
        mask = closed_higher.mask(closed_higher == True, 1)
        mask[mask == False] = 0
        return mask.rolling(number_of_days).sum() / number_of_days

    def spread(self,asset_series, benchmark_series, time_frame, mode='standard'):
        if mode == 'standard':
            asset_returns = asset_series.pct_change(time_frame)
            benchmark_returns= benchmark_series.pct_change(time_frame)
        elif mode == 'sortino':
            asset_returns = self.calculate_risk_adjusted_returns(asset_series, time_frame)
            benchmark_returns= self.calculate_risk_adjusted_returns(benchmark_series, time_frame)

        benchmark_minus_asset = asset_returns.apply(lambda x: benchmark_returns - x)
        benchmark_minus_asset.columns = ["Benchmark" + "_minus_" + col for col in benchmark_minus_asset.columns]
        return benchmark_minus_asset    

    def calculate_differences(self,df):
        diff_ = pd.DataFrame()
        for column in df.columns:
            for column2 in df.columns:
                diff_[f'{column} min {column2}'] = df[column] - df[column2]
        return diff_

    def returns(self, data, frequency='monthly'):
        """
        Calculate returns for the specified frequency.
    
        Parameters:
        - data: pd.DataFrame with DateTimeIndex and 'Close' prices.
        - frequency: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly' to specify the frequency of the returns.
    
        Returns:
        - pd.Series with the calculated returns.
        """
        # Ensure DateTime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Determine the resampling frequency
        if frequency == 'daily':
            resample_freq = 'D'
        elif frequency == 'weekly':
            resample_freq = 'W'
        elif frequency == 'monthly':
            resample_freq = 'M'
        elif frequency == 'quarterly':
            resample_freq = 'Q'
        elif frequency == 'yearly':
            resample_freq = 'A'
        else:
            raise ValueError("Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.")
        
        # Resample to the specified frequency, taking the last observation of each period
        resampled_data = data.resample(resample_freq).last()
        
        # Calculate returns
        returns = resampled_data['Close'].pct_change().dropna()
        
        return returns 
        
    def load_and_prepare_data(self, tickers='SPY', period='5y', gen_returns=False, gen_log_returns=False, gen_cumulative_returns=False, train_percentage=0.8):
        """
        Retrieve, prepare, and split stock data from Yahoo Finance (using yf.Ticker) 
        and return it as a dictionary with nested DataFrames.

        Parameters:
        - tickers: str or list of str, stock ticker symbols.
        - period: str, the period for which to retrieve data (e.g., '5y').
        - gen_returns: bool, whether to calculate percent returns along with raw prices.
        - gen_log_returns: bool, whether to calculate log returns along with raw prices.
        - gen_cumulative_returns: bool, whether to calculate cumulative returns.
        - train_percentage: float, the proportion of data to use for training.

        Returns:
        - data_dict: dict, a dictionary where each key is a ticker and each value is 
                    another dictionary containing 'full', 'train', and 'test' DataFrames.
        """
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Initialize dictionary to hold DataFrames for each ticker
        data_dict = {}

        for ticker in tickers:
            # Retrieve data using yf.Ticker
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period)

            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'Date'

            # Fill missing values with forward fill
            df = df.asfreq('B')  # Set frequency to business day
            df = df.fillna(method='ffill')  # Forward fill missing values

            # Ensure a continuous index
            df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')).fillna(method='ffill')

            # Store raw data for calculations
            raw_data_df = df.copy()

            # Calculate percentage returns if requested
            if gen_returns:
                pct_returns_df = raw_data_df.pct_change()
                pct_returns_df.columns = [col + '_Returns' for col in pct_returns_df.columns]
                df = pd.concat([df, pct_returns_df], axis=1)

            # Calculate log returns if requested
            if gen_log_returns:
                log_returns_df = np.log1p(raw_data_df.pct_change())
                log_returns_df.columns = [col + '_Log_Returns' for col in log_returns_df.columns]
                df = pd.concat([df, log_returns_df], axis=1)

            # Calculate cumulative returns if requested
            if gen_cumulative_returns:
                pct_returns_raw_df = raw_data_df.pct_change()
                cumulative_returns_raw_df = (1 + pct_returns_raw_df).cumprod() - 1
                cumulative_returns_raw_df.columns = [col + '_Cumulative_Returns' for col in cumulative_returns_raw_df.columns]
                df = pd.concat([df, cumulative_returns_raw_df], axis=1)

            # Split data into training and test sets
            split_index = int(len(df) * train_percentage)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()

            # Store data in dictionary
            data_dict[ticker] = {
                'full': df,
                'train': train_df,
                'test': test_df
            }
        
        return data_dict
    
    def vix_fix(self, data, window=22):
        """
        Computes the VIX Fix (Larry Williams) for a given price series.

        Parameters:
        - series (pd.Series): The input time series of prices.
        - window (int): The number of periods to look back for the highest high.

        Returns:
        - pd.Series: VIX Fix values.
        """
        highest_close = data['Close'].rolling(window=window).max()
        vix_fix_values = 100 * (highest_close - data['Close']) / highest_close
        return vix_fix_values

    def create_sortino_negative_indicators(self, sortino_diff_50, sortino_diff_200):
        """
        Creates a DataFrame indicating whether the latest Sortino ratio difference for each ticker is less than zero
        for 21-day, 50-day, and 200-day rolling windows.

        Parameters:
            sortino_diff_21 (pd.DataFrame): DataFrame of 21-day Sortino differences.
            sortino_diff_50 (pd.DataFrame): DataFrame of 50-day Sortino differences.
            sortino_diff_200 (pd.DataFrame): DataFrame of 200-day Sortino differences.

        Returns:
            pd.DataFrame: DataFrame with tickers as rows and columns ['21_Day', '50_Day', '200_Day'],
                        with True indicating the latest Sortino difference is <0, and False otherwise.
        """
        # Extract the latest row from each Sortino difference DataFrame
        latest_50 = sortino_diff_50.iloc[-1]
        latest_200 = sortino_diff_200.iloc[-1]

        # Create a new DataFrame with indicators
        indicators_df = pd.DataFrame({
            '50_Day': latest_50 > 0,
            '200_Day': latest_200 > 0
        })

        # Reset index to have tickers as a column
        indicators_df = indicators_df.reset_index()
        indicators_df.columns = ['Ticker',  'Relative performance: 50 Day Sortino (Benchmark - asset)', 'Relative performance: 200 Day Sortino (Benchmark - asset)']

        return indicators_df

    def create_sortino_std_deviation_table(self, rolling_sortino_ratio):
        """
        Creates a DataFrame indicating how many standard deviations the latest Sortino ratio 
        for each ticker is above or below its historical mean, using integer buckets.

        Parameters:
            rolling_sortino_ratio (pd.DataFrame): DataFrame containing the rolling 
                                                    Sortino ratios for each ticker.

        Returns:
            pd.DataFrame: DataFrame with 'Ticker' and 'Std Dev Direction' columns.
        """
        # Obtain the z-scores using the separated function
        z_scores = qc.calculate_z_scores(rolling_sortino_ratio)
        # Categorize each z-score using the separate categorize_z_score function
        categories = z_scores.apply(qc.categorize_z_score)

        # Create and return the deviation table DataFrame
        deviation_table = pd.DataFrame({
            'Ticker': rolling_sortino_ratio.columns.tolist(),
            'Std Dev Direction': categories.tolist()
        })

        return deviation_table
    
    def create_price_std_deviation_table(self, price_data, window_sizes=[21, 50, 200]):
        """
        Creates a DataFrame indicating how many standard deviations the latest price is
        above or below its rolling mean over specified window sizes, using integer buckets.

        Integer categories (based on z-score):
        3  => z > 3
        2  => 2 < z <= 3
        1  => 1 < z <= 2
        0  => -1 < z <= 1
        -1  => -2 < z <= -1
        -2  => -3 < z <= -2
        -3  => z <= -3
        """
        # Initialize dictionary to store deviation categories
        deviation_data = {'Ticker': price_data.columns.tolist()}

        for window in window_sizes:
            categories = []
            for ticker in price_data.columns:
                ticker_prices = price_data[ticker].dropna()
                if len(ticker_prices) >= window:
                    rolling_mean = ticker_prices.rolling(window=window).mean()
                    rolling_std = ticker_prices.rolling(window=window).std()

                    latest_price = ticker_prices.iloc[-1]
                    latest_mean = rolling_mean.iloc[-1]
                    latest_std = rolling_std.iloc[-1]

                    if latest_std == 0 or pd.isna(latest_std):
                        category = 'Insufficient Data'
                    else:
                        z_score = (latest_price - latest_mean) / latest_std

                        def categorize_z(z):
                            if z > 3:
                                return 3
                            elif z > 2:
                                return 2
                            elif z > 1:
                                return 1
                            elif z > -1:
                                return 0
                            elif z > -2:
                                return -1
                            elif z > -3:
                                return -2
                            else:
                                return -3

                        category = categorize_z(z_score)
                else:
                    category = 'Insufficient Data'
                categories.append(category)
            deviation_data[f'Std Dev Direction for {window}_Day Price'] = categories

        deviation_table = pd.DataFrame(deviation_data)
        return deviation_table
    
    def filter_assets_by_positive_spread_std(self,asset_spreads):
        spreads = asset_spreads
        positive_spreads = spreads[spreads >= 0] 
        
        mean = positive_spreads.mean()
        std_dev = positive_spreads.std()

        latest_spread = spreads.iloc[-1]
        threshold = mean + std_dev

        return latest_spread>=threshold

    def filter_assets_below_negative_std(self,asset_spreads):
        if not isinstance(asset_spreads, pd.Series):
            raise TypeError("asset_spreads must be a pandas Series")

        negative_spreads = asset_spreads[asset_spreads < 0]
        if negative_spreads.empty:
            return pd.Series(dtype=bool)  
        
        mean_negative = negative_spreads.mean()
        std_dev_negative = negative_spreads.std()

        threshold_negative = mean_negative - 0.75 * std_dev_negative
        return asset_spreads < threshold_negative
    
    def pairwise_spreads(self, etf_dataframes, window=20):
        """
        Creates pairwise spreads of rolling returns between assets within each category.
        
        Parameters:
            etf_dataframes (dict): Dictionary of DataFrames where each key is a category 
                                and values are DataFrames with ticker columns
            window (int): Rolling window period for calculating returns (default=20)
        
        Returns:
            dict: Dictionary where keys match the input categories and values are DataFrames
                containing the spreads between rolling returns of unique asset pairs
        """
        pairwise_spreads = {}

        for category, df in etf_dataframes.items():
            # Skip categories with only one asset
            if df.shape[1] <= 1:
                continue
                
            # Get valid tickers in this category (those without all NaN values)
            valid_tickers = [ticker for ticker in df.columns if not df[ticker].isna().all()]
            
            if len(valid_tickers) < 2:
                continue
            
            # Create an empty DataFrame for this category's spreads
            category_spreads = pd.DataFrame(index=df.index)
            
            # For each unique pair of tickers, compute the spread of rolling returns
            for i in range(len(valid_tickers)):
                for j in range(i+1, len(valid_tickers)):  # Start from i+1 to avoid duplicates
                    ticker1, ticker2 = valid_tickers[i], valid_tickers[j]
                    
                    # Find valid data for both assets
                    valid_data = df[[ticker1, ticker2]].dropna()
                    if valid_data.empty:
                        continue
                    
                    # Calculate rolling returns for both assets
                    returns1 = df[ticker1].pct_change(window)
                    returns2 = df[ticker2].pct_change(window)
                    
                    # Calculate and store the spread between rolling returns
                    spread_name = f"{ticker1}-{ticker2}"
                    category_spreads[spread_name] = returns1 - returns2
            
            # Store only if we have valid spreads
            if not category_spreads.empty:
                pairwise_spreads[category] = category_spreads
        
        return pairwise_spreads
    
    def get_sorted_correlations(self, corr_matrix):
        # Get the lower triangular part of the correlation matrix (excluding diagonal)
        mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
        corr_pairs = []
        
        # Extract all pairwise correlations
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                if mask[i, j]:  # Only take lower triangle to avoid duplicates
                    corr_pairs.append((f"{row}-{col}", corr_matrix.iloc[i, j]))
        
        # Sort by correlation value
        corr_pairs.sort(key=lambda x: x[1])
        
        # Return pair names and correlation values
        return zip(*corr_pairs)
    
    def get_cointegration_pvals(self,df, correlation_pairs):
        pairs = []
        p_values = []
        
        for pair_name in correlation_pairs:
            # Split the ticker pair
            ticker1, ticker2 = pair_name.split('-')
            
            # Skip pairs with insufficient data
            series1 = df[ticker1].dropna()
            series2 = df[ticker2].dropna()
            
            common_idx = series1.index.intersection(series2.index)
            if len(common_idx) < 10:  # Reduced minimum data points needed for monthly data
                p_values.append(1.0)  # Use 1.0 as default p-value when test can't be run
                continue
                
            s1 = series1.loc[common_idx]
            s2 = series2.loc[common_idx]
            
            try:
                # Regression to get residuals
                X = np.array(s2).reshape(-1, 1)
                X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
                y = np.array(s1).reshape(-1, 1)
                
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X.dot(beta)
                
                # ADF test on residuals
                adf_result = adfuller(residuals.flatten(), autolag='AIC')
                p_value = adf_result[1]
                
                p_values.append(p_value)
            except:
                p_values.append(1.0)  # Use 1.0 as default p-value when test fails
        
        return correlation_pairs, p_values

    # Delegated non-rolling components (latest definitions intentionally override earlier legacy bodies)
    def compute_holiday_features(self, df, country='US'):
        return self.time_features.compute_holiday_features(df, country=country)

    def compute_seasonal_decompositions(self, df, seasonal_periods=[5, 21, 63, 125, 253]):
        return self.time_features.compute_seasonal_decompositions(df, seasonal_periods=seasonal_periods)

    def compute_pairwise(self, df, operations=['differences', 'products', 'sums', 'ratios']):
        return self.feature_engineering.compute_pairwise(df, operations=operations)

    def compute_lags(self, df, lags=range(5, 200), steps=1):
        return self.feature_engineering.compute_lags(df, lags=lags, steps=steps)

    def compute_non_linear(
        self,
        df,
        transformations=['polynomial'],
        degrees=range(2, 3),
        roots=[2, 3],
        logs=True,
        exponentials=True,
    ):
        return self.feature_engineering.compute_non_linear(
            df,
            transformations=transformations,
            degrees=degrees,
            roots=roots,
            logs=logs,
            exponentials=exponentials,
        )

    def calculate_differences(self, df):
        return self.feature_engineering.calculate_differences(df)

    def compute_average_return(self, close_series, window, percent=True):
        return self.momentum_analytics.compute_average_return(close_series, window=window, percent=percent)

    def compute_momentum_diff(self, close_series, short_window, long_window):
        return self.momentum_analytics.compute_momentum_diff(
            close_series,
            short_window=short_window,
            long_window=long_window,
        )

    def compute_momentum_zscore(
        self,
        close_series,
        short_window,
        long_window,
        normalizer_window=None,
        ddof=0,
    ):
        return self.momentum_analytics.compute_momentum_zscore(
            close_series=close_series,
            short_window=short_window,
            long_window=long_window,
            normalizer_window=normalizer_window,
            ddof=ddof,
        )

    def compute_momentum_zscore_map(self, close_series, window_pairs, normalizer_window=None, ddof=0):
        return self.momentum_analytics.compute_momentum_zscore_map(
            close_series=close_series,
            window_pairs=window_pairs,
            normalizer_window=normalizer_window,
            ddof=ddof,
        )

    def compute_asset_ratio_maps(self, asset_close, time_frame_map):
        return self.risk_relative_analytics.compute_asset_ratio_maps(
            analytics=self,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
        )

    def compute_ratio_spread_map(self, left_map, right_map):
        return self.risk_relative_analytics.compute_ratio_spread_map(left_map, right_map)

    def compute_benchmark_metrics(self, benchmark_data, asset_close, asset_sharpe_map, time_frame_map):
        return self.risk_relative_analytics.compute_benchmark_metrics(
            analytics=self,
            benchmark_data=benchmark_data,
            asset_close=asset_close,
            asset_sharpe_map=asset_sharpe_map,
            time_frame_map=time_frame_map,
        )

    def build_sharpe_sortino_term_config_map(self, asset_sharpe_map, asset_sortino_map, time_frame_map):
        return self.risk_relative_analytics.build_term_config_map(
            asset_sharpe_map=asset_sharpe_map,
            asset_sortino_map=asset_sortino_map,
            time_frame_map=time_frame_map,
        )

    def build_sharpe_sortino_context(self, asset_close, time_frame_map, benchmark_data=None):
        return self.risk_relative_analytics.build_sharpe_sortino_context(
            analytics=self,
            asset_close=asset_close,
            time_frame_map=time_frame_map,
            benchmark_data=benchmark_data,
        )

    def build_benchmark_plot_payload(self, asset_sharpe_map, benchmark_metrics, spread_plot_data, time_frame_map):
        return self.risk_relative_analytics.build_benchmark_plot_payload(
            asset_sharpe_map=asset_sharpe_map,
            benchmark_metrics=benchmark_metrics,
            spread_plot_data=spread_plot_data,
            time_frame_map=time_frame_map,
        )

    def build_risk_distribution_context(self, close_series, windows, default_window=None):
        return self.risk_distribution_analytics.build_risk_distribution_context(
            close_series=close_series,
            windows=windows,
            default_window=default_window,
        )

    def build_momentum_window_diagnostics_context(
        self,
        close_series,
        window_sizes,
        highlight_windows=(7, 21, 50, 200),
        surface_years=10,
    ):
        return self.momentum_analytics.build_momentum_window_diagnostics_context(
            analytics=self,
            close_series=close_series,
            window_sizes=window_sizes,
            highlight_windows=highlight_windows,
            surface_years=surface_years,
        )

    def z_score(self, time_series_data):
        return self.signal_labels.z_score(time_series_data)

    def categorize_z_score(self, z):
        return self.signal_labels.categorize_z_score(z)

    def create_sortino_negative_indicators(self, sortino_diff_50, sortino_diff_200):
        return self.signal_labels.create_sortino_negative_indicators(sortino_diff_50, sortino_diff_200)

    def create_sortino_std_deviation_table(self, rolling_sortino_ratio):
        return self.signal_labels.create_sortino_std_deviation_table(rolling_sortino_ratio)

    def create_price_std_deviation_table(self, price_data, window_sizes=[21, 50, 200]):
        return self.signal_labels.create_price_std_deviation_table(price_data, window_sizes=window_sizes)

    def filter_assets_by_positive_spread_std(self, asset_spreads):
        return self.signal_labels.filter_assets_by_positive_spread_std(asset_spreads)

    def filter_assets_below_negative_std(self, asset_spreads):
        return self.signal_labels.filter_assets_below_negative_std(asset_spreads)

    def pairwise_spreads(self, etf_dataframes, window=20):
        return self.cross_section_stats.pairwise_spreads(etf_dataframes, window=window)

    def get_sorted_correlations(self, corr_matrix):
        return self.cross_section_stats.get_sorted_correlations(corr_matrix)

    def get_cointegration_pvals(self, df, correlation_pairs):
        return self.cross_section_stats.get_cointegration_pvals(df, correlation_pairs)


# Backward-compatible alias for existing notebook and package imports.
Rolling = TimeSeriesAnalytics

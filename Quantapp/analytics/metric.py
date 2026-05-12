import numpy as np
from statsmodels.tsa.stattools import coint


ANNUALIZATION_FACTOR = np.sqrt(252)


class Metric:
    
    def __init__(self):
        self.a = 1
        self.calculate_returns = False

    def percent_change(self, arr, window=1):
        returns = arr.pct_change(window).dropna()
        if returns.empty:
            return np.nan
        return returns.iloc[-1]
    
    def percent_return(self, arr):
        return (arr.iloc[-1] - arr.iloc[0]) / np.abs(arr.iloc[0])
    
    def z_score(self, arr):
        return (arr.iloc[-1] - arr.mean()) / arr.std()

    def vix_fix(self, arr):
        close = arr["Close"] if hasattr(arr, "columns") else arr
        close = close.dropna()
        if close.empty:
            return np.nan

        highest_close = close.max()
        if highest_close == 0:
            return np.nan
        return 100 * (highest_close - close.iloc[-1]) / highest_close
    
    def log_returns(self, arr):
        returns = np.log(arr / arr.shift()).dropna()
        if returns.empty:
            return np.nan
        return returns.iloc[-1]
    
    def average(self, arr):
        return arr.mean()
    
    def ewma_average(self,arr,alpha=None):
        arr = arr.dropna()
        if arr.empty:
            return np.nan
        if alpha is None:
            alpha = 2.0 / (len(arr) + 1.0)
        ewma = arr.ewm(alpha=alpha, adjust=False, min_periods=len(arr)).mean()
        return ewma.dropna().iloc[-1]

    def exp_average(self,arr):
        arr = arr.dropna()
        if arr.empty:
            return np.nan
        alpha = 2.0 / (len(arr) + 1.0)
        ewma = arr.ewm(alpha=alpha, adjust=False, min_periods=len(arr)).mean()
        return ewma.dropna().iloc[-1]

    def ewma_variance(self,arr,alpha=None):
        arr = arr.dropna()
        if len(arr) < 2:
            return np.nan
        if alpha is None:
            alpha = 2.0 / (len(arr) + 1.0)
        ewma = arr.ewm(alpha=alpha, adjust=False, min_periods=len(arr)).var(bias=False)
        return ewma.dropna().iloc[-1]
    
    def median(self,arr):
        return arr.median()
    
    def mode(self,arr):
        modes = arr.mode()
        if modes.empty:
            return np.nan
        return modes.iloc[0]
    
    def skew(self,arr):
        return arr.skew()
    
    def kurtosis(self,arr):
        return arr.kurtosis()

    def close_to_close_variance(self,arr):
        if hasattr(arr, "columns"):
            if "Close" not in arr.columns:
                raise ValueError("arr must contain a 'Close' column.")
            close = arr["Close"]
        else:
            close = arr
        returns = close.pct_change().dropna()
        return returns.var()

    def garman_klass_variance(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_hl = np.log(frame["High"] / frame["Low"])
        log_co = np.log(frame["Close"] / frame["Open"])
        variance = (
            0.5 * (log_hl ** 2)
            - ((2 * np.log(2)) - 1) * (log_co ** 2)
        ).dropna()
        if variance.empty:
            return np.nan
        return max(float(variance.mean()), 0.0)

    def yang_zhang_variance(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        if len(frame) < 2:
            return np.nan

        log_ho = np.log(frame["High"] / frame["Open"])
        log_lo = np.log(frame["Low"] / frame["Open"])
        log_co = np.log(frame["Close"] / frame["Open"])
        log_oc = np.log(frame["Open"] / frame["Close"].shift(1))
        log_hc = np.log(frame["High"] / frame["Close"])
        log_lc = np.log(frame["Low"] / frame["Close"])
        rs_variance = (log_hc * log_ho) + (log_lc * log_lo)
        k = 0.34 / (1.34 + ((len(frame) + 1) / (len(frame) - 1)))
        yz_variance = (
            log_oc.dropna().var()
            + (k * log_co.dropna().var())
            + ((1 - k) * rs_variance.dropna().mean())
        )
        return max(float(yz_variance), 0.0) if not np.isnan(yz_variance) else np.nan

    def rogers_satchell_variance(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_ho = np.log(frame["High"] / frame["Open"])
        log_lo = np.log(frame["Low"] / frame["Open"])
        log_hc = np.log(frame["High"] / frame["Close"])
        log_lc = np.log(frame["Low"] / frame["Close"])
        variance = ((log_hc * log_ho) + (log_lc * log_lo)).dropna()
        if variance.empty:
            return np.nan
        return max(float(variance.mean()), 0.0)

    def parkinson_variance(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_hl = np.log(frame["High"] / frame["Low"])
        variance = ((log_hl ** 2) / (4 * np.log(2))).dropna()
        if variance.empty:
            return np.nan
        return max(float(variance.mean()), 0.0)

    def ewma_realized_variance(self,arr,method="close-to-close"):
        method = method.lower()
        observation_count = len(arr)
        if observation_count == 0:
            return np.nan
        alpha = 2.0 / (observation_count + 1.0)

        if method == "close-to-close":
            if hasattr(arr, "columns"):
                if "Close" not in arr.columns:
                    raise ValueError("arr must contain a 'Close' column.")
                close = arr["Close"]
            else:
                close = arr
            returns = close.pct_change().dropna()
            if returns.empty:
                return np.nan
            ewma_variance = returns.pow(2).ewm(
                alpha=alpha,
                adjust=False,
                min_periods=len(returns),
            ).mean()
            return max(float(ewma_variance.dropna().iloc[-1]), 0.0)

        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr

        if method == "parkinson":
            log_hl = np.log(frame["High"] / frame["Low"])
            variance = (log_hl ** 2) / (4 * np.log(2))
        elif method == "garman-klass":
            log_hl = np.log(frame["High"] / frame["Low"])
            log_co = np.log(frame["Close"] / frame["Open"])
            variance = 0.5 * (log_hl ** 2) - ((2 * np.log(2)) - 1) * (log_co ** 2)
        elif method == "rogers-satchell":
            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            variance = (log_hc * log_ho) + (log_lc * log_lo)
        elif method == "yang-zhang":
            if observation_count < 2:
                return np.nan

            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_co = np.log(frame["Close"] / frame["Open"])
            log_oc = np.log(frame["Open"] / frame["Close"].shift(1))
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            rs_variance = (log_hc * log_ho) + (log_lc * log_lo)
            k = 0.34 / (1.34 + ((observation_count + 1) / (observation_count - 1)))
            log_oc = log_oc.dropna()
            log_co = log_co.dropna()
            rs_variance = rs_variance.dropna()
            if len(log_oc) < 2 or len(log_co) < 2 or rs_variance.empty:
                return np.nan
            overnight_variance = log_oc.ewm(alpha=alpha, adjust=False, min_periods=len(log_oc)).var(bias=False)
            open_to_close_variance = log_co.ewm(alpha=alpha, adjust=False, min_periods=len(log_co)).var(bias=False)
            rs_component = rs_variance.ewm(alpha=alpha, adjust=False, min_periods=len(rs_variance)).mean()
            yz_variance = (
                overnight_variance.dropna().iloc[-1]
                + (k * open_to_close_variance.dropna().iloc[-1])
                + ((1 - k) * rs_component.dropna().iloc[-1])
            )
            return max(float(yz_variance), 0.0) if not np.isnan(yz_variance) else np.nan
        else:
            raise ValueError(f"Unsupported EWMA volatility method: {method}")

        variance = variance.dropna()
        if variance.empty:
            return np.nan
        ewma_variance = variance.ewm(
            alpha=alpha,
            adjust=False,
            min_periods=len(variance),
        ).mean()
        return max(float(ewma_variance.dropna().iloc[-1]), 0.0)

    def close_to_close_volatility(self,arr):
        if hasattr(arr, "columns"):
            if "Close" not in arr.columns:
                raise ValueError("arr must contain a 'Close' column.")
            close = arr["Close"]
        else:
            close = arr
        returns = close.pct_change().dropna()
        return returns.std() * ANNUALIZATION_FACTOR
    
    def garman_klass_volatility(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_hl = np.log(frame["High"] / frame["Low"])
        log_co = np.log(frame["Close"] / frame["Open"])
        variance = (
            0.5 * (log_hl ** 2)
            - ((2 * np.log(2)) - 1) * (log_co ** 2)
        ).dropna()
        if variance.empty:
            return np.nan
        return np.sqrt(max(float(variance.mean()), 0.0)) * ANNUALIZATION_FACTOR

    def yang_zhang_volatility(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        if len(frame) < 2:
            return np.nan

        log_ho = np.log(frame["High"] / frame["Open"])
        log_lo = np.log(frame["Low"] / frame["Open"])
        log_co = np.log(frame["Close"] / frame["Open"])
        log_oc = np.log(frame["Open"] / frame["Close"].shift(1))
        log_hc = np.log(frame["High"] / frame["Close"])
        log_lc = np.log(frame["Low"] / frame["Close"])
        rs_variance = (log_hc * log_ho) + (log_lc * log_lo)
        k = 0.34 / (1.34 + ((len(frame) + 1) / (len(frame) - 1)))
        yz_variance = (
            log_oc.dropna().var()
            + (k * log_co.dropna().var())
            + ((1 - k) * rs_variance.dropna().mean())
        )
        if np.isnan(yz_variance):
            return np.nan
        return np.sqrt(max(float(yz_variance), 0.0)) * ANNUALIZATION_FACTOR
    
    def hodges_tompkins_volatility(self,arr):
        if hasattr(arr, "columns"):
            if "Close" not in arr.columns:
                raise ValueError("arr must contain a 'Close' column.")
            close = arr["Close"]
        else:
            close = arr
        returns = np.log(close / close.shift()).dropna()
        volatility = returns.std() * ANNUALIZATION_FACTOR
        if returns.empty:
            return volatility

        window = len(returns)
        subseries_count = len(returns)
        if subseries_count <= 0:
            return volatility

        denominator = (
            1
            - (window / subseries_count)
            + ((window ** 2 - 1) / (3 * (subseries_count ** 2)))
        )
        if denominator <= 0:
            return volatility
        return volatility * np.sqrt(1 / denominator)
    
    def rogers_satchell_volatility(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_ho = np.log(frame["High"] / frame["Open"])
        log_lo = np.log(frame["Low"] / frame["Open"])
        log_hc = np.log(frame["High"] / frame["Close"])
        log_lc = np.log(frame["Low"] / frame["Close"])
        variance = ((log_hc * log_ho) + (log_lc * log_lo)).dropna()
        if variance.empty:
            return np.nan
        return np.sqrt(max(float(variance.mean()), 0.0)) * ANNUALIZATION_FACTOR
    
    def parkinson_volatility(self,arr):
        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        log_hl = np.log(frame["High"] / frame["Low"])
        variance = ((log_hl ** 2) / (4 * np.log(2))).dropna()
        if variance.empty:
            return np.nan
        return np.sqrt(max(float(variance.mean()), 0.0)) * ANNUALIZATION_FACTOR

    def rolling_realized_volatility(self,arr,method="close-to-close"):
        method = method.lower()

        if method == "close-to-close":
            if hasattr(arr, "columns"):
                if "Close" not in arr.columns:
                    raise ValueError("arr must contain a 'Close' column.")
                close = arr["Close"]
            else:
                close = arr
            returns = close.pct_change().dropna()
            return returns.std() * ANNUALIZATION_FACTOR

        if method == "hodges-tompkins":
            if hasattr(arr, "columns"):
                if "Close" not in arr.columns:
                    raise ValueError("arr must contain a 'Close' column.")
                close = arr["Close"]
            else:
                close = arr
            returns = np.log(close / close.shift()).dropna()
            volatility = returns.std() * ANNUALIZATION_FACTOR
            if returns.empty:
                return volatility

            window = len(returns)
            subseries_count = len(returns)
            if subseries_count <= 0:
                return volatility

            denominator = (
                1
                - (window / subseries_count)
                + ((window ** 2 - 1) / (3 * (subseries_count ** 2)))
            )
            if denominator <= 0:
                return volatility
            return volatility * np.sqrt(1 / denominator)

        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr

        if method == "parkinson":
            log_hl = np.log(frame["High"] / frame["Low"])
            variance = ((log_hl ** 2) / (4 * np.log(2))).dropna()
        elif method == "garman-klass":
            log_hl = np.log(frame["High"] / frame["Low"])
            log_co = np.log(frame["Close"] / frame["Open"])
            variance = (
                0.5 * (log_hl ** 2)
                - ((2 * np.log(2)) - 1) * (log_co ** 2)
            ).dropna()
        elif method == "rogers-satchell":
            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            variance = ((log_hc * log_ho) + (log_lc * log_lo)).dropna()
        elif method == "yang-zhang":
            if len(frame) < 2:
                return np.nan
            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_co = np.log(frame["Close"] / frame["Open"])
            log_oc = np.log(frame["Open"] / frame["Close"].shift(1))
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            rs_variance = (log_hc * log_ho) + (log_lc * log_lo)
            k = 0.34 / (1.34 + ((len(frame) + 1) / (len(frame) - 1)))
            yz_variance = (
                log_oc.dropna().var()
                + (k * log_co.dropna().var())
                + ((1 - k) * rs_variance.dropna().mean())
            )
            if np.isnan(yz_variance):
                return np.nan
            return np.sqrt(max(float(yz_variance), 0.0)) * ANNUALIZATION_FACTOR
        else:
            raise ValueError(f"Unsupported rolling volatility method: {method}")

        if variance.empty:
            return np.nan
        return np.sqrt(max(float(variance.mean()), 0.0)) * ANNUALIZATION_FACTOR

    def ewma_realized_volatility(self,arr,method="close-to-close"):
        method = method.lower()
        observation_count = len(arr)
        if observation_count == 0:
            return np.nan
        alpha = 2.0 / (observation_count + 1.0)

        if method == "close-to-close":
            if hasattr(arr, "columns"):
                if "Close" not in arr.columns:
                    raise ValueError("arr must contain a 'Close' column.")
                close = arr["Close"]
            else:
                close = arr
            returns = close.pct_change().dropna()
            if returns.empty:
                return np.nan
            ewma_variance = returns.pow(2).ewm(
                alpha=alpha,
                adjust=False,
                min_periods=len(returns),
            ).mean()
            variance = max(float(ewma_variance.dropna().iloc[-1]), 0.0)
            return np.sqrt(variance * 252)

        required_columns = ("Open", "High", "Low", "Close")
        if not hasattr(arr, "columns"):
            raise TypeError("arr must be a pandas DataFrame with OHLC columns.")
        missing_columns = [column for column in required_columns if column not in arr.columns]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"arr is missing required OHLC columns: {missing_text}.")
        frame = arr
        if method == "parkinson":
            log_hl = np.log(frame["High"] / frame["Low"])
            variance = (log_hl ** 2) / (4 * np.log(2))
        elif method == "garman-klass":
            log_hl = np.log(frame["High"] / frame["Low"])
            log_co = np.log(frame["Close"] / frame["Open"])
            variance = 0.5 * (log_hl ** 2) - ((2 * np.log(2)) - 1) * (log_co ** 2)
        elif method == "rogers-satchell":
            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            variance = (log_hc * log_ho) + (log_lc * log_lo)
        elif method == "yang-zhang":
            if observation_count < 2:
                return np.nan

            log_ho = np.log(frame["High"] / frame["Open"])
            log_lo = np.log(frame["Low"] / frame["Open"])
            log_co = np.log(frame["Close"] / frame["Open"])
            log_oc = np.log(frame["Open"] / frame["Close"].shift(1))
            log_hc = np.log(frame["High"] / frame["Close"])
            log_lc = np.log(frame["Low"] / frame["Close"])
            rs_variance = (log_hc * log_ho) + (log_lc * log_lo)
            k = 0.34 / (1.34 + ((observation_count + 1) / (observation_count - 1)))
            log_oc = log_oc.dropna()
            log_co = log_co.dropna()
            rs_variance = rs_variance.dropna()
            if len(log_oc) < 2 or len(log_co) < 2 or rs_variance.empty:
                return np.nan
            overnight_variance = log_oc.ewm(alpha=alpha, adjust=False, min_periods=len(log_oc)).var(bias=False)
            open_to_close_variance = log_co.ewm(alpha=alpha, adjust=False, min_periods=len(log_co)).var(bias=False)
            rs_component = rs_variance.ewm(alpha=alpha, adjust=False, min_periods=len(rs_variance)).mean()
            yz_variance = (
                overnight_variance.dropna().iloc[-1]
                + (k * open_to_close_variance.dropna().iloc[-1])
                + ((1 - k) * rs_component.dropna().iloc[-1])
            )
            if np.isnan(yz_variance):
                return np.nan
            return np.sqrt(max(float(yz_variance), 0.0) * 252)
        else:
            raise ValueError(f"Unsupported EWMA volatility method: {method}")

        variance = variance.dropna()
        if variance.empty:
            return np.nan
        ewma_variance = variance.ewm(
            alpha=alpha,
            adjust=False,
            min_periods=len(variance),
        ).mean()
        return np.sqrt(max(float(ewma_variance.dropna().iloc[-1]), 0.0) * 252)
    
    def correlation(self,a,b):
        return a.corr(b)
    
    def cointegration(self,a,b):
        score, pvalue, _ = coint(a,b, maxlag=1)
        return pvalue
    
    def standard_deviation(self,arr):
        return arr.pct_change().std()
    
    def semi_standard_deviation(self,arr):
        return_series = arr.pct_change().dropna()
        return return_series[return_series<0].std() 
    
    def up_down_diff(self,arr):
        returns = arr.pct_change().dropna()
        return returns.std() - returns[returns<0].std()

    def beta(self,arr, benchmark):  
        arr_returns = (arr / arr.shift()).dropna()
        benchmark_log_returns = np.log(benchmark / benchmark.shift()).dropna()
        benchmark_returns = (benchmark / benchmark.shift()).dropna()
        cov = arr_returns.cov(benchmark_log_returns)
        var = benchmark_returns.var()
        return cov / var
        
    def alpha(self,arr, benchmark):
        arr_return = (arr.iloc[-1] - arr.iloc[0]) / np.abs(arr.iloc[0])
        benchmark_return = (benchmark.iloc[-1] - benchmark.iloc[0]) / np.abs(benchmark.iloc[0])
        arr_returns = (arr / arr.shift()).dropna()
        benchmark_log_returns = np.log(benchmark / benchmark.shift()).dropna()
        benchmark_returns = (benchmark / benchmark.shift()).dropna()
        beta = arr_returns.cov(benchmark_log_returns) / benchmark_returns.var()
        return arr_return - (beta * benchmark_return)

    def sharpe(self, arr, risk_free_rate):
        R_f = risk_free_rate[0]
        portfolio_return = (arr.iloc[-1] - arr.iloc[0]) / np.abs(arr.iloc[0])
        portfolio_std = np.log(arr / arr.shift()).dropna().std()
        return ((portfolio_return) / portfolio_std) * np.sqrt(len(arr) / 252)
    
    def sortino(self,arr, risk_free_rate):
        R_f = risk_free_rate
        return_series = arr.pct_change().dropna()
        expected_R_a = return_series.mean() 
        R_a_std_neg =return_series[return_series<0].std() 
        return  (expected_R_a / R_a_std_neg)* np.sqrt(252)
        
    def treynor(self,arr,benchmark):
        return_series = arr.pct_change().dropna()
        expected_R_a = return_series.mean()
        arr_returns = (arr / arr.shift()).dropna()
        benchmark_log_returns = np.log(benchmark / benchmark.shift()).dropna()
        benchmark_returns = (benchmark / benchmark.shift()).dropna()
        portfolio_beta = arr_returns.cov(benchmark_log_returns) / benchmark_returns.var()
        return (expected_R_a / portfolio_beta) #* np.sqrt(252)
    
    def calmar(self, arr):
        return_series = arr.pct_change().dropna()
        expected_R_a = return_series.mean()
        total_return = arr.pct_change().dropna().cumsum()
        drawdown = total_return - total_return.cummax()
        max_drawdown = drawdown.min()
        return (expected_R_a / abs(max_drawdown))* np.sqrt( 252)

    def omega(self,arr, benchmark):
        threshold = benchmark.pct_change().dropna()
        daily_threshold = (threshold + 1) ** np.sqrt(1/252) -1
        daily_return = arr.pct_change().dropna()
        excess = daily_return - daily_threshold
        PositiveSum = excess[excess > 0].sum()
        NegativeSum = excess[excess < 0].sum()
        return PositiveSum / (-NegativeSum)
    
    def information(self,arr, benchmark):
        benchmark_return = benchmark.pct_change().dropna()
        return_series = arr.pct_change().dropna()
        difference = return_series-benchmark_return
        volatility = difference.std() * np.sqrt(252)
        information = difference.mean()/volatility
        return information
    
    def M2(self,returns, benchmark_returns, risk_free_rate):
        portfolio_return = (returns.iloc[-1] - returns.iloc[0]) / np.abs(returns.iloc[0])
        portfolio_std = np.log(returns / returns.shift()).dropna().std()
        sharpe = ((portfolio_return) / portfolio_std) * np.sqrt(len(returns) / 252)
        r_f = risk_free_rate[-1]
        benchmark_std = benchmark_returns.std()
        return (sharpe * benchmark_std) + r_f
         
    def max_drawdown(self,arr):
        total_return = arr.pct_change().dropna().cumsum()
        drawdown = total_return - total_return.cummax()
        return drawdown.min()


Algorithm = Metric

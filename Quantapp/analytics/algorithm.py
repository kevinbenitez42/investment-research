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

class Algorithm:
    
    def __init__(self):
        self.a = 1
        self.calculate_returns = False
        self.volatility_models = [
            self.yang_zhang_volatility.__name__,
            self.garman_glass_volatility.__name__,
            self.rogers_satchell_volatility.__name__,
            self.parkinson_volatility.__name__,
            self.hodges_tompkins_volatility.__name__
        ]
    
    def percent_change(self, arr, window=1):
        arr= arr.pct_change(window).fillna(0)
        arr = arr.drop(arr.index[0])
        return arr
    
    def percent_return(self, arr):
        return (arr[-1] - arr[0]) / np.abs(arr[0])
    
    def rate_of_change(self,arr):
        return ((arr[0]- arr[-1]) / arr[-1]) * 100
    
    def z_score(self,arr):
        return (arr - arr.mean()) / np.std(arr)
    
    def log_returns(self, arr):
        return np.log(arr/arr.shift())
    
    def returns(self, arr):
        return arr/arr.shift()
    
    def average(self, arr):
        return arr.mean()
    
    def exp_average(self,arr):
        pass
    
    def median(self,arr):
        return arr.median()
    
    def mode(self,arr):
        return arr.mode()
    
    def skew(self,arr):
        return arr.skew()
    
    def kurtosis(self,arr):
        return arr.kurtosis()
    
    def garman_glass_volatility(self,arr,window=21):
        return GarmanKlass.get_estimator(arr, window=window)
    
    def yang_zhang_volatility(self,arr, window=21):
        return YangZhang.get_estimator(arr, window=window)
    
    def hodges_tompkins_volatility(self,arr,window=21):
        return HodgesTompkins.get_estimator(arr, window=window)
    
    def rogers_satchell_volatility(self,arr,window=21):
        return RogersSatchell.get_estimator(arr, window=window)
    
    def parkinson_volatility(self,arr,window=21):
       return Parkinson.get_estimator(arr, window=window)
    
    def correlation(self,a,b):
        return a.corr(b)
    
    def cointegration(self,a,b):
        score, pvalue, _ = coint(a,b, maxlag=1)
        return pvalue
    
    def standard_deviation(self,arr):
        return arr.pct_change().std()
    
    def semi_standard_deviation(self,arr):
        return_series = self.percent_change(arr) 
        return return_series[return_series<0].std() 
    
    def up_down_diff(self,arr):
        return self.standard_deviation(arr) - self.semi_standard_deviation(arr)

    def beta(self,arr, benchmark):  
        cov = self.returns(arr).cov(self.log_returns(benchmark))
        var = self.returns(benchmark).var()
        return cov / var
        
    def alpha(self,arr, benchmark):
        return self.percent_return(arr) - (self.beta(arr,benchmark) * self.percent_return(benchmark))

    def sharpe(self, arr, risk_free_rate):
        R_f = risk_free_rate[0]
        portfolio_return = self.percent_return(arr) 
        portfolio_std =  self.log_returns(arr).std()
        return ((portfolio_return) / portfolio_std) * np.sqrt(len(arr) / 252)
    
    def sortino(self,arr, risk_free_rate):
        R_f = risk_free_rate
        return_series = self.percent_change(arr)
        expected_R_a = return_series.mean() 
        R_a_std_neg =return_series[return_series<0].std() 
        return  (expected_R_a / R_a_std_neg)* np.sqrt(252)
        
    def treynor(self,arr,benchmark):
        return_series = self.percent_change(arr)
        expected_R_a = return_series.mean()
        portfolio_beta = self.beta(arr,benchmark)
        return (expected_R_a / portfolio_beta) #* np.sqrt(252)
    
    def calmar(self, arr):
        return_series = self.percent_change(arr)
        expected_R_a = return_series.mean()
        return (expected_R_a / abs(self.max_drawdown(arr)))* np.sqrt( 252)

    def omega(self,arr, benchmark):
        threshold = self.percent_change(benchmark)
        daily_threshold = (threshold + 1) ** np.sqrt(1/252) -1
        daily_return = self.percent_change(arr) 
        excess = daily_return - daily_threshold
        PositiveSum = excess[excess > 0].sum()
        NegativeSum = excess[excess < 0].sum()
        return PositiveSum / (-NegativeSum)
    
    def information(self,arr, benchmark):
        benchmark_return = self.percent_change(benchmark)
        return_series = self.percent_change(arr)
        difference = return_series-benchmark_return
        volatility = difference.std() * np.sqrt(252)
        information = difference.mean()/volatility
        return information
    
    def M2(self,returns, benchmark_returns, risk_free_rate):
        sharpe = self.sharpe(returns,risk_free_rate)
        r_f = risk_free_rate[-1]
        benchmark_std = benchmark_returns.std()
        return (sharpe * benchmark_std) + r_f
         
    def max_drawdown(self,arr):
        total_return = self.percent_change(arr).cumsum()
        drawdown = total_return - total_return.cummax()
        return drawdown.min()

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

class SequenceGenerator:

    def linear(self, x, dtype='decimal'):
        result = x  # Linear growth simply returns the input values as the output
        
        if dtype == 'int':
            result = np.round(result).astype(int)  # Round to nearest integer if requested
        
        return result
    
    #create exponential sequences
    def exponential(self, x, exp_bases=[np.e, 2, 3], exp_rates=[0.05, 0.1, 0.15, 0.2], output_type='decimal'):       
        growth_dict = {}  # Dictionary to store different exponential growths

        # Iterate over each base for exponential growths
        for base in exp_bases:
            result = base**x  # Calculate base^x for each base
            if output_type == 'int':
                result = np.round(result).astype(int)  # Round to nearest integer if requested
            growth_dict[f'Exponential (base={base})'] = result
        
        # Iterate over each rate for exponential growths
        for rate in exp_rates:
            result = np.exp(rate * x)  # Calculate exp(rate * x) for each rate
            if output_type == 'int':
                result = np.round(result).astype(int)  # Round to nearest integer if requested
            growth_dict[f'Exponential (rate={rate})'] = result

        return growth_dict

    #create polynomial sequences
    def polynomial(self, x, powers=[1.5, 2.5, 4, 5], output_type='decimal'):
        growth_dict = {}  # Dictionary to store different polynomial growths

        # Iterate over each power and compute x raised to the power
        for p in powers:
            result = x**p  # Calculate x raised to the power p
            if output_type == 'int':
                result = np.round(result).astype(int)  # Round to nearest integer if requested
            growth_dict[f'Polynomial (x^{p})'] = result
        
        return growth_dict  # Return the dictionary of polynomial growth patterns
    
    #create root sequences
    def root(self, x, roots=[2, 3, 4, 6, 8, 10, 12], output_type='decimal'):
        growth_dict = {}  # Dictionary to store different root-based growths
        
        # Iterate over each root and compute the corresponding root (x^(1/root))
        for r in roots:
            result = np.power(x, 1 / r)  # Calculate the root x^(1/root)
            if output_type == 'int':
                result = np.round(result).astype(int)  # Round to nearest integer if requested
            growth_dict[f'{r}-th Root (x^(1/{r}))'] = result
        
        return growth_dict

    #create logistic sequences
    def logistic(self, x, L_values, k_values, x0_values, output_type='decimal'):
        growth_dict = {}
        
        # Iterate over all combinations of L, k, and x0
        for L in L_values:
            for k in k_values:
                for x0 in x0_values:
                    key = f'Logistic (L={L}, k={k}, x0={x0})'
                    result = L / (1 + np.exp(-k * (x - x0)))  # Logistic growth formula
                    
                    if output_type == 'int':
                        result = np.round(result).astype(int)  # Round to nearest integer if requested
                    growth_dict[key] = result
        
        return growth_dict
 
    # Downsample a sequence or DataFrame
    def downsample(self, data, step):
        """
        Downsample a pandas Series or DataFrame by picking every nth value.

        Parameters:
        - data (pd.Series or pd.DataFrame): The input data to downsample.
        - step (int): The step size for downsampling.

        Returns:
        - pd.Series or pd.DataFrame: The downsampled data.
        """
        if isinstance(data, pd.Series):
            return data[::step]  # Downsample the Series
        elif isinstance(data, pd.DataFrame):
            downsampled_dict = {}
            for column in data.columns:
                downsampled_dict[column] = data[column][::step]  # Downsample each column
            return pd.DataFrame(downsampled_dict)  # Convert the dictionary to a new DataFrame
        else:
            raise TypeError("Input must be a pandas Series or DataFrame")

    #scale a sequence to a specific range
    def scale(self, y_values, final_value):
        if isinstance(y_values, pd.Series):
            scale_factor = final_value / y_values.iloc[-1]
            return y_values * scale_factor
        elif isinstance(y_values, pd.DataFrame):
            scaled_df = y_values.copy()
            for column in scaled_df.columns:
                scale_factor = final_value / scaled_df[column].iloc[-1]
                scaled_df[column] = scaled_df[column] * scale_factor
            return scaled_df
        else:
            raise TypeError("Input must be a pandas Series or DataFrame")

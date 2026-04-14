from numpy import correlate, short
import yfinance as yf
import pandas   as pd
import numpy as np
from collections import Counter
from IPython.display import display

from Quantapp.analytics import Rolling
from py_vollib.black_scholes.greeks.analytical import *
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes import *

from datetime import datetime
date_format = "%Y-%m-%d"

import os
import json

class Portfolio:
    def __init__(self, universe):
        self.assets = {}
        self.position_details = pd.DataFrame()
        self.comp = Rolling()
        self.universe = universe
    
    def add(self, ticker, direction='long'):
        self.assets[ticker] = { 'asset': Asset(ticker, universe=self.universe),
                               'direction': direction, 
                               'weight': 0.0 }
        
    def add_tickers(self,tickers):
        for ticker, direction in tickers.items():
            self.add(ticker,direction)
    
    def add_csv(self,file_path):
        
        files = os.listdir(file_path)


        capital_requirements_csv = pd.read_csv(file_path +'/'+ files[0])
        positions_csv = pd.read_csv(file_path + '/' + files[1])

        positions_drop_column = ['Account','Exp Date', 'Underlying Last Price', 'DTE','β Delta','Chg %', 'Bid', 'Ask','Indicators']
        capital_requirements_drop_column = ['Margin Type', 'Instrument Type', 'Quantity', 'Exp Date', 'Strike Price', 'Call/Put','DTE', 'Maintenance', 'Initial Requirement']

        positions_csv = positions_csv.drop(positions_drop_column,axis=1)
        positions_csv['Symbol'] = positions_csv['Symbol'].apply(lambda x: x.split()[0])

        capital_requirements_csv  = capital_requirements_csv.drop(capital_requirements_drop_column,axis=1)
        capital_requirements_csv  = capital_requirements_csv[capital_requirements_csv.Underlying != 'Spread']
        capital_requirements_csv  = capital_requirements_csv.drop(capital_requirements_csv.index[0])
        capital_requirements_csv  = capital_requirements_csv[capital_requirements_csv.Description.str.contains('Total')]

        grouped_drop_column = ['Call/Put', 'Quantity', 'Strike Price', 'PoP']
        grouped_csv = positions_csv.drop(grouped_drop_column, axis=1)
        grouped_csv = grouped_csv.groupby(['Symbol']).sum()
        grouped_csv = pd.merge(capital_requirements_csv, grouped_csv, left_on='Underlying', right_on='Symbol')
        grouped_csv['directional bias'] = grouped_csv['Delta'].apply(lambda x: 'long' if x > 0 else 'short')

        universe_data = self.universe.retrieve_universe()[['Symbol','Market Cap', 'Sector', 'Industry', 'Sub-Industry']]


        self.position_details = grouped_csv
        self.position_details = pd.merge(universe_data,self.position_details,left_on='Symbol',right_on='Underlying').drop_duplicates('Symbol')
        self.position_details = self.position_details.drop('Underlying', axis=1)

        self.position_details['Theta / Delta'] = self.position_details['Theta'] / self.position_details['Delta']
        self.position_details['Delta / Gamma'] = self.position_details['Delta'] / self.position_details['Gamma']
        self.position_details['Vega / Delta'] = self.position_details['Vega'] / self.position_details['Delta']
        
        self.position_details['BP Usage %'] = self.position_details['BP Usage %'].apply(lambda s: float(s.replace('%', '')))
        self.position_details['BP Usage % ex cash'] = self.position_details['BP Usage %'] / self.position_details['BP Usage %'].sum() * 100

        #display(self.position_details)
        data = grouped_csv.loc[:, ['Underlying', 'directional bias']]
        portfolio_dict = {}

        for index, row in data.iterrows():
            portfolio_dict[row['Underlying']] = row['directional bias']

        self.add_tickers(portfolio_dict)
        return portfolio_dict

    def load(self, period='1y', interval='1D'):
        data = {}
        for key,value in self.assets.items():
            data[key] = value['asset'].load(period=period,
                                            interval=interval,
                                            direction=value['direction'])
        return data
    
    def load_portfolio_config(self,file_name):
        config = json.load( open( f'portfolio_configs/{file_name}.json') )
        for key,value in config.items():
            self.assets[key] = {
                'asset'     : Asset(key),
                'direction' : value['direction'],
                'weight' : value['weight']
            }

    def save_portfolio_config(self,file_name='a'):
        config = {}
        for key,value in self.assets.items():
            config[key] = { 
                'direction' : value['direction'],
                'weight' : value['weight']
            }
        json.dump( config, open( f'portfolio_configs/{file_name}.json', 'w' ) )
    
    def delete_portfolio_config(self):
        pass
    
    def retrieve_assets(self):
        return self.assets
    
    def remove(self,ticker):
        self.assets.pop(ticker)
    
    def clear(self):
        self.assets = {}
        
    def retrieve_ticker_data(self,mode='Close'):
        prices = pd.DataFrame()
        for ticker, data in self.assets.items():
            prices[ticker] = data['asset'].data['Ticker Data'][mode]
        prices.fillna(0, inplace=True)
        return prices


    def retrieve_sectors(self,mode='Close'):
        sectors = {}
        counts  = {}
        for ticker, data in self.assets.items():
            sectors[data['asset'].data['Symbol']] = data['asset'].data['Sector']
        for key,value in dict(Counter(sectors.values())).items():
            sector = key
            count = value
            counts[sector] = count / len(sectors)
        return counts
    
    def weighted_average_rolling(self,formula,window,*args):
        rolling_calculations          = self.comp.rolling_multi(formula, window, self.retrieve_ticker_data(), *args)
        weighted_rolling_calculations  = pd.DataFrame()

        for symbol in self.position_details['Symbol']:
            weight         = self.position_details[self.position_details['Symbol'] == symbol]['BP Usage % ex cash'] 
            weighted_rolling_calculations = rolling_calculations[symbol] * weight.iloc[0]
            
        return weighted_rolling_calculations

    def rolling(self,formula,window,*args):
        return self.comp.rolling_multi(formula, window, self.retrieve_ticker_data(), *args)

    def rolling_multi_timeframe(self, formula, windows,*args ):
        return self.comp.rolling_multi_timeframes(formula, windows, self.retrieve_ticker_data(), *args)

    def __init__(self):
        pass

    def days_to_expiration(self, expiration_date):
        exp_date = datetime.strptime(expiration_date, date_format)
        today = datetime.today()
        diff = exp_date - today
        return diff.days

    def calculate_greeks(self,ticker):
        oc = dm.retrieve_option_chain(ticker)
        current_price=  dm.retrieve_ticker_data(ticker)['Close'][-1]
        expirations = oc.columns
        for expiration in expirations:

            for index, row in oc[expiration][0].iterrows():
                days_to_expiration = self.days_to_expiration(expiration)
                flag = 'c'  # 'c' for call, 'p' for put
                F = current_price # Underlying asset price
                strike = row['strike']  # Strike
                t = days_to_expiration / 365  # (Annualized) time-to-expiration in years
                r = 0.01  # Interest free rate
                sigma = row['impliedVolatility']

                oc[expiration][0].iloc[index]['delta']  = delta(flag, F, strike, t, r, sigma)
                oc[expiration][0].iloc[index]['gamma']  = gamma(flag, F, strike, t, r, sigma)
                oc[expiration][0].iloc[index]['theta']  =  theta(flag, F, strike, t, r, sigma)
                oc[expiration][0].iloc[index]['vega']   =  vega(flag, F , strike, t, r, sigma)
        
        return oc


    def retrieve_option_chain(self,ticker):
        oc = dm.retrieve_option_chain(ticker)


    def retrieve_iv_term_structure(self, ticker='spy', mode='mean'):
        oc = self.retrieve_option_chain(ticker)
        expirations = oc.columns
        iv_term_structure_calls = []
        iv_term_structure_puts  = []
        iv_term_structure       = pd.DataFrame()
        for expiration in expirations:
            iv_term_structure_calls.append(oc[expiration][0]['impliedVolatility'].mean())
            iv_term_structure_puts.append(oc[expiration][1]['impliedVolatility'].mean())
        
        iv_term_structure['calls'] = pd.Series(iv_term_structure_calls)
        iv_term_structure['puts'] = pd.Series(iv_term_structure_puts)
        iv_term_structure = iv_term_structure.set_index(expirations)

        return iv_term_structure

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
#import Quantapps Computation libarary
from Quantapp.analytics.rolling import Rolling
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor
from plotly.subplots import make_subplots
from datetime import datetime
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
from collections.abc import Mapping
from Quantapp.data.market_data_client import MarketDataClient

market_data = MarketDataClient()

class PieChartPlotter:
    def __init__(self):
        pass
    
    def plot_sector_market_cap(self, sector):
        data = market_data.retrieve_market_data()
        stocks = data['SP500'][data['SP500']['Sector'] == sector]
        #go through each row and append the market cap to a new list
        market_caps = []
        for index, row in stocks.iterrows():
            try:
                ticker = yf.Ticker(row['Symbol'])
                market_cap = ticker.info['marketCap']
                market_caps.append(market_cap)
            except:
                market_caps.append(np.nan)
        stocks['Market Cap'] = market_caps
        sub_industries = stocks['Sub-Industry'].unique().tolist()

        #dictionary of tickers in each sub-industry
        sub_industry_tickers_dict = {}
        for sub_industry in sub_industries:
            tickers = stocks[stocks['Sub-Industry'] == sub_industry]['Symbol'].tolist()
            sub_industry_tickers_dict[sub_industry] = tickers


        #create a pie chart of the market cap of each sub-industry

        sub_industry_market_caps = []
        for sub_industry in sub_industries:
            tickers = sub_industry_tickers_dict[sub_industry]
            market_cap = stocks[stocks['Symbol'].isin(tickers)]['Market Cap'].sum()
            sub_industry_market_caps.append(market_cap)

        #create subplots for side-by-side pie charts
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'pie'}]], subplot_titles=['Market Cap by Sub-Industry', 'Market Cap by Company'])

        # add the overall sub-industry pie to the first subplot
        fig.add_trace(go.Pie(values=sub_industry_market_caps, labels=sub_industries, textinfo='label'), row=1, col=1)

        # add traces for each sub-industry pie to the second subplot
        for sub_industry in sub_industries:
            tickers = sub_industry_tickers_dict[sub_industry]
            market_caps = stocks[stocks['Symbol'].isin(tickers)][['Symbol', 'Market Cap']]
            fig.add_trace(go.Pie(labels=market_caps['Symbol'], values=market_caps['Market Cap'], name=sub_industry, visible=False, textinfo='label+percent'), row=1, col=2)

        # make the first sub-industry pie visible
        fig.data[1].visible = True

        # create buttons for the dropdown
        buttons = []
        for i, sub_industry in enumerate(sub_industries):
            visible_list = [True] + [j == i for j in range(len(sub_industries))]
            button = dict(method='update',
                        label=sub_industry,
                        args=[{'visible': visible_list},
                                {'title': 'Market Cap by Company in ' + sub_industry + ' Sub-Industry'}])
            buttons.append(button)

        # update layout with updatemenus
        fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=1.0, y=1.0, xanchor='right', yanchor='top')])

        #height a little taller
        fig.update_layout(height=900, title_text=sector + " Sector Market Cap Breakdown")

        return fig

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
#import Quantapps Computation libarary
from Quantapp.analytics import Rolling
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

rolling = Rolling()

class CandleStickPlotter:
    def __init__(self):
        pass
    
    def create_candlestick_fig(self,ticker_data, drop_window=14, period='1Y', bollinger_window=21, title="Candlestick With Bollinger Bands"):
        """
        Plots the candlestick chart with Bollinger Bands for the given stock data.
        
        Parameters:
        - ticker_data: DataFrame containing candlestick data with 'Open', 'High', 'Low', 'Close' columns.
        - drop_window: Number of days for calculating the percentage drop.
        - period: Period to filter the data.
        - bollinger_window: Window for the moving average to calculate Bollinger Bands.
        - title: Title of the plot.
        """
        # Remove weekends/holidays and calculate percentage drop
        ticker_data = ticker_data[ticker_data.index.dayofweek < 5]
        holidays = pd.to_datetime(['2023-01-01', '2023-12-25'])  # Add more holidays as needed
        ticker_data = ticker_data[~ticker_data.index.isin(holidays)]
        ticker_data = rolling.calculate_percentage_drop(ticker_data, windows=drop_window)
        mean_drop = ticker_data['PercentageDrop'].mean()
        std_drop = ticker_data['PercentageDrop'].std()
    
        # Filter data for the specified period
        period_data = ticker_data.last(period)

        # Define bar colors
        colors = [
            'red' if drop < mean_drop - 0.5 * std_drop
            else 'blue' if drop < mean_drop + 0.25 * std_drop
            else 'green'
            for drop in period_data['PercentageDrop']
        ]
        
        

        # Calculate Bollinger Bands
        ma = period_data['Close'].rolling(window=bollinger_window).mean()
        std = period_data['Close'].rolling(window=bollinger_window).std()

        bollinger_bands = {}
        for k in [1, 2, 3]:
            bollinger_bands[f'Upper_{k}'] = ma + (std * k)
            bollinger_bands[f'Lower_{k}'] = ma - (std * k)
        bollinger_df = pd.DataFrame(bollinger_bands)
        
        # Create a single-figure candlestick chart
        fig = go.Figure()

        # Add candlestick data
        for i, color in enumerate(colors):
            fig.add_trace(go.Candlestick(
                x=[period_data.index[i]],
                open=[period_data['Open'].iloc[i]],
                high=[period_data['High'].iloc[i]],
                low=[period_data['Low'].iloc[i]],
                close=[period_data['Close'].iloc[i]],
                increasing_line_color=color,
                decreasing_line_color=color,
                showlegend=False
            ))

        # Add Bollinger Bands
        for k in [1, 2, 3]:
            fig.add_trace(go.Scatter(
                x=period_data.index,
                y=bollinger_df[f'Upper_{k}'],
                mode='lines',
                line=dict(width=1, dash='dash'),
                name=f'Upper Band {k} SD'
            ))
            fig.add_trace(go.Scatter(
                x=period_data.index,
                y=bollinger_df[f'Lower_{k}'],
                mode='lines',
                line=dict(width=1, dash='dash'),
                name=f'Lower Band {k} SD'
            ))
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            yaxis=dict(autorange=True, fixedrange=False),
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False
            )
        )

        return fig

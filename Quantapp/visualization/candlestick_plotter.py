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
        band_colors = {
            1: '#4cc9f0',
            2: '#ffd166',
            3: '#ef476f',
        }
        moving_average_color = '#f8f9fa'
        
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
        fig.add_trace(go.Scatter(
            x=period_data.index,
            y=ma,
            mode='lines',
            line=dict(width=2, color=moving_average_color),
            name=f'{bollinger_window}-Period MA'
        ))

        for k in [1, 2, 3]:
            band_color = band_colors[k]
            fig.add_trace(go.Scatter(
                x=period_data.index,
                y=bollinger_df[f'Upper_{k}'],
                mode='lines',
                line=dict(width=1, dash='dash', color=band_color),
                name=f'Upper +{k}\u03c3'
            ))
            fig.add_trace(go.Scatter(
                x=period_data.index,
                y=bollinger_df[f'Lower_{k}'],
                mode='lines',
                line=dict(width=1, dash='dash', color=band_color),
                name=f'Lower -{k}\u03c3'
            ))

        # Label the latest available Bollinger values so the active sigma levels are obvious.
        for k in [1, 2, 3]:
            band_color = band_colors[k]
            for band_name, sigma_label, yshift in [
                (f'Upper_{k}', f'+{k}\u03c3', 14),
                (f'Lower_{k}', f'-{k}\u03c3', -14),
            ]:
                latest_band = bollinger_df[band_name].dropna()
                if latest_band.empty:
                    continue

                latest_x = latest_band.index[-1]
                latest_y = latest_band.iloc[-1]
                fig.add_trace(go.Scatter(
                    x=[latest_x],
                    y=[latest_y],
                    mode='markers',
                    marker=dict(size=7, color=band_color),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_annotation(
                    x=latest_x,
                    y=latest_y,
                    text=f'{sigma_label}: {latest_y:,.2f}',
                    showarrow=False,
                    xanchor='left',
                    xshift=12,
                    yshift=yshift,
                    font=dict(size=11, color=band_color),
                    bgcolor='rgba(17, 24, 39, 0.85)',
                    bordercolor=band_color,
                    borderwidth=1,
                    borderpad=4,
                )

        latest_ma = ma.dropna()
        if not latest_ma.empty:
            ma_x = latest_ma.index[-1]
            ma_y = latest_ma.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[ma_x],
                y=[ma_y],
                mode='markers',
                marker=dict(size=7, color=moving_average_color, line=dict(color='#111827', width=1)),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_annotation(
                x=ma_x,
                y=ma_y,
                text=f'MA({bollinger_window}): {ma_y:,.2f}',
                showarrow=False,
                xanchor='left',
                xshift=12,
                yshift=28,
                font=dict(size=11, color=moving_average_color),
                bgcolor='rgba(17, 24, 39, 0.92)',
                bordercolor=moving_average_color,
                borderwidth=1,
                borderpad=4,
            )

        latest_close = period_data['Close'].dropna()
        if not latest_close.empty:
            close_x = latest_close.index[-1]
            close_y = latest_close.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[close_x],
                y=[close_y],
                mode='markers',
                marker=dict(size=8, color='white', line=dict(color='#111827', width=1)),
                name='Latest Close',
                showlegend=False
            ))
            fig.add_annotation(
                x=close_x,
                y=close_y,
                text=f'Close: {close_y:,.2f}',
                showarrow=False,
                xanchor='left',
                xshift=12,
                font=dict(size=11, color='white'),
                bgcolor='rgba(17, 24, 39, 0.92)',
                bordercolor='white',
                borderwidth=1,
                borderpad=4,
            )

        xaxis_range = None
        if len(period_data.index) > 1:
            span_days = max((period_data.index[-1] - period_data.index[0]).days, 1)
            xaxis_range = [
                period_data.index[0],
                period_data.index[-1] + pd.Timedelta(days=max(10, int(span_days * 0.08)))
            ]
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            margin=dict(r=180),
            yaxis=dict(autorange=True, fixedrange=False),
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False,
                range=xaxis_range
            )
        )

        return fig

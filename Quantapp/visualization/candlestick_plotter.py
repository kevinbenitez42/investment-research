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
from Quantapp.analytics.series_utils import calculate_textbook_rolling_max_drawdown
from .traces.candlestick import (
    build_candlestick_trace_bundle as trace_build_candlestick_trace_bundle,
    build_candlestick_y_range as trace_build_candlestick_y_range,
    build_numeric_axis_range as trace_build_numeric_axis_range,
    build_time_range as trace_build_time_range,
    slice_series_to_range as trace_slice_series_to_range,
)

rolling = Rolling()

class CandleStickPlotter:
    def __init__(self):
        pass

    @staticmethod
    def build_time_range(period_start, period_end, offset=None):
        return trace_build_time_range(period_start, period_end, offset=offset)

    @staticmethod
    def _slice_series_to_range(series, start=None, end=None):
        return trace_slice_series_to_range(series, start=start, end=end)

    @staticmethod
    def build_numeric_axis_range(series_list, include_zero=False, padding_ratio=0.08):
        return trace_build_numeric_axis_range(
            series_list,
            include_zero=include_zero,
            padding_ratio=padding_ratio,
        )

    @classmethod
    def build_candlestick_y_range(cls, bundle, start=None, end=None, overlay_mode="all"):
        return trace_build_candlestick_y_range(
            bundle,
            start=start,
            end=end,
            overlay_mode=overlay_mode,
        )

    def build_candlestick_trace_bundle(
        self,
        ticker_data,
        drop_window=14,
        period='1Y',
        bollinger_window=21,
        max_drawdown_price_windows=None,
    ):
        """
        Prepare reusable candlestick traces plus overlay group metadata.
        """
        return trace_build_candlestick_trace_bundle(
            ticker_data=ticker_data,
            drop_window=drop_window,
            period=period,
            bollinger_window=bollinger_window,
            max_drawdown_price_windows=max_drawdown_price_windows,
        )
    
    def create_candlestick_fig(
        self,
        ticker_data,
        drop_window=14,
        period='1Y',
        bollinger_window=21,
        title="Candlestick With Bollinger Bands",
        max_drawdown_price_windows=None,
    ):
        """
        Plots the candlestick chart with Bollinger Bands for the given stock data.
        
        Parameters:
        - ticker_data: DataFrame containing candlestick data with 'Open', 'High', 'Low', 'Close' columns.
        - drop_window: Number of days for calculating the percentage drop.
        - period: Period to filter the data.
        - bollinger_window: Window for the moving average to calculate Bollinger Bands.
        - title: Title of the plot.
        - max_drawdown_price_windows: Optional iterable of trailing windows whose latest
          textbook max-drawdown values should be mapped back into rolling price-level overlays.
        """
        bundle = self.build_candlestick_trace_bundle(
            ticker_data=ticker_data,
            drop_window=drop_window,
            period=period,
            bollinger_window=bollinger_window,
            max_drawdown_price_windows=max_drawdown_price_windows,
        )

        fig = go.Figure()
        for trace in bundle['traces']:
            fig.add_trace(copy.deepcopy(trace))

        if bundle['latest_close_annotation'] is not None:
            fig.add_annotation(**copy.deepcopy(bundle['latest_close_annotation']))

        bollinger_trace_indices = list(bundle['overlay_trace_groups']['bollinger'])
        mdd_trace_indices = list(bundle['overlay_trace_groups']['mapped_mdd'])
        overlay_trace_indices = bollinger_trace_indices + mdd_trace_indices

        updatemenus = []
        if bollinger_trace_indices and mdd_trace_indices:
            initial_visibility = ([False] * len(bollinger_trace_indices)) + ([True] * len(mdd_trace_indices))
            for trace_idx, is_visible in zip(overlay_trace_indices, initial_visibility):
                fig.data[trace_idx].visible = is_visible

            updatemenus.append(
                dict(
                    type='dropdown',
                    buttons=[
                        dict(
                            label='Bollinger',
                            method='update',
                            args=[
                                {'visible': ([True] * len(bollinger_trace_indices)) + ([False] * len(mdd_trace_indices))},
                                {},
                                overlay_trace_indices,
                            ],
                        ),
                        dict(
                            label='Mapped MDD',
                            method='update',
                            args=[
                                {'visible': ([False] * len(bollinger_trace_indices)) + ([True] * len(mdd_trace_indices))},
                                {},
                                overlay_trace_indices,
                            ],
                        ),
                    ],
                    direction='down',
                    showactive=True,
                    active=1,
                    x=0.18,
                    xanchor='left',
                    y=1.12,
                    yanchor='top',
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            margin=dict(r=bundle['right_margin']),
            yaxis=dict(autorange=True, fixedrange=False),
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False,
                range=bundle['xaxis_range']
            ),
            updatemenus=updatemenus,
        )

        return fig

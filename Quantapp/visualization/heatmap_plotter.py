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

class HeatmapPlotter:
    def __init__(self):
        pass
    
    def plot_correlation_heatmap(self, data, title):
        corr_matrix = data.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1,
            zmax=1,
            colorbar_title='Correlation Coefficient'
        ))

        fig.update_layout(
            title=title,
            xaxis_nticks=len(corr_matrix.columns),
            yaxis_nticks=len(corr_matrix.index),
            height=800,
            width=800
        )

        return fig

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import coint
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor
from plotly.subplots import make_subplots
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

class BarChartPlotter:
    def __init__(self):
        pass
    
    def plot_sector_market_cap(self, sector):
        import yfinance as yf

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

        stocks_sorted = stocks.sort_values(by='Market Cap', ascending=False)
        stocks_sorted.reset_index(drop=True, inplace=True)

        # Create subplots: 2 rows, 1 column
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Market Cap of All {} Companies'.format(sector), 'Market Cap by Sub-Industry'],
            shared_xaxes=False
        )

        # First subplot: Overall bar chart
        fig.add_trace(
            go.Bar(
                x=stocks_sorted['Symbol'],
                y=stocks_sorted['Market Cap'],
                name='All {}'.format(sector),
                showlegend=False
            ),
            row=1, col=1
        )

        # Add vertical line for top 10% in first subplot
        n_top_10pct_overall = max(1, int(len(stocks_sorted) * 0.10))
        last_top_idx_overall = n_top_10pct_overall - 1
        fig.add_shape(
            type="line",
            x0=last_top_idx_overall + 0.5,
            x1=last_top_idx_overall + 0.5,
            y0=0,
            y1=stocks_sorted['Market Cap'].max() * 1.05,
            line=dict(color="red", width=3, dash="dash"),
            row=1, col=1
        )

        # Second subplot: Sub-industry traces
        for sub in sub_industries:
            tickers = sub_industry_tickers_dict[sub]
            df_sub = stocks_sorted[stocks_sorted['Symbol'].isin(tickers)]
            
            # Determine top 10% within sub-industry
            n_top_10pct = max(1, int(len(df_sub) * 0.10))
            last_top_idx = n_top_10pct - 1 if n_top_10pct > 0 else 0
            
            fig.add_trace(
                go.Bar(
                    x=df_sub['Symbol'],
                    y=df_sub['Market Cap'],
                    name=sub,
                    visible=(sub == sub_industries[0])  # Show first by default
                ),
                row=2, col=1
            )
            
            # Add vertical line for top 10% in sub-industry (initially only first visible)
            fig.add_shape(
                type="line",
                x0=last_top_idx + 0.5,
                x1=last_top_idx + 0.5,
                y0=0,
                y1=df_sub['Market Cap'].max() * 1.05,
                line=dict(color="red", width=3, dash="dash"),
                visible=(sub == sub_industries[0]),
                row=2, col=1
            )

        # Create buttons for dropdown (only affects second subplot traces and shapes)
        buttons = []
        num_sub = len(sub_industries)
        for i, sub in enumerate(sub_industries):
            # Visibility: first trace (overall) always True, then sub-industry traces
            visible_traces = [True] + [j == i for j in range(num_sub)]
            
            # Shapes: first shape (overall) always visible, then sub-industry shapes
            visible_shapes = [True] + [j == i for j in range(num_sub)]
            
            button = dict(
                method="update",
                label=sub,
                args=[
                    {"visible": visible_traces},
                    {"shapes": [fig.layout.shapes[k] for k in range(len(fig.layout.shapes))] if k < 1 or (k >= 1 and (k-1) == i) else None for k in range(len(fig.layout.shapes))}
                ]
            )
            buttons.append(button)

        # Update layout
        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=1.0, y=1.15, xanchor='right', yanchor='top')],
            height=1000,  # Adjust height for two subplots
            xaxis_tickangle=-45,
            xaxis2_tickangle=-45
        )

        return fig
    
    def plot_market_cap_weights(self,calculated_weights, chart_title):
        fig = px.bar(calculated_weights, x=calculated_weights.index, y=calculated_weights.columns,
                    title=chart_title,
                    labels={"value": "Market Cap Weight", "index": "Date"},
                    barmode='stack')
        return fig

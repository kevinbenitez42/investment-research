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

class BarChartPlotter:
    def __init__(self):
        pass
    
    def create_seasonality_fig(self, data, title, frequency='monthly'):
        
        """Plot seasonality of returns using Plotly.
        Parameters:
        - data: pd.Series or pd.DataFrame with DateTimeIndex and return values.
        - title: Title of the plot.
        - frequency: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.
        Returns:
        - fig: Plotly Figure object.
        """
        
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("data must be a Series or single-column DataFrame.")
            data = data.iloc[:, 0]
        elif not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series or single-column DataFrame.")

        data = data.copy().dropna().sort_index()

        # Ensure the index is a DateTimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        current_returns = None
            
     

        if frequency == 'monthly':
            frequency_label = 'Month'
            periods = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            current_period_label = datetime.now().strftime('%b')

            # Calculate mean and median returns for each month
            period_mean = data.groupby(data.index.month).mean().reindex(range(1, 13))
            period_median = data.groupby(data.index.month).median().reindex(range(1, 13))
            # Map period indices to month abbreviations
            period_mean.index = periods
            period_median.index = periods

            current_year_returns = data.loc[data.index.year == datetime.now().year]
            if not current_year_returns.empty:
                current_returns = current_year_returns.groupby(current_year_returns.index.month).last().reindex(range(1, 13))
                current_returns.index = periods

        elif frequency == 'weekly':
            weekly_data = data.to_frame(name='Return')

            # Assign month number and week of month
            weekly_data['Month_Num'] = weekly_data.index.month
            weekly_data['Month_Name'] = weekly_data.index.strftime('%b')
            weekly_data['Week_of_Month'] = weekly_data.index.to_series().apply(lambda d: (d.day - 1) // 7 + 1)

            # Create period labels in the format 'Month / Week X'
            weekly_data['Period_Label'] = weekly_data['Month_Name'] + ' / Week ' + weekly_data['Week_of_Month'].astype(str)

            # Create a numerical representation for sorting
            weekly_data['Period_Num'] = weekly_data['Month_Num'] * 10 + weekly_data['Week_of_Month']

            # Calculate mean and median returns for each period
            period_stats = weekly_data.groupby(['Period_Num', 'Period_Label'])['Return'].agg(['mean', 'median']).reset_index()

            # Sort the data chronologically
            period_stats = period_stats.sort_values('Period_Num')

            # Extract values for plotting
            period_mean = period_stats.set_index('Period_Label')['mean']
            period_median = period_stats.set_index('Period_Label')['median']

            frequency_label = 'Month / Week of Month'

            # Determine current period label
            current_month_num = datetime.now().month
            current_week_of_month = (datetime.now().day - 1) // 7 + 1
            current_period_num = current_month_num * 10 + current_week_of_month
            current_period_label_array = period_stats.loc[period_stats['Period_Num'] == current_period_num, 'Period_Label'].values
            current_period_label = current_period_label_array[0] if len(current_period_label_array) > 0 else None

            weekly_data_current_year = weekly_data.loc[weekly_data.index.year == datetime.now().year]
            if not weekly_data_current_year.empty:
                current_returns = weekly_data_current_year.set_index('Period_Label')['Return']
                current_returns = current_returns[~current_returns.index.duplicated(keep='last')].reindex(period_mean.index)
            
    
        elif frequency == 'quarterly':
            frequency_label = 'Quarter'
            periods = ['Q1', 'Q2', 'Q3', 'Q4']
            current_quarter = (datetime.now().month - 1) // 3 + 1
            current_period_label = f'Q{current_quarter}'

            # Calculate mean and median returns for each quarter
            period_mean = data.groupby(data.index.quarter).mean().reindex(range(1, 5))
            period_median = data.groupby(data.index.quarter).median().reindex(range(1, 5))
            # Map period indices to quarters
            period_mean.index = [f'Q{i}' for i in period_mean.index]
            period_median.index = [f'Q{i}' for i in period_median.index]

            current_year_returns = data.loc[data.index.year == datetime.now().year]
            if not current_year_returns.empty:
                current_returns = current_year_returns.groupby(current_year_returns.index.quarter).last().reindex(range(1, 5))
                current_returns.index = [f'Q{i}' for i in current_returns.index]

        elif frequency == 'daily':
            # Group by month and day in MM-DD format
            frequency_label = 'Day (MM/DD)'
            periods = sorted(data.index.strftime('%m/%d').unique())
            current_day = datetime.now().strftime('%m/%d')
            current_period_label = current_day if current_day in periods else None

            # Calculate mean and median returns for each day
            period_mean = data.groupby(data.index.strftime('%m-%d')).mean()
            period_median = data.groupby(data.index.strftime('%m-%d')).median()
            # Map period indices to MM/DD format
            period_mean.index = periods
            period_median.index = periods

            current_year_returns = data.loc[data.index.year == datetime.now().year].copy()
            if not current_year_returns.empty:
                current_year_returns.index = current_year_returns.index.strftime('%m/%d')
                current_returns = current_year_returns
            
            window_size = 30
            if current_day in period_mean.index:
                current_idx = period_mean.index.get_loc(current_day)
                start_idx = current_idx - window_size
                end_idx = current_idx + window_size + 1  # +1 to include the end day

                # Handle wrap-around
                if start_idx < 0:
                    period_mean_window = pd.concat([period_mean.iloc[start_idx:], period_mean.iloc[:end_idx]])
                    period_median_window = pd.concat([period_median.iloc[start_idx:], period_median.iloc[:end_idx]])
                elif end_idx > len(period_mean):
                    period_mean_window = pd.concat([period_mean.iloc[start_idx:], period_mean.iloc[:end_idx - len(period_mean)]])
                    period_median_window = pd.concat([period_median.iloc[start_idx:], period_median.iloc[:end_idx - len(period_median)]])
                else:
                    period_mean_window = period_mean.iloc[start_idx:end_idx]
                    period_median_window = period_median.iloc[start_idx:end_idx]
            else:
                # If current_day not in periods, display the entire year
                period_mean_window = period_mean
                period_median_window = period_median

            period_mean = period_mean_window
            period_median = period_median_window
            if current_returns is not None:
                current_returns = current_returns.reindex(period_mean.index)
            

        
    
        elif frequency == 'yearly':
            group_by = data.index.year
            frequency_label = 'Year'
            periods = sorted(data.index.year.unique().astype(str))
            current_year = str(datetime.now().year)
            current_period_label = current_year if current_year in periods else None

            # Calculate mean and median returns for each year
            period_mean = data.groupby(group_by).mean()
            period_median = data.groupby(group_by).median()
            period_mean.index = periods
            period_median.index = periods

            current_returns = data.loc[data.index.year == datetime.now().year]
            if not current_returns.empty:
                current_returns = current_returns.groupby(current_returns.index.year).last()
                current_returns.index = current_returns.index.astype(str)
                current_returns = current_returns.reindex(period_mean.index)

        else:
            raise ValueError("Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.")

        # Determine colors for bars
        default_bar_color = '#3B82F6'
        highlight_bar_color = '#F59E0B'
        if current_period_label and current_period_label in period_mean.index:
            colors = [highlight_bar_color if period == current_period_label else default_bar_color for period in period_mean.index]
        else:
            colors = [highlight_bar_color if period == period_mean.index[-1] else default_bar_color for period in period_mean.index]

        # Create the figure
        fig = go.Figure()

        # Add bar trace for mean returns
        fig.add_trace(go.Bar(
            x=period_mean.index,
            y=period_mean.values,
            name='Mean Return',
            marker_color=colors,
            hovertemplate='Mean: %{y:.4f}<extra></extra>'
        ))

        median_stem_x = []
        median_stem_y = []
        for period, value in zip(period_median.index, period_median.values):
            if pd.isna(value):
                continue
            median_stem_x.extend([period, period, None])
            median_stem_y.extend([0, value, None])

        fig.add_trace(go.Scatter(
            x=median_stem_x,
            y=median_stem_y,
            mode='lines',
            name='Median Stem',
            line=dict(color='red', width=2),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Add lollipop head trace for median returns
        fig.add_trace(go.Scatter(
            x=period_median.index,
            y=period_median.values,
            mode='markers',
            name='Median Return',
            marker=dict(size=9, color='red'),
            hovertemplate='Median: %{y:.4f}<extra></extra>'
        ))

        if current_returns is not None and pd.Series(current_returns).notna().any():
            current_returns = pd.Series(current_returns).reindex(period_mean.index)
            fig.add_trace(go.Scatter(
                x=current_returns.index,
                y=current_returns.values,
                mode='lines+markers',
                name=f'{datetime.now().year} Return',
                line=dict(color='#22C55E', width=2),
                marker=dict(size=10, color='#22C55E', symbol='diamond'),
                hovertemplate='Current Year: %{y:.4f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=frequency_label,
            yaxis_title='Return',
            xaxis_tickangle=-45,
            xaxis=dict(type='category'),
            template='plotly_dark',
            legend=dict(title='Metrics'),
            hovermode='x unified'
        )

        return fig
    
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

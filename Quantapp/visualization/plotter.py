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

class Plotter:
    def __init__(self):
        pass

    def create_side_by_side_subplots(self,fig1, fig2):
        fig = make_subplots(rows=1, cols=2, subplot_titles=(fig1.layout.title.text, fig2.layout.title.text))
        
        for trace in fig1.data:
            fig.add_trace(trace,row=1,col=1)

        for trace in fig2.data:
            fig.add_trace(trace,row=1,col=2)
        
        return fig
    
    def add_fig_to_subplot(self,fig, traces, layout, row, col):
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)
        
        # Transfer annotations
        for annotation in layout.annotations:
            fig.add_annotation(annotation.update(xref='paper', yref='paper', x=(col-1)*0.5 + 0.25, y=1 - (row-1)*0.5 - 0.25))

        # Transfer shapes
        for shape in layout.shapes:
            fig.add_shape(shape.update(xref='paper', yref='paper'))
        
        # Add vertical line to current month
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_date = datetime.now()
        current_month = months[current_date.month - 1]   

    def plot_seasonality(self, data, title, frequency='monthly'):
        
        """Plot seasonality of returns using Plotly.
        Parameters:
        - data: pd.Series or pd.DataFrame with DateTimeIndex and return values.
        - title: Title of the plot.
        - frequency: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.
        Returns:
        - fig: Plotly Figure object.
        """
        
        # Ensure the index is a DateTimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        if frequency == 'monthly':
            group_by = data.index.month
            frequency_label = 'Month'
            periods = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            current_period_label = datetime.now().strftime('%b')

            # Calculate mean and median returns for each month
            period_mean = data.groupby(group_by).mean()
            period_median = data.groupby(group_by).median()
            # Map period indices to month abbreviations
            period_mean.index = periods
            period_median.index = periods
            
            #calculate the returns of each month this year up to the current month
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            current_returns = data.loc[data.index.year == current_year]
            current_returns = current_returns.loc[current_returns.index.month <= current_month]
            
            

        elif frequency == 'weekly':
            # Resample data to weekly frequency (weeks starting on Monday)
            weekly_data = data.resample('W-MON').mean()
            weekly_data = weekly_data.to_frame(name='Return')

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
            
            
            # Calculate the current returns of each week for the current year
            current_year = datetime.now().year

            # Filter data for the current year and resample to weekly frequency
            weekly_data_current_year = data.loc[data.index.year == current_year].resample('W-MON').mean()
            weekly_data_current_year = weekly_data_current_year.to_frame(name='Return')

            # Assign month number and week of month
            weekly_data_current_year['Month_Num'] = weekly_data_current_year.index.month
            weekly_data_current_year['Month_Name'] = weekly_data_current_year.index.strftime('%b')
            weekly_data_current_year['Week_of_Month'] = weekly_data_current_year.index.to_series().apply(lambda d: (d.day - 1) // 7 + 1)

            # Create period labels in the format 'Month / Week X'
            weekly_data_current_year['Period_Label'] = weekly_data_current_year['Month_Name'] + ' / Week ' + weekly_data_current_year['Week_of_Month'].astype(str)

            # Create a numerical representation for sorting
            weekly_data_current_year['Period_Num'] = weekly_data_current_year['Month_Num'] * 10 + weekly_data_current_year['Week_of_Month']

            # Extract current returns and set index to 'Period_Label'
            current_returns = weekly_data_current_year.set_index('Period_Label')['Return']

            # Reindex current_returns to match period_mean index, filling missing values with NaN
            current_returns = current_returns.reindex(period_mean.index)
            
    
        elif frequency == 'quarterly':
            group_by = data.index.quarter
            frequency_label = 'Quarter'
            periods = ['Q1', 'Q2', 'Q3', 'Q4']
            current_quarter = (datetime.now().month - 1) // 3 + 1
            current_period_label = f'Q{current_quarter}'

            # Calculate mean and median returns for each quarter
            period_mean = data.groupby(group_by).mean()
            period_median = data.groupby(group_by).median()
            # Map period indices to quarters
            period_mean.index = [f'Q{i}' for i in period_mean.index]
            period_median.index = [f'Q{i}' for i in period_median.index]
            
            #calculate the returns of quarters this year up to the current quarter
            current_year = datetime.now().year
            current_quarter = (datetime.now().month - 1) // 3 + 1
            
            current_returns = data.loc[data.index.year == current_year]
            current_returns = current_returns.loc[current_returns.index.quarter <= current_quarter]
            
            
        elif frequency == 'daily':
            # Group by month and day in MM-DD format
            group_by = data.index.strftime('%m-%d')
            frequency_label = 'Day (MM/DD)'
            periods = sorted(data.index.strftime('%m/%d').unique())
            current_day = datetime.now().strftime('%m/%d')
            current_period_label = current_day if current_day in periods else None

            # Calculate mean and median returns for each day
            period_mean = data.groupby(group_by).mean()
            period_median = data.groupby(group_by).median()
            # Map period indices to MM/DD format
            period_mean.index = periods
            period_median.index = periods
            
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

        else:
            raise ValueError("Invalid frequency. Choose 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.")

        # Determine colors for bars
        if current_period_label and current_period_label in period_mean.index:
            colors = ['red' if period == current_period_label else 'blue' for period in period_mean.index]
        else:
            colors = ['red' if period == period_mean.index[-1] else 'blue' for period in period_mean.index]

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

        # Add scatter trace for median returns
        fig.add_trace(go.Scatter(
            x=period_median.index,
            y=period_median.values,
            mode='lines+markers',
            name='Median Return',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            hovertemplate='Median: %{y:.4f}<extra></extra>'
        ))
        
        # Add scatter trace for current year returns
        # make sure the index matches the index of the average returns
        # dont plot as a line, plot as small gray markers in the shape of a line 
        '''       
        if 'current_returns' in locals():
            fig.add_trace(go.Scatter(
            x=period_mean.index,
            y=current_returns,
            mode='markers',
            name='Current Year Returns',
            marker=dict(
                symbol='line-ew',  # Use 'line-ns' for vertical lines
                size=12,           # Adjust size as needed
                color='#FFFF00',    #bright yellow is #FFFF00
                line=dict(
                width=2        # Line width of the marker
                )
            ),
            hovertemplate='Current Year: %{y:.4f}<extra></extra>'
            ))
        '''
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

    def create_spread_plot(self, asset_spreads, title='Placeholder / Title', default_years=10):
        import plotly.graph_objects as go

        if isinstance(asset_spreads, pd.Series):
            spreads = {'Spread': asset_spreads}
        elif isinstance(asset_spreads, Mapping):
            spreads = dict(asset_spreads)
        elif isinstance(asset_spreads, (list, tuple)):
            spreads = {name: series for name, series in asset_spreads}
        else:
            raise TypeError("asset_spreads must be a Series, mapping, or list/tuple of (name, Series).")

        normalized_spreads = {}
        for label, series in spreads.items():
            if not isinstance(series, pd.Series):
                raise TypeError(f"Spread '{label}' is not a pandas Series.")
            if not isinstance(series.index, pd.DatetimeIndex):
                raise TypeError(f"Spread '{label}' must have a DatetimeIndex.")
            series = series.sort_index()
            if series.dropna().empty:
                continue
            normalized_spreads[label] = series

        if not normalized_spreads:
            raise ValueError("No valid spreads with data were provided.")

        spreads = normalized_spreads
        combined_index = None
        for series in spreads.values():
            combined_index = series.index if combined_index is None else combined_index.union(series.index)

        global_start = combined_index.min()
        global_end = combined_index.max()

        range_start = global_end - pd.DateOffset(years=default_years)
        if range_start < global_start:
            range_start = global_start

        default_view_years = 3
        default_view_start = global_end - pd.DateOffset(years=default_view_years)
        if default_view_start < global_start:
            default_view_start = global_start

        fig = go.Figure()
        layout_options = {}
        visibility_matrix = []
        labels = list(spreads.keys())

        for idx, (label, series) in enumerate(spreads.items()):
            filtered = series.loc[range_start:global_end]
            fig.add_trace(
                go.Scatter(
                    x=filtered.index,
                    y=filtered.values,
                    mode='lines',
                    name=label,
                    visible=(idx == 0)
                )
            )

            visibility_row = [False] * len(spreads)
            visibility_row[idx] = True
            visibility_matrix.append(visibility_row)

            cleaned = filtered.dropna()
            if cleaned.empty:
                layout_options[label] = {"shapes": [], "annotations": []}
                continue

            positive = cleaned[cleaned >= 0]
            if positive.empty:
                positive = cleaned

            mean = positive.mean()
            std_dev = positive.std()
            if pd.isna(mean):
                mean = 0.0
            if pd.isna(std_dev):
                std_dev = 0.0

            spread_max = cleaned.max()
            spread_min = cleaned.min()
            x_min = cleaned.index.min()
            x_max = cleaned.index.max()
            middle_idx = cleaned.index[len(cleaned) // 2]

            shapes = [
                dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
                    y0=mean - std_dev, y1=mean + std_dev,
                    fillcolor="grey", opacity=0.2, line=dict(width=0)),
                dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
                    y0=mean + std_dev, y1=spread_max,
                    fillcolor="limegreen", opacity=0.3, line=dict(width=0)),
                dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
                    y0=spread_min, y1=mean - 2 * std_dev,
                    fillcolor="orangered", opacity=0.3, line=dict(width=0)),
                dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=mean, y1=mean,
                    line=dict(color="orange", width=2)),
                dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=mean + std_dev, y1=mean + std_dev,
                    line=dict(color="blue", width=2, dash="dash")),
                dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=mean - std_dev, y1=mean - std_dev,
                    line=dict(color="blue", width=2, dash="dash")),
                dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=mean + 2 * std_dev, y1=mean + 2 * std_dev,
                    line=dict(color="lightgreen", width=2, dash="dot")),
                dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=mean - 2 * std_dev, y1=mean - 2 * std_dev,
                    line=dict(color="lightgreen", width=2, dash="dot")),
            ]

            annotations = [
                dict(x=x_max, y=mean, text="Mean", showarrow=True, arrowhead=1,
                    ax=-60, ay=0, arrowcolor="orange", xref="x", yref="y"),
                dict(x=x_max, y=mean + std_dev, text="Mean + 1 Std Dev", showarrow=True, arrowhead=1,
                    ax=-120, ay=0, arrowcolor="blue", xref="x", yref="y"),
                dict(x=x_max, y=mean - std_dev, text="Mean - 1 Std Dev", showarrow=True, arrowhead=1,
                    ax=-120, ay=0, arrowcolor="blue", xref="x", yref="y"),
                dict(x=x_max, y=mean + 2 * std_dev, text="Mean + 2 Std Dev", showarrow=True, arrowhead=1,
                    ax=-120, ay=0, arrowcolor="lightgreen", xref="x", yref="y"),
                dict(x=x_max, y=mean - 2 * std_dev, text="Mean - 2 Std Dev", showarrow=True, arrowhead=1,
                    ax=-120, ay=0, arrowcolor="lightgreen", xref="x", yref="y"),
                dict(x=middle_idx, y=mean - std_dev, text="Neutral", showarrow=True, arrowhead=1,
                    ax=0, ay=-50, arrowcolor="grey", xref="x", yref="y", font=dict(size=14)),
                dict(x=middle_idx, y=mean + 2 * std_dev, text="Buy", showarrow=True, arrowhead=1,
                    ax=0, ay=-50, arrowcolor="limegreen", xref="x", yref="y", font=dict(size=14)),
                dict(x=middle_idx, y=mean - 2 * std_dev * 2, text="Sell", showarrow=True, arrowhead=1,
                    ax=0, ay=50, arrowcolor="orangered", xref="x", yref="y", font=dict(size=14)),
            ]

            layout_options[label] = {"shapes": shapes, "annotations": annotations}

        def _range(years):
            start = global_end - pd.DateOffset(years=years)
            if start < global_start:
                start = global_start
            return [start, global_end]

        spread_buttons = []
        for idx, label in enumerate(labels):
            opts = layout_options.get(label, {"shapes": [], "annotations": []})
            spread_buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visibility_matrix[idx]},
                        {
                            "title": f"{title} - {label}" if len(labels) > 1 else title,
                            "shapes": copy.deepcopy(opts["shapes"]),
                            "annotations": copy.deepcopy(opts["annotations"]),
                        },
                    ],
                )
            )

        range_buttons = [
            dict(label="Max", method="relayout", args=[{"xaxis.range": [global_start, global_end]}]),
            dict(label="10 Years", method="relayout", args=[{"xaxis.range": _range(10)}]),
            dict(label="5 Years", method="relayout", args=[{"xaxis.range": _range(5)}]),
            dict(label="3 Years", method="relayout", args=[{"xaxis.range": _range(3)}]),
            dict(label="1 Year", method="relayout", args=[{"xaxis.range": _range(1)}]),
        ]

        spread_menu = dict(
            buttons=spread_buttons,
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.18,
            yanchor="top",
        )

        range_menu = dict(
            buttons=range_buttons,
            direction="down",
            showactive=True,
            x=0.25,
            xanchor="left",
            y=1.18,
            yanchor="top",
        )

        initial_label = labels[0]
        initial_opts = layout_options.get(initial_label, {"shapes": [], "annotations": []})

        fig.update_layout(
            height=1000,
            title=f"{title} - {initial_label}" if len(labels) > 1 else title,
            template='plotly_dark',
            xaxis_title="Date",
            yaxis_title="Spread",
            xaxis=dict(range=[default_view_start, global_end]),
            updatemenus=[spread_menu, range_menu],
            shapes=copy.deepcopy(initial_opts["shapes"]),
            annotations=copy.deepcopy(initial_opts["annotations"]),
            showlegend=False,
        )

        return fig
    
    def add_recession_bands(self,fig, nber_series, fillcolor='grey', opacity=0.3):
        """Add NBER recession shading to an existing Plotly figure, aligned to the figure's x-axis."""
        if fig is None:
            raise ValueError("fig must be a Plotly Figure.")
        recession = nber_series.rename('Recession').copy()
        if not isinstance(recession.index, pd.DatetimeIndex):
            recession.index = pd.to_datetime(recession.index)

        # Extract x-axis range from the figure
        xaxis = fig.layout.xaxis
        x_start = pd.to_datetime(xaxis.range[0]) if xaxis.range else recession.index.min()
        x_end = pd.to_datetime(xaxis.range[1]) if xaxis.range else recession.index.max()

        # Filter recession data to match the figure's x-axis range
        filtered_recession = recession[(recession.index >= x_start) & (recession.index <= x_end)]

        in_recession = False
        start = None
        for timestamp, value in filtered_recession.items():
            if value == 1 and not in_recession:
                in_recession = True
                start = timestamp
            elif value == 0 and in_recession:
                fig.add_vrect(
                    x0=start,
                    x1=timestamp,
                    fillcolor=fillcolor,
                    opacity=opacity,
                    layer="below",
                    line_width=0
                )
                in_recession = False

        if in_recession and start is not None:
            end = filtered_recession.index.max()
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=fillcolor,
                opacity=opacity,
                layer="below",
                line_width=0
            )
            
        return fig
    
    def plot_percentage_drop(self,data, n=14, title='Percentage Drop from Highest Peak'):
        """
        Plot the percentage drop from the highest peak using Plotly, including lines for the mean percentage drop
        and negative standard deviation levels. Color bars blue if they fall below the mean, and red if below 0.5 
        standard deviations below the mean.
        
        Parameters:
        - daily_close: pd.DataFrame with 'Close', 'HighestHigh', and 'PercentageDrop' columns.
        - n: Number of days to plot (the most recent n days).
        - title: Title of the plot.
        """
        # Ensure the index is a DateTimeIndex
        
        percentage_drop = rolling.calculate_percentage_drop(data, n=n)['PercentageDrop']
     
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Compute statistics on the entire dataset
        mean_percentage_drop = percentage_drop.mean()
        std_dev = percentage_drop.std()
        
        # Slice the dataframe to get the last n days
        data = data.tail(n)
        
        # Define bar colors based on the percentage drop
        colors = ['red' if drop < mean_percentage_drop - 0.5 * std_dev
                else 'blue' if drop < mean_percentage_drop +.25 * std_dev
                else 'green' for drop in percentage_drop]
        
        # Create the bar chart
        fig = go.Figure()
        
        # Add the bars to the plot
        fig.add_trace(go.Bar(
            x=data.index,
            y=percentage_drop,
            marker_color=colors,
            name='Percentage Drop'
        ))
        
        # Add horizontal lines for mean and standard deviations (computed using the entire dataset)
        fig.add_trace(go.Scatter(
            x=[data.index.min(), data.index.max()],
            y=[mean_percentage_drop, mean_percentage_drop],
            mode='lines',
            line=dict(color='blue', dash='dash', width=2),
            name='Mean Percentage Drop',
            visible='legendonly'  # This makes the line hidden by default
        ))
        
        # Add horizontal lines for mean percentage drop + 0.25 standard deviations
        fig.add_trace(go.Scatter(
            x=[data.index.min(), data.index.max()],
            y=[mean_percentage_drop + 0.25 * std_dev, mean_percentage_drop + 0.25 * std_dev],
            mode='lines',
            line=dict(color='purple', dash='dash', width=1),
            name='Mean + 0.25 Std Dev'
        ))
        
        # Add horizontal lines for mean percentage drop - 0.5 standard deviations   
        fig.add_trace(go.Scatter(
            x=[data.index.min(), data.index.max()],
            y=[mean_percentage_drop - 0.5 * std_dev, mean_percentage_drop - 0.5 * std_dev],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name='Mean - 0.5 Std Dev'
        ))
        
        #add horizontal lines for mean percentage drop - 1 standard deviation
        fig.add_trace(go.Scatter(
            x=[data.index.min(), data.index.max()],
            y=[mean_percentage_drop - 1.0 * std_dev, mean_percentage_drop - 1.0 * std_dev],
            mode='lines',
            line=dict(color='purple', dash='dash', width=1),
            name='Mean - 1 Std Dev',
            visible='legendonly'  # This makes the line hidden by default
        ))
        
        
        # Add annotations for color meanings
        fig.add_annotation(
            x=data.index[int(len(data) * 0.1)],  # Position on the x-axis
            y=mean_percentage_drop + 0.5 * std_dev,  # Position on the y-axis
            text="Green: Bullish",
            showarrow=False,
            font=dict(size=12, color="green"),
            align="center"
        )
        fig.add_annotation(
            x=data.index[int(len(data) * 0.1)],  # Position on the x-axis
            y=mean_percentage_drop,  # Position on the y-axis
            text="Blue: Neutral",
            showarrow=False,
            font=dict(size=12, color="blue"),
            align="center"
        )
        fig.add_annotation(
            x=data.index[int(len(data) * 0.1)],  # Position on the x-axis
            y=mean_percentage_drop - 0.75 * std_dev,  # Position on the y-axis
            text="Red: Bearish",
            showarrow=False,
            font=dict(size=12, color="red"),
            align="center"
        )
        # Update layout to remove gaps by treating dates as categories
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Percentage Drop',
            xaxis=dict(
                type='category',  # Treat x-axis as categorical to prevent gaps
                tickangle=-45,
                showgrid=True,
                zeroline=False
            ),
            barmode='overlay'
        )
        
        # Show the plot
        fig.show()
        
    def plot_series_with_stdev_bands(
        self,
        data_series,
        stdev_values=[-0.5, 0.5, 1.5, 3],
        num_years=5,
        title="Series with Mean & Standard Deviations"
    ):
        """
        Plots a given data series, adding horizontal lines for mean and multiple standard deviations.
        Shades the regions between standard deviation bands with distinct colors:
        - Between -0.5 and 0.5 standard deviations: Green
        - Between 0.5 and 1.5 standard deviations: Yellow
        - Between 1.5 and 3 standard deviations: Red

        Parameters:
        - data_series (pd.Series): Any precomputed series of values to plot.
        - stdev_values (list of float): Multipliers for standard deviations to plot (e.g., [-0.5, 0.5, 1.5, 3]).
        - num_years (int): Number of years to zoom in on the chart.
        - title (str): Chart title.
        """
        # Filter data to the specified number of years
        zoom_start = data_series.index[-1] - pd.DateOffset(years=num_years)
        zoom_data = data_series.loc[data_series.index >= zoom_start]

        fig = go.Figure()

        # Plot data_series
        fig.add_trace(go.Scatter(
            x=data_series.index,
            y=data_series,
            mode='lines',
            name='Data',
            line=dict(color='yellow')
        ))

        # Compute mean and std
        mean_val = data_series.mean()
        std_val = data_series.std()

        # Add horizontal line for mean
        fig.add_hline(
            y=mean_val,
            line_color="white",
            line_dash="dash",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="bottom right"
        )

        # Add horizontal lines for each standard deviation
        for stdev in stdev_values:
            sd_line = mean_val + stdev * std_val
            fig.add_hline(
                y=sd_line,
                line_color="white",
                line_dash="dot",
                annotation_text=f"{stdev} SD: {sd_line:.2f}",
                annotation_position="bottom right"
            )

        # Define colors for each shading band
        shade_colors = [
            "rgba(0, 255, 0, 0.3)",    # Green for -0.5 to 0.5
            "rgba(255, 255, 0, 0.5)",  # Yellow for 0.5 to 1.5
            "rgba(255, 0, 0, 0.7)"     # Red for 1.5 to 3
        ]

        # Sort stdev_values for consistent shading
        stdev_values_sorted = sorted(stdev_values)

        # Shade regions between consecutive standard deviation bands
        for i in range(len(stdev_values_sorted) - 1):
            lower_stdev = stdev_values_sorted[i]
            upper_stdev = stdev_values_sorted[i + 1]
            y0 = mean_val + lower_stdev * std_val
            y1 = mean_val + upper_stdev * std_val
            color = shade_colors[i] if i < len(shade_colors) else "rgba(255, 0, 0, 0.7)"

            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=data_series.index.min(),
                y0=y0,
                x1=data_series.index.max(),
                y1=y1,
                fillcolor=color,
                layer="below",
                line_width=0
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_dark',
            height=800,
            xaxis=dict(
                # Set range to the latest num_years only
                range=[zoom_start, data_series.index[-1]]
            ),
            # Adjust y-axis range based on filtered data
            yaxis=dict(
                range=[zoom_data.min(), zoom_data.max()]
            ),
        )

        fig.show()
            
    def plot_candlestick(self,ticker_data, drop_window=14, period='1Y', bollinger_window=21, title="Candlestick With Bollinger Bands"):
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
        ticker_data = rolling.calculate_percentage_drop(ticker_data, n=drop_window)
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
            height=900,
            template='plotly_dark',
            yaxis=dict(autorange=True, fixedrange=False),
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False
            )
        )

        fig.show()
    
    def create_candlestick_chart(self, df, title='Candlestick Chart with SMAs'):
        # Calculate the Simple Moving Averages (SMAs)
        df['SMA21'] = df['Close'].rolling(window=21).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()

        # Create the candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'],
                                            name='Candlesticks')])

        # Add the 21, 50, and 200-day SMAs to the chart
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA21'],
                                mode='lines', line=dict(color='red', width=2),
                                name='21-day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                                mode='lines', line=dict(color='violet', width=2),  # Brighter purple
                                name='50-day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'],
                                mode='lines', line=dict(color='yellow', width=2),
                                name='200-day SMA'))

        # Add alternating gray background for each month with subtle opacity
        months = pd.to_datetime(df.index).to_period("M")
        for i, (month, group) in enumerate(df.groupby(months)):
            if i % 2 == 0:  # Shade every other month
                fig.add_shape(
                    type="rect",
                    x0=group.index[0],
                    x1=group.index[-1],
                    y0=0,  # Extend from the bottom of the graph
                    y1=1,  # Extend to the top of the graph
                    xref='x',  # x-axis reference
                    yref='paper',  # y-axis reference as the paper (full height of the plot)
                    fillcolor="rgba(200, 200, 200, 0.1)",  # Lighter gray with 10% opacity
                    line=dict(width=0),  # No border line
                    layer="below"  # Ensure it appears behind the candlesticks
                )

        # Add vertical lines for each year
        years = pd.date_range(start=df.index.min().replace(month=1, day=1), end=df.index.max(), freq='Y')
        for year in years:
            fig.add_vline(x=year, line_width=2, line_dash="dash", line_color="lightgray", name='Year')

        # Add vertical lines for each fiscal quarter end
        quarters = pd.date_range(start=df.index.min(), end=df.index.max(), freq='Q')
        for quarter in quarters:
            fig.add_vline(x=quarter, line_width=1, line_dash="dot", line_color="gray", name='Quarter')

        # Update x-axes to hide weekends and specific dates
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(values=["2021-01-01", "2021-12-25"])  # Hide Christmas
            ]
        )

        # Apply the Plotly dark theme and update layout
        fig.update_layout(
            title=title,
            template="plotly_dark",  # Apply the dark theme
            height=800,  # Set the height to make the chart taller
            yaxis_title='Price',
            xaxis_title='Date'
        )

        # Define the time frame options for the dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=[{"xaxis.range": [df.index.min(), df.index.max()]}],
                            label="Max",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [df.index.max() - pd.DateOffset(years=10), df.index.max()]}],
                            label="10 Years",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [df.index.max() - pd.DateOffset(years=5), df.index.max()]}],
                            label="5 Years",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [df.index.max() - pd.DateOffset(years=3), df.index.max()]}],
                            label="3 Years",
                            method="relayout"
                        ),
                        dict(
                            args=[{"xaxis.range": [df.index.max() - pd.DateOffset(years=1), df.index.max()]}],
                            label="1 Year",
                            method="relayout"
                        ),
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.15,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ]
        )

        return fig
    
    def plot_combined_table(self, df, title='Combined Sortino Indicators'):
        """
        Plots a combined table with indicators for Sortino differences and standard deviation categories,
        highlighting the direction of deviations (positive or negative).
        
        Parameters:
            df (pd.DataFrame): Combined DataFrame with 'Ticker', Sortino differences, and deviation categories.
            title (str): Title of the table.
        """
        # Define color mappings for positive and negative deviations
        positive_deviation_colors = {
            '>+3 SD': 'lightgreen',
            '+2-3 SD': 'yellow',
            '+1-2 SD': 'orange',
            '+<1 SD': 'white'
        }
        
        negative_deviation_colors = {
            '<-3 SD': 'lightcoral',
            '-2-3 SD': 'coral',
            '-1-2 SD': 'lightblue',
            '-<1 SD': 'white'
        }
        
        # Initialize fill colors based on deviation categories and Sortino differences
        fill_colors = []
        for _, row in df.iterrows():
            row_colors = []
            for col in df.columns:
                if col == 'Ticker':
                    row_colors.append('lightgrey')  # Default color for Ticker column
                elif 'Relative performance' in col:
                    if row[col]:  # Underperforming if True
                        row_colors.append('lightgreen')  # Highlight underperforming assets
                    else:
                        row_colors.append('white')        # Default color
                else:
                    deviation = row[col]
                    if deviation.startswith('+'):
                        # Positive Deviation
                        color = positive_deviation_colors.get(deviation, 'white')
                    elif deviation.startswith('-'):
                        # Negative Deviation
                        color = negative_deviation_colors.get(deviation, 'white')
                    else:
                        color = 'white'  # Default color for any other case
                    row_colors.append(color)
            fill_colors.append(row_colors)
        
        # Transpose fill_colors to match Plotly's column-wise format
        fill_colors_transposed = list(map(list, zip(*fill_colors)))
        
        # Replace boolean values with descriptive text for Sortino differences
        display_df = df.copy()
        for col in df.columns:
            if 'Relative performance' in col:
                display_df[col] = display_df[col].apply(lambda x: 'Underperforming' if x else 'Overperforming')
        
        # Create the Plotly table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>' + col.replace('_', ' ') + '</b>' for col in display_df.columns],
                fill_color='paleturquoise',
                align='center',
                font=dict(color='black', size=12)
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=fill_colors_transposed,
                align='center',
                font=dict(color='black', size=11)
            )
        )])
        
        # Update layout for aesthetics
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=800,
            margin=dict(l=50, r=50, t=80, b=200)  # Increased bottom margin for legend
        )
        
        # Add a comprehensive legend using annotations
        legend_text = (
            "<b>Legend:</b><br>"
            "<b>Deviation Directions:</b><br>"
            "Light Green: >+3 SD (Significantly Above Mean)<br>"
            "Yellow: +2-3 SD (Above Mean)<br>"
            "Orange: +1-2 SD (Slightly Above Mean)<br>"
            "Light Coral: <-3 SD (Significantly Below Mean)<br>"
            "Coral: -2-3 SD (Below Mean)<br>"
            "Light Blue: -1-2 SD (Slightly Below Mean)<br>"
            "White: Within 1 SD<br><br>"
            "<b>Performance Indicators:</b><br>"
            "Light Green: Underperforming<br>"
            "White: Overperforming"
        )
        
        fig.add_annotation(
            text=legend_text,
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.3,
            xanchor='center',
            yanchor='top',
            font=dict(color='black', size=12)
        )
        
        # Show the table
        fig.show()
    
    def plot_time_series(self, all_series, time_frame='1y', title='Time Series Data of Ticker Symbols'):
        """
        Plots the time series data for a DataFrame where each column is a ticker symbol and each row is a price.

        Parameters:
            all_series (pd.DataFrame): DataFrame containing time series data with ticker symbols as columns and prices as rows.
            title (str): Title for the plot (default is 'Time Series Data of Ticker Symbols').
        """
        # Filter out the data based on the specified time frame
        if time_frame == '1y':
            all_series = all_series.loc[all_series.index >= all_series.index[-1] - pd.DateOffset(years=1)]
        elif time_frame == '3y':
            all_series = all_series.loc[all_series.index >= all_series.index[-1] - pd.DateOffset(years=3)]
        elif time_frame == '5y':
            all_series = all_series.loc[all_series.index >= all_series.index[-1] - pd.DateOffset(years=5)]
        elif time_frame == '10y':
            all_series = all_series.loc[all_series.index >= all_series.index[-1] - pd.DateOffset(years=10)]
        else:
            # Error handling for invalid time frame
            print("Error: Invalid time frame")
            return
        
        # Create a Plotly figure
        fig = px.line(all_series, title=title)
        
        # Add a dashed horizontal line at zero
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        
        # Update layout for the figure
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            xaxis=dict(
                tickangle=-45,
                showgrid=True,
                zeroline=True  # Add zero line for x-axis
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True  # Add zero line for y-axis
            )
        )

        return fig
    
    def plot_return_difference(self,sp500, frequency='daily'):
        """
        Plot the difference between the average and median returns based on the specified frequency.

        Parameters:
        - sp500: pd.DataFrame with a DateTimeIndex and 'Close' column.
        - frequency: 'daily' or 'weekly' to specify the plot frequency.
        """
        # Calculate daily returns
        sp500_daily_returns = sp500.pct_change() * 100
        
        # Add columns for day names, month/day, and week number
        sp500_daily_returns['Day of Week'] = sp500_daily_returns.index.day_name()
        sp500_daily_returns['Month/Day'] = sp500_daily_returns.index.strftime('%m/%d')
        sp500_daily_returns['Week Number'] = sp500_daily_returns.index.isocalendar().week
        
        if frequency == 'daily':
            # Calculate average and median returns for each day of the year
            daily_avg_returns = sp500_daily_returns.groupby('Month/Day')['Close'].mean()
            daily_median_returns = sp500_daily_returns.groupby('Month/Day')['Close'].median()
            
            # Calculate the difference between mean and median
            daily_diff = daily_avg_returns - daily_median_returns
            
            fig = go.Figure()
            
            # Plot the difference as bars
            fig.add_trace(go.Bar(
                x=daily_diff.index,
                y=daily_diff,
                name='Difference (Mean - Median)',
                marker_color='purple'
            ))
            
            fig.update_layout(
                title='Difference Between Average and Median Returns for Each Day of the Year',
                xaxis_title='Date',
                yaxis_title='Difference (%)',
                xaxis_tickangle=-45  # Rotate x-axis labels for readability
            )
            fig.show()
        
        elif frequency == 'weekly':
            # Calculate average and median returns for each week of the year
            weekly_avg_returns = sp500_daily_returns.groupby('Week Number')['Close'].mean()
            weekly_median_returns = sp500_daily_returns.groupby('Week Number')['Close'].median()
            
            # Calculate the difference between mean and median
            weekly_diff = weekly_avg_returns - weekly_median_returns
            
            fig = go.Figure()
            
            # Plot the difference as bars
            fig.add_trace(go.Bar(
                x=weekly_diff.index,
                y=weekly_diff,
                name='Difference (Mean - Median)',
                marker_color='purple'
            ))
            
            fig.update_layout(
                title='Difference Between Average and Median Returns for Each Week of the Year',
                xaxis_title='Week Number',
                yaxis_title='Difference (%)'
            )
            fig.show()
        
        else:
            raise ValueError("Invalid frequency. Choose 'daily' or 'weekly'.")

    def plot_average_returns(self,sp500, frequency='daily', line_style='lines'):
        
        """
        Plot average returns based on the specified frequency, including a line graph of the median returns.
        
        Parameters:
        - sp500: pd.DataFrame with a DateTimeIndex and 'Close' column.
        - frequency: 'daily', 'weekly' to specify the plot frequency.
        - line_style: 'lines' for a regular line, 'lines+markers' for a line with markers.
        """
        # Calculate daily returns
        sp500_daily_returns = sp500.pct_change() * 100
        
        # Add columns for day names, month/day, and week number
        sp500_daily_returns['Day of Week'] = sp500_daily_returns.index.day_name()
        sp500_daily_returns['Month/Day'] = sp500_daily_returns.index.strftime('%m/%d')
        sp500_daily_returns['Week Number'] = sp500_daily_returns.index.isocalendar().week
        
        # Get the current date and week number
        current_date = datetime.now()
        current_day = current_date.strftime('%m/%d')
        current_week_number = current_date.isocalendar().week
        
        if frequency == 'daily':
            # Average and median returns for each day of the year
            daily_avg_returns = sp500_daily_returns.groupby('Month/Day')['Close'].mean()
            daily_median_returns = sp500_daily_returns.groupby('Month/Day')['Close'].median()
            
            fig = go.Figure()
            
            # Add average returns as bars
            fig.add_trace(go.Bar(
                x=daily_avg_returns.index,
                y=daily_avg_returns,
                name='Average Return',
                marker_color='blue'
            ))
            
            # Add median returns as a line with or without markers
            fig.add_trace(go.Scatter(
                x=daily_median_returns.index,
                y=daily_median_returns,
                mode=line_style,  # Choose between 'lines' or 'lines+markers'
                line=dict(color='red', width=2),
                marker=dict(size=6, color='red') if line_style == 'lines+markers' else None,
                name='Median Return'
            ))
            
            # Highlight and annotate the bar for the current day
            if current_day in daily_avg_returns.index:
                fig.update_traces(
                    marker_color=[('red' if x == current_day else 'blue') for x in daily_avg_returns.index],
                    selector=dict(type='bar')
                )
                fig.add_annotation(
                    x=current_day,
                    y=daily_avg_returns[current_day],
                    text='Current Day',
                    showarrow=True,
                    arrowhead=2
                )
            
            fig.update_layout(
                title='Average and Median Returns for Each Day of the Year',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                xaxis_tickangle=-45  # Rotate x-axis labels for readability
            )
            fig.show()
        
        elif frequency == 'weekly':
            # Average and median returns for each week of the year
            weekly_avg_returns = sp500_daily_returns.groupby('Week Number')['Close'].mean()
            weekly_median_returns = sp500_daily_returns.groupby('Week Number')['Close'].median()
            
            fig = go.Figure()
            
            # Add average returns as bars
            fig.add_trace(go.Bar(
                x=weekly_avg_returns.index,
                y=weekly_avg_returns,
                name='Average Return',
                marker_color='blue'
            ))
            
            # Add median returns as a line with or without markers
            fig.add_trace(go.Scatter(
                x=weekly_median_returns.index,
                y=weekly_median_returns,
                mode=line_style,  # Choose between 'lines' or 'lines+markers'
                line=dict(color='red', width=2),
                marker=dict(size=6, color='red') if line_style == 'lines+markers' else None,
                name='Median Return'
            ))
            
            # Highlight and annotate the bar for the current week
            fig.update_traces(
                marker_color=[('red' if x == current_week_number else 'blue') for x in weekly_avg_returns.index],
                selector=dict(type='bar')
            )
            fig.add_annotation(
                x=current_week_number,
                y=weekly_avg_returns[current_week_number],
                text='Current Week',
                showarrow=True,
                arrowhead=2
            )
            
            fig.update_layout(
                title='Average and Median Returns for Each Week of the Year',
                xaxis_title='Week Number',
                yaxis_title='Return (%)'
            )
            fig.show()
        
        else:
            raise ValueError("Invalid frequency. Choose 'daily' or 'weekly'.")
    
    def plot_z_score_combined(self, z_score_combined):
        # Define the columns for the dropdown options
        columns = z_score_combined.columns.tolist()
        
        # Create the initial figure with the data sorted by the first column
        fig = go.Figure()
        
        # Add the initial table trace (sorted by first column)
        sorted_df = z_score_combined.sort_values(by=columns[0], ascending=True)
        
        # Define a function to determine cell color based on z-score value and column type
        def get_cell_color(value, column_name):
            if pd.isna(value):
                return 'white'
            
            # Check if this is a "Benchmark Minus ETF" column (inverse coloring logic)
            if "Benchmark Minus ETF" in column_name:
                # For benchmark minus ETF columns: green for above 1, red for below -0.5
                if value > 1:
                    return 'lightgreen'
                elif value < -0.5:
                    return 'lightcoral'
                else:
                    return 'white'
            else:
                # For regular sortino columns: red for above 1, green for below -0.5
                if value > 1:
                    return 'lightcoral'
                elif value < -0.5:
                    return 'lightgreen'
                else:
                    return 'white'
        
        table = go.Table(
            header=dict(
                values=['Ticker'] + columns,
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=[sorted_df.index] + [sorted_df[col] for col in columns],
                fill_color=[
                    'lightgrey',  # Ticker column color
                    # For each data column, color cells based on value and column name
                    *[[get_cell_color(val, col) for val in sorted_df[col]] for col in columns]
                ],
                align='center',
                format=[None] + ['.2f'] * len(columns)  # Format numbers to 2 decimal places
            )
        )
        
        fig.add_trace(table)
        
        # Create dropdown menu options for sorting
        buttons = []
        
        # Add buttons for each column (ascending only)
        for i, col in enumerate(columns):
            buttons.append(dict(
                args=[{
                    'cells': {
                        'values': [z_score_combined.sort_values(by=col, ascending=True).index] + 
                                [z_score_combined.sort_values(by=col, ascending=True)[c] for c in columns],
                        'fill': {
                            'color': [
                                'lightgrey',  # Ticker column color
                                # For each data column, color cells based on value and column name
                                *[[get_cell_color(val, c) for val in z_score_combined.sort_values(by=col, ascending=True)[c]] for c in columns]
                            ]
                        }
                    }
                }],
                label=f"{col} (Ascending)",
                method="update"
            ))
        
        # Update layout with dropdown menu
        fig.update_layout(
            title='Combined Z-Scores for Sortino Ratios',
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            template='plotly_white',
            height=600,
            margin=dict(l=10, r=10, t=100, b=10)  # Increased top margin for dropdown
        )
        
        # Add a color legend annotation with updated descriptions
        legend_text = (
            "Color coding for Asset Sortino Ratio:<br>" +
            "<span style='color:lightcoral'>â– </span> z > 1: Significantly above average (potential overvaluation)<br>" +
            "<span style='color:lightgreen'>â– </span> z < -0.5: Significantly below average (potential undervaluation)<br>" +
            "<br>Color coding for Benchmark Minus ETF:<br>" +
            "<span style='color:lightgreen'>â– </span> z > 1: ETF underperforming benchmark (potential buying opportunity)<br>" +
            "<span style='color:lightcoral'>â– </span> z < -0.5: ETF outperforming benchmark (potentially overvalued)<br>" 
        )
        
        fig.add_annotation(
            text=legend_text,
            showarrow=False,
            xref="paper", yref="paper",
            x=1.0, y=1.2,
            xanchor='right',
            yanchor='top',
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def plot_prices_and_returns(self,df_dict, n=200):
        """
        Plots, for each group of assets in a dictionary of DataFrames:
        - The prices in the first subplot,
        - The n-window returns in the second subplot,
        - The n-window Sharpe ratio in the third subplot.

        Provides a single dropdown to toggle which group to display.

        Args:
            df_dict (dict): A dictionary where each key is a group name and each value is a
                            pandas DataFrame with a DateTime index (prices) and columns as asset tickers.
            n (int): The window length for computing returns (periods=n in pct_change).

        Returns:
            None
        """
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create a 3-row figure:
        # 1) first row for prices,
        # 2) second row for n-window returns,
        # 3) third row for n-window Sharpe ratio.
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[
                "Prices",
                f"{n}-Window Returns",
                f"{n}-Window Sharpe Ratio"
            ]
        )

        group_names = list(df_dict.keys())
        total_traces = 0
        group_traces_visibility = []  # Will store (start_idx, end_idx) for each group

        # For each group, add 3 traces per asset:
        #   1) Price trace
        #   2) Returns trace
        #   3) Sharpe ratio trace
        for i, (group_name, df) in enumerate(df_dict.items()):
            # Calculate n-window returns
            df_returns = df.pct_change(periods=n)

            # Calculate approximate n-window Sharpe (no RF, daily frequency assumed)
            #  rolling_mean: average returns over the window
            #  volatility: std of all returns over the window
            #  ratio = sqrt(n) * rolling_mean / volatility
            rolling_mean = df_returns.rolling(window=n).mean()
            volatility = df_returns.rolling(window=n).std()
            sharpe_ratio = (rolling_mean * np.sqrt(n)) / volatility

            start_idx = total_traces
            for col in df.columns:
                # 1) Price trace (row=1)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=f"{col} ({group_name}) - Price",
                        visible=(True if i == 0 else False)
                    ),
                    row=1, col=1
                )
                total_traces += 1

                # 2) Returns trace (row=2)
                fig.add_trace(
                    go.Scatter(
                        x=df_returns.index,
                        y=df_returns[col],
                        mode='lines',
                        name=f"{col} ({group_name}) - {n}-Win Return",
                        visible=(True if i == 0 else False)
                    ),
                    row=2, col=1
                )
                total_traces += 1

                # 3) Sharpe ratio trace (row=3)
                fig.add_trace(
                    go.Scatter(
                        x=sharpe_ratio.index,
                        y=sharpe_ratio[col],
                        mode='lines',
                        name=f"{col} ({group_name}) - Sharpe",
                        visible=(True if i == 0 else False)
                    ),
                    row=3, col=1
                )
                total_traces += 1

            end_idx = total_traces - 1
            group_traces_visibility.append((start_idx, end_idx))

        # Build dropdown buttons to toggle each group's traces
        buttons = []
        for i, group_name in enumerate(group_names):
            visible_config = [False] * total_traces

            start_idx, end_idx = group_traces_visibility[i]
            # Make only this group's traces visible
            for j in range(start_idx, end_idx + 1):
                visible_config[j] = True

            buttons.append({
                "label": group_name,
                "method": "update",
                "args": [{"visible": visible_config}],
            })

        # Add the dropdown menu & layout options
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": buttons,
                    "direction": "down",
                    "showactive": True,
                }
            ],
            title="Prices, Returns & Sharpe by Asset Group",
            template="plotly_dark",
            height=2400
        )

        # Label axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Returns", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe", row=3, col=1)

        # Add a horizontal line at y=0 for returns subplot (row=2)
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y2", y0=0, y1=0,
            line=dict(color="white", dash="dash")
        )    # Add a horizontal line at y=0 for returns subplot (row=1)
        
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y3", y0=0, y1=0,
            line=dict(color="white", dash="dash")
        )
        # Show the figure
        fig.show()

    def plot_diff_from_average(self, df_dict, n=200):
        """
        Plots, for each group of assets in a dictionary of DataFrames:
        - The difference between each asset's n-window returns and the average returns in the first subplot,
        - The difference between each asset's n-window Sharpe ratio and the average Sharpe ratio in the second subplot.
        (Horizontal lines removed as requested.)

        Args:
            df_dict (dict): Dictionary of group names and DataFrames (with DateTime index and asset columns)
            n (int): Window length for computing returns (pct_change(periods=n)) and metrics.

        Returns:
            None
        """
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[
                f"{n}-Window Returns Difference (Asset - Average)",
                f"{n}-Window Sharpe Ratio Difference (Asset - Average)"
            ]
        )

        group_names = list(df_dict.keys())
        total_traces = 0
        group_traces_visibility = []

        all_returns_diff = []
        all_sharpe_diff = []

        for i, (group_name, df) in enumerate(df_dict.items()):
            df_returns = df.pct_change(periods=n)
            rolling_mean = df_returns.rolling(window=n).mean()
            volatility = df_returns.rolling(window=n).std()
            sharpe_ratio = (rolling_mean * np.sqrt(n)) / (volatility)

            avg_returns = df_returns.mean(axis=1)
            avg_sharpe = sharpe_ratio.mean(axis=1)

            start_idx = total_traces
            for col in df.columns:
                diff_returns = df_returns[col] - avg_returns
                diff_sharpe = sharpe_ratio[col] - avg_sharpe

                all_returns_diff.append(diff_returns)
                all_sharpe_diff.append(diff_sharpe)

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=diff_returns,
                        mode='lines',
                        name=f"{col} ({group_name}) - {n}-Win Return Diff",
                        visible=(True if i == 0 else False)
                    ),
                    row=1, col=1
                )
                total_traces += 1

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=diff_sharpe,
                        mode='lines',
                        name=f"{col} ({group_name}) - Sharpe Diff",
                        visible=(True if i == 0 else False)
                    ),
                    row=2, col=1
                )
                total_traces += 1

            group_traces_visibility.append((start_idx, total_traces - 1))

        buttons = []
        for i, group_name in enumerate(group_names):
            visible_config = [False] * total_traces
            start_idx, end_idx = group_traces_visibility[i]
            for j in range(start_idx, end_idx + 1):
                visible_config[j] = True
            buttons.append({
                "label": group_name,
                "method": "update",
                "args": [{"visible": visible_config}],
            })

        all_returns_diff = pd.concat(all_returns_diff).dropna()
        all_sharpe_diff = pd.concat(all_sharpe_diff).dropna()
        #add horizontal line at y=0
        fig.add_shape(
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y1", y0=0, y1=0,
            line=dict(color="white", dash="dash")
        )
        
        fig.add_shape(    
            type="line",
            xref="paper", x0=0, x1=1,
            yref="y2", y0=0, y1=0,
            line=dict(color="white", dash="dash")
        )
        
        fig.update_layout(
            updatemenus=[{
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
            }],
            title="Differences from Average by Asset Group",
            template="plotly_dark",
            height=1600
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Returns Diff", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Diff", row=2, col=1)

        fig.show()
    
    def plot_pairwise_spreads(self, pairwise_spreads_dict_by_timeframe, title="Pairwise Spreads", time_frames=None):
        """
        Creates an interactive plot showing pairwise spreads for each category and time frame with dropdown selections.
        Uses subplots to show both the spreads over time and their z-scores.
        
        Parameters:
            pairwise_spreads_dict_by_timeframe (dict): Dictionary with time frames as keys and dictionaries of category spreads as values
            title (str): Main title for the plot
            time_frames (dict): Dictionary mapping time frame keys to display names (e.g. {'short': 21})
        """
        # Get list of time frames and categories
        time_frame_keys = list(pairwise_spreads_dict_by_timeframe.keys())
        first_time_frame = time_frame_keys[0]
        categories = list(pairwise_spreads_dict_by_timeframe[first_time_frame].keys())
        first_category = categories[0]
        
        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.1,
                            subplot_titles=("Pairwise Spreads", "Z-Scores"),
                            row_heights=[0.7, 0.3])
        
        # Dictionary to track traces by their ID
        trace_indices = {}
        trace_idx = 0
        
        # Add all traces for all time frames and categories (initially hide most)
        for time_frame_key in time_frame_keys:
            for category in categories:
                df = pairwise_spreads_dict_by_timeframe[time_frame_key][category]
                
                # Store starting index for this combination
                current_combo = f"{time_frame_key}_{category}"
                trace_indices[current_combo] = []
                
                # For each spread in the category
                for spread in df.columns:
                    # Add spread line
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[spread],
                            mode="lines",
                            name=spread,
                            line=dict(width=1.5),
                            opacity=0.8,
                            visible=(time_frame_key == first_time_frame and category == first_category),
                            hovertemplate='%{y:.4f}<extra>%{fullData.name} (%{x})</extra>'
                        ),
                        row=1, col=1
                    )
                    trace_indices[current_combo].append(trace_idx)
                    trace_idx += 1
                    
                    # Calculate z-score and add bar
                    z_score = (df[spread].iloc[-1] - df[spread].mean()) / df[spread].std()
                    fig.add_trace(
                        go.Bar(
                            x=[spread],
                            y=[z_score],
                            name=f"Z-Score: {spread}",
                            text=f"{z_score:.2f}",
                            textposition='auto',
                            visible=(time_frame_key == first_time_frame and category == first_category),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    trace_indices[current_combo].append(trace_idx)
                    trace_idx += 1
        
        # Create timeframe buttons
        timeframe_buttons = []
        for time_frame_key in time_frame_keys:
            tf_days = time_frames.get(time_frame_key) if time_frames else time_frame_key
            timeframe_buttons.append(
                dict(
                    label=f"{tf_days} Days" if isinstance(tf_days, int) else time_frame_key,
                    method="update",
                    args=[
                        {"visible": [False] * trace_idx},  # Hide all traces initially
                        {"title": f"Pairwise Spreads for {first_category} ({tf_days} Days)"}
                    ]
                )
            )
            # Set visibility for the selected timeframe and first category
            visible_traces = trace_indices[f"{time_frame_key}_{first_category}"]
            for i in visible_traces:
                timeframe_buttons[-1]["args"][0]["visible"][i] = True
        
        # Create category buttons for each time frame
        category_buttons = []
        for category in categories:
            category_buttons.append(
                dict(
                    label=category,
                    method="update",
                    args=[
                        {"visible": [False] * trace_idx},  # Hide all traces initially
                        {"title": f"Pairwise Spreads for {category} ({time_frames.get(first_time_frame)} Days)"}
                    ]
                )
            )
            # Set visibility for the selected category and first time frame
            visible_traces = trace_indices[f"{first_time_frame}_{category}"]
            for i in visible_traces:
                category_buttons[-1]["args"][0]["visible"][i] = True
        
        # Update layout with dropdown menus
        fig.update_layout(
            title=f"Pairwise Spreads for {first_category} ({time_frames.get(first_time_frame)} Days)",
            template="plotly_dark",
            updatemenus=[
                # Time frame dropdown
                dict(
                    active=0,
                    buttons=timeframe_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.05,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="rgba(50, 50, 50, 0.7)",
                    font=dict(color="white"),
                    name="Time Frame"
                ),
                # Category dropdown
                dict(
                    active=0,
                    buttons=category_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.35,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    bgcolor="rgba(50, 50, 50, 0.7)",
                    font=dict(color="white"),
                    name="Category"
                )
            ],
            # Add annotations for the dropdowns
            annotations=[
                dict(
                    text="Time Frame:",
                    x=0.01,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                ),
                dict(
                    text="Category:",
                    x=0.3,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ],
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.6
            )
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
        fig.add_hline(y=2, line_dash="dot", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=-2, line_dash="dot", line_color="red", opacity=0.5, row=2, col=1)
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Spread Value", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        
        return fig
    
    def plot_etf_correlation_cointegration(self,etf_dataframes):
        """
        Plots interactive monthly correlation and cointegration charts for ETF dataframes.
        """
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create monthly resampled ETF dataframes
        etf_dataframes_monthly = {key: value.resample('M').last() for key, value in etf_dataframes.items()}

        # Create correlation matrices for each ETF dataframe using monthly data
        etf_dataframes_correlation_matrices = {key: np.log(value).diff().dropna().corr() for key, value in etf_dataframes_monthly.items()}

        # Use the first key as the default
        first_key = list(etf_dataframes_correlation_matrices.keys())[0]
        first_df = etf_dataframes_monthly[first_key]
        first_matrix = etf_dataframes_correlation_matrices[first_key]

        # Get sorted correlations
        pair_names, pair_corrs = rolling.get_sorted_correlations(first_matrix)
        pair_names = list(pair_names)  # Convert to list for reuse

        # Get cointegration p-values for the same pairs in same order
        coint_pair_names, coint_p_values = rolling.get_cointegration_pvals(first_df, pair_names)

        # Convert p-values to -log10(p) for better visualization
        log_p_values = [-np.log10(p) if p > 0 else 15 for p in coint_p_values] if coint_p_values else []

        # Create subplots: heatmap on left, correlation and cointegration bar charts on right
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5],
            specs=[[{"rowspan": 2}, {}], 
                [None, {}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            subplot_titles=('Monthly Correlation Matrix', 'Monthly Sorted Pairwise Correlations', 'Monthly Cointegration Test (-log10 p-value)')
        )

        # Add heatmap trace on left side (spanning both rows)
        heatmap = go.Heatmap(
            z=first_matrix.values,
            x=first_matrix.columns,
            y=first_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='Correlation', y=0.5, len=0.85)
        )
        fig.add_trace(heatmap, row=1, col=1)

        # Add correlation bar chart on top right
        bar = go.Bar(
            x=pair_names,
            y=pair_corrs,
            marker=dict(
                color=pair_corrs,
                colorscale='RdBu_r',
                showscale=False
            )
        )
        fig.add_trace(bar, row=1, col=2)

        # Add cointegration bar chart on bottom right
        if coint_pair_names and log_p_values:
            coint_bar = go.Bar(
                x=coint_pair_names,
                y=log_p_values,
                marker=dict(
                    color=log_p_values,
                    colorscale='Viridis',
                    colorbar=dict(title='-log10(p)', x=1.15, y=0.25, len=0.4)
                )
            )
            fig.add_trace(coint_bar, row=2, col=2)

        # Add a horizontal line at .05 for cointegration
        fig.add_hline(y=-np.log10(0.05), line_dash='dash', line_color='red', row=2, col=2)

        # Create dropdown menu buttons
        buttons = []
        for key in etf_dataframes_correlation_matrices.keys():
            matrix = etf_dataframes_correlation_matrices[key]
            df = etf_dataframes_monthly[key]
            pair_names, pair_corrs = rolling.get_sorted_correlations(matrix)
            pair_names_list = list(pair_names)
            coint_pair_names, coint_p_values = rolling.get_cointegration_pvals(df, pair_names_list)
            log_p_values = [-np.log10(p) if p > 0 else 15 for p in coint_p_values] if coint_p_values else []

            buttons.append(
                dict(
                    method='update',
                    label=key,
                    args=[{
                        'z': [matrix.values, None, None],
                        'x': [matrix.columns, pair_names_list, coint_pair_names],
                        'y': [matrix.index, pair_corrs, log_p_values],
                        'marker.color': [None, pair_corrs, log_p_values]
                    }]
                )
            )

        # Update layout
        fig.update_layout(
            title='ETF Analysis: Monthly Correlation and Cointegration',
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            height=900,
        )

        # Format x-axes
        fig.update_xaxes(tickangle=90, tickfont=dict(size=8), row=1, col=2)
        fig.update_xaxes(tickangle=90, tickfont=dict(size=8), row=2, col=2)

        # Add axis titles
        fig.update_yaxes(title='Correlation', row=1, col=2)
        fig.update_yaxes(title='-log10(p-value)', row=2, col=2)

        # Show the figure
        fig.show()
    
    def plot_rolling_regression(self, rolling_results, ticker_str, factor_returns):
        # Plot the alpha
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_results.index,
            y=rolling_results['alpha'],
            mode='lines',
            name='Alpha'
        ))
        # Add a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=rolling_results.index[0],
            y0=0,
            x1=rolling_results.index[-1],
            y1=0,
            line=dict(
                color="white",
                width=1,
                dash="dash"
            )
        )
        
        # Add a horizontal line for the mean of alpha
        mean_alpha = rolling_results['alpha'].mean()
        std_alpha = rolling_results['alpha'].std()
        fig.add_hline(
            y=mean_alpha,
            line_color="white",
            line_dash="dot",
            annotation_text=f"Mean: {mean_alpha:.2f}",
            annotation_position="bottom right"
        )
        
        # Add horizontal lines for Â±1, Â±1.5, Â±2, Â±3 standard deviations from the mean
        for i in [1, 1.5, 2, 3]:
            fig.add_hline(
                y=mean_alpha + i * std_alpha,
                line_color="red",
                line_dash="dash",
                annotation_text=f"+{i}Ïƒ: {mean_alpha + i * std_alpha:.2f}",
                annotation_position="top right"
            )
            fig.add_hline(
                y=mean_alpha - i * std_alpha,
                line_color="green",
                line_dash="dash",
                annotation_text=f"-{i}Ïƒ: {mean_alpha - i * std_alpha:.2f}",
                annotation_position="bottom right"
            )
        
        fig.update_layout(
            title=f'{ticker_str} Rolling Alpha',
            xaxis_title='Date',
            yaxis_title='Alpha',
            template='plotly_dark',
            height=600,
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False
            )
        )
        fig.show()
        # Create subplots: one row per factor
        # Exclude the 'RF' column so that we only plot factor betas (e.g., Mkt-RF, SMB, HML)
        factors_to_plot = [factor for factor in factor_returns.columns if factor != "RF"]
        num_factors = len(factors_to_plot)

        # Create subplots for each factor beta
        fig = make_subplots(
            rows=num_factors,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{factor} Beta" for factor in factors_to_plot]
        )

        # Add each beta trace along with its horizontal baseline:
        for i, factor in enumerate(factors_to_plot, start=1):
            fig.add_trace(
                go.Scatter(
                    x=rolling_results.index,
                    y=rolling_results[f'{factor}_beta'],
                    mode='lines',
                    name=f'{factor} Beta'
                ),
                row=i,
                col=1
            )
            baseline = 1 if factor == "Mkt-RF" else 0
            fig.add_hline(
                y=baseline,
                row=i,
                col=1,
                line=dict(color="white", dash="dash"),
                annotation_text=f"Baseline: {baseline}",
                annotation_position="bottom right"
            )

        fig.update_layout(
            title=f'{ticker_str} Rolling Betas',
            template='plotly_dark',
            height=400 * num_factors,
            showlegend=False
        )
        fig.show()
        # Plot the R-squared and adjusted R-squared
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_results.index,
            y=rolling_results['r_squared'],
            mode='lines',
            name='R-Squared'
        ))
        fig.add_trace(go.Scatter(
            x=rolling_results.index,
            y=rolling_results['adj_r_squared'],
            mode='lines',
            name='Adjusted R-Squared'
        ))
        
        #add a horizontal line at mean of the r-squared
        mean_r_squared = rolling_results['r_squared'].mean()
        fig.add_hline(
            y=mean_r_squared,
            line_color="white",
            line_dash="dot",
            annotation_text=f"Mean: {mean_r_squared:.2f}",
            annotation_position="bottom right"
        )
        
        #add a horizontal line at 0
        fig.add_shape(
            type="line",
            x0=rolling_results.index[0],
            y0=0,
            x1=rolling_results.index[-1],
            y1=0,
            line=dict(
                color="white",
                width=1,
                dash="dash"
            )
        )
        
        fig.update_layout(
            title=f'{ticker_str} Rolling R-Squared',
            xaxis_title='Date',
            yaxis_title='R-Squared',
            template='plotly_dark',
            height=600,
            xaxis=dict(
                rangeslider=dict(visible=False),
                tickangle=-45,
                showgrid=True,
                zeroline=False
            )
        )
        fig.show()

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

class LineChartPlotter:
    def __init__(self):
        pass
    
    def plot_series(self, data, title):
        fig = go.Figure()
        if isinstance(data, pd.DataFrame):
            for ticker in data.index:
                fig.add_trace(go.Scatter(x=data.columns, y=data.loc[ticker], name=ticker))
        elif isinstance(data, pd.Series):
            fig.add_trace(go.Scatter(x=data.index, y=data.values, name=data.name or 'Series'))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Close Price')
        return fig

    
    def plot_returns(
        self,
        series,
        benchmark_series=None,
        plot_type="returns",
        title="Returns Analysis",
        windows=None,
        ratio_type="sharpe",
        default_years=10,
    ):
        import copy
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        if windows is None:
            raise ValueError("Provide at least one window via 'windows'.")

        try:
            windows_iterable = list(windows)
        except TypeError:
            windows_iterable = [windows]

        sanitized_windows = []
        for win in windows_iterable:
            if win is None or pd.isna(win):
                continue
            try:
                win_int = int(win)
            except (TypeError, ValueError):
                continue
            if win_int > 0:
                sanitized_windows.append(win_int)

        windows = list(dict.fromkeys(sanitized_windows))
        if not windows:
            raise ValueError("No valid windows supplied.")

        def _metric_label(metric_key):
            return "Returns" if metric_key is None else metric_key.capitalize()

        series = series.sort_index()
        if benchmark_series is not None:
            benchmark_series = benchmark_series.sort_index()

        end_date = series.index.max()
        start_date = end_date - pd.DateOffset(years=default_years)
        series = series.loc[start_date:end_date]
        if benchmark_series is not None:
            benchmark_series = benchmark_series.loc[start_date:end_date]

        has_benchmark = benchmark_series is not None and not benchmark_series.empty
        window_series_map: dict[str | None, dict[int, pd.Series]] = {}
        benchmark_window_map: dict[str | None, dict[int, pd.Series]] = {}

        if plot_type == "risk_adjusted":
            metric_choices = ["sharpe", "sortino", "omega", "calmar", "sterling"]
            for metric in metric_choices:
                metric_series = {}
                metric_benchmark = {}
                for win in windows:
                    ratio_df = rolling.risk_adjusted_returns(series, windows=[win], ratio_type=metric)
                    ratio_series = ratio_df.iloc[:, 0].dropna()
                    if ratio_series.empty:
                        continue
                    metric_series[win] = ratio_series
                    if has_benchmark:
                        bench_df = rolling.risk_adjusted_returns(
                            benchmark_series, windows=[win], ratio_type=metric
                        )
                        metric_benchmark[win] = bench_df.iloc[:, 0].reindex(ratio_series.index)
                if metric_series:
                    window_series_map[metric] = metric_series
                    if has_benchmark and metric_benchmark:
                        benchmark_window_map[metric] = metric_benchmark
            if not window_series_map:
                raise ValueError("No risk-adjusted data available for the requested windows.")
        else:
            base_series_map = {}
            base_benchmark_map = {}
            for win in windows:
                win_series = series.pct_change(win).dropna()
                if win_series.empty:
                    continue
                base_series_map[win] = win_series
                if has_benchmark:
                    base_benchmark_map[win] = benchmark_series.pct_change(win).reindex(win_series.index)
            if not base_series_map:
                raise ValueError("No data available for the requested windows.")
            window_series_map[None] = base_series_map
            if has_benchmark and base_benchmark_map:
                benchmark_window_map[None] = base_benchmark_map

        available_metrics = list(window_series_map.keys())
        if plot_type == "risk_adjusted":
            requested_metric = str(ratio_type).lower()
            default_metric = requested_metric if requested_metric in window_series_map else available_metrics[0]
        else:
            default_metric = available_metrics[0]

        metric_windows_map = {
            metric_key: list(series_map.keys()) for metric_key, series_map in window_series_map.items()
        }
        default_window = metric_windows_map[default_metric][0]

        global_min = series.index.min()
        global_max = series.index.max()
        if has_benchmark:
            global_min = min(global_min, benchmark_series.index.min())
            global_max = max(global_max, benchmark_series.index.max())

        def clamp(years):
            start = global_max - pd.DateOffset(years=years)
            start = max(start, global_min)
            return [start, global_max]

        timeframe_buttons = [
            dict(label="Max", method="relayout", args=[{"xaxis.range": [global_min, global_max]}]),
            dict(label="10 Years", method="relayout", args=[{"xaxis.range": clamp(10)}]),
            dict(label="5 Years", method="relayout", args=[{"xaxis.range": clamp(5)}]),
            dict(label="3 Years", method="relayout", args=[{"xaxis.range": clamp(3)}]),
            dict(label="1 Year", method="relayout", args=[{"xaxis.range": clamp(1)}]),
        ]
        timeframe_menu = dict(
            buttons=timeframe_buttons,
            direction="down",
            showactive=True,
            x=0.22,
            xanchor="left",
            y=1.15,
            yanchor="top",
            active=0,
        )

        def build_layout_components(window_series, current_window, metric_key=None):
            shapes, annotations = [], []
            idx_min = window_series.index.min()
            idx_max = window_series.index.max()
            center_x = idx_min + (idx_max - idx_min) / 2

            negatives = window_series[window_series < 0]
            positives = window_series[window_series > 0]

            std_minus_05 = None
            if not negatives.empty:
                std_neg = negatives.std()
                mean_neg = negatives.mean()
                if pd.notna(std_neg):
                    std_minus_05 = mean_neg - 0.5 * std_neg

            std_plus_05 = std_plus_15 = std_plus_3 = None
            if not positives.empty:
                std_pos = positives.std()
                mean_pos = positives.mean()
                if pd.notna(std_pos):
                    std_plus_05 = mean_pos + 0.5 * std_pos
                    std_plus_15 = mean_pos + 1.5 * std_pos
                    std_plus_3 = mean_pos + 3.0 * std_pos

            mean_value = window_series.mean()

            shapes.append(
                dict(type="line", xref="x", yref="y", x0=idx_min, x1=idx_max, y0=0, y1=0, line=dict(color="black", dash="dash"))
            )
            annotations.append(
                dict(x=idx_max, y=0, showarrow=False, xanchor="left", yanchor="bottom", text="Zero Line", font=dict(color="black", size=10))
            )

            shapes.append(
                dict(type="line", xref="x", yref="y", x0=idx_min, x1=idx_max, y0=mean_value, y1=mean_value, line=dict(color="cyan"))
            )
            annotations.append(
                dict(x=idx_max, y=mean_value, showarrow=False, xanchor="left", yanchor="middle", text="Mean", font=dict(color="cyan", size=10))
            )

            if std_minus_05 is not None:
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_minus_05,
                        y1=std_minus_05,
                        line=dict(color="orange", dash="dashdot"),
                    )
                )
                annotations.append(
                    dict(
                        x=idx_max,
                        y=std_minus_05,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        text=f"-0.5 Std Dev ({std_minus_05:.2f})",
                        font=dict(color="orange", size=10),
                    )
                )
                lower = min(window_series.min(), std_minus_05)
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=lower,
                        y1=std_minus_05,
                        fillcolor="darkgreen",
                        opacity=0.4,
                        line=dict(width=0),
                    )
                )
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_minus_05,
                        y1=0,
                        fillcolor="lightgreen",
                        opacity=0.4,
                        line=dict(width=0),
                    )
                )
                annotations.append(
                    dict(x=center_x, y=(0 + std_minus_05) / 2, showarrow=False, text="Buy Zone", font=dict(size=12, color="black"), align="center")
                )
                annotations.append(
                    dict(
                        x=center_x,
                        y=(lower + std_minus_05) / 2,
                        showarrow=False,
                        text="Definite Buy Zone",
                        font=dict(size=12, color="black"),
                        align="center",
                    )
                )

            if std_plus_05 is not None:
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_plus_05,
                        y1=std_plus_05,
                        line=dict(color="blue", dash="dashdot"),
                    )
                )
                annotations.append(
                    dict(
                        x=idx_max,
                        y=std_plus_05,
                        showarrow=False,
                        xanchor="left",
                        yanchor="middle",
                        text=f"+0.5 Std Dev ({std_plus_05:.2f})",
                        font=dict(color="blue", size=10),
                    )
                )
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=0,
                        y1=std_plus_05,
                        fillcolor="yellow",
                        opacity=0.4,
                        line=dict(width=0),
                    )
                )
                annotations.append(
                    dict(x=center_x, y=std_plus_05 / 2, showarrow=False, text="Hold Zone", font=dict(size=12, color="black"), align="center")
                )

            if std_plus_05 is not None and std_plus_15 is not None:
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_plus_15,
                        y1=std_plus_15,
                        line=dict(color="magenta", dash="dashdot"),
                    )
                )
                annotations.append(
                    dict(x=idx_max, y=std_plus_15, showarrow=False, xanchor="left", yanchor="middle", text="+1.5 Std Dev", font=dict(color="magenta", size=10))
                )
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_plus_05,
                        y1=std_plus_15,
                        fillcolor="magenta",
                        opacity=0.2,
                        line=dict(width=0),
                    )
                )
                annotations.append(
                    dict(
                        x=center_x,
                        y=(std_plus_05 + std_plus_15) / 2,
                        showarrow=False,
                        text="Reduce Risk Zone",
                        font=dict(size=12, color="black"),
                        align="center",
                    )
                )

            if std_plus_15 is not None and std_plus_3 is not None:
                shapes.append(
                    dict(type="line", xref="x", yref="y", x0=idx_min, x1=idx_max, y0=std_plus_3, y1=std_plus_3, line=dict(color="red", dash="dashdot"))
                )
                annotations.append(
                    dict(x=idx_max, y=std_plus_3, showarrow=False, xanchor="left", yanchor="middle", text="+3.0 Std Dev", font=dict(color="red", size=10))
                )
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=idx_min,
                        x1=idx_max,
                        y0=std_plus_15,
                        y1=std_plus_3,
                        fillcolor="red",
                        opacity=0.2,
                        line=dict(width=0),
                    )
                )
                annotations.append(
                    dict(
                        x=center_x,
                        y=(std_plus_15 + std_plus_3) / 2,
                        showarrow=False,
                        text="Liquidation Zone",
                        font=dict(size=12, color="black"),
                        align="center",
                    )
                )

            latest = window_series.index.max()
            default_start = latest - pd.DateOffset(years=3)
            default_start = max(default_start, idx_min)

            prefix = "Risk-Adjusted" if plot_type == "risk_adjusted" else "Regular Returns"
            if plot_type == "risk_adjusted" and metric_key is not None:
                metric_title = _metric_label(metric_key)
                title_text = f"{prefix} {title} ({metric_title}, Window={current_window} days)"
                yaxis_title = f"{metric_title} Ratio"
            else:
                title_text = f"{prefix} {title} (Window={current_window} days)"
                yaxis_title = "Return"

            return {
                "shapes": shapes,
                "annotations": annotations,
                "title": title_text,
                "xaxis_range": [default_start, latest],
                "yaxis_title": yaxis_title,
            }

        combination_data = {}
        for metric_key, series_map in window_series_map.items():
            for win, win_series in series_map.items():
                layout = build_layout_components(win_series, win, metric_key)
                x_payload = [win_series.index]
                y_payload = [win_series.values]
                names = [f"{win}-Day {_metric_label(metric_key)}"]

                bench_series_dict = benchmark_window_map.get(metric_key, {})
                bench_series = bench_series_dict.get(win)
                if has_benchmark and bench_series is not None:
                    aligned = bench_series.reindex(win_series.index)
                    x_payload.append(aligned.index)
                    y_payload.append(aligned.values)
                    names.append("Benchmark")

                combination_data[(metric_key, win)] = {
                    "x": x_payload,
                    "y": y_payload,
                    "names": names,
                    "layout": layout,
                }

        def make_window_menu(metric_key, active_window=None):
            metric_windows = metric_windows_map[metric_key]
            buttons = []
            for win in metric_windows:
                combo = combination_data[(metric_key, win)]
                buttons.append(
                    dict(
                        label=f"{win}-Day",
                        method="update",
                        args=[
                            {"x": combo["x"], "y": combo["y"], "name": combo["names"]},
                            {
                                "title": combo["layout"]["title"],
                                "shapes": combo["layout"]["shapes"],
                                "annotations": combo["layout"]["annotations"],
                                "xaxis": {"range": combo["layout"]["xaxis_range"]},
                                "yaxis": {"title": combo["layout"]["yaxis_title"]},
                            },
                        ],
                    )
                )
            active_index = 0
            if active_window in metric_windows:
                active_index = metric_windows.index(active_window)
            return dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
                active=active_index,
            )

        default_combo = combination_data[(default_metric, default_window)]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=default_combo["x"][0],
                y=default_combo["y"][0],
                mode="lines",
                name=default_combo["names"][0],
            )
        )
        if has_benchmark and len(default_combo["x"]) > 1:
            fig.add_trace(
                go.Scatter(
                    x=default_combo["x"][1],
                    y=default_combo["y"][1],
                    mode="lines",
                    name="Benchmark",
                    line=dict(dash="dash"),
                )
            )

        defaults_layout = default_combo["layout"]
        fig.update_layout(
            height=1000,
            title=defaults_layout["title"],
            shapes=defaults_layout["shapes"],
            annotations=defaults_layout["annotations"],
            xaxis=dict(range=defaults_layout["xaxis_range"]),
            yaxis=dict(title=defaults_layout["yaxis_title"]),
            hovermode="x unified",
        )

        window_menu_initial = make_window_menu(default_metric, default_window)
        menus = [window_menu_initial, copy.deepcopy(timeframe_menu)]

        if plot_type == "risk_adjusted":
            metric_buttons = []
            metric_key_order = []
            for metric_key in available_metrics:
                metric_windows = metric_windows_map[metric_key]
                metric_default_window = metric_windows[0]
                combo = combination_data[(metric_key, metric_default_window)]
                metric_buttons.append(
                    dict(
                        label=_metric_label(metric_key),
                        method="update",
                        args=[
                            {"x": combo["x"], "y": combo["y"], "name": combo["names"]},
                            {
                                "title": combo["layout"]["title"],
                                "shapes": combo["layout"]["shapes"],
                                "annotations": combo["layout"]["annotations"],
                                "xaxis": {"range": combo["layout"]["xaxis_range"]},
                                "yaxis": {"title": combo["layout"]["yaxis_title"]},
                            },
                        ],
                    )
                )
                metric_key_order.append(metric_key)

            metric_menu_template = dict(
                buttons=metric_buttons,
                direction="down",
                showactive=True,
                x=0.43,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )

            for idx, metric_key in enumerate(metric_key_order):
                button = metric_buttons[idx]
                metric_menu_active = copy.deepcopy(metric_menu_template)
                metric_menu_active["active"] = idx
                button["args"][1]["updatemenus"] = [
                    make_window_menu(metric_key),
                    copy.deepcopy(timeframe_menu),
                    metric_menu_active,
                ]

            metric_menu_initial = copy.deepcopy(metric_menu_template)
            metric_menu_initial["active"] = available_metrics.index(default_metric)
            menus.append(metric_menu_initial)

        fig.update_layout(updatemenus=menus)
        return fig

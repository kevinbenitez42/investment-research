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
from .figure_helpers import (
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    add_std_annotations,
    add_zone_annotation,
    build_detail_visibility_mask,
    build_time_range_buttons,
    build_visibility_mask,
)

rolling = Rolling()

class LineChartPlotter:
    HEADER_TOP_MARGIN = 150
    HEADER_TITLE_Y = 0.97
    HEADER_MENU_Y = 1.08

    def __init__(self):
        pass

    @classmethod
    def _header_margin(cls, top=None):
        return dict(t=cls.HEADER_TOP_MARGIN if top is None else int(top))

    @classmethod
    def _header_title(cls, text):
        return dict(
            text=str(text),
            x=0.5,
            xanchor="center",
            y=cls.HEADER_TITLE_Y,
            yanchor="top",
        )

    @classmethod
    def _dropdown_menu(
        cls,
        *,
        buttons,
        x,
        active=None,
        y=None,
        direction="down",
        showactive=True,
        xanchor="left",
        yanchor="top",
        **overrides,
    ):
        menu = dict(
            type="dropdown",
            buttons=buttons,
            direction=direction,
            showactive=showactive,
            x=x,
            xanchor=xanchor,
            y=cls.HEADER_MENU_Y if y is None else y,
            yanchor=yanchor,
        )
        if active is not None:
            menu["active"] = active
        menu.update(overrides)
        return menu

    @staticmethod
    def _coerce_positive_int(value):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced > 0 else None

    @classmethod
    def _preferred_numeric_window(cls, options, preferred=200):
        normalized = []
        seen = set()
        for option in options:
            coerced = cls._coerce_positive_int(option)
            if coerced is None or coerced in seen:
                continue
            normalized.append(coerced)
            seen.add(coerced)

        if not normalized:
            return None
        if preferred in seen:
            return preferred
        return max(normalized)

    @classmethod
    def _window_value_from_label(cls, label, config=None):
        if isinstance(config, Mapping):
            config_window = cls._coerce_positive_int(config.get("time_frame"))
            if config_window is not None:
                return config_window

        digits = "".join(ch if ch.isdigit() else " " for ch in str(label)).split()
        if not digits:
            return None
        return cls._coerce_positive_int(digits[0])

    @classmethod
    def _preferred_window_label(cls, label_map, preferred=200):
        labels = list(label_map.keys())
        if not labels:
            return None

        exact_label = f"{preferred}-day"
        if exact_label in label_map:
            return exact_label

        for label, config in label_map.items():
            if cls._window_value_from_label(label, config) == preferred:
                return label

        return max(
            labels,
            key=lambda label: cls._window_value_from_label(label, label_map.get(label)) or -1,
        )

    @classmethod
    def _preferred_term_key(cls, time_frame_map, term_options=None, preferred=200):
        if term_options is None:
            candidates = [term for term in time_frame_map.keys()]
        else:
            allowed = set(term_options)
            candidates = [term for term in time_frame_map.keys() if term in allowed]
            if not candidates:
                candidates = list(term_options)

        if not candidates:
            return None
        if "long" in candidates:
            return "long"

        for term in candidates:
            if cls._coerce_positive_int(time_frame_map.get(term)) == preferred:
                return term

        return max(
            candidates,
            key=lambda term: cls._coerce_positive_int(time_frame_map.get(term)) or -1,
        )
    
    def plot_series(self, data, title):
        fig = go.Figure()
        if isinstance(data, pd.DataFrame):
            for ticker in data.index:
                fig.add_trace(go.Scatter(x=data.columns, y=data.loc[ticker], name=ticker))
        elif isinstance(data, pd.Series):
            fig.add_trace(go.Scatter(x=data.index, y=data.values, name=data.name or 'Series'))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Close Price')
        return fig

    def plot_momentum_zscore_comparison(
        self,
        zscore_data,
        ticker_label="Asset",
        default_label=None,
        default_time_label="3 Years",
        sigma_levels=(0.5, 1.0, 1.5),
        show_zones=True,
    ):
        """
        Plot interactive momentum z-score comparisons with window and time dropdowns.
        """
        if not isinstance(zscore_data, Mapping) or not zscore_data:
            raise ValueError("zscore_data must be a non-empty mapping of label -> pandas Series.")

        prepared_data = {}
        for label, series in zscore_data.items():
            if not isinstance(series, pd.Series):
                raise TypeError(f"zscore_data['{label}'] must be a pandas Series.")
            cleaned = series.dropna().sort_index()
            if not cleaned.empty:
                prepared_data[str(label)] = cleaned

        if not prepared_data:
            raise ValueError("No non-empty momentum z-score series found in zscore_data.")

        labels = list(prepared_data.keys())
        if default_label not in prepared_data:
            default_label = self._preferred_window_label(prepared_data) or labels[0]

        fig = make_subplots(rows=1, cols=1)
        for label in labels:
            series = prepared_data[label]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=label,
                    visible=(label == default_label),
                ),
                row=1,
                col=1,
            )

        global_start = min(series.index.min() for series in prepared_data.values())
        global_end = max(series.index.max() for series in prepared_data.values())
        x_ref = prepared_data[default_label].index

        overlay_start_idx = len(fig.data)
        add_sigma_reference_lines(
            fig,
            row=1,
            col=1,
            x_ref=x_ref,
            levels=sigma_levels,
            sigma=1.0,
            center=0.0,
            line_color="rgba(160, 160, 160, 0.50)",
            line_dash="dash",
        )
        add_mean_reference_line(
            fig,
            row=1,
            col=1,
            x_ref=x_ref,
            line_color="rgba(0, 0, 0, 0.85)",
        )

        if show_zones:
            add_horizontal_zone_trace(
                fig,
                row=1,
                col=1,
                x_ref=x_ref,
                y0=-1.5,
                y1=-1.0,
                fillcolor="rgba(0, 170, 0, 0.15)",
            )
            add_horizontal_zone_trace(
                fig,
                row=1,
                col=1,
                x_ref=x_ref,
                y0=1.0,
                y1=1.5,
                fillcolor="rgba(220, 0, 0, 0.15)",
            )

        constant_trace_indices = list(range(overlay_start_idx, len(fig.data)))
        total_traces = len(fig.data)
        buttons_window = []
        for idx, label in enumerate(labels):
            visibility = build_visibility_mask(
                total_traces=total_traces,
                active_window_index=idx,
                traces_per_window=1,
                constant_trace_indices=constant_trace_indices,
            )
            buttons_window.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": self._header_title(f"{ticker_label} Momentum Z-Score {label}")},
                    ],
                )
            )

        buttons_time = build_time_range_buttons(global_start, global_end, axis_count=1)
        year_map = {"10 Years": 10, "5 Years": 5, "3 Years": 3, "1 Year": 1}
        default_years = year_map.get(default_time_label, 3)
        default_start = max(global_start, global_end - pd.DateOffset(years=default_years))

        fig.update_layout(
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons_window,
                    x=0.10,
                    direction="down",
                ),
                self._dropdown_menu(
                    buttons=buttons_time,
                    x=0.33,
                    direction="down",
                ),
            ],
            height=600,
            margin=self._header_margin(),
            title=self._header_title(f"{ticker_label} Momentum Z-Score {default_label}"),
            template="plotly_white",
            yaxis_title="Z-Score",
            xaxis=dict(range=[default_start, global_end]),
        )
        return fig

    def plot_sharpe_sortino_comparison(self, term_config_map, ticker_label="Asset", default_label=None):
        """
        Plot Sharpe/Sortino ratio and spread z-score panels with a term dropdown.
        """
        if not isinstance(term_config_map, Mapping) or not term_config_map:
            raise ValueError("term_config_map must be a non-empty mapping.")

        term_labels = list(term_config_map.keys())
        if default_label not in term_config_map:
            default_label = self._preferred_window_label(term_config_map) or term_labels[0]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Sharpe and Sortino Ratios",
                "Sharpe-Sortino Spread (z-score)",
            ),
        )

        term_trace_map = {}
        for term_label in term_labels:
            cfg = term_config_map[term_label]
            sharpe = cfg["sharpe"]
            sortino = cfg["sortino"]
            time_frame = cfg.get("time_frame", term_label)

            mean_sharpe = sharpe.mean()
            mean_sortino = sortino.mean()

            spread = (sharpe - sortino).dropna()
            spread_mean = spread.mean()
            spread_std = spread.std(ddof=0)
            if pd.isna(spread_std) or spread_std == 0:
                spread_std = 1.0
            z_spread = (spread - spread_mean) / spread_std
            z_index = z_spread.index if not z_spread.empty else sharpe.dropna().index

            visible = term_label == default_label
            trace_indices = []

            fig.add_trace(
                go.Scatter(
                    x=sharpe.index,
                    y=sharpe,
                    mode="lines",
                    name=f"Sharpe Ratio ({time_frame}-day)",
                    line=dict(color="blue"),
                    visible=visible,
                    legendgroup="Sharpe",
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=sortino.index,
                    y=sortino,
                    mode="lines",
                    name=f"Sortino Ratio ({time_frame}-day)",
                    line=dict(color="orange"),
                    visible=visible,
                    legendgroup="Sortino",
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=sharpe.index,
                    y=np.full(len(sharpe.index), mean_sharpe),
                    mode="lines",
                    name=f"Mean Sharpe ({mean_sharpe:.3f})",
                    line=dict(color="blue", dash="dash"),
                    opacity=0.7,
                    visible=visible,
                    legendgroup="Sharpe Mean",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=sortino.index,
                    y=np.full(len(sortino.index), mean_sortino),
                    mode="lines",
                    name=f"Mean Sortino ({mean_sortino:.3f})",
                    line=dict(color="orange", dash="dash"),
                    opacity=0.7,
                    visible=visible,
                    legendgroup="Sortino Mean",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=z_index,
                    y=z_spread.reindex(z_index),
                    mode="lines",
                    name=f"Sharpe-Sortino Spread ({time_frame}-day)",
                    line=dict(color="green"),
                    visible=visible,
                    legendgroup="Spread",
                ),
                row=2,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            if len(z_index) > 0:
                for level, color in [(0, "black"), (1, "red"), (-1, "red"), (2, "purple"), (-2, "purple")]:
                    fig.add_trace(
                        go.Scatter(
                            x=z_index,
                            y=np.full(len(z_index), level),
                            mode="lines",
                            line=dict(color=color, dash="dot"),
                            hoverinfo="skip",
                            showlegend=False,
                            visible=visible,
                        ),
                        row=2,
                        col=1,
                    )
                    trace_indices.append(len(fig.data) - 1)

            if not z_spread.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[z_spread.index[-1]],
                        y=[z_spread.mean()],
                        mode="markers+text",
                        text=[f"Mean z: {z_spread.mean():.2f}"],
                        textposition="middle right",
                        marker=dict(color="purple", size=6),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )
                trace_indices.append(len(fig.data) - 1)

            term_trace_map[term_label] = trace_indices

        total_traces = len(fig.data)
        buttons = []
        for term_label in term_labels:
            visibility = [False] * total_traces
            for trace_idx in term_trace_map[term_label]:
                visibility[trace_idx] = True

            buttons.append(
                dict(
                    label=term_label,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": self._header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({term_label})")},
                    ],
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=900,
            margin=self._header_margin(),
            legend=dict(x=0.01, y=0.99),
            xaxis2_title="Date",
            yaxis_title="Ratio Value",
            yaxis2_title="Spread (z-score)",
            title=self._header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({default_label})"),
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.01,
                    active=term_labels.index(default_label),
                )
            ],
        )
        return fig

    def plot_multi_benchmark_sharpe_spread_summary(
        self,
        summary_zscore_map,
        time_frame_map,
        ticker_label="Asset",
        default_term=None,
        template="plotly_dark",
    ):
        """
        Plot benchmark Sharpe-spread z-score summary with term/time dropdowns.
        """
        if not isinstance(summary_zscore_map, Mapping) or not summary_zscore_map:
            raise ValueError("summary_zscore_map must be a non-empty mapping.")
        if not isinstance(time_frame_map, Mapping) or not time_frame_map:
            raise ValueError("time_frame_map must be a non-empty mapping.")

        term_order = [term for term in time_frame_map.keys() if term in summary_zscore_map]
        if not term_order:
            raise ValueError("No overlapping term keys between summary_zscore_map and time_frame_map.")

        if default_term not in term_order:
            default_term = self._preferred_term_key(time_frame_map, term_order) or term_order[0]
        benchmark_order = []
        for term in term_order:
            for symbol in summary_zscore_map.get(term, {}).keys():
                if symbol not in benchmark_order:
                    benchmark_order.append(symbol)

        fig = make_subplots(rows=1, cols=1)
        benchmark_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        term_ranges = {}
        term_trace_bounds = {}

        for term in term_order:
            term_series_map = summary_zscore_map.get(term, {})
            visible = term == default_term
            non_empty_series = []
            x_ref = pd.Index([])

            term_trace_start = len(fig.data)
            for idx, symbol in enumerate(benchmark_order):
                zscore_series = term_series_map.get(symbol, pd.Series(dtype=float)).dropna()
                if not zscore_series.empty:
                    non_empty_series.append(zscore_series)
                    if len(zscore_series.index) > len(x_ref):
                        x_ref = zscore_series.index

            add_horizontal_zone_trace(fig, 1, x_ref, -2, -1.5, "rgba(180, 0, 0, 0.40)", visible=visible)
            add_horizontal_zone_trace(fig, 1, x_ref, 1.5, 2, "rgba(0, 128, 0, 0.55)", visible=visible)

            for idx, symbol in enumerate(benchmark_order):
                zscore_series = term_series_map.get(symbol, pd.Series(dtype=float)).dropna()
                fig.add_trace(
                    go.Scatter(
                        x=zscore_series.index,
                        y=zscore_series,
                        mode="lines",
                        name=symbol,
                        legendgroup=symbol,
                        showlegend=visible,
                        line=dict(color=benchmark_palette[idx % len(benchmark_palette)]),
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )

            add_mean_reference_line(fig, 1, x_ref, visible=visible)
            add_sigma_reference_lines(fig, 1, x_ref, levels=(0.5, 1, 1.5, 2), visible=visible)

            term_trace_bounds[term] = (term_trace_start, len(fig.data))
            if non_empty_series:
                max_index = max(series.index.max() for series in non_empty_series)
                min_index = min(series.index.min() for series in non_empty_series)
                term_ranges[term] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
            else:
                term_ranges[term] = None

        add_zone_annotation(fig, 1, -2, -1.5, "Liquidate", "rgba(255, 235, 235, 0.95)")
        add_zone_annotation(fig, 1, 1.5, 2, "Accumulate", "rgba(235, 255, 235, 0.95)")
        add_std_annotations(fig, 1, levels=(0.5, 1, 1.5, 2))

        timeframe_buttons = []
        total_traces = len(fig.data)
        for term in term_order:
            visibility = [False] * total_traces
            start, end = term_trace_bounds[term]
            for trace_idx in range(start, end):
                visibility[trace_idx] = True

            layout_updates = {
                "title": self._header_title(
                    f"{ticker_label} {term.title()} Sharpe Spread Z-Scores vs Benchmarks ({time_frame_map[term]}-Day)"
                ),
                "yaxis": {"title": "Sharpe Spread Z-Score"},
            }
            if term_ranges.get(term) is not None:
                layout_updates["xaxis"] = {"range": term_ranges[term]}

            timeframe_buttons.append(
                dict(
                    label=f"{term.title()} ({time_frame_map[term]})",
                    method="update",
                    args=[{"visible": visibility}, layout_updates],
                )
            )

        available_ranges = [date_range for date_range in term_ranges.values() if date_range is not None]
        if available_ranges:
            global_start = min(date_range[0] for date_range in available_ranges)
            global_end = max(date_range[1] for date_range in available_ranges)
            fig.update_xaxes(range=term_ranges[default_term] or [global_start, global_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(global_start, global_end),
                x=0.22,
            )
        else:
            time_range_menu = None

        fig.update_yaxes(title_text="Sharpe Spread Z-Score", row=1, col=1)
        fig.update_layout(
            title=self._header_title(
                f"{ticker_label} {default_term.title()} Sharpe Spread Z-Scores vs Benchmarks ({time_frame_map[default_term]}-Day)"
            ),
            height=650,
            margin=self._header_margin(),
            template=template,
            showlegend=True,
            updatemenus=[
                self._dropdown_menu(
                    buttons=timeframe_buttons,
                    x=0.0,
                )
            ]
            + ([time_range_menu] if time_range_menu is not None else []),
        )
        return fig

    def plot_benchmark_zscore_detail(
        self,
        detail_zscore_map,
        benchmark_order,
        time_frame_map,
        ticker_label="Asset",
        default_benchmark=None,
        default_term=None,
        template="plotly_dark",
    ):
        """
        Plot benchmark detail panel: z-score, Sharpe, excess return, and volatility decomposition.
        """
        if not benchmark_order:
            raise ValueError("benchmark_order is empty.")
        if not isinstance(detail_zscore_map, Mapping) or not detail_zscore_map:
            raise ValueError("detail_zscore_map must be a non-empty mapping.")
        if not isinstance(time_frame_map, Mapping) or not time_frame_map:
            raise ValueError("time_frame_map must be a non-empty mapping.")

        term_order = list(time_frame_map.keys())
        if default_term not in term_order:
            default_term = self._preferred_term_key(time_frame_map, term_order) or term_order[0]
        default_benchmark = default_benchmark if default_benchmark in benchmark_order else benchmark_order[0]

        detail_fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.32, 0.23, 0.23, 0.22],
            subplot_titles=(
                "Risk-Adjusted Return Z-Score Comparison",
                "Rolling Sharpe Ratio",
                "Annualized Excess Return",
                "Annualized Volatility",
            ),
        )

        term_styles = {
            term_order[0]: {"color": "#1f77b4"},
            term_order[1] if len(term_order) > 1 else term_order[0]: {"color": "#ff7f0e"},
            term_order[2] if len(term_order) > 2 else term_order[0]: {"color": "#2ca02c"},
        }
        for term in term_order:
            term_styles.setdefault(term, {"color": "#7f7f7f"})

        detail_view_order = [(symbol, term) for symbol in benchmark_order for term in term_order]
        default_detail_view = (default_benchmark, default_term)
        traces_per_view = None

        for symbol, term in detail_view_order:
            visible = (symbol, term) == default_detail_view
            metric_set = detail_zscore_map.get(symbol, {}).get(term, {})
            style = term_styles.get(term, {"color": "#7f7f7f"})
            trace_start = len(detail_fig.data)

            asset_zscore = metric_set.get("asset", pd.Series(dtype=float)).dropna()
            benchmark_zscore = metric_set.get("benchmark", pd.Series(dtype=float)).dropna()
            asset_sharpe = metric_set.get("asset_sharpe", pd.Series(dtype=float)).dropna()
            benchmark_sharpe = metric_set.get("benchmark_sharpe", pd.Series(dtype=float)).dropna()
            asset_excess_return = metric_set.get("asset_excess_return", pd.Series(dtype=float)).dropna()
            benchmark_excess_return = metric_set.get("benchmark_excess_return", pd.Series(dtype=float)).dropna()
            asset_volatility = metric_set.get("asset_volatility", pd.Series(dtype=float)).dropna()
            benchmark_volatility = metric_set.get("benchmark_volatility", pd.Series(dtype=float)).dropna()

            detail_fig.add_trace(
                go.Scatter(
                    x=asset_zscore.index,
                    y=asset_zscore,
                    mode="lines",
                    name=f"{ticker_label} {term.title()} Sharpe Z-Score",
                    legendgroup=f"asset-{term}",
                    line=dict(color=style["color"], dash="solid", width=2),
                    visible=visible,
                    showlegend=visible,
                ),
                row=1,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=benchmark_zscore.index,
                    y=benchmark_zscore,
                    mode="lines",
                    name=f"{symbol} {term.title()} Sharpe Z-Score",
                    legendgroup=f"{symbol}-{term}",
                    line=dict(color=style["color"], dash="dot", width=2),
                    visible=visible,
                    showlegend=visible,
                ),
                row=1,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=asset_sharpe.index,
                    y=asset_sharpe,
                    mode="lines",
                    name=f"{ticker_label} {term.title()} Sharpe",
                    legendgroup=f"asset-{term}",
                    line=dict(color=style["color"], dash="solid", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=benchmark_sharpe.index,
                    y=benchmark_sharpe,
                    mode="lines",
                    name=f"{symbol} {term.title()} Sharpe",
                    legendgroup=f"{symbol}-{term}",
                    line=dict(color=style["color"], dash="dot", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=asset_excess_return.index,
                    y=asset_excess_return,
                    mode="lines",
                    name=f"{ticker_label} {term.title()} Excess Return",
                    legendgroup=f"asset-{term}",
                    line=dict(color=style["color"], dash="solid", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=benchmark_excess_return.index,
                    y=benchmark_excess_return,
                    mode="lines",
                    name=f"{symbol} {term.title()} Excess Return",
                    legendgroup=f"{symbol}-{term}",
                    line=dict(color=style["color"], dash="dot", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=asset_volatility.index,
                    y=asset_volatility,
                    mode="lines",
                    name=f"{ticker_label} {term.title()} Volatility",
                    legendgroup=f"asset-{term}",
                    line=dict(color=style["color"], dash="solid", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
            detail_fig.add_trace(
                go.Scatter(
                    x=benchmark_volatility.index,
                    y=benchmark_volatility,
                    mode="lines",
                    name=f"{symbol} {term.title()} Volatility",
                    legendgroup=f"{symbol}-{term}",
                    line=dict(color=style["color"], dash="dot", width=2),
                    visible=visible,
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

            added_traces = len(detail_fig.data) - trace_start
            if traces_per_view is None:
                traces_per_view = added_traces

        dynamic_trace_count = len(detail_fig.data)
        detail_x_ref = max((trace.x for trace in detail_fig.data if len(trace.x) > 0), key=len, default=None)
        if detail_x_ref is not None:
            add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, -3, -2, "rgba(0, 128, 0, 0.30)")
            add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, 2, 3, "rgba(180, 0, 0, 0.30)")
            add_zone_annotation(detail_fig, 1, -3, -2, "Accumulate", "rgba(235, 255, 235, 0.95)")
            add_zone_annotation(detail_fig, 1, 2, 3, "Liquidate", "rgba(255, 235, 235, 0.95)")
            add_sigma_reference_lines(detail_fig, 1, detail_x_ref)
            add_mean_reference_line(detail_fig, 2, detail_x_ref)
            add_mean_reference_line(detail_fig, 3, detail_x_ref)
            add_mean_reference_line(detail_fig, 4, detail_x_ref)

        total_traces = len(detail_fig.data)
        buttons = []
        for idx, (symbol, term) in enumerate(detail_view_order):
            visibility = build_detail_visibility_mask(dynamic_trace_count, total_traces, idx, traces_per_view)
            buttons.append(
                dict(
                    label=f"{symbol} | {term.title()} ({time_frame_map[term]})",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": self._header_title(
                                f"{ticker_label} vs {symbol} Risk-Adjusted Return Decomposition [{term.title()} {time_frame_map[term]}-Day]"
                            )
                        },
                    ],
                )
            )

        detail_fig.update_yaxes(title_text="Sharpe Z-Score", row=1, col=1)
        detail_fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        detail_fig.update_yaxes(title_text="Excess Return", tickformat=".1%", row=3, col=1)
        detail_fig.update_yaxes(title_text="Volatility", tickformat=".1%", row=4, col=1)

        detail_start = min((trace.x[0] for trace in detail_fig.data if len(trace.x) > 0), default=None)
        detail_end = max((trace.x[-1] for trace in detail_fig.data if len(trace.x) > 0), default=None)
        if detail_start is not None and detail_end is not None:
            detail_default_start = max(detail_start, detail_end - pd.DateOffset(years=3))
            detail_fig.update_xaxes(range=[detail_default_start, detail_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(detail_start, detail_end, axis_count=4),
                x=0.18,
            )
        else:
            time_range_menu = None

        detail_fig.update_layout(
            title=self._header_title(
                f"{ticker_label} vs {default_benchmark} Risk-Adjusted Return Decomposition [{default_term.title()} {time_frame_map[default_term]}-Day]"
            ),
            height=1350,
            margin=self._header_margin(),
            template=template,
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                )
            ]
            + ([time_range_menu] if time_range_menu is not None else []),
        )
        return detail_fig

    def plot_risk_distribution_zscores(
        self,
        metrics_by_window,
        window_options=None,
        default_window=None,
        ticker_label="Asset",
        template="plotly_white",
    ):
        """
        Plot rolling drawdown, skew, kurtosis, and gini z-scores with window dropdown.
        """
        if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
            raise ValueError("metrics_by_window must be a non-empty mapping of window -> metric map.")

        if window_options is None:
            window_options = list(metrics_by_window.keys())
        else:
            try:
                window_options = [int(w) for w in window_options]
            except Exception as exc:
                raise ValueError("window_options must be iterable integers.") from exc
            window_options = [w for w in window_options if w in metrics_by_window]

        if not window_options:
            raise ValueError("No valid window options available for plotting.")

        if default_window not in window_options:
            default_window = self._preferred_numeric_window(window_options) or window_options[0]

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Rolling Max Drawdown",
                "Rolling Skew Z-Score",
                "Rolling Excess Kurtosis Z-Score",
                "Rolling Gini Coefficient Z-Score",
            ),
        )

        traces_per_window = None
        zscore_index_candidates = []

        for window in window_options:
            visible = window == default_window
            metric_set = metrics_by_window.get(window, {})
            drawdown_series = metric_set.get("max_drawdown", pd.Series(dtype=float)).dropna()
            skew_series = metric_set.get("skew_z", pd.Series(dtype=float)).dropna()
            kurtosis_series = metric_set.get("kurtosis_z", pd.Series(dtype=float)).dropna()
            gini_series = metric_set.get("gini_z", pd.Series(dtype=float)).dropna()

            trace_start = len(fig.data)

            fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series,
                    mode="lines",
                    name=f"{window}-Day Max Drawdown",
                    line=dict(color="red"),
                    fill="tozeroy",
                    visible=visible,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=skew_series.index,
                    y=skew_series,
                    mode="lines",
                    name=f"{window}-Day Skew Z-Score",
                    line=dict(color="blue"),
                    visible=visible,
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=kurtosis_series.index,
                    y=kurtosis_series,
                    mode="lines",
                    name=f"{window}-Day Excess Kurtosis Z-Score",
                    line=dict(color="purple"),
                    visible=visible,
                ),
                row=3,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=gini_series.index,
                    y=gini_series,
                    mode="lines",
                    name=f"{window}-Day Gini Coefficient Z-Score",
                    line=dict(color="green"),
                    visible=visible,
                ),
                row=4,
                col=1,
            )

            added_traces = len(fig.data) - trace_start
            if traces_per_window is None:
                traces_per_window = added_traces

            for series in (skew_series, kurtosis_series, gini_series):
                if not series.empty:
                    zscore_index_candidates.append(series.index)

        if traces_per_window is None or traces_per_window <= 0:
            traces_per_window = 4

        x_ref = max(zscore_index_candidates, key=len, default=pd.Index([]))
        if len(x_ref) > 0:
            for row in (2, 3, 4):
                add_mean_reference_line(fig, row, x_ref)
                add_sigma_reference_lines(fig, row, x_ref, line_color="gray")

        total_traces = len(fig.data)
        first_constant_trace = traces_per_window * len(window_options)
        constant_trace_indices = list(range(first_constant_trace, total_traces))

        buttons = []
        for idx, window in enumerate(window_options):
            visibility = build_visibility_mask(
                total_traces=total_traces,
                active_window_index=idx,
                traces_per_window=traces_per_window,
                constant_trace_indices=constant_trace_indices,
            )
            buttons.append(
                dict(
                    label=str(window),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": self._header_title(f"{ticker_label} Rolling Risk Metrics Z-Scores ({window}-Day Window)")},
                    ],
                )
            )

        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=1, col=1)
        fig.update_yaxes(title_text="Skew Z", row=2, col=1)
        fig.update_yaxes(title_text="Excess Kurtosis Z", row=3, col=1)
        fig.update_yaxes(title_text="Gini Z", row=4, col=1)

        fig.update_layout(
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.1,
                )
            ],
            title=self._header_title(f"{ticker_label} Rolling Risk Metrics Z-Scores ({default_window}-Day Window)"),
            height=1500,
            margin=self._header_margin(),
            template=template,
            showlegend=False,
        )
        return fig

    def plot_momentum_window_diagnostics(
        self,
        diagnostics_context,
        ticker_label="Asset",
        template="plotly_white",
    ):
        """
        Build momentum-window diagnostics figures from a precomputed context payload.

        Returns
        -------
        dict[str, go.Figure]
            {
                "optimal_window",
                "optimal_window_histogram",
                "sharpe_mean_median",
                "volatility_mean_median",
                "sharpe_surface_3d",
            }
        """
        required_keys = {
            "window_sizes",
            "highlight_windows",
            "sharpe_table",
            "optimal_windows_int",
            "mean_sharpe",
            "median_sharpe",
            "mean_volatility",
            "median_volatility",
            "sharpe_surface",
            "surface_years",
        }
        if not isinstance(diagnostics_context, Mapping):
            raise TypeError("diagnostics_context must be a mapping.")
        missing = [key for key in required_keys if key not in diagnostics_context]
        if missing:
            raise ValueError(f"diagnostics_context missing required keys: {missing}")

        window_sizes = diagnostics_context["window_sizes"]
        highlight_windows = diagnostics_context["highlight_windows"]
        sharpe_table = diagnostics_context["sharpe_table"]
        optimal_windows_int = diagnostics_context["optimal_windows_int"]
        mean_sharpe = diagnostics_context["mean_sharpe"]
        median_sharpe = diagnostics_context["median_sharpe"]
        mean_volatility = diagnostics_context["mean_volatility"]
        median_volatility = diagnostics_context["median_volatility"]
        sharpe_surface = diagnostics_context["sharpe_surface"]
        surface_years = diagnostics_context["surface_years"]

        line_style_map = {
            7: ("orange", "7 Days"),
            21: ("red", "21 Days"),
            50: ("blue", "50 Days"),
            200: ("green", "200 Days"),
        }

        def add_reference_vlines(fig):
            for window in highlight_windows:
                color, label = line_style_map.get(window, ("gray", f"{window} Days"))
                fig.add_vline(
                    x=window,
                    line_color=color,
                    line_dash="dash",
                    annotation_text=label,
                    annotation_position="top left",
                )

        fig_optimal = go.Figure()
        fig_optimal.add_trace(
            go.Scatter(
                x=sharpe_table.index,
                y=sharpe_table["Optimal_Window"],
                mode="lines",
                name="Optimal Window Size",
            )
        )
        fig_optimal.update_layout(
            title="Rolling Optimal Momentum Window",
            xaxis_title="Date",
            yaxis_title="Window Size (Days)",
            template=template,
            showlegend=False,
        )

        fig_dist = px.histogram(
            optimal_windows_int,
            nbins=len(window_sizes),
            labels={"value": "Optimal Window Size (Days)", "count": "Frequency"},
            title="Distribution of Optimal Sharpe Momentum Windows Over Time",
        )
        add_reference_vlines(fig_dist)
        fig_dist.update_layout(bargap=0.1, template=template)

        fig_mean_median = go.Figure()
        fig_mean_median.add_trace(
            go.Scatter(
                x=mean_sharpe.index,
                y=mean_sharpe.values,
                mode="lines+markers",
                name="Mean Sharpe Ratio",
            )
        )
        fig_mean_median.add_trace(
            go.Scatter(
                x=median_sharpe.index,
                y=median_sharpe.values,
                mode="lines+markers",
                name="Median Sharpe Ratio",
            )
        )
        fig_mean_median.update_layout(
            title="Mean and Median Sharpe Ratios Across All Dates by Momentum Window",
            xaxis_title="Momentum Window Size (Days)",
            yaxis_title="Sharpe Ratio",
            template=template,
        )
        add_reference_vlines(fig_mean_median)

        fig_volatility = go.Figure()
        fig_volatility.add_trace(
            go.Scatter(
                x=mean_volatility.index,
                y=mean_volatility.values,
                mode="lines+markers",
                name="Mean Volatility",
            )
        )
        fig_volatility.add_trace(
            go.Scatter(
                x=median_volatility.index,
                y=median_volatility.values,
                mode="lines+markers",
                name="Median Volatility",
            )
        )
        fig_volatility.update_layout(
            title="Mean and Median Volatility Across All Dates by Momentum Window",
            xaxis_title="Momentum Window Size (Days)",
            yaxis_title="Volatility (Rolling Std of Returns)",
            template=template,
        )
        add_reference_vlines(fig_volatility)

        def create_vertical_plane(x_val, y_vals, z_min, z_max, opacity=0.3, color="red"):
            y_grid, z_grid = np.meshgrid(y_vals, np.linspace(z_min, z_max, 2))
            x_grid = np.full_like(y_grid, x_val)
            return go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                showscale=False,
                opacity=opacity,
                colorscale=[[0, color], [1, color]],
                hoverinfo="skip",
            )

        if sharpe_surface.empty:
            fig_surface = go.Figure()
            fig_surface.update_layout(
                title=f"3D Surface of Sharpe Ratios by Window and Date (Last {surface_years} Years)",
                template=template,
            )
            fig_surface.add_annotation(
                text="No Sharpe surface data available.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        else:
            z_values = sharpe_surface.values
            finite_values = z_values[np.isfinite(z_values)]
            if finite_values.size == 0:
                z_min, z_max = 0.0, 1.0
            else:
                z_min, z_max = float(finite_values.min()), float(finite_values.max())
                if z_min == z_max:
                    z_max = z_min + 1.0

            date_labels = sharpe_surface.index.strftime("%Y-%m-%d")
            y_vals = np.arange(len(sharpe_surface))

            surface_trace = go.Surface(
                z=sharpe_surface.values,
                x=sharpe_surface.columns,
                y=y_vals,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe Ratio"),
                name="Sharpe Surface",
            )

            traces = [surface_trace]
            plane_color_map = {7: "orange", 21: "red", 50: "blue", 200: "green"}
            for window in highlight_windows:
                traces.append(
                    create_vertical_plane(
                        window,
                        y_vals,
                        z_min,
                        z_max,
                        opacity=0.3,
                        color=plane_color_map.get(window, "gray"),
                    )
                )

            tick_step = max(1, len(date_labels) // 10)
            fig_surface = go.Figure(data=traces)
            fig_surface.update_layout(
                title=f"3D Surface of Sharpe Ratios by Window and Date (Last {surface_years} Years) with Highlighted Windows",
                scene=dict(
                    xaxis_title="Momentum Window Size (Days)",
                    yaxis_title="Date",
                    yaxis=dict(
                        tickmode="array",
                        tickvals=np.arange(0, len(date_labels), step=tick_step),
                        ticktext=date_labels[::tick_step],
                    ),
                    zaxis_title="Sharpe Ratio",
                ),
                height=900,
                template=template,
            )

        return {
            "optimal_window": fig_optimal,
            "optimal_window_histogram": fig_dist,
            "sharpe_mean_median": fig_mean_median,
            "volatility_mean_median": fig_volatility,
            "sharpe_surface_3d": fig_surface,
        }

    
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
        default_window = self._preferred_numeric_window(metric_windows_map[default_metric]) or metric_windows_map[default_metric][0]

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
        timeframe_menu = self._dropdown_menu(
            buttons=timeframe_buttons,
            x=0.22,
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
                "title": self._header_title(title_text),
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
            return self._dropdown_menu(
                buttons=buttons,
                x=0.01,
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
            margin=self._header_margin(),
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
                metric_default_window = self._preferred_numeric_window(metric_windows) or metric_windows[0]
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

            metric_menu_template = self._dropdown_menu(
                buttons=metric_buttons,
                x=0.43,
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

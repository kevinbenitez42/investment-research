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
from .candlestick_plotter import CandleStickPlotter
from .figure_helpers import (
    add_horizontal_zone,
    add_horizontal_zone_trace,
    add_mean_reference_line,
    add_sigma_reference_lines,
    add_std_annotations,
    add_zone_annotation,
    build_detail_visibility_mask,
    build_time_range_buttons,
    build_visibility_mask,
)
from Quantapp.analytics.series_utils import calculate_zscore

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

    @staticmethod
    def _coerce_timestamp(value):
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, datetime):
            return pd.Timestamp(value)
        if isinstance(value, np.datetime64):
            return pd.Timestamp(value)
        if isinstance(value, str):
            try:
                coerced = pd.to_datetime(value, errors="raise")
            except (TypeError, ValueError):
                return None
            return None if pd.isna(coerced) else pd.Timestamp(coerced)
        return None

    @classmethod
    def _trace_datetime_bounds(cls, traces):
        starts = []
        ends = []
        for trace in traces:
            x_values = getattr(trace, "x", None)
            if x_values is None or len(x_values) == 0:
                continue

            start = cls._coerce_timestamp(x_values[0])
            end = cls._coerce_timestamp(x_values[-1])
            if start is None or end is None:
                continue

            starts.append(start)
            ends.append(end)

        if not starts:
            return None, None
        return min(starts), max(ends)

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
        Plot Sharpe/Sortino z-scores, raw ratios, and Sortino-minus-Sharpe spread z-scores.
        """
        if not isinstance(term_config_map, Mapping) or not term_config_map:
            raise ValueError("term_config_map must be a non-empty mapping.")

        term_labels = list(term_config_map.keys())
        if default_label not in term_config_map:
            default_label = self._preferred_window_label(term_config_map) or term_labels[0]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.30, 0.38, 0.32],
            subplot_titles=(
                "Risk-Adjusted Return Z-Score Comparison",
                "Rolling Sharpe and Sortino Ratios",
                "Sortino-Sharpe Spread (z-score)",
            ),
        )
        fig.update_xaxes(matches="x3", row=1, col=1)
        fig.update_xaxes(matches="x3", row=2, col=1)

        term_trace_map = {}
        term_default_ranges = {}
        term_full_ranges = {}
        for term_label in term_labels:
            cfg = term_config_map[term_label]
            sharpe = cfg["sharpe"]
            sortino = cfg["sortino"]
            spread = cfg.get("spread", sortino - sharpe)
            sharpe_zscore = cfg.get("sharpe_zscore")
            sortino_zscore = cfg.get("sortino_zscore")
            spread_zscore = cfg.get("spread_zscore")
            time_frame = cfg.get("time_frame", term_label)

            sharpe_clean = sharpe.dropna()
            sortino_clean = sortino.dropna()
            spread_clean = spread.dropna()
            sharpe_zscore = (
                sharpe_zscore
                if isinstance(sharpe_zscore, pd.Series)
                else calculate_zscore(sharpe_clean).dropna()
                if not sharpe_clean.empty
                else pd.Series(dtype=float)
            )
            sortino_zscore = (
                sortino_zscore
                if isinstance(sortino_zscore, pd.Series)
                else calculate_zscore(sortino_clean).dropna()
                if not sortino_clean.empty
                else pd.Series(dtype=float)
            )
            spread_zscore = (
                spread_zscore
                if isinstance(spread_zscore, pd.Series)
                else calculate_zscore(spread_clean).dropna()
                if not spread_clean.empty
                else pd.Series(dtype=float)
            )
            mean_sharpe = sharpe_clean.mean()
            mean_sortino = sortino_clean.mean()

            zscore_x_ref = max(
                [series.index for series in (sharpe_zscore, sortino_zscore, spread_zscore) if not series.empty],
                key=len,
                default=pd.Index([]),
            )
            non_empty_series = [
                series
                for series in (sharpe_clean, sortino_clean, sharpe_zscore, sortino_zscore, spread_zscore)
                if not series.empty
            ]

            visible = term_label == default_label
            trace_indices = []

            fig.add_trace(
                go.Scatter(
                    x=sharpe_zscore.index,
                    y=sharpe_zscore,
                    mode="lines",
                    name=f"Sharpe Z-Score ({time_frame}-day)",
                    line=dict(color="blue"),
                    visible=visible,
                    legendgroup="Sharpe Z",
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=sortino_zscore.index,
                    y=sortino_zscore,
                    mode="lines",
                    name=f"Sortino Z-Score ({time_frame}-day)",
                    line=dict(color="orange"),
                    visible=visible,
                    legendgroup="Sortino Z",
                ),
                row=1,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            if len(zscore_x_ref) > 0:
                add_mean_reference_line(
                    fig,
                    1,
                    zscore_x_ref,
                    line_color="rgba(0, 0, 0, 0.70)",
                    visible=visible,
                )
                trace_indices.append(len(fig.data) - 1)
                add_sigma_reference_lines(
                    fig,
                    1,
                    zscore_x_ref,
                    levels=(1, 2),
                    line_color="rgba(110, 110, 110, 0.45)",
                    visible=visible,
                )
                trace_indices.extend(range(len(fig.data) - 4, len(fig.data)))

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
                row=2,
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
                row=2,
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
                row=2,
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
                row=2,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            fig.add_trace(
                go.Scatter(
                    x=spread_zscore.index,
                    y=spread_zscore,
                    mode="lines",
                    name=f"Sortino-Sharpe Spread ({time_frame}-day)",
                    line=dict(color="green"),
                    visible=visible,
                    legendgroup="Spread",
                ),
                row=3,
                col=1,
            )
            trace_indices.append(len(fig.data) - 1)

            if len(zscore_x_ref) > 0:
                add_horizontal_zone_trace(
                    fig,
                    3,
                    zscore_x_ref,
                    2,
                    3,
                    "rgba(180, 0, 0, 0.20)",
                    visible=visible,
                )
                trace_indices.append(len(fig.data) - 1)
                add_horizontal_zone_trace(
                    fig,
                    3,
                    zscore_x_ref,
                    -1,
                    -0.5,
                    "rgba(0, 128, 0, 0.18)",
                    visible=visible,
                )
                trace_indices.append(len(fig.data) - 1)
                fig.add_trace(
                    go.Scatter(
                        x=zscore_x_ref,
                        y=np.full(len(zscore_x_ref), 3.0),
                        mode="lines",
                        line=dict(color="rgba(128, 0, 128, 0.75)", dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )
                trace_indices.append(len(fig.data) - 1)
                add_mean_reference_line(
                    fig,
                    3,
                    zscore_x_ref,
                    line_color="rgba(0, 0, 0, 0.70)",
                    visible=visible,
                )
                trace_indices.append(len(fig.data) - 1)
                for level in (-1, -0.5, 1, 2):
                    fig.add_trace(
                        go.Scatter(
                            x=zscore_x_ref,
                            y=np.full(len(zscore_x_ref), float(level)),
                            mode="lines",
                            line=dict(color="rgba(110, 110, 110, 0.45)", dash="dot"),
                            hoverinfo="skip",
                            showlegend=False,
                            visible=visible,
                        ),
                        row=3,
                        col=1,
                    )
                    trace_indices.append(len(fig.data) - 1)

            if not spread_zscore.empty:
                fig.add_trace(
                    go.Scatter(
                        x=[spread_zscore.index[-1]],
                        y=[spread_zscore.iloc[-1]],
                        mode="markers+text",
                        text=[f"Latest z: {spread_zscore.iloc[-1]:.2f}"],
                        textposition="middle right",
                        marker=dict(color="purple", size=6),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )
                trace_indices.append(len(fig.data) - 1)

            term_trace_map[term_label] = trace_indices
            if non_empty_series:
                max_index = max(series.index.max() for series in non_empty_series)
                min_index = min(series.index.min() for series in non_empty_series)
                term_full_ranges[term_label] = [min_index, max_index]
                term_default_ranges[term_label] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
            else:
                term_full_ranges[term_label] = None
                term_default_ranges[term_label] = None

        total_traces = len(fig.data)
        buttons = []
        for term_label in term_labels:
            visibility = [False] * total_traces
            for trace_idx in term_trace_map[term_label]:
                visibility[trace_idx] = True

            layout_updates = {
                "title": self._header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({term_label})")
            }
            if term_default_ranges.get(term_label) is not None:
                layout_updates["xaxis"] = {"range": term_default_ranges[term_label]}
                layout_updates["xaxis2"] = {"range": term_default_ranges[term_label]}
                layout_updates["xaxis3"] = {"range": term_default_ranges[term_label]}

            buttons.append(
                dict(
                    label=term_label,
                    method="update",
                    args=[
                        {"visible": visibility},
                        layout_updates,
                    ],
                )
            )

        available_ranges = [date_range for date_range in term_full_ranges.values() if date_range is not None]
        if available_ranges:
            global_start = min(date_range[0] for date_range in available_ranges)
            global_end = max(date_range[1] for date_range in available_ranges)
            fig.update_xaxes(range=term_default_ranges[default_label] or [global_start, global_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(global_start, global_end, axis_count=3),
                x=0.22 if len(term_labels) > 1 else 0.01,
            )
        else:
            time_range_menu = None

        updatemenus = []
        if len(term_labels) > 1:
            updatemenus.append(
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.01,
                    active=term_labels.index(default_label),
                )
            )
        if time_range_menu is not None:
            updatemenus.append(time_range_menu)

        fig.update_layout(
            template="plotly_white",
            height=1100,
            margin=self._header_margin(),
            legend=dict(x=0.01, y=0.99),
            xaxis3_title="Date",
            yaxis_title="Z-Score",
            yaxis2_title="Ratio Value",
            yaxis3_title="Z-Score",
            title=self._header_title(f"Sharpe & Sortino Analysis for {ticker_label} ({default_label})"),
            updatemenus=updatemenus,
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
        Plot benchmark Sharpe-spread z-score summary with optional term/time controls.
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
        term_default_ranges = {}
        term_full_ranges = {}
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
                term_full_ranges[term] = [min_index, max_index]
                term_default_ranges[term] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
            else:
                term_full_ranges[term] = None
                term_default_ranges[term] = None

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
            if term_default_ranges.get(term) is not None:
                layout_updates["xaxis"] = {"range": term_default_ranges[term]}

            timeframe_buttons.append(
                dict(
                    label=f"{term.title()} ({time_frame_map[term]})",
                    method="update",
                    args=[{"visible": visibility}, layout_updates],
                )
            )

        available_ranges = [date_range for date_range in term_full_ranges.values() if date_range is not None]
        if available_ranges:
            global_start = min(date_range[0] for date_range in available_ranges)
            global_end = max(date_range[1] for date_range in available_ranges)
            fig.update_xaxes(range=term_default_ranges[default_term] or [global_start, global_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(global_start, global_end),
                x=0.22 if len(term_order) > 1 else 0.0,
            )
        else:
            time_range_menu = None

        updatemenus = []
        if len(term_order) > 1:
            updatemenus.append(
                self._dropdown_menu(
                    buttons=timeframe_buttons,
                    x=0.0,
                )
            )
        if time_range_menu is not None:
            updatemenus.append(time_range_menu)

        fig.update_yaxes(title_text="Sharpe Spread Z-Score", row=1, col=1)
        fig.update_layout(
            title=self._header_title(
                f"{ticker_label} {default_term.title()} Sharpe Spread Z-Scores vs Benchmarks ({time_frame_map[default_term]}-Day)"
            ),
            height=650,
            margin=self._header_margin(),
            template=template,
            showlegend=True,
            updatemenus=updatemenus,
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
            add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, -1, 1, "rgba(211, 211, 211, 0.18)")
            add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, -2, -1, "rgba(0, 128, 0, 0.30)")
            add_horizontal_zone_trace(detail_fig, 1, detail_x_ref, 1, 2, "rgba(180, 0, 0, 0.30)")
            add_zone_annotation(detail_fig, 1, -1, 1, "Neutral", "rgba(235, 235, 235, 0.95)")
            add_zone_annotation(detail_fig, 1, -0.85, -0.25, "Bullish Neutral (on the way up) ↑", "rgba(235, 235, 235, 0.90)")
            add_zone_annotation(detail_fig, 1, 0.25, 0.85, "Bearish Neutral (on the way down) ↓", "rgba(235, 235, 235, 0.90)")
            add_zone_annotation(detail_fig, 1, -2, -1, "Accumulate", "rgba(235, 255, 235, 0.95)")
            add_zone_annotation(detail_fig, 1, 1, 2, "Liquidate", "rgba(255, 235, 235, 0.95)")
            add_sigma_reference_lines(detail_fig, 1, detail_x_ref)
            add_mean_reference_line(detail_fig, 2, detail_x_ref)
            add_mean_reference_line(detail_fig, 3, detail_x_ref)
            add_mean_reference_line(detail_fig, 4, detail_x_ref)

        total_traces = len(detail_fig.data)
        buttons = []
        single_term_view = len(term_order) == 1
        for idx, (symbol, term) in enumerate(detail_view_order):
            visibility = build_detail_visibility_mask(dynamic_trace_count, total_traces, idx, traces_per_view)
            buttons.append(
                dict(
                    label=symbol if single_term_view else f"{symbol} | {term.title()} ({time_frame_map[term]})",
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

        detail_start, detail_end = self._trace_datetime_bounds(detail_fig.data)
        if detail_start is not None and detail_end is not None:
            detail_default_start = max(detail_start, detail_end - pd.DateOffset(years=3))
            detail_fig.update_xaxes(range=[detail_default_start, detail_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(detail_start, detail_end, axis_count=4),
                x=0.18 if len(detail_view_order) > 1 else 0.0,
            )
        else:
            time_range_menu = None

        updatemenus = []
        if len(detail_view_order) > 1:
            updatemenus.append(
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                )
            )
        if time_range_menu is not None:
            updatemenus.append(time_range_menu)

        detail_fig.update_layout(
            title=self._header_title(
                f"{ticker_label} vs {default_benchmark} Risk-Adjusted Return Decomposition [{default_term.title()} {time_frame_map[default_term]}-Day]"
            ),
            height=1350,
            margin=self._header_margin(),
            template=template,
            updatemenus=updatemenus,
        )
        return detail_fig

    def plot_risk_distribution_zscores(
        self,
        metrics_by_window,
        window_options=None,
        default_window=None,
        ticker_label="Asset",
        template="plotly_dark",
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

        metric_colors = {
            "drawdown": "#1f77b4",
            "skew": "#ff7f0e",
            "kurtosis": "#2ca02c",
            "gini": "#d62728",
        }
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
                    line=dict(color=metric_colors["drawdown"]),
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
                    line=dict(color=metric_colors["skew"]),
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
                    line=dict(color=metric_colors["kurtosis"]),
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
                    line=dict(color=metric_colors["gini"]),
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
            for row in (2, 4):
                add_mean_reference_line(fig, row, x_ref)
                add_sigma_reference_lines(
                    fig,
                    row,
                    x_ref,
                    levels=(1, 1.5, 2, 3) if row == 2 else (1, 2, 3),
                )
            add_mean_reference_line(fig, 3, x_ref)
            for value in (-1.5, -1, -0.5, 1, 2, 3):
                fig.add_hline(
                    y=value,
                    row=3,
                    col=1,
                    line_dash="dot",
                    line_color="rgba(220, 220, 220, 0.55)",
                    line_width=1,
                )

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
        add_horizontal_zone(
            fig,
            row=2,
            col=1,
            y0=-3,
            y1=-1.5,
            fillcolor="rgba(180, 0, 0, 0.30)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=2,
            col=1,
            y0=-1.5,
            y1=1.5,
            fillcolor="rgba(211, 211, 211, 0.18)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=2,
            col=1,
            y0=-1,
            y1=1,
            fillcolor="rgba(169, 169, 169, 0.24)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=2,
            col=1,
            y0=1.5,
            y1=3,
            fillcolor="rgba(180, 0, 0, 0.30)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_zone_annotation(
            fig,
            row=2,
            col=1,
            y0=-1.5,
            y1=-1,
            text="More Likely to Have Larger Upside Moves ↑",
            font_color="rgba(235, 235, 235, 0.92)",
        )
        add_zone_annotation(
            fig,
            row=2,
            col=1,
            y0=1,
            y1=1.5,
            text="More Likely to Have Larger Downside Moves ↓",
            font_color="rgba(235, 235, 235, 0.92)",
        )

        fig.update_layout(
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                )
            ],
            title=self._header_title(f"{ticker_label} Rolling Risk Metrics Z-Scores ({default_window}-Day Window)"),
            height=1500,
            margin=self._header_margin(),
            template=template,
            showlegend=False,
        )
        return fig

    def plot_rolling_max_drawdown(
        self,
        metrics_by_window,
        window_options=None,
        default_window=None,
        ticker_label="Asset",
        template="plotly_dark",
    ):
        """
        Plot textbook rolling max drawdown with a window dropdown.
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

        fig = go.Figure()
        drawdown_series_map = {}

        for window in window_options:
            drawdown_series = metrics_by_window.get(window, {}).get("max_drawdown", pd.Series(dtype=float)).dropna()
            drawdown_series_map[window] = drawdown_series
            fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series,
                    mode="lines",
                    name=f"{window}-Day Max Drawdown",
                    line=dict(color="#1f77b4"),
                    fill="tozeroy",
                    visible=window == default_window,
                )
            )

        buttons = []
        total_traces = len(fig.data)
        for idx, window in enumerate(window_options):
            visibility = [trace_idx == idx for trace_idx in range(total_traces)]
            buttons.append(
                dict(
                    label=str(window),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": self._header_title(f"{ticker_label} Textbook Rolling Max Drawdown ({window}-Day Window)")},
                    ],
                )
            )

        available_indexes = [series.index for series in drawdown_series_map.values() if not series.empty]
        if available_indexes:
            global_start = min(index[0] for index in available_indexes)
            global_end = max(index[-1] for index in available_indexes)
            default_start = max(global_start, global_end - pd.DateOffset(years=3))
            fig.update_xaxes(range=[default_start, global_end])
            time_range_menu = self._dropdown_menu(
                buttons=build_time_range_buttons(global_start, global_end),
                x=0.18,
            )
            updatemenus = [
                self._dropdown_menu(buttons=buttons, x=0.0),
                time_range_menu,
            ]
        else:
            updatemenus = [self._dropdown_menu(buttons=buttons, x=0.0)]

        fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
        fig.update_layout(
            updatemenus=updatemenus,
            title=self._header_title(f"{ticker_label} Textbook Rolling Max Drawdown ({default_window}-Day Window)"),
            height=650,
            margin=self._header_margin(),
            template=template,
            showlegend=False,
        )
        return fig

    def plot_peak_pullback_and_rolling_drawdown(
        self,
        price_frame,
        metrics_by_window,
        window_options=None,
        default_window=None,
        ticker_label="Asset",
        title=None,
        display_days=None,
        template="plotly_dark",
    ):
        """
        Plot underwater drawdown against textbook rolling max drawdown,
        plus rolling recovery time, with a shared window dropdown.
        """
        if "Close" not in price_frame.columns:
            raise ValueError("price_frame must contain a 'Close' column.")
        if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
            raise ValueError("metrics_by_window must be a non-empty mapping of window -> metric map.")

        plot_data = price_frame.copy()
        if not isinstance(plot_data.index, pd.DatetimeIndex):
            plot_data.index = pd.to_datetime(plot_data.index)
        plot_data = plot_data.sort_index().dropna(subset=["Close"])
        if plot_data.empty:
            raise ValueError("No non-null close data available for pullback plotting.")

        if window_options is None:
            window_options = list(metrics_by_window.keys())
        else:
            try:
                window_options = [int(window) for window in window_options]
            except Exception as exc:
                raise ValueError("window_options must be iterable integers.") from exc
            window_options = [window for window in window_options if window in metrics_by_window]

        if not window_options:
            raise ValueError("No valid window options available for plotting.")

        if default_window not in window_options:
            default_window = self._preferred_numeric_window(window_options) or window_options[0]

        if display_days is not None:
            display_days = self._coerce_positive_int(display_days)
            if display_days is None:
                raise ValueError("display_days must be a positive integer.")

        combined_title = str(
            title or f"{ticker_label} Drawdown and Recovery Profile"
        )

        def _annotation_payload(annotation):
            if hasattr(annotation, "to_plotly_json"):
                return copy.deepcopy(annotation).to_plotly_json()
            return copy.deepcopy(dict(annotation))

        def _window_title(window):
            return f"{combined_title} ({window}-Day Window)"

        def _percentile_rank(series, value):
            cleaned = pd.Series(series).dropna()
            if cleaned.empty or pd.isna(value):
                return np.nan
            return float(cleaned.le(value).mean() * 100)

        def _format_percentile(value):
            return "n/a" if pd.isna(value) else f"{value:.0f}th pctile"

        def _line_annotation(*, x, y, text, color, xref, yref, yshift=0):
            if x is None or pd.isna(y):
                return None
            return dict(
                x=x,
                y=float(y),
                xref=xref,
                yref=yref,
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=8,
                yshift=yshift,
                font=dict(size=10, color=color),
                bgcolor="rgba(15, 23, 42, 0.70)",
                bordercolor=color,
                borderwidth=1,
            )

        def _slice_visible(series, start=None, end=None):
            return CandleStickPlotter._slice_series_to_range(series, start=start, end=end)

        def _constant_axis_series(value):
            if pd.isna(value):
                return None
            return pd.Series([float(value)])

        def _rolling_recovery_time(close_series, window):
            def recovery_window(window_values):
                values = np.asarray(window_values, dtype=float)
                values = values[~np.isnan(values)]
                if values.size == 0:
                    return np.nan

                peaks = np.maximum.accumulate(values)
                drawdowns = values / peaks - 1.0
                trough_idx = int(np.nanargmin(drawdowns))
                if trough_idx >= values.size - 1:
                    return np.nan

                peak_at_trough = peaks[trough_idx]
                recovery_candidates = np.flatnonzero(values[trough_idx + 1:] >= peak_at_trough)
                if recovery_candidates.size == 0:
                    return np.nan

                recovery_idx = trough_idx + 1 + int(recovery_candidates[0])
                return float(recovery_idx - trough_idx)

            return close_series.rolling(window=window).apply(recovery_window, raw=True).dropna()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            row_heights=[0.58, 0.42],
            subplot_titles=(
                "Underwater vs Textbook Rolling Max Drawdown",
                "Rolling Recovery Time (Trough to Recovered High)",
            ),
        )
        subplot_title_annotations = [_annotation_payload(annotation) for annotation in fig.layout.annotations]

        trace_state_map = {}
        annotation_map = {}
        drawdown_series_map = {}
        recovery_series_map = {}

        for window in window_options:
            rolling_peak = plot_data["Close"].rolling(window=window, min_periods=1).max()
            underwater_series = (
                plot_data["Close"]
                .div(rolling_peak)
                .sub(1.0)
                .dropna()
            )
            drawdown_series = (
                metrics_by_window.get(window, {})
                .get("max_drawdown", pd.Series(dtype=float))
                .dropna()
            )
            recovery_series = _rolling_recovery_time(plot_data["Close"], window)

            shared_index = underwater_series.index.intersection(drawdown_series.index)
            if display_days is not None:
                shared_index = shared_index[-display_days:]

            visible_underwater = underwater_series.reindex(shared_index).dropna()
            visible_drawdown = drawdown_series.reindex(shared_index).dropna()
            drawdown_series_map[window] = visible_drawdown
            if display_days is not None and len(shared_index) > 0:
                recovery_start = shared_index[0]
                recovery_end = shared_index[-1]
                visible_recovery = recovery_series.loc[recovery_start:recovery_end].dropna()
            else:
                visible_recovery = recovery_series
            recovery_series_map[window] = visible_recovery

            underwater_p05 = underwater_series.quantile(0.05) if not underwater_series.empty else np.nan
            drawdown_p05 = drawdown_series.quantile(0.05) if not drawdown_series.empty else np.nan
            recovery_p95 = recovery_series.quantile(0.95) if not recovery_series.empty else np.nan

            underwater_current = underwater_series.iloc[-1] if not underwater_series.empty else np.nan
            drawdown_current = drawdown_series.iloc[-1] if not drawdown_series.empty else np.nan
            recovery_current = recovery_series.iloc[-1] if not recovery_series.empty else np.nan

            underwater_rank = _percentile_rank(underwater_series, underwater_current)
            drawdown_rank = _percentile_rank(drawdown_series, drawdown_current)
            recovery_rank = _percentile_rank(recovery_series, recovery_current)

            drawdown_x_ref = visible_underwater.index if not visible_underwater.empty else visible_drawdown.index
            recovery_x_ref = visible_recovery.index
            drawdown_last_x = drawdown_x_ref[-1] if len(drawdown_x_ref) > 0 else None
            recovery_last_x = recovery_x_ref[-1] if len(recovery_x_ref) > 0 else None

            default_states = [True, True, True, True, True, True]
            visible_states = [
                state if window == default_window else False
                for state in default_states
            ]
            trace_indices = []

            fig.add_trace(
                go.Scatter(
                    x=visible_underwater.index,
                    y=visible_underwater,
                    mode="lines",
                    name=f"{window}-Day Underwater",
                    line=dict(color="#0ea5e9", width=2),
                    visible=visible_states[0],
                ),
                row=1,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[0]))

            fig.add_trace(
                go.Scatter(
                    x=visible_drawdown.index,
                    y=visible_drawdown,
                    mode="lines",
                    name=f"{window}-Day Textbook Max Drawdown",
                    line=dict(color="#f97316", width=2, dash="dot"),
                    visible=visible_states[1],
                ),
                row=1,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[1]))

            fig.add_trace(
                go.Scatter(
                    x=drawdown_x_ref,
                    y=[underwater_p05] * len(drawdown_x_ref),
                    mode="lines",
                    name="Underwater 5th Percentile",
                    line=dict(color="rgba(14, 165, 233, 0.55)", width=1.5, dash="dash"),
                    hovertemplate="Underwater 5th percentile: %{y:.1%}<extra></extra>",
                    visible=visible_states[2],
                ),
                row=1,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[2]))

            fig.add_trace(
                go.Scatter(
                    x=drawdown_x_ref,
                    y=[drawdown_p05] * len(drawdown_x_ref),
                    mode="lines",
                    name="Textbook 5th Percentile",
                    line=dict(color="rgba(249, 115, 22, 0.55)", width=1.5, dash="dash"),
                    hovertemplate="Textbook 5th percentile: %{y:.1%}<extra></extra>",
                    visible=visible_states[3],
                ),
                row=1,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[3]))

            fig.add_trace(
                go.Scatter(
                    x=visible_recovery.index,
                    y=visible_recovery,
                    mode="lines",
                    name=f"{window}-Day Recovery Time",
                    line=dict(color="#22c55e", width=2),
                    visible=visible_states[4],
                ),
                row=2,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[4]))

            fig.add_trace(
                go.Scatter(
                    x=recovery_x_ref,
                    y=[recovery_p95] * len(recovery_x_ref),
                    mode="lines",
                    name="Recovery 95th Percentile",
                    line=dict(color="rgba(34, 197, 94, 0.60)", width=1.5, dash="dash"),
                    hovertemplate="Recovery 95th percentile: %{y:.0f} sessions<extra></extra>",
                    visible=visible_states[5],
                ),
                row=2,
                col=1,
            )
            trace_indices.append((len(fig.data) - 1, default_states[5]))

            line_annotations = [
                _line_annotation(
                    x=visible_underwater.index[-1] if not visible_underwater.empty else None,
                    y=visible_underwater.iloc[-1] if not visible_underwater.empty else np.nan,
                    text="Underwater",
                    color="#0ea5e9",
                    xref="x",
                    yref="y",
                    yshift=12,
                ),
                _line_annotation(
                    x=visible_drawdown.index[-1] if not visible_drawdown.empty else None,
                    y=visible_drawdown.iloc[-1] if not visible_drawdown.empty else np.nan,
                    text="Textbook Max DD",
                    color="#f97316",
                    xref="x",
                    yref="y",
                    yshift=-12,
                ),
                _line_annotation(
                    x=drawdown_last_x,
                    y=underwater_p05,
                    text="Underwater 5th pct",
                    color="rgba(14, 165, 233, 0.90)",
                    xref="x",
                    yref="y",
                    yshift=-12,
                ),
                _line_annotation(
                    x=drawdown_last_x,
                    y=drawdown_p05,
                    text="Textbook 5th pct",
                    color="rgba(249, 115, 22, 0.90)",
                    xref="x",
                    yref="y",
                    yshift=12,
                ),
                _line_annotation(
                    x=visible_recovery.index[-1] if not visible_recovery.empty else None,
                    y=visible_recovery.iloc[-1] if not visible_recovery.empty else np.nan,
                    text="Recovery Time",
                    color="#22c55e",
                    xref="x2",
                    yref="y2",
                    yshift=12,
                ),
                _line_annotation(
                    x=recovery_last_x,
                    y=recovery_p95,
                    text="Recovery 95th pct",
                    color="rgba(34, 197, 94, 0.90)",
                    xref="x2",
                    yref="y2",
                    yshift=-12,
                ),
            ]

            trace_state_map[window] = trace_indices
            annotation_map[window] = subplot_title_annotations + [annotation for annotation in line_annotations if annotation is not None] + [
                dict(
                    x=0.995,
                    y=0.88,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                    bgcolor="rgba(15, 23, 42, 0.72)",
                    bordercolor="rgba(148, 163, 184, 0.35)",
                    borderwidth=1,
                    text=(
                        f"Latest underwater: {underwater_current:.1%} ({_format_percentile(underwater_rank)})"
                        "<br>"
                        f"Latest textbook: {drawdown_current:.1%} ({_format_percentile(drawdown_rank)})"
                    ),
                ),
                dict(
                    x=0.995,
                    y=0.26,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                    bgcolor="rgba(15, 23, 42, 0.72)",
                    bordercolor="rgba(148, 163, 184, 0.35)",
                    borderwidth=1,
                    text=(
                        f"Latest completed recovery: {recovery_current:.0f} sessions ({_format_percentile(recovery_rank)})"
                        if not pd.isna(recovery_current)
                        else "Latest completed recovery: n/a"
                    ),
                ),
            ]

        fig.update_xaxes(
            title_text="Date",
            type="date",
            showgrid=True,
            zeroline=False,
            rangebreaks=[dict(bounds=["sat", "mon"])],
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=1, col=1)
        fig.update_xaxes(
            title_text="Date",
            type="date",
            rangebreaks=[dict(bounds=["sat", "mon"])],
            matches="x",
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Sessions", rangemode="tozero", row=2, col=1)
        fig.add_hline(
            y=0,
            row=1,
            col=1,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.45)",
            line_width=1,
        )

        total_traces = len(fig.data)
        buttons = []
        for window in window_options:
            visibility = [False] * total_traces
            for trace_idx, state in trace_state_map[window]:
                visibility[trace_idx] = state
            buttons.append(
                dict(
                    label=f"{window}-Day",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": self._header_title(_window_title(window)),
                            "annotations": annotation_map[window],
                        },
                    ],
                )
            )

        updatemenus = [
            self._dropdown_menu(
                buttons=buttons,
                x=0.0,
                active=window_options.index(default_window),
            )
        ]

        available_indexes = [series.index for series in drawdown_series_map.values() if not series.empty]
        if available_indexes:
            global_start = min(index[0] for index in available_indexes)
            global_end = max(index[-1] for index in available_indexes)
            recovery_indexes = [series.index for series in recovery_series_map.values() if not series.empty]
            if recovery_indexes:
                global_start = min(global_start, min(index[0] for index in recovery_indexes))
                global_end = max(global_end, max(index[-1] for index in recovery_indexes))
            default_start = global_start
            fig.update_xaxes(range=[default_start, global_end], row=1, col=1)
            fig.update_xaxes(range=[default_start, global_end], row=2, col=1)

            def _drawdown_range(years=None):
                start = global_start if years is None else max(global_start, global_end - pd.DateOffset(years=years))
                return {
                    "xaxis.range": [start, global_end],
                    "xaxis2.range": [start, global_end],
                }

            updatemenus.append(
                self._dropdown_menu(
                    buttons=[
                        dict(label="Full", method="relayout", args=[_drawdown_range()]),
                        dict(label="10 Years", method="relayout", args=[_drawdown_range(10)]),
                        dict(label="5 Years", method="relayout", args=[_drawdown_range(5)]),
                        dict(label="3 Years", method="relayout", args=[_drawdown_range(3)]),
                        dict(label="1 Year", method="relayout", args=[_drawdown_range(1)]),
                    ],
                    x=0.18,
                )
            )

        fig.update_layout(
            updatemenus=updatemenus,
            title=self._header_title(_window_title(default_window)),
            height=1180,
            margin=self._header_margin(top=190),
            template=template,
            showlegend=True,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            annotations=annotation_map[default_window],
        )
        return fig

    def plot_candlestick_drawdown_recovery_profile(
        self,
        price_frame,
        metrics_by_window,
        window_options=None,
        default_window=None,
        show_window_menu=True,
        ticker_label="Asset",
        title=None,
        candlestick_period=None,
        candlestick_bollinger_window=50,
        candlestick_mapped_drawdown_windows=None,
        timeframe_options=None,
        default_timeframe_label="Full",
        template="plotly_dark",
    ):
        """
        Plot a stacked candlestick, drawdown, and recovery profile with linked zoom.
        """
        if "Close" not in price_frame.columns:
            raise ValueError("price_frame must contain a 'Close' column.")
        if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
            raise ValueError("metrics_by_window must be a non-empty mapping of window -> metric map.")

        plot_data = price_frame.copy()
        if not isinstance(plot_data.index, pd.DatetimeIndex):
            plot_data.index = pd.to_datetime(plot_data.index)
        plot_data = plot_data.sort_index().dropna(subset=["Close"])
        if plot_data.empty:
            raise ValueError("No non-null close data available for plotting.")

        if window_options is None:
            window_options = list(metrics_by_window.keys())
        else:
            try:
                window_options = [int(window) for window in window_options]
            except Exception as exc:
                raise ValueError("window_options must be iterable integers.") from exc
            window_options = [window for window in window_options if window in metrics_by_window]

        if not window_options:
            raise ValueError("No valid window options available for plotting.")

        if default_window not in window_options:
            default_window = self._preferred_numeric_window(window_options) or window_options[0]

        if candlestick_mapped_drawdown_windows is None:
            candlestick_mapped_drawdown_windows = list(window_options)
        else:
            candlestick_mapped_drawdown_windows = [
                int(window) for window in candlestick_mapped_drawdown_windows
            ]

        if timeframe_options is None:
            timeframe_options = [
                ("Full", None),
                ("10 Years", pd.DateOffset(years=10)),
                ("5 Years", pd.DateOffset(years=5)),
                ("3 Years", pd.DateOffset(years=3)),
                ("2 Years", pd.DateOffset(years=2)),
                ("1 Year", pd.DateOffset(years=1)),
                ("6 Months", pd.DateOffset(months=6)),
                ("3 Months", pd.DateOffset(months=3)),
            ]

        timeframe_labels = [str(label) for label, _ in timeframe_options]
        try:
            default_timeframe_index = next(
                idx
                for idx, label in enumerate(timeframe_labels)
                if label.lower() == str(default_timeframe_label).lower()
            )
        except StopIteration:
            default_timeframe_index = 0

        combined_title = str(
            title or f"{ticker_label} Candlestick, Drawdown, and Recovery Profile"
        )

        def _annotation_payload(annotation):
            if hasattr(annotation, "to_plotly_json"):
                return copy.deepcopy(annotation).to_plotly_json()
            return copy.deepcopy(dict(annotation))

        def _window_title(window):
            return f"{combined_title} ({window}-Day Window)"

        def _percentile_rank(series, value):
            cleaned = pd.Series(series).dropna()
            if cleaned.empty or pd.isna(value):
                return np.nan
            return float(cleaned.le(value).mean() * 100)

        def _format_percentile(value):
            return "n/a" if pd.isna(value) else f"{value:.0f}th pctile"

        def _line_annotation(*, x, y, text, color, xref, yref, yshift=0):
            if x is None or pd.isna(y):
                return None
            return dict(
                x=x,
                y=float(y),
                xref=xref,
                yref=yref,
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=8,
                yshift=yshift,
                font=dict(size=10, color=color),
                bgcolor="rgba(15, 23, 42, 0.70)",
                bordercolor=color,
                borderwidth=1,
            )

        def _slice_visible(series, start=None, end=None):
            return CandleStickPlotter._slice_series_to_range(series, start=start, end=end)

        def _constant_axis_series(value):
            if pd.isna(value):
                return None
            return pd.Series([float(value)])

        def _rolling_recovery_time(close_series, window):
            def recovery_window(window_values):
                values = np.asarray(window_values, dtype=float)
                values = values[~np.isnan(values)]
                if values.size == 0:
                    return np.nan

                peaks = np.maximum.accumulate(values)
                drawdowns = values / peaks - 1.0
                trough_idx = int(np.nanargmin(drawdowns))
                if trough_idx >= values.size - 1:
                    return np.nan

                peak_at_trough = peaks[trough_idx]
                recovery_candidates = np.flatnonzero(values[trough_idx + 1:] >= peak_at_trough)
                if recovery_candidates.size == 0:
                    return np.nan

                recovery_idx = trough_idx + 1 + int(recovery_candidates[0])
                return float(recovery_idx - trough_idx)

            return close_series.rolling(window=window).apply(recovery_window, raw=True).dropna()

        candlestick_plotter = CandleStickPlotter()
        candlestick_bundle = candlestick_plotter.build_candlestick_trace_bundle(
            ticker_data=plot_data,
            period=candlestick_period,
            bollinger_window=candlestick_bollinger_window,
            max_drawdown_price_windows=candlestick_mapped_drawdown_windows,
        )

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            row_heights=[0.50, 0.30, 0.20],
            subplot_titles=(
                f"Candlestick Price with {candlestick_bollinger_window}-Period Bollinger or Mapped MDD Overlay",
                "Underwater vs Textbook Rolling Max Drawdown",
                "Rolling Recovery Time (Trough to Recovered High)",
            ),
        )
        subplot_title_annotations = [_annotation_payload(annotation) for annotation in fig.layout.annotations]

        for trace in candlestick_bundle["traces"]:
            fig.add_trace(copy.deepcopy(trace), row=1, col=1)

        candlestick_overlay_groups = {
            overlay_name: list(trace_indices)
            for overlay_name, trace_indices in candlestick_bundle["overlay_trace_groups"].items()
        }
        overlay_trace_indices = (
            candlestick_overlay_groups["bollinger"] + candlestick_overlay_groups["mapped_mdd"]
        )
        default_overlay = (
            "mapped_mdd"
            if candlestick_overlay_groups["mapped_mdd"]
            else "bollinger"
        )
        for trace_idx in candlestick_overlay_groups["bollinger"]:
            fig.data[trace_idx].visible = default_overlay == "bollinger"
        for trace_idx in candlestick_overlay_groups["mapped_mdd"]:
            fig.data[trace_idx].visible = default_overlay == "mapped_mdd"

        static_annotations = list(subplot_title_annotations)
        if candlestick_bundle["latest_close_annotation"] is not None:
            top_annotation = _annotation_payload(candlestick_bundle["latest_close_annotation"])
            top_annotation["xref"] = "x"
            top_annotation["yref"] = "y"
            static_annotations.append(top_annotation)

        timeframe_menu_x = 0.18 if show_window_menu else 0.0
        overlay_menu_x = 0.38 if show_window_menu else 0.20
        menu_specs = []
        if show_window_menu:
            menu_specs.append(("Damage window", 0.00))
        menu_specs.append(("View timeframe", timeframe_menu_x))
        if overlay_trace_indices:
            menu_specs.append(("Overlay mode", overlay_menu_x))
        for label, x_pos in menu_specs:
            static_annotations.append(
                dict(
                    text=label,
                    x=x_pos,
                    xref="paper",
                    y=1.115,
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                )
            )

        trace_state_map = {}
        annotation_map = {}
        underwater_series_map = {}
        drawdown_series_map = {}
        recovery_series_map = {}
        underwater_percentile_map = {}
        drawdown_percentile_map = {}
        recovery_percentile_map = {}
        drawdown_trace_indices = []

        for window in window_options:
            rolling_peak = plot_data["Close"].rolling(window=window, min_periods=1).max()
            underwater_series = (
                plot_data["Close"]
                .div(rolling_peak)
                .sub(1.0)
                .dropna()
            )
            drawdown_series = (
                metrics_by_window.get(window, {})
                .get("max_drawdown", pd.Series(dtype=float))
                .dropna()
            )
            recovery_series = _rolling_recovery_time(plot_data["Close"], window)

            shared_index = underwater_series.index.intersection(drawdown_series.index)
            visible_underwater = underwater_series.reindex(shared_index).dropna()
            visible_drawdown = drawdown_series.reindex(shared_index).dropna()
            visible_recovery = recovery_series

            underwater_series_map[window] = visible_underwater
            drawdown_series_map[window] = visible_drawdown
            recovery_series_map[window] = visible_recovery

            underwater_p05 = underwater_series.quantile(0.05) if not underwater_series.empty else np.nan
            drawdown_p05 = drawdown_series.quantile(0.05) if not drawdown_series.empty else np.nan
            recovery_p95 = recovery_series.quantile(0.95) if not recovery_series.empty else np.nan

            underwater_percentile_map[window] = underwater_p05
            drawdown_percentile_map[window] = drawdown_p05
            recovery_percentile_map[window] = recovery_p95

            underwater_current = underwater_series.iloc[-1] if not underwater_series.empty else np.nan
            drawdown_current = drawdown_series.iloc[-1] if not drawdown_series.empty else np.nan
            recovery_current = recovery_series.iloc[-1] if not recovery_series.empty else np.nan

            underwater_rank = _percentile_rank(underwater_series, underwater_current)
            drawdown_rank = _percentile_rank(drawdown_series, drawdown_current)
            recovery_rank = _percentile_rank(recovery_series, recovery_current)

            drawdown_x_ref = visible_underwater.index if not visible_underwater.empty else visible_drawdown.index
            recovery_x_ref = visible_recovery.index
            drawdown_last_x = drawdown_x_ref[-1] if len(drawdown_x_ref) > 0 else None
            recovery_last_x = recovery_x_ref[-1] if len(recovery_x_ref) > 0 else None

            default_states = [True, True, True, True, True, True]
            trace_positions = []

            fig.add_trace(
                go.Scatter(
                    x=visible_underwater.index,
                    y=visible_underwater,
                    mode="lines",
                    name=f"{window}-Day Underwater",
                    line=dict(color="#0ea5e9", width=2),
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[0]))

            fig.add_trace(
                go.Scatter(
                    x=visible_drawdown.index,
                    y=visible_drawdown,
                    mode="lines",
                    name=f"{window}-Day Textbook Max Drawdown",
                    line=dict(color="#f97316", width=2, dash="dot"),
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[1]))

            fig.add_trace(
                go.Scatter(
                    x=drawdown_x_ref,
                    y=[underwater_p05] * len(drawdown_x_ref),
                    mode="lines",
                    name="Underwater 5th Percentile",
                    line=dict(color="rgba(14, 165, 233, 0.55)", width=1.5, dash="dash"),
                    hovertemplate="Underwater 5th percentile: %{y:.1%}<extra></extra>",
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[2]))

            fig.add_trace(
                go.Scatter(
                    x=drawdown_x_ref,
                    y=[drawdown_p05] * len(drawdown_x_ref),
                    mode="lines",
                    name="Textbook 5th Percentile",
                    line=dict(color="rgba(249, 115, 22, 0.55)", width=1.5, dash="dash"),
                    hovertemplate="Textbook 5th percentile: %{y:.1%}<extra></extra>",
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[3]))

            fig.add_trace(
                go.Scatter(
                    x=visible_recovery.index,
                    y=visible_recovery,
                    mode="lines",
                    name=f"{window}-Day Recovery Time",
                    line=dict(color="#22c55e", width=2),
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[4]))

            fig.add_trace(
                go.Scatter(
                    x=recovery_x_ref,
                    y=[recovery_p95] * len(recovery_x_ref),
                    mode="lines",
                    name="Recovery 95th Percentile",
                    line=dict(color="rgba(34, 197, 94, 0.60)", width=1.5, dash="dash"),
                    hovertemplate="Recovery 95th percentile: %{y:.0f} sessions<extra></extra>",
                    visible=window == default_window,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            drawdown_trace_indices.append(len(fig.data) - 1)
            trace_positions.append((len(drawdown_trace_indices) - 1, default_states[5]))

            line_annotations = [
                _line_annotation(
                    x=visible_underwater.index[-1] if not visible_underwater.empty else None,
                    y=visible_underwater.iloc[-1] if not visible_underwater.empty else np.nan,
                    text="Underwater",
                    color="#0ea5e9",
                    xref="x2",
                    yref="y2",
                    yshift=12,
                ),
                _line_annotation(
                    x=visible_drawdown.index[-1] if not visible_drawdown.empty else None,
                    y=visible_drawdown.iloc[-1] if not visible_drawdown.empty else np.nan,
                    text="Textbook Max DD",
                    color="#f97316",
                    xref="x2",
                    yref="y2",
                    yshift=-12,
                ),
                _line_annotation(
                    x=drawdown_last_x,
                    y=underwater_p05,
                    text="Underwater 5th pct",
                    color="rgba(14, 165, 233, 0.90)",
                    xref="x2",
                    yref="y2",
                    yshift=-12,
                ),
                _line_annotation(
                    x=drawdown_last_x,
                    y=drawdown_p05,
                    text="Textbook 5th pct",
                    color="rgba(249, 115, 22, 0.90)",
                    xref="x2",
                    yref="y2",
                    yshift=12,
                ),
                _line_annotation(
                    x=visible_recovery.index[-1] if not visible_recovery.empty else None,
                    y=visible_recovery.iloc[-1] if not visible_recovery.empty else np.nan,
                    text="Recovery Time",
                    color="#22c55e",
                    xref="x3",
                    yref="y3",
                    yshift=12,
                ),
                _line_annotation(
                    x=recovery_last_x,
                    y=recovery_p95,
                    text="Recovery 95th pct",
                    color="rgba(34, 197, 94, 0.90)",
                    xref="x3",
                    yref="y3",
                    yshift=-12,
                ),
            ]

            trace_state_map[window] = trace_positions
            annotation_map[window] = static_annotations + [annotation for annotation in line_annotations if annotation is not None] + [
                dict(
                    x=0.995,
                    y=0.95,
                    xref="paper",
                    yref="y2 domain",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                    bgcolor="rgba(15, 23, 42, 0.72)",
                    bordercolor="rgba(148, 163, 184, 0.35)",
                    borderwidth=1,
                    text=(
                        f"Latest underwater: {underwater_current:.1%} ({_format_percentile(underwater_rank)})"
                        "<br>"
                        f"Latest textbook: {drawdown_current:.1%} ({_format_percentile(drawdown_rank)})"
                    ),
                ),
                dict(
                    x=0.995,
                    y=0.95,
                    xref="paper",
                    yref="y3 domain",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    align="right",
                    font=dict(size=11),
                    bgcolor="rgba(15, 23, 42, 0.72)",
                    bordercolor="rgba(148, 163, 184, 0.35)",
                    borderwidth=1,
                    text=(
                        f"Latest completed recovery: {recovery_current:.0f} sessions ({_format_percentile(recovery_rank)})"
                        if not pd.isna(recovery_current)
                        else "Latest completed recovery: n/a"
                    ),
                ),
            ]

        fig.update_xaxes(
            title_text="Date",
            type="date",
            rangeslider=dict(visible=False),
            showgrid=True,
            zeroline=False,
            rangebreaks=[dict(bounds=["sat", "mon"])],
            showticklabels=False,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Price", autorange=True, fixedrange=False, row=1, col=1)
        fig.update_xaxes(
            title_text="Date",
            type="date",
            showgrid=True,
            zeroline=False,
            rangebreaks=[dict(bounds=["sat", "mon"])],
            matches="x",
            showticklabels=False,
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(
            title_text="Date",
            type="date",
            rangebreaks=[dict(bounds=["sat", "mon"])],
            matches="x",
            showticklabels=True,
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="Sessions", rangemode="tozero", row=3, col=1)
        fig.add_hline(
            y=0,
            row=2,
            col=1,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.45)",
            line_width=1,
        )

        window_buttons = []
        for window in window_options:
            visibility = [False] * len(drawdown_trace_indices)
            for trace_position, state in trace_state_map[window]:
                visibility[trace_position] = state
            window_buttons.append(
                dict(
                    label=f"{window}-Day",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": self._header_title(_window_title(window)),
                            "annotations": annotation_map[window],
                        },
                        drawdown_trace_indices,
                    ],
                )
            )

        global_start_candidates = []
        global_end_candidates = []
        if candlestick_bundle["period_start"] is not None and candlestick_bundle["period_end"] is not None:
            global_start_candidates.append(candlestick_bundle["period_start"])
            global_end_candidates.append(candlestick_bundle["period_end"])
        available_indexes = [series.index for series in drawdown_series_map.values() if not series.empty]
        recovery_indexes = [series.index for series in recovery_series_map.values() if not series.empty]
        if available_indexes:
            global_start_candidates.append(min(index[0] for index in available_indexes))
            global_end_candidates.append(max(index[-1] for index in available_indexes))
        if recovery_indexes:
            global_start_candidates.append(min(index[0] for index in recovery_indexes))
            global_end_candidates.append(max(index[-1] for index in recovery_indexes))

        updatemenus = []
        if show_window_menu:
            updatemenus.append(
                self._dropdown_menu(
                    buttons=window_buttons,
                    x=0.0,
                    active=window_options.index(default_window),
                )
            )

        if global_start_candidates and global_end_candidates:
            global_start = min(global_start_candidates)
            global_end = max(global_end_candidates)

            def _visible_drawdown_axis_series(start=None, end=None):
                axis_series = []
                for window in window_options:
                    underwater_slice = _slice_visible(underwater_series_map.get(window), start=start, end=end)
                    drawdown_slice = _slice_visible(drawdown_series_map.get(window), start=start, end=end)
                    if not underwater_slice.empty:
                        axis_series.append(underwater_slice)
                        underwater_percentile = _constant_axis_series(underwater_percentile_map.get(window))
                        if underwater_percentile is not None:
                            axis_series.append(underwater_percentile)
                    if not drawdown_slice.empty:
                        axis_series.append(drawdown_slice)
                        drawdown_percentile = _constant_axis_series(drawdown_percentile_map.get(window))
                        if drawdown_percentile is not None:
                            axis_series.append(drawdown_percentile)
                return axis_series

            def _visible_recovery_axis_series(start=None, end=None):
                axis_series = []
                for window in window_options:
                    recovery_slice = _slice_visible(recovery_series_map.get(window), start=start, end=end)
                    if recovery_slice.empty:
                        continue
                    axis_series.append(recovery_slice)
                    recovery_percentile = _constant_axis_series(recovery_percentile_map.get(window))
                    if recovery_percentile is not None:
                        axis_series.append(recovery_percentile)
                return axis_series

            def _combined_range(offset=None):
                visible_start = global_start if offset is None else max(global_start, global_end - offset)
                new_range = CandleStickPlotter.build_time_range(global_start, global_end, offset)
                layout_updates = {
                    "xaxis.range": new_range,
                    "xaxis2.range": new_range,
                    "xaxis3.range": new_range,
                }
                price_range = CandleStickPlotter.build_candlestick_y_range(
                    candlestick_bundle,
                    start=visible_start,
                    end=global_end,
                )
                if price_range is not None:
                    layout_updates["yaxis.range"] = price_range

                drawdown_range = CandleStickPlotter.build_numeric_axis_range(
                    _visible_drawdown_axis_series(start=visible_start, end=global_end),
                    include_zero=True,
                    padding_ratio=0.08,
                )
                if drawdown_range is not None:
                    layout_updates["yaxis2.range"] = drawdown_range

                recovery_range = CandleStickPlotter.build_numeric_axis_range(
                    _visible_recovery_axis_series(start=visible_start, end=global_end),
                    include_zero=True,
                    padding_ratio=0.08,
                )
                if recovery_range is not None:
                    layout_updates["yaxis3.range"] = recovery_range

                return layout_updates

            updatemenus.append(
                self._dropdown_menu(
                    buttons=[
                        dict(label=label, method="relayout", args=[_combined_range(offset)])
                        for label, offset in timeframe_options
                    ],
                    x=timeframe_menu_x,
                    active=default_timeframe_index,
                )
            )

            initial_range = CandleStickPlotter.build_time_range(
                global_start,
                global_end,
                timeframe_options[default_timeframe_index][1],
            )
            fig.update_xaxes(range=initial_range, row=1, col=1)
            fig.update_xaxes(range=initial_range, row=2, col=1)
            fig.update_xaxes(range=initial_range, row=3, col=1)
            initial_layout = _combined_range(timeframe_options[default_timeframe_index][1])
            if "yaxis.range" in initial_layout:
                fig.update_yaxes(range=initial_layout["yaxis.range"], row=1, col=1)
            if "yaxis2.range" in initial_layout:
                fig.update_yaxes(range=initial_layout["yaxis2.range"], row=2, col=1)
            if "yaxis3.range" in initial_layout:
                fig.update_yaxes(range=initial_layout["yaxis3.range"], row=3, col=1)

        if overlay_trace_indices:
            updatemenus.append(
                self._dropdown_menu(
                    buttons=[
                        dict(
                            label="Bollinger",
                            method="update",
                            args=[
                                {
                                    "visible": ([True] * len(candlestick_overlay_groups["bollinger"])) +
                                    ([False] * len(candlestick_overlay_groups["mapped_mdd"]))
                                },
                                {},
                                overlay_trace_indices,
                            ],
                        ),
                        dict(
                            label="Mapped MDD",
                            method="update",
                            args=[
                                {
                                    "visible": ([False] * len(candlestick_overlay_groups["bollinger"])) +
                                    ([True] * len(candlestick_overlay_groups["mapped_mdd"]))
                                },
                                {},
                                overlay_trace_indices,
                            ],
                        ),
                    ],
                    x=overlay_menu_x,
                    active=0 if default_overlay == "bollinger" else 1,
                )
            )

        fig.update_layout(
            updatemenus=updatemenus,
            title=self._header_title(_window_title(default_window)),
            height=1650,
            margin=dict(t=220, r=max(candlestick_bundle["right_margin"], 240)),
            template=template,
            showlegend=False,
            hovermode="x unified",
            annotations=annotation_map[default_window],
        )
        return fig

    def plot_distribution_shape_zscores(
        self,
        metrics_by_window,
        window_options=None,
        default_window=None,
        ticker_label="Asset",
        template="plotly_dark",
        include_return_panel=True,
    ):
        """
        Plot skew/kurtosis/gini z-scores, optionally with a leading daily-return panel.
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

        if include_return_panel:
            subplot_titles = (
                "Daily Returns with Rolling Quantile Bands",
                "Rolling Skew Z-Score",
                "Rolling Excess Kurtosis Z-Score",
                "Rolling Gini Coefficient Z-Score",
            )
            row_count = 4
            row_map = {"returns": 1, "skew": 2, "kurtosis": 3, "gini": 4}
            figure_height = 1450
            figure_title = f"{ticker_label} Daily Returns & Distribution Z-Scores ({default_window}-Day Window)"
        else:
            subplot_titles = (
                "Rolling Skew Z-Score",
                "Rolling Excess Kurtosis Z-Score",
                "Rolling Gini Coefficient Z-Score",
            )
            row_count = 3
            row_map = {"skew": 1, "kurtosis": 2, "gini": 3}
            figure_height = 1150
            figure_title = f"{ticker_label} Distribution Z-Scores ({default_window}-Day Window)"

        fig = make_subplots(
            rows=row_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
        )

        metric_colors = {
            "return_median": "#f8fafc",
            "outer_band": "rgba(148, 163, 184, 0.18)",
            "inner_band": "rgba(148, 163, 184, 0.32)",
            "skew": "#ff7f0e",
            "kurtosis": "#2ca02c",
            "gini": "#d62728",
        }
        traces_per_window = None
        zscore_index_candidates = []

        for window in window_options:
            visible = window == default_window
            metric_set = metrics_by_window.get(window, {})
            daily_return_series = metric_set.get("daily_returns", pd.Series(dtype=float)).dropna()
            return_q10 = metric_set.get("return_q10", pd.Series(dtype=float)).dropna()
            return_q25 = metric_set.get("return_q25", pd.Series(dtype=float)).dropna()
            return_median = metric_set.get("return_median", pd.Series(dtype=float)).dropna()
            return_q75 = metric_set.get("return_q75", pd.Series(dtype=float)).dropna()
            return_q90 = metric_set.get("return_q90", pd.Series(dtype=float)).dropna()
            skew_series = metric_set.get("skew_z", pd.Series(dtype=float)).dropna()
            kurtosis_series = metric_set.get("kurtosis_z", pd.Series(dtype=float)).dropna()
            gini_series = metric_set.get("gini_z", pd.Series(dtype=float)).dropna()

            trace_start = len(fig.data)

            if include_return_panel:
                return_colors = np.where(
                    daily_return_series >= 0,
                    "rgba(34, 197, 94, 0.45)",
                    "rgba(239, 68, 68, 0.45)",
                ).tolist()

                fig.add_trace(
                    go.Bar(
                        x=daily_return_series.index,
                        y=daily_return_series,
                        name=f"{window}-Day 1D Returns",
                        marker_color=return_colors,
                        visible=visible,
                        opacity=0.75,
                    ),
                    row=row_map["returns"],
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=return_q90.index,
                        y=return_q90,
                        mode="lines",
                        name=f"{window}-Day 90th Percentile",
                        line=dict(color="rgba(148, 163, 184, 0.0)", width=1),
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=row_map["returns"],
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=return_q10.index,
                        y=return_q10,
                        mode="lines",
                        name=f"{window}-Day 10th Percentile",
                        line=dict(color="rgba(148, 163, 184, 0.0)", width=1),
                        fill="tonexty",
                        fillcolor=metric_colors["outer_band"],
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=row_map["returns"],
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=return_q75.index,
                        y=return_q75,
                        mode="lines",
                        name=f"{window}-Day 75th Percentile",
                        line=dict(color="rgba(148, 163, 184, 0.0)", width=1),
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=row_map["returns"],
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=return_q25.index,
                        y=return_q25,
                        mode="lines",
                        name=f"{window}-Day 25th Percentile",
                        line=dict(color="rgba(148, 163, 184, 0.0)", width=1),
                        fill="tonexty",
                        fillcolor=metric_colors["inner_band"],
                        visible=visible,
                        hoverinfo="skip",
                    ),
                    row=row_map["returns"],
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=return_median.index,
                        y=return_median,
                        mode="lines",
                        name=f"{window}-Day Rolling Median",
                        line=dict(color=metric_colors["return_median"], width=1.6, dash="dash"),
                        visible=visible,
                    ),
                    row=row_map["returns"],
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=skew_series.index,
                    y=skew_series,
                    mode="lines",
                    name=f"{window}-Day Skew Z-Score",
                    line=dict(color=metric_colors["skew"]),
                    visible=visible,
                ),
                row=row_map["skew"],
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=kurtosis_series.index,
                    y=kurtosis_series,
                    mode="lines",
                    name=f"{window}-Day Excess Kurtosis Z-Score",
                    line=dict(color=metric_colors["kurtosis"]),
                    visible=visible,
                ),
                row=row_map["kurtosis"],
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=gini_series.index,
                    y=gini_series,
                    mode="lines",
                    name=f"{window}-Day Gini Coefficient Z-Score",
                    line=dict(color=metric_colors["gini"]),
                    visible=visible,
                ),
                row=row_map["gini"],
                col=1,
            )

            added_traces = len(fig.data) - trace_start
            if traces_per_window is None:
                traces_per_window = added_traces

            for series in (skew_series, kurtosis_series, gini_series):
                if not series.empty:
                    zscore_index_candidates.append(series.index)

        if traces_per_window is None or traces_per_window <= 0:
            traces_per_window = 9 if include_return_panel else 3

        x_ref = max(zscore_index_candidates, key=len, default=pd.Index([]))
        if len(x_ref) > 0:
            for row in (row_map["skew"], row_map["gini"]):
                add_mean_reference_line(fig, row, x_ref)
                add_sigma_reference_lines(
                    fig,
                    row,
                    x_ref,
                    levels=(1, 1.5, 2, 3) if row == row_map["skew"] else (1, 2, 3),
                )
            add_mean_reference_line(fig, row_map["kurtosis"], x_ref)
            for value in (-1.5, -1, -0.5, 1, 2, 3):
                fig.add_hline(
                    y=value,
                    row=row_map["kurtosis"],
                    col=1,
                    line_dash="dot",
                    line_color="rgba(220, 220, 220, 0.55)",
                    line_width=1,
                )

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
                        {"title": self._header_title(f"{ticker_label} Rolling Distribution Z-Scores ({window}-Day Window)")},
                    ],
                )
            )

        if include_return_panel:
            fig.update_yaxes(title_text="1D Return", tickformat=".1%", row=row_map["returns"], col=1)
            fig.add_hline(
                y=0,
                row=row_map["returns"],
                col=1,
                line_dash="dash",
                line_color="rgba(248, 250, 252, 0.45)",
                line_width=1,
            )
        fig.update_yaxes(title_text="Skew Z", row=row_map["skew"], col=1)
        fig.update_yaxes(title_text="Excess Kurtosis Z", row=row_map["kurtosis"], col=1)
        fig.update_yaxes(title_text="Gini Z", row=row_map["gini"], col=1)
        add_horizontal_zone(
            fig,
            row=row_map["skew"],
            col=1,
            y0=-3,
            y1=-1.5,
            fillcolor="rgba(180, 0, 0, 0.30)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=row_map["skew"],
            col=1,
            y0=-1.5,
            y1=1.5,
            fillcolor="rgba(211, 211, 211, 0.18)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=row_map["skew"],
            col=1,
            y0=-1,
            y1=1,
            fillcolor="rgba(169, 169, 169, 0.24)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=row_map["skew"],
            col=1,
            y0=1.5,
            y1=3,
            fillcolor="rgba(180, 0, 0, 0.30)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_zone_annotation(
            fig,
            row=row_map["skew"],
            col=1,
            y0=-1.5,
            y1=-1,
            text="More Likely to Have Larger Upside Moves",
            font_color="rgba(235, 235, 235, 0.92)",
        )
        add_zone_annotation(
            fig,
            row=row_map["skew"],
            col=1,
            y0=1,
            y1=1.5,
            text="More Likely to Have Larger Downside Moves",
            font_color="rgba(235, 235, 235, 0.92)",
        )
        add_horizontal_zone(
            fig,
            row=row_map["kurtosis"],
            col=1,
            y0=-1.5,
            y1=-1,
            fillcolor="rgba(180, 0, 0, 0.24)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )
        add_horizontal_zone(
            fig,
            row=row_map["kurtosis"],
            col=1,
            y0=-1,
            y1=-0.5,
            fillcolor="rgba(56, 189, 248, 0.16)",
            opacity=1.0,
            line_color="rgba(0, 0, 0, 0)",
            line_width=0,
        )

        fig.update_layout(
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                )
            ],
            title=self._header_title(figure_title),
            height=figure_height,
            margin=self._header_margin(),
            template=template,
            showlegend=False,
        )
        return fig

    def plot_value_at_risk_profile(
        self,
        metrics_by_window,
        window_options=None,
        confidence_levels=None,
        default_window=None,
        ticker_label="Asset",
        template="plotly_dark",
    ):
        """
        Plot daily returns, rolling historical VaR / CVaR (Expected Shortfall), and breach diagnostics.
        """
        if not isinstance(metrics_by_window, Mapping) or not metrics_by_window:
            raise ValueError("metrics_by_window must be a non-empty mapping of window -> confidence -> metric map.")

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

        if confidence_levels is None:
            confidence_levels = sorted(metrics_by_window.get(default_window, {}).keys())
        else:
            try:
                confidence_levels = sorted(
                    float(level) / 100.0 if float(level) > 1 else float(level)
                    for level in confidence_levels
                )
            except Exception as exc:
                raise ValueError("confidence_levels must be iterable numeric confidence values.") from exc

        if not confidence_levels:
            raise ValueError("No valid confidence levels available for plotting.")

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.42, 0.31, 0.27],
            subplot_titles=(
                "Close-to-Close Returns with Historical Downside VaR Thresholds",
                "Rolling Close-to-Close Downside Loss Forecasts: VaR and CVaR",
                "Rolling Close-to-Close Downside Breach Rate vs Expected",
            ),
        )

        palette = ["#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899"]
        confidence_styles = {}
        for idx, confidence in enumerate(confidence_levels):
            base_color = palette[idx % len(palette)]
            confidence_styles[confidence] = {
                "var": base_color,
                "es": px.colors.qualitative.Dark24[idx % len(px.colors.qualitative.Dark24)],
                "breach": base_color,
            }

        traces_per_window = None
        index_candidates = []

        for window in window_options:
            visible = window == default_window
            trace_start = len(fig.data)
            window_metric_map = metrics_by_window.get(window, {})

            default_metric_set = window_metric_map.get(
                confidence_levels[0],
                next(iter(window_metric_map.values()), {}),
            )
            daily_return_series = default_metric_set.get("daily_returns", pd.Series(dtype=float)).dropna()
            return_colors = np.where(
                daily_return_series >= 0,
                "rgba(34, 197, 94, 0.45)",
                "rgba(239, 68, 68, 0.45)",
            ).tolist()

            fig.add_trace(
                go.Bar(
                    x=daily_return_series.index,
                    y=daily_return_series,
                    name="1D Returns",
                    marker_color=return_colors,
                    visible=visible,
                    opacity=0.75,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            if not daily_return_series.empty:
                index_candidates.append(daily_return_series.index)

            for confidence in confidence_levels:
                metric_set = window_metric_map.get(confidence, {})
                label = f"{confidence:.0%}"
                style = confidence_styles[confidence]

                var_threshold = metric_set.get("var_threshold", pd.Series(dtype=float)).dropna()
                var_loss = metric_set.get("var", pd.Series(dtype=float)).dropna()
                expected_shortfall = metric_set.get("expected_shortfall", pd.Series(dtype=float)).dropna()
                breaches = metric_set.get("breaches", pd.Series(dtype=float)).dropna()
                rolling_breach_rate = metric_set.get("rolling_breach_rate", pd.Series(dtype=float)).dropna()
                expected_breach_rate = metric_set.get("expected_breach_rate", pd.Series(dtype=float)).dropna()

                breach_index = breaches.index[breaches.astype(bool)] if not breaches.empty else pd.Index([])
                breach_returns = daily_return_series.reindex(breach_index).dropna()

                fig.add_trace(
                    go.Scatter(
                        x=var_threshold.index,
                        y=var_threshold,
                        mode="lines",
                        name=f"{label} Close-to-Close VaR Threshold",
                        line=dict(color=style["var"], width=2),
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=breach_returns.index,
                        y=breach_returns,
                        mode="markers",
                        name=f"{label} Close-to-Close Breaches",
                        marker=dict(color=style["breach"], size=7, symbol="x"),
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=var_loss.index,
                        y=var_loss,
                        mode="lines",
                        name=f"{label} Close-to-Close VaR",
                        line=dict(color=style["var"], width=2),
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=expected_shortfall.index,
                        y=expected_shortfall,
                        mode="lines",
                        name=f"{label} Close-to-Close CVaR",
                        line=dict(color=style["es"], width=2, dash="dash"),
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=rolling_breach_rate.index,
                        y=rolling_breach_rate,
                        mode="lines",
                        name=f"{label} Close-to-Close Breach Rate",
                        line=dict(color=style["var"], width=2),
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=expected_breach_rate.index,
                        y=expected_breach_rate,
                        mode="lines",
                        name=f"{label} Expected Close-to-Close Breach Rate",
                        line=dict(color=style["es"], width=1.5, dash="dot"),
                        visible=visible,
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

                for series in (
                    var_threshold,
                    var_loss,
                    expected_shortfall,
                    rolling_breach_rate,
                    expected_breach_rate,
                ):
                    if not series.empty:
                        index_candidates.append(series.index)

            added_traces = len(fig.data) - trace_start
            if traces_per_window is None:
                traces_per_window = added_traces

        if traces_per_window is None or traces_per_window <= 0:
            raise ValueError("Unable to build VaR traces from the supplied metric payload.")

        total_traces = len(fig.data)
        buttons = []
        for idx, window in enumerate(window_options):
            visibility = build_visibility_mask(
                total_traces=total_traces,
                active_window_index=idx,
                traces_per_window=traces_per_window,
                constant_trace_indices=[],
            )
            buttons.append(
                dict(
                    label=str(window),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": self._header_title(f"{ticker_label} Historical Close-to-Close Downside VaR / CVaR Profile ({window}-Day Window)")},
                    ],
                )
            )

        fig.update_yaxes(title_text="Close-to-Close Return", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="Downside Loss Forecast", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title_text="Downside Breach Rate", tickformat=".1%", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.add_hline(
            y=0,
            row=1,
            col=1,
            line_dash="dash",
            line_color="rgba(248, 250, 252, 0.40)",
            line_width=1,
        )

        updatemenus = [self._dropdown_menu(buttons=buttons, x=0.0)]
        static_annotations = [
            dict(
                text="VaR Lookback Window",
                x=0.0,
                xref="paper",
                y=1.115,
                yref="paper",
                showarrow=False,
                xanchor="left",
            )
        ]
        if index_candidates:
            global_start = min(index[0] for index in index_candidates if len(index) > 0)
            global_end = max(index[-1] for index in index_candidates if len(index) > 0)
            default_start = max(global_start, global_end - pd.DateOffset(years=3))
            fig.update_xaxes(range=[default_start, global_end])
            updatemenus.append(
                self._dropdown_menu(
                    buttons=build_time_range_buttons(global_start, global_end),
                    x=0.18,
                )
            )
            static_annotations.append(
                dict(
                    text="View timeframe",
                    x=0.18,
                    xref="paper",
                    y=1.115,
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                )
            )

        fig.update_layout(
            updatemenus=updatemenus,
            title=self._header_title(f"{ticker_label} Historical Close-to-Close Downside VaR / CVaR Profile ({default_window}-Day Window)"),
            annotations=list(fig.layout.annotations) + static_annotations,
            height=1200,
            margin=self._header_margin(),
            template=template,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    def plot_session_probability_cone(
        self,
        cone_context,
        ticker_label="Asset",
        template="plotly_dark",
    ):
        """
        Plot an open-anchored end-of-day probability cone for the latest session using
        trailing open-to-close returns, with VaR / CVaR downside markers.
        """
        required_keys = {
            "session_date",
            "window",
            "effective_window",
            "anchor_price",
            "latest_price",
            "sample_returns",
            "interval_confidence_levels",
            "var_confidence_levels",
            "intervals",
            "var_levels",
            "median_return",
            "median_price",
        }
        if not isinstance(cone_context, Mapping):
            raise TypeError("cone_context must be a mapping.")
        missing = [key for key in required_keys if key not in cone_context]
        if missing:
            raise ValueError(f"cone_context missing required keys: {missing}")

        interval_levels = sorted(cone_context["interval_confidence_levels"], reverse=True)
        var_levels = sorted(cone_context["var_confidence_levels"])
        interval_map = cone_context["intervals"]
        var_level_map = cone_context["var_levels"]
        sample_returns = pd.Series(cone_context["sample_returns"]).dropna()

        if sample_returns.empty:
            raise ValueError("cone_context sample_returns must contain at least one return observation.")

        anchor_price = float(cone_context["anchor_price"])
        latest_price = float(cone_context["latest_price"])
        median_price = float(cone_context["median_price"])
        session_date = pd.Timestamp(cone_context["session_date"])
        effective_window = int(cone_context["effective_window"])
        requested_window = int(cone_context["window"])

        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.12,
            row_heights=[0.58, 0.42],
            subplot_titles=(
                "Open-Anchored Projected Close Cone (Open-to-Close)",
                "Trailing Open-to-Close Return Distribution",
            ),
        )

        band_colors = [
            "rgba(59, 130, 246, 0.18)",
            "rgba(34, 197, 94, 0.22)",
            "rgba(245, 158, 11, 0.26)",
            "rgba(239, 68, 68, 0.30)",
        ]
        band_line_colors = [
            "rgba(59, 130, 246, 0.75)",
            "rgba(34, 197, 94, 0.82)",
            "rgba(245, 158, 11, 0.86)",
            "rgba(239, 68, 68, 0.88)",
        ]

        close_end_prices = [anchor_price, median_price, latest_price]
        for confidence in interval_levels:
            price_band = interval_map.get(confidence, {})
            close_end_prices.extend(
                [
                    float(price_band.get("lower_price", np.nan)),
                    float(price_band.get("upper_price", np.nan)),
                ]
            )
        for confidence in var_levels:
            tail_band = var_level_map.get(confidence, {})
            close_end_prices.extend(
                [
                    float(tail_band.get("var_price", np.nan)),
                    float(tail_band.get("expected_shortfall_price", np.nan)),
                ]
            )

        valid_prices = [price for price in close_end_prices if np.isfinite(price)]
        if not valid_prices:
            raise ValueError("cone_context does not contain any finite price levels to plot.")
        price_min = min(valid_prices)
        price_max = max(valid_prices)
        price_padding = max((price_max - price_min) * 0.10, anchor_price * 0.01)

        for idx, confidence in enumerate(interval_levels):
            price_band = interval_map[confidence]
            fill_color = band_colors[idx % len(band_colors)]
            line_color = band_line_colors[idx % len(band_line_colors)]
            fig.add_trace(
                go.Scatter(
                    x=[0.0, 1.0, 1.0, 0.0, 0.0],
                    y=[
                        anchor_price,
                        price_band["upper_price"],
                        price_band["lower_price"],
                        anchor_price,
                        anchor_price,
                    ],
                    mode="lines",
                    line=dict(color=line_color, width=1.5),
                    fill="toself",
                    fillcolor=fill_color,
                    name=f"{confidence:.0%} Open-to-Close Central Range",
                    hovertemplate=(
                        f"{confidence:.0%} Open-to-Close Central Range"
                        "<br>Lower Close: %{customdata[0]:,.2f}"
                        "<br>Upper Close: %{customdata[1]:,.2f}"
                        "<extra></extra>"
                    ),
                    customdata=[
                        [price_band["lower_price"], price_band["upper_price"]],
                        [price_band["lower_price"], price_band["upper_price"]],
                        [price_band["lower_price"], price_band["upper_price"]],
                        [price_band["lower_price"], price_band["upper_price"]],
                        [price_band["lower_price"], price_band["upper_price"]],
                    ],
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[anchor_price, median_price],
                mode="lines+markers",
                name="Median Close",
                line=dict(color="#f8fafc", width=2, dash="dash"),
                marker=dict(size=8, color="#f8fafc"),
                hovertemplate="Median Close: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[anchor_price],
                mode="markers+text",
                name="Session Open",
                marker=dict(size=10, color="#22c55e", symbol="diamond"),
                text=[f"Open {anchor_price:,.2f}"],
                textposition="top left",
                hovertemplate="Session Open: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[1.0],
                y=[latest_price],
                mode="markers+text",
                name="Latest Session Price",
                marker=dict(size=9, color="#38bdf8", symbol="circle"),
                text=[f"Last {latest_price:,.2f}"],
                textposition="middle right",
                hovertemplate="Latest Session Price: %{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        var_marker_symbols = ["triangle-down", "square"]
        for idx, confidence in enumerate(var_levels):
            tail_band = var_level_map[confidence]
            base_color = "#ef4444" if idx == 0 else "#b91c1c"
            fig.add_trace(
                go.Scatter(
                    x=[1.0],
                    y=[tail_band["var_price"]],
                    mode="markers+text",
                    name=f"{confidence:.0%} Open-to-Close VaR Floor",
                    marker=dict(size=11, color=base_color, symbol=var_marker_symbols[idx % len(var_marker_symbols)]),
                    text=[f"{confidence:.0%} VaR Floor {tail_band['var_price']:,.2f}"],
                    textposition="middle right",
                    hovertemplate=(
                        f"{confidence:.0%} Open-to-Close VaR Floor"
                        "<br>Close Price: %{y:,.2f}"
                        f"<br>Return Threshold: {tail_band['var_return']:.2%}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[1.0],
                    y=[tail_band["expected_shortfall_price"]],
                    mode="markers+text",
                    name=f"{confidence:.0%} Open-to-Close CVaR Floor",
                    marker=dict(size=10, color=base_color, symbol="x"),
                    text=[f"{confidence:.0%} CVaR Floor {tail_band['expected_shortfall_price']:,.2f}"],
                    textposition="middle right",
                    hovertemplate=(
                        f"{confidence:.0%} Open-to-Close CVaR Floor"
                        "<br>Close Price: %{y:,.2f}"
                        f"<br>Expected Shortfall: {tail_band['expected_shortfall_return']:.2%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.add_vline(
            x=1.0,
            row=1,
            col=1,
            line_dash="dot",
            line_color="rgba(248, 250, 252, 0.40)",
            line_width=1,
        )
        fig.add_hline(
            y=anchor_price,
            row=1,
            col=1,
            line_dash="dot",
            line_color="rgba(34, 197, 94, 0.45)",
            line_width=1,
        )

        histogram_values = sample_returns.mul(100.0)
        fig.add_trace(
            go.Histogram(
                x=histogram_values,
                nbinsx=min(max(effective_window // 8, 20), 60),
                name="200-Session Open-to-Close Returns",
                marker_color="rgba(59, 130, 246, 0.65)",
                opacity=0.85,
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_vline(
            x=cone_context["median_return"] * 100.0,
            row=2,
            col=1,
            line_dash="dash",
            line_color="#f8fafc",
            line_width=2,
        )
        for idx, confidence in enumerate(var_levels):
            tail_band = var_level_map[confidence]
            base_color = "#ef4444" if idx == 0 else "#b91c1c"
            fig.add_vline(
                x=tail_band["var_return"] * 100.0,
                row=2,
                col=1,
                line_dash="dash",
                line_color=base_color,
                line_width=2,
            )
            fig.add_vline(
                x=tail_band["expected_shortfall_return"] * 100.0,
                row=2,
                col=1,
                line_dash="dot",
                line_color=base_color,
                line_width=2,
            )

        fig.update_xaxes(
            row=1,
            col=1,
            range=[-0.10, 1.18],
            tickmode="array",
            tickvals=[0.0, 1.0],
            ticktext=["Today Open", "Projected Close"],
            title_text="Session Path",
        )
        fig.update_yaxes(
            row=1,
            col=1,
            title_text="Price",
            range=[price_min - price_padding, price_max + price_padding],
            tickprefix="$",
        )
        fig.update_xaxes(
            row=2,
            col=1,
            title_text="Open-to-Close Return",
            ticksuffix="%",
        )
        fig.update_yaxes(
            row=2,
            col=1,
            title_text="Count",
        )

        title_date = session_date.strftime("%Y-%m-%d")
        fig.update_layout(
            title=self._header_title(
                f"{ticker_label} Today Open-to-Close Probability Cone From Open "
                f"({requested_window}-Session Model, {title_date})"
            ),
            height=1050,
            margin=self._header_margin(),
            template=template,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            bargap=0.08,
        )

        fig.add_annotation(
            x=0.0,
            y=1.12,
            xref="paper",
            yref="paper",
            text=(
                f"Using the last {effective_window} completed open-to-close sessions. "
                f"Open anchor: {anchor_price:,.2f}."
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=12, color="rgba(226, 232, 240, 0.92)"),
        )
        return fig


    def plot_trade_range_probability_cone(
        self,
        cone_context,
        ticker_label="Asset",
        template="plotly_dark",
    ):
        """
        Plot a close-based two-sided probability cone with long and short tail markers
        for same-day trade planning.
        """
        required_keys = {
            "session_date",
            "window",
            "effective_window",
            "anchor_price",
            "latest_price",
            "sample_returns",
            "interval_confidence_levels",
            "tail_confidence_levels",
            "intervals",
            "long_tail_levels",
            "short_tail_levels",
            "median_return",
            "median_price",
        }
        if not isinstance(cone_context, Mapping):
            raise TypeError("cone_context must be a mapping.")

        raw_contexts_by_window = cone_context.get("cone_contexts_by_window")
        window_options = cone_context.get("windows")
        default_window = self._coerce_positive_int(
            cone_context.get("default_window", cone_context.get("window"))
        )

        if isinstance(raw_contexts_by_window, Mapping) and raw_contexts_by_window:
            contexts_by_window = {}
            for key, context in raw_contexts_by_window.items():
                coerced_window = self._coerce_positive_int(key)
                if coerced_window is None:
                    continue
                contexts_by_window[coerced_window] = context

            if window_options is None:
                window_options = list(contexts_by_window.keys())
            else:
                try:
                    window_options = [int(window) for window in window_options]
                except Exception as exc:
                    raise ValueError("cone_context windows must be iterable integers.") from exc
                window_options = [window for window in window_options if window in contexts_by_window]

            if not window_options:
                raise ValueError("cone_context does not contain any valid probability-cone windows.")
            if default_window not in window_options:
                default_window = self._preferred_numeric_window(window_options) or window_options[0]
        else:
            missing = [key for key in required_keys if key not in cone_context]
            if missing:
                raise ValueError(f"cone_context missing required keys: {missing}")
            default_window = int(cone_context["window"])
            window_options = [default_window]
            contexts_by_window = {default_window: cone_context}

        default_context = contexts_by_window[default_window]
        missing = [key for key in required_keys if key not in default_context]
        if missing:
            raise ValueError(f"cone_context missing required keys: {missing}")

        def _trade_range_labels(context):
            horizon_sessions = max(1, int(context.get("horizon_sessions", 1)))
            if horizon_sessions == 1:
                return {
                    "panel_title": lambda confidence: (
                        f"Today Prior-Close-Anchored {confidence:.0%} Projected Close Range (Close-to-Close)"
                    ),
                    "distribution_title": "Trailing Close-to-Close Return Distribution",
                    "header_title": "Two-Sided Close-to-Close Trade Range Cone From Prior Close",
                    "annotation_basis": "completed close-to-close sessions",
                    "path_ticktext": ["Prior Close", "Projected Close"],
                    "path_axis_title": "Close Path",
                    "return_axis_title": "Close-to-Close Return",
                    "histogram_name": "Close-to-Close Returns",
                    "range_name": "Projected Close Range",
                    "lower_price_label": "Lower Close",
                    "upper_price_label": "Upper Close",
                    "median_name": "Median Close",
                    "anchor_name": "Prior Close",
                    "anchor_text_prefix": "Prior Close",
                    "show_latest_price": True,
                    "latest_name": "Latest Price So Far",
                    "latest_text_prefix": "Last",
                    "tail_basis": "Close-to-Close",
                }

            horizon_label = f"{horizon_sessions}-Session"
            horizon_lower = f"{horizon_sessions}-session"
            return {
                "panel_title": lambda confidence: (
                    f"Reference-Close-Anchored {confidence:.0%} Projected Close Range ({horizon_label} Horizon)"
                ),
                "distribution_title": f"Trailing {horizon_label} Forward Return Distribution",
                "header_title": f"Two-Sided {horizon_label} Trade Range Cone From Reference Close",
                "annotation_basis": f"completed {horizon_lower} close-based forward returns",
                "path_ticktext": ["Reference Close", "Projected Close"],
                "path_axis_title": "Holding-Period Close Path",
                "return_axis_title": f"{horizon_label} Forward Return",
                "histogram_name": f"{horizon_label} Returns",
                "range_name": f"{horizon_label} Projected Exit Range",
                "lower_price_label": "Lower Exit Price",
                "upper_price_label": "Upper Exit Price",
                "median_name": "Median Exit Close",
                "anchor_name": "Reference Close",
                "anchor_text_prefix": "Reference Close",
                "show_latest_price": False,
                "latest_name": "Current Close",
                "latest_text_prefix": "Current",
                "tail_basis": horizon_label,
            }

        default_labels = _trade_range_labels(default_context)

        interval_levels = sorted(default_context["interval_confidence_levels"], reverse=True)
        tail_levels = sorted(default_context["tail_confidence_levels"])
        panel_confidence_levels = sorted(set(interval_levels).intersection(tail_levels))
        if not panel_confidence_levels:
            panel_confidence_levels = sorted(set(interval_levels).union(tail_levels))

        distribution_row = len(panel_confidence_levels) + 1
        distribution_height = 0.32
        panel_height = (1.0 - distribution_height) / len(panel_confidence_levels)
        row_heights = [panel_height] * len(panel_confidence_levels) + [distribution_height]
        subplot_titles = tuple(
            [
                default_labels["panel_title"](confidence)
                for confidence in panel_confidence_levels
            ]
            + [default_labels["distribution_title"]]
        )

        fig = make_subplots(
            rows=distribution_row,
            cols=1,
            vertical_spacing=0.08,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )
        subplot_title_annotations = [copy.deepcopy(annotation) for annotation in fig.layout.annotations]

        interval_fill_colors = [
            "rgba(59, 130, 246, 0.18)",
            "rgba(34, 197, 94, 0.24)",
            "rgba(245, 158, 11, 0.30)",
            "rgba(239, 68, 68, 0.34)",
        ]
        interval_line_colors = [
            "rgba(59, 130, 246, 0.72)",
            "rgba(34, 197, 94, 0.84)",
            "rgba(245, 158, 11, 0.88)",
            "rgba(239, 68, 68, 0.90)",
        ]

        global_price_min = None
        global_price_max = None
        for window in window_options:
            context = contexts_by_window[window]
            missing = [key for key in required_keys if key not in context]
            if missing:
                raise ValueError(f"cone_context for window {window} missing required keys: {missing}")

            window_interval_levels = sorted(context["interval_confidence_levels"], reverse=True)
            window_tail_levels = sorted(context["tail_confidence_levels"])
            window_panel_levels = sorted(set(window_interval_levels).intersection(window_tail_levels))
            if not window_panel_levels:
                window_panel_levels = sorted(set(window_interval_levels).union(window_tail_levels))
            if window_panel_levels != panel_confidence_levels:
                raise ValueError("All cone windows must share the same confidence-level layout.")

            interval_map = context["intervals"]
            long_tail_map = context["long_tail_levels"]
            short_tail_map = context["short_tail_levels"]
            anchor_price = float(context["anchor_price"])
            latest_price = float(context["latest_price"])
            median_price = float(context["median_price"])

            candidate_prices = [anchor_price, latest_price, median_price]
            for confidence in window_interval_levels:
                band = interval_map[confidence]
                candidate_prices.extend([band["lower_price"], band["upper_price"]])
            for confidence in window_tail_levels:
                long_tail = long_tail_map[confidence]
                short_tail = short_tail_map[confidence]
                candidate_prices.extend(
                    [
                        long_tail["var_price"],
                        long_tail["expected_shortfall_price"],
                        short_tail["var_price"],
                        short_tail["expected_shortfall_price"],
                    ]
                )

            valid_prices = [float(price) for price in candidate_prices if np.isfinite(price)]
            if not valid_prices:
                raise ValueError(
                    f"cone_context for window {window} does not contain any finite price levels to plot."
                )
            window_min = min(valid_prices)
            window_max = max(valid_prices)
            global_price_min = window_min if global_price_min is None else min(global_price_min, window_min)
            global_price_max = window_max if global_price_max is None else max(global_price_max, window_max)

        if global_price_min is None or global_price_max is None:
            raise ValueError("cone_context does not contain any finite price levels to plot.")

        default_anchor_price = float(default_context["anchor_price"])
        price_padding = max((global_price_max - global_price_min) * 0.10, default_anchor_price * 0.01)
        price_axis_min = global_price_min - price_padding
        price_axis_max = global_price_max + price_padding

        def _axis_ref(row_idx, axis_name):
            return axis_name if row_idx == 1 else f"{axis_name}{row_idx}"

        def _window_title(context):
            requested_window = int(context["window"])
            title_date = pd.Timestamp(context["session_date"]).strftime("%Y-%m-%d")
            labels = _trade_range_labels(context)
            return self._header_title(
                f"{ticker_label} {labels['header_title']} "
                f"({requested_window}-Session Lookback, {title_date})"
            )

        def _window_annotation(context):
            labels = _trade_range_labels(context)
            return dict(
                x=0.0,
                y=1.12,
                xref="paper",
                yref="paper",
                text=(
                    f"Using the last {int(context['effective_window'])} {labels['annotation_basis']}. "
                    f"Entry anchor: {float(context['anchor_price']):,.2f}. "
                    "Red markers show long-risk floors; purple markers show short-risk ceilings."
                ),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=12, color="rgba(226, 232, 240, 0.92)"),
            )

        def _window_shapes(context):
            shapes = []
            anchor_price = float(context["anchor_price"])
            median_return = float(context["median_return"])
            long_tail_map = context["long_tail_levels"]
            short_tail_map = context["short_tail_levels"]

            for row_idx in range(1, distribution_row):
                xref = _axis_ref(row_idx, "x")
                yref = _axis_ref(row_idx, "y")
                shapes.append(
                    dict(
                        type="line",
                        xref=xref,
                        yref=f"{yref} domain",
                        x0=1.0,
                        x1=1.0,
                        y0=0.0,
                        y1=1.0,
                        line=dict(dash="dot", color="rgba(248, 250, 252, 0.40)", width=1),
                    )
                )
                shapes.append(
                    dict(
                        type="line",
                        xref=xref,
                        yref=yref,
                        x0=-0.10,
                        x1=1.20,
                        y0=anchor_price,
                        y1=anchor_price,
                        line=dict(dash="dot", color="rgba(34, 197, 94, 0.45)", width=1),
                    )
                )

            xref = _axis_ref(distribution_row, "x")
            yref = _axis_ref(distribution_row, "y")
            shapes.append(
                dict(
                    type="line",
                    xref=xref,
                    yref=f"{yref} domain",
                    x0=median_return * 100.0,
                    x1=median_return * 100.0,
                    y0=0.0,
                    y1=1.0,
                    line=dict(dash="dash", color="#f8fafc", width=2),
                )
            )

            for idx, confidence in enumerate(tail_levels):
                long_tail = long_tail_map[confidence]
                short_tail = short_tail_map[confidence]
                long_color = "#ef4444" if idx == 0 else "#991b1b"
                short_color = "#a855f7" if idx == 0 else "#6d28d9"

                for value, color, dash in (
                    (long_tail["var_return"] * 100.0, long_color, "dash"),
                    (long_tail["expected_shortfall_return"] * 100.0, long_color, "dot"),
                    (short_tail["var_return"] * 100.0, short_color, "dash"),
                    (short_tail["expected_shortfall_return"] * 100.0, short_color, "dot"),
                ):
                    shapes.append(
                        dict(
                            type="line",
                            xref=xref,
                            yref=f"{yref} domain",
                            x0=value,
                            x1=value,
                            y0=0.0,
                            y1=1.0,
                            line=dict(dash=dash, color=color, width=2),
                        )
                    )

            return shapes

        window_annotations = {}
        window_shapes = {}
        traces_per_window = None

        for window in window_options:
            visible = window == default_window
            trace_start = len(fig.data)
            context = contexts_by_window[window]
            labels = _trade_range_labels(context)
            interval_map = context["intervals"]
            long_tail_map = context["long_tail_levels"]
            short_tail_map = context["short_tail_levels"]
            sample_returns = pd.Series(context["sample_returns"]).dropna()

            if sample_returns.empty:
                raise ValueError(
                    f"cone_context sample_returns for window {window} must contain at least one return observation."
                )

            anchor_price = float(context["anchor_price"])
            latest_price = float(context["latest_price"])
            median_price = float(context["median_price"])
            effective_window = int(context["effective_window"])

            def add_projected_close_row(target_row, confidence, *, show_shared_legend):
                band = interval_map.get(confidence)
                style_index = panel_confidence_levels.index(confidence)
                fill_color = interval_fill_colors[style_index % len(interval_fill_colors)]
                line_color = interval_line_colors[style_index % len(interval_line_colors)]

                if band is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[0.0, 1.0, 1.0, 0.0, 0.0],
                            y=[anchor_price, band["upper_price"], band["lower_price"], anchor_price, anchor_price],
                            mode="lines",
                            line=dict(color=line_color, width=1.5),
                            fill="toself",
                            fillcolor=fill_color,
                            name=f"{confidence:.0%} {labels['range_name']}",
                            hovertemplate=(
                                f"{confidence:.0%} {labels['range_name']}"
                                f"<br>{labels['lower_price_label']}: %{{customdata[0]:,.2f}}"
                                f"<br>{labels['upper_price_label']}: %{{customdata[1]:,.2f}}"
                                "<br>Lower Return: %{customdata[2]:.4%}"
                                "<br>Upper Return: %{customdata[3]:.4%}"
                                "<extra></extra>"
                            ),
                            customdata=[
                                [
                                    band["lower_price"],
                                    band["upper_price"],
                                    band["lower_return"],
                                    band["upper_return"],
                                ]
                            ] * 5,
                            showlegend=True,
                            legendgroup=f"range-{confidence:.0%}",
                            visible=visible,
                        ),
                        row=target_row,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0, 1.0],
                            y=[band["lower_price"], band["upper_price"]],
                            mode="markers",
                            name=f"{confidence:.0%} Range Edges",
                            marker=dict(color=line_color, size=8, symbol="diamond"),
                            hovertemplate=(
                                f"{confidence:.0%} Range Edge"
                                "<br>Close Price: %{y:,.2f}"
                                "<br>Return Threshold: %{customdata[0]:.4%}"
                                "<extra></extra>"
                            ),
                            customdata=[[band["lower_return"]], [band["upper_return"]]],
                            showlegend=False,
                            legendgroup=f"range-{confidence:.0%}",
                            visible=visible,
                        ),
                        row=target_row,
                        col=1,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=[0.0, 1.0],
                        y=[anchor_price, median_price],
                        mode="lines+markers",
                        name=labels["median_name"],
                        line=dict(color="#f8fafc", width=2, dash="dash"),
                        marker=dict(size=8, color="#f8fafc"),
                        hovertemplate=f"{labels['median_name']}: %{{y:,.2f}}<extra></extra>",
                        showlegend=show_shared_legend,
                        legendgroup="median-close",
                        visible=visible,
                    ),
                    row=target_row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0.0],
                        y=[anchor_price],
                        mode="markers+text",
                        name=labels["anchor_name"],
                        marker=dict(size=10, color="#22c55e", symbol="diamond"),
                        text=[f"{labels['anchor_text_prefix']} {anchor_price:,.2f}"],
                        textposition="top left",
                        hovertemplate=f"{labels['anchor_name']}: %{{y:,.2f}}<extra></extra>",
                        showlegend=show_shared_legend,
                        legendgroup="session-open",
                        visible=visible,
                    ),
                    row=target_row,
                    col=1,
                )
                if labels["show_latest_price"]:
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0],
                            y=[latest_price],
                            mode="markers+text",
                            name=labels["latest_name"],
                            marker=dict(size=9, color="#38bdf8", symbol="circle"),
                            text=[f"{labels['latest_text_prefix']} {latest_price:,.2f}"],
                            textposition="middle right",
                            hovertemplate=f"{labels['latest_name']}: %{{y:,.2f}}<extra></extra>",
                            showlegend=show_shared_legend,
                            legendgroup="latest-session-price",
                            visible=visible,
                        ),
                        row=target_row,
                        col=1,
                    )

            for row_idx, confidence in enumerate(panel_confidence_levels, start=1):
                add_projected_close_row(row_idx, confidence, show_shared_legend=row_idx == 1)

                long_tail = long_tail_map.get(confidence)
                short_tail = short_tail_map.get(confidence)
                style_index = panel_confidence_levels.index(confidence)
                long_color = "#ef4444" if style_index == 0 else "#991b1b"
                short_color = "#a855f7" if style_index == 0 else "#6d28d9"

                if long_tail is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0],
                            y=[long_tail["var_price"]],
                            mode="markers+text",
                            name=f"{confidence:.0%} {labels['tail_basis']} Long VaR Floor",
                            marker=dict(size=11, color=long_color, symbol="triangle-down"),
                            text=[f"{confidence:.0%} Long VaR Floor {long_tail['var_price']:,.2f}"],
                            textposition="middle right",
                            hovertemplate=(
                                f"{confidence:.0%} {labels['tail_basis']} Long VaR Floor"
                                "<br>Close Price: %{y:,.2f}"
                                f"<br>Return Threshold: {long_tail['var_return']:.4%}"
                                "<extra></extra>"
                            ),
                            visible=visible,
                        ),
                        row=row_idx,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0],
                            y=[long_tail["expected_shortfall_price"]],
                            mode="markers+text",
                            name=f"{confidence:.0%} {labels['tail_basis']} Long CVaR Floor",
                            marker=dict(size=10, color=long_color, symbol="x"),
                            text=[f"{confidence:.0%} Long CVaR Floor {long_tail['expected_shortfall_price']:,.2f}"],
                            textposition="middle right",
                            hovertemplate=(
                                f"{confidence:.0%} {labels['tail_basis']} Long CVaR Floor"
                                "<br>Close Price: %{y:,.2f}"
                                f"<br>Expected Shortfall: {long_tail['expected_shortfall_return']:.4%}"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                            visible=visible,
                        ),
                        row=row_idx,
                        col=1,
                    )
                if short_tail is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0],
                            y=[short_tail["var_price"]],
                            mode="markers+text",
                            name=f"{confidence:.0%} {labels['tail_basis']} Short VaR Ceiling",
                            marker=dict(size=11, color=short_color, symbol="triangle-up"),
                            text=[f"{confidence:.0%} Short VaR Ceiling {short_tail['var_price']:,.2f}"],
                            textposition="middle right",
                            hovertemplate=(
                                f"{confidence:.0%} {labels['tail_basis']} Short VaR Ceiling"
                                "<br>Close Price: %{y:,.2f}"
                                f"<br>Return Threshold: {short_tail['var_return']:.4%}"
                                "<extra></extra>"
                            ),
                            visible=visible,
                        ),
                        row=row_idx,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[1.0],
                            y=[short_tail["expected_shortfall_price"]],
                            mode="markers+text",
                            name=f"{confidence:.0%} {labels['tail_basis']} Short CVaR Ceiling",
                            marker=dict(size=10, color=short_color, symbol="x"),
                            text=[f"{confidence:.0%} Short CVaR Ceiling {short_tail['expected_shortfall_price']:,.2f}"],
                            textposition="middle right",
                            hovertemplate=(
                                f"{confidence:.0%} {labels['tail_basis']} Short CVaR Ceiling"
                                "<br>Close Price: %{y:,.2f}"
                                f"<br>Expected Shortfall: {short_tail['expected_shortfall_return']:.4%}"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                            visible=visible,
                        ),
                        row=row_idx,
                        col=1,
                    )

            histogram_values = sample_returns.mul(100.0)
            fig.add_trace(
                go.Histogram(
                    x=histogram_values,
                    nbinsx=min(max(effective_window // 8, 20), 60),
                    name=labels["histogram_name"],
                    marker_color="rgba(59, 130, 246, 0.65)",
                    opacity=0.85,
                    hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
                    showlegend=False,
                    visible=visible,
                ),
                row=distribution_row,
                col=1,
            )

            added_traces = len(fig.data) - trace_start
            if traces_per_window is None:
                traces_per_window = added_traces

            window_shapes[window] = _window_shapes(context)
            window_annotations[window] = subplot_title_annotations + [_window_annotation(context)]

        if traces_per_window is None or traces_per_window <= 0:
            raise ValueError("Unable to build cone traces from the supplied probability payload.")

        for row_idx in range(1, distribution_row):
            fig.update_xaxes(
                row=row_idx,
                col=1,
                range=[-0.10, 1.20],
                tickmode="array",
                tickvals=[0.0, 1.0],
                ticktext=default_labels["path_ticktext"],
                title_text=default_labels["path_axis_title"],
            )
            fig.update_yaxes(
                row=row_idx,
                col=1,
                title_text="Price",
                range=[price_axis_min, price_axis_max],
                tickprefix="$",
            )

        fig.update_xaxes(
            row=distribution_row,
            col=1,
            title_text=default_labels["return_axis_title"],
            ticksuffix="%",
        )
        fig.update_yaxes(row=distribution_row, col=1, title_text="Count")

        updatemenus = []
        if len(window_options) > 1:
            total_traces = len(fig.data)
            buttons = []
            for idx, window in enumerate(window_options):
                visibility = build_visibility_mask(
                    total_traces=total_traces,
                    active_window_index=idx,
                    traces_per_window=traces_per_window,
                    constant_trace_indices=[],
                )
                buttons.append(
                    dict(
                        label=str(window),
                        method="update",
                        args=[
                            {"visible": visibility},
                            {
                                "title": _window_title(contexts_by_window[window]),
                                "shapes": window_shapes[window],
                                "annotations": window_annotations[window],
                            },
                        ],
                    )
                )
            updatemenus = [
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                    active=window_options.index(default_window),
                )
            ]

        fig.update_layout(
            updatemenus=updatemenus,
            title=_window_title(default_context),
            annotations=window_annotations[default_window],
            shapes=window_shapes[default_window],
            height=1450,
            margin=self._header_margin(),
            template=template,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            bargap=0.08,
        )
        return fig

    def plot_trade_range_history_profile(
        self,
        history_context,
        ticker_label="Asset",
        template="plotly_dark",
    ):
        """
        Plot an ex-ante historical long/short trade-range profile using close-to-close
        returns and prior-session information only.
        """
        required_keys = {
            "interval_confidence_levels",
            "tail_confidence_levels",
            "session_returns",
            "session_open",
            "session_close",
        }
        if not isinstance(history_context, Mapping):
            raise TypeError("history_context must be a mapping.")
        missing = [key for key in required_keys if key not in history_context]
        if missing:
            raise ValueError(f"history_context missing required keys: {missing}")

        metrics_by_window = history_context.get("metrics_by_window")
        window_options = history_context.get("windows")
        default_window = history_context.get("default_window", history_context.get("window"))

        if isinstance(metrics_by_window, Mapping) and metrics_by_window:
            if window_options is None:
                window_options = list(metrics_by_window.keys())
            else:
                try:
                    window_options = [int(window) for window in window_options]
                except Exception as exc:
                    raise ValueError("history_context windows must be iterable integers.") from exc
                window_options = [window for window in window_options if window in metrics_by_window]

            if not window_options:
                raise ValueError("history_context does not contain any valid rolling windows.")
            if default_window not in window_options:
                default_window = self._preferred_numeric_window(window_options) or window_options[0]
        else:
            metrics_by_confidence = history_context.get("metrics_by_confidence")
            if not isinstance(metrics_by_confidence, Mapping) or not metrics_by_confidence:
                raise ValueError(
                    "history_context metrics_by_confidence must be a non-empty mapping."
                )
            default_window = int(history_context["window"])
            window_options = [default_window]
            metrics_by_window = {default_window: metrics_by_confidence}

        interval_levels = sorted(history_context["interval_confidence_levels"], reverse=True)
        tail_levels = sorted(history_context["tail_confidence_levels"])
        session_returns = pd.Series(history_context["session_returns"]).dropna()
        session_close = pd.Series(history_context["session_close"]).dropna()
        if session_returns.empty or session_close.empty:
            raise ValueError("history_context does not contain enough session data to plot.")

        horizon_sessions = max(1, int(history_context.get("horizon_sessions", 1)))
        if horizon_sessions == 1:
            history_labels = {
                "return_name": "Close-to-Close Return",
                "returns_panel": "Close-to-Close Returns with Two-Sided Tail Thresholds",
                "forecast_panel": "Rolling Close-to-Close Tail Forecasts: Long Floors and Short Ceilings",
                "breach_panel": "Rolling Close-to-Close Breach Rates vs Expected",
                "header_basis": "Historical Two-Sided Close-to-Close Trade Range Profile",
            }
        else:
            horizon_label = f"{horizon_sessions}-Session Forward"
            history_labels = {
                "return_name": f"{horizon_label} Return",
                "returns_panel": f"{horizon_label} Returns with Two-Sided Tail Thresholds",
                "forecast_panel": f"Rolling {horizon_label} Tail Forecasts: Long Floors and Short Ceilings",
                "breach_panel": f"Rolling {horizon_label} Breach Rates vs Expected",
                "header_basis": f"Historical Two-Sided {horizon_label} Trade Range Profile",
            }

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.33, 0.27, 0.40],
            subplot_titles=(
                history_labels["returns_panel"],
                history_labels["forecast_panel"],
                history_labels["breach_panel"],
            ),
        )

        interval_fill_colors = {
            0.95: "rgba(34, 197, 94, 0.18)",
            0.99: "rgba(59, 130, 246, 0.15)",
        }
        interval_line_colors = {
            0.95: "rgba(34, 197, 94, 0.82)",
            0.99: "rgba(59, 130, 246, 0.82)",
        }
        long_colors = {
            0.95: "#ef4444",
            0.99: "#991b1b",
        }
        short_colors = {
            0.95: "#a855f7",
            0.99: "#6d28d9",
        }

        index_candidates = [session_returns.index]
        traces_per_window = None
        return_colors = np.where(
            session_returns >= 0,
            "rgba(34, 197, 94, 0.45)",
            "rgba(239, 68, 68, 0.45)",
        ).tolist()

        for window in window_options:
            visible = window == default_window
            trace_start = len(fig.data)
            metrics_by_confidence = metrics_by_window.get(window, {})

            fig.add_trace(
                go.Bar(
                    x=session_returns.index,
                    y=session_returns,
                    name=history_labels["return_name"],
                    marker_color=return_colors,
                    opacity=0.75,
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}"
                        f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=1,
            )

            for confidence in tail_levels:
                metric_set = metrics_by_confidence.get(confidence, {})
                lower_var_return = metric_set.get("lower_var_return", pd.Series(dtype=float)).dropna()
                upper_var_return = metric_set.get("upper_var_return", pd.Series(dtype=float)).dropna()
                lower_es_return = metric_set.get(
                    "lower_expected_shortfall_return",
                    pd.Series(dtype=float),
                ).dropna()
                upper_es_return = metric_set.get(
                    "upper_expected_shortfall_return",
                    pd.Series(dtype=float),
                ).dropna()
                lower_breaches = metric_set.get("lower_breaches", pd.Series(dtype=float)).dropna()
                upper_breaches = metric_set.get("upper_breaches", pd.Series(dtype=float)).dropna()

                lower_breach_index = (
                    lower_breaches.index[lower_breaches.astype(bool)]
                    if not lower_breaches.empty
                    else pd.Index([])
                )
                upper_breach_index = (
                    upper_breaches.index[upper_breaches.astype(bool)]
                    if not upper_breaches.empty
                    else pd.Index([])
                )
                lower_breach_returns = session_returns.reindex(lower_breach_index).dropna()
                upper_breach_returns = session_returns.reindex(upper_breach_index).dropna()

                fig.add_trace(
                    go.Scatter(
                        x=lower_var_return.index,
                        y=lower_var_return,
                        mode="lines",
                        name=f"{confidence:.0%} Long VaR Return",
                        line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Long VaR Return"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Return Threshold: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=upper_var_return.index,
                        y=upper_var_return,
                        mode="lines",
                        name=f"{confidence:.0%} Short VaR Return",
                        line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Short VaR Return"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Return Threshold: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=lower_breach_returns.index,
                        y=lower_breach_returns,
                        mode="markers",
                        name=f"{confidence:.0%} Long Breaches",
                        marker=dict(
                            color=long_colors.get(confidence, "#ef4444"),
                            size=7,
                            symbol="x",
                        ),
                        hovertemplate=(
                            f"{confidence:.0%} Long Breach"
                            "<br>Date: %{x|%Y-%m-%d}"
                            f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=upper_breach_returns.index,
                        y=upper_breach_returns,
                        mode="markers",
                        name=f"{confidence:.0%} Short Breaches",
                        marker=dict(
                            color=short_colors.get(confidence, "#a855f7"),
                            size=7,
                            symbol="x",
                        ),
                        hovertemplate=(
                            f"{confidence:.0%} Short Breach"
                            "<br>Date: %{x|%Y-%m-%d}"
                            f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=lower_var_return.index,
                        y=lower_var_return,
                        mode="lines",
                        name=f"{confidence:.0%} Long VaR Floor",
                        line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Long VaR Floor"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Forecast Threshold: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=lower_es_return.index,
                        y=lower_es_return,
                        mode="lines",
                        name=f"{confidence:.0%} Long CVaR Floor",
                        line=dict(
                            color=long_colors.get(confidence, "#ef4444"),
                            width=2,
                            dash="dot",
                        ),
                        hovertemplate=(
                            f"{confidence:.0%} Long CVaR Floor"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Expected Shortfall: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=upper_var_return.index,
                        y=upper_var_return,
                        mode="lines",
                        name=f"{confidence:.0%} Short VaR Ceiling",
                        line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Short VaR Ceiling"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Forecast Threshold: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=upper_es_return.index,
                        y=upper_es_return,
                        mode="lines",
                        name=f"{confidence:.0%} Short CVaR Ceiling",
                        line=dict(
                            color=short_colors.get(confidence, "#a855f7"),
                            width=2,
                            dash="dot",
                        ),
                        hovertemplate=(
                            f"{confidence:.0%} Short CVaR Ceiling"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Expected Shortfall: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=2,
                    col=1,
                )

                for series in (
                    lower_var_return,
                    upper_var_return,
                    lower_es_return,
                    upper_es_return,
                    lower_breach_returns,
                    upper_breach_returns,
                ):
                    if not series.empty:
                        index_candidates.append(series.index)

            for confidence in tail_levels:
                metric_set = metrics_by_confidence.get(confidence, {})
                lower_breach_rate = metric_set.get(
                    "lower_rolling_breach_rate",
                    pd.Series(dtype=float),
                ).dropna()
                upper_breach_rate = metric_set.get(
                    "upper_rolling_breach_rate",
                    pd.Series(dtype=float),
                ).dropna()
                either_side_breach_rate = metric_set.get(
                    "either_side_rolling_breach_rate",
                    pd.Series(dtype=float),
                ).dropna()

                fig.add_trace(
                    go.Scatter(
                        x=lower_breach_rate.index,
                        y=lower_breach_rate,
                        mode="lines",
                        name=f"{confidence:.0%} Long Breach Rate",
                        line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Long Breach Rate"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Breach Rate: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=upper_breach_rate.index,
                        y=upper_breach_rate,
                        mode="lines",
                        name=f"{confidence:.0%} Short Breach Rate",
                        line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                        hovertemplate=(
                            f"{confidence:.0%} Short Breach Rate"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Breach Rate: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=either_side_breach_rate.index,
                        y=either_side_breach_rate,
                        mode="lines",
                        name=f"{confidence:.0%} Either-Side Breach Rate",
                        line=dict(
                            color="#f59e0b" if confidence == 0.95 else "#b45309",
                            width=2,
                        ),
                        hovertemplate=(
                            f"{confidence:.0%} Either-Side Breach Rate"
                            "<br>Date: %{x|%Y-%m-%d}"
                            "<br>Breach Rate: %{y:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=3,
                    col=1,
                )

                for series in (lower_breach_rate, upper_breach_rate, either_side_breach_rate):
                    if not series.empty:
                        index_candidates.append(series.index)

            added_traces = len(fig.data) - trace_start
            if traces_per_window is None:
                traces_per_window = added_traces

        for confidence in tail_levels:
            expected_tail_rate = 1.0 - confidence
            reference_color = "#94a3b8" if confidence == 0.95 else "#64748b"
            fig.add_hline(
                y=expected_tail_rate,
                row=3,
                col=1,
                line_dash="dot",
                line_color=reference_color,
                line_width=1.5,
            )
            fig.add_annotation(
                x=0.99,
                y=expected_tail_rate,
                xref="x3 domain",
                yref="y3",
                text=f"{confidence:.0%} Expected Long / Short {expected_tail_rate:.2%}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10, color=reference_color),
            )
            either_side_expected_rate = min(1.0, 2.0 * expected_tail_rate)
            fig.add_hline(
                y=either_side_expected_rate,
                row=3,
                col=1,
                line_dash="dashdot",
                line_color=reference_color,
                line_width=1.5,
            )
            fig.add_annotation(
                x=0.99,
                y=either_side_expected_rate,
                xref="x3 domain",
                yref="y3",
                text=f"{confidence:.0%} Expected Either-Side {either_side_expected_rate:.2%}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10, color=reference_color),
            )

        fig.add_hline(
            y=0,
            row=1,
            col=1,
            line_dash="dash",
            line_color="rgba(248, 250, 252, 0.40)",
            line_width=1,
        )
        fig.add_hline(
            y=0,
            row=2,
            col=1,
            line_dash="dash",
            line_color="rgba(248, 250, 252, 0.40)",
            line_width=1,
        )
        fig.update_yaxes(title_text=history_labels["return_name"], tickformat=".2%", row=1, col=1)
        fig.update_yaxes(title_text="Forecast Return Threshold", tickformat=".2%", row=2, col=1)
        fig.update_yaxes(title_text="Breach Rate", tickformat=".2%", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        if traces_per_window is None or traces_per_window <= 0:
            raise ValueError("Unable to build trade-range traces from the supplied history payload.")

        non_empty_indices = [index for index in index_candidates if len(index) > 0]
        if non_empty_indices:
            global_start = min(index[0] for index in non_empty_indices)
            global_end = max(index[-1] for index in non_empty_indices)
            default_start = max(global_start, global_end - pd.DateOffset(years=3))
            fig.update_xaxes(range=[default_start, global_end])

        total_traces = len(fig.data)
        buttons = []
        for idx, window in enumerate(window_options):
            visibility = build_visibility_mask(
                total_traces=total_traces,
                active_window_index=idx,
                traces_per_window=traces_per_window,
                constant_trace_indices=[],
            )
            buttons.append(
                dict(
                    label=str(window),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": self._header_title(
                                f"{ticker_label} {history_labels['header_basis']} ({window}-Session Lookback)"
                            )
                        },
                    ],
                )
            )

        fig.update_layout(
            updatemenus=[
                self._dropdown_menu(
                    buttons=buttons,
                    x=0.0,
                    active=window_options.index(default_window),
                )
            ],
            title=self._header_title(
                f"{ticker_label} {history_labels['header_basis']} ({default_window}-Session Lookback)"
            ),
            height=1325,
            margin=self._header_margin(top=170),
            template=template,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
            yaxis_title="Annualized Volatility (Rolling Std of Excess Returns)",
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

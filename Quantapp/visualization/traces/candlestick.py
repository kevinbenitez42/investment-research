"""Candlestick trace builders and price-axis helpers."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Quantapp.analytics.rolling import Rolling
from Quantapp.analytics.series_utils import calculate_textbook_rolling_max_drawdown

rolling = Rolling()


def build_time_range(period_start, period_end, offset=None):
    """Build a padded time-axis range for a visible period."""
    if period_start is None or period_end is None:
        return None
    start = period_start if offset is None else max(period_start, period_end - offset)
    span_days = max((period_end - start).days, 1)
    padding_days = max(10, int(span_days * 0.08))
    return [start, period_end + pd.Timedelta(days=padding_days)]


def slice_series_to_range(series, start=None, end=None):
    """Return the visible slice of a numeric series."""
    if series is None:
        return pd.Series(dtype=float)
    sliced = pd.Series(series).dropna()
    if sliced.empty:
        return sliced
    if start is not None:
        sliced = sliced.loc[sliced.index >= start]
    if end is not None:
        sliced = sliced.loc[sliced.index <= end]
    return sliced


def build_numeric_axis_range(series_list, include_zero=False, padding_ratio=0.08):
    """Build a padded numeric axis range from one or more series."""
    numeric_arrays = []
    for series in series_list:
        cleaned = pd.Series(series).dropna()
        if cleaned.empty:
            continue
        numeric_arrays.append(cleaned.astype(float).to_numpy())

    if not numeric_arrays:
        return None

    min_value = min(float(np.nanmin(values)) for values in numeric_arrays)
    max_value = max(float(np.nanmax(values)) for values in numeric_arrays)

    if include_zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)

    span = max_value - min_value
    if span <= 0:
        padding = max(abs(max_value), abs(min_value), 1.0) * padding_ratio
    else:
        padding = span * padding_ratio

    return [min_value - padding, max_value + padding]


def build_candlestick_y_range(bundle, start=None, end=None, overlay_mode="all"):
    """Build a visible y-axis range for a candlestick panel plus overlays."""
    price_frame = bundle.get("price_frame")
    overlay_series_map = bundle.get("overlay_series_map", {})

    series_list = []
    if isinstance(price_frame, pd.DataFrame) and not price_frame.empty:
        for column in ("High", "Low"):
            series_list.append(slice_series_to_range(price_frame[column], start=start, end=end))

    if overlay_mode == "all":
        overlay_groups = overlay_series_map.values()
    else:
        overlay_groups = [overlay_series_map.get(overlay_mode, [])]

    for overlay_group in overlay_groups:
        for series in overlay_group:
            series_list.append(slice_series_to_range(series, start=start, end=end))

    return build_numeric_axis_range(series_list, include_zero=False, padding_ratio=0.08)


def build_candlestick_trace_bundle(
    ticker_data,
    drop_window=14,
    period="1Y",
    bollinger_window=21,
    max_drawdown_price_windows=None,
):
    """Prepare candlestick traces plus overlay metadata for composed views."""
    ticker_data = ticker_data.sort_index().copy()
    ticker_data = ticker_data[ticker_data.index.dayofweek < 5]
    holidays = pd.to_datetime(["2023-01-01", "2023-12-25"])
    ticker_data = ticker_data[~ticker_data.index.isin(holidays)]
    ticker_data = rolling.calculate_percentage_drop(ticker_data, windows=drop_window)
    mean_drop = ticker_data["PercentageDrop"].mean()
    std_drop = ticker_data["PercentageDrop"].std()

    period_data = ticker_data.copy() if period is None else ticker_data.last(period)
    if period_data.empty:
        raise ValueError("No candlestick data available for the selected period.")

    colors = [
        "red"
        if drop < mean_drop - (0.5 * std_drop)
        else "blue"
        if drop < mean_drop + (0.25 * std_drop)
        else "green"
        for drop in period_data["PercentageDrop"]
    ]

    moving_average = period_data["Close"].rolling(window=bollinger_window).mean()
    rolling_std = period_data["Close"].rolling(window=bollinger_window).std()

    bollinger_bands = {}
    for level in (1, 2, 3):
        bollinger_bands[f"Upper_{level}"] = moving_average + (rolling_std * level)
        bollinger_bands[f"Lower_{level}"] = moving_average - (rolling_std * level)
    bollinger_df = pd.DataFrame(bollinger_bands)

    band_colors = {
        1: "#4cc9f0",
        2: "#ffd166",
        3: "#ef476f",
    }
    moving_average_color = "#f8f9fa"

    traces = []
    always_visible_trace_indices = []
    overlay_trace_groups = {
        "bollinger": [],
        "mapped_mdd": [],
    }
    overlay_series_map = {
        "bollinger": [moving_average.copy()],
        "mapped_mdd": [],
    }
    overlay_series_map["bollinger"].extend(
        bollinger_df[column].copy() for column in bollinger_df.columns
    )

    def append_trace(trace, group):
        traces.append(copy.deepcopy(trace))
        trace_index = len(traces) - 1
        if group == "always":
            always_visible_trace_indices.append(trace_index)
        else:
            overlay_trace_groups[group].append(trace_index)

    for index, color in enumerate(colors):
        append_trace(
            go.Candlestick(
                x=[period_data.index[index]],
                open=[period_data["Open"].iloc[index]],
                high=[period_data["High"].iloc[index]],
                low=[period_data["Low"].iloc[index]],
                close=[period_data["Close"].iloc[index]],
                increasing_line_color=color,
                decreasing_line_color=color,
                showlegend=False,
            ),
            "always",
        )

    append_trace(
        go.Scatter(
            x=period_data.index,
            y=moving_average,
            mode="lines",
            line=dict(width=2, color=moving_average_color),
            name=f"{bollinger_window}-Period MA",
        ),
        "bollinger",
    )

    for level in (1, 2, 3):
        band_color = band_colors[level]
        append_trace(
            go.Scatter(
                x=period_data.index,
                y=bollinger_df[f"Upper_{level}"],
                mode="lines",
                line=dict(width=1, dash="dash", color=band_color),
                name=f"Upper +{level} SD",
            ),
            "bollinger",
        )
        append_trace(
            go.Scatter(
                x=period_data.index,
                y=bollinger_df[f"Lower_{level}"],
                mode="lines",
                line=dict(width=1, dash="dash", color=band_color),
                name=f"Lower -{level} SD",
            ),
            "bollinger",
        )

    for level in (1, 2, 3):
        band_color = band_colors[level]
        for band_name, label_prefix, text_position in (
            (f"Upper_{level}", f"+{level} SD", "top right"),
            (f"Lower_{level}", f"-{level} SD", "bottom right"),
        ):
            latest_band = bollinger_df[band_name].dropna()
            if latest_band.empty:
                continue

            latest_x = latest_band.index[-1]
            latest_y = latest_band.iloc[-1]
            append_trace(
                go.Scatter(
                    x=[latest_x],
                    y=[latest_y],
                    mode="markers+text",
                    marker=dict(size=7, color=band_color),
                    text=[f"{label_prefix}: {latest_y:,.2f}"],
                    textposition=text_position,
                    textfont=dict(size=11, color=band_color),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                "bollinger",
            )

    latest_ma = moving_average.dropna()
    if not latest_ma.empty:
        ma_x = latest_ma.index[-1]
        ma_y = latest_ma.iloc[-1]
        append_trace(
            go.Scatter(
                x=[ma_x],
                y=[ma_y],
                mode="markers+text",
                marker=dict(size=7, color=moving_average_color, line=dict(color="#111827", width=1)),
                text=[f"MA({bollinger_window}): {ma_y:,.2f}"],
                textposition="top right",
                textfont=dict(size=11, color=moving_average_color),
                showlegend=False,
                hoverinfo="skip",
            ),
            "bollinger",
        )

    latest_close_annotation = None
    latest_close = period_data["Close"].dropna()
    if not latest_close.empty:
        close_x = latest_close.index[-1]
        close_y = latest_close.iloc[-1]
        append_trace(
            go.Scatter(
                x=[close_x],
                y=[close_y],
                mode="markers",
                marker=dict(size=8, color="white", line=dict(color="#111827", width=1)),
                name="Latest Close",
                showlegend=False,
            ),
            "always",
        )
        latest_close_annotation = dict(
            x=close_x,
            y=close_y,
            text=f"Close: {close_y:,.2f}",
            showarrow=False,
            xanchor="left",
            xshift=12,
            font=dict(size=11, color="white"),
            bgcolor="rgba(17, 24, 39, 0.92)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
        )

    if max_drawdown_price_windows:
        drawdown_level_colors = {
            21: "#ef4444",
            50: "#f59e0b",
            200: "#8b5cf6",
        }
        close_series = ticker_data["Close"].dropna()
        for window in [int(value) for value in max_drawdown_price_windows]:
            rolling_peak = close_series.rolling(window=window, min_periods=1).max()
            textbook_drawdown = calculate_textbook_rolling_max_drawdown(
                close_series,
                window=window,
            ).dropna()
            mapped_drawdown_level = (
                rolling_peak.reindex(textbook_drawdown.index)
                .mul(1.0 + textbook_drawdown)
                .reindex(period_data.index)
                .dropna()
            )
            if mapped_drawdown_level.empty:
                continue

            level_color = drawdown_level_colors.get(window, "#f97316")
            level_label = f"{window}-Day Mapped Textbook Max DD Price"
            overlay_series_map["mapped_mdd"].append(mapped_drawdown_level.copy())
            append_trace(
                go.Scatter(
                    x=mapped_drawdown_level.index,
                    y=mapped_drawdown_level.values,
                    mode="lines",
                    line=dict(width=1.5, dash="dot", color=level_color),
                    name=level_label,
                    hovertemplate=f"{level_label}: %{{y:,.2f}}<extra></extra>",
                ),
                "mapped_mdd",
            )
            latest_drawdown_x = mapped_drawdown_level.index[-1]
            latest_drawdown_y = mapped_drawdown_level.iloc[-1]
            append_trace(
                go.Scatter(
                    x=[latest_drawdown_x],
                    y=[latest_drawdown_y],
                    mode="markers+text",
                    marker=dict(size=6, color=level_color),
                    text=[f"Mapped MDD {window}d: {latest_drawdown_y:,.2f}"],
                    textposition="middle right",
                    textfont=dict(size=11, color=level_color),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                "mapped_mdd",
            )

    period_start = period_data.index.min()
    period_end = period_data.index.max()

    return {
        "traces": traces,
        "always_visible_trace_indices": always_visible_trace_indices,
        "overlay_trace_groups": overlay_trace_groups,
        "default_overlay": "bollinger",
        "latest_close_annotation": latest_close_annotation,
        "period_start": period_start,
        "period_end": period_end,
        "xaxis_range": build_time_range(period_start, period_end),
        "right_margin": 240 if max_drawdown_price_windows else 180,
        "price_frame": period_data[["Open", "High", "Low", "Close"]].copy(),
        "overlay_series_map": overlay_series_map,
    }

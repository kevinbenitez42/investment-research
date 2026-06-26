"""Shared helpers for momentum diagnostic views."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Quantapp.visualization.core.theme import apply_plotly_dark_theme

HEADER_TOP_MARGIN = 150
HEADER_TITLE_Y = 0.97
HEADER_MENU_Y = 1.08

REFERENCE_WINDOW_STYLE_MAP = {
    7: ("orange", "7 Days"),
    21: ("red", "21 Days"),
    50: ("blue", "50 Days"),
    200: ("green", "200 Days"),
}
DEFAULT_HIGHLIGHT_WINDOWS = tuple(REFERENCE_WINDOW_STYLE_MAP)
DEFAULT_SURFACE_YEARS = 10


def header_margin(top=None):
    return dict(t=HEADER_TOP_MARGIN if top is None else int(top))


def header_title(text):
    return dict(
        text=str(text),
        x=0.5,
        xanchor="center",
        y=HEADER_TITLE_Y,
        yanchor="top",
    )


def dropdown_menu(
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
        y=HEADER_MENU_Y if y is None else y,
        yanchor=yanchor,
    )
    if active is not None:
        menu["active"] = active
    menu.update(overrides)
    return menu


def finalize_dark_figure(fig: go.Figure) -> go.Figure:
    """Apply the shared dark theme before returning a notebook-ready figure."""
    fig.update_layout(autosize=True)
    return apply_plotly_dark_theme(fig)


def coerce_positive_int(value):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def coerce_timestamp(value):
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


def trace_datetime_bounds(traces):
    starts = []
    ends = []
    for trace in traces:
        x_values = getattr(trace, "x", None)
        if x_values is None or len(x_values) == 0:
            continue

        start = coerce_timestamp(x_values[0])
        end = coerce_timestamp(x_values[-1])
        if start is None or end is None:
            continue

        starts.append(start)
        ends.append(end)

    if not starts:
        return None, None
    return min(starts), max(ends)


def preferred_numeric_window(options, preferred=200):
    normalized = []
    seen = set()
    for option in options:
        coerced = coerce_positive_int(option)
        if coerced is None or coerced in seen:
            continue
        normalized.append(coerced)
        seen.add(coerced)

    if not normalized:
        return None
    if preferred in seen:
        return preferred
    return max(normalized)


def window_value_from_label(label, config=None):
    if isinstance(config, Mapping):
        config_window = coerce_positive_int(config.get("time_frame"))
        if config_window is not None:
            return config_window

    digits = "".join(ch if ch.isdigit() else " " for ch in str(label)).split()
    if not digits:
        return None
    return coerce_positive_int(digits[0])


def preferred_window_label(label_map, preferred=200):
    labels = list(label_map.keys())
    if not labels:
        return None

    exact_label = f"{preferred}-day"
    if exact_label in label_map:
        return exact_label

    for label, config in label_map.items():
        if window_value_from_label(label, config) == preferred:
            return label

    return max(
        labels,
        key=lambda label: window_value_from_label(label, label_map.get(label)) or -1,
    )


def preferred_term_key(time_frame_map, term_options=None, preferred=200):
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
        if coerce_positive_int(time_frame_map.get(term)) == preferred:
            return term

    return max(
        candidates,
        key=lambda term: coerce_positive_int(time_frame_map.get(term)) or -1,
    )


def _coerce_diagnostics_mapping(diagnostics_context, required_keys):
    if not isinstance(diagnostics_context, dict):
        try:
            diagnostics_context = dict(diagnostics_context)
        except Exception as exc:
            raise TypeError("diagnostics_context must be a mapping.") from exc

    missing = [key for key in required_keys if key not in diagnostics_context]
    if missing:
        raise ValueError(f"diagnostics_context missing required keys: {missing}")

    return dict(diagnostics_context)


def _add_sharpe_display_fields(context):
    sharpe_table = context["sharpe_table"]
    sharpe_only = sharpe_table.drop(columns="Optimal_Window", errors="ignore")
    surface_years = int(context.get("surface_years", DEFAULT_SURFACE_YEARS))

    if sharpe_only.empty:
        sharpe_surface = sharpe_only
    else:
        last_date = sharpe_only.index[-1]
        start_date = last_date - pd.Timedelta(days=365 * surface_years)
        sharpe_surface = sharpe_only.loc[start_date:last_date]

    context["highlight_windows"] = context.get("highlight_windows", DEFAULT_HIGHLIGHT_WINDOWS)
    context["sharpe_only"] = sharpe_only
    context["sharpe_surface"] = sharpe_surface
    context["surface_years"] = surface_years
    return context


def coerce_sharpe_surface_context(diagnostics_context):
    context = _coerce_diagnostics_mapping(diagnostics_context, {"sharpe_table"})
    return _add_sharpe_display_fields(context)


def coerce_momentum_diagnostics_context(diagnostics_context):
    context = _coerce_diagnostics_mapping(
        diagnostics_context,
        {"sharpe_table", "volatility_df"},
    )
    context = _add_sharpe_display_fields(context)
    sharpe_only = context["sharpe_only"]
    optimal_windows = sharpe_only.idxmax(axis=1).dropna()
    volatility_df = context["volatility_df"]
    window_sizes = context.get("window_sizes", list(sharpe_only.columns))

    latest_sharpe_row = sharpe_only.dropna(how="all").tail(1)
    if latest_sharpe_row.empty:
        current_sharpe = pd.Series(dtype=float)
    else:
        current_sharpe = latest_sharpe_row.iloc[0].reindex(window_sizes)

    def sharpe_zscore(series):
        clean = pd.Series(series).dropna().sort_index()
        if clean.empty:
            return pd.Series(dtype=float)
        std = clean.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=clean.index)
        return (clean - clean.mean()) / std

    sharpe_zscore_frame = sharpe_only.apply(sharpe_zscore)
    sharpe_zscore_mean_by_window = sharpe_zscore_frame.mean().reindex(window_sizes)
    sharpe_zscore_std_by_window = sharpe_zscore_frame.std().reindex(window_sizes)
    cross_window_mean = sharpe_zscore_frame.mean(axis=1)
    cross_window_std = sharpe_zscore_frame.std(axis=1)
    cross_window_zscore_frame = sharpe_zscore_frame.sub(cross_window_mean, axis=0).div(
        cross_window_std.replace(0, np.nan),
        axis=0,
    )
    cross_window_zscore_frame.loc[cross_window_std == 0] = 0.0
    cross_window_zscore_mean_by_window = cross_window_zscore_frame.mean().reindex(window_sizes)
    cross_window_zscore_std_by_window = cross_window_zscore_frame.std().reindex(window_sizes)

    latest_sharpe_zscore_row = sharpe_zscore_frame.dropna(how="all").tail(1)
    if latest_sharpe_zscore_row.empty:
        current_sharpe_zscore = pd.Series(dtype=float)
        current_sharpe_zscore_date = None
        current_sharpe_cross_window_zscore = pd.Series(dtype=float)
    else:
        current_sharpe_zscore = latest_sharpe_zscore_row.iloc[0].reindex(window_sizes)
        current_sharpe_zscore_date = latest_sharpe_zscore_row.index[0]
        latest_cross_window_zscore_row = cross_window_zscore_frame.dropna(how="all").tail(1)
        if latest_cross_window_zscore_row.empty:
            current_sharpe_cross_window_zscore = pd.Series(dtype=float)
        else:
            current_sharpe_cross_window_zscore = latest_cross_window_zscore_row.iloc[0].reindex(window_sizes)

    context["window_sizes"] = window_sizes
    context["optimal_windows"] = optimal_windows
    context["optimal_windows_int"] = optimal_windows.astype(int)
    context["current_sharpe"] = current_sharpe
    context["current_sharpe_zscore"] = current_sharpe_zscore
    context["current_sharpe_zscore_date"] = current_sharpe_zscore_date
    context["sharpe_zscore_mean_by_window"] = sharpe_zscore_mean_by_window
    context["sharpe_zscore_std_by_window"] = sharpe_zscore_std_by_window
    context["current_sharpe_cross_window_zscore"] = current_sharpe_cross_window_zscore
    context["cross_window_zscore_mean_by_window"] = cross_window_zscore_mean_by_window
    context["cross_window_zscore_std_by_window"] = cross_window_zscore_std_by_window
    context["mean_sharpe"] = sharpe_only.mean()
    context["median_sharpe"] = sharpe_only.median()
    context["std_sharpe"] = sharpe_only.std()
    context["mean_volatility"] = volatility_df.mean()
    context["median_volatility"] = volatility_df.median()
    return context


def add_reference_vlines(fig: go.Figure, highlight_windows, *, row=None, col=None) -> None:
    for window in highlight_windows:
        color, label = REFERENCE_WINDOW_STYLE_MAP.get(window, ("gray", f"{window} Days"))
        subplot_kwargs = {}
        if row is not None and col is not None:
            subplot_kwargs = {"row": row, "col": col}
        fig.add_vline(
            x=window,
            line_color=color,
            line_dash="dash",
            annotation_text=label,
            annotation_position="top left",
            **subplot_kwargs,
        )

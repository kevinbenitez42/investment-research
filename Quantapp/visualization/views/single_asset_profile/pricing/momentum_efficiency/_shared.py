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


def coerce_momentum_diagnostics_context(diagnostics_context):
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
    if not isinstance(diagnostics_context, dict):
        try:
            diagnostics_context = dict(diagnostics_context)
        except Exception as exc:
            raise TypeError("diagnostics_context must be a mapping.") from exc

    missing = [key for key in required_keys if key not in diagnostics_context]
    if missing:
        raise ValueError(f"diagnostics_context missing required keys: {missing}")

    return diagnostics_context


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

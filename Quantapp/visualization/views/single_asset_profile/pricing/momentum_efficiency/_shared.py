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


def _normalize_ratio_type(value):
    ratio_type = str(value or "sharpe").strip()
    if ratio_type.lower().startswith("correlation::") and ratio_type.split("::", 1)[1].strip():
        return "correlation::" + ratio_type.split("::", 1)[1].strip()
    for metric_name in ("appraisal", "treynor", "information"):
        prefix = f"{metric_name}::"
        if ratio_type.lower().startswith(prefix) and ratio_type.split("::", 1)[1].strip():
            return prefix + ratio_type.split("::", 1)[1].strip()
    ratio_type = ratio_type.lower()
    if ratio_type not in {
        "sharpe", "sortino", "return", "volatility", "downside_volatility"
    }:
        raise ValueError(
            "ratio_type must be 'sharpe', 'sortino', 'return', 'volatility', "
            "or 'downside_volatility'."
        )
    return ratio_type


def _ratio_label(ratio_type):
    ratio_type = _normalize_ratio_type(ratio_type)
    if ratio_type.startswith("correlation::"):
        return f"Correlation vs {ratio_type.split('::', 1)[1]}"
    for metric_name in ("appraisal", "treynor", "information"):
        if ratio_type.startswith(f"{metric_name}::"):
            return f"{metric_name.title()} vs {ratio_type.split('::', 1)[1]}"
    return {
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "return": "Return",
        "volatility": "Volatility",
        "downside_volatility": "Downside Volatility",
    }[ratio_type]


def _coerce_ratio_context(diagnostics_context, required_keys=()):
    context = _coerce_diagnostics_mapping(diagnostics_context, set(required_keys))
    ratio_type = _normalize_ratio_type(context.get("ratio_type", "sharpe"))
    ratio_table = context.get("ratio_table")
    if ratio_table is None:
        ratio_table = context.get(f"{ratio_type}_table")
    if ratio_table is None:
        # ``sharpe_table`` was the original public schema. It remains the final
        # fallback so older notebook contexts continue to coerce unchanged.
        ratio_table = context.get("sharpe_table")
    if ratio_table is None:
        raise ValueError(
            "diagnostics_context must provide ratio_table, "
            f"{ratio_type}_table, or the legacy sharpe_table key."
        )

    ratio_table = (
        ratio_table
        if isinstance(ratio_table, pd.DataFrame)
        else pd.DataFrame(ratio_table)
    )
    ratio_label = str(context.get("ratio_label") or _ratio_label(ratio_type))
    context["ratio_type"] = ratio_type
    context["ratio_label"] = ratio_label
    context["ratio_table"] = ratio_table
    context[f"{ratio_type}_table"] = ratio_table
    # Compatibility alias: existing visualization code can consume a Sortino
    # context before all callers have migrated to the generic field names.
    context["sharpe_table"] = ratio_table
    return context


def _add_ratio_display_fields(context):
    ratio_table = context["ratio_table"]
    ratio_only = (
        ratio_table.drop(columns="Optimal_Window")
        if "Optimal_Window" in ratio_table.columns
        else ratio_table
    )
    surface_years = int(context.get("surface_years", DEFAULT_SURFACE_YEARS))

    if ratio_only.empty:
        ratio_surface = ratio_only
    else:
        last_date = ratio_only.index[-1]
        start_date = last_date - pd.Timedelta(days=365 * surface_years)
        ratio_surface = ratio_only.loc[start_date:last_date]

    context["highlight_windows"] = context.get("highlight_windows", DEFAULT_HIGHLIGHT_WINDOWS)
    context["ratio_only"] = ratio_only
    context["ratio_surface"] = ratio_surface
    context[f"{context['ratio_type']}_only"] = ratio_only
    context[f"{context['ratio_type']}_surface"] = ratio_surface
    # Legacy aliases deliberately point at the selected ratio, not always at
    # Sharpe. That lets older renderers display the correct values while their
    # labels are migrated separately.
    context["sharpe_only"] = ratio_only
    context["sharpe_surface"] = ratio_surface
    context["surface_years"] = surface_years
    return context


def _add_sharpe_display_fields(context):
    """Backward-compatible wrapper for the ratio-generic display schema."""
    return _add_ratio_display_fields(_coerce_ratio_context(context))


def coerce_ratio_surface_context(diagnostics_context):
    context = _coerce_ratio_context(diagnostics_context)
    return _add_ratio_display_fields(context)


def coerce_sharpe_surface_context(diagnostics_context):
    """Backward-compatible name for :func:`coerce_ratio_surface_context`."""
    return coerce_ratio_surface_context(diagnostics_context)


def coerce_momentum_diagnostics_context(diagnostics_context):
    context = _coerce_ratio_context(diagnostics_context, {"volatility_df"})
    context = _add_ratio_display_fields(context)
    ratio_only = context["ratio_only"]
    optimal_windows = ratio_only.dropna(how="all").idxmax(axis=1).dropna()
    volatility_df = context["volatility_df"]
    window_sizes = context.get("window_sizes", list(ratio_only.columns))

    latest_ratio_row = ratio_only.dropna(how="all").tail(1)
    if latest_ratio_row.empty:
        current_ratio = pd.Series(dtype=float)
    else:
        current_ratio = latest_ratio_row.iloc[0].reindex(window_sizes)

    def ratio_zscore(series):
        clean = pd.Series(series).dropna().sort_index()
        if clean.empty:
            return pd.Series(dtype=float)
        std = clean.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=clean.index)
        return (clean - clean.mean()) / std

    ratio_zscore_frame = ratio_only.apply(ratio_zscore)
    ratio_zscore_mean_by_window = ratio_zscore_frame.mean().reindex(window_sizes)
    ratio_zscore_std_by_window = ratio_zscore_frame.std().reindex(window_sizes)
    cross_window_mean = ratio_zscore_frame.mean(axis=1)
    cross_window_std = ratio_zscore_frame.std(axis=1)
    cross_window_zscore_frame = ratio_zscore_frame.sub(cross_window_mean, axis=0).div(
        cross_window_std.replace(0, np.nan),
        axis=0,
    )
    cross_window_zscore_frame.loc[cross_window_std == 0] = 0.0
    cross_window_zscore_mean_by_window = cross_window_zscore_frame.mean().reindex(window_sizes)
    cross_window_zscore_std_by_window = cross_window_zscore_frame.std().reindex(window_sizes)

    latest_ratio_zscore_row = ratio_zscore_frame.dropna(how="all").tail(1)
    if latest_ratio_zscore_row.empty:
        current_ratio_zscore = pd.Series(dtype=float)
        current_ratio_zscore_date = None
        current_ratio_cross_window_zscore = pd.Series(dtype=float)
    else:
        current_ratio_zscore = latest_ratio_zscore_row.iloc[0].reindex(window_sizes)
        current_ratio_zscore_date = latest_ratio_zscore_row.index[0]
        latest_cross_window_zscore_row = cross_window_zscore_frame.dropna(how="all").tail(1)
        if latest_cross_window_zscore_row.empty:
            current_ratio_cross_window_zscore = pd.Series(dtype=float)
        else:
            current_ratio_cross_window_zscore = latest_cross_window_zscore_row.iloc[0].reindex(window_sizes)

    context["window_sizes"] = window_sizes
    context["optimal_windows"] = optimal_windows
    context["optimal_windows_int"] = optimal_windows.astype(int)
    context["current_ratio"] = current_ratio
    context["current_ratio_zscore"] = current_ratio_zscore
    context["current_ratio_zscore_date"] = current_ratio_zscore_date
    context["ratio_zscore_mean_by_window"] = ratio_zscore_mean_by_window
    context["ratio_zscore_std_by_window"] = ratio_zscore_std_by_window
    context["current_ratio_cross_window_zscore"] = current_ratio_cross_window_zscore
    context["cross_window_zscore_mean_by_window"] = cross_window_zscore_mean_by_window
    context["cross_window_zscore_std_by_window"] = cross_window_zscore_std_by_window
    context["mean_ratio"] = ratio_only.mean()
    context["median_ratio"] = ratio_only.median()
    context["std_ratio"] = ratio_only.std()

    ratio_type = context["ratio_type"]
    context[f"current_{ratio_type}"] = current_ratio
    context[f"current_{ratio_type}_zscore"] = current_ratio_zscore
    context[f"current_{ratio_type}_zscore_date"] = current_ratio_zscore_date
    context[f"{ratio_type}_zscore_mean_by_window"] = ratio_zscore_mean_by_window
    context[f"{ratio_type}_zscore_std_by_window"] = ratio_zscore_std_by_window
    context[f"current_{ratio_type}_cross_window_zscore"] = current_ratio_cross_window_zscore
    context[f"mean_{ratio_type}"] = context["mean_ratio"]
    context[f"median_{ratio_type}"] = context["median_ratio"]
    context[f"std_{ratio_type}"] = context["std_ratio"]

    # Original public field names remain selected-ratio aliases until every
    # downstream view has adopted the generic schema.
    context["current_sharpe"] = current_ratio
    context["current_sharpe_zscore"] = current_ratio_zscore
    context["current_sharpe_zscore_date"] = current_ratio_zscore_date
    context["sharpe_zscore_mean_by_window"] = ratio_zscore_mean_by_window
    context["sharpe_zscore_std_by_window"] = ratio_zscore_std_by_window
    context["current_sharpe_cross_window_zscore"] = current_ratio_cross_window_zscore
    context["mean_sharpe"] = context["mean_ratio"]
    context["median_sharpe"] = context["median_ratio"]
    context["std_sharpe"] = context["std_ratio"]
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

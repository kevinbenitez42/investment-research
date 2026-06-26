"""Option expiration P/L views for portfolio performance notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PORTFOLIO_EQUAL_MOVE = "ALL"
PORTFOLIO_BETA_MOVE = "ALL_BETA"
PORTFOLIO_BETA_PRICE = "ALL_BETA_PRICE"
PORTFOLIO_MODES = {PORTFOLIO_EQUAL_MOVE, PORTFOLIO_BETA_MOVE, PORTFOLIO_BETA_PRICE}
TOTAL_PL_LINE_COLOR = "#38BDF8"
PROFIT_FILL_COLOR = "rgba(34, 197, 94, 0.18)"
LOSS_FILL_COLOR = "rgba(239, 68, 68, 0.18)"
LEG_LINE_COLOR = "rgba(203, 213, 225, 0.55)"
ZERO_LINE_COLOR = "rgba(226, 232, 240, 0.75)"
CURRENT_LINE_COLOR = "#FACC15"
BREAKEVEN_LINE_COLOR = "#F472B6"
HALF_MAX_PROFIT_LINE_COLOR = "#A78BFA"
MAX_PROFIT_LINE_COLOR = "#22C55E"
AVG_MAX_PROFIT_LINE_COLOR = "rgba(134, 239, 172, 0.9)"
AVG_MAX_LOSS_LINE_COLOR = "rgba(252, 165, 165, 0.9)"
MEDIAN_MAX_PROFIT_LINE_COLOR = "rgba(74, 222, 128, 0.9)"
MEDIAN_MAX_LOSS_LINE_COLOR = "rgba(248, 113, 113, 0.9)"
DARK_PAPER_COLOR = "#0F172A"
DARK_PLOT_COLOR = "#111827"
DARK_GRID_COLOR = "rgba(148, 163, 184, 0.22)"
DARK_FONT_COLOR = "#E5E7EB"
BREAKEVEN_LEGEND_GROUP = "breakeven_guides"
HALF_MAX_PROFIT_LEGEND_GROUP = "half_max_profit_guides"
MAX_PROFIT_LEGEND_GROUP = "max_profit_guides"
AVG_MAX_PROFIT_LEGEND_GROUP = "avg_max_profit_guides"
AVG_MAX_LOSS_LEGEND_GROUP = "avg_max_loss_guides"
MEDIAN_MAX_PROFIT_LEGEND_GROUP = "median_max_profit_guides"
MEDIAN_MAX_LOSS_LEGEND_GROUP = "median_max_loss_guides"
GUIDE_VISIBLE_DEFAULT = "legendonly"


def _available_underlyings(positions_df: pd.DataFrame) -> list[str]:
    if positions_df.empty or "underlying" not in positions_df.columns:
        return []
    return sorted(positions_df["underlying"].dropna().unique().tolist())


def _ticker_lookup_key(ticker: str) -> str:
    return "".join(character for character in str(ticker).upper() if character.isalnum())


def _get_price_for_underlying(
    underlying_name: str,
    portfolio_latest_prices: Mapping[str, float],
) -> float:
    current_price = portfolio_latest_prices.get(underlying_name, np.nan)

    if pd.notna(current_price):
        return float(current_price)

    target_key = _ticker_lookup_key(underlying_name)

    for price_ticker, price_value in portfolio_latest_prices.items():
        if _ticker_lookup_key(price_ticker) == target_key and pd.notna(price_value):
            return float(price_value)

    return np.nan


def _portfolio_dropdown_options(benchmark_label: str) -> list[tuple[str, str]]:
    return [
        ("All assets (% move)", PORTFOLIO_EQUAL_MOVE),
        (f"All assets (beta % vs {benchmark_label})", PORTFOLIO_BETA_MOVE),
        (f"All assets (beta price vs {benchmark_label})", PORTFOLIO_BETA_PRICE),
    ]


def _asset_dropdown_options(underlyings: list[str]) -> list[tuple[str, str]]:
    return [(underlying_name, underlying_name) for underlying_name in underlyings]


def _underlying_dropdown_options(
    underlyings: list[str],
    benchmark_label: str,
    *,
    include_portfolio_modes: bool = True,
    include_assets: bool = True,
) -> list[tuple[str, str]]:
    options = []

    if include_portfolio_modes:
        options.extend(_portfolio_dropdown_options(benchmark_label))

    if include_assets:
        options.extend(_asset_dropdown_options(underlyings))

    return options


def _get_reference_price(
    underlying_name: str,
    legs_frame: pd.DataFrame,
    portfolio_latest_prices: Mapping[str, float],
) -> float:
    current_price = _get_price_for_underlying(underlying_name, portfolio_latest_prices)

    if pd.notna(current_price):
        return float(current_price)

    strikes = legs_frame["strike"].dropna()

    if not strikes.empty:
        return float(strikes.mean())

    return np.nan


def _get_underlying_beta(underlying_name: str, underlying_beta_map: Mapping[str, float]) -> float:
    beta_value = underlying_beta_map.get(underlying_name, 1.0)

    if pd.isna(beta_value) or not np.isfinite(beta_value):
        return 1.0

    return float(beta_value)


def _select_strikes_for_frame(
    legs_frame: pd.DataFrame,
    reference_price: float,
    selected_strike_range: str | int | float,
) -> tuple[np.ndarray, float, float, str]:
    unique_strikes = np.sort(legs_frame["strike"].dropna().unique())

    if len(unique_strikes) == 0:
        return unique_strikes, np.nan, np.nan, "No strikes"

    if selected_strike_range == "All":
        return unique_strikes, float(unique_strikes.min()), float(unique_strikes.max()), "All strikes"

    strike_range = float(selected_strike_range)
    lower_bound = reference_price - strike_range
    upper_bound = reference_price + strike_range
    selected_strikes = unique_strikes[(unique_strikes >= lower_bound) & (unique_strikes <= upper_bound)]

    if len(selected_strikes) == 0:
        nearest_idx = np.argmin(np.abs(unique_strikes - reference_price))
        selected_strikes = np.array([unique_strikes[nearest_idx]])
        lower_bound = float(selected_strikes[0])
        upper_bound = float(selected_strikes[0])

    return selected_strikes, float(lower_bound), float(upper_bound), f"+/- {strike_range:g}"


def _add_zero_and_current_guides(
    fig: go.Figure,
    *,
    current_marker_x: float,
    current_marker_text: str,
) -> None:
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color=ZERO_LINE_COLOR)
    fig.add_vline(
        x=current_marker_x,
        line_width=1.5,
        line_dash="dash",
        line_color=CURRENT_LINE_COLOR,
        annotation_text=current_marker_text,
        annotation_position="top right",
    )


def _add_current_price_guide(
    fig: go.Figure,
    *,
    current_marker_x: float,
    current_marker_text: str,
) -> None:
    if pd.isna(current_marker_x):
        return

    fig.add_vline(
        x=float(current_marker_x),
        line_width=1.5,
        line_dash="dash",
        line_color=CURRENT_LINE_COLOR,
        annotation_text=current_marker_text,
        annotation_position="top right",
    )


def _find_breakeven_points(x_values, total_pl) -> list[float]:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(total_pl, dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]

    if len(x_array) < 2:
        return []

    finite_abs_y = np.abs(y_array[np.isfinite(y_array)])
    zero_tolerance = max(float(finite_abs_y.max()) * 1e-9, 1e-8) if len(finite_abs_y) else 1e-8
    zero_mask = np.isclose(y_array, 0.0, atol=zero_tolerance)
    breakevens = []
    idx = 0

    while idx < len(x_array):
        if not zero_mask[idx]:
            idx += 1
            continue

        start_idx = idx

        while idx + 1 < len(x_array) and zero_mask[idx + 1]:
            idx += 1

        breakevens.append(float((x_array[start_idx] + x_array[idx]) / 2))
        idx += 1

    for idx in range(len(x_array) - 1):
        if zero_mask[idx] or zero_mask[idx + 1]:
            continue

        y_start = y_array[idx]
        y_end = y_array[idx + 1]

        if y_start * y_end < 0:
            x_start = x_array[idx]
            x_end = x_array[idx + 1]
            breakevens.append(float(x_start - y_start * (x_end - x_start) / (y_end - y_start)))

    deduped = []

    for breakeven in sorted(breakevens):
        if not deduped or not np.isclose(breakeven, deduped[-1], rtol=1e-5, atol=1e-5):
            deduped.append(breakeven)

    return deduped


def _find_pl_level_points(x_values, total_pl, target_pl: float) -> list[float]:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(total_pl, dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]

    if len(x_array) < 2 or not np.isfinite(target_pl):
        return []

    y_diff = y_array - target_pl
    finite_abs_y = np.abs(np.concatenate([y_array[np.isfinite(y_array)], np.array([target_pl])]))
    level_tolerance = max(float(finite_abs_y.max()) * 1e-9, 1e-8) if len(finite_abs_y) else 1e-8
    level_mask = np.isclose(y_diff, 0.0, atol=level_tolerance)
    level_points = []
    idx = 0

    while idx < len(x_array):
        if not level_mask[idx]:
            idx += 1
            continue

        start_idx = idx

        while idx + 1 < len(x_array) and level_mask[idx + 1]:
            idx += 1

        level_points.append(float((x_array[start_idx] + x_array[idx]) / 2))
        idx += 1

    for idx in range(len(x_array) - 1):
        if level_mask[idx] or level_mask[idx + 1]:
            continue

        y_start = y_diff[idx]
        y_end = y_diff[idx + 1]

        if y_start * y_end < 0:
            x_start = x_array[idx]
            x_end = x_array[idx + 1]
            level_points.append(float(x_start - y_start * (x_end - x_start) / (y_end - y_start)))

    deduped = []

    for level_point in sorted(level_points):
        if not deduped or not np.isclose(level_point, deduped[-1], rtol=1e-5, atol=1e-5):
            deduped.append(level_point)

    return deduped


def _least_loss_point(x_values, total_pl) -> tuple[float, float] | None:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(total_pl, dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]

    if len(x_array) == 0:
        return None

    best_idx = int(np.nanargmax(y_array))
    return float(x_array[best_idx]), float(y_array[best_idx])


def _max_profit_point(x_values, total_pl) -> tuple[float, float] | None:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(total_pl, dtype=float)
    valid_mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[valid_mask]
    y_array = y_array[valid_mask]

    if len(x_array) == 0:
        return None

    max_pl = float(np.nanmax(y_array))
    tolerance = max(abs(max_pl) * 1e-9, 1e-8)
    max_mask = np.isclose(y_array, max_pl, atol=tolerance)
    max_x_values = x_array[max_mask]

    if len(max_x_values) == 0:
        best_idx = int(np.nanargmax(y_array))
        return float(x_array[best_idx]), max_pl

    max_x = float((max_x_values.min() + max_x_values.max()) / 2)
    return max_x, max_pl


def _shape_legend_exists(fig: go.Figure, legendgroup: str) -> bool:
    for shape in fig.layout.shapes:
        if getattr(shape, "legendgroup", None) == legendgroup and getattr(shape, "showlegend", False):
            return True

    return False


def _add_vertical_guide_shape(
    fig: go.Figure,
    *,
    x_value: float,
    legendgroup: str,
    legend_name: str,
    label_text: str,
    line_color: str,
    line_dash: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    vline_kwargs = dict(
        x=x_value,
        line_width=2,
        line_dash=line_dash,
        line_color=line_color,
        visible=GUIDE_VISIBLE_DEFAULT,
        showlegend=not _shape_legend_exists(fig, legendgroup),
        name=legend_name,
        legendgroup=legendgroup,
        label=dict(text=label_text, font=dict(color=line_color, size=11)),
    )

    if row is not None and col is not None:
        fig.add_vline(**vline_kwargs, row=row, col=col)
    else:
        fig.add_vline(**vline_kwargs)


def _add_horizontal_guide_shape(
    fig: go.Figure,
    *,
    y_value: float,
    legendgroup: str,
    legend_name: str,
    label_text: str,
    line_color: str,
    line_dash: str = "dash",
) -> None:
    fig.add_hline(
        y=y_value,
        line_width=1.75,
        line_dash=line_dash,
        line_color=line_color,
        visible=GUIDE_VISIBLE_DEFAULT,
        showlegend=not _shape_legend_exists(fig, legendgroup),
        name=legend_name,
        legendgroup=legendgroup,
        label=dict(text=label_text, font=dict(color=line_color, size=11)),
    )


def _add_breakeven_guides(
    fig: go.Figure,
    *,
    x_values,
    total_pl,
    value_formatter=None,
    row: int | None = None,
    col: int | None = None,
) -> list[float]:
    breakevens = _find_breakeven_points(x_values, total_pl)
    formatter = value_formatter or (lambda value: f"{value:.2f}")

    if not breakevens:
        least_loss = _least_loss_point(x_values, total_pl)

        if least_loss is None or least_loss[1] >= 0:
            return []

        breakeven_x, least_loss_pl = least_loss
        _add_vertical_guide_shape(
            fig,
            x_value=breakeven_x,
            legendgroup=BREAKEVEN_LEGEND_GROUP,
            legend_name="Break Even / Least Loss",
            label_text=f"Least Loss: {formatter(breakeven_x)} | ${least_loss_pl:,.0f}",
            line_color=BREAKEVEN_LINE_COLOR,
            line_dash="dashdot",
            row=row,
            col=col,
        )
        return [breakeven_x]

    for idx, breakeven in enumerate(breakevens, start=1):
        prefix = "BE" if len(breakevens) == 1 else f"BE {idx}"
        _add_vertical_guide_shape(
            fig,
            x_value=breakeven,
            legendgroup=BREAKEVEN_LEGEND_GROUP,
            legend_name="Break Even / Least Loss",
            label_text=f"{prefix}: {formatter(breakeven)}",
            line_color=BREAKEVEN_LINE_COLOR,
            line_dash="dashdot",
            row=row,
            col=col,
        )

    return breakevens


def _add_max_profit_guide(
    fig: go.Figure,
    *,
    x_values,
    total_pl,
    value_formatter=None,
    skip_x_values: list[float] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> float | None:
    max_profit = _max_profit_point(x_values, total_pl)

    if max_profit is None:
        return None

    max_profit_x, max_profit_pl = max_profit

    for skip_x in skip_x_values or []:
        if np.isclose(max_profit_x, skip_x, rtol=1e-5, atol=1e-5):
            return None

    formatter = value_formatter or (lambda value: f"{value:.2f}")
    _add_vertical_guide_shape(
        fig,
        x_value=max_profit_x,
        legendgroup=MAX_PROFIT_LEGEND_GROUP,
        legend_name="Max Profit",
        label_text=f"Max Profit: {formatter(max_profit_x)} | ${max_profit_pl:,.0f}",
        line_color=MAX_PROFIT_LINE_COLOR,
        line_dash="dot",
        row=row,
        col=col,
    )
    return max_profit_x


def _add_half_max_profit_guides(
    fig: go.Figure,
    *,
    x_values,
    total_pl,
    value_formatter=None,
    skip_x_values: list[float] | None = None,
    row: int | None = None,
    col: int | None = None,
) -> list[float]:
    max_profit = _max_profit_point(x_values, total_pl)

    if max_profit is None:
        return []

    _, max_profit_pl = max_profit

    if max_profit_pl <= 0:
        return []

    half_profit_pl = max_profit_pl / 2
    half_profit_x_values = _find_pl_level_points(x_values, total_pl, half_profit_pl)
    formatter = value_formatter or (lambda value: f"{value:.2f}")
    drawn_x_values = []

    for idx, half_profit_x in enumerate(half_profit_x_values, start=1):
        if any(np.isclose(half_profit_x, skip_x, rtol=1e-5, atol=1e-5) for skip_x in skip_x_values or []):
            continue

        prefix = "Half Max Profit" if len(half_profit_x_values) == 1 else f"Half Max {idx}"
        _add_vertical_guide_shape(
            fig,
            x_value=half_profit_x,
            legendgroup=HALF_MAX_PROFIT_LEGEND_GROUP,
            legend_name="Half Max Profit",
            label_text=f"{prefix}: {formatter(half_profit_x)} | ${half_profit_pl:,.0f}",
            line_color=HALF_MAX_PROFIT_LINE_COLOR,
            line_dash="dash",
            row=row,
            col=col,
        )
        drawn_x_values.append(half_profit_x)

    return drawn_x_values


def _total_pl_trace_from_figure(fig: go.Figure) -> go.Scatter | None:
    for trace in reversed(fig.data):
        trace_name = str(trace.name or "")

        if trace.visible == "legendonly":
            continue

        if trace_name.startswith("Total"):
            return trace

    return None


def _add_panel_payoff_guides(
    fig: go.Figure,
    panel_figure: go.Figure,
    *,
    row: int,
    col: int,
    value_formatter=None,
) -> None:
    total_trace = _total_pl_trace_from_figure(panel_figure)

    if total_trace is None:
        return

    guide_x_values = _add_breakeven_guides(
        fig,
        x_values=total_trace.x,
        total_pl=total_trace.y,
        value_formatter=value_formatter,
        row=row,
        col=col,
    )
    half_profit_x_values = _add_half_max_profit_guides(
        fig,
        x_values=total_trace.x,
        total_pl=total_trace.y,
        value_formatter=value_formatter,
        skip_x_values=guide_x_values,
        row=row,
        col=col,
    )
    _add_max_profit_guide(
        fig,
        x_values=total_trace.x,
        total_pl=total_trace.y,
        value_formatter=value_formatter,
        skip_x_values=[*guide_x_values, *half_profit_x_values],
        row=row,
        col=col,
    )


def _expiration_label(leg: pd.Series) -> str:
    expiration = leg.get("expiration")
    return expiration.strftime("%b %d") if pd.notna(expiration) else "No Exp"


def _add_leg_trace(
    fig: go.Figure,
    *,
    x_values,
    leg_pl,
    trace_name: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=leg_pl,
            mode="lines",
            name=trace_name,
            line=dict(width=1.25, dash="dot", color=LEG_LINE_COLOR),
            visible="legendonly",
        )
    )


def _add_total_pl_trace(
    fig: go.Figure,
    *,
    x_values,
    total_pl,
    trace_name: str,
) -> None:
    positive_pl = np.where(total_pl >= 0, total_pl, np.nan)
    negative_pl = np.where(total_pl < 0, total_pl, np.nan)

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=positive_pl,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor=PROFIT_FILL_COLOR,
            hoverinfo="skip",
            showlegend=False,
            name="Profit area",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=negative_pl,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor=LOSS_FILL_COLOR,
            hoverinfo="skip",
            showlegend=False,
            name="Loss area",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=total_pl,
            mode="lines",
            name=trace_name,
            line=dict(width=4, color=TOTAL_PL_LINE_COLOR),
        )
    )


def _calculate_option_pl_at_prices(legs_frame: pd.DataFrame, prices: np.ndarray) -> np.ndarray:
    total_pl = np.zeros(len(prices))

    for _, leg in legs_frame.iterrows():
        strike = leg["strike"]
        opt_type = leg["option_type"]
        qty = leg["net_quantity"]
        avg_px = leg["average_price"]
        intrinsic = np.maximum(prices - strike, 0) if opt_type == "C" else np.maximum(strike - prices, 0)
        total_pl += qty * (intrinsic - avg_px) * 100

    return total_pl


def _format_currency_or_na(value: float) -> str:
    return f"${value:,.2f}" if pd.notna(value) else "N/A"


def _format_number_or_na(value: float) -> str:
    return f"{value:,.2f}" if pd.notna(value) else "N/A"


def _format_extreme_value(value: float) -> str:
    return "Unbounded" if np.isinf(value) else f"${value:,.2f}"


def _format_directional_axis_ticker(ticker: str, direction: str) -> str:
    escaped_ticker = escape(str(ticker))

    if direction == "down":
        return f"<span style='color:#F87171'>({escaped_ticker})</span>"

    return escaped_ticker


def _profit_direction_sign(
    *,
    has_unbounded_profit: bool,
    max_profit: float,
    best_price: float,
    current_price: float,
    current_pl: float,
) -> str:
    if has_unbounded_profit:
        return "up"

    if max_profit <= 0:
        return "none"

    if pd.isna(best_price):
        return "unknown"

    if pd.isna(current_price):
        return "unknown"

    if pd.notna(current_pl) and np.isclose(current_pl, max_profit):
        return "flat"

    if np.isclose(best_price, current_price):
        return "flat"

    return "up" if best_price > current_price else "down"


def _summarize_option_extremes_by_underlying(
    positions_df: pd.DataFrame,
    portfolio_latest_prices: Mapping[str, float],
) -> pd.DataFrame:
    rows = []

    for underlying_name, legs_frame in positions_df.groupby("underlying"):
        legs_frame = legs_frame.dropna(subset=["strike", "option_type", "net_quantity", "average_price"])

        if legs_frame.empty:
            continue

        candidate_prices = np.array(sorted({0.0, *legs_frame["strike"].astype(float).tolist()}))
        candidate_pl = _calculate_option_pl_at_prices(legs_frame, candidate_prices)
        worst_idx = int(np.nanargmin(candidate_pl))
        best_idx = int(np.nanargmax(candidate_pl))
        worst_finite_pl = float(candidate_pl[worst_idx])
        best_finite_pl = float(candidate_pl[best_idx])
        worst_price = float(candidate_prices[worst_idx])
        best_price = float(candidate_prices[best_idx])
        net_call_quantity = legs_frame.loc[legs_frame["option_type"] == "C", "net_quantity"].sum()
        has_unbounded_loss = bool(net_call_quantity < 0)
        has_unbounded_profit = bool(net_call_quantity > 0)
        current_price = _get_price_for_underlying(underlying_name, portfolio_latest_prices)
        current_pl = np.nan

        if pd.notna(current_price):
            current_pl = float(_calculate_option_pl_at_prices(legs_frame, np.array([float(current_price)]))[0])

        rows.append(
            {
                "underlying": underlying_name,
                "max_loss": np.inf if has_unbounded_loss else max(-worst_finite_pl, 0.0),
                "max_profit": np.inf if has_unbounded_profit else max(best_finite_pl, 0.0),
                "worst_finite_pl": worst_finite_pl,
                "best_finite_pl": best_finite_pl,
                "worst_price": worst_price,
                "best_price": best_price,
                "current_price": current_price,
                "current_pl": current_pl,
                "unbounded_loss": has_unbounded_loss,
                "unbounded_profit": has_unbounded_profit,
            }
        )

    return pd.DataFrame(rows)


def build_option_profit_loss_extremes_table(
    positions_df: pd.DataFrame,
    *,
    portfolio_latest_prices: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build a display table of max profit, max loss, and max-profit direction."""
    if positions_df.empty:
        return pd.DataFrame(columns=["max_profit", "max_loss", "direction"])

    portfolio_latest_prices = portfolio_latest_prices or {}
    extremes_frame = _summarize_option_extremes_by_underlying(positions_df, portfolio_latest_prices)

    if extremes_frame.empty:
        return pd.DataFrame(columns=["max_profit", "max_loss", "direction"])

    table = extremes_frame.copy()
    table["direction"] = table.apply(
        lambda row: _profit_direction_sign(
            has_unbounded_profit=row["unbounded_profit"],
            max_profit=row["max_profit"],
            best_price=row["best_price"],
            current_price=row["current_price"],
            current_pl=row["current_pl"],
        ),
        axis=1,
    )
    table["max_profit"] = table["max_profit"].map(_format_extreme_value)
    table["max_loss"] = table["max_loss"].map(_format_extreme_value)
    return (
        table.set_index("underlying")[["max_profit", "max_loss", "direction"]]
        .sort_index()
        .rename_axis("ticker")
    )


def build_option_max_loss_by_underlying_figure(
    positions_df: pd.DataFrame,
    *,
    portfolio_latest_prices: Mapping[str, float] | None = None,
) -> go.Figure | None:
    """Build a bar chart of theoretical max loss and max profit by option underlying."""
    if positions_df.empty:
        print("No option positions found in positions_df.")
        return None

    portfolio_latest_prices = portfolio_latest_prices or {}
    extremes_frame = _summarize_option_extremes_by_underlying(positions_df, portfolio_latest_prices)

    if extremes_frame.empty:
        print("No option positions were available for max-loss analysis.")
        return None

    extremes_frame["direction"] = extremes_frame.apply(
        lambda row: _profit_direction_sign(
            has_unbounded_profit=row["unbounded_profit"],
            max_profit=row["max_profit"],
            best_price=row["best_price"],
            current_price=row["current_price"],
            current_pl=row["current_pl"],
        ),
        axis=1,
    )
    finite_magnitudes = pd.concat(
        [
            extremes_frame.loc[~extremes_frame["unbounded_loss"], "max_loss"],
            extremes_frame.loc[~extremes_frame["unbounded_profit"], "max_profit"],
        ]
    )
    finite_cap = float(finite_magnitudes.max()) if not finite_magnitudes.empty else 0.0
    unbounded_bar_value = max(finite_cap * 1.15, 1.0)
    extremes_frame["display_loss"] = -np.where(
        extremes_frame["unbounded_loss"],
        unbounded_bar_value,
        extremes_frame["max_loss"],
    )
    extremes_frame["display_profit"] = np.where(
        extremes_frame["unbounded_profit"],
        unbounded_bar_value,
        extremes_frame["max_profit"],
    )

    def _risk_reward_value(row: pd.Series) -> float:
        if row["unbounded_profit"] and row["unbounded_loss"]:
            return np.nan

        if row["unbounded_profit"]:
            return np.inf

        if row["unbounded_loss"]:
            return 0.0

        max_profit = row["max_profit"]
        max_loss = row["max_loss"]

        if pd.isna(max_profit) or pd.isna(max_loss):
            return np.nan

        if max_loss <= 0:
            return np.inf if max_profit > 0 else np.nan

        return max_profit / max_loss

    def _format_risk_reward(value: float) -> str:
        if pd.isna(value):
            return "N/A"

        if np.isinf(value):
            return "Unbounded"

        return f"{value:,.2f}x"

    extremes_frame["risk_reward"] = extremes_frame.apply(_risk_reward_value, axis=1)
    finite_risk_reward_values = (
        extremes_frame["risk_reward"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    finite_risk_reward_cap = (
        float(finite_risk_reward_values.max())
        if not finite_risk_reward_values.empty
        else 0.0
    )
    unbounded_risk_reward_value = max(finite_risk_reward_cap * 1.15, 1.0)
    extremes_frame["display_risk_reward"] = np.where(
        np.isposinf(extremes_frame["risk_reward"]),
        unbounded_risk_reward_value,
        extremes_frame["risk_reward"],
    )
    extremes_frame["sort_magnitude"] = np.maximum(
        extremes_frame["display_loss"].abs(),
        extremes_frame["display_profit"].abs(),
    )
    extremes_frame = extremes_frame.sort_values("sort_magnitude", ascending=False)

    def _chart_payload(chart_frame: pd.DataFrame) -> dict:
        x_values = chart_frame["underlying"].tolist()
        return {
            "x": x_values,
            "display_profit": chart_frame["display_profit"].tolist(),
            "display_loss": chart_frame["display_loss"].tolist(),
            "display_risk_reward": chart_frame["display_risk_reward"].tolist(),
            "profit_text": [
                "Unbounded" if row.unbounded_profit else f"${row.max_profit:,.0f}"
                for row in chart_frame.itertuples()
            ],
            "loss_text": [
                "Unbounded" if row.unbounded_loss else f"-${row.max_loss:,.0f}"
                for row in chart_frame.itertuples()
            ],
            "risk_reward_text": [
                _format_risk_reward(row.risk_reward)
                for row in chart_frame.itertuples()
            ],
            "profit_hover": [
                (
                    f"{row.underlying}<br>"
                    f"Direction: {row.direction}<br>"
                    f"Max profit: {'Unbounded' if row.unbounded_profit else _format_currency_or_na(row.max_profit)}<br>"
                    f"Best finite P/L: {_format_currency_or_na(row.best_finite_pl)}<br>"
                    f"Current price: {_format_number_or_na(row.current_price)}<br>"
                    f"Current option P/L: {_format_currency_or_na(row.current_pl)}"
                )
                for row in chart_frame.itertuples()
            ],
            "loss_hover": [
                (
                    f"{row.underlying}<br>"
                    f"Direction: {row.direction}<br>"
                    f"Max loss: {'Unbounded' if row.unbounded_loss else _format_currency_or_na(row.max_loss)}<br>"
                    f"Worst finite P/L: {_format_currency_or_na(row.worst_finite_pl)}<br>"
                    f"Current price: {_format_number_or_na(row.current_price)}<br>"
                    f"Current option P/L: {_format_currency_or_na(row.current_pl)}"
                )
                for row in chart_frame.itertuples()
            ],
            "risk_reward_hover": [
                (
                    f"{row.underlying}<br>"
                    f"Direction: {row.direction}<br>"
                    f"Risk Reward: {_format_risk_reward(row.risk_reward)}<br>"
                    f"Max profit: {'Unbounded' if row.unbounded_profit else _format_currency_or_na(row.max_profit)}<br>"
                    f"Max loss: {'Unbounded' if row.unbounded_loss else _format_currency_or_na(row.max_loss)}"
                )
                for row in chart_frame.itertuples()
            ],
            "profit_color": np.where(chart_frame["unbounded_profit"], "#22C55E", "#16A34A").tolist(),
            "loss_color": np.where(chart_frame["unbounded_loss"], "#EF4444", "#F97316").tolist(),
            "risk_reward_color": [
                "#22C55E" if np.isposinf(row.risk_reward)
                else "#38BDF8" if pd.notna(row.risk_reward)
                else "rgba(148, 163, 184, 0.45)"
                for row in chart_frame.itertuples()
            ],
            "ticktext": [
                _format_directional_axis_ticker(row.underlying, row.direction)
                for row in chart_frame.itertuples()
            ],
        }

    def _chart_shapes(chart_frame: pd.DataFrame) -> list[dict]:
        shape_fig = go.Figure()
        shape_fig.add_hline(
            y=0,
            line_width=1.5,
            line_dash="dash",
            line_color=ZERO_LINE_COLOR,
        )
        finite_profit_values = chart_frame.loc[~chart_frame["unbounded_profit"], "max_profit"].replace([np.inf, -np.inf], np.nan).dropna()
        finite_loss_values = chart_frame.loc[~chart_frame["unbounded_loss"], "max_loss"].replace([np.inf, -np.inf], np.nan).dropna()

        if not finite_profit_values.empty:
            avg_max_profit = float(finite_profit_values.mean())
            median_max_profit = float(finite_profit_values.median())
            _add_horizontal_guide_shape(
                shape_fig,
                y_value=avg_max_profit,
                legendgroup=AVG_MAX_PROFIT_LEGEND_GROUP,
                legend_name="Avg Max Profit",
                label_text=f"Avg Max Profit: ${avg_max_profit:,.0f}",
                line_color=AVG_MAX_PROFIT_LINE_COLOR,
            )
            _add_horizontal_guide_shape(
                shape_fig,
                y_value=median_max_profit,
                legendgroup=MEDIAN_MAX_PROFIT_LEGEND_GROUP,
                legend_name="Median Max Profit",
                label_text=f"Median Max Profit: ${median_max_profit:,.0f}",
                line_color=MEDIAN_MAX_PROFIT_LINE_COLOR,
                line_dash="dot",
            )

        if not finite_loss_values.empty:
            avg_max_loss = float(finite_loss_values.mean())
            median_max_loss = float(finite_loss_values.median())
            _add_horizontal_guide_shape(
                shape_fig,
                y_value=-avg_max_loss,
                legendgroup=AVG_MAX_LOSS_LEGEND_GROUP,
                legend_name="Avg Max Loss",
                label_text=f"Avg Max Loss: -${avg_max_loss:,.0f}",
                line_color=AVG_MAX_LOSS_LINE_COLOR,
            )
            _add_horizontal_guide_shape(
                shape_fig,
                y_value=-median_max_loss,
                legendgroup=MEDIAN_MAX_LOSS_LEGEND_GROUP,
                legend_name="Median Max Loss",
                label_text=f"Median Max Loss: -${median_max_loss:,.0f}",
                line_color=MEDIAN_MAX_LOSS_LINE_COLOR,
                line_dash="dot",
            )

        return [shape.to_plotly_json() for shape in shape_fig.layout.shapes]

    def _risk_reward_guide_shapes() -> list[dict]:
        guide_styles = [
            (1, "#FBBF24", "dash"),
            (2, "#22C55E", "dot"),
        ]
        return [
            {
                "type": "line",
                "x0": 0,
                "x1": 1,
                "xref": "x2 domain",
                "y0": level,
                "y1": level,
                "yref": "y2",
                "line": {"color": line_color, "dash": line_dash, "width": 1.6},
                "label": {"text": f"{level}x", "font": {"color": line_color, "size": 11}},
            }
            for level, line_color, line_dash in guide_styles
        ]

    def _combined_chart_shapes(chart_frame: pd.DataFrame) -> list[dict]:
        return [*_chart_shapes(chart_frame), *_risk_reward_guide_shapes()]

    def _xaxis_layout(payload: dict, *, domain: list[float] | None = None, anchor: str | None = None) -> dict:
        layout = {
            "title": {"text": "Ticker"},
            "gridcolor": DARK_GRID_COLOR,
            "zerolinecolor": ZERO_LINE_COLOR,
            "tickmode": "array",
            "tickvals": payload["x"],
            "ticktext": payload["ticktext"],
            "categoryorder": "array",
            "categoryarray": payload["x"],
        }

        if domain is not None:
            layout["domain"] = domain

        if anchor is not None:
            layout["anchor"] = anchor

        return layout

    yaxis_layout = {
        "title": {"text": "P/L ($)"},
        "gridcolor": DARK_GRID_COLOR,
        "zerolinecolor": ZERO_LINE_COLOR,
        "autorange": True,
    }

    def _risk_reward_yaxis_layout(payload: dict) -> dict:
        risk_reward_values = (
            pd.Series(payload["display_risk_reward"], dtype="float64")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        max_risk_reward = float(risk_reward_values.max()) if not risk_reward_values.empty else 0.0
        y_max = max(max_risk_reward, 2.0) * 1.18
        return {
            "title": {"text": "Risk Reward"},
            "gridcolor": DARK_GRID_COLOR,
            "zerolinecolor": ZERO_LINE_COLOR,
            "rangemode": "tozero",
            "range": [0, y_max],
        }

    initial_payload = _chart_payload(extremes_frame)
    initial_risk_reward_yaxis_layout = _risk_reward_yaxis_layout(initial_payload)

    def _total_extreme_labels(chart_frame: pd.DataFrame) -> tuple[str, str]:
        if chart_frame.empty:
            return "N/A", "N/A"

        if chart_frame["unbounded_profit"].any():
            profit_label = "Unbounded"
        else:
            profit_label = f"${chart_frame['max_profit'].replace([np.inf, -np.inf], np.nan).dropna().sum():,.0f}"

        if chart_frame["unbounded_loss"].any():
            loss_label = "Unbounded"
        else:
            loss_label = f"-${chart_frame['max_loss'].replace([np.inf, -np.inf], np.nan).dropna().sum():,.0f}"

        return profit_label, loss_label

    def _total_summary_row(label: str, chart_frame: pd.DataFrame, color: str) -> str:
        profit_label, loss_label = _total_extreme_labels(chart_frame)
        return (
            f"<span style='color:{color}'><b>{label}</b></span>: "
            f"P {profit_label} | L {loss_label}"
        )

    totals_summary_text = (
        "<b>Ticker-Sum Totals</b><br>"
        + _total_summary_row("Bullish", extremes_frame[extremes_frame["direction"] == "up"], "#86EFAC")
        + "<br>"
        + _total_summary_row("Bearish", extremes_frame[extremes_frame["direction"] == "down"], "#FCA5A5")
        + "<br>"
        + _total_summary_row("Net", extremes_frame, DARK_FONT_COLOR)
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        subplot_titles=("Max Profit and Max Loss", "Risk Reward"),
        vertical_spacing=0.13,
    )
    subplot_axis_domains = {
        "xaxis": list(fig.layout.xaxis.domain),
        "xaxis2": list(fig.layout.xaxis2.domain),
        "yaxis": list(fig.layout.yaxis.domain),
        "yaxis2": list(fig.layout.yaxis2.domain),
    }
    filter_payloads = []

    for filter_label, direction_value in [
        ("All Directions", None),
        ("Directionally Bullish", "up"),
        ("Directionally Bearish", "down"),
    ]:
        filtered_frame = (
            extremes_frame
            if direction_value is None
            else extremes_frame[extremes_frame["direction"] == direction_value]
        )
        filter_payloads.append((filter_label, filtered_frame, _chart_payload(filtered_frame)))

    for filter_index, (_, _, payload) in enumerate(filter_payloads):
        trace_visible = filter_index == 0
        fig.add_trace(
            go.Bar(
                x=payload["x"],
                y=payload["display_profit"],
                name="Max Profit",
                text=payload["profit_text"],
                textposition="outside",
                hovertext=payload["profit_hover"],
                hoverinfo="text",
                marker=dict(
                    color=payload["profit_color"],
                    line=dict(color="rgba(255, 255, 255, 0.35)", width=1),
                ),
                visible=trace_visible,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=payload["x"],
                y=payload["display_loss"],
                name="Max Loss",
                text=payload["loss_text"],
                textposition="outside",
                hovertext=payload["loss_hover"],
                hoverinfo="text",
                marker=dict(
                    color=payload["loss_color"],
                    line=dict(color="rgba(255, 255, 255, 0.35)", width=1),
                ),
                visible=trace_visible,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=payload["x"],
                y=payload["display_risk_reward"],
                name="Risk Reward",
                text=payload["risk_reward_text"],
                textposition="outside",
                hovertext=payload["risk_reward_hover"],
                hoverinfo="text",
                marker=dict(
                    color=payload["risk_reward_color"],
                    line=dict(color="rgba(255, 255, 255, 0.35)", width=1),
                ),
                visible=trace_visible,
            ),
            row=2,
            col=1,
        )
    fig.update_layout(shapes=_combined_chart_shapes(extremes_frame))

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.18,
        text=totals_summary_text,
        showarrow=False,
        align="left",
        font=dict(color=DARK_FONT_COLOR, size=11),
        bgcolor="rgba(15, 23, 42, 0.82)",
        bordercolor="rgba(148, 163, 184, 0.35)",
        borderwidth=1,
        borderpad=6,
        xanchor="right",
        yanchor="top",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=1.08,
        text="Unbounded bars are scaled for display",
        showarrow=False,
        font=dict(color="rgba(229, 231, 235, 0.78)", size=11),
        xanchor="right",
    )
    filter_buttons = []

    for filter_index, (filter_label, filtered_frame, payload) in enumerate(filter_payloads):
        visible = [False] * len(fig.data)
        visible[filter_index * 3] = True
        visible[filter_index * 3 + 1] = True
        visible[filter_index * 3 + 2] = True
        filter_buttons.append(
            {
                "label": filter_label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "xaxis": _xaxis_layout(
                            payload,
                            domain=subplot_axis_domains["xaxis"],
                            anchor="y",
                        ),
                        "xaxis2": _xaxis_layout(
                            payload,
                            domain=subplot_axis_domains["xaxis2"],
                            anchor="y2",
                        ),
                        "yaxis": {
                            **yaxis_layout,
                            "domain": subplot_axis_domains["yaxis"],
                            "anchor": "x",
                        },
                        "yaxis2": {
                            **_risk_reward_yaxis_layout(payload),
                            "domain": subplot_axis_domains["yaxis2"],
                            "anchor": "x2",
                        },
                        "shapes": _combined_chart_shapes(filtered_frame),
                    },
                ],
            }
        )

    fig.update_layout(
        title="Option Max Profit and Max Loss by Ticker<br><sup>Losses plot below zero; unbounded reflects net call exposure</sup>",
        xaxis_title="Ticker",
        yaxis_title="P/L ($)",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_COLOR,
        plot_bgcolor=DARK_PLOT_COLOR,
        font={"color": DARK_FONT_COLOR},
        height=1180,
        margin=dict(t=170, b=115),
        barmode="relative",
        updatemenus=[
            {
                "buttons": filter_buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "bgcolor": "#1F2937",
                "bordercolor": "#475569",
                "font": {"color": DARK_FONT_COLOR},
                "x": 0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            font={"color": DARK_FONT_COLOR},
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0,
            groupclick="togglegroup",
        )
    )
    fig.update_xaxes(
        **_xaxis_layout(initial_payload),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        **_xaxis_layout(initial_payload),
        row=2,
        col=1,
    )
    fig.update_yaxes(**yaxis_layout, row=1, col=1)
    fig.update_yaxes(**initial_risk_reward_yaxis_layout, row=2, col=1)
    return fig


def _build_portfolio_mode_figure(
    positions_df: pd.DataFrame,
    selected_strike_range: str | int | float,
    *,
    net_cost_basis: pd.Series,
    portfolio_latest_prices: Mapping[str, float],
    benchmark_current_price: float,
    underlying_beta_map: Mapping[str, float],
    benchmark_label: str,
    use_beta_weighting: bool,
    use_beta_price_axis: bool,
) -> go.Figure | None:
    scoped_legs = positions_df.copy()
    reference_price_map = {}
    selected_frames = []
    total_unique_strikes = 0
    selected_strike_count = 0

    for underlying_name, group in scoped_legs.groupby("underlying"):
        reference_price = _get_reference_price(underlying_name, group, portfolio_latest_prices)

        if pd.isna(reference_price):
            continue

        reference_price_map[underlying_name] = reference_price
        selected_strikes, _, _, _ = _select_strikes_for_frame(group, reference_price, selected_strike_range)
        total_unique_strikes += group["strike"].dropna().nunique()
        selected_strike_count += len(selected_strikes)
        selected_frames.append(group[group["strike"].isin(selected_strikes)].copy())

    if not selected_frames:
        print("No portfolio option legs matched the selected strike filter.")
        return None

    fig = go.Figure()
    option_legs = pd.concat(selected_frames, ignore_index=True)
    scenario_trigger_levels = []

    for underlying_name, group in option_legs.groupby("underlying"):
        reference_price = reference_price_map.get(underlying_name, np.nan)

        if not pd.notna(reference_price) or reference_price <= 0:
            continue

        relative_moves = group["strike"].dropna() / reference_price - 1

        if use_beta_weighting:
            beta_value = _get_underlying_beta(underlying_name, underlying_beta_map)

            if abs(beta_value) > 1e-6:
                scenario_trigger_levels.extend((relative_moves / beta_value).tolist())
        else:
            scenario_trigger_levels.extend(relative_moves.tolist())

    if scenario_trigger_levels:
        scenario_min = min(min(scenario_trigger_levels), -0.05)
        scenario_max = max(max(scenario_trigger_levels), 0.05)
    else:
        scenario_min, scenario_max = -0.25, 0.25

    pad = max((scenario_max - scenario_min) * 0.25, 0.05)
    scenario_move_range = np.linspace(
        max(scenario_min - pad, -0.95),
        min(scenario_max + pad, 0.95),
        500,
    )
    benchmark_price_range = np.maximum(benchmark_current_price * (1 + scenario_move_range), 0)
    scenario_x_values = benchmark_price_range if use_beta_price_axis else scenario_move_range * 100
    total_pl = np.zeros(len(scenario_move_range))

    for _, leg in option_legs.iterrows():
        underlying_name = leg["underlying"]
        reference_price = reference_price_map.get(underlying_name, np.nan)

        if pd.isna(reference_price):
            continue

        strike = leg["strike"]
        opt_type = leg["option_type"]
        qty = leg["net_quantity"]
        avg_px = leg["average_price"]
        exp_label = _expiration_label(leg)
        beta_value = _get_underlying_beta(underlying_name, underlying_beta_map)

        if use_beta_weighting:
            scenario_prices = np.maximum(reference_price * (1 + beta_value * scenario_move_range), 0)
            leg_label_prefix = f"{underlying_name} (beta {beta_value:.2f})"
        else:
            scenario_prices = np.maximum(reference_price * (1 + scenario_move_range), 0)
            leg_label_prefix = underlying_name

        intrinsic = np.maximum(scenario_prices - strike, 0) if opt_type == "C" else np.maximum(strike - scenario_prices, 0)
        leg_pl = qty * (intrinsic - avg_px) * 100
        total_pl += leg_pl

        direction = "Long" if qty > 0 else "Short"
        _add_leg_trace(
            fig,
            x_values=scenario_x_values,
            leg_pl=leg_pl,
            trace_name=f"{leg_label_prefix} | {direction} {abs(qty):.0f}x {strike:.0f}{opt_type} exp {exp_label}",
        )

    if use_beta_price_axis:
        portfolio_trace_name = f"Total Portfolio P/L (Beta Weighted vs {benchmark_label} Price)"
        title_text = f"Entire Portfolio Options - Beta-Weighted P/L vs {benchmark_label} Price at Expiration"
        xaxis_title = f"{benchmark_label} Price at Expiration"
        current_marker_x = benchmark_current_price
        current_marker_text = f"Current {benchmark_label}: {benchmark_current_price:.2f}"
    elif use_beta_weighting:
        portfolio_trace_name = f"Total Portfolio P/L (Beta Weighted vs {benchmark_label} %)"
        title_text = f"Entire Portfolio Options - Beta-Weighted P/L vs {benchmark_label} Move % at Expiration"
        xaxis_title = f"{benchmark_label} Move at Expiration (%)"
        current_marker_x = 0
        current_marker_text = f"Current {benchmark_label}: 0%"
    else:
        portfolio_trace_name = "Total Portfolio P/L"
        title_text = "Entire Portfolio Options - Equal-Move P/L at Expiration"
        xaxis_title = "Underlying Move at Expiration (%)"
        current_marker_x = 0
        current_marker_text = "Current prices"

    _add_total_pl_trace(
        fig,
        x_values=scenario_x_values,
        total_pl=total_pl,
        trace_name=portfolio_trace_name,
    )
    portfolio_breakeven_formatter = (
        (lambda value: f"{value:.2f}")
        if use_beta_price_axis
        else (lambda value: f"{value:.2f}%")
    )
    guide_x_values = _add_breakeven_guides(
        fig,
        x_values=scenario_x_values,
        total_pl=total_pl,
        value_formatter=portfolio_breakeven_formatter,
    )
    half_profit_x_values = _add_half_max_profit_guides(
        fig,
        x_values=scenario_x_values,
        total_pl=total_pl,
        value_formatter=portfolio_breakeven_formatter,
        skip_x_values=guide_x_values,
    )
    _add_max_profit_guide(
        fig,
        x_values=scenario_x_values,
        total_pl=total_pl,
        value_formatter=portfolio_breakeven_formatter,
        skip_x_values=[*guide_x_values, *half_profit_x_values],
    )
    _add_zero_and_current_guides(fig, current_marker_x=current_marker_x, current_marker_text=current_marker_text)

    displayed_cost = option_legs["net_cost_basis"].sum() if "net_cost_basis" in option_legs.columns else np.nan
    total_cost = float(net_cost_basis.sum()) if not net_cost_basis.empty else np.nan
    range_text = (
        "All strikes"
        if selected_strike_range == "All"
        else f"+/- {float(selected_strike_range):g} around each underlying current price"
    )

    if use_beta_price_axis:
        mode_text = f"Beta weighted to {benchmark_label} with {benchmark_label} price axis"
    elif use_beta_weighting:
        mode_text = f"Beta weighted to {benchmark_label} with {benchmark_label} move % axis"
    else:
        mode_text = "Equal % move across all underlyings"

    subtitle = (
        f"Mode: {mode_text}"
        f" | Range: {range_text}"
        f" | Underlyings shown: {option_legs['underlying'].nunique()} of {scoped_legs['underlying'].nunique()}"
        f" | Strike groups shown: {selected_strike_count} of {total_unique_strikes}"
        f" | Displayed cost basis: ${displayed_cost:,.2f}"
        f" | Total portfolio cost basis: ${total_cost:,.2f}"
    )
    fig.update_layout(
        title=f"{title_text}<br><sup>{subtitle}</sup>",
        xaxis_title=xaxis_title,
        yaxis_title="P/L ($)",
        template="plotly_dark",
        hovermode="x unified",
        height=700,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, groupclick="togglegroup"),
    )

    return fig


def _build_underlying_mode_figure(
    positions_df: pd.DataFrame,
    selected_underlying: str,
    selected_strike_range: str | int | float,
    *,
    net_cost_basis: pd.Series,
    portfolio_latest_prices: Mapping[str, float],
) -> go.Figure | None:
    scoped_legs = positions_df[positions_df["underlying"] == selected_underlying].copy()

    if scoped_legs.empty:
        print(f"No {selected_underlying} option positions found in positions_df.")
        return None

    reference_price = _get_reference_price(selected_underlying, scoped_legs, portfolio_latest_prices)
    current_px = _get_price_for_underlying(selected_underlying, portfolio_latest_prices)
    selected_strikes, lower_bound, upper_bound, range_label = _select_strikes_for_frame(
        scoped_legs,
        reference_price,
        selected_strike_range,
    )
    total_unique_strikes = scoped_legs["strike"].dropna().nunique()
    option_legs = scoped_legs[scoped_legs["strike"].isin(selected_strikes)].copy()

    if option_legs.empty:
        print(f"No {selected_underlying} option legs matched the selected strike filter.")
        return None

    fig = go.Figure()
    strike_min = option_legs["strike"].min()
    strike_max = option_legs["strike"].max()
    reference_min = min(strike_min, current_px) if pd.notna(current_px) else strike_min
    reference_max = max(strike_max, current_px) if pd.notna(current_px) else strike_max
    pad = max((reference_max - reference_min) * 0.25, reference_max * 0.10, 1.0)
    s_range = np.linspace(max(reference_min - pad, 0), reference_max + pad, 500)
    total_pl = np.zeros(len(s_range))

    for _, leg in option_legs.iterrows():
        strike = leg["strike"]
        opt_type = leg["option_type"]
        qty = leg["net_quantity"]
        avg_px = leg["average_price"]
        exp_label = _expiration_label(leg)

        intrinsic = np.maximum(s_range - strike, 0) if opt_type == "C" else np.maximum(strike - s_range, 0)
        leg_pl = qty * (intrinsic - avg_px) * 100
        total_pl += leg_pl

        direction = "Long" if qty > 0 else "Short"
        _add_leg_trace(
            fig,
            x_values=s_range,
            leg_pl=leg_pl,
            trace_name=f"{direction} {abs(qty):.0f}x {strike:.0f}{opt_type} exp {exp_label}",
        )

    _add_total_pl_trace(
        fig,
        x_values=s_range,
        total_pl=total_pl,
        trace_name="Total P/L",
    )
    guide_x_values = _add_breakeven_guides(
        fig,
        x_values=s_range,
        total_pl=total_pl,
        value_formatter=lambda value: f"{value:.2f}",
    )
    half_profit_x_values = _add_half_max_profit_guides(
        fig,
        x_values=s_range,
        total_pl=total_pl,
        value_formatter=lambda value: f"{value:.2f}",
        skip_x_values=guide_x_values,
    )
    _add_max_profit_guide(
        fig,
        x_values=s_range,
        total_pl=total_pl,
        value_formatter=lambda value: f"{value:.2f}",
        skip_x_values=[*guide_x_values, *half_profit_x_values],
    )
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color=ZERO_LINE_COLOR)

    if pd.notna(current_px):
        current_marker_x = current_px
        current_marker_text = f"Current {selected_underlying}: {current_px:.2f}"
    else:
        current_marker_x = reference_price
        current_marker_text = f"{selected_underlying} reference: {reference_price:.2f}"

    _add_current_price_guide(
        fig,
        current_marker_x=current_marker_x,
        current_marker_text=current_marker_text,
    )

    total_cost = net_cost_basis.get(selected_underlying, np.nan)
    displayed_cost = option_legs["net_cost_basis"].sum() if "net_cost_basis" in option_legs.columns else np.nan
    subtitle = (
        f"Range: {range_label} around {reference_price:.2f}"
        f" | Window: [{lower_bound:.2f}, {upper_bound:.2f}]"
        f" | Strikes shown: {len(selected_strikes)} of {total_unique_strikes}"
        f" | Displayed cost basis: ${displayed_cost:,.2f}"
        f" | Total ticker cost basis: ${total_cost:,.2f}"
    )
    fig.update_layout(
        title=f"{selected_underlying} Options - P/L at Expiration<br><sup>{subtitle}</sup>",
        xaxis_title=f"{selected_underlying} Price at Expiration",
        yaxis_title="P/L ($)",
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, groupclick="togglegroup"),
    )

    return fig


def build_option_expiration_pl_figure(
    positions_df: pd.DataFrame,
    selected_underlying: str,
    selected_strike_range: str | int | float,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
    benchmark_current_price: float = np.nan,
    underlying_beta_map: Mapping[str, float] | None = None,
    benchmark_label: str = "SPY",
) -> go.Figure | None:
    """Build a portfolio or single-underlying option P/L-at-expiration figure."""
    if positions_df.empty:
        print("No option positions found in positions_df.")
        return None

    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    portfolio_latest_prices = portfolio_latest_prices or {}
    underlying_beta_map = underlying_beta_map or {}
    use_portfolio_mode = selected_underlying in PORTFOLIO_MODES
    use_beta_weighting = selected_underlying in {PORTFOLIO_BETA_MOVE, PORTFOLIO_BETA_PRICE}
    use_beta_price_axis = selected_underlying == PORTFOLIO_BETA_PRICE

    if use_portfolio_mode:
        return _build_portfolio_mode_figure(
            positions_df,
            selected_strike_range,
            net_cost_basis=net_cost_basis,
            portfolio_latest_prices=portfolio_latest_prices,
            benchmark_current_price=benchmark_current_price,
            underlying_beta_map=underlying_beta_map,
            benchmark_label=benchmark_label,
            use_beta_weighting=use_beta_weighting,
            use_beta_price_axis=use_beta_price_axis,
        )

    return _build_underlying_mode_figure(
        positions_df,
        selected_underlying,
        selected_strike_range,
        net_cost_basis=net_cost_basis,
        portfolio_latest_prices=portfolio_latest_prices,
    )


def _visible_payoff_traces(fig: go.Figure) -> list[go.Scatter]:
    traces = []

    for trace in fig.data:
        if trace.visible == "legendonly":
            continue
        traces.append(go.Scatter(trace.to_plotly_json()))

    return traces


def _style_portfolio_panel_axes(fig: go.Figure, *, row: int, col: int, benchmark_label: str) -> None:
    fig.update_xaxes(
        title_text=f"{benchmark_label} Price at Expiration",
        gridcolor=DARK_GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text="P/L ($)",
        gridcolor=DARK_GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        row=row,
        col=col,
    )


def _add_portfolio_panel_guides(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    benchmark_current_price: float,
    benchmark_label: str,
) -> None:
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color=ZERO_LINE_COLOR, row=row, col=col)

    if pd.notna(benchmark_current_price):
        fig.add_vline(
            x=benchmark_current_price,
            line_width=1.5,
            line_dash="dash",
            line_color=CURRENT_LINE_COLOR,
            annotation_text=f"Current {benchmark_label}: {benchmark_current_price:.2f}",
            annotation_position="top right",
            row=row,
            col=col,
        )


def _filter_positions_by_dte(positions_df: pd.DataFrame, lower_bound: int | None, upper_bound: int | None) -> pd.DataFrame:
    if "days_to_expiration" not in positions_df.columns:
        return positions_df.iloc[0:0].copy()

    dte = pd.to_numeric(positions_df["days_to_expiration"], errors="coerce")
    mask = dte.notna()

    if lower_bound is not None:
        mask &= dte >= lower_bound

    if upper_bound is not None:
        mask &= dte <= upper_bound

    return positions_df.loc[mask].copy()


def _dte_panel_configs(full_panel_title: str, positions_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, int, int]]:
    return [
        (full_panel_title, positions_df, 1, 1),
        ("< 50 DTE", _filter_positions_by_dte(positions_df, None, 49), 2, 1),
        ("50-200 DTE", _filter_positions_by_dte(positions_df, 50, 200), 2, 2),
        (">= 200 DTE", _filter_positions_by_dte(positions_df, 200, None), 2, 3),
    ]


def build_portfolio_beta_price_dte_grid_figure(
    positions_df: pd.DataFrame,
    selected_strike_range: str | int | float,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
    benchmark_current_price: float = np.nan,
    underlying_beta_map: Mapping[str, float] | None = None,
    benchmark_label: str = "SPY",
) -> go.Figure | None:
    """Build the beta-price portfolio P/L figure plus DTE-bucket panels."""
    if positions_df.empty:
        print("No option positions found in positions_df.")
        return None

    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    portfolio_latest_prices = portfolio_latest_prices or {}
    underlying_beta_map = underlying_beta_map or {}
    panels = _dte_panel_configs("Full Portfolio", positions_df)
    panel_figures = []

    for panel_title, panel_positions, row, col in panels:
        if panel_positions.empty:
            panel_figures.append((panel_title, None, row, col))
            continue

        panel_figures.append(
            (
                panel_title,
                build_option_expiration_pl_figure(
                    panel_positions,
                    PORTFOLIO_BETA_PRICE,
                    selected_strike_range,
                    net_cost_basis=net_cost_basis,
                    portfolio_latest_prices=portfolio_latest_prices,
                    benchmark_current_price=benchmark_current_price,
                    underlying_beta_map=underlying_beta_map,
                    benchmark_label=benchmark_label,
                ),
                row,
                col,
            )
        )

    if panel_figures[0][1] is None:
        return None

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        subplot_titles=[panel_title for panel_title, _, _, _ in panel_figures],
        vertical_spacing=0.16,
        horizontal_spacing=0.07,
    )

    for panel_title, panel_figure, row, col in panel_figures:
        if panel_figure is None:
            fig.add_annotation(
                text="No positions",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="rgba(229, 231, 235, 0.72)", size=13),
                row=row,
                col=col,
            )
            _style_portfolio_panel_axes(fig, row=row, col=col, benchmark_label=benchmark_label)
            continue

        for trace in _visible_payoff_traces(panel_figure):
            trace.showlegend = row == 1
            fig.add_trace(trace, row=row, col=col)

        _add_panel_payoff_guides(
            fig,
            panel_figure,
            row=row,
            col=col,
            value_formatter=lambda value: f"{value:.2f}",
        )
        _add_portfolio_panel_guides(
            fig,
            row=row,
            col=col,
            benchmark_current_price=benchmark_current_price,
            benchmark_label=benchmark_label,
        )
        _style_portfolio_panel_axes(fig, row=row, col=col, benchmark_label=benchmark_label)

    fig.update_layout(
        title=f"Entire Portfolio Options - Beta-Weighted P/L vs {benchmark_label} Price at Expiration",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_COLOR,
        plot_bgcolor=DARK_PLOT_COLOR,
        font={"color": DARK_FONT_COLOR},
        hovermode="x unified",
        height=1050,
        margin=dict(t=120, b=90),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            font={"color": DARK_FONT_COLOR},
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="left",
            x=0,
            groupclick="togglegroup",
        ),
    )
    return fig


def _style_underlying_panel_axes(fig: go.Figure, *, row: int, col: int, selected_underlying: str) -> None:
    fig.update_xaxes(
        title_text=f"{selected_underlying} Price at Expiration",
        gridcolor=DARK_GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text="P/L ($)",
        gridcolor=DARK_GRID_COLOR,
        zerolinecolor=ZERO_LINE_COLOR,
        row=row,
        col=col,
    )


def _underlying_current_marker(
    positions_df: pd.DataFrame,
    selected_underlying: str,
    portfolio_latest_prices: Mapping[str, float],
) -> tuple[float, str]:
    current_px = _get_price_for_underlying(selected_underlying, portfolio_latest_prices)

    if pd.notna(current_px):
        return float(current_px), f"Current {selected_underlying}: {current_px:.2f}"

    scoped_legs = positions_df[positions_df["underlying"] == selected_underlying].copy()

    if scoped_legs.empty:
        return np.nan, ""

    reference_price = _get_reference_price(selected_underlying, scoped_legs, portfolio_latest_prices)

    if pd.notna(reference_price):
        return float(reference_price), f"{selected_underlying} reference: {reference_price:.2f}"

    return np.nan, ""


def _add_underlying_panel_guides(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    current_marker_x: float,
    current_marker_text: str,
) -> None:
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color=ZERO_LINE_COLOR, row=row, col=col)

    if pd.notna(current_marker_x):
        fig.add_vline(
            x=current_marker_x,
            line_width=1.5,
            line_dash="dash",
            line_color=CURRENT_LINE_COLOR,
            annotation_text=current_marker_text,
            annotation_position="top right",
            row=row,
            col=col,
        )


def build_underlying_option_expiration_pl_dte_grid_figure(
    positions_df: pd.DataFrame,
    selected_underlying: str,
    selected_strike_range: str | int | float,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
) -> go.Figure | None:
    """Build one asset's option P/L figure plus DTE-bucket panels."""
    if positions_df.empty:
        print("No option positions found in positions_df.")
        return None

    scoped_positions = positions_df[positions_df["underlying"] == selected_underlying].copy()

    if scoped_positions.empty:
        print(f"No {selected_underlying} option positions found in positions_df.")
        return None

    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    portfolio_latest_prices = portfolio_latest_prices or {}
    panels = _dte_panel_configs(f"{selected_underlying} Full Position", scoped_positions)
    panel_figures = []

    for panel_title, panel_positions, row, col in panels:
        if panel_positions.empty:
            panel_figures.append((panel_title, None, row, col))
            continue

        panel_figures.append(
            (
                panel_title,
                build_option_expiration_pl_figure(
                    panel_positions,
                    selected_underlying,
                    selected_strike_range,
                    net_cost_basis=net_cost_basis,
                    portfolio_latest_prices=portfolio_latest_prices,
                ),
                row,
                col,
            )
        )

    if panel_figures[0][1] is None:
        return None

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        subplot_titles=[panel_title for panel_title, _, _, _ in panel_figures],
        vertical_spacing=0.16,
        horizontal_spacing=0.07,
    )
    current_marker_x, current_marker_text = _underlying_current_marker(
        scoped_positions,
        selected_underlying,
        portfolio_latest_prices,
    )

    for panel_title, panel_figure, row, col in panel_figures:
        if panel_figure is None:
            fig.add_annotation(
                text="No positions",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(color="rgba(229, 231, 235, 0.72)", size=13),
                row=row,
                col=col,
            )
            _style_underlying_panel_axes(fig, row=row, col=col, selected_underlying=selected_underlying)
            continue

        for trace in _visible_payoff_traces(panel_figure):
            trace.showlegend = row == 1
            fig.add_trace(trace, row=row, col=col)

        _add_panel_payoff_guides(
            fig,
            panel_figure,
            row=row,
            col=col,
            value_formatter=lambda value: f"{value:.2f}",
        )
        _add_underlying_panel_guides(
            fig,
            row=row,
            col=col,
            current_marker_x=current_marker_x,
            current_marker_text=current_marker_text,
        )
        _style_underlying_panel_axes(fig, row=row, col=col, selected_underlying=selected_underlying)

    fig.update_layout(
        title=f"{selected_underlying} Options - P/L at Expiration by DTE Bucket",
        template="plotly_dark",
        paper_bgcolor=DARK_PAPER_COLOR,
        plot_bgcolor=DARK_PLOT_COLOR,
        font={"color": DARK_FONT_COLOR},
        hovermode="x unified",
        height=1050,
        margin=dict(t=120, b=90),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            font={"color": DARK_FONT_COLOR},
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="left",
            x=0,
            groupclick="togglegroup",
        ),
    )
    return fig


def _trace_visible_value(trace) -> bool | str:
    return True if trace.visible is None else trace.visible


def _selector_layout_update(fig: go.Figure) -> dict:
    axis_updates = {}

    for layout_key, layout_value in fig.layout.to_plotly_json().items():
        if not (layout_key.startswith("xaxis") or layout_key.startswith("yaxis")):
            continue

        axis_layout = layout_value.copy()
        axis_layout["autorange"] = True
        axis_layout["gridcolor"] = DARK_GRID_COLOR
        axis_layout["zerolinecolor"] = ZERO_LINE_COLOR
        axis_updates[layout_key] = axis_layout

    return {
        "title": fig.layout.title.to_plotly_json(),
        "shapes": [shape.to_plotly_json() for shape in fig.layout.shapes],
        "annotations": [annotation.to_plotly_json() for annotation in fig.layout.annotations],
        "height": fig.layout.height,
        "hovermode": fig.layout.hovermode,
        "template": "plotly_dark",
        "paper_bgcolor": DARK_PAPER_COLOR,
        "plot_bgcolor": DARK_PLOT_COLOR,
        "font": {"color": DARK_FONT_COLOR},
        **axis_updates,
    }


def _build_dropdown_selector_figure(
    selector_figures: list[tuple[str, str, go.Figure]],
    default_value: str,
    *,
    legend_y: float = -0.18,
    margin: dict | None = None,
) -> go.Figure:
    selector_values = {option_value for _, option_value, _ in selector_figures}

    if default_value not in selector_values:
        default_value = selector_figures[0][1]

    dropdown_values = [value for _, value, _ in selector_figures]
    active_idx = dropdown_values.index(default_value)
    combined_fig = go.Figure()
    trace_groups = {}

    for option_label, option_value, option_fig in selector_figures:
        start_idx = len(combined_fig.data)
        trace_visibilities = []

        for trace in option_fig.data:
            trace_visibilities.append(_trace_visible_value(trace))
            combined_fig.add_trace(trace.to_plotly_json())
            combined_fig.data[-1].visible = (
                trace_visibilities[-1]
                if option_value == default_value
                else False
            )

        trace_groups[option_value] = {
            "label": option_label,
            "start": start_idx,
            "end": len(combined_fig.data),
            "visibilities": trace_visibilities,
            "layout": _selector_layout_update(option_fig),
        }

    total_trace_count = len(combined_fig.data)
    buttons = []

    for option_value in dropdown_values:
        trace_group = trace_groups[option_value]
        visible = [False] * total_trace_count

        for trace_idx, trace_visibility in zip(
            range(trace_group["start"], trace_group["end"]),
            trace_group["visibilities"],
        ):
            visible[trace_idx] = trace_visibility

        buttons.append(
            {
                "label": trace_group["label"],
                "method": "update",
                "args": [{"visible": visible}, trace_group["layout"]],
            }
        )

    default_layout = trace_groups[default_value]["layout"]
    combined_fig.update_layout(**default_layout)
    combined_fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": active_idx,
                "bgcolor": "#1F2937",
                "bordercolor": "#475569",
                "font": {"color": DARK_FONT_COLOR},
                "x": 0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        margin=margin or dict(t=120, b=110),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor="rgba(148, 163, 184, 0.35)",
            borderwidth=1,
            font={"color": DARK_FONT_COLOR},
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="left",
            x=0,
            traceorder="normal",
            itemwidth=30,
            groupclick="togglegroup",
        ),
    )

    return combined_fig


def build_asset_option_expiration_pl_dte_selector_figure(
    positions_df: pd.DataFrame,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
    benchmark_current_price: float = np.nan,
    underlying_beta_map: Mapping[str, float] | None = None,
    benchmark_label: str = "SPY",
    default_underlying: str | None = None,
    selected_strike_range: str | int | float = "All",
    include_portfolio_mode: bool = True,
) -> go.Figure | None:
    """Build a dropdown selector where assets and portfolio mode are split into DTE-bucket panels."""
    available_underlyings = _available_underlyings(positions_df)

    if not available_underlyings:
        print("No option positions found in positions_df.")
        return None

    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    portfolio_latest_prices = portfolio_latest_prices or {}
    underlying_beta_map = underlying_beta_map or {}
    dropdown_options = _asset_dropdown_options(available_underlyings)

    if include_portfolio_mode:
        dropdown_options = [
            (f"All assets (beta price vs {benchmark_label})", PORTFOLIO_BETA_PRICE),
            *dropdown_options,
        ]

    dropdown_values = {option_value for _, option_value in dropdown_options}
    default_underlying = default_underlying or (
        PORTFOLIO_BETA_PRICE
        if include_portfolio_mode
        else available_underlyings[0]
    )

    if default_underlying not in dropdown_values:
        default_underlying = PORTFOLIO_BETA_PRICE if include_portfolio_mode else available_underlyings[0]

    selector_figures = []

    for option_label, option_value in dropdown_options:
        if option_value == PORTFOLIO_BETA_PRICE:
            fig = build_portfolio_beta_price_dte_grid_figure(
                positions_df,
                selected_strike_range,
                net_cost_basis=net_cost_basis,
                portfolio_latest_prices=portfolio_latest_prices,
                benchmark_current_price=benchmark_current_price,
                underlying_beta_map=underlying_beta_map,
                benchmark_label=benchmark_label,
            )
        else:
            fig = build_underlying_option_expiration_pl_dte_grid_figure(
                positions_df,
                option_value,
                selected_strike_range,
                net_cost_basis=net_cost_basis,
                portfolio_latest_prices=portfolio_latest_prices,
            )

        if fig is not None:
            selector_figures.append((option_label, option_value, fig))

    if not selector_figures:
        print("No DTE option P/L selector figures could be built for the available positions.")
        return None

    return _build_dropdown_selector_figure(
        selector_figures,
        default_underlying,
        legend_y=-0.08,
        margin=dict(t=120, b=90),
    )


def build_option_expiration_pl_selector_figure(
    positions_df: pd.DataFrame,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
    benchmark_current_price: float = np.nan,
    underlying_beta_map: Mapping[str, float] | None = None,
    benchmark_label: str = "SPY",
    default_underlying: str | None = None,
    selected_strike_range: str | int | float = "All",
    include_portfolio_modes: bool = True,
    include_assets: bool = True,
) -> go.Figure | None:
    """Build one option P/L figure with a Plotly dropdown for portfolio modes and assets."""
    available_underlyings = _available_underlyings(positions_df)

    if not available_underlyings:
        print("No option positions found in positions_df.")
        return None

    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    portfolio_latest_prices = portfolio_latest_prices or {}
    underlying_beta_map = underlying_beta_map or {}
    dropdown_options = _underlying_dropdown_options(
        available_underlyings,
        benchmark_label,
        include_portfolio_modes=include_portfolio_modes,
        include_assets=include_assets,
    )

    if not dropdown_options:
        print("No option P/L selector choices are available.")
        return None

    dropdown_values = {option_value for _, option_value in dropdown_options}
    default_underlying = default_underlying or dropdown_options[0][1]

    if default_underlying not in dropdown_values:
        default_underlying = dropdown_options[0][1]

    selector_figures = []

    for option_label, option_value in dropdown_options:
        fig = build_option_expiration_pl_figure(
            positions_df,
            option_value,
            selected_strike_range,
            net_cost_basis=net_cost_basis,
            portfolio_latest_prices=portfolio_latest_prices,
            benchmark_current_price=benchmark_current_price,
            underlying_beta_map=underlying_beta_map,
            benchmark_label=benchmark_label,
        )

        if fig is not None:
            selector_figures.append((option_label, option_value, fig))

    if not selector_figures:
        print("No option P/L figures could be built for the available positions.")
        return None

    return _build_dropdown_selector_figure(selector_figures, default_underlying)


def display_option_expiration_pl_view(
    positions_df: pd.DataFrame,
    *,
    net_cost_basis: pd.Series | None = None,
    portfolio_latest_prices: Mapping[str, float] | None = None,
    benchmark_current_price: float = np.nan,
    underlying_beta_map: Mapping[str, float] | None = None,
    benchmark_label: str = "SPY",
    default_underlying: str | None = None,
    selected_strike_range: str | int | float = "All",
) -> tuple[go.Figure | None, go.Figure | None, go.Figure | None]:
    """Display the combined P/L selector and max-loss option figure."""
    pl_selector_fig = build_asset_option_expiration_pl_dte_selector_figure(
        positions_df,
        net_cost_basis=net_cost_basis,
        portfolio_latest_prices=portfolio_latest_prices,
        benchmark_current_price=benchmark_current_price,
        underlying_beta_map=underlying_beta_map,
        benchmark_label=benchmark_label,
        default_underlying=default_underlying,
        selected_strike_range=selected_strike_range,
    )
    max_loss_fig = build_option_max_loss_by_underlying_figure(
        positions_df,
        portfolio_latest_prices=portfolio_latest_prices,
    )

    if pl_selector_fig is not None:
        pl_selector_fig.show()

    if max_loss_fig is not None:
        max_loss_fig.show()

    return pl_selector_fig, None, max_loss_fig

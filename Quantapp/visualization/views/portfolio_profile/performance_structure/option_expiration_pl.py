"""Option expiration P/L views for portfolio performance notebooks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PORTFOLIO_EQUAL_MOVE = "ALL"
PORTFOLIO_BETA_MOVE = "ALL_BETA"
PORTFOLIO_BETA_PRICE = "ALL_BETA_PRICE"
PORTFOLIO_MODES = {PORTFOLIO_EQUAL_MOVE, PORTFOLIO_BETA_MOVE, PORTFOLIO_BETA_PRICE}


def _available_underlyings(positions_df: pd.DataFrame) -> list[str]:
    if positions_df.empty or "underlying" not in positions_df.columns:
        return []
    return sorted(positions_df["underlying"].dropna().unique().tolist())


def _default_underlying(underlyings: list[str]) -> str | None:
    if not underlyings:
        return None
    return "URA" if "URA" in underlyings else PORTFOLIO_EQUAL_MOVE


def _get_reference_price(
    underlying_name: str,
    legs_frame: pd.DataFrame,
    portfolio_latest_prices: Mapping[str, float],
) -> float:
    current_price = portfolio_latest_prices.get(underlying_name, np.nan)

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
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(
        x=current_marker_x,
        line_width=1,
        line_dash="dash",
        line_color="yellow",
        annotation_text=current_marker_text,
        annotation_position="top right",
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
            line=dict(width=1, dash="dot"),
            opacity=0.6,
            visible="legendonly",
        )
    )


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

    fig.add_trace(
        go.Scatter(
            x=scenario_x_values,
            y=total_pl,
            mode="lines",
            name=portfolio_trace_name,
            line=dict(width=3, color="white"),
        )
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    current_px = portfolio_latest_prices.get(selected_underlying, np.nan)
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

    fig.add_trace(
        go.Scatter(
            x=s_range,
            y=total_pl,
            mode="lines",
            name="Total P/L",
            line=dict(width=3, color="white"),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

    if pd.notna(current_px):
        fig.add_vline(
            x=current_px,
            line_width=1,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Current: {current_px:.2f}",
            annotation_position="top right",
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
) -> go.Figure | None:
    """Display one option expiration P/L figure without notebook widgets."""
    available_underlyings = _available_underlyings(positions_df)

    if not available_underlyings:
        print("No option positions found in positions_df.")
        return None

    portfolio_latest_prices = portfolio_latest_prices or {}
    underlying_beta_map = underlying_beta_map or {}
    net_cost_basis = net_cost_basis if net_cost_basis is not None else pd.Series(dtype=float, name="net_cost_basis")
    default_underlying = default_underlying or _default_underlying(available_underlyings)
    fig = build_option_expiration_pl_figure(
        positions_df,
        default_underlying,
        selected_strike_range,
        net_cost_basis=net_cost_basis,
        portfolio_latest_prices=portfolio_latest_prices,
        benchmark_current_price=benchmark_current_price,
        underlying_beta_map=underlying_beta_map,
        benchmark_label=benchmark_label,
    )

    if fig is not None:
        fig.show()

    return fig

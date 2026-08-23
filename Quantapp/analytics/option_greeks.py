"""Option Greek calculations for normalized portfolio option positions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


DEFAULT_CONTRACT_MULTIPLIER = 100.0
DEFAULT_ANNUALIZATION_DAYS = 365.25
DEFAULT_DTE_BUCKETS = (
    ("Total", None, None),
    ("< 50 DTE", None, 49),
    ("50-200 DTE", 50, 200),
    (">= 200 DTE", 200, None),
)


def _lookup_key(value) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def _option_type_key(value) -> str | None:
    normalized = str(value).strip().upper()
    if normalized in {"C", "CALL"}:
        return "C"
    if normalized in {"P", "PUT"}:
        return "P"
    return None


def _finite_positive(value) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric > 0


def _normalize_implied_volatility(value) -> float:
    try:
        volatility = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(volatility) or volatility <= 0:
        return np.nan
    # Most providers return IV as a decimal. Treat large values as percent inputs.
    if volatility > 5:
        volatility = volatility / 100.0
    return float(volatility)


def _coerce_numeric_map(values: Mapping | None, *, default: float | None = None) -> dict[str, float]:
    if values is None:
        return {}
    coerced = {}
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            coerced[_lookup_key(key)] = numeric
    if default is not None:
        coerced.setdefault("", float(default))
    return coerced


def _column_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index)


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    annual_rate: float,
    volatility: float,
    option_type: str,
    *,
    dividend_yield: float = 0.0,
) -> float:
    """Return Black-Scholes-Merton option price with continuous dividend yield."""
    option_key = _option_type_key(option_type)
    if option_key not in {"C", "P"}:
        return np.nan
    if not all(_finite_positive(value) for value in (spot, strike, time_years, volatility)):
        return np.nan

    spot = float(spot)
    strike = float(strike)
    time_years = float(time_years)
    annual_rate = float(annual_rate)
    volatility = float(volatility)
    dividend_yield = float(dividend_yield or 0.0)

    volatility_time = volatility * np.sqrt(time_years)
    d1 = (
        np.log(spot / strike)
        + (annual_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / volatility_time
    d2 = d1 - volatility_time
    discounted_spot = spot * np.exp(-dividend_yield * time_years)
    discounted_strike = strike * np.exp(-annual_rate * time_years)

    if option_key == "C":
        return float(discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1))


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_years: float,
    annual_rate: float,
    volatility: float,
    option_type: str,
    *,
    dividend_yield: float = 0.0,
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS,
) -> dict[str, float]:
    """Return per-share Black-Scholes-Merton Greeks.

    Vega is per one volatility point, rho is per one rate point, and theta is
    per calendar day.
    """
    option_key = _option_type_key(option_type)
    empty = {
        "theoretical_price": np.nan,
        "delta": np.nan,
        "gamma": np.nan,
        "theta_per_day": np.nan,
        "vega_per_vol_point": np.nan,
        "rho_per_rate_point": np.nan,
    }
    if option_key not in {"C", "P"}:
        return empty
    if not all(_finite_positive(value) for value in (spot, strike, time_years, volatility)):
        return empty

    spot = float(spot)
    strike = float(strike)
    time_years = float(time_years)
    annual_rate = float(annual_rate)
    volatility = float(volatility)
    dividend_yield = float(dividend_yield or 0.0)
    annualization_days = float(annualization_days)

    volatility_time = volatility * np.sqrt(time_years)
    d1 = (
        np.log(spot / strike)
        + (annual_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / volatility_time
    d2 = d1 - volatility_time
    discounted_spot_factor = np.exp(-dividend_yield * time_years)
    discounted_strike_factor = np.exp(-annual_rate * time_years)
    density_d1 = norm.pdf(d1)

    gamma = discounted_spot_factor * density_d1 / (spot * volatility_time)
    vega_per_vol_point = spot * discounted_spot_factor * density_d1 * np.sqrt(time_years) / 100.0

    theta_common = -(
        spot * discounted_spot_factor * density_d1 * volatility
    ) / (2.0 * np.sqrt(time_years))

    if option_key == "C":
        theoretical_price = black_scholes_price(
            spot,
            strike,
            time_years,
            annual_rate,
            volatility,
            option_key,
            dividend_yield=dividend_yield,
        )
        delta = discounted_spot_factor * norm.cdf(d1)
        theta_annual = (
            theta_common
            - annual_rate * strike * discounted_strike_factor * norm.cdf(d2)
            + dividend_yield * spot * discounted_spot_factor * norm.cdf(d1)
        )
        rho_per_rate_point = strike * time_years * discounted_strike_factor * norm.cdf(d2) / 100.0
    else:
        theoretical_price = black_scholes_price(
            spot,
            strike,
            time_years,
            annual_rate,
            volatility,
            option_key,
            dividend_yield=dividend_yield,
        )
        delta = discounted_spot_factor * (norm.cdf(d1) - 1.0)
        theta_annual = (
            theta_common
            + annual_rate * strike * discounted_strike_factor * norm.cdf(-d2)
            - dividend_yield * spot * discounted_spot_factor * norm.cdf(-d1)
        )
        rho_per_rate_point = -strike * time_years * discounted_strike_factor * norm.cdf(-d2) / 100.0

    return {
        "theoretical_price": float(theoretical_price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_per_day": float(theta_annual / annualization_days),
        "vega_per_vol_point": float(vega_per_vol_point),
        "rho_per_rate_point": float(rho_per_rate_point),
    }


def _black_scholes_greeks_arrays(
    spot,
    strike,
    time_years,
    annual_rate,
    volatility,
    option_type_keys,
    dividend_yield=0.0,
    *,
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS,
) -> dict[str, np.ndarray]:
    """Vectorized Black-Scholes-Merton Greeks for same-shaped arrays."""
    spot, strike, time_years, annual_rate, volatility, dividend_yield = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_years, dtype=float),
        np.asarray(annual_rate, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(dividend_yield, dtype=float),
    )
    option_type_keys = np.asarray(option_type_keys, dtype=object)
    if option_type_keys.shape != spot.shape:
        option_type_keys = np.broadcast_to(option_type_keys, spot.shape)

    result = {
        "theoretical_price": np.full(spot.shape, np.nan, dtype=float),
        "delta": np.full(spot.shape, np.nan, dtype=float),
        "gamma": np.full(spot.shape, np.nan, dtype=float),
        "theta_per_day": np.full(spot.shape, np.nan, dtype=float),
        "vega_per_vol_point": np.full(spot.shape, np.nan, dtype=float),
        "rho_per_rate_point": np.full(spot.shape, np.nan, dtype=float),
    }
    option_valid = np.isin(option_type_keys, ["C", "P"])
    numeric_valid = (
        np.isfinite(spot)
        & np.isfinite(strike)
        & np.isfinite(time_years)
        & np.isfinite(annual_rate)
        & np.isfinite(volatility)
        & np.isfinite(dividend_yield)
        & (spot > 0)
        & (strike > 0)
        & (time_years > 0)
        & (volatility > 0)
    )
    valid = option_valid & numeric_valid
    if not valid.any():
        return result

    valid_spot = spot[valid]
    valid_strike = strike[valid]
    valid_time = time_years[valid]
    valid_rate = annual_rate[valid]
    valid_volatility = volatility[valid]
    valid_dividend_yield = dividend_yield[valid]
    valid_option_type = option_type_keys[valid]

    sqrt_time = np.sqrt(valid_time)
    volatility_time = valid_volatility * sqrt_time
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (
            np.log(valid_spot / valid_strike)
            + (
                valid_rate
                - valid_dividend_yield
                + 0.5 * valid_volatility**2
            )
            * valid_time
        ) / volatility_time
        d2 = d1 - volatility_time
        discounted_spot_factor = np.exp(-valid_dividend_yield * valid_time)
        discounted_strike_factor = np.exp(-valid_rate * valid_time)
        density_d1 = norm.pdf(d1)

        price = np.empty_like(valid_spot, dtype=float)
        delta = np.empty_like(valid_spot, dtype=float)
        theta_annual = np.empty_like(valid_spot, dtype=float)
        rho_per_rate_point = np.empty_like(valid_spot, dtype=float)

        discounted_spot = valid_spot * discounted_spot_factor
        discounted_strike = valid_strike * discounted_strike_factor
        theta_common = -(
            valid_spot * discounted_spot_factor * density_d1 * valid_volatility
        ) / (2.0 * sqrt_time)

        call_mask = valid_option_type == "C"
        put_mask = ~call_mask
        price[call_mask] = (
            discounted_spot[call_mask] * norm.cdf(d1[call_mask])
            - discounted_strike[call_mask] * norm.cdf(d2[call_mask])
        )
        price[put_mask] = (
            discounted_strike[put_mask] * norm.cdf(-d2[put_mask])
            - discounted_spot[put_mask] * norm.cdf(-d1[put_mask])
        )
        delta[call_mask] = discounted_spot_factor[call_mask] * norm.cdf(d1[call_mask])
        delta[put_mask] = discounted_spot_factor[put_mask] * (
            norm.cdf(d1[put_mask]) - 1.0
        )
        theta_annual[call_mask] = (
            theta_common[call_mask]
            - valid_rate[call_mask]
            * valid_strike[call_mask]
            * discounted_strike_factor[call_mask]
            * norm.cdf(d2[call_mask])
            + valid_dividend_yield[call_mask]
            * valid_spot[call_mask]
            * discounted_spot_factor[call_mask]
            * norm.cdf(d1[call_mask])
        )
        theta_annual[put_mask] = (
            theta_common[put_mask]
            + valid_rate[put_mask]
            * valid_strike[put_mask]
            * discounted_strike_factor[put_mask]
            * norm.cdf(-d2[put_mask])
            - valid_dividend_yield[put_mask]
            * valid_spot[put_mask]
            * discounted_spot_factor[put_mask]
            * norm.cdf(-d1[put_mask])
        )
        rho_per_rate_point[call_mask] = (
            valid_strike[call_mask]
            * valid_time[call_mask]
            * discounted_strike_factor[call_mask]
            * norm.cdf(d2[call_mask])
            / 100.0
        )
        rho_per_rate_point[put_mask] = (
            -valid_strike[put_mask]
            * valid_time[put_mask]
            * discounted_strike_factor[put_mask]
            * norm.cdf(-d2[put_mask])
            / 100.0
        )
        gamma = discounted_spot_factor * density_d1 / (
            valid_spot * volatility_time
        )
        vega_per_vol_point = (
            valid_spot
            * discounted_spot_factor
            * density_d1
            * sqrt_time
            / 100.0
        )

    valid_index = np.flatnonzero(valid)
    result["theoretical_price"].flat[valid_index] = price
    result["delta"].flat[valid_index] = delta
    result["gamma"].flat[valid_index] = gamma
    result["theta_per_day"].flat[valid_index] = theta_annual / float(annualization_days)
    result["vega_per_vol_point"].flat[valid_index] = vega_per_vol_point
    result["rho_per_rate_point"].flat[valid_index] = rho_per_rate_point
    return result


def _prepare_position_keys(positions_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"underlying", "expiration", "option_type", "strike", "net_quantity"}
    missing_columns = sorted(required_columns - set(positions_df.columns))
    if missing_columns:
        raise ValueError(
            "positions_df is missing required option columns: "
            + ", ".join(missing_columns)
        )

    frame = positions_df.copy()
    frame["_underlying_key"] = frame["underlying"].map(_lookup_key)
    frame["_expiration_key"] = pd.to_datetime(frame["expiration"], errors="coerce").dt.normalize()
    frame["_option_type_key"] = frame["option_type"].map(_option_type_key)
    frame["_strike_key"] = pd.to_numeric(frame["strike"], errors="coerce").round(6)
    frame["net_quantity"] = pd.to_numeric(frame["net_quantity"], errors="coerce")
    return frame


def _prepare_chain_frame(current_chains_by_underlying: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for underlying, chain in (current_chains_by_underlying or {}).items():
        if chain is None or len(chain) == 0:
            continue
        chain_frame = pd.DataFrame(chain).copy()
        if chain_frame.empty:
            continue

        chain_frame["_underlying_key"] = _lookup_key(underlying)
        chain_frame["_expiration_key"] = pd.to_datetime(
            _column_or_nan(chain_frame, "Expiration Date"),
            errors="coerce",
        ).dt.normalize()
        chain_frame["_option_type_key"] = _column_or_nan(chain_frame, "Type").map(_option_type_key)
        chain_frame["_strike_key"] = pd.to_numeric(
            _column_or_nan(chain_frame, "strike"),
            errors="coerce",
        ).round(6)
        chain_frame["chain_implied_volatility"] = _column_or_nan(
            chain_frame,
            "impliedVolatility",
        ).map(_normalize_implied_volatility)

        for column in ("bid", "ask", "lastPrice", "mid", "openInterest", "Days Till Expiration"):
            chain_frame[f"chain_{column}"] = pd.to_numeric(
                _column_or_nan(chain_frame, column),
                errors="coerce",
            )

        midpoint = chain_frame["chain_mid"]
        fallback_midpoint = (chain_frame["chain_bid"] + chain_frame["chain_ask"]) / 2.0
        chain_frame["chain_option_mark"] = midpoint.where(midpoint.gt(0), fallback_midpoint)
        chain_frame["chain_option_mark"] = chain_frame["chain_option_mark"].where(
            chain_frame["chain_option_mark"].gt(0),
            chain_frame["chain_lastPrice"],
        )
        chain_frame["chain_contractSymbol"] = _column_or_nan(chain_frame, "contractSymbol")
        chain_frame["_chain_matched"] = True

        rows.append(
            chain_frame[
                [
                    "_underlying_key",
                    "_expiration_key",
                    "_option_type_key",
                    "_strike_key",
                    "_chain_matched",
                    "chain_contractSymbol",
                    "chain_implied_volatility",
                    "chain_option_mark",
                    "chain_bid",
                    "chain_ask",
                    "chain_lastPrice",
                    "chain_mid",
                    "chain_openInterest",
                    "chain_Days Till Expiration",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "_underlying_key",
                "_expiration_key",
                "_option_type_key",
                "_strike_key",
                "_chain_matched",
                "chain_contractSymbol",
                "chain_implied_volatility",
                "chain_option_mark",
                "chain_bid",
                "chain_ask",
                "chain_lastPrice",
                "chain_mid",
                "chain_openInterest",
                "chain_Days Till Expiration",
            ]
        )

    chain_frame = pd.concat(rows, ignore_index=True)
    return chain_frame.drop_duplicates(
        ["_underlying_key", "_expiration_key", "_option_type_key", "_strike_key"],
        keep="first",
    )


def build_option_greeks_frame(
    positions_df: pd.DataFrame,
    current_chains_by_underlying: Mapping[str, pd.DataFrame],
    spot_price_by_underlying: Mapping[str, float],
    *,
    annual_rate: float = 0.02,
    dividend_yield_by_underlying: Mapping[str, float] | None = None,
    as_of_date: pd.Timestamp | str | None = None,
    contract_multiplier: float = DEFAULT_CONTRACT_MULTIPLIER,
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS,
) -> pd.DataFrame:
    """Join option positions to current chains and calculate per-leg Greeks."""
    if positions_df is None or positions_df.empty:
        return pd.DataFrame()

    as_of = (
        pd.Timestamp.today().normalize()
        if as_of_date is None
        else pd.Timestamp(as_of_date).normalize()
    )
    positions = _prepare_position_keys(positions_df)
    chain = _prepare_chain_frame(current_chains_by_underlying)
    frame = positions.merge(
        chain,
        how="left",
        on=["_underlying_key", "_expiration_key", "_option_type_key", "_strike_key"],
    )
    frame["_chain_matched"] = frame["_chain_matched"].astype("boolean").fillna(False).astype(bool)

    spot_lookup = _coerce_numeric_map(spot_price_by_underlying)
    dividend_lookup = _coerce_numeric_map(dividend_yield_by_underlying)
    frame["spot"] = frame["_underlying_key"].map(spot_lookup)
    frame["dividend_yield"] = frame["_underlying_key"].map(dividend_lookup).fillna(0.0)
    expiration_days = (frame["_expiration_key"] - as_of).dt.days
    frame["days_to_expiration_for_greeks"] = expiration_days
    frame["time_years"] = expiration_days.clip(lower=1) / float(annualization_days)
    frame["risk_free_rate"] = float(annual_rate)

    greek_rows = pd.DataFrame(
        _black_scholes_greeks_arrays(
            spot=pd.to_numeric(frame["spot"], errors="coerce"),
            strike=pd.to_numeric(frame["strike"], errors="coerce"),
            time_years=pd.to_numeric(frame["time_years"], errors="coerce"),
            annual_rate=pd.to_numeric(frame["risk_free_rate"], errors="coerce"),
            volatility=pd.to_numeric(frame["chain_implied_volatility"], errors="coerce"),
            option_type_keys=frame["_option_type_key"].to_numpy(dtype=object),
            dividend_yield=pd.to_numeric(frame["dividend_yield"], errors="coerce"),
            annualization_days=annualization_days,
        ),
        index=frame.index,
    )
    frame = pd.concat([frame, greek_rows], axis=1)

    multiplier = float(contract_multiplier)
    signed_contracts = pd.to_numeric(frame["net_quantity"], errors="coerce")
    frame["contract_multiplier"] = multiplier
    frame["abs_contracts"] = signed_contracts.abs()
    frame["position_market_value"] = frame["chain_option_mark"] * signed_contracts * multiplier
    frame["position_delta_shares"] = frame["delta"] * signed_contracts * multiplier
    frame["position_delta_notional"] = frame["position_delta_shares"] * frame["spot"]
    frame["position_gamma_delta_per_dollar"] = frame["gamma"] * signed_contracts * multiplier
    frame["position_theta_per_day"] = frame["theta_per_day"] * signed_contracts * multiplier
    frame["position_vega_per_vol_point"] = frame["vega_per_vol_point"] * signed_contracts * multiplier
    frame["position_rho_per_rate_point"] = frame["rho_per_rate_point"] * signed_contracts * multiplier

    invalid_terms = (
        pd.to_numeric(frame["strike"], errors="coerce").le(0)
        | signed_contracts.isna()
        | frame["_option_type_key"].isna()
    )
    conditions = [
        ~frame["_chain_matched"],
        frame["spot"].isna() | frame["spot"].le(0),
        frame["chain_implied_volatility"].isna() | frame["chain_implied_volatility"].le(0),
        frame["time_years"].isna() | frame["time_years"].le(0),
        invalid_terms,
    ]
    choices = [
        "no_current_chain_match",
        "missing_spot",
        "missing_implied_volatility",
        "invalid_time_to_expiration",
        "invalid_contract_terms",
    ]
    frame["greek_status"] = np.select(conditions, choices, default="ok")

    helper_columns = [column for column in frame.columns if column.startswith("_")]
    return frame.drop(columns=helper_columns)


def summarize_option_greeks(
    greeks_frame: pd.DataFrame,
    *,
    by: str | list[str] = "underlying",
) -> pd.DataFrame:
    """Aggregate per-leg Greek exposures by underlying, with a portfolio total row."""
    if greeks_frame is None or greeks_frame.empty:
        return pd.DataFrame()

    group_columns = [by] if isinstance(by, str) else list(by)
    exposure_columns = [
        "position_market_value",
        "position_delta_shares",
        "position_delta_notional",
        "position_gamma_delta_per_dollar",
        "position_theta_per_day",
        "position_vega_per_vol_point",
        "position_rho_per_rate_point",
    ]
    available_exposure_columns = [
        column for column in exposure_columns if column in greeks_frame.columns
    ]
    summary = (
        greeks_frame.groupby(group_columns, dropna=False)[available_exposure_columns]
        .sum(min_count=1)
        .reset_index()
    )

    counts = (
        greeks_frame.assign(
            option_legs=1,
            priced_legs=greeks_frame["greek_status"].eq("ok").astype(int),
            unmatched_legs=greeks_frame["greek_status"].ne("ok").astype(int),
            gross_contracts=pd.to_numeric(greeks_frame["net_quantity"], errors="coerce").abs(),
        )
        .groupby(group_columns, dropna=False)[
            ["option_legs", "priced_legs", "unmatched_legs", "gross_contracts"]
        ]
        .sum()
        .reset_index()
    )
    summary = counts.merge(summary, on=group_columns, how="left")

    total = {
        column: "TOTAL" if idx == 0 else ""
        for idx, column in enumerate(group_columns)
    }
    for column in ["option_legs", "priced_legs", "unmatched_legs", "gross_contracts"]:
        total[column] = counts[column].sum()
    for column in available_exposure_columns:
        total[column] = summary[column].sum()

    return pd.concat([summary, pd.DataFrame([total])], ignore_index=True)


def build_option_greek_sensitivity_frame(
    greeks_frame: pd.DataFrame,
    *,
    scenario_moves: np.ndarray | list[float] | None = None,
    dte_buckets: list[tuple[str, int | None, int | None]] | tuple[tuple[str, int | None, int | None], ...] | None = None,
    include_total: bool = True,
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS,
) -> pd.DataFrame:
    """Recalculate aggregate Greek exposures across percentage spot shocks.

    The same percentage move is applied to each underlying. IV, time to
    expiration, risk-free rate, and dividend yield are held constant.
    """
    if greeks_frame is None or greeks_frame.empty:
        return pd.DataFrame()

    scenario_moves = (
        np.linspace(-0.30, 0.30, 121)
        if scenario_moves is None
        else np.asarray(scenario_moves, dtype=float)
    )
    scenario_moves = scenario_moves[np.isfinite(scenario_moves)]
    if scenario_moves.size == 0:
        return pd.DataFrame()

    dte_buckets = DEFAULT_DTE_BUCKETS if dte_buckets is None else dte_buckets
    valid = greeks_frame.copy()
    if "greek_status" in valid.columns:
        valid = valid[valid["greek_status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()

    required_columns = {
        "underlying",
        "spot",
        "strike",
        "time_years",
        "risk_free_rate",
        "chain_implied_volatility",
        "option_type",
        "net_quantity",
        "days_to_expiration_for_greeks",
    }
    missing_columns = sorted(required_columns - set(valid.columns))
    if missing_columns:
        raise ValueError(
            "greeks_frame is missing required sensitivity columns: "
            + ", ".join(missing_columns)
        )

    numeric_columns = [
        "spot",
        "strike",
        "time_years",
        "risk_free_rate",
        "chain_implied_volatility",
        "dividend_yield",
        "net_quantity",
        "contract_multiplier",
        "days_to_expiration_for_greeks",
    ]
    for column in numeric_columns:
        if column in valid.columns:
            valid[column] = pd.to_numeric(valid[column], errors="coerce")

    if "contract_multiplier" not in valid.columns:
        valid["contract_multiplier"] = DEFAULT_CONTRACT_MULTIPLIER
    if "dividend_yield" not in valid.columns:
        valid["dividend_yield"] = 0.0
    valid["dividend_yield"] = valid["dividend_yield"].fillna(0.0)

    valid["_option_type_key"] = valid["option_type"].map(_option_type_key)
    valid["_contract_multiplier"] = valid["contract_multiplier"].where(
        valid["contract_multiplier"].gt(0),
        DEFAULT_CONTRACT_MULTIPLIER,
    )
    valid = valid[
        valid["underlying"].notna()
        & valid["_option_type_key"].notna()
        & valid["spot"].gt(0)
        & valid["strike"].gt(0)
        & valid["time_years"].gt(0)
        & valid["chain_implied_volatility"].gt(0)
        & valid["net_quantity"].notna()
        & valid["_contract_multiplier"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    exposure_columns = [
        "scenario_position_value",
        "position_delta_shares",
        "position_delta_notional",
        "position_gamma_delta_per_dollar",
        "position_theta_per_day",
        "position_vega_per_vol_point",
        "position_rho_per_rate_point",
    ]

    scenario_count = len(scenario_moves)
    leg_count = len(valid)
    leg_indexes = np.tile(np.arange(leg_count), scenario_count)
    scenario_indexes = np.repeat(np.arange(scenario_count), leg_count)

    base_spot = valid["spot"].to_numpy(dtype=float)
    scenario_spot = base_spot[leg_indexes] * (1.0 + scenario_moves[scenario_indexes])
    finite_scenario = np.isfinite(scenario_spot) & (scenario_spot > 0)
    if not finite_scenario.any():
        return pd.DataFrame()

    leg_indexes = leg_indexes[finite_scenario]
    scenario_indexes = scenario_indexes[finite_scenario]
    scenario_spot = scenario_spot[finite_scenario]

    signed_multiplier = (
        valid["net_quantity"].to_numpy(dtype=float)
        * valid["_contract_multiplier"].to_numpy(dtype=float)
    )
    greek_arrays = _black_scholes_greeks_arrays(
        spot=scenario_spot,
        strike=valid["strike"].to_numpy(dtype=float)[leg_indexes],
        time_years=valid["time_years"].to_numpy(dtype=float)[leg_indexes],
        annual_rate=valid["risk_free_rate"].to_numpy(dtype=float)[leg_indexes],
        volatility=valid["chain_implied_volatility"].to_numpy(dtype=float)[leg_indexes],
        option_type_keys=valid["_option_type_key"].to_numpy(dtype=object)[leg_indexes],
        dividend_yield=valid["dividend_yield"].to_numpy(dtype=float)[leg_indexes],
        annualization_days=annualization_days,
    )
    position_multiplier = signed_multiplier[leg_indexes]
    delta_shares = greek_arrays["delta"] * position_multiplier
    scenario_frame = pd.DataFrame(
        {
            "scenario_move": scenario_moves[scenario_indexes],
            "underlying": valid["underlying"].astype(str).to_numpy()[leg_indexes],
            "days_to_expiration_for_greeks": valid[
                "days_to_expiration_for_greeks"
            ].to_numpy(dtype=float)[leg_indexes],
            "scenario_spot": scenario_spot,
            "scenario_position_value": (
                greek_arrays["theoretical_price"] * position_multiplier
            ),
            "position_delta_shares": delta_shares,
            "position_delta_notional": delta_shares * scenario_spot,
            "position_gamma_delta_per_dollar": (
                greek_arrays["gamma"] * position_multiplier
            ),
            "position_theta_per_day": (
                greek_arrays["theta_per_day"] * position_multiplier
            ),
            "position_vega_per_vol_point": (
                greek_arrays["vega_per_vol_point"] * position_multiplier
            ),
            "position_rho_per_rate_point": (
                greek_arrays["rho_per_rate_point"] * position_multiplier
            ),
        }
    )
    finite_exposure = scenario_frame[exposure_columns].notna().any(axis=1)
    scenario_frame = scenario_frame[finite_exposure].copy()
    if scenario_frame.empty:
        return pd.DataFrame()

    bucketed_rows = []
    dte = pd.to_numeric(
        scenario_frame["days_to_expiration_for_greeks"],
        errors="coerce",
    )
    for bucket_label, lower_bound, upper_bound in dte_buckets:
        mask = dte.notna()
        if lower_bound is not None:
            mask &= dte >= float(lower_bound)
        if upper_bound is not None:
            mask &= dte <= float(upper_bound)
        bucket_frame = scenario_frame[mask]
        if bucket_frame.empty:
            continue

        grouped = (
            bucket_frame.groupby(["scenario_move", "underlying"], as_index=False)
            .agg(
                scenario_spot=("scenario_spot", "first"),
                **{column: (column, "sum") for column in exposure_columns},
            )
        )
        grouped["dte_bucket"] = bucket_label
        bucketed_rows.append(grouped)

        if include_total:
            total = (
                grouped.groupby("scenario_move", as_index=False)[exposure_columns]
                .sum()
            )
            total["underlying"] = "TOTAL"
            total["dte_bucket"] = bucket_label
            total["scenario_spot"] = np.nan
            total = total[
                [
                    "scenario_move",
                    "underlying",
                    "scenario_spot",
                    *exposure_columns,
                    "dte_bucket",
                ]
            ]
            bucketed_rows.append(total)

    if not bucketed_rows:
        return pd.DataFrame()

    return (
        pd.concat(bucketed_rows, ignore_index=True)
        .sort_values(["underlying", "dte_bucket", "scenario_move"])
        .reset_index(drop=True)
    )

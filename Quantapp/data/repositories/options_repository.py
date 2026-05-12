"""Stable options data access interfaces for notebooks and apps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from Quantapp.data.adapters.options import (
    build_historical_eod_row,
    build_options_chain_dicts,
    concat_options_chain_values,
    normalize_massive_snapshot,
    normalize_reference_contracts,
)
from Quantapp.data.market_history import get_market_history
from Quantapp.data.schemas.options import HISTORICAL_OPTIONS_EOD_COLUMNS
from Quantapp.data.sources.massive_options import (
    fetch_option_daily_bar,
    fetch_options_chain_snapshot,
    fetch_reference_options_contracts,
)


@dataclass
class CurrentOptionsChain:
    """Normalized current options chain data."""

    chain: pd.DataFrame
    underlying_price: float
    calls_by_expiration: dict[str, pd.DataFrame]
    puts_by_expiration: dict[str, pd.DataFrame]
    expirations: list[str]
    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass
class HistoricalOptionsEodPanel:
    """Historical option EOD bars and retrieval summary."""

    chain: pd.DataFrame
    summary: pd.DataFrame


def _require_massive_provider(provider: str) -> None:
    normalized_provider = str(provider).strip().lower()
    if normalized_provider not in {"massive", "polygon"}:
        raise ValueError(f"Unsupported options provider '{provider}'.")


def get_current_options_chain(
    underlying_ticker: str,
    *,
    provider: str = "massive",
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_underlying_price: float | None = None,
    as_of_date: pd.Timestamp | str | None = None,
) -> CurrentOptionsChain:
    """Fetch and normalize the current options chain for one underlying."""
    _require_massive_provider(provider)

    snapshot_results = fetch_options_chain_snapshot(
        underlying_ticker,
        api_key=api_key,
        base_url=base_url,
    )
    chain_df, underlying_price = normalize_massive_snapshot(
        snapshot_results,
        as_of_date=as_of_date,
        fallback_underlying_price=fallback_underlying_price,
    )
    calls_by_expiration, puts_by_expiration, expirations = build_options_chain_dicts(chain_df)
    calls = concat_options_chain_values(calls_by_expiration, chain_df.columns)
    puts = concat_options_chain_values(puts_by_expiration, chain_df.columns)

    return CurrentOptionsChain(
        chain=chain_df,
        underlying_price=underlying_price,
        calls_by_expiration=calls_by_expiration,
        puts_by_expiration=puts_by_expiration,
        expirations=expirations,
        calls=calls,
        puts=puts,
    )


def _load_underlying_history(
    underlying_ticker: str,
    *,
    underlying_history: pd.DataFrame | None,
    lookback_days: int,
) -> pd.DataFrame:
    if underlying_history is not None:
        return underlying_history.copy()

    history_period = f"{max(lookback_days + 10, 60)}d"
    history_map = get_market_history(
        symbols=[underlying_ticker],
        period=history_period,
        interval="1d",
        provider="yfinance",
        align=False,
    )
    return history_map.get(str(underlying_ticker).strip().upper(), pd.DataFrame())


def _underlying_close_map(
    underlying_history: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, float]:
    if underlying_history.empty or "Close" not in underlying_history.columns:
        return {}

    history = underlying_history.copy()
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)
    if history.index.tz is not None:
        history.index = history.index.tz_convert("UTC").tz_localize(None)
    history.index = history.index.normalize()

    history = history.loc[
        (history.index >= start_date - pd.Timedelta(days=5))
        & (history.index <= end_date)
    ]
    return {
        pd.to_datetime(idx).strftime("%Y-%m-%d"): float(val)
        for idx, val in history["Close"].dropna().items()
    }


def _select_contracts(
    contracts_df: pd.DataFrame,
    *,
    as_of_date: str,
    day_spot: float,
    max_contracts_per_day: int,
    max_days_to_expiry: int,
    moneyness_band: float,
) -> tuple[pd.DataFrame, str | None]:
    contracts_df = contracts_df.copy()
    as_of_ts = pd.to_datetime(as_of_date)
    contracts_df["days_to_expiry"] = (contracts_df["expiration_date"] - as_of_ts).dt.days
    contracts_df = contracts_df[
        (contracts_df["days_to_expiry"] >= 0)
        & (contracts_df["days_to_expiry"] <= max_days_to_expiry)
    ].copy()
    if contracts_df.empty:
        return contracts_df, "no contracts in expiry window"

    contracts_df["moneyness_distance_pct"] = (contracts_df["strike_price"] - day_spot).abs() / max(day_spot, 1e-9)
    contracts_df = contracts_df[contracts_df["moneyness_distance_pct"] <= moneyness_band].copy()
    if contracts_df.empty:
        return contracts_df, "no contracts in moneyness band"

    return (
        contracts_df.sort_values(
            ["days_to_expiry", "moneyness_distance_pct", "strike_price"]
        ).head(max_contracts_per_day),
        None,
    )


def get_historical_options_eod_panel(
    underlying_ticker: str,
    *,
    underlying_history: pd.DataFrame | None = None,
    provider: str = "massive",
    api_key: str | None = None,
    base_url: str | None = None,
    lookback_days: int = 30,
    max_contracts_per_day: int = 12,
    max_days_to_expiry: int = 120,
    moneyness_band: float = 0.25,
    end_date: pd.Timestamp | str | None = None,
) -> HistoricalOptionsEodPanel:
    """Fetch a historical options EOD panel for selected near-ATM contracts."""
    _require_massive_provider(provider)

    resolved_end_date = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    )
    start_date = resolved_end_date - pd.Timedelta(days=lookback_days - 1)
    asof_dates = pd.bdate_range(start=start_date, end=resolved_end_date).strftime("%Y-%m-%d").tolist()

    underlying_frame = _load_underlying_history(
        underlying_ticker,
        underlying_history=underlying_history,
        lookback_days=lookback_days,
    )
    close_map = _underlying_close_map(
        underlying_frame,
        start_date=start_date,
        end_date=resolved_end_date,
    )

    historical_rows = []
    daily_counts = []

    for as_of_date in asof_dates:
        day_spot = close_map.get(as_of_date)
        if day_spot is None:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": "missing underlying close",
                }
            )
            continue

        try:
            contracts = fetch_reference_options_contracts(
                underlying_ticker,
                as_of_date,
                api_key=api_key,
                base_url=base_url,
            )
        except requests.RequestException as exc:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": f"reference error: {exc}",
                }
            )
            continue

        if not contracts:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": "no contracts returned",
                }
            )
            continue

        try:
            contracts_df = normalize_reference_contracts(contracts)
        except ValueError:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": "missing required metadata fields",
                }
            )
            continue

        contracts_df, empty_reason = _select_contracts(
            contracts_df,
            as_of_date=as_of_date,
            day_spot=day_spot,
            max_contracts_per_day=max_contracts_per_day,
            max_days_to_expiry=max_days_to_expiry,
            moneyness_band=moneyness_band,
        )
        selected_count = int(len(contracts_df))
        if empty_reason or selected_count == 0:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": empty_reason or "no contracts in moneyness band",
                }
            )
            continue

        eod_count = 0
        for contract in contracts_df.to_dict(orient="records"):
            try:
                eod_bar = fetch_option_daily_bar(
                    contract["ticker"],
                    as_of_date,
                    api_key=api_key,
                    base_url=base_url,
                )
            except requests.RequestException:
                continue

            if not eod_bar:
                continue

            eod_count += 1
            historical_rows.append(
                build_historical_eod_row(
                    contract=contract,
                    eod_bar=eod_bar,
                    as_of_date=as_of_date,
                    day_spot=day_spot,
                )
            )

        daily_counts.append(
            {
                "as_of_date": as_of_date,
                "contracts_selected": selected_count,
                "contracts_with_eod": eod_count,
                "reason": "ok",
            }
        )

    chain = pd.DataFrame(historical_rows, columns=HISTORICAL_OPTIONS_EOD_COLUMNS)
    if not chain.empty:
        chain = chain.sort_values(
            ["as_of_date", "Expiration Date", "Type", "strike"]
        ).reset_index(drop=True)

    summary = pd.DataFrame(daily_counts)
    return HistoricalOptionsEodPanel(chain=chain, summary=summary)

"""Stable options data access interfaces for notebooks and apps."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    fetch_option_daily_bars,
    fetch_options_chain_snapshot,
    fetch_reference_options_contracts_paginated,
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
    query_summary: dict[str, int]


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
    max_contracts_per_day: int | None,
    max_days_to_expiry: int | None,
    moneyness_band: float,
    selection_mode: str,
    target_dtes: Sequence[int] | None,
) -> tuple[pd.DataFrame, str | None]:
    contracts_df = contracts_df.copy()
    as_of_ts = pd.to_datetime(as_of_date)
    contracts_df["days_to_expiry"] = (contracts_df["expiration_date"] - as_of_ts).dt.days
    expiration_mask = contracts_df["days_to_expiry"] >= 0
    if max_days_to_expiry is not None:
        expiration_mask &= (
            contracts_df["days_to_expiry"] <= max_days_to_expiry
        )
    contracts_df = contracts_df[expiration_mask].copy()
    if contracts_df.empty:
        return contracts_df, "no contracts in expiry window"

    contracts_df["moneyness_distance_pct"] = (contracts_df["strike_price"] - day_spot).abs() / max(day_spot, 1e-9)
    contracts_df = contracts_df[contracts_df["moneyness_distance_pct"] <= moneyness_band].copy()
    if contracts_df.empty:
        return contracts_df, "no contracts in moneyness band"

    contracts_df = contracts_df.sort_values(
        ["days_to_expiry", "moneyness_distance_pct", "strike_price"]
    )
    if selection_mode in {
        "nearest_atm_per_expiration",
        "nearest_atm_by_target_dte",
    }:
        contracts_df = (
            contracts_df.groupby(
                ["expiration_date", "contract_type"],
                sort=True,
                as_index=False,
                group_keys=False,
            )
            .head(1)
            .sort_values(["days_to_expiry", "moneyness_distance_pct", "strike_price"])
        )
        if selection_mode == "nearest_atm_by_target_dte":
            normalized_target_dtes = sorted(
                {
                    int(target_dte)
                    for target_dte in (target_dtes or [])
                    if int(target_dte) >= 0
                }
            )
            if not normalized_target_dtes:
                raise ValueError(
                    "target_dtes must contain at least one non-negative DTE when "
                    "selection_mode='nearest_atm_by_target_dte'."
                )
            expiration_candidates = contracts_df[
                ["expiration_date", "days_to_expiry"]
            ].drop_duplicates()
            selected_expirations = set()
            for target_dte in normalized_target_dtes:
                available_expirations = expiration_candidates[
                    ~expiration_candidates["expiration_date"].isin(
                        selected_expirations
                    )
                ]
                if available_expirations.empty:
                    break
                selected_expirations.add(
                    available_expirations.loc[
                        (
                            available_expirations["days_to_expiry"]
                            - target_dte
                        ).abs().idxmin(),
                        "expiration_date",
                    ]
                )
            contracts_df = contracts_df[
                contracts_df["expiration_date"].isin(selected_expirations)
            ]
    elif selection_mode != "nearest_contracts":
        raise ValueError(
            "selection_mode must be 'nearest_contracts', "
            "'nearest_atm_per_expiration', or 'nearest_atm_by_target_dte'."
        )

    if max_contracts_per_day is not None:
        contracts_df = contracts_df.head(max_contracts_per_day)
    return contracts_df, None


def _daily_bar_date_keys(eod_bar: dict) -> set[str]:
    """Return plausible session dates for a Massive daily aggregate timestamp."""
    timestamp = pd.to_numeric(eod_bar.get("t"), errors="coerce")
    if pd.isna(timestamp):
        return set()
    timestamp_utc = pd.to_datetime(timestamp, unit="ms", utc=True)
    return {
        timestamp_utc.strftime("%Y-%m-%d"),
        timestamp_utc.tz_convert("America/New_York").strftime("%Y-%m-%d"),
    }


def get_historical_options_eod_panel(
    underlying_ticker: str,
    *,
    underlying_history: pd.DataFrame | None = None,
    provider: str = "massive",
    api_key: str | None = None,
    base_url: str | None = None,
    lookback_days: int = 30,
    max_contracts_per_day: int | None = 12,
    max_days_to_expiry: int | None = 120,
    moneyness_band: float = 0.25,
    selection_mode: str = "nearest_contracts",
    target_dtes: Sequence[int] | None = None,
    reference_refresh: str = "daily",
    bar_retrieval_mode: str = "per_contract_range",
    max_workers: int = 16,
    end_date: pd.Timestamp | str | None = None,
) -> HistoricalOptionsEodPanel:
    """Fetch a historical options EOD panel for selected near-ATM contracts."""
    _require_massive_provider(provider)
    if bar_retrieval_mode not in {"per_contract_range", "per_day"}:
        raise ValueError(
            "bar_retrieval_mode must be 'per_contract_range' or 'per_day'."
        )
    if reference_refresh not in {"daily", "weekly"}:
        raise ValueError("reference_refresh must be 'daily' or 'weekly'.")
    if int(max_workers) < 1:
        raise ValueError("max_workers must be at least 1.")

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
    selected_contract_requests = []
    selected_option_tickers = set()
    reference_query_count = 0
    bar_query_count = 0
    eligible_trading_date_count = 0
    reference_cache = {}
    reference_errors = {}

    def reference_key_for(as_of_date):
        if reference_refresh == "weekly":
            iso_calendar = pd.Timestamp(as_of_date).isocalendar()
            return f"{iso_calendar.year}-W{iso_calendar.week:02d}"
        return as_of_date

    def fetch_reference_snapshot(as_of_date, day_spot):
        expiration_date_lte = (
            (
                pd.Timestamp(as_of_date)
                + pd.Timedelta(days=max_days_to_expiry)
            ).strftime("%Y-%m-%d")
            if max_days_to_expiry is not None
            else None
        )
        try:
            contracts, page_query_count = (
                fetch_reference_options_contracts_paginated(
                    underlying_ticker,
                    as_of_date,
                    api_key=api_key,
                    base_url=base_url,
                    expiration_date_gte=as_of_date,
                    expiration_date_lte=expiration_date_lte,
                    strike_price_gte=day_spot * (1 - moneyness_band),
                    strike_price_lte=day_spot * (1 + moneyness_band),
                )
            )
            if not contracts:
                return None, page_query_count, "no contracts returned"
            return (
                normalize_reference_contracts(contracts),
                page_query_count,
                None,
            )
        except requests.RequestException as exc:
            return None, 1, f"reference error: {exc}"
        except ValueError:
            return None, 1, "missing required metadata fields"

    if reference_refresh == "weekly":
        weekly_requests = {}
        for as_of_date in asof_dates:
            day_spot = close_map.get(as_of_date)
            if day_spot is None:
                continue
            weekly_requests.setdefault(
                reference_key_for(as_of_date), (as_of_date, day_spot)
            )

        worker_count = min(int(max_workers), max(len(weekly_requests), 1))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    fetch_reference_snapshot, as_of_date, day_spot
                ): reference_key
                for reference_key, (as_of_date, day_spot) in weekly_requests.items()
            }
            for future in as_completed(futures):
                reference_key = futures[future]
                contracts_df, page_query_count, error = future.result()
                reference_query_count += page_query_count
                if error:
                    reference_errors[reference_key] = error
                else:
                    reference_cache[reference_key] = contracts_df

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
        eligible_trading_date_count += 1

        reference_key = reference_key_for(as_of_date)

        if reference_key not in reference_cache and reference_key not in reference_errors:
            contracts_df, page_query_count, error = fetch_reference_snapshot(
                as_of_date, day_spot
            )
            reference_query_count += page_query_count
            if error:
                reference_errors[reference_key] = error
            else:
                reference_cache[reference_key] = contracts_df

        if reference_key in reference_errors:
            daily_counts.append(
                {
                    "as_of_date": as_of_date,
                    "contracts_selected": 0,
                    "contracts_with_eod": 0,
                    "reason": reference_errors[reference_key],
                }
            )
            continue

        contracts_df = reference_cache[reference_key].copy()
        contracts_df, empty_reason = _select_contracts(
            contracts_df,
            as_of_date=as_of_date,
            day_spot=day_spot,
            max_contracts_per_day=max_contracts_per_day,
            max_days_to_expiry=max_days_to_expiry,
            moneyness_band=moneyness_band,
            selection_mode=selection_mode,
            target_dtes=target_dtes,
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

        daily_count = {
            "as_of_date": as_of_date,
            "contracts_selected": selected_count,
            "contracts_with_eod": 0,
            "reason": "ok",
        }
        contract_records = contracts_df.to_dict(orient="records")
        selected_option_tickers.update(
            contract["ticker"] for contract in contract_records
        )
        if bar_retrieval_mode == "per_day":
            for contract in contract_records:
                try:
                    bar_query_count += 1
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

                daily_count["contracts_with_eod"] += 1
                historical_rows.append(
                    build_historical_eod_row(
                        contract=contract,
                        eod_bar=eod_bar,
                        as_of_date=as_of_date,
                        day_spot=day_spot,
                    )
                )
        else:
            selected_contract_requests.extend(
                {
                    "contract": contract,
                    "as_of_date": as_of_date,
                    "day_spot": day_spot,
                    "daily_count": daily_count,
                }
                for contract in contract_records
            )
        daily_counts.append(daily_count)

    if bar_retrieval_mode == "per_contract_range" and selected_contract_requests:
        requests_by_contract = defaultdict(list)
        for selected_request in selected_contract_requests:
            requests_by_contract[selected_request["contract"]["ticker"]].append(
                selected_request
            )

        def fetch_contract_range(option_ticker, contract_requests):
            selected_dates = [request["as_of_date"] for request in contract_requests]
            try:
                return fetch_option_daily_bars(
                    option_ticker,
                    min(selected_dates),
                    max(selected_dates),
                    api_key=api_key,
                    base_url=base_url,
                )
            except requests.RequestException:
                return []

        bar_query_count += len(requests_by_contract)
        fetched_bars_by_contract = {}
        worker_count = min(int(max_workers), len(requests_by_contract))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    fetch_contract_range, option_ticker, contract_requests
                ): option_ticker
                for option_ticker, contract_requests in requests_by_contract.items()
            }
            for future in as_completed(futures):
                fetched_bars_by_contract[futures[future]] = future.result()

        for option_ticker, contract_requests in requests_by_contract.items():
            eod_bars = fetched_bars_by_contract.get(option_ticker, [])
            bars_by_date = {}
            for eod_bar in eod_bars:
                for bar_date in _daily_bar_date_keys(eod_bar):
                    bars_by_date.setdefault(bar_date, eod_bar)

            for selected_request in contract_requests:
                eod_bar = bars_by_date.get(selected_request["as_of_date"])
                if not eod_bar:
                    continue
                selected_request["daily_count"]["contracts_with_eod"] += 1
                historical_rows.append(
                    build_historical_eod_row(
                        contract=selected_request["contract"],
                        eod_bar=eod_bar,
                        as_of_date=selected_request["as_of_date"],
                        day_spot=selected_request["day_spot"],
                    )
                )

    chain = pd.DataFrame(historical_rows, columns=HISTORICAL_OPTIONS_EOD_COLUMNS)
    if not chain.empty:
        chain = chain.sort_values(
            ["as_of_date", "Expiration Date", "Type", "strike"]
        ).reset_index(drop=True)

    summary = pd.DataFrame(daily_counts)
    selected_observation_count = int(
        summary["contracts_selected"].sum() if not summary.empty else 0
    )
    unique_contract_count = len(selected_option_tickers)
    query_summary = {
        "reference_queries": reference_query_count,
        "reference_queries_avoided": max(
            eligible_trading_date_count - reference_query_count, 0
        ),
        "bar_queries": bar_query_count,
        "total_queries": reference_query_count + bar_query_count,
        "selected_contract_observations": selected_observation_count,
        "unique_contracts": unique_contract_count,
        "bar_queries_avoided": max(selected_observation_count - bar_query_count, 0),
    }
    return HistoricalOptionsEodPanel(
        chain=chain,
        summary=summary,
        query_summary=query_summary,
    )

"""Normalize provider options data into Quantapp shapes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Quantapp.data.schemas.options import (
    CURRENT_OPTIONS_CHAIN_COLUMNS,
    HISTORICAL_OPTIONS_EOD_COLUMNS,
    REFERENCE_CONTRACT_REQUIRED_COLUMNS,
    SNAPSHOT_REQUIRED_COLUMNS,
)


def _numeric_column(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[column], errors="coerce")


def normalize_massive_snapshot(
    snapshot_results: list[dict],
    *,
    as_of_date: pd.Timestamp | str | None = None,
    fallback_underlying_price: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Normalize a Massive options snapshot into the notebook's chain shape."""
    if not snapshot_results:
        raise ValueError("Massive snapshot returned no contracts.")

    raw = pd.json_normalize(snapshot_results)
    missing_cols = sorted(SNAPSHOT_REQUIRED_COLUMNS - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Massive response missing required fields: {', '.join(missing_cols)}")

    today = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()

    df = pd.DataFrame(index=raw.index)
    df["contractSymbol"] = raw["details.ticker"]
    df["Expiration Date"] = raw["details.expiration_date"]
    df["strike"] = _numeric_column(raw, "details.strike_price")
    df["Type"] = raw["details.contract_type"].astype(str).str.lower().map({"call": "Call", "put": "Put"})
    df["openInterest"] = _numeric_column(raw, "open_interest")
    df["impliedVolatility"] = _numeric_column(raw, "implied_volatility")
    df["bid"] = _numeric_column(raw, "last_quote.bid")
    df["ask"] = _numeric_column(raw, "last_quote.ask")
    df["lastPrice"] = _numeric_column(raw, "day.close")
    df["volume"] = _numeric_column(raw, "day.volume")
    df["change"] = _numeric_column(raw, "day.change")
    df["percentChange"] = _numeric_column(raw, "day.change_percent")
    df["currency"] = "USD"
    df["contractSize"] = "REGULAR"

    underlying_prices = _numeric_column(raw, "underlying_asset.price")
    if underlying_prices.notna().any():
        underlying_price = float(underlying_prices.dropna().iloc[0])
    elif fallback_underlying_price is not None:
        underlying_price = float(fallback_underlying_price)
    else:
        underlying_price = np.nan

    expiration_dt = pd.to_datetime(df["Expiration Date"])
    df["Days Till Expiration"] = (expiration_dt - today).dt.days
    df["Expiration day"] = expiration_dt.dt.day
    df["Expiration day name"] = expiration_dt.dt.strftime("%A")
    df["bid-ask spread"] = df["ask"] - df["bid"]
    df["mid"] = (df["bid"] + df["ask"]) / 2

    if np.isnan(underlying_price):
        df["inTheMoney"] = np.nan
    else:
        df["inTheMoney"] = np.where(
            df["Type"] == "Call",
            df["strike"] < underlying_price,
            np.where(df["Type"] == "Put", df["strike"] > underlying_price, np.nan),
        )

    df = df[df["Type"].isin(["Call", "Put"])].copy()
    if df.empty:
        raise ValueError("Massive snapshot did not include call/put contract types.")

    return df[CURRENT_OPTIONS_CHAIN_COLUMNS], underlying_price


def build_options_chain_dicts(
    chain_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Split a normalized options chain into call/put maps by expiration."""
    call_chain = {}
    put_chain = {}

    for expiration_date, expiration_df in chain_df.groupby("Expiration Date", sort=True):
        call_chain[expiration_date] = expiration_df[expiration_df["Type"] == "Call"].reset_index(drop=True)
        put_chain[expiration_date] = expiration_df[expiration_df["Type"] == "Put"].reset_index(drop=True)

    all_expirations = sorted(set(call_chain.keys()) | set(put_chain.keys()))
    empty_frame = chain_df.iloc[0:0].copy()
    for expiration_date in all_expirations:
        call_chain.setdefault(expiration_date, empty_frame.copy())
        put_chain.setdefault(expiration_date, empty_frame.copy())

    return call_chain, put_chain, all_expirations


def concat_options_chain_values(
    chain_dict: dict[str, pd.DataFrame],
    reference_columns: list[str] | pd.Index,
) -> pd.DataFrame:
    """Concatenate non-empty expiration frames while preserving expected columns."""
    non_empty = [df for df in chain_dict.values() if not df.empty]
    if non_empty:
        return pd.concat(non_empty, ignore_index=True)
    return pd.DataFrame(columns=reference_columns)


def normalize_reference_contracts(contracts: list[dict]) -> pd.DataFrame:
    """Normalize Massive reference contract metadata."""
    if not contracts:
        return pd.DataFrame(columns=sorted(REFERENCE_CONTRACT_REQUIRED_COLUMNS))

    contracts_df = pd.json_normalize(contracts)
    missing_cols = sorted(REFERENCE_CONTRACT_REQUIRED_COLUMNS - set(contracts_df.columns))
    if missing_cols:
        raise ValueError(f"Massive reference response missing required fields: {', '.join(missing_cols)}")

    contracts_df = contracts_df.copy()
    contracts_df["strike_price"] = pd.to_numeric(contracts_df["strike_price"], errors="coerce")
    contracts_df["expiration_date"] = pd.to_datetime(contracts_df["expiration_date"], errors="coerce")
    return contracts_df.dropna(subset=["strike_price", "expiration_date"])


def select_contracts_for_eod_panel(
    contracts_df: pd.DataFrame,
    *,
    as_of_date: str,
    day_spot: float,
    max_contracts_per_day: int,
    max_days_to_expiry: int,
    moneyness_band: float,
) -> pd.DataFrame:
    """Select a small near-ATM contract set for daily EOD bar retrieval."""
    if contracts_df.empty:
        return contracts_df.copy()

    selected = contracts_df.copy()
    as_of_ts = pd.to_datetime(as_of_date)
    selected["days_to_expiry"] = (selected["expiration_date"] - as_of_ts).dt.days
    selected = selected[
        (selected["days_to_expiry"] >= 0)
        & (selected["days_to_expiry"] <= max_days_to_expiry)
    ].copy()
    if selected.empty:
        return selected

    selected["moneyness_distance_pct"] = (selected["strike_price"] - day_spot).abs() / max(day_spot, 1e-9)
    selected = selected[selected["moneyness_distance_pct"] <= moneyness_band].copy()
    if selected.empty:
        return selected

    return selected.sort_values(
        ["days_to_expiry", "moneyness_distance_pct", "strike_price"]
    ).head(max_contracts_per_day)


def build_historical_eod_row(
    *,
    contract: dict,
    eod_bar: dict,
    as_of_date: str,
    day_spot: float,
) -> dict:
    """Combine selected contract metadata with one Massive EOD aggregate bar."""
    option_type = str(contract.get("contract_type", "")).lower()
    option_label = "Call" if option_type == "call" else "Put" if option_type == "put" else option_type.capitalize()
    expiration_date = contract.get("expiration_date")

    row = {
        "as_of_date": as_of_date,
        "contractSymbol": contract.get("ticker"),
        "Expiration Date": expiration_date.strftime("%Y-%m-%d") if pd.notna(expiration_date) else None,
        "strike": float(contract.get("strike_price")) if pd.notna(contract.get("strike_price")) else np.nan,
        "Type": option_label,
        "Days Till Expiration": int(contract.get("days_to_expiry")) if pd.notna(contract.get("days_to_expiry")) else np.nan,
        "open": eod_bar.get("o"),
        "high": eod_bar.get("h"),
        "low": eod_bar.get("l"),
        "lastPrice": eod_bar.get("c"),
        "volume": eod_bar.get("v"),
        "vwap": eod_bar.get("vw"),
        "transactions": eod_bar.get("n"),
        "day_spot": day_spot,
        "moneyness_distance_pct": (
            float(contract.get("moneyness_distance_pct"))
            if pd.notna(contract.get("moneyness_distance_pct"))
            else np.nan
        ),
        "source": "massive_eod_bar",
    }
    return {column: row.get(column) for column in HISTORICAL_OPTIONS_EOD_COLUMNS}

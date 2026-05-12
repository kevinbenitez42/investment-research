"""Shared data contracts for normalized options data."""

from __future__ import annotations

SNAPSHOT_REQUIRED_COLUMNS = frozenset(
    {
        "details.ticker",
        "details.expiration_date",
        "details.strike_price",
        "details.contract_type",
    }
)

REFERENCE_CONTRACT_REQUIRED_COLUMNS = frozenset(
    {
        "ticker",
        "expiration_date",
        "strike_price",
        "contract_type",
    }
)

CURRENT_OPTIONS_CHAIN_COLUMNS = [
    "contractSymbol",
    "Expiration Date",
    "strike",
    "Type",
    "openInterest",
    "impliedVolatility",
    "bid",
    "ask",
    "lastPrice",
    "volume",
    "change",
    "percentChange",
    "currency",
    "contractSize",
    "Days Till Expiration",
    "Expiration day",
    "Expiration day name",
    "bid-ask spread",
    "mid",
    "inTheMoney",
]

HISTORICAL_OPTIONS_EOD_COLUMNS = [
    "as_of_date",
    "contractSymbol",
    "Expiration Date",
    "strike",
    "Type",
    "Days Till Expiration",
    "open",
    "high",
    "low",
    "lastPrice",
    "volume",
    "vwap",
    "transactions",
    "day_spot",
    "moneyness_distance_pct",
    "source",
]

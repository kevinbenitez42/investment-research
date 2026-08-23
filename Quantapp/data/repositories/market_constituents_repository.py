"""Stable accessors for market constituent data."""

from __future__ import annotations

import pandas as pd
import requests

from Quantapp.data.adapters.market_constituents import (
    build_market_data_from_tables,
    normalize_market_index_tables,
)
from Quantapp.data.sources.wikipedia import fetch_wikipedia_market_index_tables


def get_market_index_tables(
    *,
    provider: str = "wikipedia",
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch normalized market-index constituent tables."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "wikipedia":
        raise ValueError(f"Unsupported market constituent provider '{provider}'.")

    raw_tables = fetch_wikipedia_market_index_tables(timeout=timeout, session=session)
    return normalize_market_index_tables(raw_tables)


def get_market_constituent_data(
    *,
    provider: str = "wikipedia",
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch market constituent tables plus S&P sector slices."""
    tables = get_market_index_tables(
        provider=provider,
        timeout=timeout,
        session=session,
    )
    return build_market_data_from_tables(tables)

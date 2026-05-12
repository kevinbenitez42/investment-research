"""Top-level market history interface for Quantapp.data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from Quantapp.data.adapters.market_history import prepare_history_map
from Quantapp.data.sources.yfinance_history import fetch_history_many


def get_market_history(
    *,
    symbols: Iterable[str],
    period: str = "max",
    interval: str = "1d",
    provider: str = "yfinance",
    align: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch, normalize, and optionally align market history for requested symbols."""
    requested_symbols = list(symbols)
    if not requested_symbols:
        return {}

    # Keep the public interface vendor-agnostic even though only yfinance is wired up today.
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "yfinance":
        raise ValueError(f"Unsupported provider '{provider}'.")

    # Source modules fetch raw provider data; adapters convert it into Quantapp shape.
    raw_history = fetch_history_many(
        requested_symbols,
        period=period,
        interval=interval,
    )
    return prepare_history_map(raw_history, align=align)

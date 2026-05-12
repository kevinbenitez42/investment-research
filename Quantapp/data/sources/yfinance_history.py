"""yfinance-backed market history fetch helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _get_yfinance():
    """Import yfinance only when a yfinance-backed call is actually made."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for yfinance-backed market history calls.") from exc
    return yf


def fetch_history(symbol: str, *, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Fetch market history for a single symbol from yfinance."""
    # Clean the symbol once here so every caller gets the same vendor-facing format.
    normalized_symbol = str(symbol).strip()
    if not normalized_symbol:
        raise ValueError("symbol is required.")

    yf = _get_yfinance()
    return yf.Ticker(normalized_symbol).history(period=period, interval=interval)


def fetch_history_many(
    symbols: Iterable[str],
    *,
    period: str = "max",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch market history for multiple symbols from yfinance."""
    # Reuse the single-symbol helper so validation and fetch behavior stay consistent.
    return {
        str(symbol).strip(): fetch_history(symbol, period=period, interval=interval)
        for symbol in symbols
    }


def download_history(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Fetch market history through yfinance.download from the data layer."""
    yf = _get_yfinance()
    return yf.download(*args, **kwargs)


class QuantappTicker:
    """Small yfinance-compatible ticker wrapper for notebook migration."""

    def __init__(self, symbol: str):
        normalized_symbol = str(symbol).strip()
        if not normalized_symbol:
            raise ValueError("symbol is required.")
        self.symbol = normalized_symbol
        yf = _get_yfinance()
        self._ticker = yf.Ticker(normalized_symbol)

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Fetch historical prices through the Quantapp data source helper."""
        return self._ticker.history(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ticker, name)


class QuantappTickers:
    """Small yfinance-compatible multi-ticker wrapper for notebook migration."""

    def __init__(self, tickers: str | Iterable[str]):
        self.tickers = tickers
        yf = _get_yfinance()
        self._tickers = yf.Tickers(tickers)

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self._tickers.history(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tickers, name)


def Ticker(symbol: str) -> QuantappTicker:
    """Return a Quantapp data-layer ticker wrapper."""
    return QuantappTicker(symbol)


def Tickers(tickers: str | Iterable[str]) -> QuantappTickers:
    """Return a Quantapp data-layer multi-ticker wrapper."""
    return QuantappTickers(tickers)

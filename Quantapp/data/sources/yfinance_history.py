"""yfinance-backed market history fetch helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from Quantapp.data.cache import DEFAULT_CACHE_TTL_SECONDS, load_cached_frame, save_cached_frame


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
    """Fetch market history for multiple symbols in one cached batch."""
    normalized_symbols = list(
        dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    )
    if not normalized_symbols:
        return {}

    panel = download_history(
        normalized_symbols,
        period=period,
        interval=interval,
        group_by="column",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if panel.empty:
        return {}

    if len(normalized_symbols) == 1 and not isinstance(panel.columns, pd.MultiIndex):
        return {normalized_symbols[0]: panel.dropna(how="all")}

    if not isinstance(panel.columns, pd.MultiIndex):
        return {}

    ticker_level = None
    for level in range(panel.columns.nlevels):
        level_values = {str(value).upper() for value in panel.columns.get_level_values(level)}
        if level_values.intersection(normalized_symbols):
            ticker_level = level
            break
    if ticker_level is None:
        return {}

    histories = {}
    for symbol in normalized_symbols:
        try:
            history = panel.xs(symbol, axis=1, level=ticker_level).dropna(how="all")
        except KeyError:
            continue
        if not history.empty:
            histories[symbol] = history
    return histories


def download_history(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Fetch market history through yfinance.download from the data layer."""
    cache_ttl_seconds = kwargs.pop("cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS)
    cache_key = repr((args, sorted(kwargs.items(), key=lambda item: item[0])))
    cached = load_cached_frame("yfinance", cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached.copy()
    yf = _get_yfinance()
    result = yf.download(*args, **kwargs)
    save_cached_frame("yfinance", cache_key, result)
    return result


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

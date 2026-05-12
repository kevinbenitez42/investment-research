"""Yfinance-style compatibility facade backed by Quantapp.data sources."""

from __future__ import annotations

from Quantapp.data.sources.yfinance_history import Ticker, Tickers, download_history

download = download_history

__all__ = ["Ticker", "Tickers", "download"]

"""Helpers for benchmark ticker normalization and aligned benchmark loading."""

from __future__ import annotations

import yfinance as yf


def normalize_benchmark_tickers(benchmark_tickers, asset_ticker, *, include_asset=False):
    """Normalize benchmark tickers and remove blanks/duplicates, optionally preserving the asset ticker."""
    normalized = []
    seen = set()
    asset_ticker = str(asset_ticker).strip().upper()
    if benchmark_tickers is None:
        candidates = []
    elif isinstance(benchmark_tickers, str):
        candidates = benchmark_tickers.split(",")
    else:
        candidates = benchmark_tickers

    for symbol in candidates:
        if symbol is None:
            continue
        symbol = str(symbol).strip().upper()
        if not symbol or symbol in seen:
            continue
        if not include_asset and symbol == asset_ticker:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def load_benchmark_data(benchmark_tickers, period, interval, helper):
    """Download benchmark frames and normalize their date indexes."""
    benchmark_frames = {}
    skipped = []
    for symbol in benchmark_tickers:
        frame = yf.Ticker(symbol).history(period=period, interval=interval)
        if frame.empty:
            skipped.append(symbol)
            continue
        benchmark_frames[symbol] = helper.simplify_datetime_index(frame)
    return benchmark_frames, skipped


def align_series_to_common_index(ticker_frame, vix_frame, benchmark_frames):
    """Align asset, VIX, and benchmarks to their shared index."""
    analysis_index = ticker_frame.index.intersection(vix_frame.index)
    for benchmark_frame in benchmark_frames.values():
        analysis_index = analysis_index.intersection(benchmark_frame.index)
    analysis_index = analysis_index.sort_values()
    aligned_benchmarks = {symbol: frame.loc[analysis_index] for symbol, frame in benchmark_frames.items()}
    return analysis_index, ticker_frame.loc[analysis_index], vix_frame.loc[analysis_index], aligned_benchmarks

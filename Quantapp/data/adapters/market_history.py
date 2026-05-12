"""Normalization and alignment helpers for market history frames."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    """Normalize symbols by trimming, uppercasing, and dropping duplicates."""
    normalized_symbols = []
    seen = set()

    for symbol in symbols:
        # Uppercased, de-duplicated symbols become the stable keys used downstream.
        normalized_symbol = str(symbol).strip().upper()
        if not normalized_symbol or normalized_symbol in seen:
            continue
        normalized_symbols.append(normalized_symbol)
        seen.add(normalized_symbol)

    return normalized_symbols


def normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a market history frame to Quantapp conventions."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    normalized_frame = frame.copy()
    if not isinstance(normalized_frame.index, pd.DatetimeIndex):
        normalized_frame.index = pd.to_datetime(normalized_frame.index)

    # Strip timezone/time-of-day detail so histories from different vendors align cleanly.
    if normalized_frame.index.tz is not None:
        normalized_frame.index = normalized_frame.index.tz_convert("UTC").tz_localize(None)

    normalized_frame.index = normalized_frame.index.normalize()
    return normalized_frame.sort_index()


def normalize_history_map(frames_by_symbol: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Normalize the keys and frames in a market history mapping."""
    normalized_frames = {}

    for symbol, frame in frames_by_symbol.items():
        # Normalize the symbol and the frame together so the mapping has consistent semantics.
        normalized_keys = normalize_symbols([symbol])
        if not normalized_keys:
            continue
        normalized_frames[normalized_keys[0]] = normalize_history_frame(frame)

    return normalized_frames


def align_history_map(frames_by_symbol: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align a market history mapping to the shared date index."""
    # Ignore empty frames before intersecting indexes so one bad series does not poison the set.
    populated_frames = {
        symbol: frame
        for symbol, frame in frames_by_symbol.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    if not populated_frames:
        return {}

    frame_iterator = iter(populated_frames.values())
    aligned_index = next(frame_iterator).index
    for frame in frame_iterator:
        aligned_index = aligned_index.intersection(frame.index)

    # Every returned frame is restricted to the same dates for apples-to-apples analysis.
    aligned_index = aligned_index.sort_values()
    return {
        symbol: frame.loc[aligned_index]
        for symbol, frame in populated_frames.items()
    }


def prepare_history_map(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    *,
    align: bool = True,
) -> dict[str, pd.DataFrame]:
    """Normalize and optionally align a market history mapping."""
    # First standardize keys/frames, then optionally collapse everything to a shared index.
    normalized_frames = {
        symbol: frame
        for symbol, frame in normalize_history_map(frames_by_symbol).items()
        if not frame.empty
    }
    if not align:
        return normalized_frames
    return align_history_map(normalized_frames)

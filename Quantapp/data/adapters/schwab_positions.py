"""Schwab position normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from Quantapp.data.schemas.positions import (
    SCHWAB_NET_DIRECTION_COLUMNS,
    SCHWAB_OPTION_POSITION_COLUMNS,
    SCHWAB_OPTION_SYMBOL_PATTERN_TEXT,
)

SCHWAB_OPTION_SYMBOL_PATTERN = re.compile(SCHWAB_OPTION_SYMBOL_PATTERN_TEXT)


def parse_schwab_option_positions(
    positions: Iterable[dict],
    *,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parse Schwab option positions into a normalized option-leg table."""
    option_rows = []
    today_normalized = (
        pd.Timestamp(as_of_date).normalize()
        if as_of_date is not None
        else pd.Timestamp.today().normalize()
    )

    for position in positions:
        if not isinstance(position, dict):
            continue
        instrument = position.get("instrument", {})
        if not isinstance(instrument, dict) or instrument.get("assetType") != "OPTION":
            continue

        position_symbol = "".join((instrument.get("symbol") or "").split())
        match = SCHWAB_OPTION_SYMBOL_PATTERN.match(position_symbol)
        if not match:
            continue

        expiration_code = match.group("expiration")
        expiration_date = pd.to_datetime("20" + expiration_code, format="%Y%m%d", errors="coerce")
        days_to_expiration = (
            int((expiration_date.normalize() - today_normalized).days)
            if pd.notna(expiration_date)
            else pd.NA
        )
        strike_price = int(match.group("strike")) / 1000
        long_quantity = position.get("longQuantity", 0.0)
        short_quantity = position.get("shortQuantity", 0.0)

        option_rows.append(
            {
                "symbol": position_symbol,
                "underlying": match.group("underlying"),
                "expiration": expiration_date,
                "days_to_expiration": days_to_expiration,
                "option_type": match.group("option_type"),
                "strike": strike_price,
                "underlying_symbol": instrument.get("underlyingSymbol"),
                "put_call": instrument.get("putCall"),
                "long_quantity": long_quantity,
                "short_quantity": short_quantity,
                "net_quantity": long_quantity - short_quantity,
                "average_price": position.get("averagePrice", 0.0),
            }
        )

    frame = pd.DataFrame(option_rows)
    for column in SCHWAB_OPTION_POSITION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, [column for column in SCHWAB_OPTION_POSITION_COLUMNS if column in frame.columns]]


def build_schwab_option_direction_summary(
    positions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Calculate option directional value and sentiment by underlying."""
    if positions_df.empty:
        empty_direction = pd.DataFrame(columns=SCHWAB_NET_DIRECTION_COLUMNS)
        return empty_direction, {}

    direction_factor = positions_df["option_type"].map({"C": 1.0, "P": -1.0}).fillna(0.0)
    positions_df["net_quantity"] = positions_df["long_quantity"] - positions_df["short_quantity"]
    positions_df["directional_value"] = (
        positions_df["net_quantity"]
        * positions_df["average_price"]
        * direction_factor
        * 100
    )
    net_direction_series = positions_df.groupby("underlying")["directional_value"].sum()
    sentiment_map = net_direction_series.apply(
        lambda value: "bullish" if value > 0 else ("bearish" if value < 0 else "neutral")
    )
    net_direction = net_direction_series.to_frame("directional_value").join(sentiment_map.rename("sentiment"))
    net_direction["sign"] = net_direction["sentiment"].map({"bullish": 1, "bearish": -1, "neutral": 0})
    return net_direction, sentiment_map.to_dict()


def organize_positions_by_underlying(positions: Iterable[dict]) -> dict[str, list[dict]]:
    """Group raw Schwab positions by instrument underlying symbol."""
    organized_positions: dict[str, list[dict]] = {}
    for position in positions:
        if not isinstance(position, dict):
            continue
        symbol = position.get("instrument", {}).get("underlyingSymbol", "")
        if symbol:
            organized_positions.setdefault(symbol, []).append(position)
    return organized_positions


def calculate_schwab_position_amounts(
    organized_positions: dict[str, list[dict]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate notebook-compatible net invested amounts and margin by underlying."""
    net_invested_amounts = {}
    total_margin = {}

    for symbol, positions_list in organized_positions.items():
        net_sum = 0.0
        for position in positions_list:
            average_price = position.get("averagePrice", 0)
            maintenance_requirement = position.get("maintenanceRequirement", 0) / 100
            put_call = position.get("instrument", {}).get("putCall", "N/A")
            short_quantity = position.get("shortQuantity", 0)
            long_quantity = position.get("longQuantity", 0)
            if put_call == "PUT":
                average_price = -average_price

            net_invested = (long_quantity - short_quantity) * average_price * 100
            maintenance_requirement = maintenance_requirement * 100
            total_margin[symbol] = total_margin.get(symbol, 0) + maintenance_requirement
            net_sum += net_invested

        net_invested_amounts[symbol] = net_sum

    return net_invested_amounts, total_margin


def summarize_schwab_positions(
    positions: Iterable[dict],
    *,
    as_of_date: pd.Timestamp | str | None = None,
) -> dict:
    """Build all normalized position outputs used by the portfolio notebooks."""
    positions_list = list(positions or [])
    option_positions = parse_schwab_option_positions(positions_list, as_of_date=as_of_date)
    net_direction, option_sentiment = build_schwab_option_direction_summary(option_positions)
    organized_positions = organize_positions_by_underlying(positions_list)
    invested_symbols = list(organized_positions.keys())
    net_invested_amounts, total_margin = calculate_schwab_position_amounts(organized_positions)

    return {
        "option_positions": option_positions,
        "option_sentiment": option_sentiment,
        "net_direction": net_direction,
        "organized_positions": organized_positions,
        "invested_symbols": invested_symbols,
        "net_invested_amounts": net_invested_amounts,
        "total_margin": total_margin,
    }

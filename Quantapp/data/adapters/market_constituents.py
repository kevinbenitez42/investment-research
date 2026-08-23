"""Normalize market constituent tables from external providers."""

from __future__ import annotations

import re
import warnings

import pandas as pd


def find_table(
    tables: list[pd.DataFrame],
    *,
    required_columns: list[str] | None = None,
    column_prefixes: list[str] | None = None,
    table_name: str = "table",
) -> pd.DataFrame:
    """Select the first table matching required column names and prefixes."""
    required_columns = required_columns or []
    column_prefixes = column_prefixes or []

    for table in tables:
        table_columns = [str(column) for column in table.columns]
        has_required_columns = all(column in table_columns for column in required_columns)
        has_prefixed_columns = all(
            any(column.startswith(prefix) for column in table_columns)
            for prefix in column_prefixes
        )

        if has_required_columns and has_prefixed_columns:
            return table.copy()

    expected_columns = required_columns + [f"{prefix}*" for prefix in column_prefixes]
    raise ValueError(f"Could not find {table_name} with columns: {expected_columns}")


def normalize_sp500_constituents(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Normalize the Wikipedia S&P 500 constituents table."""
    table = find_table(
        tables,
        required_columns=["Symbol", "GICS Sector", "GICS Sub-Industry"],
        table_name="S&P 500 holdings table",
    )
    return table[["Symbol", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"GICS Sector": "Sector", "GICS Sub-Industry": "Sub-Industry"}
    )


def normalize_nasdaq_100_constituents(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Normalize the Wikipedia NASDAQ-100 constituents table."""
    aliases = {
        "symbol": ("ticker", "ticker symbol", "symbol"),
        "company": ("company", "company name", "security", "name"),
        "sector": ("icb sector", "icb industry", "gics sector", "sector"),
        "sub_industry": (
            "icb subsector",
            "icb sub sector",
            "gics sub industry",
            "gics subindustry",
            "sub industry",
            "subsector",
            "industry",
        ),
    }

    def normalized_column_name(column: object) -> str:
        if isinstance(column, tuple):
            column = " ".join(
                str(part) for part in column if not str(part).startswith("Unnamed")
            )
        value = re.sub(r"\[[^]]*]", "", str(column))
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    for table in tables:
        normalized_columns = {
            column: normalized_column_name(column) for column in table.columns
        }
        selected: dict[str, object] = {}
        used_columns: set[object] = set()
        for field, field_aliases in aliases.items():
            selected[field] = next(
                (
                    column
                    for alias in field_aliases
                    for column, normalized in normalized_columns.items()
                    if column not in used_columns
                    and (normalized == alias or normalized.startswith(f"{alias} "))
                ),
                None,
            )
            if selected[field] is not None:
                used_columns.add(selected[field])

        if all(column is not None for column in selected.values()):
            normalized = table[
                [
                    selected["symbol"],
                    selected["company"],
                    selected["sector"],
                    selected["sub_industry"],
                ]
            ].copy()
            normalized.columns = ["Symbol", "Company", "Sector", "Sub-Industry"]
            normalized["Symbol"] = normalized["Symbol"].astype(str).str.strip()
            return normalized[
                normalized["Symbol"].ne("") & normalized["Symbol"].ne("nan")
            ]

    available_schemas = [list(map(str, table.columns)) for table in tables]
    raise ValueError(
        "Could not find a NASDAQ-100 holdings table with symbol, company, sector, "
        f"and sub-industry columns. Available schemas: {available_schemas}"
    )


def normalize_dow_jones_constituents(
    tables: list[pd.DataFrame],
    *,
    sp500_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalize the Wikipedia Dow Jones Industrial Average constituents table."""
    try:
        table = find_table(
            tables,
            required_columns=["Symbol", "Sector"],
            table_name="Dow Jones holdings table",
        )
        sector_column = "Sector"
    except ValueError:
        table = find_table(
            tables,
            required_columns=["Symbol", "Industry"],
            table_name="Dow Jones holdings table",
        )
        sector_column = "Industry"

    table = table[["Symbol", sector_column]].rename(columns={sector_column: "Sector"})
    if sp500_table is not None and "Sub-Industry" in sp500_table:
        table = pd.merge(table, sp500_table[["Symbol", "Sub-Industry"]], on="Symbol", how="left")
    return table


def normalize_russell_1000_constituents(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Normalize the Wikipedia Russell 1000 constituents table."""
    table = find_table(
        tables,
        required_columns=["Symbol", "GICS Sector", "GICS Sub-Industry"],
        table_name="Russell 1000 holdings table",
    )
    return table[["Symbol", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"GICS Sector": "Sector", "GICS Sub-Industry": "Sub-Industry"}
    )


def normalize_market_index_tables(raw_tables: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Normalize raw market-index table lists into the legacy MarketDataClient shape."""
    sp500_table = normalize_sp500_constituents(raw_tables["SP500"])
    try:
        qqq_table = normalize_nasdaq_100_constituents(raw_tables["NASDAQ_100"])
    except ValueError as exc:
        # Wikipedia occasionally removes or restructures the components table. Keep
        # independently fetched market tables usable while that source is unavailable.
        concise_error = str(exc).partition(" Available schemas:")[0]
        warnings.warn(
            f"NASDAQ-100 constituents are unavailable from Wikipedia: {concise_error}",
            RuntimeWarning,
            stacklevel=2,
        )
        qqq_table = pd.DataFrame(
            columns=["Symbol", "Company", "Sector", "Sub-Industry"]
        )
    dia_table = normalize_dow_jones_constituents(raw_tables["DIA"], sp500_table=sp500_table)
    try:
        russell_1000_table = normalize_russell_1000_constituents(
            raw_tables["Russell_1000"]
        )
    except ValueError as exc:
        warnings.warn(
            f"Russell 1000 constituents are unavailable from Wikipedia: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        russell_1000_table = pd.DataFrame(
            columns=["Symbol", "Sector", "Sub-Industry"]
        )

    return {
        "SP500_TABLE": sp500_table,
        "NASDAQ_100_TABLE": qqq_table,
        "DIA_TABLE": dia_table,
        "Russell_1000_TABLE": russell_1000_table,
    }


def build_market_data_from_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build the legacy MarketDataClient market-data dictionary from normalized tables."""
    sp500_table = tables["SP500_TABLE"]
    qqq_table = tables["NASDAQ_100_TABLE"]
    dia_table = tables["DIA_TABLE"]
    russell_1000_table = tables["Russell_1000_TABLE"]

    return {
        "SP500": sp500_table,
        "NASDAQ_100": qqq_table,
        "DIA": dia_table,
        "Russell_1000": russell_1000_table,
        "Information Technology": sp500_table[sp500_table["Sector"] == "Information Technology"],
        "Financials": sp500_table[sp500_table["Sector"] == "Financials"],
        "Health Care": sp500_table[sp500_table["Sector"] == "Health Care"],
        "Industrials": sp500_table[sp500_table["Sector"] == "Industrials"],
        "Consumer Discretionary": sp500_table[sp500_table["Sector"] == "Consumer Discretionary"],
        "Energy": sp500_table[sp500_table["Sector"] == "Energy"],
        "Materials": sp500_table[sp500_table["Sector"] == "Materials"],
        "Communication Services": sp500_table[sp500_table["Sector"] == "Communication Services"],
        "Real Estate": sp500_table[sp500_table["Sector"] == "Real Estate"],
        "Consumer Staples": sp500_table[sp500_table["Sector"] == "Consumer Staples"],
        "Utilities": sp500_table[sp500_table["Sector"] == "Utilities"],
    }

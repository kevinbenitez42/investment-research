"""Stable portfolio data access interfaces for notebooks and apps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from Quantapp.data.adapters.schwab_positions import summarize_schwab_positions
from Quantapp.data.sources.schwab_accounts import fetch_schwab_account_snapshot


@dataclass
class SchwabPortfolioSnapshot:
    """Schwab account and normalized position data for one selected account."""

    account_information: Any
    account_numbers: Any
    account_hash: str
    account: dict
    raw_positions: list[dict]
    option_positions: pd.DataFrame
    option_sentiment: dict[str, str]
    net_direction: pd.DataFrame
    organized_positions: dict[str, list[dict]]
    invested_symbols: list[str]
    net_invested_amounts: dict[str, float]
    total_margin: dict[str, float]


def get_schwab_portfolio_snapshot(
    client: Any,
    *,
    account_hash: str | None = None,
    account_index: int = 0,
    as_of_date: pd.Timestamp | str | None = None,
    fields: Any | None = None,
) -> SchwabPortfolioSnapshot:
    """Fetch Schwab account positions and return normalized portfolio inputs."""
    snapshot = fetch_schwab_account_snapshot(
        client,
        account_hash=account_hash,
        account_index=account_index,
        fields=fields,
    )
    position_summary = summarize_schwab_positions(
        snapshot["positions"],
        as_of_date=as_of_date,
    )

    return SchwabPortfolioSnapshot(
        account_information=snapshot["account_information"],
        account_numbers=snapshot["account_numbers"],
        account_hash=snapshot["account_hash"],
        account=snapshot["account"],
        raw_positions=snapshot["positions"],
        option_positions=position_summary["option_positions"],
        option_sentiment=position_summary["option_sentiment"],
        net_direction=position_summary["net_direction"],
        organized_positions=position_summary["organized_positions"],
        invested_symbols=position_summary["invested_symbols"],
        net_invested_amounts=position_summary["net_invested_amounts"],
        total_margin=position_summary["total_margin"],
    )

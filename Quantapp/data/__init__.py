"""Shared data access and normalization interfaces for Quantapp."""

from importlib import import_module

from Quantapp.data.benchmark_utils import (
    align_series_to_common_index,
    load_benchmark_data,
    normalize_benchmark_tickers,
)
from Quantapp.data.market_history import get_market_history
from Quantapp.data.gics_peers import (
    GICSPeerFrames,
    build_capitalization_count_table,
    build_gics_peer_frames,
    build_gics_peer_table,
    choose_gics_peer_level,
    normalize_peer_symbol,
    select_gics_peer_rows,
)
from Quantapp.data.sources.fred import get_historical_treasury_yields
from Quantapp.data.repositories.fundamentals_repository import (
    BalanceSheetHistory,
    CashFlowHistory,
    IncomeStatementHistory,
    PeerAnalysisData,
    get_balance_sheet_history,
    get_cash_flow_history,
    get_income_statement_history,
    get_peer_analysis_data,
)
from Quantapp.data.repositories.options_repository import (
    CurrentOptionsChain,
    HistoricalOptionsEodPanel,
    get_current_options_chain,
    get_historical_options_eod_panel,
)
from Quantapp.data.repositories.market_constituents_repository import (
    get_market_constituent_data,
    get_market_index_tables,
)
from Quantapp.data.repositories.portfolio_repository import (
    SchwabPortfolioSnapshot,
    get_schwab_portfolio_snapshot,
)
from . import yf

_LAZY_EXPORTS = {
    "CompanyDataClient": ("Quantapp.data.company_data_client", "CompanyDataClient"),
    "GICSDataClient": ("Quantapp.data.gics_data_client", "GICSDataClient"),
    "MacroDataClient": ("Quantapp.data.macro_data_client", "MacroDataClient"),
    "MarketDataClient": ("Quantapp.data.market_data_client", "MarketDataClient"),
}


def __getattr__(name: str):
    """Load legacy data clients only when callers request them."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute

__all__ = [
    "MacroDataClient",
    "MarketDataClient",
    "CompanyDataClient",
    "BalanceSheetHistory",
    "CashFlowHistory",
    "CurrentOptionsChain",
    "GICSDataClient",
    "GICSPeerFrames",
    "HistoricalOptionsEodPanel",
    "IncomeStatementHistory",
    "PeerAnalysisData",
    "SchwabPortfolioSnapshot",
    "normalize_benchmark_tickers",
    "load_benchmark_data",
    "align_series_to_common_index",
    "get_balance_sheet_history",
    "get_cash_flow_history",
    "get_current_options_chain",
    "get_historical_options_eod_panel",
    "get_historical_treasury_yields",
    "get_income_statement_history",
    "get_market_history",
    "build_capitalization_count_table",
    "build_gics_peer_frames",
    "build_gics_peer_table",
    "choose_gics_peer_level",
    "get_market_constituent_data",
    "get_market_index_tables",
    "get_peer_analysis_data",
    "get_schwab_portfolio_snapshot",
    "normalize_peer_symbol",
    "select_gics_peer_rows",
    "yf",
]

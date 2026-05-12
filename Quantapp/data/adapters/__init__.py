"""Adapters for converting provider data into Quantapp shapes."""

from Quantapp.data.adapters.balance_sheet import (
    normalize_balance_sheet_frame,
    prepare_balance_sheet_frame,
)
from Quantapp.data.adapters.cash_flow import (
    normalize_cash_flow_frame,
    prepare_cash_flow_frame,
)
from Quantapp.data.adapters.income_statement import (
    normalize_income_statement_frame,
    normalize_revenue_segmentation,
    prepare_income_statement_frame,
)
from Quantapp.data.adapters.market_history import (
    align_history_map,
    normalize_history_frame,
    normalize_history_map,
    normalize_symbols,
    prepare_history_map,
)
from Quantapp.data.adapters.market_constituents import (
    build_market_data_from_tables,
    find_table,
    normalize_dow_jones_constituents,
    normalize_market_index_tables,
    normalize_nasdaq_100_constituents,
    normalize_russell_1000_constituents,
    normalize_sp500_constituents,
)
from Quantapp.data.adapters.options import (
    build_historical_eod_row,
    build_options_chain_dicts,
    concat_options_chain_values,
    normalize_massive_snapshot,
    normalize_reference_contracts,
    select_contracts_for_eod_panel,
)
from Quantapp.data.adapters.peers import (
    build_peer_metrics_frame,
    build_relative_value_summary,
    company_context,
    extract_peer_symbols,
    filter_peer_symbols,
    pick_numeric,
    safe_first_record,
)
from Quantapp.data.adapters.schwab_positions import (
    SCHWAB_OPTION_SYMBOL_PATTERN,
    build_schwab_option_direction_summary,
    calculate_schwab_position_amounts,
    organize_positions_by_underlying,
    parse_schwab_option_positions,
    summarize_schwab_positions,
)

__all__ = [
    "align_history_map",
    "build_peer_metrics_frame",
    "build_relative_value_summary",
    "build_historical_eod_row",
    "build_market_data_from_tables",
    "build_options_chain_dicts",
    "company_context",
    "build_schwab_option_direction_summary",
    "calculate_schwab_position_amounts",
    "concat_options_chain_values",
    "extract_peer_symbols",
    "filter_peer_symbols",
    "find_table",
    "normalize_balance_sheet_frame",
    "normalize_cash_flow_frame",
    "normalize_dow_jones_constituents",
    "normalize_history_frame",
    "normalize_history_map",
    "normalize_income_statement_frame",
    "normalize_market_index_tables",
    "normalize_massive_snapshot",
    "normalize_nasdaq_100_constituents",
    "normalize_reference_contracts",
    "normalize_revenue_segmentation",
    "normalize_russell_1000_constituents",
    "normalize_sp500_constituents",
    "organize_positions_by_underlying",
    "parse_schwab_option_positions",
    "pick_numeric",
    "prepare_balance_sheet_frame",
    "prepare_cash_flow_frame",
    "normalize_symbols",
    "prepare_history_map",
    "prepare_income_statement_frame",
    "safe_first_record",
    "SCHWAB_OPTION_SYMBOL_PATTERN",
    "select_contracts_for_eod_panel",
    "summarize_schwab_positions",
]

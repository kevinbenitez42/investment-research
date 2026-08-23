"""Valuation-focused Single Asset Profile visualization views."""

from importlib import import_module

_FAIR_VALUE = ".fair_value"
_FUNDAMENTALS = ".fundamentals"

_LAZY_EXPORTS = {
    "plot_analyst_price_target_band": (_FAIR_VALUE, "plot_analyst_price_target_band"),
    "plot_annual_vs_quarterly_dcf": (_FAIR_VALUE, "plot_annual_vs_quarterly_dcf"),
    "plot_backfilled_dcf_vs_price": (_FAIR_VALUE, "plot_backfilled_dcf_vs_price"),
    "plot_balance_sheet_trends": (_FUNDAMENTALS, "plot_balance_sheet_trends"),
    "plot_cash_conversion_capital_allocation": (_FUNDAMENTALS, "plot_cash_conversion_capital_allocation"),
    "plot_cash_flow_trends": (_FUNDAMENTALS, "plot_cash_flow_trends"),
    "plot_dcf_snapshot_vs_price": (_FAIR_VALUE, "plot_dcf_snapshot_vs_price"),
    "plot_eps_dilution": (_FUNDAMENTALS, "plot_eps_dilution"),
    "plot_gross_margin_drivers": (_FUNDAMENTALS, "plot_gross_margin_drivers"),
    "plot_income_statement_trends": (_FUNDAMENTALS, "plot_income_statement_trends"),
    "plot_liquidity_capital_structure": (_FUNDAMENTALS, "plot_liquidity_capital_structure"),
    "plot_market_based_implied_pricing": (_FAIR_VALUE, "plot_market_based_implied_pricing"),
    "plot_net_margin_drivers": (_FUNDAMENTALS, "plot_net_margin_drivers"),
    "plot_operating_margin_drivers": (_FUNDAMENTALS, "plot_operating_margin_drivers"),
    "plot_price_target_premium_discount": (_FAIR_VALUE, "plot_price_target_premium_discount"),
    "plot_relative_value_vs_peer_medians": (_FUNDAMENTALS, "plot_relative_value_vs_peer_medians"),
    "plot_revenue_segmentation": (_FUNDAMENTALS, "plot_revenue_segmentation"),
    "plot_seasonal_growth_rates": (_FUNDAMENTALS, "plot_seasonal_growth_rates"),
    "plot_ttm_profit_conversion": (_FUNDAMENTALS, "plot_ttm_profit_conversion"),
}

__all__ = [
    "plot_analyst_price_target_band",
    "plot_annual_vs_quarterly_dcf",
    "plot_backfilled_dcf_vs_price",
    "plot_balance_sheet_trends",
    "plot_cash_conversion_capital_allocation",
    "plot_cash_flow_trends",
    "plot_dcf_snapshot_vs_price",
    "plot_eps_dilution",
    "plot_gross_margin_drivers",
    "plot_income_statement_trends",
    "plot_liquidity_capital_structure",
    "plot_market_based_implied_pricing",
    "plot_net_margin_drivers",
    "plot_operating_margin_drivers",
    "plot_price_target_premium_discount",
    "plot_relative_value_vs_peer_medians",
    "plot_revenue_segmentation",
    "plot_seasonal_growth_rates",
    "plot_ttm_profit_conversion",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

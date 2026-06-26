"""Single Asset Profile visualization views."""

from importlib import import_module

_LAZY_EXPORTS = {
    "plot_analyst_price_target_band": (".valuation", "plot_analyst_price_target_band"),
    "plot_annual_vs_quarterly_dcf": (".valuation", "plot_annual_vs_quarterly_dcf"),
    "plot_backfilled_dcf_vs_price": (".valuation", "plot_backfilled_dcf_vs_price"),
    "plot_balance_sheet_trends": (".valuation", "plot_balance_sheet_trends"),
    "plot_cash_conversion_capital_allocation": (".valuation", "plot_cash_conversion_capital_allocation"),
    "plot_cash_flow_trends": (".valuation", "plot_cash_flow_trends"),
    "plot_dcf_snapshot_vs_price": (".valuation", "plot_dcf_snapshot_vs_price"),
    "plot_eps_dilution": (".valuation", "plot_eps_dilution"),
    "plot_gross_margin_drivers": (".valuation", "plot_gross_margin_drivers"),
    "plot_income_statement_trends": (".valuation", "plot_income_statement_trends"),
    "plot_liquidity_capital_structure": (".valuation", "plot_liquidity_capital_structure"),
    "plot_market_based_implied_pricing": (".valuation", "plot_market_based_implied_pricing"),
    "plot_net_margin_drivers": (".valuation", "plot_net_margin_drivers"),
    "plot_operating_margin_drivers": (".valuation", "plot_operating_margin_drivers"),
    "plot_price_target_premium_discount": (".valuation", "plot_price_target_premium_discount"),
    "plot_relative_value_vs_peer_medians": (".valuation", "plot_relative_value_vs_peer_medians"),
    "plot_revenue_segmentation": (".valuation", "plot_revenue_segmentation"),
    "plot_seasonal_growth_rates": (".valuation", "plot_seasonal_growth_rates"),
    "plot_ttm_profit_conversion": (".valuation", "plot_ttm_profit_conversion"),
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

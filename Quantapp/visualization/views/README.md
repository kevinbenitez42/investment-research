# Visualization Views

This folder is for high-level analysis-oriented figure builders.

The target structure mirrors the active `Research/` workflows where the view code is notebook-scoped. That makes each Python view package a prototype for a future React feature/module.

A view should usually own:

- subplot layout
- figure-specific shading and annotations
- dropdowns or time controls unique to that figure
- assembly of traces into a complete chart or dashboard-style view

Organization rule:

- every view group gets its own folder, even if it currently exposes only one public view
- keep one view per module inside that group folder
- use the group package `__init__.py` file to re-export the public view functions

Active mirrored structure:

- `portfolio_profile/performance_structure/`
  Views used by `Research/Portfolio/Portfolio- Performance & Structure.ipynb`.

- `single_asset_profile/pricing/backtesting/`
  Views used by `Research/Single Asset/Pricing/Backtesting.ipynb`.

- `single_asset_profile/pricing/distribution/`
  Views used by `Research/Single Asset/Pricing/Distribution.ipynb`, including trade-range and volatility-model comparison views.

- `single_asset_profile/pricing/factor_analysis/`
  Views used by `Research/Single Asset/Pricing/Factor Analysis.ipynb`.

- `single_asset_profile/pricing/momentum_efficiency/`
  Views used by `Research/Single Asset/Pricing/Momentum & Efficiency.ipynb`.

- `single_asset_profile/pricing/options_pricing/`
  Views used by `Research/Single Asset/Pricing/Options Pricing.ipynb`.

- `single_asset_profile/pricing/predictive_modeling/`
  Views used by `Research/Single Asset/Pricing/Predictive modeling.ipynb`.

- `single_asset_profile/valuation/fair_value/analyst_price_targets_vs_price/`
  Views used by `Research/Single Asset/Valuation/Fair value/Analyst Price Targets vs Price.ipynb`.

- `single_asset_profile/valuation/fair_value/dcf_vs_price/`
  Views used by `Research/Single Asset/Valuation/Fair value/DCF vs Price.ipynb`.

- `single_asset_profile/valuation/fair_value/market_based_pricing/`
  Views used by `Research/Single Asset/Valuation/Fair value/Market based pricing.ipynb`.

- `single_asset_profile/valuation/fundamentals/balance_sheet/`
  Views used by `Research/Single Asset/Valuation/Fundamentals/Balance Sheet.ipynb`.

- `single_asset_profile/valuation/fundamentals/cash_flow_statement/`
  Views used by `Research/Single Asset/Valuation/Fundamentals/Cash flow Statement.ipynb`.

- `single_asset_profile/valuation/fundamentals/income_statement/`
  Views used by `Research/Single Asset/Valuation/Fundamentals/Income Statement.ipynb`.

- `single_asset_profile/valuation/fundamentals/peer_analysis/`
  Views used by `Research/Single Asset/Valuation/Fundamentals/Peer analysis.ipynb`.

This is the preferred destination for notebook extractions and future visualization refactors.

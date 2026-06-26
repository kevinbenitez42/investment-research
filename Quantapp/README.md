# Quantapp

`Quantapp` is the current reusable Python library for this repository.

It sits between exploratory notebooks and future application surfaces, and is where shared research logic should live when it is no longer notebook-specific.

## Package Map

- [`analytics/`](analytics/README.md): return, momentum, volatility, rolling-statistics, and risk-analysis utilities
- [`data/`](data/README.md): market, macro, company, benchmark, and GICS data access helpers
- [`visualization/`](visualization/README.md): Plotly figure builders, helper utilities, and the in-progress `views` migration
- [`models/`](models/README.md): modeling helpers and package-level model abstractions
- [`accounts/`](accounts/README.md): account-related logic
- [`config/`](config/README.md): reserved for shared package configuration

Representative modules:

- [`data/macro_data_client.py`](data/macro_data_client.py): macro and FRED-style financial data client
- [`data/market_data_client.py`](data/market_data_client.py): ETF, index, and broader market dataset access
- [`data/company_data_client.py`](data/company_data_client.py): company-level fundamentals and metadata access
- [`data/gics_data_client.py`](data/gics_data_client.py): GICS structure, company classification, and sector or industry aggregation
- [`analytics/compute.py`](analytics/compute.py): prepared-data metric, rolling-statistic, and transform helpers
- [`visualization/plotter.py`](visualization/plotter.py): legacy chart-type plotting helper for prepared data series

## Working Rule

When logic is:

- exploratory and one-off, it can stay in a notebook
- reused across notebooks, it should move into `Quantapp`
- application-facing, it should be callable from an app layer without notebook assumptions

Layer boundaries:

- `analytics/` owns heavy, reusable, financially meaningful calculations and metric preparation.
- `visualization/` owns lightweight display-only summaries needed for labels, annotations, hover text, axis ranges, and menus.
- notebooks should act as controllers: choose inputs, call analytics/model code, pass results to views, and display figures.

## Current Direction

`Quantapp` is in a transition period:

- the analytics and data layers are already shared
- the visualization layer is being reorganized from plot-type modules toward analysis-oriented views

# Data

This package contains shared data-access helpers and provider-facing clients.

Current focus:

- `market_history.py`: vendor-agnostic entry point for market history retrieval
- `sources/wikipedia.py` and `repositories/market_constituents_repository.py`: Wikipedia-backed market constituent tables
- `sources/databento.py`: Databento historical client setup and futures curve pulls
- `sources/fred.py`: FRED API helpers and Treasury yield history retrieval
- `sources/fmp.py` and `repositories/fundamentals_repository.py`: Financial Modeling Prep-backed fundamentals retrieval
- `repositories/options_repository.py`: portable options-chain and options-EOD retrieval interfaces
- `sources/schwab_accounts.py` and `repositories/portfolio_repository.py`: Schwab account/position retrieval and normalized portfolio inputs
- `yf.py`: temporary yfinance-style compatibility facade for notebooks migrating into `Quantapp.data`
- `sources/`: provider-specific fetchers such as `yfinance`
- `adapters/`: normalization and alignment helpers for provider outputs

This package should stay focused on loading, normalizing, and aligning data. Downstream analytics should live in `Quantapp/analytics`.

## QuickFS Status

QuickFS is deprecated for this project and is being phased out. The legacy QuickFS-backed `CompanyDataClient` remains in place for fallback/reference purposes, but new fundamentals retrieval should use the FMP-backed `sources` / `adapters` / `repositories` path.

## Refactor Scaffold

The data layer is being prepared around these boundaries:

- `sources/`: provider-facing clients that fetch raw external data
- `adapters/`: normalization code that converts provider data into Quantapp shapes
- `schemas/`: shared data contracts, names, and validation rules
- `repositories/`: stable internal access interfaces for normalized data

# Data

This package contains shared data-access helpers and provider-facing clients.

Current areas include:

- `market_data_client.py`: market price/history access
- `macro_data_client.py`: macroeconomic series access
- `company_data_client.py`: company-level data access
- `gics_data_client.py`: sector, industry, and classification access
- `benchmark_utils.py`: alignment and benchmark normalization helpers

This package should stay focused on loading, normalizing, and aligning data. Downstream analytics should live in `Quantapp/analytics`.

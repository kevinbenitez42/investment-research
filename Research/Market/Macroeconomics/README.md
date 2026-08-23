# Macroeconomics

This folder contains global and country-level macroeconomic research notebooks.

Use `_Macroeconomics - Global.ipynb` for broad macro context and the country or region notebooks for focused local series, policy, and market-cycle work.

## Country notebook framework

The non-U.S. country notebooks use a shared configuration-driven framework:

- `Quantapp/data/country_macro.py` defines each country's FRED mappings,
  transformation frequency, FX quote convention, freshness threshold, cycle
  indicator, and country-specific risk series.
- `Quantapp/analytics/macro.py` contains frequency-aware transformations.
- `Quantapp/visualization/macro_dashboard.py` builds the common section charts,
  timeframe controls, and efficient cycle shading.
- FRED responses are fetched concurrently subject to the shared rate limiter and
  cached locally for 12 hours.

Every notebook displays a freshness table and warnings before its charts. A
warning is analytically important: international OECD series distributed by
FRED can lag national releases. Use the named national statistics agency or
central bank as the live-data fallback before making a current policy or asset
allocation conclusion.

Currency performance is normalized so a positive value always represents
local-currency strength against the U.S. dollar, regardless of the vendor's
quote convention.

This product uses the FRED® API but is not endorsed or certified by the Federal
Reserve Bank of St. Louis.

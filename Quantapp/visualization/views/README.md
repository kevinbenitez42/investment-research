# Visualization Views

This folder is for high-level analysis-oriented figure builders.

A view should usually own:

- subplot layout
- figure-specific shading and annotations
- dropdowns or time controls unique to that figure
- assembly of traces into a complete chart or dashboard-style view

Current migration note:

- `volatility.py` contains the first extracted notebook-specific view, `plot_vix_fix_bands`

This is the preferred destination for future extractions from the `Momentum & Efficiency` notebook.

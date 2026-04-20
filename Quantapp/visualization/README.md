# Visualization

This package contains shared Plotly-based visualization code.

## Current Structure

- `plotter.py`: older general-purpose figure builder namespace
- `line_chart_plotter.py`, `bar_chart_plotter.py`, `candlestick_plotter.py`, `heatmap_plotter.py`, `pie_chart_plotter.py`: legacy plot-type-oriented modules
- `figure_helpers.py`: shared figure utility functions
- [`core/`](core/README.md): shared visualization utilities used by multiple views
- [`traces/`](traces/README.md): reusable trace builders and trace bundles
- [`views/`](views/README.md): higher-level analysis-oriented figure builders

## Direction

The package is being reorganized gradually:

- legacy plotter modules remain in place while existing notebooks continue to work
- new extractions should prefer `views/` when the figure represents a specific analysis block
- `core/` and `traces/` should only hold genuinely reusable pieces

In other words, the target organization is closer to "analysis view" than to "chart type" for complex notebook figures.

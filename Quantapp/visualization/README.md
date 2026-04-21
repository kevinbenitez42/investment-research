# Visualization

This package contains shared Plotly-based visualization code.

## Package Responsibilities

- [`core/`](core/README.md)
  Shared visualization infrastructure reused across multiple figures.
  Good fits:
  - axis/range helpers
  - layout/menu helpers
  - theme helpers
  - annotation or figure-level utilities that are not themselves traces

- [`traces/`](traces/README.md)
  Plot-type-oriented trace constructors and small reusable trace groups.
  Good fits:
  - functions returning `go.Scatter`, `go.Candlestick`, `go.Bar`, `go.Surface`, etc.
  - reusable overlays
  - reference traces
  - small trace bundles when the bundle itself is reused

  `traces/` should not own full-figure composition. In particular, trace modules should not be the place that decides subplot structure, page-level layout, dropdown menus, or chart-specific figure orchestration.

- [`views/`](views/README.md)
  Fully composed figures built from traces plus layout code.
  A view is a figure-level composition layer. A module belongs in `views/` when it owns things like:
  - `make_subplots(...)`
  - `update_layout(...)`
  - subplot wiring
  - figure-specific annotations
  - dropdowns, menus, legend behavior, and axis configuration
  - composition of multiple traces into one finished `go.Figure`

  Simply grouping related charts is not enough to make something a view. Under this package structure, a view should return a composed figure, not just collect loosely related plotting helpers.

## Legacy Modules

- `plotter.py`
  Older general-purpose figure builder namespace.

- `line_chart_plotter.py`, `bar_chart_plotter.py`, `candlestick_plotter.py`, `heatmap_plotter.py`, `pie_chart_plotter.py`
  Legacy plot-type-oriented modules that remain in place while notebooks are migrated.
  These can continue to serve as compatibility wrappers, but new extractions should prefer `traces/` plus `views/`.

- `figure_helpers.py`
  Shared figure utilities used by older and newer code. Truly general figure-level helpers may eventually move under `core/`, but this file remains a valid shared utility layer during migration.

## Decision Rules

When adding new visualization code, use this boundary:

1. If the function returns Plotly traces, it belongs in `traces/`.
2. If the function composes a full figure and owns layout/subplot behavior, it belongs in `views/`.
3. If the function is generic figure infrastructure reused by multiple views, it belongs in `core/`.
4. If the code exists mainly to preserve notebook compatibility, it may stay in the legacy top-level modules and delegate to the newer layers.

## Direction

The package is being reorganized gradually:

- legacy plotter modules remain in place while existing notebooks continue to work
- new extractions should prefer `views/` for composed figures
- `traces/` should contain reusable trace construction, not figure assembly
- `core/` should contain reusable figure infrastructure, not analysis-specific logic

In other words, the target organization is:

- `traces/` builds plot layers
- `views/` composes finished figures
- `core/` supports both with shared figure infrastructure

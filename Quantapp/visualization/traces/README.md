# Visualization Traces

This folder is for reusable, shared plotting primitives.

Good fits:

- generic line, bar, heatmap, or candlestick builders
- shared price-axis helpers
- low-level trace constructors reused by more than one view

Not a good fit:

- view-specific semantic traces that only exist for one figure
- one-off wrappers whose only caller is a single view module

If a trace pattern is only used by one figure, it should stay in that view module as a private helper.

# Quantapp

`Quantapp` is the current reusable Python library for this repository.

It sits between exploratory notebooks and future application surfaces, and is where shared research logic should live when it is no longer notebook-specific.

## Package Map

- [`analytics/`](analytics/README.md): return, momentum, volatility, rolling-statistics, and risk-analysis utilities
- [`data/`](data/README.md): market, macro, company, benchmark, and GICS data access helpers
- [`visualization/`](visualization/README.md): Plotly figure builders, helper utilities, and the in-progress `views` migration
- [`workflows/`](workflows/README.md): higher-level assembled flows that orchestrate analytics and visualization together
- [`models/`](models/README.md): modeling helpers and package-level model abstractions
- [`accounts/`](accounts/README.md): account-related logic
- [`config/`](config/README.md): reserved for shared package configuration

## Working Rule

When logic is:

- exploratory and one-off, it can stay in a notebook
- reused across notebooks, it should move into `Quantapp`
- application-facing, it should be callable from a workflow or app layer without notebook assumptions

## Current Direction

`Quantapp` is in a transition period:

- the analytics and data layers are already shared
- the visualization layer is being reorganized from plot-type modules toward analysis-oriented views
- workflows are starting to assemble reusable dashboard payloads outside of notebooks

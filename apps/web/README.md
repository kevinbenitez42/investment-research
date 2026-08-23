# Web App

This folder contains browser and notebook-embedded application entry points.

Intended responsibilities:

- present research and dashboard workflows in a browser UI
- consume the API layer rather than calling data providers directly
- share view models and types with other app surfaces when appropriate

Status:

- `momentum_dashboard.py` contains the executable Momentum & Risk-Adjusted Performance Dash workflow
- its research notebook is intentionally a one-cell inline launcher
- `options_pricing_dashboard.py` contains Options Pricing Blocks 1–15A and its inline tabbed dashboard
- the Options research notebook retains Blocks 16 onward after a single setup/dashboard launcher

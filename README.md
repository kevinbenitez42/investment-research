# Investment Analysis

This project is a personal investment research workspace for studying assets, markets, sectors, and companies.

It combines:
- reusable Python analysis code
- Jupyter notebooks for research
- locally saved company financial data
- dashboard experiments
- older archived projects

## Main Folders

### `Quantapp`
Core Python library for the project.

Main packages:
- `data/`: data-access clients for macro, market, company, and GICS datasets
- `analytics/`: computations such as returns, volatility, spreads, rolling statistics, and factor-style analysis
- `visualization/`: Plotly chart builders for price analysis, seasonality, spreads, heatmaps, and related visualizations
- `models/`: forecasting and model helpers
- `accounts/`: account-related code

### `Notebooks`
Jupyter notebooks used for analysis and exploration.

Current top-level notebook groups:
- `Asset`: asset-level studies such as price and financial statement analysis
- `Market`: market, sector, industry, and macro analysis

This is likely the main place to work when doing research.

### `company_data`
Local storage for company-level data.

Each company usually has:
- annual financial statement data
- quarterly financial statement data
- metadata

This folder acts as a local cache of company fundamentals.

### `Dashboard-Application`
Contains dashboard-related scripts that use the `Quantapp` library to display analysis visually.

### `old projects`
Archived notebooks and older code kept for reference.

These files may contain useful ideas, but they do not appear to be the main current working area.

## Other Files

- `gics_structure.csv`: classification structure for sectors, industries, and sub-industries
- `Project_structure.txt`: raw folder structure listing
- `TODOS`: project notes or unfinished tasks
- `.env`: environment variables and secrets configuration

## Suggested Starting Points

If you are learning the project, start here:
1. Read the notebooks in `Notebooks/Asset` or `Notebooks/Market`
2. Use `Quantapp.data`, `Quantapp.analytics`, and `Quantapp.visualization` to understand how data is loaded, computed, and plotted
3. Check `company_data` to see what local data is already available

## Project Purpose

The purpose of this project is to support personal portfolio and market analysis by combining:
- data collection
- financial and statistical computation
- charting and visualization
- notebook-based research workflows

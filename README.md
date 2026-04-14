# Investment Platform

This repository is an investment research workspace that is being restructured into an application platform.

The long-term application pipeline is:

`Electron -> React -> FastAPI -> PostgreSQL`

That means the project is evolving from a notebook-first research repo into:
- an `Electron` desktop shell
- a `React` frontend
- a `FastAPI` backend
- a PostgreSQL-backed data layer
- shared Python analytics code reused across research and application features

## Project Structure

### `apps/`
Application entry points and runtime surfaces.

- `apps/desktop`: planned `Electron` desktop application shell
- `apps/web`: planned `React` frontend
- `apps/api`: planned `FastAPI` backend and primary application server
- `apps/worker`: planned background jobs, sync tasks, and long-running processing

### `packages/`
Shared code intended to be reused across multiple app layers.

- `packages/python`: future home for reusable Python business logic and analytics packages
- `packages/typescript`: future home for shared frontend types, API clients, and common TypeScript utilities

### `Quantapp`
Current core Python library for the project.

This is the existing reusable analytics layer and is the main source of finance logic that will eventually be migrated or reorganized into the new app structure.

Main packages include:
- `data/`: data-access clients for market, macro, company, and GICS datasets
- `analytics/`: returns, volatility, rolling statistics, momentum, and risk analysis
- `visualization/`: Plotly-based chart builders and helpers
- `models/`: model and forecasting helpers
- `accounts/`: account-related logic
- `workflows/`: higher-level assembled analysis workflows

### `Notebooks`
Research and exploration environment.

Current notebook groups include:
- `Single Asset Profile`
- `Market Profile`
- `Portfolio Profile`
- `Development`
- `research`: reserved for active notebook work under the new structure
- `archived`: reserved for notebook archives under the new structure

The notebooks remain the main place for exploratory analysis while features are being migrated into the application stack.

### `company_data`
Local company-level data store and cache.

This currently holds locally saved company financial data and related metadata. Over time, more of this data is expected to move behind the backend and into PostgreSQL-backed workflows.

### `infra`
Infrastructure-level project support.

This folder is intended for operational setup such as PostgreSQL runtime support, local environment infrastructure, containers, deployment helpers, and other environment-level configuration.

### `docs`
Project notes, migration plans, architecture notes, and other documentation.

### `old projects`
Archived notebooks and older code kept for reference.

These files are retained for ideas and historical context but are not the main target structure going forward.

### `scripts`
Helper scripts for development, cleanup, data maintenance, and local workflows.

### `thinkorswim scripts`
Separate scripts related to thinkorswim studies and trading platform workflows.

## Supporting Files

- `pyproject.toml`: Python package configuration for the current codebase
- `.env`: local environment variables and secrets
- `gics_structure.csv`: classification data for sectors, industries, and sub-industries
- `Project_structure.txt`: older raw structure notes
- `TODOS`: project notes and unfinished work

## Current Direction

The repo currently contains both:
- the existing research workspace
- the early scaffold for the application platform

The migration path is expected to look like this:
1. keep exploratory work in notebooks
2. move reusable logic into shared Python modules
3. expose that logic through `FastAPI`
4. build product screens in `React`
5. package the desktop experience with `Electron`

## Purpose

The purpose of this project is to support investment research and portfolio analysis while gradually becoming a full desktop application built around `Electron`, `React`, `FastAPI`, and `PostgreSQL`.

# Investment Research Workspace

This repository is an investment research workspace that is gradually being reshaped into an application platform.

The long-term direction is:

`Electron -> React -> FastAPI -> shared Python analytics/data layer`

Today, the repo still has a strong notebook-first workflow, but more reusable logic is being moved into `Quantapp` so it can eventually support desktop, web, and API surfaces without notebook-specific assumptions.

## How To Navigate

Start with the README closest to the folder you are working in:

- [`apps/`](apps/README.md): application runtime surfaces
- [`Quantapp/`](Quantapp/README.md): shared Python library
- [`Notebooks/`](Notebooks/README.md): research and exploratory work
- [`company_data/`](company_data/README.md): local ticker-based data cache
- [`scripts/`](scripts/README.md): maintenance and utility scripts
- [`thinkorswim scripts/`](<thinkorswim scripts/README.md>): platform-specific thinkScript files

The older `Project_structure.txt` note has been retired. Folder-level Markdown READMEs are now the source of truth for project organization.

## Repository Map

### [`apps/`](apps/README.md)

Application entry points and runtime shells.

- [`apps/desktop/`](apps/desktop/README.md): active Electron prototype
- [`apps/api/`](apps/api/README.md): reserved for the future FastAPI backend
- [`apps/web/`](apps/web/README.md): reserved for the future browser frontend
- [`apps/worker/`](apps/worker/README.md): reserved for background jobs and batch processing

### [`Quantapp/`](Quantapp/README.md)

Current shared Python library for analytics, data loading, charting, and reusable workflows.

Important subpackages:

- [`analytics/`](Quantapp/analytics/README.md)
- [`data/`](Quantapp/data/README.md)
- [`visualization/`](Quantapp/visualization/README.md)
- [`workflows/`](Quantapp/workflows/README.md)
- [`models/`](Quantapp/models/README.md)
- [`accounts/`](Quantapp/accounts/README.md)
- [`config/`](Quantapp/config/README.md)

### [`Notebooks/`](Notebooks/README.md)

Research and exploratory notebook environment.

Current notebook groups:

- [`Single Asset Profile/`](Notebooks/Single%20Asset%20Profile/README.md)
- [`Market Profile/`](Notebooks/Market%20Profile/README.md)
- [`Portfolio Profile/`](Notebooks/Portfolio%20Profile/README.md)
- [`Development/`](Notebooks/Development/README.md)
- [`_archived/`](Notebooks/_archived/README.md)

### [`company_data/`](company_data/README.md)

Local ticker-based company data cache and working store.

### [`scripts/`](scripts/README.md)

Repository maintenance and helper scripts.

### [`thinkorswim scripts/`](<thinkorswim scripts/README.md>)

Standalone thinkScript studies and platform-specific utilities.

## Supporting Files

- `pyproject.toml`: Python package configuration
- `.env`: local environment variables and secrets
- `gics_structure.csv`: GICS classification source data
- `TODOS.md`: current project notes and follow-up work

## Working Model

The current migration path looks like this:

1. keep exploratory sequencing and one-off analysis in notebooks
2. move reused computation into `Quantapp/analytics` and `Quantapp/data`
3. move reused figure builders into `Quantapp/visualization`
4. assemble stable reusable flows in `Quantapp/workflows`
5. expose those flows through application surfaces in `apps/`

## Purpose

The goal of this repository is to support investment research and portfolio analysis while steadily evolving into a more durable application platform built on shared Python logic and dedicated runtime surfaces.

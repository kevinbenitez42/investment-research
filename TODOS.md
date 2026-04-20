Top priority for a scalable web app

- define your FastAPI service and core endpoints
- make the backend the single source of truth for data access and business logic
- standardize data access patterns in Quantapp/data
- scaffold apps/web as a React frontend connected to the API
- add tests and lint/type checks for Python and TypeScript
- document the architecture, API schema, and shared package usage in README.md

Current cleanup and organization

- decide whether `Quantapp` remains the main shared analytics library or should be split into app-specific modules
- continue extracting notebook-specific visualization blocks from `Notebooks/Single Asset Profile/Pricing/Momentum & Efficiency.ipynb` into `Quantapp/visualization/views`, using Block 7 as the pilot and then working through Blocks 8-15
- clean up notebooks and archive obsolete research files when core app structure is stable
- refactor `Quantapp/data` so source-specific providers are separated from shared data helpers
- replace deprecated QuickFS data flows with Financial Modeling Prep, Databento, and Massive providers

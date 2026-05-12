# Research

This folder is the experimental execution environment for investment analysis, Plotly visualization exploration, and prototype UI work that may later inform the front-end app.

The code here is allowed to be notebook-shaped and research-driven. Stable or reused logic should eventually move into `Quantapp`, while polished application surfaces should live under `apps/`.

Current research groups:

- [`Single Asset Profile/`](Single%20Asset%20Profile/README.md): asset-specific pricing and valuation work
- [`Market Profile/`](Market%20Profile/README.md): cross-market, sector, asset-class, and macro work
- [`Portfolio Profile/`](Portfolio%20Profile/README.md): reserved for portfolio-level notebook work
- [`Development/`](Development/README.md): scratch, debugging, and migration work
- [`_archived/`](./_archived/README.md): older notebooks kept for reference

Working rule:

- keep exploratory sequencing, narrative, Plotly experiments, and UI prototypes here
- move reused computation into `Quantapp/analytics` or `Quantapp/data`
- move reused figure builders and shared view logic into `Quantapp/visualization`

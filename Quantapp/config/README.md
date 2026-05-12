# Config

This package is reserved for shared `Quantapp` configuration.

Potential future uses:

- provider configuration
- package-wide constants
- environment-aware settings helpers
- shared application defaults consumed by workflows

Status:

- currently empty

Provider note:

- QuickFS is deprecated for this project and is being phased out. Legacy QuickFS code may remain for fallback/reference, but new fundamentals retrieval should use the FMP-backed `Quantapp.data` interfaces.

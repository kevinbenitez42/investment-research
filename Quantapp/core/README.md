# Core

This package contains reusable application infrastructure used across Quantapp.

Core code should stay independent of finance and investing domain logic. It is the place for shared configuration, database/session infrastructure, caching interfaces and generic implementations, logging setup, general utilities, and common exceptions or base interfaces.

Suggested boundaries:

- `config/`: application configuration and environment handling
- `db/`: database connection and session infrastructure
- `cache/`: caching interfaces and generic cache implementations
- `logging/`: application logging setup
- `utils/`: shared utilities that are not finance-specific
- `exceptions/`: common exceptions and error base classes


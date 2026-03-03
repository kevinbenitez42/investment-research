"""Helpers for loading local project secrets from the repo .env file."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the nearest parent directory that looks like the project root."""
    current = Path(start_path or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    package_root = Path(__file__).resolve().parents[1]

    if package_root not in candidates:
        candidates.append(package_root)

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return package_root


def load_project_env(
    env_path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load key/value pairs from the repo .env into ``os.environ``."""
    resolved_env_path = Path(env_path).expanduser().resolve() if env_path else find_project_root() / ".env"

    if not resolved_env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in resolved_env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value

    return loaded


def require_secret(name: str, env_path: str | os.PathLike[str] | None = None) -> str:
    """Return a required secret from the repo .env file."""
    load_project_env(env_path)
    value = os.getenv(name)
    if value:
        return value

    raise KeyError(f"Missing required secret '{name}' in the project .env file.")

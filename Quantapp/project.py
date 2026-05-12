"""Project-location helpers for notebooks and scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start_path: str | os.PathLike[str] | None = None) -> Path:
    """Return the nearest parent directory containing the Quantapp project."""
    current = Path(start_path or Path.cwd()).expanduser().resolve()
    candidates = [current, *current.parents]
    package_root = Path(__file__).resolve().parents[1]

    if package_root not in candidates:
        candidates.append(package_root)

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "Quantapp").is_dir():
            return candidate

    for candidate in candidates:
        if (candidate / "Quantapp").is_dir():
            return candidate

    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return package_root


def ensure_project_root_on_path(start_path: str | os.PathLike[str] | None = None) -> Path:
    """Add the project root to ``sys.path`` and return it."""
    project_root = find_project_root(start_path)
    root_text = str(project_root)

    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    return project_root

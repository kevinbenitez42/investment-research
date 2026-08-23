"""Small local time-to-live cache for downloaded DataFrames."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pickle
import sys
import threading
import time

import numpy as np
import pandas as pd


DEFAULT_CACHE_TTL_SECONDS = 12 * 60 * 60
CACHE_SCHEMA_VERSION = 2


def _runtime_cache_fingerprint() -> str:
    """Separate pickle caches that cannot safely share serialized objects."""
    return (
        f"schema={CACHE_SCHEMA_VERSION}|"
        f"python={sys.version_info.major}.{sys.version_info.minor}|"
        f"numpy={np.__version__}|pandas={pd.__version__}"
    )


def _cache_path(namespace: str, key: str) -> Path:
    root = Path(os.getenv("QUANTAPP_CACHE_DIR", Path.home() / ".cache" / "quantapp"))
    compatible_key = f"{_runtime_cache_fingerprint()}|{key}"
    digest = hashlib.sha256(compatible_key.encode("utf-8")).hexdigest()
    return root / namespace / f"{digest}.pkl"


def load_cached_frame(namespace: str, key: str, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
    path = _cache_path(namespace, key)
    try:
        if time.time() - path.stat().st_mtime >= ttl_seconds:
            return None
        return pd.read_pickle(path)
    except (FileNotFoundError, OSError):
        return None
    except (ImportError, AttributeError, ValueError, EOFError, pickle.UnpicklingError):
        # A cache can outlive the Python/NumPy/pandas environment that wrote it.
        # Remove only this derived artifact so the caller can fetch and replace it.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def save_cached_frame(namespace: str, key: str, frame: pd.DataFrame) -> None:
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    frame.to_pickle(temporary)
    os.replace(temporary, path)

"""Databento-backed market data fetch helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

import pandas as pd

from Quantapp.secrets import load_project_env

DEFAULT_DATABENTO_DATASET = "GLBX.MDP3"
DEFAULT_DATABENTO_SCHEMA = "ohlcv-1d"
DATABENTO_API_KEY_ENV_NAMES = ("DATABENTO_API_KEY",)
FUTURES_MONTH_CYCLE = (
    (1, "F"),
    (2, "G"),
    (3, "H"),
    (4, "J"),
    (5, "K"),
    (6, "M"),
    (7, "N"),
    (8, "Q"),
    (9, "U"),
    (10, "V"),
    (11, "X"),
    (12, "Z"),
)


@dataclass(frozen=True)
class DatabentoFuturesCurvePull:
    """Result from one Databento futures-curve history pull."""

    data: pd.DataFrame
    symbols: list[str]
    start_date: date
    end_date: date
    dataset: str
    schema: str

    @property
    def symbol_column(self) -> str | None:
        """Return the symbol column used by Databento's DataFrame, if present."""
        return get_databento_symbol_column(self.data)

    @property
    def returned_symbols(self) -> list[str]:
        """Return sorted symbols present in the retrieved data."""
        symbol_col = self.symbol_column
        if not symbol_col:
            return []

        frame = self.data.reset_index()
        return sorted(set(frame[symbol_col].astype(str)))

    @property
    def missing_symbols(self) -> list[str]:
        """Return requested symbols with no rows in the retrieved data."""
        return sorted(set(self.symbols) - set(self.returned_symbols))


def resolve_databento_api_key(
    api_key: str | None = None,
    *,
    use_keyring: bool = True,
    keyring_service: str = "databento",
    keyring_username: str = "default",
) -> str | None:
    """Resolve a Databento API key from an explicit value, env vars, or keyring."""
    if api_key:
        resolved_api_key = api_key.strip()
        if resolved_api_key:
            return resolved_api_key

    load_project_env()
    for env_name in DATABENTO_API_KEY_ENV_NAMES:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value.strip()

    if not use_keyring:
        return None

    try:
        import keyring
        keyring_value = keyring.get_password(keyring_service, keyring_username)
    except Exception:
        return None

    return keyring_value.strip() if keyring_value else None


def get_databento_historical_client(api_key: str | None = None):
    """Create a Databento historical client."""
    try:
        import databento as db
    except ImportError as exc:
        raise ImportError(
            "databento is required for Databento-backed data calls. "
            "Install it with `pip install databento`."
        ) from exc

    resolved_api_key = resolve_databento_api_key(api_key)
    if not resolved_api_key:
        raise ValueError(
            "Missing Databento API key. Set DATABENTO_API_KEY in your project .env "
            "or save it to keyring under service='databento', username='default'."
        )

    os.environ["DATABENTO_API_KEY"] = resolved_api_key
    return db.Historical(resolved_api_key)


def build_futures_contract_symbols(
    root: str,
    *,
    n_contracts: int,
    as_of: date | datetime | str | None = None,
    month_cycle: Sequence[tuple[int, str]] = FUTURES_MONTH_CYCLE,
) -> list[str]:
    """Build raw Databento futures symbols for the next ``n_contracts`` months."""
    if n_contracts < 1:
        raise ValueError("n_contracts must be at least 1.")

    as_of_date = _coerce_date(as_of) if as_of else date.today()
    normalized_root = str(root).strip().upper()
    if not normalized_root:
        raise ValueError("root must be a non-empty futures root symbol.")

    start_idx = 0
    for idx, (month_num, _) in enumerate(month_cycle):
        if as_of_date.month <= month_num:
            start_idx = idx
            break

    symbols: list[str] = []
    year = as_of_date.year
    idx = start_idx
    for _ in range(n_contracts):
        _, month_code = month_cycle[idx]
        yy = year % 10
        symbols.append(f"{normalized_root}{month_code}{yy}")

        idx += 1
        if idx >= len(month_cycle):
            idx = 0
            year += 1

    return symbols


def fetch_databento_timeseries(
    symbols: Iterable[str],
    *,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    schema: str = DEFAULT_DATABENTO_SCHEMA,
    start: date | datetime | str,
    end: date | datetime | str,
    stype_in: str = "raw_symbol",
    client=None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch Databento timeseries data and return the provider DataFrame."""
    requested_symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if not requested_symbols:
        raise ValueError("At least one Databento symbol is required.")

    resolved_client = client or get_databento_historical_client(api_key)
    bars = resolved_client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=requested_symbols,
        stype_in=stype_in,
        start=_coerce_date(start).isoformat(),
        end=_coerce_date(end).isoformat(),
    )
    return bars.to_df()


def fetch_databento_futures_curve_bars(
    root: str = "CL",
    *,
    n_contracts: int = 24,
    days_back: int = 60,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    schema: str = DEFAULT_DATABENTO_SCHEMA,
    as_of: date | datetime | str | None = None,
    max_contracts: int = 24,
    max_days_back: int = 93,
    client=None,
    api_key: str | None = None,
) -> DatabentoFuturesCurvePull:
    """Fetch daily bars for a raw-symbol futures curve in a single Databento query."""
    if n_contracts > max_contracts:
        raise ValueError(f"Cost guard: keep n_contracts <= {max_contracts}.")
    if days_back > max_days_back:
        raise ValueError(f"Cost guard: keep days_back <= {max_days_back}.")
    if days_back < 1:
        raise ValueError("days_back must be at least 1.")

    end_date = _coerce_date(as_of) if as_of else date.today()
    start_date = end_date - timedelta(days=days_back)
    symbols = build_futures_contract_symbols(root, n_contracts=n_contracts, as_of=end_date)
    data = fetch_databento_timeseries(
        symbols,
        dataset=dataset,
        schema=schema,
        start=start_date,
        end=end_date,
        client=client,
        api_key=api_key,
    )

    return DatabentoFuturesCurvePull(
        data=data,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        dataset=dataset,
        schema=schema,
    )


def get_databento_symbol_column(frame: pd.DataFrame) -> str | None:
    """Return the provider symbol column used by a Databento DataFrame."""
    columns = set(frame.reset_index().columns)
    if "symbol" in columns:
        return "symbol"
    if "raw_symbol" in columns:
        return "raw_symbol"
    return None


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))

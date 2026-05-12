"""FRED-backed macro data fetch helpers."""

from __future__ import annotations

import datetime as dt
import os
from urllib.parse import urlencode

import pandas as pd
import requests

from Quantapp.secrets import load_project_env

DEFAULT_FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV_NAMES = ("FRED_API_KEY",)
FRED_TREASURY_YIELD_SERIES_IDS = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}
FRED_REAL_TREASURY_YIELD_SERIES_IDS = {
    "5Y": "DFII5",
    "7Y": "DFII7",
    "10Y": "DFII10",
    "20Y": "DFII20",
    "30Y": "DFII30",
}


def resolve_fred_api_key(api_key: str | None = None) -> str | None:
    """Resolve a FRED API key from an explicit value or supported env vars."""
    if api_key:
        resolved_api_key = api_key.strip()
        if resolved_api_key:
            return resolved_api_key

    load_project_env()
    for env_name in FRED_API_KEY_ENV_NAMES:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value.strip()
    return None


def fred_api_url(
    path: str,
    *,
    base_url: str | None = None,
    params: dict | None = None,
) -> str:
    """Build a FRED API URL from an endpoint path and query parameters."""
    resolved_base_url = (base_url or os.getenv("FRED_API_BASE_URL") or DEFAULT_FRED_API_BASE_URL).rstrip("/")
    endpoint = str(path).lstrip("/")
    query = urlencode(params or {})
    return f"{resolved_base_url}/{endpoint}" + (f"?{query}" if query else "")


def fetch_fred_observations(
    series_id: str,
    *,
    api_key: str | None = None,
    start_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one FRED observations series as a date-indexed DataFrame."""
    resolved_api_key = resolve_fred_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("Missing FRED API key. Set FRED_API_KEY in your project .env.")

    params = {
        "series_id": str(series_id).strip().upper(),
        "api_key": resolved_api_key,
        "file_type": "json",
    }
    if start_date is not None:
        params["observation_start"] = format_fred_date(start_date)
    if end_date is not None:
        params["observation_end"] = format_fred_date(end_date)

    http = session or requests
    response = http.get(
        fred_api_url("series/observations", base_url=base_url, params=params),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error_message"):
        raise RuntimeError(payload["error_message"])

    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="date"))

    frame = pd.DataFrame(observations)
    value = pd.to_numeric(frame["value"], errors="coerce")
    index = pd.to_datetime(frame["date"])
    return pd.DataFrame({"value": value.values}, index=index)


def fetch_fred_observations_query(
    query: str,
    *,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a fully-built FRED observations query URL as a date-indexed DataFrame."""
    http = session or requests
    response = http.get(query, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error_message"):
        raise RuntimeError(payload["error_message"])

    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame(columns=["value"], index=pd.DatetimeIndex([], name="date"))

    frame = pd.DataFrame(observations).drop(columns=["realtime_start", "realtime_end"], errors="ignore")
    value = pd.to_numeric(frame["value"], errors="coerce")
    index = pd.to_datetime(frame["date"])
    return pd.DataFrame({"value": value.values}, index=index)


def fetch_fred_series_frame(
    series_ids: dict[str, str],
    *,
    api_key: str | None = None,
    start_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch multiple FRED series into one DataFrame keyed by display name."""
    resolved_api_key = resolve_fred_api_key(api_key)
    frames = [
        fetch_fred_observations(
            series_id,
            api_key=resolved_api_key,
            start_date=start_date,
            end_date=end_date,
            base_url=base_url,
            timeout=timeout,
            session=session,
        ).rename(columns={"value": name})
        for name, series_id in series_ids.items()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def fetch_historical_treasury_yields(
    maturities: str | list[str] | tuple[str, ...] | None = None,
    *,
    api_key: str | None = None,
    start_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    real: bool = False,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch historical US Treasury constant maturity yields from FRED."""
    series_ids = FRED_REAL_TREASURY_YIELD_SERIES_IDS if real else FRED_TREASURY_YIELD_SERIES_IDS
    selected_series = select_fred_treasury_series(series_ids, maturities)
    return fetch_fred_series_frame(
        selected_series,
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )


def get_historical_treasury_yields(
    maturities: str | list[str] | tuple[str, ...] | None = None,
    *,
    api_key: str | None = None,
    start_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    real: bool = False,
) -> pd.DataFrame:
    """Stable data-package entry point for Treasury yield history."""
    return fetch_historical_treasury_yields(
        maturities=maturities,
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
        real=real,
    )


def select_fred_treasury_series(
    series_ids: dict[str, str],
    maturities: str | list[str] | tuple[str, ...] | None,
) -> dict[str, str]:
    """Filter a Treasury maturity map to the requested maturities."""
    if maturities is None:
        return dict(series_ids)
    if isinstance(maturities, str):
        maturities = [maturities]

    normalized_maturities = [str(maturity).strip().upper() for maturity in maturities]
    unsupported = [maturity for maturity in normalized_maturities if maturity not in series_ids]
    if unsupported:
        available = ", ".join(series_ids.keys())
        requested = ", ".join(unsupported)
        raise ValueError(f"Unsupported Treasury maturities: {requested}. Available maturities: {available}.")

    return {maturity: series_ids[maturity] for maturity in normalized_maturities}


def format_fred_date(value: str | dt.date | dt.datetime | pd.Timestamp) -> str:
    """Format a FRED-compatible date value as YYYY-MM-DD."""
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime("%Y-%m-%d")
    raise TypeError("Dates must be strings or datetime-like values.")

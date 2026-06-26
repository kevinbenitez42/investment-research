"""Massive-backed options data fetch helpers."""

from __future__ import annotations

import os
from urllib.parse import quote

import requests

DEFAULT_MASSIVE_API_BASE_URL = "https://api.massive.com"
MASSIVE_API_KEY_ENV_NAMES = (
    "MASSIVE_API_KEY",
    "POLYGON_API_KEY",
    "API_POLYGON",
    "POLYGON_KEY",
)


def resolve_massive_api_key(api_key: str | None = None) -> str | None:
    """Resolve a Massive API key from an explicit value or supported env vars."""
    if api_key:
        return api_key

    for env_name in MASSIVE_API_KEY_ENV_NAMES:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return None


def massive_api_url(path_or_url: str, *, base_url: str | None = None) -> str:
    """Build a Massive API URL from a path, preserving absolute URLs."""
    url = str(path_or_url)
    if url.startswith(("http://", "https://")):
        return url

    resolved_base_url = (base_url or os.getenv("MASSIVE_API_BASE_URL") or DEFAULT_MASSIVE_API_BASE_URL).rstrip("/")
    return f"{resolved_base_url}/{url.lstrip('/')}"


def massive_request_json(
    path_or_url: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    params: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict:
    """Fetch a Massive REST endpoint and return its decoded JSON payload."""
    resolved_api_key = resolve_massive_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("Missing Massive API key. Set MASSIVE_API_KEY in your project .env.")

    request_params = dict(params or {})
    request_params.setdefault("apiKey", resolved_api_key)

    http = session or requests
    response = http.get(
        massive_api_url(path_or_url, base_url=base_url),
        params=request_params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_massive_paginated(
    path_or_url: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    initial_params: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch all pages from a Massive list endpoint."""
    records = []
    next_url = path_or_url
    first_request = True

    while next_url:
        payload = massive_request_json(
            next_url,
            api_key=api_key,
            base_url=base_url,
            params=initial_params if first_request else None,
            timeout=timeout,
            session=session,
        )
        records.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        first_request = False

    return records


def fetch_options_chain_snapshot(
    underlying_ticker: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    limit: int = 250,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch the current options chain snapshot for an underlying ticker."""
    return fetch_massive_paginated(
        f"/v3/snapshot/options/{str(underlying_ticker).strip().upper()}",
        api_key=api_key,
        base_url=base_url,
        initial_params={"limit": limit},
        timeout=timeout,
        session=session,
    )


def fetch_reference_options_contracts(
    underlying_ticker: str,
    as_of_date: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    expired: bool = False,
    limit: int = 1000,
    timeout: int = 20,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch one page of options contract reference metadata for a date."""
    payload = massive_request_json(
        "/v3/reference/options/contracts",
        api_key=api_key,
        base_url=base_url,
        params={
            "underlying_ticker": str(underlying_ticker).strip().upper(),
            "as_of": as_of_date,
            "expired": str(expired).lower(),
            "limit": limit,
            "sort": "expiration_date",
            "order": "asc",
        },
        timeout=timeout,
        session=session,
    )
    return payload.get("results", [])


def fetch_reference_options_contracts_paginated(
    underlying_ticker: str,
    as_of_date: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    expired: bool = False,
    expiration_date_gte: str | None = None,
    expiration_date_lte: str | None = None,
    strike_price_gte: float | None = None,
    strike_price_lte: float | None = None,
    limit: int = 1000,
    timeout: int = 20,
    session: requests.Session | None = None,
) -> tuple[list[dict], int]:
    """Fetch every reference-contract page and report the exact request count."""
    params = {
        "underlying_ticker": str(underlying_ticker).strip().upper(),
        "as_of": as_of_date,
        "expired": str(expired).lower(),
        "limit": int(limit),
        "sort": "expiration_date",
        "order": "asc",
    }
    optional_filters = {
        "expiration_date.gte": expiration_date_gte,
        "expiration_date.lte": expiration_date_lte,
        "strike_price.gte": strike_price_gte,
        "strike_price.lte": strike_price_lte,
    }
    params.update(
        {
            filter_name: filter_value
            for filter_name, filter_value in optional_filters.items()
            if filter_value is not None
        }
    )

    records = []
    request_count = 0
    next_url = "/v3/reference/options/contracts"
    first_request = True
    while next_url:
        request_count += 1
        payload = massive_request_json(
            next_url,
            api_key=api_key,
            base_url=base_url,
            params=params if first_request else None,
            timeout=timeout,
            session=session,
        )
        records.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        first_request = False
    return records, request_count


def fetch_option_daily_bar(
    option_ticker: str,
    as_of_date: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    adjusted: bool = True,
    timeout: int = 15,
    session: requests.Session | None = None,
) -> dict | None:
    """Fetch the EOD aggregate bar for one option contract on one date."""
    safe_ticker = quote(str(option_ticker).strip(), safe=":")
    payload = massive_request_json(
        f"/v2/aggs/ticker/{safe_ticker}/range/1/day/{as_of_date}/{as_of_date}",
        api_key=api_key,
        base_url=base_url,
        params={"adjusted": str(adjusted).lower()},
        timeout=timeout,
        session=session,
    )
    results = payload.get("results", [])
    return results[0] if results else None


def fetch_option_daily_bars(
    option_ticker: str,
    start_date: str,
    end_date: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    adjusted: bool = True,
    limit: int = 50000,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch a contract's complete daily-bar range in one aggregate request."""
    safe_ticker = quote(str(option_ticker).strip(), safe=":")
    payload = massive_request_json(
        f"/v2/aggs/ticker/{safe_ticker}/range/1/day/{start_date}/{end_date}",
        api_key=api_key,
        base_url=base_url,
        params={
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": int(limit),
        },
        timeout=timeout,
        session=session,
    )
    return payload.get("results", [])

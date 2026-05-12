"""Financial Modeling Prep REST fetch helpers."""

from __future__ import annotations

import os

import requests

from Quantapp.secrets import load_project_env

DEFAULT_FMP_API_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_API_KEY_ENV_NAMES = (
    "FMP_API_KEY",
    "FINANCIAL_MODELING_PREP_API_KEY",
)


def resolve_fmp_api_key(api_key: str | None = None) -> str | None:
    """Resolve an FMP API key from an explicit value or supported env vars."""
    if api_key:
        return api_key

    load_project_env()
    for env_name in FMP_API_KEY_ENV_NAMES:
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    return None


def normalize_fmp_symbol(ticker: str) -> str:
    """Normalize a ticker symbol to FMP's common dot-delimited convention."""
    normalized = str(ticker).strip().upper()
    for old, new in (("\\", "."), ("/", "."), ("-", ".")):
        normalized = normalized.replace(old, new)
    return normalized


def fmp_api_url(endpoint_or_url: str, *, base_url: str | None = None) -> str:
    """Build an FMP URL from an endpoint, preserving absolute URLs."""
    url = str(endpoint_or_url)
    if url.startswith(("http://", "https://")):
        return url

    resolved_base_url = (base_url or os.getenv("FMP_API_BASE_URL") or DEFAULT_FMP_API_BASE_URL).rstrip("/")
    return f"{resolved_base_url}/{url.lstrip('/')}"


def fmp_request_json(
    endpoint_or_url: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    params: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
    **query_params,
):
    """Fetch an FMP endpoint and return its decoded JSON payload."""
    resolved_api_key = resolve_fmp_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("Missing FMP API key. Set FMP_API_KEY in your project .env.")

    request_params = dict(params or {})
    request_params.update(query_params)
    request_params.setdefault("apikey", resolved_api_key)

    http = session or requests
    response = http.get(
        fmp_api_url(endpoint_or_url, base_url=base_url),
        params=request_params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(payload["Error Message"])
    return payload


def fetch_fmp_quote(
    symbol: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch the current FMP quote snapshot for one symbol."""
    payload = fmp_request_json(
        "quote",
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
    )
    return payload if isinstance(payload, list) else []


def fetch_fmp_profile(
    symbol: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch the FMP company profile snapshot for one symbol."""
    payload = fmp_request_json(
        "profile",
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
    )
    return payload if isinstance(payload, list) else []


def fetch_fmp_statement(
    endpoint: str,
    symbol: str,
    *,
    period: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch one FMP financial statement endpoint for one symbol."""
    params: dict[str, object] = {"symbol": normalize_fmp_symbol(symbol)}
    if period:
        params["period"] = period
    if limit is not None:
        params["limit"] = limit

    payload = fmp_request_json(
        endpoint,
        api_key=api_key,
        base_url=base_url,
        params=params,
        timeout=timeout,
        session=session,
    )
    return payload if isinstance(payload, list) else []


def fetch_fmp_balance_sheet_statement(
    symbol: str,
    *,
    period: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP balance sheet statement rows for one symbol."""
    return fetch_fmp_statement(
        "balance-sheet-statement",
        symbol,
        period=period,
        limit=limit,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )


def fetch_fmp_income_statement(
    symbol: str,
    *,
    period: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP income statement rows for one symbol."""
    return fetch_fmp_statement(
        "income-statement",
        symbol,
        period=period,
        limit=limit,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )


def fetch_fmp_cash_flow_statement(
    symbol: str,
    *,
    period: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP cash flow statement rows for one symbol."""
    return fetch_fmp_statement(
        "cash-flow-statement",
        symbol,
        period=period,
        limit=limit,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )


def fetch_fmp_stock_peers(
    symbol: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
):
    """Fetch the FMP stock-peers payload for one symbol."""
    return fmp_request_json(
        "stock-peers",
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
    )


def fetch_fmp_ratios_ttm(
    symbol: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP trailing-twelve-month ratio metrics for one symbol."""
    payload = fmp_request_json(
        "ratios-ttm",
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
    )
    return payload if isinstance(payload, list) else []


def fetch_fmp_key_metrics_ttm(
    symbol: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP trailing-twelve-month key metrics for one symbol."""
    payload = fmp_request_json(
        "key-metrics-ttm",
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
    )
    return payload if isinstance(payload, list) else []


def fetch_fmp_revenue_segmentation(
    symbol: str,
    *,
    endpoint: str,
    period: str = "annual",
    structure: str = "flat",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch FMP revenue segmentation rows."""
    payload = fmp_request_json(
        endpoint,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
        symbol=normalize_fmp_symbol(symbol),
        period=period,
        structure=structure,
    )
    return payload if isinstance(payload, list) else []

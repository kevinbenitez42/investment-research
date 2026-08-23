"""World Bank Indicators API data helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pandas as pd
import requests

from Quantapp.data.cache import DEFAULT_CACHE_TTL_SECONDS, load_cached_frame, save_cached_frame


WORLD_BANK_API_URL = "https://api.worldbank.org/v2"
NOMINAL_GDP_CURRENT_USD = "NY.GDP.MKTP.CD"


def fetch_world_bank_indicator(
    countries: Mapping[str, str],
    indicator: str,
    *,
    start_year: int = 1960,
    end_year: int | None = None,
    timeout: int = 30,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one annual indicator for multiple economies in a single request.

    ``countries`` maps World Bank economy codes to desired display names.
    """
    if not countries:
        return pd.DataFrame()
    end_year = end_year or date.today().year
    codes = tuple(dict.fromkeys(code.strip().upper() for code in countries))
    cache_key = f"{indicator}|{';'.join(codes)}|{start_year}|{end_year}"
    cached = load_cached_frame("world_bank", cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached.copy()

    http = session or requests
    url = f"{WORLD_BANK_API_URL}/country/{';'.join(codes)}/indicator/{indicator}"
    response = http.get(
        url,
        params={
            "format": "json",
            "date": f"{start_year}:{end_year}",
            "per_page": max(1000, len(codes) * (end_year - start_year + 1)),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("World Bank returned an unexpected indicator response")

    rows = payload[1] or []
    records = [
        {
            "Year": int(item["date"]),
            "Economy": countries[item["countryiso3code"]],
            "Value": item["value"],
        }
        for item in rows
        if item.get("value") is not None and item.get("countryiso3code") in countries
    ]
    if not records:
        return pd.DataFrame(columns=list(countries.values()), dtype=float)
    frame = (
        pd.DataFrame(records)
        .pivot(index="Year", columns="Economy", values="Value")
        .sort_index()
        .reindex(columns=list(countries.values()))
        .astype(float)
    )
    frame.attrs["indicator"] = indicator
    frame.attrs["last_updated"] = payload[0].get("lastupdated")
    save_cached_frame("world_bank", cache_key, frame)
    return frame


def fetch_nominal_gdp_current_usd(
    countries: Mapping[str, str],
    **kwargs,
) -> pd.DataFrame:
    """Fetch annual nominal GDP measured in comparable current U.S. dollars."""
    return fetch_world_bank_indicator(
        countries,
        NOMINAL_GDP_CURRENT_USD,
        **kwargs,
    )


__all__ = [
    "NOMINAL_GDP_CURRENT_USD",
    "fetch_nominal_gdp_current_usd",
    "fetch_world_bank_indicator",
]

"""Wikipedia-backed table fetch helpers."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

DEFAULT_WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0"}
WIKIPEDIA_GICS_STRUCTURE_URL = "https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard"
WIKIPEDIA_MARKET_INDEX_URLS = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ_100": "https://en.wikipedia.org/wiki/NASDAQ-100",
    # The main DJIA article no longer contains the constituents table.
    "DIA": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
    "Russell_1000": "https://en.wikipedia.org/wiki/Russell_1000_Index",
}
WIKIPEDIA_SP_MARKET_CAP_INDEX_URLS = {
    "Large Cap": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "Mid Cap": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "Small Cap": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def fetch_wikipedia_html(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> str:
    """Fetch one Wikipedia page and return its HTML."""
    http = session or requests
    response = http.get(
        url,
        headers=headers or DEFAULT_WIKIPEDIA_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def fetch_wikipedia_tables(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> list[pd.DataFrame]:
    """Fetch all HTML tables from one Wikipedia page."""
    html = fetch_wikipedia_html(
        url,
        headers=headers,
        timeout=timeout,
        session=session,
    )
    return pd.read_html(StringIO(html))


def fetch_wikipedia_market_index_tables(
    *,
    urls: dict[str, str] | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, list[pd.DataFrame]]:
    """Fetch raw Wikipedia table lists for supported market indexes."""
    resolved_urls = urls or WIKIPEDIA_MARKET_INDEX_URLS
    return {
        index_name: fetch_wikipedia_tables(
            url,
            headers=headers,
            timeout=timeout,
            session=session,
        )
        for index_name, url in resolved_urls.items()
    }

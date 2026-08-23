"""Peer analysis normalization helpers."""

from __future__ import annotations

import pandas as pd

from Quantapp.data.sources.fmp import normalize_fmp_symbol


def pick_numeric(record: dict, *keys: str) -> float:
    """Pick the first numeric value available for the supplied keys."""
    for key in keys:
        if not isinstance(record, dict):
            continue
        value = record.get(key)
        if value in (None, "", "None"):
            continue
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value):
            return float(numeric_value)
    return float("nan")


def safe_first_record(payload) -> dict:
    """Return a first record from a list/dict payload."""
    if isinstance(payload, list):
        return payload[0] if payload else {}
    if isinstance(payload, dict):
        return payload
    return {}


def extract_peer_symbols(peer_payload, *, target_symbol: str) -> list[str]:
    """Extract a de-duplicated symbol list from FMP stock-peers payloads."""
    target = normalize_fmp_symbol(target_symbol)
    raw_peer_symbols = []
    peer_candidates = peer_payload if isinstance(peer_payload, list) else [peer_payload]

    for record in peer_candidates:
        if isinstance(record, str):
            candidate_values = [record]
        elif isinstance(record, dict):
            candidate_values = []
            for key in ("peersList", "peers", "symbols", "companies"):
                value = record.get(key)
                if isinstance(value, list):
                    candidate_values.extend(value)
            for key in ("symbol", "peerSymbol"):
                value = record.get(key)
                if value:
                    candidate_values.append(value)
        else:
            candidate_values = []

        for candidate in candidate_values:
            candidate_symbol = normalize_fmp_symbol(candidate)
            if candidate_symbol and candidate_symbol != target:
                raw_peer_symbols.append(candidate_symbol)

    return list(dict.fromkeys(raw_peer_symbols))


def company_context(symbol: str, quote: dict, profile: dict) -> dict:
    """Build normalized target-company context from quote/profile payloads."""
    normalized_symbol = normalize_fmp_symbol(symbol)
    company_name = quote.get("name") or profile.get("companyName") or normalized_symbol
    industry_name = profile.get("industry") or quote.get("industry") or ""
    sector_name = profile.get("sector") or quote.get("sector") or ""
    exchange = (
        profile.get("exchangeShortName")
        or profile.get("exchange")
        or quote.get("exchange")
        or quote.get("exchangeShortName")
        or ""
    )
    market_cap = pick_numeric(quote, "marketCap", "marketCapTTM")
    return {
        "symbol": normalized_symbol,
        "company_name": company_name,
        "industry_name": industry_name,
        "sector_name": sector_name,
        "exchange": exchange,
        "market_cap": market_cap,
    }


def filter_peer_symbols(
    raw_peer_symbols: list[str],
    *,
    target_context: dict,
    peer_context_by_symbol: dict[str, tuple[dict, dict]],
    require_same_sector: bool = True,
    require_same_industry: bool = False,
    require_same_exchange: bool = False,
    market_cap_lower_multiple: float = 0.2,
    market_cap_upper_multiple: float = 5.0,
) -> tuple[list[str], pd.DataFrame, str]:
    """Filter peer symbols using the same sector/industry/exchange/size rules as the notebook."""
    filtered_peer_symbols = []
    peer_filter_rows = []
    target_market_cap = target_context.get("market_cap")
    target_sector = target_context.get("sector_name") or ""
    target_industry = target_context.get("industry_name") or ""
    target_exchange = target_context.get("exchange") or ""

    for candidate_symbol in raw_peer_symbols:
        quote, profile = peer_context_by_symbol.get(candidate_symbol, ({}, {}))
        if not quote and not profile:
            peer_filter_rows.append(
                {
                    "symbol": candidate_symbol,
                    "sector": None,
                    "industry": None,
                    "exchange": None,
                    "marketCapBn": pd.NA,
                    "sameSector": False,
                    "sameIndustry": False,
                    "sameExchange": False,
                    "withinMarketCapBand": False,
                    "passesFilters": False,
                    "note": "Lookup failed",
                }
            )
            continue

        candidate_sector = profile.get("sector") or quote.get("sector") or ""
        candidate_industry = profile.get("industry") or quote.get("industry") or ""
        candidate_exchange = (
            profile.get("exchangeShortName")
            or profile.get("exchange")
            or quote.get("exchange")
            or quote.get("exchangeShortName")
            or ""
        )
        candidate_market_cap = pick_numeric(quote, "marketCap", "marketCapTTM")

        same_sector = (not require_same_sector) or (not target_sector) or (candidate_sector == target_sector)
        same_industry = (not require_same_industry) or (not target_industry) or (candidate_industry == target_industry)
        same_exchange = (not require_same_exchange) or (not target_exchange) or (candidate_exchange == target_exchange)
        within_market_cap_band = True
        if pd.notna(target_market_cap) and target_market_cap > 0 and pd.notna(candidate_market_cap):
            within_market_cap_band = (
                target_market_cap * market_cap_lower_multiple
                <= candidate_market_cap
                <= target_market_cap * market_cap_upper_multiple
            )

        passes_filters = same_sector and same_industry and same_exchange and within_market_cap_band
        peer_filter_rows.append(
            {
                "symbol": candidate_symbol,
                "sector": candidate_sector,
                "industry": candidate_industry,
                "exchange": candidate_exchange,
                "marketCapBn": candidate_market_cap / 1_000_000_000 if pd.notna(candidate_market_cap) else pd.NA,
                "sameSector": same_sector,
                "sameIndustry": same_industry,
                "sameExchange": same_exchange,
                "withinMarketCapBand": within_market_cap_band,
                "passesFilters": passes_filters,
                "note": "",
            }
        )

        if passes_filters:
            filtered_peer_symbols.append(candidate_symbol)

    selection_note = "Applied configured peer filters."
    if not filtered_peer_symbols and raw_peer_symbols:
        filtered_peer_symbols = raw_peer_symbols.copy()
        selection_note = (
            "Filters returned no peers, so the notebook fell back to the raw FMP peer list. "
            "Loosen the filter settings or add manual peers if needed."
        )

    return filtered_peer_symbols, pd.DataFrame(peer_filter_rows), selection_note


def build_peer_metrics_frame(metric_records: list[dict]) -> pd.DataFrame:
    """Build and type a peer valuation metrics table."""
    peer_metrics = pd.DataFrame(metric_records).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    numeric_columns = [
        "price",
        "marketCapBn",
        "peRatio",
        "priceToSales",
        "priceToBook",
        "evToRevenue",
        "evToEbitda",
        "freeCashFlowYieldPct",
        "dividendYieldPct",
    ]
    for column in numeric_columns:
        if column not in peer_metrics.columns:
            peer_metrics[column] = pd.NA
        peer_metrics[column] = pd.to_numeric(peer_metrics[column], errors="coerce")

    return peer_metrics.sort_values(["marketCapBn", "symbol"], ascending=[False, True]).reset_index(drop=True)


def build_relative_value_summary(peer_metrics: pd.DataFrame, *, target_symbol: str) -> pd.DataFrame:
    """Summarize target valuation versus peer medians."""
    target = normalize_fmp_symbol(target_symbol)
    target_rows = peer_metrics.loc[peer_metrics["symbol"].eq(target)]
    if target_rows.empty:
        raise ValueError(f"Target ticker {target} was not found in the peer metric table.")

    target_row = target_rows.iloc[0]
    metrics_to_compare = {
        "peRatio": "P/E",
        "priceToSales": "Price / Sales",
        "priceToBook": "Price / Book",
        "evToRevenue": "EV / Revenue",
        "evToEbitda": "EV / EBITDA",
        "freeCashFlowYieldPct": "FCF Yield (%)",
        "dividendYieldPct": "Dividend Yield (%)",
    }

    relative_value_rows = []
    for column, label in metrics_to_compare.items():
        peer_series = peer_metrics.loc[peer_metrics["symbol"].ne(target), column].dropna()
        target_value = pd.to_numeric(pd.Series([target_row[column]]), errors="coerce").iloc[0]
        if pd.isna(target_value) or peer_series.empty:
            continue

        peer_median = float(peer_series.median())
        is_yield_metric = column.endswith("YieldPct")
        if peer_median == 0:
            discount_to_peer_median_pct = pd.NA
        elif is_yield_metric:
            discount_to_peer_median_pct = (target_value / peer_median - 1) * 100
        else:
            discount_to_peer_median_pct = (1 - target_value / peer_median) * 100

        cheaper_than_peers_pct = (
            (peer_series <= target_value).mean() * 100
            if is_yield_metric
            else (peer_series >= target_value).mean() * 100
        )

        relative_value_rows.append(
            {
                "metric": label,
                "targetValue": round(float(target_value), 2),
                "peerMedian": round(peer_median, 2),
                "discountToPeerMedianPct": (
                    round(float(discount_to_peer_median_pct), 2)
                    if pd.notna(discount_to_peer_median_pct)
                    else pd.NA
                ),
                "cheaperThanPeersPct": round(float(cheaper_than_peers_pct), 2),
            }
        )

    return pd.DataFrame(relative_value_rows)

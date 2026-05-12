"""Stable fundamentals data access interfaces for notebooks and apps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from Quantapp.data.adapters.cash_flow import normalize_cash_flow_frame
from Quantapp.data.adapters.balance_sheet import normalize_balance_sheet_frame
from Quantapp.data.adapters.income_statement import (
    normalize_income_statement_frame,
    normalize_revenue_segmentation,
)
from Quantapp.data.adapters.peers import (
    build_peer_metrics_frame,
    build_relative_value_summary,
    company_context,
    extract_peer_symbols,
    filter_peer_symbols,
    pick_numeric,
    safe_first_record,
)
from Quantapp.data.sources.fmp import (
    fetch_fmp_balance_sheet_statement,
    fetch_fmp_cash_flow_statement,
    fetch_fmp_income_statement,
    fetch_fmp_key_metrics_ttm,
    fetch_fmp_profile,
    fetch_fmp_quote,
    fetch_fmp_ratios_ttm,
    fetch_fmp_revenue_segmentation,
    fetch_fmp_stock_peers,
    normalize_fmp_symbol,
)


@dataclass
class BalanceSheetHistory:
    """Normalized balance sheet history for one company."""

    symbol: str
    company_name: str
    chart_label: str
    annual: pd.DataFrame
    quarterly: pd.DataFrame
    quote: list[dict]


@dataclass
class IncomeStatementHistory:
    """Normalized income statement history and revenue segmentation for one company."""

    symbol: str
    company_name: str
    chart_label: str
    annual: pd.DataFrame
    quarterly: pd.DataFrame
    quote: list[dict]
    annual_product_segmentation: pd.DataFrame
    annual_product_segments: list[str]
    annual_geographic_segmentation: pd.DataFrame
    annual_geographic_segments: list[str]
    available_revenue_segmentations: pd.Series


@dataclass
class CashFlowHistory:
    """Normalized cash flow statement history for one company."""

    symbol: str
    company_name: str
    chart_label: str
    annual: pd.DataFrame
    quarterly: pd.DataFrame
    quote: list[dict]


@dataclass
class PeerAnalysisData:
    """Peer-analysis context, peer universe, and valuation tables."""

    symbol: str
    company_name: str
    industry_name: str
    sector_name: str
    target_exchange: str
    target_market_cap: float
    quote: dict
    profile: dict
    raw_peer_symbols: list[str]
    filtered_peer_symbols: list[str]
    peer_symbols: list[str]
    peer_filter_summary: pd.DataFrame
    peer_overview: pd.DataFrame
    peer_metrics: pd.DataFrame
    relative_value_summary: pd.DataFrame
    relative_value: pd.DataFrame


def get_balance_sheet_history(
    ticker: str,
    *,
    provider: str = "fmp",
    annual_limit: int = 20,
    quarterly_limit: int = 40,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> BalanceSheetHistory:
    """Fetch and normalize annual and quarterly balance sheet history."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "fmp":
        raise ValueError(f"Unsupported fundamentals provider '{provider}'.")

    symbol = normalize_fmp_symbol(ticker)
    quote = fetch_fmp_quote(
        symbol,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )
    company_name = quote[0].get("name", symbol) if quote else symbol
    chart_label = f"{company_name} ({symbol})" if company_name != symbol else symbol

    annual = normalize_balance_sheet_frame(
        pd.DataFrame(
            fetch_fmp_balance_sheet_statement(
                symbol,
                limit=annual_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="annual",
    )
    if annual.empty:
        raise RuntimeError(f"FMP did not return annual balance sheet history for {symbol}.")

    quarterly = normalize_balance_sheet_frame(
        pd.DataFrame(
            fetch_fmp_balance_sheet_statement(
                symbol,
                period="quarter",
                limit=quarterly_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="quarterly",
    )
    if quarterly.empty:
        raise RuntimeError(f"FMP did not return quarterly balance sheet history for {symbol}.")

    return BalanceSheetHistory(
        symbol=symbol,
        company_name=company_name,
        chart_label=chart_label,
        annual=annual,
        quarterly=quarterly,
        quote=quote,
    )


def _quote_company_label(
    symbol: str,
    *,
    api_key: str | None,
    base_url: str | None,
    timeout: int,
    session: requests.Session | None,
) -> tuple[list[dict], str, str]:
    quote = fetch_fmp_quote(
        symbol,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )
    company_name = quote[0].get("name", symbol) if quote else symbol
    chart_label = f"{company_name} ({symbol})" if company_name != symbol else symbol
    return quote, company_name, chart_label


def get_income_statement_history(
    ticker: str,
    *,
    provider: str = "fmp",
    annual_limit: int = 20,
    quarterly_limit: int = 40,
    include_revenue_segmentation: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> IncomeStatementHistory:
    """Fetch and normalize annual/quarterly income statement history."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "fmp":
        raise ValueError(f"Unsupported fundamentals provider '{provider}'.")

    symbol = normalize_fmp_symbol(ticker)
    quote, company_name, chart_label = _quote_company_label(
        symbol,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )

    annual = normalize_income_statement_frame(
        pd.DataFrame(
            fetch_fmp_income_statement(
                symbol,
                limit=annual_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="annual",
    )
    if annual.empty:
        raise RuntimeError(f"FMP did not return annual income statement history for {symbol}.")

    quarterly = normalize_income_statement_frame(
        pd.DataFrame(
            fetch_fmp_income_statement(
                symbol,
                period="quarter",
                limit=quarterly_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="quarterly",
    )
    if quarterly.empty:
        raise RuntimeError(f"FMP did not return quarterly income statement history for {symbol}.")

    annual_product_segmentation = pd.DataFrame()
    annual_product_segments: list[str] = []
    annual_geographic_segmentation = pd.DataFrame()
    annual_geographic_segments: list[str] = []
    if include_revenue_segmentation:
        try:
            annual_product_segmentation, annual_product_segments = normalize_revenue_segmentation(
                fetch_fmp_revenue_segmentation(
                    symbol,
                    endpoint="revenue-product-segmentation",
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception:
            annual_product_segmentation, annual_product_segments = pd.DataFrame(), []

        try:
            annual_geographic_segmentation, annual_geographic_segments = normalize_revenue_segmentation(
                fetch_fmp_revenue_segmentation(
                    symbol,
                    endpoint="revenue-geographic-segmentation",
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception:
            annual_geographic_segmentation, annual_geographic_segments = pd.DataFrame(), []

    available_revenue_segmentations = pd.Series(
        {
            "annualProduct": bool(annual_product_segments),
            "annualGeographic": bool(annual_geographic_segments),
        },
        name="available",
    )

    return IncomeStatementHistory(
        symbol=symbol,
        company_name=company_name,
        chart_label=chart_label,
        annual=annual,
        quarterly=quarterly,
        quote=quote,
        annual_product_segmentation=annual_product_segmentation,
        annual_product_segments=annual_product_segments,
        annual_geographic_segmentation=annual_geographic_segmentation,
        annual_geographic_segments=annual_geographic_segments,
        available_revenue_segmentations=available_revenue_segmentations,
    )


def get_cash_flow_history(
    ticker: str,
    *,
    provider: str = "fmp",
    annual_limit: int = 20,
    quarterly_limit: int = 40,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> CashFlowHistory:
    """Fetch and normalize annual/quarterly cash flow statement history."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "fmp":
        raise ValueError(f"Unsupported fundamentals provider '{provider}'.")

    symbol = normalize_fmp_symbol(ticker)
    quote, company_name, chart_label = _quote_company_label(
        symbol,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        session=session,
    )

    annual = normalize_cash_flow_frame(
        pd.DataFrame(
            fetch_fmp_cash_flow_statement(
                symbol,
                limit=annual_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="annual",
    )
    if annual.empty:
        raise RuntimeError(f"FMP did not return annual cash flow history for {symbol}.")

    quarterly = normalize_cash_flow_frame(
        pd.DataFrame(
            fetch_fmp_cash_flow_statement(
                symbol,
                period="quarter",
                limit=quarterly_limit,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                session=session,
            )
        ),
        frequency="quarterly",
    )
    if quarterly.empty:
        raise RuntimeError(f"FMP did not return quarterly cash flow history for {symbol}.")

    return CashFlowHistory(
        symbol=symbol,
        company_name=company_name,
        chart_label=chart_label,
        annual=annual,
        quarterly=quarterly,
        quote=quote,
    )


def get_peer_analysis_data(
    ticker: str,
    *,
    provider: str = "fmp",
    peer_limit: int = 12,
    manual_peer_symbols: list[str] | None = None,
    require_same_sector: bool = True,
    require_same_industry: bool = False,
    require_same_exchange: bool = False,
    market_cap_lower_multiple: float = 0.2,
    market_cap_upper_multiple: float = 5.0,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 30,
    session: requests.Session | None = None,
) -> PeerAnalysisData:
    """Fetch FMP peer context and valuation tables for one company."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider != "fmp":
        raise ValueError(f"Unsupported fundamentals provider '{provider}'.")

    symbol = normalize_fmp_symbol(ticker)
    quote = safe_first_record(
        fetch_fmp_quote(
            symbol,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            session=session,
        )
    )
    profile = safe_first_record(
        fetch_fmp_profile(
            symbol,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            session=session,
        )
    )
    target_context = company_context(symbol, quote, profile)

    try:
        peer_payload = fetch_fmp_stock_peers(
            symbol,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            session=session,
        )
    except Exception:
        peer_payload = []

    raw_peer_symbols = extract_peer_symbols(peer_payload, target_symbol=symbol)
    manual_symbols = [
        normalize_fmp_symbol(manual_symbol)
        for manual_symbol in (manual_peer_symbols or [])
        if str(manual_symbol).strip()
    ]

    peer_context_by_symbol: dict[str, tuple[dict, dict]] = {}
    for candidate_symbol in raw_peer_symbols:
        try:
            candidate_quote = safe_first_record(
                fetch_fmp_quote(
                    candidate_symbol,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
            candidate_profile = safe_first_record(
                fetch_fmp_profile(
                    candidate_symbol,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception:
            candidate_quote, candidate_profile = {}, {}
        peer_context_by_symbol[candidate_symbol] = (candidate_quote, candidate_profile)

    filtered_peer_symbols, peer_filter_summary, selection_note = filter_peer_symbols(
        raw_peer_symbols,
        target_context=target_context,
        peer_context_by_symbol=peer_context_by_symbol,
        require_same_sector=require_same_sector,
        require_same_industry=require_same_industry,
        require_same_exchange=require_same_exchange,
        market_cap_lower_multiple=market_cap_lower_multiple,
        market_cap_upper_multiple=market_cap_upper_multiple,
    )

    peer_symbols = [symbol, *manual_symbols, *filtered_peer_symbols]
    peer_symbols = [peer_symbol for peer_symbol in dict.fromkeys(peer_symbols) if peer_symbol][:peer_limit]

    metric_rows = []
    for peer_symbol in peer_symbols:
        if peer_symbol == symbol:
            peer_quote = quote
            peer_profile = profile
        else:
            peer_quote, peer_profile = peer_context_by_symbol.get(peer_symbol, ({}, {}))
            if not peer_quote:
                try:
                    peer_quote = safe_first_record(
                        fetch_fmp_quote(
                            peer_symbol,
                            api_key=api_key,
                            base_url=base_url,
                            timeout=timeout,
                            session=session,
                        )
                    )
                except Exception:
                    peer_quote = {}
            if not peer_profile:
                try:
                    peer_profile = safe_first_record(
                        fetch_fmp_profile(
                            peer_symbol,
                            api_key=api_key,
                            base_url=base_url,
                            timeout=timeout,
                            session=session,
                        )
                    )
                except Exception:
                    peer_profile = {}

        try:
            peer_ratios = safe_first_record(
                fetch_fmp_ratios_ttm(
                    peer_symbol,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception:
            peer_ratios = {}

        try:
            peer_key_metrics = safe_first_record(
                fetch_fmp_key_metrics_ttm(
                    peer_symbol,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    session=session,
                )
            )
        except Exception:
            peer_key_metrics = {}

        metric_rows.append(
            {
                "symbol": peer_symbol,
                "companyName": peer_quote.get("name") or peer_profile.get("companyName") or peer_symbol,
                "sector": peer_profile.get("sector"),
                "industry": peer_profile.get("industry"),
                "price": pick_numeric(peer_quote, "price"),
                "marketCapBn": pick_numeric(peer_quote, "marketCap") / 1_000_000_000,
                "peRatio": pick_numeric(peer_ratios, "peRatioTTM", "priceEarningsRatioTTM", "peRatio"),
                "priceToSales": pick_numeric(peer_ratios, "priceToSalesRatioTTM", "priceToSalesRatio"),
                "priceToBook": pick_numeric(peer_ratios, "priceToBookRatioTTM", "priceToBookRatio"),
                "evToRevenue": pick_numeric(peer_key_metrics, "enterpriseValueOverRevenueTTM", "enterpriseValueOverRevenue"),
                "evToEbitda": pick_numeric(peer_key_metrics, "enterpriseValueOverEBITDATTM", "enterpriseValueOverEBITDA"),
                "freeCashFlowYieldPct": pick_numeric(peer_key_metrics, "freeCashFlowYieldTTM", "freeCashFlowYield") * 100,
                "dividendYieldPct": pick_numeric(
                    peer_ratios,
                    "dividendYielTTM",
                    "dividendYieldTTM",
                    "dividendYieldPercentageTTM",
                    "dividendYield",
                )
                * 100,
            }
        )

    peer_metrics = build_peer_metrics_frame(metric_rows)
    relative_value_summary = build_relative_value_summary(peer_metrics, target_symbol=symbol)
    peer_overview = pd.DataFrame(
        [
            {
                "company": target_context["company_name"],
                "symbol": symbol,
                "sector": target_context["sector_name"],
                "industry": target_context["industry_name"],
                "exchange": target_context["exchange"],
                "rawPeerCount": len(raw_peer_symbols),
                "filteredPeerCount": len(filtered_peer_symbols),
                "peerCount": len(peer_symbols),
                "peerSource": "FMP stock-peers endpoint",
                "selectionNote": selection_note,
            }
        ]
    )

    return PeerAnalysisData(
        symbol=symbol,
        company_name=target_context["company_name"],
        industry_name=target_context["industry_name"],
        sector_name=target_context["sector_name"],
        target_exchange=target_context["exchange"],
        target_market_cap=target_context["market_cap"],
        quote=quote,
        profile=profile,
        raw_peer_symbols=raw_peer_symbols,
        filtered_peer_symbols=filtered_peer_symbols,
        peer_symbols=peer_symbols,
        peer_filter_summary=peer_filter_summary,
        peer_overview=peer_overview,
        peer_metrics=peer_metrics,
        relative_value_summary=relative_value_summary,
        relative_value=relative_value_summary,
    )

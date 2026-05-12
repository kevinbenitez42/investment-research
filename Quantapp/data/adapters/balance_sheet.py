"""Balance sheet normalization helpers."""

from __future__ import annotations

import pandas as pd

from Quantapp.data.schemas.balance_sheet import (
    BALANCE_SHEET_BASE_COLUMNS,
    BALANCE_SHEET_NUMERIC_COLUMNS,
)


def prepare_balance_sheet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw FMP balance sheet rows to the base Quantapp shape."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    prepared = frame.copy()
    if prepared.empty:
        return prepared

    for column in BALANCE_SHEET_BASE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    if "calendarYear" in frame.columns:
        prepared["calendarYear"] = pd.to_numeric(prepared["calendarYear"], errors="coerce")
    elif "fiscalYear" in frame.columns:
        prepared["calendarYear"] = pd.to_numeric(prepared["fiscalYear"], errors="coerce")
    else:
        prepared["calendarYear"] = pd.NA

    if prepared["calendarYear"].isna().all():
        prepared["calendarYear"] = prepared["date"].dt.year

    for column in BALANCE_SHEET_NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def normalize_balance_sheet_frame(
    frame: pd.DataFrame,
    *,
    frequency: str = "annual",
) -> pd.DataFrame:
    """Clean and enrich FMP balance sheet rows for notebook and app consumption."""
    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency not in {"annual", "quarterly"}:
        raise ValueError("frequency must be either 'annual' or 'quarterly'.")

    balance = prepare_balance_sheet_frame(frame)
    if balance.empty:
        return balance

    balance = (
        balance.loc[:, BALANCE_SHEET_BASE_COLUMNS]
        .dropna(subset=["date", "totalAssets"])
        .copy()
    )
    if balance.empty:
        return balance

    balance["totalEquity"] = balance["totalEquity"].fillna(balance["totalStockholdersEquity"])
    balance["totalStockholdersEquity"] = balance["totalStockholdersEquity"].fillna(balance["totalEquity"])
    balance["cashAndShortTermInvestments"] = balance["cashAndShortTermInvestments"].fillna(
        balance["cashAndCashEquivalents"].fillna(0) + balance["shortTermInvestments"].fillna(0)
    )
    balance["goodwillAndIntangibleAssets"] = balance["goodwillAndIntangibleAssets"].fillna(
        balance["goodwill"].fillna(0) + balance["intangibleAssets"].fillna(0)
    )
    balance["totalDebt"] = balance["totalDebt"].fillna(
        balance["shortTermDebt"].fillna(0) + balance["longTermDebt"].fillna(0)
    )
    balance["totalNonCurrentAssets"] = balance["totalNonCurrentAssets"].fillna(
        balance["totalAssets"] - balance["totalCurrentAssets"]
    )
    balance["totalNonCurrentLiabilities"] = balance["totalNonCurrentLiabilities"].fillna(
        balance["totalLiabilities"] - balance["totalCurrentLiabilities"]
    )
    balance["netDebt"] = balance["netDebt"].fillna(
        balance["totalDebt"] - balance["cashAndCashEquivalents"].fillna(0)
    )

    balance = balance.sort_values("date").reset_index(drop=True)
    if normalized_frequency == "quarterly":
        balance["quarterLabel"] = balance["period"].astype("string").str.upper().str.extract(r"(Q[1-4])", expand=False)
        missing_quarter_label = balance["quarterLabel"].isna()
        balance.loc[missing_quarter_label, "quarterLabel"] = (
            "Q" + balance.loc[missing_quarter_label, "date"].dt.quarter.astype("Int64").astype(str)
        )
        balance["quarterlyYear"] = balance["date"].dt.year

    total_assets = balance["totalAssets"].replace(0, pd.NA)
    current_liabilities = balance["totalCurrentLiabilities"].replace(0, pd.NA)
    total_equity = balance["totalEquity"].replace(0, pd.NA)
    yoy_periods = 4 if normalized_frequency == "quarterly" else 1

    balance["workingCapital"] = balance["totalCurrentAssets"] - balance["totalCurrentLiabilities"]
    balance["cashAndShortTermInvestmentsPctAssets"] = balance["cashAndShortTermInvestments"].div(total_assets)
    balance["receivablesPctAssets"] = balance["netReceivables"].div(total_assets)
    balance["inventoryPctAssets"] = balance["inventory"].div(total_assets)
    balance["ppePctAssets"] = balance["propertyPlantEquipmentNet"].div(total_assets)
    balance["goodwillIntangiblePctAssets"] = balance["goodwillAndIntangibleAssets"].div(total_assets)
    balance["longTermInvestmentsPctAssets"] = balance["longTermInvestments"].div(total_assets)
    balance["currentAssetsPctAssets"] = balance["totalCurrentAssets"].div(total_assets)
    balance["nonCurrentAssetsPctAssets"] = balance["totalNonCurrentAssets"].div(total_assets)
    balance["currentLiabilitiesPctAssets"] = balance["totalCurrentLiabilities"].div(total_assets)
    balance["nonCurrentLiabilitiesPctAssets"] = balance["totalNonCurrentLiabilities"].div(total_assets)
    balance["totalDebtPctAssets"] = balance["totalDebt"].div(total_assets)
    balance["totalLiabilitiesPctAssets"] = balance["totalLiabilities"].div(total_assets)
    balance["totalEquityPctAssets"] = balance["totalEquity"].div(total_assets)
    balance["currentRatio"] = balance["totalCurrentAssets"].div(current_liabilities)
    balance["cashRatio"] = balance["cashAndShortTermInvestments"].div(current_liabilities)
    balance["workingCapitalPctAssets"] = balance["workingCapital"].div(total_assets)
    balance["debtToAssets"] = balance["totalDebt"].div(total_assets)
    balance["liabilityToAssets"] = balance["totalLiabilities"].div(total_assets)
    balance["equityRatio"] = balance["totalEquity"].div(total_assets)
    balance["debtToEquity"] = balance["totalDebt"].div(total_equity)
    balance["netDebtToEquity"] = balance["netDebt"].div(total_equity)
    balance["totalAssetsYoY"] = balance["totalAssets"].pct_change(yoy_periods, fill_method=None)
    balance["totalLiabilitiesYoY"] = balance["totalLiabilities"].pct_change(yoy_periods, fill_method=None)
    balance["totalEquityYoY"] = balance["totalEquity"].pct_change(yoy_periods, fill_method=None)
    balance["cashYoY"] = balance["cashAndShortTermInvestments"].pct_change(yoy_periods, fill_method=None)
    balance["totalDebtYoY"] = balance["totalDebt"].pct_change(yoy_periods, fill_method=None)
    balance["workingCapitalYoY"] = balance["workingCapital"].pct_change(yoy_periods, fill_method=None)
    balance["currentRatioChange"] = balance["currentRatio"].diff(yoy_periods)
    balance["debtToEquityChange"] = balance["debtToEquity"].diff(yoy_periods)
    balance["liabilityToAssetsChange"] = balance["liabilityToAssets"].diff(yoy_periods)
    balance["equityRatioChange"] = balance["equityRatio"].diff(yoy_periods)

    return balance

"""Cash flow statement normalization helpers."""

from __future__ import annotations

import pandas as pd

from Quantapp.data.schemas.cash_flow import (
    CASH_FLOW_BASE_COLUMNS,
    CASH_FLOW_NUMERIC_COLUMNS,
)


def prepare_cash_flow_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw FMP cash flow statement rows to the base Quantapp shape."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    prepared = frame.copy()
    if prepared.empty:
        return prepared

    for column in CASH_FLOW_BASE_COLUMNS:
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

    for column in CASH_FLOW_NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def normalize_cash_flow_frame(
    frame: pd.DataFrame,
    *,
    frequency: str = "annual",
) -> pd.DataFrame:
    """Clean and enrich FMP cash flow statement rows for notebooks and apps."""
    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency not in {"annual", "quarterly"}:
        raise ValueError("frequency must be either 'annual' or 'quarterly'.")

    cash = prepare_cash_flow_frame(frame)
    if cash.empty:
        return cash

    cash = (
        cash.loc[:, CASH_FLOW_BASE_COLUMNS]
        .dropna(subset=["date", "netIncome"])
        .copy()
    )
    if cash.empty:
        return cash

    cash["operatingCashFlow"] = cash["operatingCashFlow"].fillna(cash["netCashProvidedByOperatingActivities"])
    cash["capitalExpenditure"] = cash["capitalExpenditure"].fillna(cash["investmentsInPropertyPlantAndEquipment"])
    cash["netDividendsPaid"] = cash["netDividendsPaid"].fillna(cash["commonDividendsPaid"])
    cash["freeCashFlow"] = cash["freeCashFlow"].fillna(cash["operatingCashFlow"] + cash["capitalExpenditure"])
    cash["shareholderReturns"] = cash["commonStockRepurchased"].fillna(0) + cash["netDividendsPaid"].fillna(0)
    cash["discretionaryCashFlow"] = cash["operatingCashFlow"] + cash["capitalExpenditure"]

    cash = cash.sort_values("date").reset_index(drop=True)
    if normalized_frequency == "quarterly":
        cash["quarterLabel"] = cash["period"].astype("string").str.upper().str.extract(r"(Q[1-4])", expand=False)
        missing_quarter_label = cash["quarterLabel"].isna()
        cash.loc[missing_quarter_label, "quarterLabel"] = (
            "Q" + cash.loc[missing_quarter_label, "date"].dt.quarter.astype("Int64").astype(str)
        )
        cash["quarterlyYear"] = cash["date"].dt.year

    yoy_periods = 4 if normalized_frequency == "quarterly" else 1
    operating_cash_flow_denominator = cash["operatingCashFlow"].replace(0, pd.NA)
    free_cash_flow_denominator = cash["freeCashFlow"].replace(0, pd.NA)

    cash["operatingCashFlowYoY"] = cash["operatingCashFlow"].pct_change(yoy_periods, fill_method=None)
    cash["freeCashFlowYoY"] = cash["freeCashFlow"].pct_change(yoy_periods, fill_method=None)
    cash["netIncomeYoY"] = cash["netIncome"].pct_change(yoy_periods, fill_method=None)
    cash["netCashChangeYoY"] = cash["netChangeInCash"].pct_change(yoy_periods, fill_method=None)
    cash["cfoToNetIncome"] = cash["operatingCashFlow"].div(cash["netIncome"].replace(0, pd.NA))
    cash["capexToCfo"] = cash["capitalExpenditure"].mul(-1).div(operating_cash_flow_denominator)
    cash["fcfToCfo"] = cash["freeCashFlow"].div(operating_cash_flow_denominator)
    cash["stockCompToCfo"] = cash["stockBasedCompensation"].div(operating_cash_flow_denominator)
    cash["workingCapitalDragToCfo"] = cash["changeInWorkingCapital"].mul(-1).div(operating_cash_flow_denominator)
    cash["buybacksToFcf"] = cash["commonStockRepurchased"].mul(-1).div(free_cash_flow_denominator)
    cash["dividendsToFcf"] = cash["netDividendsPaid"].mul(-1).div(free_cash_flow_denominator)
    cash["shareholderReturnsToFcf"] = cash["shareholderReturns"].mul(-1).div(free_cash_flow_denominator)

    if normalized_frequency == "quarterly":
        cash["ttmNetIncome"] = cash["netIncome"].rolling(4).sum()
        cash["ttmOperatingCashFlow"] = cash["operatingCashFlow"].rolling(4).sum()
        cash["ttmFreeCashFlow"] = cash["freeCashFlow"].rolling(4).sum()
        cash["ttmCapitalExpenditure"] = cash["capitalExpenditure"].rolling(4).sum()
        cash["ttmShareholderReturns"] = cash["shareholderReturns"].rolling(4).sum()
        cash["ttmNetChangeInCash"] = cash["netChangeInCash"].rolling(4).sum()
        cash["ttmDiscretionaryCashFlow"] = cash["discretionaryCashFlow"].rolling(4).sum()
        cash["ttmBuybacks"] = cash["commonStockRepurchased"].rolling(4).sum()
        cash["ttmDividends"] = cash["netDividendsPaid"].rolling(4).sum()
        cash["ttmCfoToNetIncome"] = cash["ttmOperatingCashFlow"].div(cash["ttmNetIncome"].replace(0, pd.NA))
        cash["ttmCapexToCfo"] = cash["ttmCapitalExpenditure"].mul(-1).div(
            cash["ttmOperatingCashFlow"].replace(0, pd.NA)
        )
        cash["ttmFcfToCfo"] = cash["ttmFreeCashFlow"].div(cash["ttmOperatingCashFlow"].replace(0, pd.NA))
        cash["ttmShareholderReturnsToFcf"] = cash["ttmShareholderReturns"].mul(-1).div(
            cash["ttmFreeCashFlow"].replace(0, pd.NA)
        )
        cash["ttmBuybacksToFcf"] = cash["ttmBuybacks"].mul(-1).div(cash["ttmFreeCashFlow"].replace(0, pd.NA))
        cash["ttmDividendsToFcf"] = cash["ttmDividends"].mul(-1).div(cash["ttmFreeCashFlow"].replace(0, pd.NA))

    return cash

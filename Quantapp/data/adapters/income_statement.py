"""Income statement normalization helpers."""

from __future__ import annotations

import pandas as pd

from Quantapp.data.schemas.income_statement import (
    INCOME_STATEMENT_BASE_COLUMNS,
    INCOME_STATEMENT_NUMERIC_COLUMNS,
)


def prepare_income_statement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw FMP income statement rows to the base Quantapp shape."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    prepared = frame.copy()
    if prepared.empty:
        return prepared

    for column in INCOME_STATEMENT_BASE_COLUMNS:
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

    for column in INCOME_STATEMENT_NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def normalize_income_statement_frame(
    frame: pd.DataFrame,
    *,
    frequency: str = "annual",
) -> pd.DataFrame:
    """Clean and enrich FMP income statement rows for notebooks and apps."""
    normalized_frequency = str(frequency).strip().lower()
    if normalized_frequency not in {"annual", "quarterly"}:
        raise ValueError("frequency must be either 'annual' or 'quarterly'.")

    revenue = prepare_income_statement_frame(frame)
    if revenue.empty:
        return revenue

    revenue = (
        revenue.loc[:, INCOME_STATEMENT_BASE_COLUMNS]
        .dropna(subset=["date", "revenue"])
        .copy()
    )
    if revenue.empty:
        return revenue

    revenue["costOfRevenue"] = revenue["costOfRevenue"].fillna(
        revenue["revenue"] - revenue["grossProfit"]
    )
    revenue["incomeBeforeTax"] = revenue["incomeBeforeTax"].fillna(
        revenue["netIncome"] + revenue["incomeTaxExpense"]
    )
    revenue["totalOtherIncomeExpensesNet"] = revenue["totalOtherIncomeExpensesNet"].fillna(
        revenue["incomeBeforeTax"] - revenue["operatingIncome"]
    )
    revenue["epsDiluted"] = revenue["epsDiluted"].fillna(revenue["eps"])
    revenue["weightedAverageShsOutDil"] = revenue["weightedAverageShsOutDil"].fillna(
        revenue["weightedAverageShsOut"]
    )

    revenue = revenue.sort_values("date").reset_index(drop=True)
    if normalized_frequency == "quarterly":
        revenue["quarterLabel"] = revenue["period"].astype("string").str.upper().str.extract(r"(Q[1-4])", expand=False)
        missing_quarter_label = revenue["quarterLabel"].isna()
        revenue.loc[missing_quarter_label, "quarterLabel"] = (
            "Q" + revenue.loc[missing_quarter_label, "date"].dt.quarter.astype("Int64").astype(str)
        )
        revenue["quarterlyYear"] = revenue["date"].dt.year

    yoy_periods = 4 if normalized_frequency == "quarterly" else 1
    revenue_denominator = revenue["revenue"].replace(0, pd.NA)
    income_before_tax_denominator = revenue["incomeBeforeTax"].replace(0, pd.NA)
    basic_shares_denominator = revenue["weightedAverageShsOut"].replace(0, pd.NA)

    revenue["revenueYoY"] = revenue["revenue"].pct_change(yoy_periods, fill_method=None)
    revenue["grossProfitYoY"] = revenue["grossProfit"].pct_change(yoy_periods, fill_method=None)
    revenue["operatingIncomeYoY"] = revenue["operatingIncome"].pct_change(yoy_periods, fill_method=None)
    revenue["incomeBeforeTaxYoY"] = revenue["incomeBeforeTax"].pct_change(yoy_periods, fill_method=None)
    revenue["incomeTaxExpenseYoY"] = revenue["incomeTaxExpense"].pct_change(yoy_periods, fill_method=None)
    revenue["netIncomeYoY"] = revenue["netIncome"].pct_change(yoy_periods, fill_method=None)
    revenue["costOfRevenueYoY"] = revenue["costOfRevenue"].pct_change(yoy_periods, fill_method=None)
    revenue["operatingExpensesYoY"] = revenue["operatingExpenses"].pct_change(yoy_periods, fill_method=None)
    revenue["gaYoY"] = revenue["generalAndAdministrativeExpenses"].pct_change(yoy_periods, fill_method=None)
    revenue["marketingYoY"] = revenue["sellingAndMarketingExpenses"].pct_change(yoy_periods, fill_method=None)
    revenue["sgaYoY"] = revenue["sellingGeneralAndAdministrativeExpenses"].pct_change(yoy_periods, fill_method=None)
    revenue["rdYoY"] = revenue["researchAndDevelopmentExpenses"].pct_change(yoy_periods, fill_method=None)
    revenue["epsYoY"] = revenue["eps"].pct_change(yoy_periods, fill_method=None)
    revenue["epsDilutedYoY"] = revenue["epsDiluted"].pct_change(yoy_periods, fill_method=None)
    revenue["dilutedSharesYoY"] = revenue["weightedAverageShsOutDil"].pct_change(yoy_periods, fill_method=None)

    if normalized_frequency == "quarterly":
        revenue["ttmRevenue"] = revenue["revenue"].rolling(4).sum()
        revenue["ttmGrossProfit"] = revenue["grossProfit"].rolling(4).sum()
        revenue["ttmOperatingIncome"] = revenue["operatingIncome"].rolling(4).sum()
        revenue["ttmIncomeBeforeTax"] = revenue["incomeBeforeTax"].rolling(4).sum()
        revenue["ttmNetIncome"] = revenue["netIncome"].rolling(4).sum()
        revenue["ttmRevenueYoY"] = revenue["ttmRevenue"].pct_change(4, fill_method=None)
        revenue["ttmGrossProfitYoY"] = revenue["ttmGrossProfit"].pct_change(4, fill_method=None)
        revenue["ttmOperatingIncomeYoY"] = revenue["ttmOperatingIncome"].pct_change(4, fill_method=None)
        revenue["ttmIncomeBeforeTaxYoY"] = revenue["ttmIncomeBeforeTax"].pct_change(4, fill_method=None)
        revenue["ttmNetIncomeYoY"] = revenue["ttmNetIncome"].pct_change(4, fill_method=None)

    revenue["grossMargin"] = revenue["grossProfit"].div(revenue_denominator)
    revenue["operatingMargin"] = revenue["operatingIncome"].div(revenue_denominator)
    revenue["incomeBeforeTaxMargin"] = revenue["incomeBeforeTax"].div(revenue_denominator)
    revenue["netMargin"] = revenue["netIncome"].div(revenue_denominator)
    revenue["grossProfitPctRevenue"] = revenue["grossProfit"].div(revenue_denominator)
    revenue["costOfRevenuePctRevenue"] = revenue["costOfRevenue"].div(revenue_denominator)
    revenue["operatingExpensesPctRevenue"] = revenue["operatingExpenses"].div(revenue_denominator)
    revenue["gaPctRevenue"] = revenue["generalAndAdministrativeExpenses"].div(revenue_denominator)
    revenue["marketingPctRevenue"] = revenue["sellingAndMarketingExpenses"].div(revenue_denominator)
    revenue["sgaPctRevenue"] = revenue["sellingGeneralAndAdministrativeExpenses"].div(revenue_denominator)
    revenue["rdPctRevenue"] = revenue["researchAndDevelopmentExpenses"].div(revenue_denominator)
    revenue["otherIncomeExpensePctRevenue"] = revenue["totalOtherIncomeExpensesNet"].div(revenue_denominator)
    revenue["incomeTaxExpensePctRevenue"] = revenue["incomeTaxExpense"].div(revenue_denominator)
    revenue["incomeTaxRate"] = revenue["incomeTaxExpense"].div(income_before_tax_denominator)
    revenue["dilutionPct"] = revenue["weightedAverageShsOutDil"].div(basic_shares_denominator) - 1

    if normalized_frequency == "quarterly":
        ttm_revenue_denominator = revenue["ttmRevenue"].replace(0, pd.NA)
        revenue["ttmGrossMargin"] = revenue["ttmGrossProfit"].div(ttm_revenue_denominator)
        revenue["ttmOperatingMargin"] = revenue["ttmOperatingIncome"].div(ttm_revenue_denominator)
        revenue["ttmIncomeBeforeTaxMargin"] = revenue["ttmIncomeBeforeTax"].div(ttm_revenue_denominator)
        revenue["ttmNetMargin"] = revenue["ttmNetIncome"].div(ttm_revenue_denominator)

    revenue["grossMarginChange"] = revenue["grossMargin"].diff(yoy_periods)
    revenue["operatingMarginChange"] = revenue["operatingMargin"].diff(yoy_periods)
    revenue["netMarginChange"] = revenue["netMargin"].diff(yoy_periods)

    return revenue


def normalize_revenue_segmentation(payload: list[dict] | dict | None) -> tuple[pd.DataFrame, list[str]]:
    """Normalize FMP revenue segmentation payloads."""
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(), []

    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        data = item.get("data") or {}
        if not data:
            continue

        row = {
            "date": pd.to_datetime(item.get("date"), errors="coerce"),
            "fiscalYear": pd.to_numeric(item.get("fiscalYear"), errors="coerce"),
            "period": item.get("period"),
        }
        row.update({key: pd.to_numeric(value, errors="coerce") for key, value in data.items()})
        records.append(row)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame, []

    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    segment_columns = [column for column in frame.columns if column not in {"date", "fiscalYear", "period"}]
    if not segment_columns:
        return pd.DataFrame(), []

    segment_columns = (
        frame.loc[:, segment_columns]
        .fillna(0)
        .iloc[-1]
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    frame["totalSegmentRevenue"] = frame[segment_columns].sum(axis=1, min_count=1)
    return frame, segment_columns

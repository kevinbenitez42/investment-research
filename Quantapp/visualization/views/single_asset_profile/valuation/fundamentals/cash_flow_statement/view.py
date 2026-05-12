"""Cash flow statement fundamentals views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..._shared import apply_dark_yaxis, apply_standard_figure_layout, apply_subplot_x_grid


def plot_cash_flow_trends(cash_flow: pd.DataFrame, *, chart_label: str, period_label: str) -> go.Figure:
    period_title = period_label.title()
    is_annual = period_label.lower() == "annual"
    line_width = 2.5 if is_annual else 2.25
    secondary_line_width = 2.25 if is_annual else 2.0
    marker_size = 7 if is_annual else 6

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.30, 0.22, 0.24, 0.24],
        subplot_titles=(
            f"{period_title} Net Income, Operating Cash Flow, and Free Cash Flow",
            f"{period_title} Non-cash and Working-capital Drivers",
            f"{period_title} Investing, Financing, and Net Cash Change",
            f"{period_title} Cash Flow Growth Rates",
        ),
    )
    for metric_name, label, color in [
        ("netIncome", "Net income", "#60a5fa"),
        ("operatingCashFlow", "Operating cash flow", "#34d399"),
        ("freeCashFlow", "Free cash flow", "#f59e0b"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=cash_flow["date"],
                y=cash_flow[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": line_width},
                marker={"size": marker_size},
            ),
            row=1,
            col=1,
        )

    for metric_name, label, color, dash in [
        ("depreciationAndAmortization", "Depreciation and amortization", "#a78bfa", None),
        ("stockBasedCompensation", "Stock-based compensation", "#fb7185", None),
        ("changeInWorkingCapital", "Change in working capital", "#38bdf8", "dash"),
    ]:
        line_style = {"color": color, "width": secondary_line_width}
        if dash:
            line_style["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=cash_flow["date"],
                y=cash_flow[metric_name],
                name=label,
                mode="lines+markers",
                line=line_style,
                marker={"size": 6},
            ),
            row=2,
            col=1,
        )

    for metric_name, label, color in [
        ("netCashProvidedByInvestingActivities", "Investing cash flow", "#f97316"),
        ("netCashProvidedByFinancingActivities", "Financing cash flow", "#c084fc"),
        ("netChangeInCash", "Net change in cash", "#22c55e"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=cash_flow["date"],
                y=cash_flow[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": secondary_line_width},
                marker={"size": 6},
            ),
            row=3,
            col=1,
        )

    for metric_name, label, color in [
        ("netIncomeYoY", "Net income growth", "#60a5fa"),
        ("operatingCashFlowYoY", "Operating cash flow growth", "#34d399"),
        ("freeCashFlowYoY", "Free cash flow growth", "#f59e0b"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=cash_flow["date"],
                y=cash_flow[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": secondary_line_width},
                marker={"size": 6},
            ),
            row=4,
            col=1,
        )

    apply_standard_figure_layout(fig, f"{chart_label} {period_label.lower()} cash flow statement", 1460)
    apply_subplot_x_grid(fig, 4)
    apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", row=1)
    apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", zeroline=True, row=2)
    apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", zeroline=True, row=3)
    apply_dark_yaxis(fig, title_text="YoY growth", tickformat=".1%", zeroline=True, row=4)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def plot_cash_conversion_capital_allocation(
    annual_cash: pd.DataFrame,
    quarterly_cash: pd.DataFrame,
    *,
    chart_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Annual cash conversion ratios",
            "Quarterly and TTM cash conversion ratios",
            "Annual capital allocation as a share of FCF",
            "Quarterly and TTM capital allocation as a share of FCF",
            "Annual discretionary cash generation",
            "TTM discretionary cash generation",
        ),
    )

    for metric_name, label, color in [
        ("cfoToNetIncome", "CFO to net income", "#60a5fa"),
        ("capexToCfo", "Capex to CFO", "#f59e0b"),
        ("fcfToCfo", "FCF to CFO", "#34d399"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=annual_cash["date"],
                y=annual_cash[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                marker={"size": 7},
                legendgroup="conversion",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        ttm_metric_name = "ttm" + metric_name[0].upper() + metric_name[1:]
        fig.add_trace(
            go.Scatter(
                x=quarterly_cash["date"],
                y=quarterly_cash[ttm_metric_name] if ttm_metric_name in quarterly_cash.columns else quarterly_cash[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.25},
                marker={"size": 6},
                legendgroup="conversion",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    for metric_name, label, color in [
        ("buybacksToFcf", "Buybacks to FCF", "#fb7185"),
        ("dividendsToFcf", "Dividends to FCF", "#a78bfa"),
        ("shareholderReturnsToFcf", "Shareholder returns to FCF", "#38bdf8"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=annual_cash["date"],
                y=annual_cash[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.5},
                marker={"size": 7},
                legendgroup="allocation",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    for metric_name, source_name, label, color in [
        ("ttmBuybacksToFcf", "buybacksToFcf", "Buybacks to FCF", "#fb7185"),
        ("ttmDividendsToFcf", "dividendsToFcf", "Dividends to FCF", "#a78bfa"),
        ("ttmShareholderReturnsToFcf", "shareholderReturnsToFcf", "Shareholder returns to FCF", "#38bdf8"),
    ]:
        series = quarterly_cash[metric_name] if metric_name in quarterly_cash.columns else quarterly_cash[source_name]
        fig.add_trace(
            go.Scatter(
                x=quarterly_cash["date"],
                y=series,
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.25},
                marker={"size": 6},
                legendgroup="allocation",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    for frame, row_number, col_number, discretionary_column, fcf_column, cash_column in [
        (annual_cash, 3, 1, "discretionaryCashFlow", "freeCashFlow", "netChangeInCash"),
        (
            quarterly_cash.dropna(subset=["ttmDiscretionaryCashFlow"]),
            3,
            2,
            "ttmDiscretionaryCashFlow",
            "ttmFreeCashFlow",
            "ttmNetChangeInCash",
        ),
    ]:
        for metric_name, label, color in [
            (discretionary_column, "Discretionary cash flow", "#22c55e"),
            (fcf_column, "Free cash flow", "#f59e0b"),
            (cash_column, "Net change in cash", "#60a5fa"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[metric_name],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.25},
                    marker={"size": 7 if col_number == 1 else 6},
                    legendgroup="cash-generation",
                    showlegend=False,
                ),
                row=row_number,
                col=col_number,
            )

    apply_standard_figure_layout(fig, f"{chart_label} cash conversion and capital allocation", 1280)
    apply_subplot_x_grid(fig, 3, 2)
    for col_number in [1, 2]:
        apply_dark_yaxis(fig, title_text="Ratio", tickformat=",.2f", zeroline=True, row=1, col=col_number)
        apply_dark_yaxis(fig, title_text="Percent of FCF", tickformat=".1%", zeroline=True, row=2, col=col_number)
        apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", zeroline=True, row=3, col=col_number)
    return fig

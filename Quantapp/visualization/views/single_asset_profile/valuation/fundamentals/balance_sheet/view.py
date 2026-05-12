"""Balance sheet fundamentals views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..._shared import apply_dark_yaxis, apply_standard_figure_layout, apply_subplot_x_grid


def plot_balance_sheet_trends(balance: pd.DataFrame, *, chart_label: str, period_label: str) -> go.Figure:
    period_title = period_label.title()
    is_annual = period_label.lower() == "annual"
    line_width = 2.5 if is_annual else 2.25
    marker_size = 7 if is_annual else 6

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.30, 0.24, 0.23, 0.23],
        subplot_titles=(
            f"{period_title} Assets, Liabilities, and Equity",
            f"{period_title} Current Assets, Current Liabilities, and Working Capital",
            f"{period_title} Asset Mix as a Percent of Total Assets",
            f"{period_title} Balance Sheet Growth Rates",
        ),
    )
    for metric_name, label, color in [
        ("totalAssets", "Total assets", "#60a5fa"),
        ("totalLiabilities", "Total liabilities", "#f59e0b"),
        ("totalEquity", "Total equity", "#34d399"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=balance["date"],
                y=balance[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": line_width},
                marker={"size": marker_size},
            ),
            row=1,
            col=1,
        )

    for metric_name, label, color in [
        ("totalCurrentAssets", "Current assets", "#38bdf8"),
        ("totalCurrentLiabilities", "Current liabilities", "#fb7185"),
    ]:
        fig.add_trace(
            go.Bar(x=balance["date"], y=balance[metric_name], name=label, marker_color=color, opacity=0.55 if is_annual else 0.5),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=balance["date"],
            y=balance["workingCapital"],
            name="Working capital",
            mode="lines+markers",
            line={"color": "#a78bfa", "width": line_width},
            marker={"size": marker_size},
        ),
        row=2,
        col=1,
    )

    for metric_name, label, color in [
        ("cashAndShortTermInvestmentsPctAssets", "Cash and STI % assets", "#60a5fa"),
        ("receivablesPctAssets", "Receivables % assets", "#22c55e"),
        ("inventoryPctAssets", "Inventory % assets", "#f59e0b"),
        ("ppePctAssets", "Net PP&E % assets", "#f97316"),
        ("longTermInvestmentsPctAssets", "Long-term investments % assets", "#a78bfa"),
        ("goodwillIntangiblePctAssets", "Goodwill and intangibles % assets", "#fb7185"),
    ]:
        if metric_name in balance and balance[metric_name].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=balance["date"],
                    y=balance[metric_name],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.25 if is_annual else 2.0},
                    marker={"size": 6},
                ),
                row=3,
                col=1,
            )

    for metric_name, label, color in [
        ("totalAssetsYoY", "Assets growth", "#60a5fa"),
        ("totalLiabilitiesYoY", "Liabilities growth", "#f59e0b"),
        ("totalEquityYoY", "Equity growth", "#34d399"),
        ("totalDebtYoY", "Debt growth", "#fb7185"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=balance["date"],
                y=balance[metric_name],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2.25 if is_annual else 2.0},
                marker={"size": 6},
            ),
            row=4,
            col=1,
        )

    apply_standard_figure_layout(fig, f"{chart_label} {period_label.lower()} balance sheet", 1420)
    fig.update_layout(barmode="group")
    apply_subplot_x_grid(fig, 4)
    apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", row=1)
    apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", zeroline=True, row=2)
    apply_dark_yaxis(fig, title_text="Percent of assets", tickformat=".1%", zeroline=True, row=3)
    apply_dark_yaxis(fig, title_text="YoY growth", tickformat=".1%", zeroline=True, row=4)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig


def plot_liquidity_capital_structure(
    annual_balance: pd.DataFrame,
    quarterly_balance: pd.DataFrame,
    *,
    chart_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Annual cash versus debt",
            "Quarterly cash versus debt",
            "Annual liquidity ratios",
            "Quarterly liquidity ratios",
            "Annual capital structure percentages",
            "Quarterly capital structure percentages",
        ),
    )
    for frame, col_number in [(annual_balance, 1), (quarterly_balance, 2)]:
        marker_size = 7 if col_number == 1 else 6
        for metric_name, label, color in [
            ("cashAndShortTermInvestments", "Cash and STI", "#60a5fa"),
            ("totalDebt", "Total debt", "#f59e0b"),
            ("netDebt", "Net debt", "#f43f5e"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[metric_name],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.5},
                    marker={"size": marker_size},
                    legendgroup=metric_name,
                    showlegend=col_number == 1,
                ),
                row=1,
                col=col_number,
            )
        for metric_name, label, color in [
            ("currentRatio", "Current ratio", "#22c55e"),
            ("cashRatio", "Cash ratio", "#a78bfa"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[metric_name],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.25},
                    marker={"size": marker_size},
                    legendgroup=metric_name,
                    showlegend=False,
                ),
                row=2,
                col=col_number,
            )
        for metric_name, label, color in [
            ("debtToAssets", "Debt % assets", "#f59e0b"),
            ("liabilityToAssets", "Liabilities % assets", "#fb7185"),
            ("equityRatio", "Equity % assets", "#34d399"),
            ("workingCapitalPctAssets", "Working capital % assets", "#38bdf8"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[metric_name],
                    name=label,
                    mode="lines+markers",
                    line={"color": color, "width": 2.25},
                    marker={"size": marker_size},
                    legendgroup=metric_name,
                    showlegend=False,
                ),
                row=3,
                col=col_number,
            )

    apply_standard_figure_layout(fig, f"{chart_label} liquidity and capital structure", 1280)
    apply_subplot_x_grid(fig, 3, 2)
    for col_number in [1, 2]:
        apply_dark_yaxis(fig, title_text="Amount (USD)", tickformat="$,.3s", row=1, col=col_number)
        apply_dark_yaxis(fig, title_text="Ratio", tickformat=",.2f", zeroline=True, row=2, col=col_number)
        apply_dark_yaxis(fig, title_text="Percent of assets", tickformat=".1%", zeroline=True, row=3, col=col_number)
    return fig

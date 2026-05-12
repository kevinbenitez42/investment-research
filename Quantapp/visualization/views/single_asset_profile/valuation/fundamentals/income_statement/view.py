"""Income statement fundamentals views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..._shared import apply_dark_yaxis, apply_standard_figure_layout, apply_subplot_x_grid


def _line_style(color: str, width: float, dash: str | None = None) -> dict:
    style = {"color": color, "width": width}
    if dash:
        style["dash"] = dash
    return style


def _add_metric_trace(
    fig: go.Figure,
    frame: pd.DataFrame,
    metric_name: str,
    label: str,
    color: str,
    *,
    row: int,
    col: int = 1,
    width: float = 2.5,
    marker_size: int = 7,
    dash: str | None = None,
    legendgroup: str | None = None,
    showlegend: bool | None = None,
) -> None:
    if metric_name not in frame or not frame[metric_name].notna().any():
        return
    trace_kwargs = {}
    if showlegend is not None:
        trace_kwargs["showlegend"] = showlegend
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[metric_name],
            name=label,
            mode="lines+markers",
            line=_line_style(color, width, dash),
            marker={"size": marker_size},
            legendgroup=legendgroup,
            **trace_kwargs,
        ),
        row=row,
        col=col,
    )


def plot_income_statement_trends(
    revenue_frame: pd.DataFrame,
    *,
    chart_label: str,
    period_label: str,
) -> go.Figure:
    period_title = period_label.title()
    is_quarterly = period_label.lower() == "quarterly"
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.30, 0.22, 0.18, 0.15, 0.15],
        subplot_titles=(
            "Quarterly Revenue and TTM Revenue" if is_quarterly else "Annual Revenue",
            f"{period_title} Gross Profit, Operating Income, and Net Income",
            f"{period_title} Margins",
            f"{period_title} {'YoY ' if is_quarterly else ''}Growth Rates",
            f"{period_title} Margin Changes",
        ),
    )
    fig.add_trace(
        go.Bar(
            x=revenue_frame["date"],
            y=revenue_frame["revenue"],
            name=f"{period_title} revenue",
            marker_color="#f59e0b" if is_quarterly else "#60a5fa",
            opacity=0.45 if is_quarterly else 0.65,
        ),
        row=1,
        col=1,
    )
    if is_quarterly and "ttmRevenue" in revenue_frame:
        _add_metric_trace(
            fig,
            revenue_frame,
            "ttmRevenue",
            "TTM revenue",
            "#22c55e",
            row=1,
            width=3,
            marker_size=7,
        )

    label_prefix = period_title
    width = 2.5
    marker_size = 6 if is_quarterly else 7
    for metric_name, label, color in [
        ("grossProfit", "gross profit", "#a78bfa"),
        ("operatingIncome", "operating income", "#f97316"),
        ("netIncome", "net income", "#f43f5e"),
    ]:
        _add_metric_trace(fig, revenue_frame, metric_name, f"{label_prefix} {label}", color, row=2, marker_size=marker_size)
    for metric_name, label, color in [
        ("grossMargin", "gross margin", "#8b5cf6"),
        ("operatingMargin", "operating margin", "#fb923c"),
        ("netMargin", "net margin", "#fb7185"),
    ]:
        _add_metric_trace(fig, revenue_frame, metric_name, f"{label_prefix} {label}", color, row=3, marker_size=marker_size)
    for metric_name, label, color in [
        ("revenueYoY", "revenue growth", "#38bdf8"),
        ("grossProfitYoY", "gross profit growth", "#c084fc"),
        ("operatingIncomeYoY", "operating income growth", "#fb923c"),
        ("netIncomeYoY", "net income growth", "#fb7185"),
    ]:
        _add_metric_trace(fig, revenue_frame, metric_name, f"{label_prefix} {label}", color, row=4, marker_size=marker_size)
    for metric_name, label, color in [
        ("grossMarginChange", "gross margin change", "#8b5cf6"),
        ("operatingMarginChange", "operating margin change", "#f97316"),
        ("netMarginChange", "net margin change", "#f43f5e"),
    ]:
        _add_metric_trace(
            fig,
            revenue_frame,
            metric_name,
            f"{label_prefix} {label}",
            color,
            row=5,
            width=2,
            marker_size=6,
            dash="dash",
        )

    apply_standard_figure_layout(fig, f"{chart_label} {period_label.lower()} income statement", 1460)
    apply_subplot_x_grid(fig, 5)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    apply_dark_yaxis(fig, title_text="Revenue (USD)", tickformat="$,.3s", row=1)
    apply_dark_yaxis(fig, title_text="Income (USD)", tickformat="$,.3s", row=2)
    apply_dark_yaxis(fig, title_text="Margin", tickformat=".1%", row=3)
    apply_dark_yaxis(fig, title_text="Growth", tickformat=".1%", zeroline=True, row=4)
    apply_dark_yaxis(fig, title_text="Margin change", tickformat=".1%", zeroline=True, row=5)
    return fig


def plot_seasonal_growth_rates(quarterly_revenue: pd.DataFrame, *, chart_label: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Revenue YoY Growth by Fiscal Quarter",
            "Gross Profit YoY Growth by Fiscal Quarter",
            "Operating Income YoY Growth by Fiscal Quarter",
            "Net Income YoY Growth by Fiscal Quarter",
        ),
    )
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    quarter_colors = {"Q1": "#60a5fa", "Q2": "#34d399", "Q3": "#f59e0b", "Q4": "#f87171"}
    metric_rows = [
        ("revenueYoY", "Revenue growth", 1),
        ("grossProfitYoY", "Gross profit growth", 2),
        ("operatingIncomeYoY", "Operating income growth", 3),
        ("netIncomeYoY", "Net income growth", 4),
    ]
    for metric_name, metric_label, row_number in metric_rows:
        for quarter_label in quarter_order:
            seasonal_slice = quarterly_revenue.loc[quarterly_revenue["quarterLabel"] == quarter_label].copy()
            if seasonal_slice.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=seasonal_slice["quarterlyYear"],
                    y=seasonal_slice[metric_name],
                    mode="lines+markers",
                    name=f"{quarter_label} {metric_label}",
                    line={"color": quarter_colors[quarter_label], "width": 2.5},
                    marker={"size": 7},
                    legendgroup=quarter_label,
                    showlegend=row_number == 1,
                ),
                row=row_number,
                col=1,
            )
    apply_standard_figure_layout(fig, f"{chart_label} seasonal quarterly growth rates", 1120)
    apply_subplot_x_grid(fig, 4)
    for row_number in [1, 2, 3, 4]:
        fig.update_xaxes(dtick=1, row=row_number, col=1)
        apply_dark_yaxis(fig, title_text="YoY growth", tickformat=".1%", zeroline=True, row=row_number)
    fig.update_xaxes(title_text="Fiscal year", row=4, col=1)
    return fig


def plot_eps_dilution(
    annual_eps_frame: pd.DataFrame,
    quarterly_eps_frame: pd.DataFrame,
    *,
    analysis_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Annual EPS",
            "Quarterly EPS",
            "Annual weighted-average shares",
            "Quarterly weighted-average shares",
            "Annual dilution",
            "Quarterly dilution",
        ),
    )
    eps_colors = {"basic": "#60a5fa", "diluted": "#f59e0b", "dilution": "#34d399"}
    for frame, col_number in [(annual_eps_frame, 1), (quarterly_eps_frame, 2)]:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["eps"],
                mode="lines+markers",
                name="Basic EPS",
                line={"color": eps_colors["basic"], "width": 2.5},
                marker={"size": 7},
                legendgroup="eps",
                showlegend=col_number == 1,
            ),
            row=1,
            col=col_number,
        )
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["epsDiluted"],
                mode="lines+markers",
                name="Diluted EPS",
                line={"color": eps_colors["diluted"], "width": 2.5, "dash": "dash"},
                marker={"size": 7},
                legendgroup="eps",
                showlegend=col_number == 1,
            ),
            row=1,
            col=col_number,
        )
        for metric_name, label, color in [
            ("weightedAverageShsOut", "Basic shares", eps_colors["basic"]),
            ("weightedAverageShsOutDil", "Diluted shares", eps_colors["diluted"]),
        ]:
            fig.add_trace(
                go.Bar(
                    x=frame["date"],
                    y=frame[metric_name] / 1_000_000_000,
                    name=label,
                    marker_color=color,
                    opacity=0.75,
                    legendgroup="shares",
                    showlegend=col_number == 1,
                ),
                row=2,
                col=col_number,
            )
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["dilutionPct"],
                mode="lines+markers",
                name="Dilution %",
                line={"color": eps_colors["dilution"], "width": 2.5},
                marker={"size": 7},
                legendgroup="dilution",
                showlegend=col_number == 1,
            ),
            row=3,
            col=col_number,
        )

    apply_standard_figure_layout(fig, f"{analysis_label} EPS and dilution", 1260)
    fig.update_layout(barmode="group")
    apply_subplot_x_grid(fig, 3, 2)
    for col_number in [1, 2]:
        apply_dark_yaxis(fig, title_text="EPS", tickformat=",.2f", row=1, col=col_number)
        apply_dark_yaxis(fig, title_text="Shares (bn)", tickformat=",.1f", row=2, col=col_number)
        apply_dark_yaxis(fig, title_text="Dilution", tickformat=".2%", row=3, col=col_number)
    return fig


def plot_ttm_profit_conversion(ttm_frame: pd.DataFrame, *, analysis_label: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.36, 0.32, 0.32],
        subplot_titles=("TTM profit dollars", "TTM profit margins", "TTM YoY growth"),
    )
    for metric_name, metric_label, color in [
        ("ttmGrossProfit", "Gross profit", "#60a5fa"),
        ("ttmOperatingIncome", "Operating income", "#34d399"),
        ("ttmIncomeBeforeTax", "Pre-tax income", "#f59e0b"),
        ("ttmNetIncome", "Net income", "#f87171"),
    ]:
        _add_metric_trace(fig, ttm_frame, metric_name, metric_label, color, row=1, legendgroup="profit_dollars")
    for metric_name, metric_label, color in [
        ("ttmGrossMargin", "Gross margin", "#60a5fa"),
        ("ttmOperatingMargin", "Operating margin", "#34d399"),
        ("ttmIncomeBeforeTaxMargin", "Pre-tax margin", "#f59e0b"),
        ("ttmNetMargin", "Net margin", "#f87171"),
    ]:
        _add_metric_trace(fig, ttm_frame, metric_name, metric_label, color, row=2, legendgroup="profit_margin", showlegend=False)
    for metric_name, metric_label, color in [
        ("ttmRevenueYoY", "Revenue growth", "#c084fc"),
        ("ttmGrossProfitYoY", "Gross profit growth", "#60a5fa"),
        ("ttmOperatingIncomeYoY", "Operating income growth", "#34d399"),
        ("ttmIncomeBeforeTaxYoY", "Pre-tax income growth", "#f59e0b"),
        ("ttmNetIncomeYoY", "Net income growth", "#f87171"),
    ]:
        _add_metric_trace(fig, ttm_frame, metric_name, metric_label, color, row=3, legendgroup="profit_growth", showlegend=False)
    apply_standard_figure_layout(fig, f"{analysis_label} TTM profit conversion", 1220)
    apply_subplot_x_grid(fig, 3)
    apply_dark_yaxis(fig, title_text="Amount", tickformat=",.2s", row=1)
    apply_dark_yaxis(fig, title_text="Margin", tickformat=".1%", row=2)
    apply_dark_yaxis(fig, title_text="YoY growth", tickformat=".1%", zeroline=True, row=3)
    fig.update_xaxes(title_text="Quarter end", row=3, col=1)
    return fig


def plot_gross_margin_drivers(
    annual_revenue: pd.DataFrame,
    quarterly_revenue: pd.DataFrame,
    *,
    chart_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        row_heights=[0.38, 0.31, 0.31],
        subplot_titles=(
            "Annual Revenue vs Cost of revenue (COGS)",
            "Quarterly Revenue vs Cost of revenue (COGS)",
            "Annual Gross Margin and Cost of Revenue Ratio",
            "Quarterly Gross Margin and Cost of Revenue Ratio",
            "Annual Revenue Growth vs Cost of Revenue Growth",
            "Quarterly Revenue Growth vs Cost of Revenue Growth",
        ),
    )
    for frame, col_number, period, revenue_color, cost_color in [
        (annual_revenue, 1, "Annual", "#60a5fa", "#f97316"),
        (quarterly_revenue, 2, "Quarterly", "#38bdf8", "#fb923c"),
    ]:
        fig.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame["revenue"],
                name=f"{period} revenue",
                marker_color=revenue_color,
                opacity=0.55 if col_number == 1 else 0.5,
                legendgroup=f"{period.lower()}-revenue",
            ),
            row=1,
            col=col_number,
        )
        _add_metric_trace(
            fig,
            frame,
            "costOfRevenue",
            f"{period} cost of revenue (COGS)",
            cost_color,
            row=1,
            col=col_number,
            width=2.5 if col_number == 1 else 2.25,
            marker_size=7 if col_number == 1 else 6,
            legendgroup=f"{period.lower()}-cost-of-revenue",
        )
        _add_metric_trace(fig, frame, "grossMargin", f"{period} gross margin", "#a78bfa" if col_number == 1 else "#8b5cf6", row=2, col=col_number, marker_size=7 if col_number == 1 else 6, legendgroup="gross-margin")
        _add_metric_trace(fig, frame, "costOfRevenuePctRevenue", f"{period} cost of revenue % revenue", "#f43f5e" if col_number == 1 else "#fb7185", row=2, col=col_number, width=2.25, marker_size=6, dash="dash", legendgroup="cost-of-revenue-ratio")
        _add_metric_trace(fig, frame, "revenueYoY", f"{period} revenue growth", "#38bdf8", row=3, col=col_number, marker_size=7 if col_number == 1 else 6, legendgroup="growth")
        _add_metric_trace(fig, frame, "costOfRevenueYoY", f"{period} cost of revenue growth", cost_color, row=3, col=col_number, width=2.25, marker_size=6, dash="dash", legendgroup="growth")
    apply_standard_figure_layout(fig, f"{chart_label} gross margin drivers", 1320)
    _apply_driver_axes(fig)
    return fig


def plot_operating_margin_drivers(
    annual_revenue: pd.DataFrame,
    quarterly_revenue: pd.DataFrame,
    *,
    chart_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        row_heights=[0.38, 0.31, 0.31],
        subplot_titles=(
            "Annual Gross Profit vs Operating Expenses",
            "Quarterly Gross Profit vs Operating Expenses",
            "Annual Operating Margin Drivers",
            "Quarterly Operating Margin Drivers",
            "Annual Revenue Growth vs Operating Expense Growth",
            "Quarterly Revenue Growth vs Operating Expense Growth",
        ),
    )
    level_specs = [
        (annual_revenue, 1, "Annual", [("grossProfit", "gross profit", "#a78bfa"), ("operatingExpenses", "operating expenses", "#f97316"), ("operatingIncome", "operating income", "#22c55e")]),
        (quarterly_revenue, 2, "Quarterly", [("grossProfit", "gross profit", "#c084fc"), ("operatingExpenses", "operating expenses", "#fb923c"), ("operatingIncome", "operating income", "#4ade80")]),
    ]
    for frame, col_number, period, traces in level_specs:
        for metric_name, label, color in traces:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=1, col=col_number, width=2.5 if col_number == 1 else 2.25, marker_size=7 if col_number == 1 else 6, legendgroup=label)
    ratio_specs = [
        (annual_revenue, 1, "Annual", [("operatingMargin", "operating margin", "#38bdf8", None), ("operatingExpensesPctRevenue", "operating expenses % revenue", "#f97316", "dash"), ("sgaPctRevenue", "SG&A % revenue", "#f59e0b", None), ("rdPctRevenue", "R&D % revenue", "#fb7185", None)]),
        (quarterly_revenue, 2, "Quarterly", [("operatingMargin", "operating margin", "#0ea5e9", None), ("operatingExpensesPctRevenue", "operating expenses % revenue", "#fb923c", "dash"), ("sgaPctRevenue", "SG&A % revenue", "#fbbf24", None), ("rdPctRevenue", "R&D % revenue", "#fb7185", None)]),
    ]
    for frame, col_number, period, traces in ratio_specs:
        for metric_name, label, color, dash in traces:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=2, col=col_number, width=2.25 if "operating" in label else 2, marker_size=6, dash=dash, legendgroup=label)
    growth_specs = [
        (annual_revenue, 1, "Annual", "#f97316", "#f59e0b"),
        (quarterly_revenue, 2, "Quarterly", "#fb923c", "#fbbf24"),
    ]
    for frame, col_number, period, opex_color, sga_color in growth_specs:
        for metric_name, label, color, width, dash in [
            ("revenueYoY", "revenue growth", "#38bdf8", 2.5, None),
            ("operatingExpensesYoY", "operating expenses growth", opex_color, 2.25, "dash"),
            ("sgaYoY", "SG&A growth", sga_color, 2, None),
            ("rdYoY", "R&D growth", "#fb7185", 2, None),
        ]:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=3, col=col_number, width=width, marker_size=6, dash=dash, legendgroup="growth")
    apply_standard_figure_layout(fig, f"{chart_label} operating margin drivers", 1320)
    _apply_driver_axes(fig)
    return fig


def plot_net_margin_drivers(
    annual_revenue: pd.DataFrame,
    quarterly_revenue: pd.DataFrame,
    *,
    chart_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=2,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        row_heights=[0.38, 0.31, 0.31],
        subplot_titles=(
            "Annual Operating Income vs Pretax Income vs Net Income",
            "Quarterly Operating Income vs Pretax Income vs Net Income",
            "Annual Net Margin and Below-the-line Ratios",
            "Quarterly Net Margin and Below-the-line Ratios",
            "Annual Operating Income vs Pretax vs Net Income Growth",
            "Quarterly Operating Income vs Pretax vs Net Income Growth",
        ),
    )
    for frame, col_number, period, colors in [
        (annual_revenue, 1, "Annual", ("#22c55e", "#38bdf8", "#f43f5e")),
        (quarterly_revenue, 2, "Quarterly", ("#4ade80", "#0ea5e9", "#fb7185")),
    ]:
        for metric_name, label, color in [
            ("operatingIncome", "operating income", colors[0]),
            ("incomeBeforeTax", "pretax income", colors[1]),
            ("netIncome", "net income", colors[2]),
        ]:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=1, col=col_number, width=2.5 if col_number == 1 else 2.25, marker_size=7 if col_number == 1 else 6, legendgroup="income-levels")
    for frame, col_number, period, colors in [
        (annual_revenue, 1, "Annual", ("#22c55e", "#38bdf8", "#f43f5e", "#a78bfa", "#f97316")),
        (quarterly_revenue, 2, "Quarterly", ("#4ade80", "#0ea5e9", "#fb7185", "#c084fc", "#fb923c")),
    ]:
        for metric_name, label, color, dash, group in [
            ("operatingMargin", "operating margin", colors[0], None, "margin-ratios"),
            ("incomeBeforeTaxMargin", "pretax margin", colors[1], None, "margin-ratios"),
            ("netMargin", "net margin", colors[2], None, "margin-ratios"),
            ("otherIncomeExpensePctRevenue", "other income / expense % revenue", colors[3], "dash", "below-line-ratios"),
            ("incomeTaxExpensePctRevenue", "tax expense % revenue", colors[4], "dash", "below-line-ratios"),
        ]:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=2, col=col_number, width=2.25, marker_size=6, dash=dash, legendgroup=group)
    for frame, col_number, period, colors in [
        (annual_revenue, 1, "Annual", ("#22c55e", "#38bdf8", "#f43f5e", "#f97316")),
        (quarterly_revenue, 2, "Quarterly", ("#4ade80", "#0ea5e9", "#fb7185", "#fb923c")),
    ]:
        for metric_name, label, color, dash in [
            ("operatingIncomeYoY", "operating income growth", colors[0], None),
            ("incomeBeforeTaxYoY", "pretax income growth", colors[1], None),
            ("netIncomeYoY", "net income growth", colors[2], None),
            ("incomeTaxExpenseYoY", "tax expense growth", colors[3], "dash"),
        ]:
            _add_metric_trace(fig, frame, metric_name, f"{period} {label}", color, row=3, col=col_number, width=2.25, marker_size=6, dash=dash, legendgroup="growth")
    apply_standard_figure_layout(fig, f"{chart_label} net margin drivers", 1320)
    _apply_driver_axes(fig)
    return fig


def _apply_driver_axes(fig: go.Figure) -> None:
    apply_subplot_x_grid(fig, 3, 2)
    for col_number in [1, 2]:
        apply_dark_yaxis(fig, title_text="USD", tickformat="$,.3s", row=1, col=col_number)
        apply_dark_yaxis(fig, title_text="Percent of revenue", tickformat=".1%", zeroline=True, row=2, col=col_number)
        apply_dark_yaxis(fig, title_text="Growth", tickformat=".1%", zeroline=True, row=3, col=col_number)


def plot_revenue_segmentation(
    *,
    annual_product_segmentation: pd.DataFrame | None,
    annual_product_segments: list[str],
    annual_geographic_segmentation: pd.DataFrame | None,
    annual_geographic_segments: list[str],
    chart_label: str,
    symbol: str,
) -> go.Figure:
    segmentation_specs = []
    if annual_product_segments:
        segmentation_specs.append(
            (
                annual_product_segmentation,
                annual_product_segments,
                "Annual revenue by product",
                ["#60a5fa", "#38bdf8", "#34d399", "#f59e0b", "#f97316", "#f43f5e", "#a78bfa", "#818cf8"],
            )
        )
    if annual_geographic_segments:
        segmentation_specs.append(
            (
                annual_geographic_segmentation,
                annual_geographic_segments,
                "Annual revenue by geography",
                ["#22c55e", "#14b8a6", "#0ea5e9", "#8b5cf6", "#ec4899", "#f97316", "#eab308", "#fb7185"],
            )
        )
    if not segmentation_specs:
        raise RuntimeError(f"FMP did not return annual revenue segmentation for {symbol}.")

    fig = make_subplots(
        rows=len(segmentation_specs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12 if len(segmentation_specs) > 1 else 0.08,
        subplot_titles=tuple(spec[2] for spec in segmentation_specs),
    )
    for row_number, (segment_frame, segment_columns, subplot_title, palette) in enumerate(segmentation_specs, start=1):
        for color_index, segment_name in enumerate(segment_columns):
            fig.add_trace(
                go.Bar(
                    x=segment_frame["date"],
                    y=segment_frame[segment_name],
                    name=segment_name,
                    marker_color=palette[color_index % len(palette)],
                    legendgroup=subplot_title,
                ),
                row=row_number,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=segment_frame["date"],
                y=segment_frame["totalSegmentRevenue"],
                name=f"{subplot_title} total",
                mode="lines+markers",
                line={"color": "#e2e8f0", "width": 2.5},
                marker={"size": 7},
                legendgroup=f"{subplot_title}-total",
            ),
            row=row_number,
            col=1,
        )

    figure_height = 620 if len(segmentation_specs) == 1 else 1120
    apply_standard_figure_layout(fig, f"{chart_label} annual revenue segmentation", figure_height, bottom_margin=90)
    fig.update_layout(
        barmode="stack",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(2, 8, 23, 0.6)",
        },
        margin={"l": 60, "r": 240, "t": 120, "b": 80},
    )
    apply_subplot_x_grid(fig, len(segmentation_specs))
    for row_number in range(1, len(segmentation_specs) + 1):
        apply_dark_yaxis(fig, title_text="Revenue (USD)", tickformat="$,.3s", row=row_number)
    fig.update_xaxes(title_text="Date", row=len(segmentation_specs), col=1)
    return fig

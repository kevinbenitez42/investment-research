"""Analyst price-target fair value views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..._shared import apply_price_history_layout


def plot_analyst_price_target_band(
    plot_df: pd.DataFrame,
    latest_snapshot: pd.Series | dict,
    *,
    symbol: str,
    chart_label: str,
    rolling_window_days: int = 365,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["price"],
            name=f"{symbol} close",
            line={"color": "#e5eefc", "width": 2.5},
        )
    )

    for column, label, color, dash in [
        ("targetConsensus", f"Consensus target ({rolling_window_days}d trailing)", "#60a5fa", None),
        ("targetMedian", f"Median target ({rolling_window_days}d trailing)", "#fbbf24", None),
        ("targetHigh", f"High target ({rolling_window_days}d trailing)", "#4ade80", "dash"),
        ("targetLow", f"Low target ({rolling_window_days}d trailing)", "#f87171", "dash"),
    ]:
        trace_options = {
            "x": plot_df["date"],
            "y": plot_df[column],
            "name": label,
            "line": {"color": color, "width": 2.5},
        }
        if dash:
            trace_options["line"]["dash"] = dash
        if column == "targetLow":
            trace_options["fill"] = "tonexty"
            trace_options["fillcolor"] = "rgba(96, 165, 250, 0.14)"
        fig.add_trace(go.Scatter(**trace_options))

    fig.add_trace(
        go.Scatter(
            x=[plot_df["date"].iloc[-1]],
            y=[latest_snapshot["targetConsensus"]],
            mode="markers+text",
            name="Latest consensus API snapshot",
            marker={"color": "#38bdf8", "size": 11, "symbol": "diamond"},
            text=[f"API snapshot: {latest_snapshot['targetConsensus']:.2f}"],
            textfont={"color": "#dbeafe"},
            textposition="top center",
        )
    )
    apply_price_history_layout(fig, f"{chart_label} price vs analyst price target band")
    return fig


def plot_price_target_premium_discount(plot_df: pd.DataFrame, *, chart_label: str) -> go.Figure:
    historical_target_distance = plot_df.loc[
        :,
        ["date", "price", "targetLow", "targetMedian", "targetConsensus", "targetHigh"],
    ].copy()
    target_distance_columns = {
        "targetLow": "Low target",
        "targetMedian": "Median target",
        "targetConsensus": "Consensus target",
        "targetHigh": "High target",
    }
    for column_name, label in target_distance_columns.items():
        target_values = pd.to_numeric(historical_target_distance[column_name], errors="coerce").replace(0, pd.NA)
        historical_target_distance[label] = (historical_target_distance["price"] / target_values - 1) * 100

    relative_colors = {
        "Low target": "#f87171",
        "Median target": "#fbbf24",
        "Consensus target": "#60a5fa",
        "High target": "#4ade80",
    }
    fig = make_subplots(
        rows=len(relative_colors),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=list(relative_colors.keys()),
    )

    for row_number, (label, color) in enumerate(relative_colors.items(), start=1):
        trace_df = historical_target_distance.loc[:, ["date", label]].dropna().copy()
        trace_df["zero"] = 0.0
        trace_df["positive"] = trace_df[label].where(trace_df[label] >= 0)
        trace_df["negative"] = trace_df[label].where(trace_df[label] < 0)

        for y_column, fillcolor in [("zero", None), ("positive", "rgba(34, 197, 94, 0.22)")]:
            fig.add_trace(
                go.Scatter(
                    x=trace_df["date"],
                    y=trace_df[y_column],
                    mode="lines",
                    line={"color": "rgba(0, 0, 0, 0)", "width": 0},
                    fill="tonexty" if fillcolor else None,
                    fillcolor=fillcolor,
                    hoverinfo="skip",
                    showlegend=False,
                    connectgaps=False,
                ),
                row=row_number,
                col=1,
            )
        for y_column, fillcolor in [("zero", None), ("negative", "rgba(248, 113, 113, 0.22)")]:
            fig.add_trace(
                go.Scatter(
                    x=trace_df["date"],
                    y=trace_df[y_column],
                    mode="lines",
                    line={"color": "rgba(0, 0, 0, 0)", "width": 0},
                    fill="tonexty" if fillcolor else None,
                    fillcolor=fillcolor,
                    hoverinfo="skip",
                    showlegend=False,
                    connectgaps=False,
                ),
                row=row_number,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=trace_df["date"],
                y=trace_df[label],
                name=label,
                line={"color": color, "width": 2.5},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>" + label + "</extra>",
                showlegend=False,
            ),
            row=row_number,
            col=1,
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="#cbd5e1",
            opacity=0.8,
            row=row_number,
            col=1,
        )

        extrema = pd.to_numeric(trace_df[label], errors="coerce").dropna()
        if not extrema.empty:
            lowest_value = float(extrema.min())
            highest_value = float(extrema.max())
            for line_label, y_value, line_color in [
                ("Lowest", lowest_value, "#f87171"),
                ("Highest", highest_value, "#4ade80"),
            ]:
                fig.add_hline(
                    y=y_value,
                    line_dash="dash",
                    line_color=line_color,
                    opacity=0.7,
                    annotation_text=f"{line_label}: {y_value:.2f}%",
                    annotation_position="top right",
                    annotation_font={"color": line_color, "size": 10},
                    row=row_number,
                    col=1,
                )
        fig.update_yaxes(
            title_text=label,
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=False,
            automargin=True,
            row=row_number,
            col=1,
        )

    fig.update_layout(
        title=(
            f"{chart_label} price premium/discount to analyst target lines"
            "<br><sup>Green shading = price above target. Red shading = price below target.</sup>"
        ),
        template="plotly_dark",
        paper_bgcolor="#020817",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#0f172a", "font_color": "#e2e8f0"},
        autosize=True,
        height=1100,
        margin={"l": 90, "r": 30, "t": 110, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False, automargin=True)
    fig.update_xaxes(title_text="Date", row=len(relative_colors), col=1)
    return fig

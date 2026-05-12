"""Peer analysis valuation views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_relative_value_vs_peer_medians(relative_value: pd.DataFrame, *, company_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=relative_value["metric"],
            y=relative_value["discountToPeerMedianPct"],
            marker_color=relative_value["discountToPeerMedianPct"].apply(
                lambda value: "#22c55e" if value < 0 else "#f97316"
            ),
            hovertemplate="%{x}<br>Discount to peer median: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(
        title=f"{company_name} relative valuation vs peer medians",
        template="plotly_dark",
        height=620,
        margin={"l": 70, "r": 30, "t": 90, "b": 90},
    )
    fig.update_xaxes(title="Metric", tickangle=-20)
    fig.update_yaxes(title="Discount to peer median (%)")
    return fig


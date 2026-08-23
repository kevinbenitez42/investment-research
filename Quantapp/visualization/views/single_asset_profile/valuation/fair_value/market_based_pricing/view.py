"""Market-based fair value views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_market_based_implied_pricing(
    market_based_pricing: pd.DataFrame,
    *,
    current_price: float | None,
    chart_label: str,
) -> go.Figure | None:
    if market_based_pricing.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=market_based_pricing["Method"],
            y=market_based_pricing["Implied Price Median"],
            name="Peer median implied price",
            marker_color="#38bdf8",
            customdata=market_based_pricing[["Implied Price Low", "Implied Price High"]],
            hovertemplate=(
                "Method: %{x}<br>Median implied price: %{y:.2f}<br>"
                "Low-high range: %{customdata[0]:.2f} to %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    if pd.notna(current_price):
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#f97316",
            annotation_text=f"Current price: {current_price:.2f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{chart_label} market-based implied pricing",
        template="plotly_dark",
        paper_bgcolor="#020817",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#0f172a", "font_color": "#e2e8f0"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title="Method",
        yaxis_title="USD per share",
        autosize=True,
        height=680,
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False, automargin=True)
    return fig


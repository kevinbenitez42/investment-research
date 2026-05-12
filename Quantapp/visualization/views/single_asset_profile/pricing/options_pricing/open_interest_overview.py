"""Open-interest overview views for options-pricing workflows."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_open_interest_overview_view(
    expirations,
    total_oi_calls,
    total_oi_puts,
    total_oi_all,
    dte_list,
    *,
    template="plotly_white",
):
    """Compose total and aggregate open-interest panels by expiration."""
    exp_labels = [str(expiration) for expiration in expirations]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Total Call vs Put Open Interest", "Aggregate Open Interest (Calls + Puts)"),
    )
    fig.add_trace(
        go.Bar(
            x=exp_labels,
            y=total_oi_calls,
            name="Calls",
            marker_color="blue",
            hovertemplate="Expiration: %{x}<br>Total Call OI: %{y}<br>DTE: %{customdata}",
            customdata=dte_list,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=exp_labels,
            y=total_oi_puts,
            name="Puts",
            marker_color="green",
            hovertemplate="Expiration: %{x}<br>Total Put OI: %{y}<br>DTE: %{customdata}",
            customdata=dte_list,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=exp_labels,
            y=total_oi_all,
            name="Calls + Puts",
            marker_color="orange",
            hovertemplate="Expiration: %{x}<br>Aggregate OI: %{y}<br>DTE: %{customdata}",
            customdata=dte_list,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title="Options Open Interest Overview",
        xaxis_title="Option Expiration",
        yaxis_title="Open Interest",
        barmode="group",
        template=template,
        height=800,
    )
    fig.update_xaxes(title_text="Option Expiration", type="category", row=2, col=1)
    fig.update_xaxes(type="category", row=1, col=1)
    return fig

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
    put_call_ratio=None,
    *,
    template="plotly_white",
):
    """Compose total, aggregate, and put/call open-interest panels by expiration."""
    exp_labels = [str(expiration) for expiration in expirations]
    if put_call_ratio is None:
        put_call_ratio = [
            (put_oi / call_oi) if call_oi else float("nan")
            for call_oi, put_oi in zip(total_oi_calls, total_oi_puts)
        ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.42, 0.32, 0.26],
        subplot_titles=(
            "Total Call vs Put Open Interest",
            "Aggregate Open Interest (Calls + Puts)",
            "Put/Call Open Interest Ratio",
        ),
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
    fig.add_trace(
        go.Bar(
            x=exp_labels,
            y=put_call_ratio,
            name="Put/Call Ratio",
            marker_color="red",
            hovertemplate="Expiration: %{x}<br>Put/Call OI Ratio: %{y:.2f}<br>DTE: %{customdata}",
            customdata=dte_list,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=1, line_color="gray", line_dash="dash", row=3, col=1)
    fig.update_layout(
        title="Options Open Interest Overview",
        barmode="group",
        template=template,
        height=950,
    )
    fig.update_xaxes(title_text="Option Expiration", type="category", row=3, col=1)
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig.update_yaxes(title_text="Open Interest", row=2, col=1)
    fig.update_yaxes(title_text="Put/Call", tickformat=".2f", row=3, col=1)
    return fig

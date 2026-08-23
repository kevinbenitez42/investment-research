"""Option-chain table views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _in_the_money_colors(frame, fallback_color):
    if "inTheMoney" not in frame:
        return [fallback_color] * len(frame)
    return [
        "rgba(144, 238, 144, 0.6)" if bool(value) else fallback_color
        for value in frame["inTheMoney"]
    ]


def _table_trace(frame, *, header_color, fallback_cell_color):
    fill_colors = _in_the_money_colors(frame, fallback_cell_color)
    return go.Table(
        header=dict(
            values=list(frame.columns),
            fill_color=header_color,
            align="left",
            font=dict(size=11),
        ),
        cells=dict(
            values=[frame[column] for column in frame.columns],
            fill_color=[fill_colors for _ in frame.columns],
            align="left",
        ),
    )


def plot_option_chain_table_view(
    call_contracts_table,
    put_contracts_table,
    *,
    current_price,
    strikes_to_show,
    expiration_date,
    ticker_label,
):
    """Compose side-by-side call and put option-chain tables around the current price."""
    half_range = int(strikes_to_show) // 2
    min_strike = float(current_price) - half_range
    max_strike = float(current_price) + half_range

    calls = pd.DataFrame(call_contracts_table).copy()
    puts = pd.DataFrame(put_contracts_table).copy()
    filtered_calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
    filtered_puts = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Call Options", "Put Options"),
        specs=[[{"type": "table"}, {"type": "table"}]],
    )
    fig.add_trace(
        _table_trace(filtered_calls, header_color="paleturquoise", fallback_cell_color="lavender"),
        row=1,
        col=1,
    )
    fig.add_trace(
        _table_trace(filtered_puts, header_color="lightpink", fallback_cell_color="mistyrose"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title_text=(
            f"Option Chain for {ticker_label} - Expiration: {expiration_date}<br>"
            f"<span style='font-size:12px;'>Green = In The Money | "
            f"Showing strikes from ${min_strike:.1f} to ${max_strike:.1f}</span>"
        ),
        height=800,
        autosize=True,
    )
    return fig

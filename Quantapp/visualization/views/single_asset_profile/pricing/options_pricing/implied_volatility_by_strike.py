"""Implied-volatility by strike views."""

from __future__ import annotations

import plotly.graph_objects as go


def _strike_column(frame):
    if "StrikePrice" in frame:
        return "StrikePrice"
    return "strike"


def plot_implied_volatility_by_strike_view(
    calls_df,
    puts_df,
    *,
    current_price,
    ticker_label,
    expiration_date,
    template="plotly_white",
):
    """Compose call/put implied volatility across strike prices."""
    call_strike_column = _strike_column(calls_df)
    put_strike_column = _strike_column(puts_df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=calls_df[call_strike_column],
            y=calls_df["impliedVolatility"],
            mode="lines+markers",
            name="Calls",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=puts_df[put_strike_column],
            y=puts_df["impliedVolatility"],
            mode="lines+markers",
            name="Puts",
        )
    )
    fig.add_vline(
        x=current_price,
        line_color="red",
        line_dash="dash",
        annotation_text="Current Price",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Implied Volatility across Strike Prices for {ticker_label} Options (Expiration: {expiration_date})",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template=template,
        height=600,
        width=1200,
    )
    return fig

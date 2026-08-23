"""ATM implied-volatility and skew views."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _line_trace(frame, column, *, name, color, dash=None):
    line = dict(color=color)
    if dash is not None:
        line["dash"] = dash
    return go.Scatter(
        x=frame["Days Till Expiration"],
        y=frame[column],
        mode="lines+markers",
        name=name,
        line=line,
    )


def plot_atm_iv_realized_view(atm_df, *, ticker_label="Asset", template="plotly_white"):
    """Compose ATM IV, realized volatility, spread, and skew panels."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "ATM IV and Realized Volatility",
            "IV - Realized Vol Spread",
            "Put-Call IV Skew",
        ),
    )
    fig.add_trace(_line_trace(atm_df, "ATM IV Call", name="ATM IV Call", color="blue"), row=1, col=1)
    fig.add_trace(_line_trace(atm_df, "ATM IV Put", name="ATM IV Put", color="green"), row=1, col=1)
    fig.add_trace(_line_trace(atm_df, "Realized Vol", name="Realized Volatility", color="red"), row=1, col=1)
    fig.add_trace(_line_trace(atm_df, "IV-RV Call", name="IV-RV (Call)", color="blue", dash="dash"), row=2, col=1)
    fig.add_trace(_line_trace(atm_df, "IV-RV Put", name="IV-RV (Put)", color="green", dash="dash"), row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_dash="dash", row=2, col=1)
    fig.add_trace(_line_trace(atm_df, "IV Skew", name="Put-Call IV Skew", color="purple"), row=3, col=1)
    fig.update_layout(
        title=f"Volatility & IV Skew Analysis - {ticker_label}",
        xaxis_title="Days Till Expiration",
        yaxis_title="Volatility",
        height=1200,
        showlegend=True,
        template=template,
    )
    return fig

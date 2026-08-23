"""Median IV-minus-realized-volatility views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _series_trace(frame, *, option_type, name, color):
    side_frame = frame[frame["Type"] == option_type]
    return go.Scatter(
        x=side_frame["Expiration"],
        y=side_frame["Median"],
        mode="lines+markers",
        name=name,
        line=dict(color=color),
        hovertemplate="DTE: %{customdata} days<br>Median IV-Realized: %{y:.2%}",
        customdata=side_frame["DTE"],
    )


def _add_max_marker(fig, frame):
    if frame.empty or frame["Median"].dropna().empty:
        return
    max_row = frame.loc[frame["Median"].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[max_row["Expiration"]],
            y=[max_row["Median"]],
            mode="markers+text",
            text=["MAX"],
            textposition="top center",
            marker=dict(color="red", size=12),
            hovertemplate="DTE: %{customdata} days<br>Median IV-Realized: %{y:.2%}",
            customdata=[max_row["DTE"]],
            showlegend=False,
        ),
        row=1,
        col=1,
    )


def _ensure_dte(frame, today):
    frame = pd.DataFrame(frame).copy()
    if frame.empty:
        return frame
    if "DTE" not in frame:
        frame["DTE"] = (pd.to_datetime(frame["Expiration"]) - today).dt.days
    return frame


def plot_median_iv_minus_realized_view(
    df_all,
    df_otm,
    *,
    today=None,
    template="plotly_white",
):
    """Compose all-strike and OTM median IV-minus-realized-volatility panels."""
    today = pd.Timestamp.today().normalize() if today is None else pd.to_datetime(today)
    df_all = _ensure_dte(df_all, today)
    df_otm = _ensure_dte(df_otm, today)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("Median IV minus Realized Vol (All Strikes)", "OTM Median IV minus Realized Vol"),
    )
    fig.add_trace(_series_trace(df_all, option_type="Call", name="Calls", color="blue"), row=1, col=1)
    fig.add_trace(_series_trace(df_all, option_type="Put", name="Puts", color="green"), row=1, col=1)
    _add_max_marker(fig, df_all)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

    fig.add_trace(_series_trace(df_otm, option_type="Call_OTM", name="Calls (OTM)", color="blue"), row=2, col=1)
    fig.add_trace(_series_trace(df_otm, option_type="Put_OTM", name="Puts (OTM)", color="green"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    fig.update_layout(
        height=1000,
        title="Median IV minus Realized Vol: All Strikes vs OTM Strikes",
        template=template,
        xaxis=dict(type="date"),
        xaxis2=dict(type="date", title="Expiration"),
        yaxis=dict(title="Median IV - Realized Vol"),
        yaxis2=dict(title="Median IV - Realized Vol"),
    )
    return fig

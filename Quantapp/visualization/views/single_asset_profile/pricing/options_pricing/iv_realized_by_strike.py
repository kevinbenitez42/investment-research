"""IV-minus-realized-volatility by strike views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _trace_name(expiration, frame):
    if "expiration_label" in frame and frame["expiration_label"].notna().any():
        return str(frame["expiration_label"].dropna().iloc[0])
    if "days_till_expiration" in frame:
        days = int(round(float(frame["days_till_expiration"].iloc[0])))
    elif "Days Till Expiration" in frame:
        days = int(round(float(frame["Days Till Expiration"].iloc[0])))
    else:
        days = (pd.to_datetime(expiration) - pd.Timestamp.today()).days
    try:
        exp_label = pd.to_datetime(expiration).strftime("%b %d, %Y")
    except (TypeError, ValueError):
        exp_label = str(expiration)
    return f"{days}d - Exp: {exp_label}"


def _add_side_traces(fig, series_map, *, side_name, row, col):
    for idx, (expiration, frame) in enumerate(series_map.items()):
        if frame is None or frame.empty:
            continue
        hover_frame = pd.DataFrame(index=frame.index)
        hover_frame["contract_count"] = frame.get(
            "contract_count", pd.Series(float("nan"), index=frame.index)
        )
        hover_frame["days_till_expiration"] = frame.get(
            "days_till_expiration", pd.Series(float("nan"), index=frame.index)
        )
        hover_frame["dte_min"] = frame.get(
            "dte_min", hover_frame["days_till_expiration"]
        )
        hover_frame["dte_max"] = frame.get(
            "dte_max", hover_frame["days_till_expiration"]
        )
        hover_frame["moneyness_mid_pct"] = frame.get(
            "moneyness_mid_pct", pd.Series(float("nan"), index=frame.index)
        )
        hover_frame["avg_iv"] = frame.get(
            "avg_iv", pd.Series(float("nan"), index=frame.index)
        )
        hover_frame["avg_realized_vol"] = frame.get(
            "avg_realized_vol", pd.Series(float("nan"), index=frame.index)
        )
        fig.add_trace(
            go.Scatter(
                x=frame["strike"],
                y=frame["iv_minus_realized"],
                customdata=hover_frame.to_numpy(),
                mode="lines+markers",
                name=_trace_name(expiration, frame),
                legendgroup=f"{side_name}-{idx}",
                showlegend=True,
                hovertemplate=(
                    "Strike: %{x:,.2f}<br>"
                    "Avg IV - RV: %{y:+.2%}<br>"
                    "Contracts: %{customdata[0]:,.0f}<br>"
                    "Avg DTE: %{customdata[1]:.0f} "
                    "(%{customdata[2]:.0f}-%{customdata[3]:.0f})<br>"
                    "Moneyness bucket: %{customdata[4]:+.1%}<br>"
                    "Avg IV / RV: %{customdata[5]:.2%} / %{customdata[6]:.2%}"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=row,
            col=col,
        )


def plot_iv_minus_realized_by_strike_view(
    call_iv_minus_realized_by_expiration,
    put_iv_minus_realized_by_expiration,
    *,
    spot_price,
    template="plotly_white",
):
    """Compose call and put IV-minus-realized-volatility by strike panels."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calls: IV - Realized Vol", "Puts: IV - Realized Vol"),
    )
    _add_side_traces(fig, call_iv_minus_realized_by_expiration, side_name="Calls", row=1, col=1)
    _add_side_traces(fig, put_iv_minus_realized_by_expiration, side_name="Puts", row=1, col=2)

    for col in (1, 2):
        fig.add_hline(y=0, line_color="black", line_dash="dash", row=1, col=col)
        fig.add_vline(
            x=spot_price,
            line_color="red",
            line_dash="dash",
            annotation_text="Spot Price",
            annotation_position="top left",
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Clustered Average IV - Realized Volatility by Strike for Calls and Puts",
        height=1000,
        template=template,
        xaxis=dict(type="linear", title="Strike Price (Calls)"),
        xaxis2=dict(type="linear", title="Strike Price (Puts)"),
    )
    return fig

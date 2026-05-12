"""Open-interest and probability-of-touch range views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


def _strike_for_pot(spot, sigma, time_to_expiration, *, prob=0.15, option_type="call"):
    z_value = norm.ppf(1 - prob)
    if option_type == "call":
        return spot * np.exp(sigma * np.sqrt(time_to_expiration) * z_value)
    return spot * np.exp(-sigma * np.sqrt(time_to_expiration) * z_value)


def _open_interest_stats(chain_df):
    if chain_df is None or chain_df.empty or "openInterest" not in chain_df:
        return {"max_open_interest": 0.0, "max_open_interest_strike": None}

    open_interest = pd.to_numeric(chain_df["openInterest"], errors="coerce")
    valid_open_interest = open_interest.dropna()
    if valid_open_interest.empty:
        return {"max_open_interest": 0.0, "max_open_interest_strike": None}

    max_index = valid_open_interest.idxmax()
    return {
        "max_open_interest": float(valid_open_interest.loc[max_index]),
        "max_open_interest_strike": float(chain_df.loc[max_index, "strike"]),
    }


def _time_to_expiration(expiration, call_chain, put_chain):
    for chain_df in (call_chain, put_chain):
        if chain_df is None or chain_df.empty or "Days Till Expiration" not in chain_df:
            continue
        dte = pd.to_numeric(chain_df["Days Till Expiration"], errors="coerce").dropna()
        if not dte.empty:
            return max(float(dte.iloc[0]), 1.0) / 252
    fallback_days = max((pd.to_datetime(expiration) - pd.Timestamp.today().normalize()).days, 1)
    return fallback_days / 252


def _expiration_overlays(expiration, call_chain, put_chain, spot_price, annualized_vol):
    call_stats = _open_interest_stats(call_chain)
    put_stats = _open_interest_stats(put_chain)
    time_to_expiration = _time_to_expiration(expiration, call_chain, put_chain)

    call_lower_15 = _strike_for_pot(spot_price, annualized_vol, time_to_expiration, prob=0.15, option_type="call")
    put_upper_15 = _strike_for_pot(spot_price, annualized_vol, time_to_expiration, prob=0.15, option_type="put")
    call_lower_30 = _strike_for_pot(spot_price, annualized_vol, time_to_expiration, prob=0.30, option_type="call")
    put_upper_30 = _strike_for_pot(spot_price, annualized_vol, time_to_expiration, prob=0.30, option_type="put")

    y_max = max(call_stats["max_open_interest"], put_stats["max_open_interest"], 1.0) * 1.05
    shapes = [
        dict(
            type="line",
            x0=spot_price,
            x1=spot_price,
            y0=0,
            y1=y_max,
            line=dict(color="red", dash="dash", width=2),
        ),
        dict(
            type="rect",
            x0=put_upper_15,
            x1=call_lower_15,
            y0=0,
            y1=y_max,
            fillcolor="orange",
            opacity=0.2,
            layer="below",
            line_width=0,
        ),
        dict(
            type="rect",
            x0=put_upper_30,
            x1=call_lower_30,
            y0=0,
            y1=y_max,
            fillcolor="blue",
            opacity=0.2,
            layer="below",
            line_width=0,
        ),
    ]
    annotations = [
        dict(
            x=spot_price,
            y=y_max,
            text="Spot",
            showarrow=False,
            yshift=6,
            font=dict(color="red"),
        )
    ]

    if call_stats["max_open_interest_strike"] is not None:
        shapes.append(
            dict(
                type="line",
                x0=call_stats["max_open_interest_strike"],
                x1=call_stats["max_open_interest_strike"],
                y0=0,
                y1=y_max,
                line=dict(color="blue", dash="dot", width=2),
            )
        )
        annotations.append(
            dict(
                x=call_stats["max_open_interest_strike"],
                y=y_max,
                text="Max Call OI",
                showarrow=False,
                yshift=-14,
                font=dict(color="blue"),
            )
        )

    if put_stats["max_open_interest_strike"] is not None:
        shapes.append(
            dict(
                type="line",
                x0=put_stats["max_open_interest_strike"],
                x1=put_stats["max_open_interest_strike"],
                y0=0,
                y1=y_max,
                line=dict(color="green", dash="dot", width=2),
            )
        )
        annotations.append(
            dict(
                x=put_stats["max_open_interest_strike"],
                y=y_max,
                text="Max Put OI",
                showarrow=False,
                yshift=-34,
                font=dict(color="green"),
            )
        )
    return shapes, annotations


def _bar_trace(chain_df, *, name, color, visible):
    return go.Bar(
        x=chain_df["strike"],
        y=chain_df["openInterest"],
        name=name,
        marker_color=color,
        visible=visible,
    )


def plot_open_interest_pot_ranges_view(
    call_contract_chain,
    put_contract_chain,
    expirations,
    *,
    spot_price,
    annualized_vol,
    template="plotly_white",
):
    """Compose open-interest bars with spot and probability-of-touch ranges."""
    expirations = list(expirations)
    if not expirations:
        raise ValueError("No option expirations are available to plot.")

    fig = go.Figure()
    buttons = []
    first_expiration = expirations[0]

    for exp_index, expiration in enumerate(expirations):
        call_chain = call_contract_chain[expiration]
        put_chain = put_contract_chain[expiration]
        visible = exp_index == 0
        fig.add_trace(_bar_trace(call_chain, name="Calls", color="blue", visible=visible))
        fig.add_trace(_bar_trace(put_chain, name="Puts", color="green", visible=visible))

        trace_visibility = [False] * (len(expirations) * 2)
        trace_visibility[exp_index * 2] = True
        trace_visibility[exp_index * 2 + 1] = True
        shapes, annotations = _expiration_overlays(expiration, call_chain, put_chain, spot_price, annualized_vol)
        buttons.append(
            dict(
                label=str(expiration),
                method="update",
                args=[
                    {"visible": trace_visibility},
                    {
                        "title": f"Open Interest & PoT Ranges: {expiration}",
                        "shapes": shapes,
                        "annotations": annotations,
                    },
                ],
            )
        )

    first_shapes, first_annotations = _expiration_overlays(
        first_expiration,
        call_contract_chain[first_expiration],
        put_contract_chain[first_expiration],
        spot_price,
        annualized_vol,
    )
    fig.update_layout(
        title=f"Open Interest & PoT Ranges: {first_expiration}",
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.15, xanchor="left", yanchor="top")],
        barmode="group",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        template=template,
        height=600,
        shapes=first_shapes,
        annotations=first_annotations,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

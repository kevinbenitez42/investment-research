"""Open-interest and probability-of-touch range views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .implied_move import build_atm_implied_move_term_structure


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


def _days_to_expiration(expiration, call_chain, put_chain):
    for chain_df in (call_chain, put_chain):
        if chain_df is None or chain_df.empty or "Days Till Expiration" not in chain_df:
            continue
        dte = pd.to_numeric(chain_df["Days Till Expiration"], errors="coerce").dropna()
        if not dte.empty:
            return max(int(round(float(dte.iloc[0]))), 0)
    return max((pd.to_datetime(expiration) - pd.Timestamp.today().normalize()).days, 0)


def _expiration_overlays(expiration, call_chain, put_chain, spot_price):
    call_stats = _open_interest_stats(call_chain)
    put_stats = _open_interest_stats(put_chain)
    implied_move_frame = build_atm_implied_move_term_structure(
        {expiration: call_chain},
        {expiration: put_chain},
        [expiration],
        spot_price=spot_price,
    )

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
    ]
    range_annotations = []
    if not implied_move_frame.empty:
        implied_move = implied_move_frame.iloc[0]
        range_lower = float(implied_move["Straddle Lower"])
        range_upper = float(implied_move["Straddle Upper"])
        shapes.append(
            dict(
                type="rect",
                x0=range_lower,
                x1=range_upper,
                y0=0,
                y1=y_max,
                fillcolor="#F59E0B",
                opacity=0.2,
                layer="below",
                line=dict(color="#F59E0B", width=1, dash="dot"),
            )
        )
        range_label_style = dict(
            y=0,
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            bordercolor="#F59E0B",
            borderwidth=1,
            borderpad=3,
            bgcolor="rgba(17, 24, 39, 0.88)",
            font=dict(color="#FBBF24", size=11),
        )
        range_annotations.extend(
            [
                dict(
                    x=range_lower,
                    xanchor="right",
                    text=f"<b>Lower</b><br>${range_lower:,.2f}",
                    **range_label_style,
                ),
                dict(
                    x=range_upper,
                    xanchor="left",
                    text=f"<b>Upper</b><br>${range_upper:,.2f}",
                    **range_label_style,
                ),
            ]
        )
        spot_annotation = (
            "Spot"
            f"<br>ATM move: ±${implied_move['ATM Straddle']:,.2f} "
            f"(±{implied_move['Straddle Move %']:.2%})"
        )
    else:
        spot_annotation = "Spot<br>ATM move unavailable"

    annotations = [
        dict(
            x=spot_price,
            y=y_max,
            text=spot_annotation,
            showarrow=False,
            yshift=6,
            font=dict(color="red"),
        )
    ] + range_annotations

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


def plot_open_interest_implied_move_ranges_view(
    call_contract_chain,
    put_contract_chain,
    expirations,
    *,
    spot_price,
    template="plotly_white",
):
    """Compose open-interest bars with the selected expiration's ATM implied move."""
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
        shapes, annotations = _expiration_overlays(expiration, call_chain, put_chain, spot_price)
        days_to_expiration = _days_to_expiration(expiration, call_chain, put_chain)
        buttons.append(
            dict(
                label=f"{expiration} ({days_to_expiration} DTE)",
                method="update",
                args=[
                    {"visible": trace_visibility},
                    {
                        "title": f"Open Interest & ATM Implied Move: {expiration}",
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
    )
    fig.update_layout(
        title=f"Open Interest & ATM Implied Move: {first_expiration}",
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


def plot_open_interest_pot_ranges_view(
    call_contract_chain,
    put_contract_chain,
    expirations,
    *,
    spot_price,
    annualized_vol=None,
    template="plotly_white",
):
    """Backward-compatible alias for the former probability-of-touch view."""
    return plot_open_interest_implied_move_ranges_view(
        call_contract_chain,
        put_contract_chain,
        expirations,
        spot_price=spot_price,
        template=template,
    )

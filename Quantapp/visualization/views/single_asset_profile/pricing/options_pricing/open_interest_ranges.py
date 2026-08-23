"""Open-interest and probability-of-touch range views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _put_call_skew_trace(call_chain, put_chain, *, visible):
    """Return put IV minus call IV at strikes shared by both sides."""
    required_columns = {"strike", "impliedVolatility"}
    if not required_columns.issubset(call_chain.columns) or not required_columns.issubset(
        put_chain.columns
    ):
        return go.Scatter(x=[], y=[], name="Put - Call IV", visible=visible)

    calls = call_chain[["strike", "impliedVolatility"]].copy()
    puts = put_chain[["strike", "impliedVolatility"]].copy()
    for frame in (calls, puts):
        frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
        frame["impliedVolatility"] = pd.to_numeric(
            frame["impliedVolatility"], errors="coerce"
        )
        frame.dropna(inplace=True)

    skew = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
    skew["put_call_skew"] = (
        skew["impliedVolatility_put"] - skew["impliedVolatility_call"]
    )
    skew.sort_values("strike", inplace=True)
    return go.Scatter(
        x=skew["strike"],
        y=skew["put_call_skew"],
        customdata=skew[["impliedVolatility_put", "impliedVolatility_call"]],
        mode="lines+markers",
        name="Put - Call IV",
        line=dict(color="#A855F7"),
        visible=visible,
        hovertemplate=(
            "Strike: %{x:,.2f}<br>Put-call skew: %{y:+.2%}<br>"
            "Put IV: %{customdata[0]:.2%}<br>Call IV: %{customdata[1]:.2%}"
            "<extra></extra>"
        ),
    )


def _subplot_overlays(expiration, call_chain, put_chain, spot_price):
    """Place the original implied-move/OI overlays on the upper subplot."""
    shapes, annotations = _expiration_overlays(
        expiration, call_chain, put_chain, spot_price
    )
    for shape in shapes:
        shape.update(xref="x", yref="y")
    for annotation in annotations:
        annotation.update(xref="x", yref="y")

    shapes.extend(
        [
            dict(
                type="line",
                x0=spot_price,
                x1=spot_price,
                y0=0,
                y1=1,
                xref="x2",
                yref="y2 domain",
                line=dict(color="red", dash="dash", width=2),
            ),
            dict(
                type="line",
                x0=0,
                x1=1,
                y0=0,
                y1=0,
                xref="x2 domain",
                yref="y2",
                line=dict(color="#94A3B8", dash="dash", width=1),
            ),
        ]
    )
    return shapes, annotations


def plot_open_interest_implied_move_ranges_view(
    call_contract_chain,
    put_contract_chain,
    expirations,
    *,
    spot_price,
    template="plotly_white",
):
    """Plot one DTE at a time, with call/put OI above put-call IV skew."""
    expirations = list(expirations)
    if not expirations:
        raise ValueError("No option expirations are available to plot.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.68, 0.32],
        subplot_titles=("Call and Put Open Interest", "Put-Call IV Skew"),
    )
    buttons = []
    first_expiration = expirations[0]

    for exp_index, expiration in enumerate(expirations):
        call_chain = call_contract_chain[expiration]
        put_chain = put_contract_chain[expiration]
        visible = exp_index == 0
        fig.add_trace(
            _bar_trace(call_chain, name="Calls", color="blue", visible=visible), row=1, col=1
        )
        fig.add_trace(
            _bar_trace(put_chain, name="Puts", color="green", visible=visible), row=1, col=1
        )
        fig.add_trace(
            _put_call_skew_trace(call_chain, put_chain, visible=visible), row=2, col=1
        )

        trace_visibility = [False] * (len(expirations) * 3)
        trace_visibility[exp_index * 3 : exp_index * 3 + 3] = [True, True, True]
        days_to_expiration = _days_to_expiration(expiration, call_chain, put_chain)
        shapes, annotations = _subplot_overlays(
            expiration, call_chain, put_chain, spot_price
        )
        buttons.append(
            dict(
                label=f"{expiration} ({days_to_expiration} DTE)",
                method="update",
                args=[
                    {"visible": trace_visibility},
                    {
                        "title": f"Open Interest & Put-Call Skew: {days_to_expiration} DTE ({expiration})",
                        "shapes": shapes,
                        "annotations": annotations,
                    },
                ],
            )
        )

    first_dte = _days_to_expiration(
        first_expiration, call_contract_chain[first_expiration], put_contract_chain[first_expiration]
    )
    first_shapes, first_annotations = _subplot_overlays(
        first_expiration,
        call_contract_chain[first_expiration],
        put_contract_chain[first_expiration],
        spot_price,
    )
    fig.update_layout(
        title=f"Open Interest & Put-Call Skew: {first_dte} DTE ({first_expiration})",
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.15, xanchor="left", yanchor="top")],
        barmode="group",
        template=template,
        height=850,
        shapes=first_shapes,
        annotations=first_annotations,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_yaxes(title_text="Open Interest", row=1, col=1)
    fig.update_yaxes(title_text="Put IV - Call IV", tickformat="+.1%", row=2, col=1)
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

"""ATM option-price and volatility implied-move term structures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


IMPLIED_MOVE_COLUMNS = [
    "Expiration Date",
    "DTE",
    "ATM Strike",
    "Call Mid",
    "Put Mid",
    "ATM Straddle",
    "Straddle Move %",
    "Straddle Lower",
    "Straddle Upper",
    "ATM IV",
    "IV Move",
    "IV Move %",
    "Quote Source",
]


def _numeric(value) -> float:
    numeric_value = pd.to_numeric(value, errors="coerce")
    try:
        return float(numeric_value)
    except (TypeError, ValueError):
        return np.nan


def _quote_mid(contract: pd.Series) -> tuple[float, str]:
    bid = _numeric(contract.get("bid"))
    ask = _numeric(contract.get("ask"))
    if np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask and ask > 0:
        return (bid + ask) / 2.0, "bid/ask midpoint"

    last_price = _numeric(contract.get("lastPrice"))
    if np.isfinite(last_price) and last_price > 0:
        return last_price, "last price fallback"
    return np.nan, "unavailable"


def _expiration_dte(expiration, call_contract: pd.Series, put_contract: pd.Series) -> int:
    dte_candidates = [
        _numeric(call_contract.get("Days Till Expiration")),
        _numeric(put_contract.get("Days Till Expiration")),
    ]
    dte_candidates = [value for value in dte_candidates if np.isfinite(value)]
    if dte_candidates:
        return max(int(round(dte_candidates[0])), 0)
    return max((pd.to_datetime(expiration) - pd.Timestamp.today().normalize()).days, 0)


def build_atm_implied_move_term_structure(
    call_contract_chain: Mapping,
    put_contract_chain: Mapping,
    expirations: Sequence,
    *,
    spot_price: float,
) -> pd.DataFrame:
    """Build one paired ATM straddle and ATM-IV move observation per expiration."""
    spot_price = float(spot_price)
    if not np.isfinite(spot_price) or spot_price <= 0:
        raise ValueError("spot_price must be a positive finite number.")

    rows = []
    for expiration in expirations:
        calls = call_contract_chain.get(expiration)
        puts = put_contract_chain.get(expiration)
        if calls is None or puts is None or calls.empty or puts.empty:
            continue

        calls = calls.copy()
        puts = puts.copy()
        calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
        puts["strike"] = pd.to_numeric(puts["strike"], errors="coerce")
        calls = calls.dropna(subset=["strike"])
        puts = puts.dropna(subset=["strike"])

        common_strikes = sorted(
            set(calls["strike"]).intersection(puts["strike"]),
            key=lambda strike: abs(strike - spot_price),
        )
        selected_contracts = None
        for atm_strike in common_strikes:
            call_contract = calls.loc[calls["strike"].eq(atm_strike)].iloc[0]
            put_contract = puts.loc[puts["strike"].eq(atm_strike)].iloc[0]
            call_mid, call_quote_source = _quote_mid(call_contract)
            put_mid, put_quote_source = _quote_mid(put_contract)
            if np.isfinite(call_mid) and np.isfinite(put_mid):
                selected_contracts = (
                    atm_strike,
                    call_contract,
                    put_contract,
                    call_mid,
                    put_mid,
                    call_quote_source,
                    put_quote_source,
                )
                break

        if selected_contracts is None:
            continue

        (
            atm_strike,
            call_contract,
            put_contract,
            call_mid,
            put_mid,
            call_quote_source,
            put_quote_source,
        ) = selected_contracts

        dte = _expiration_dte(expiration, call_contract, put_contract)
        iv_values = [
            _numeric(call_contract.get("impliedVolatility")),
            _numeric(put_contract.get("impliedVolatility")),
        ]
        iv_values = [value for value in iv_values if np.isfinite(value) and value > 0]
        atm_iv = float(np.mean(iv_values)) if iv_values else np.nan

        straddle_move = call_mid + put_mid
        straddle_move_pct = straddle_move / spot_price
        iv_move_pct = atm_iv * np.sqrt(max(dte, 1) / 365.25) if np.isfinite(atm_iv) else np.nan

        rows.append(
            {
                "Expiration Date": pd.to_datetime(expiration),
                "DTE": dte,
                "ATM Strike": float(atm_strike),
                "Call Mid": call_mid,
                "Put Mid": put_mid,
                "ATM Straddle": straddle_move,
                "Straddle Move %": straddle_move_pct,
                "Straddle Lower": max(spot_price - straddle_move, 0.0),
                "Straddle Upper": spot_price + straddle_move,
                "ATM IV": atm_iv,
                "IV Move": spot_price * iv_move_pct if np.isfinite(iv_move_pct) else np.nan,
                "IV Move %": iv_move_pct,
                "Quote Source": f"Call: {call_quote_source}; Put: {put_quote_source}",
            }
        )

    return (
        pd.DataFrame(rows, columns=IMPLIED_MOVE_COLUMNS)
        .sort_values(["DTE", "Expiration Date"])
        .reset_index(drop=True)
    )


def plot_atm_implied_move_term_structure_view(
    implied_move_frame: pd.DataFrame,
    *,
    spot_price: float,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot ATM-straddle price bounds and move percentages across expirations."""
    if implied_move_frame is None or implied_move_frame.empty:
        raise ValueError("No usable paired ATM call and put quotes were available.")

    frame = implied_move_frame.sort_values(["DTE", "Expiration Date"]).reset_index(drop=True)
    x_positions = list(range(len(frame)))
    tick_labels = [
        f"{expiration:%Y-%m-%d}<br>{int(dte)} DTE"
        for expiration, dte in zip(frame["Expiration Date"], frame["DTE"])
    ]

    hover_data = frame[
        [
            "Expiration Date",
            "DTE",
            "ATM Strike",
            "Call Mid",
            "Put Mid",
            "ATM Straddle",
            "ATM IV",
            "IV Move",
            "Quote Source",
        ]
    ].copy()
    hover_data["Expiration Date"] = pd.to_datetime(hover_data["Expiration Date"]).dt.strftime("%Y-%m-%d")
    customdata = hover_data.to_numpy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.60, 0.40],
        subplot_titles=(
            "ATM Straddle-Implied Price Range",
            "Implied Move as a Percentage of Spot",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=frame["Straddle Lower"],
            mode="lines+markers",
            name="Straddle lower bound",
            showlegend=False,
            line=dict(color="#F59E0B"),
            marker=dict(size=6),
            customdata=customdata,
            hovertemplate=(
                "Expiration: %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "ATM strike: $%{customdata[2]:,.2f}<br>"
                "Call / put mid: $%{customdata[3]:,.2f} / $%{customdata[4]:,.2f}<br>"
                "ATM straddle: $%{customdata[5]:,.2f}<br>"
                "Lower bound: $%{y:,.2f}<br>"
                "%{customdata[8]}<extra>%{fullData.name}</extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=frame["Straddle Upper"],
            mode="lines+markers",
            name="ATM straddle range",
            line=dict(color="#F59E0B"),
            marker=dict(size=6),
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.18)",
            customdata=customdata,
            hovertemplate=(
                "Expiration: %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "ATM strike: $%{customdata[2]:,.2f}<br>"
                "ATM straddle: $%{customdata[5]:,.2f}<br>"
                "Upper bound: $%{y:,.2f}<extra>%{fullData.name}</extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[spot_price] * len(x_positions),
            mode="lines",
            name="Current spot",
            line=dict(color="#F87171", dash="dash", width=2),
            hovertemplate="Spot: $%{y:,.2f}<extra>%{fullData.name}</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=frame["Straddle Move %"],
            mode="lines+markers",
            name="ATM straddle / spot",
            line=dict(color="#F59E0B", width=3),
            customdata=customdata,
            hovertemplate=(
                "Expiration: %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "Price-implied move: %{y:.2%}<br>"
                "ATM straddle: $%{customdata[5]:,.2f}<extra>%{fullData.name}</extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=frame["IV Move %"],
            mode="lines+markers",
            name="ATM IV one-sigma move",
            line=dict(color="#22D3EE", dash="dot", width=2),
            customdata=customdata,
            hovertemplate=(
                "Expiration: %{customdata[0]} (%{customdata[1]} DTE)<br>"
                "ATM IV: %{customdata[6]:.2%}<br>"
                "IV one-sigma move: %{y:.2%} ($%{customdata[7]:,.2f})"
                "<extra>%{fullData.name}</extra>"
            ),
        ),
        row=2,
        col=1,
    )

    title_suffix = f": {ticker_label}" if ticker_label else ""
    fig.update_layout(
        title=f"ATM Implied Move by Expiration{title_suffix}",
        template=template,
        height=760,
        hovermode="x",
        margin=dict(t=110, b=130),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=tick_labels,
        tickangle=-45,
        automargin=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Underlying Price", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Implied Move", tickformat=".1%", row=2, col=1)
    return fig

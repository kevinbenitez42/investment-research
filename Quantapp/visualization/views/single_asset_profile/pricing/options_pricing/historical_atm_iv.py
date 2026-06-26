"""Historical ATM implied-volatility calculations and views."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy.stats import norm


_REQUIRED_COLUMNS = {
    "as_of_date",
    "Expiration Date",
    "strike",
    "Type",
    "Days Till Expiration",
    "lastPrice",
    "day_spot",
}


def _black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    annual_rate: float,
    volatility: float,
    option_type: str,
) -> float:
    volatility_time = volatility * np.sqrt(time_years)
    d1 = (
        np.log(spot / strike)
        + (annual_rate + 0.5 * volatility**2) * time_years
    ) / volatility_time
    d2 = d1 - volatility_time
    discounted_strike = strike * np.exp(-annual_rate * time_years)
    if option_type == "Call":
        return float(spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2))
    return float(discounted_strike * norm.cdf(-d2) - spot * norm.cdf(-d1))


def _implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_years: float,
    annual_rate: float,
    option_type: str,
) -> float:
    if option_type not in {"Call", "Put"}:
        return np.nan
    if not all(
        np.isfinite(value) and value > 0
        for value in (option_price, spot, strike, time_years)
    ):
        return np.nan

    discounted_strike = strike * np.exp(-annual_rate * time_years)
    if option_type == "Call":
        lower_bound = max(spot - discounted_strike, 0.0)
        upper_bound = spot
    else:
        lower_bound = max(discounted_strike - spot, 0.0)
        upper_bound = discounted_strike

    price_tolerance = max(1e-8, spot * 1e-10)
    if option_price <= lower_bound + price_tolerance or option_price >= upper_bound:
        return np.nan

    objective = lambda volatility: (
        _black_scholes_price(
            spot,
            strike,
            time_years,
            annual_rate,
            volatility,
            option_type,
        )
        - option_price
    )
    try:
        return float(brentq(objective, 1e-6, 10.0, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan


def build_historical_atm_iv_history(
    historical_chain: pd.DataFrame,
    *,
    annual_rate: float = 0.02,
    min_dte: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ATM IV per date/expiration and cross-expiration daily summaries."""
    if historical_chain is None or historical_chain.empty:
        return pd.DataFrame(), pd.DataFrame()

    missing_columns = sorted(_REQUIRED_COLUMNS - set(historical_chain.columns))
    if missing_columns:
        raise ValueError(
            "Historical options chain is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = historical_chain.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["Expiration Date"] = pd.to_datetime(
        frame["Expiration Date"], errors="coerce"
    )
    for column in ("strike", "Days Till Expiration", "lastPrice", "day_spot"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Type"] = frame["Type"].astype(str).str.title()
    frame = frame.dropna(
        subset=[
            "as_of_date",
            "Expiration Date",
            "strike",
            "Days Till Expiration",
            "lastPrice",
            "day_spot",
        ]
    )
    frame = frame[
        frame["Type"].isin(["Call", "Put"])
        & frame["Days Till Expiration"].ge(min_dte)
        & frame["lastPrice"].gt(0)
        & frame["day_spot"].gt(0)
        & frame["strike"].gt(0)
    ].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame["ATM Distance"] = (frame["strike"] - frame["day_spot"]).abs()
    atm_contracts = (
        frame.sort_values(
            [
                "as_of_date",
                "Expiration Date",
                "Type",
                "ATM Distance",
                "strike",
            ]
        )
        .groupby(
            ["as_of_date", "Expiration Date", "Type"],
            sort=True,
            as_index=False,
            group_keys=False,
        )
        .head(1)
        .copy()
    )
    atm_contracts["ATM IV"] = atm_contracts.apply(
        lambda row: _implied_volatility(
            option_price=float(row["lastPrice"]),
            spot=float(row["day_spot"]),
            strike=float(row["strike"]),
            time_years=max(float(row["Days Till Expiration"]), 1.0) / 365.25,
            annual_rate=float(annual_rate),
            option_type=row["Type"],
        ),
        axis=1,
    )
    atm_contracts = atm_contracts[
        np.isfinite(atm_contracts["ATM IV"])
        & atm_contracts["ATM IV"].between(0.001, 5.0)
    ].copy()
    if atm_contracts.empty:
        return pd.DataFrame(), pd.DataFrame()

    atm_contracts["ATM Call IV"] = atm_contracts["ATM IV"].where(
        atm_contracts["Type"].eq("Call")
    )
    atm_contracts["ATM Put IV"] = atm_contracts["ATM IV"].where(
        atm_contracts["Type"].eq("Put")
    )
    expiration_history = (
        atm_contracts.groupby(
            ["as_of_date", "Expiration Date"], as_index=False, sort=True
        )
        .agg(
            **{
                "ATM IV": ("ATM IV", "mean"),
                "ATM Call IV": ("ATM Call IV", "mean"),
                "ATM Put IV": ("ATM Put IV", "mean"),
                "DTE": ("Days Till Expiration", "median"),
                "Contracts Used": ("ATM IV", "size"),
            }
        )
        .sort_values(["as_of_date", "DTE", "Expiration Date"])
        .reset_index(drop=True)
    )

    daily_summary = (
        expiration_history.groupby("as_of_date", as_index=False, sort=True)
        .agg(
            **{
                "Mean ATM IV": ("ATM IV", "mean"),
                "Median ATM IV": ("ATM IV", "median"),
                "Expirations Used": ("Expiration Date", "nunique"),
            }
        )
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )
    return expiration_history, daily_summary


def plot_historical_atm_iv_summary_view(
    daily_summary: pd.DataFrame,
    *,
    expiration_history: pd.DataFrame | None = None,
    target_dtes: Sequence[int] | None = None,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot mean/median ATM IV against a dropdown-selected target DTE."""
    required_columns = {
        "as_of_date",
        "Mean ATM IV",
        "Median ATM IV",
        "Expirations Used",
    }
    if daily_summary is None or daily_summary.empty:
        raise ValueError("No historical ATM implied-volatility observations are available.")
    missing_columns = sorted(required_columns - set(daily_summary.columns))
    if missing_columns:
        raise ValueError(
            "Historical ATM IV summary is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = daily_summary.sort_values("as_of_date").copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date"])
    frame = frame[
        pd.to_numeric(frame["Expirations Used"], errors="coerce").ge(3)
    ].copy()
    if frame.empty:
        raise ValueError(
            "At least three valid expirations per date are required to compare "
            "selected-DTE ATM IV with cross-expiration mean and median benchmarks."
        )
    summary_hover_data = np.column_stack([frame["Expirations Used"].astype(int)])
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            "ATM IV Level Comparison",
            "Selected DTE ATM IV Minus Fixed-Tenor Benchmarks",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["as_of_date"],
            y=frame["Mean ATM IV"],
            customdata=summary_hover_data,
            mode="lines+markers",
            name="Fixed-Tenor Mean ATM IV",
            line=dict(color="#38BDF8", width=2.5),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Fixed-Tenor Mean ATM IV: %{y:.2%}<br>"
                "Expirations: %{customdata[0]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["as_of_date"],
            y=frame["Median ATM IV"],
            customdata=summary_hover_data,
            mode="lines+markers",
            name="Fixed-Tenor Median ATM IV",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Fixed-Tenor Median ATM IV: %{y:.2%}<br>"
                "Expirations: %{customdata[0]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    normalized_target_dtes = sorted(
        {
            int(target_dte)
            for target_dte in (target_dtes or [])
            if int(target_dte) > 0
        }
    )
    if expiration_history is None or expiration_history.empty:
        normalized_target_dtes = []
    elif not normalized_target_dtes:
        normalized_target_dtes = sorted(
            pd.to_numeric(expiration_history["DTE"], errors="coerce")
            .dropna()
            .round()
            .astype(int)
            .unique()
            .tolist()
        )

    target_trace_starts = {}
    default_target_dte = (
        30 if 30 in normalized_target_dtes else normalized_target_dtes[0]
        if normalized_target_dtes
        else None
    )
    if normalized_target_dtes:
        required_expiration_columns = {
            "as_of_date",
            "Expiration Date",
            "DTE",
            "ATM IV",
        }
        missing_expiration_columns = sorted(
            required_expiration_columns - set(expiration_history.columns)
        )
        if missing_expiration_columns:
            raise ValueError(
                "Historical ATM IV expiration data is missing required columns: "
                + ", ".join(missing_expiration_columns)
            )

        expiration_frame = expiration_history.copy()
        expiration_frame["as_of_date"] = pd.to_datetime(
            expiration_frame["as_of_date"], errors="coerce"
        )
        expiration_frame["Expiration Date"] = pd.to_datetime(
            expiration_frame["Expiration Date"], errors="coerce"
        )
        expiration_frame["DTE"] = pd.to_numeric(
            expiration_frame["DTE"], errors="coerce"
        )
        expiration_frame["ATM IV"] = pd.to_numeric(
            expiration_frame["ATM IV"], errors="coerce"
        )
        expiration_frame = expiration_frame.dropna(
            subset=["as_of_date", "Expiration Date", "DTE", "ATM IV"]
        )

        for target_dte in normalized_target_dtes:
            selected_dte_frame = expiration_frame.copy()
            selected_dte_frame["DTE Distance"] = (
                selected_dte_frame["DTE"] - target_dte
            ).abs()
            selected_dte_frame = (
                selected_dte_frame.sort_values(
                    ["as_of_date", "DTE Distance", "DTE", "Expiration Date"]
                )
                .groupby("as_of_date", as_index=False, sort=True)
                .head(1)
                .rename(
                    columns={
                        "ATM IV": "Selected ATM IV",
                        "DTE": "Actual DTE",
                    }
                )
            )
            selected_dte_frame = selected_dte_frame.merge(
                frame[
                    [
                        "as_of_date",
                        "Mean ATM IV",
                        "Median ATM IV",
                        "Expirations Used",
                    ]
                ],
                on="as_of_date",
                how="inner",
            )
            selected_dte_frame["Selected - Mean"] = (
                selected_dte_frame["Selected ATM IV"]
                - selected_dte_frame["Mean ATM IV"]
            )
            selected_dte_frame["Selected - Median"] = (
                selected_dte_frame["Selected ATM IV"]
                - selected_dte_frame["Median ATM IV"]
            )
            selected_hover_data = np.column_stack(
                [
                    selected_dte_frame["Expiration Date"].dt.strftime("%Y-%m-%d"),
                    selected_dte_frame["Actual DTE"].round().astype(int),
                ]
            )
            visible = target_dte == default_target_dte
            target_trace_starts[target_dte] = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=selected_dte_frame["as_of_date"],
                    y=selected_dte_frame["Selected ATM IV"],
                    customdata=selected_hover_data,
                    mode="lines+markers",
                    name=f"{target_dte} DTE ATM IV",
                    visible=visible,
                    line=dict(color="#C084FC", width=2.5),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Target DTE: "
                        f"{target_dte}<br>Actual DTE: %{{customdata[1]}}<br>"
                        "Expiration: %{customdata[0]}<br>ATM IV: %{y:.2%}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_dte_frame["as_of_date"],
                    y=selected_dte_frame["Selected - Mean"],
                    customdata=selected_hover_data,
                    mode="lines+markers",
                    name=f"{target_dte} DTE - Tenor Mean",
                    visible=visible,
                    line=dict(color="#34D399", width=2),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Actual DTE: "
                        "%{customdata[1]}<br>Selected - Mean: %{y:.2%}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_dte_frame["as_of_date"],
                    y=selected_dte_frame["Selected - Median"],
                    customdata=selected_hover_data,
                    mode="lines+markers",
                    name=f"{target_dte} DTE - Tenor Median",
                    visible=visible,
                    line=dict(color="#F87171", width=2, dash="dash"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Actual DTE: "
                        "%{customdata[1]}<br>Selected - Median: %{y:.2%}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    dropdown_buttons = []
    for target_dte, trace_start in target_trace_starts.items():
        visibility = [True, True] + [False] * (len(fig.data) - 2)
        visibility[trace_start : trace_start + 3] = [True, True, True]
        dropdown_buttons.append(
            dict(
                label=f"{target_dte} DTE",
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{ticker_label + ' ' if ticker_label else ''}"
                            f"Historical ATM IV — {target_dte} DTE vs Fixed Tenors"
                        )
                    },
                ],
            )
        )

    title_prefix = f"{ticker_label} " if ticker_label else ""
    fig.update_layout(
        title=(
            f"{title_prefix}Historical ATM IV — {default_target_dte} DTE vs Fixed Tenors"
            if default_target_dte is not None
            else f"{title_prefix}Historical ATM IV Across Fixed Tenors"
        ),
        template=template,
        height=760,
        hovermode="x unified",
        updatemenus=(
            [
                dict(
                    active=normalized_target_dtes.index(default_target_dte),
                    buttons=dropdown_buttons,
                    direction="down",
                    x=0,
                    y=1.16,
                    xanchor="left",
                    yanchor="top",
                )
            ]
            if dropdown_buttons
            else []
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(t=130),
    )
    fig.add_hline(y=0, line_color="#9CA3AF", line_dash="dot", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(
        title_text="ATM Implied Volatility", tickformat=".1%", row=1, col=1
    )
    fig.update_yaxes(
        title_text="IV Difference", tickformat="+.1%", row=2, col=1
    )
    return fig

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

_DEFAULT_TARGET_DTES = (7, 14, 30, 60, 90, 120)


def _format_target_dte(target_dte: float) -> int | float:
    return int(target_dte) if float(target_dte).is_integer() else float(target_dte)


def _fixed_tenor_iv_column(target_dte: float) -> str:
    return f"{_format_target_dte(target_dte)} DTE ATM IV"


def _normalize_target_dtes(target_dtes: Sequence[int | float] | None) -> list[float]:
    values = _DEFAULT_TARGET_DTES if target_dtes is None else target_dtes
    normalized = sorted({float(target_dte) for target_dte in values if float(target_dte) > 0})
    return normalized


def _linear_interpolate_no_extrapolate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_x: float,
) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[valid].astype(float)
    y = y_values[valid].astype(float)
    if x.size == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    target = float(target_x)
    if target < float(x.min()) or target > float(x.max()):
        return np.nan
    if x.size == 1:
        return float(y[0]) if np.isclose(target, x[0]) else np.nan

    return float(np.interp(target, x, y))


def _interpolate_atm_iv_by_total_variance(
    expiration_frame: pd.DataFrame,
    *,
    target_dte: float,
    annualization_days: float,
) -> float:
    variance_frame = expiration_frame[["DTE", "ATM IV"]].copy()
    variance_frame["DTE"] = pd.to_numeric(variance_frame["DTE"], errors="coerce")
    variance_frame["ATM IV"] = pd.to_numeric(variance_frame["ATM IV"], errors="coerce")
    variance_frame = variance_frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["DTE", "ATM IV"]
    )
    variance_frame = variance_frame[
        variance_frame["DTE"].gt(0) & variance_frame["ATM IV"].ge(0)
    ].copy()
    if variance_frame.empty:
        return np.nan

    variance_frame["total_variance"] = (
        variance_frame["ATM IV"].pow(2) * (variance_frame["DTE"] / annualization_days)
    )
    grouped = variance_frame.groupby("DTE", as_index=False, sort=True)[
        "total_variance"
    ].mean()
    interpolated_variance = _linear_interpolate_no_extrapolate(
        grouped["DTE"].to_numpy(dtype=float),
        grouped["total_variance"].to_numpy(dtype=float),
        target_dte,
    )
    if not np.isfinite(interpolated_variance):
        return np.nan

    return float(np.sqrt(max(interpolated_variance, 0.0) / (target_dte / annualization_days)))


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
    target_dtes: Sequence[int | float] | None = None,
    annualization_days: float = 365.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ATM IV per expiration and fixed-tenor daily summaries."""
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
            time_years=max(float(row["Days Till Expiration"]), 1.0)
            / annualization_days,
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

    daily_counts = (
        expiration_history.groupby("as_of_date", as_index=False, sort=True)
        .agg(
            **{
                "Expirations Used": ("Expiration Date", "nunique"),
            }
        )
        .sort_values("as_of_date")
        .reset_index(drop=True)
    )

    target_values = _normalize_target_dtes(target_dtes)
    fixed_tenor_records: list[dict[str, object]] = []
    for as_of_date, date_frame in expiration_history.groupby("as_of_date", sort=True):
        record: dict[str, object] = {"as_of_date": as_of_date}
        fixed_tenor_values = []
        fixed_tenors_used = 0
        for target_dte in target_values:
            column = _fixed_tenor_iv_column(target_dte)
            interpolated_iv = _interpolate_atm_iv_by_total_variance(
                date_frame,
                target_dte=float(target_dte),
                annualization_days=float(annualization_days),
            )
            record[column] = interpolated_iv
            if np.isfinite(interpolated_iv):
                fixed_tenor_values.append(interpolated_iv)
                fixed_tenors_used += 1

        record["Mean ATM IV"] = (
            float(np.mean(fixed_tenor_values)) if fixed_tenor_values else np.nan
        )
        record["Fixed Tenors Used"] = fixed_tenors_used
        fixed_tenor_records.append(record)

    daily_summary = pd.DataFrame.from_records(fixed_tenor_records)
    if daily_summary.empty:
        daily_summary = daily_counts.copy()
        daily_summary["Mean ATM IV"] = np.nan
        daily_summary["Fixed Tenors Used"] = 0
    else:
        daily_summary = daily_summary.merge(daily_counts, on="as_of_date", how="left")
        ordered_columns = [
            "as_of_date",
            "Mean ATM IV",
            "Fixed Tenors Used",
            "Expirations Used",
            *[_fixed_tenor_iv_column(target_dte) for target_dte in target_values],
        ]
        daily_summary = daily_summary[
            [column for column in ordered_columns if column in daily_summary.columns]
        ]

    daily_summary = daily_summary.sort_values("as_of_date").reset_index(drop=True)
    return expiration_history, daily_summary


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column

    lower_lookup = {str(column).lower(): column for column in frame.columns}
    for column in candidates:
        match = lower_lookup.get(str(column).lower())
        if match is not None:
            return str(match)

    return None


def build_current_atm_iv_by_expiration(
    option_chain: pd.DataFrame,
    *,
    spot_price: float,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a current ATM-IV snapshot by expiration/DTE from a current chain."""
    if option_chain is None or option_chain.empty:
        return pd.DataFrame()

    frame = option_chain.copy()
    strike_column = _first_existing_column(frame, ("strike", "Strike"))
    iv_column = _first_existing_column(
        frame,
        (
            "impliedVolatility",
            "implied_volatility",
            "implied_vol",
            "Implied Volatility",
            "IV",
            "iv",
        ),
    )
    type_column = _first_existing_column(frame, ("Type", "type", "option_type"))
    dte_column = _first_existing_column(
        frame,
        ("Days Till Expiration", "Date Till Expiration", "DTE", "dte"),
    )
    expiration_column = _first_existing_column(
        frame,
        ("Expiration Date", "expiration_date", "expiration", "Expiration"),
    )
    required = {
        "strike": strike_column,
        "implied volatility": iv_column,
        "expiration": expiration_column,
    }
    missing = [label for label, column in required.items() if column is None]
    if missing:
        raise ValueError(
            "Current option chain is missing required columns for ATM IV snapshot: "
            + ", ".join(missing)
        )

    snapshot = pd.DataFrame(
        {
            "Expiration Date": pd.to_datetime(frame[expiration_column], errors="coerce"),
            "Strike": pd.to_numeric(frame[strike_column], errors="coerce"),
            "ATM IV": pd.to_numeric(frame[iv_column], errors="coerce"),
        }
    )
    if type_column is not None:
        type_values = frame[type_column].astype(str).str.strip().str.lower()
        snapshot["Type"] = type_values.str[0].map({"c": "Call", "p": "Put"})
        snapshot["Type"] = snapshot["Type"].fillna(frame[type_column].astype(str))
    else:
        snapshot["Type"] = "Option"

    if dte_column is not None:
        snapshot["DTE"] = pd.to_numeric(frame[dte_column], errors="coerce")
    else:
        anchor_date = (
            pd.Timestamp.today().normalize()
            if today is None
            else pd.Timestamp(today).normalize()
        )
        snapshot["DTE"] = (snapshot["Expiration Date"].dt.normalize() - anchor_date).dt.days

    valid_spot = float(spot_price)
    if not np.isfinite(valid_spot) or valid_spot <= 0:
        return pd.DataFrame()

    snapshot = snapshot.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Expiration Date", "DTE", "Strike", "ATM IV"]
    )
    snapshot = snapshot[
        snapshot["DTE"].gt(0)
        & snapshot["Strike"].gt(0)
        & snapshot["ATM IV"].gt(0)
    ].copy()
    if snapshot.empty:
        return pd.DataFrame()

    snapshot.loc[snapshot["ATM IV"].gt(5), "ATM IV"] = (
        snapshot.loc[snapshot["ATM IV"].gt(5), "ATM IV"] / 100.0
    )
    snapshot["Strike Distance"] = (snapshot["Strike"] - valid_spot).abs()
    nearest_by_type = (
        snapshot.sort_values(["Expiration Date", "Type", "Strike Distance", "Strike"])
        .groupby(["Expiration Date", "DTE", "Type"], as_index=False)
        .first()
    )
    atm_by_expiration = (
        nearest_by_type.groupby(["Expiration Date", "DTE"], as_index=False)
        .agg(
            **{
                "ATM IV": ("ATM IV", "mean"),
                "ATM Strike": ("Strike", "mean"),
                "Contracts Used": ("ATM IV", "size"),
            }
        )
        .sort_values(["DTE", "Expiration Date"])
        .reset_index(drop=True)
    )
    if atm_by_expiration.empty:
        return atm_by_expiration

    call_iv = nearest_by_type[nearest_by_type["Type"].eq("Call")][
        ["Expiration Date", "DTE", "ATM IV"]
    ].rename(columns={"ATM IV": "ATM Call IV"})
    put_iv = nearest_by_type[nearest_by_type["Type"].eq("Put")][
        ["Expiration Date", "DTE", "ATM IV"]
    ].rename(columns={"ATM IV": "ATM Put IV"})
    atm_by_expiration = atm_by_expiration.merge(
        call_iv,
        on=["Expiration Date", "DTE"],
        how="left",
    ).merge(
        put_iv,
        on=["Expiration Date", "DTE"],
        how="left",
    )
    current_mean_atm_iv = float(atm_by_expiration["ATM IV"].mean())
    atm_by_expiration["Current Mean ATM IV"] = current_mean_atm_iv
    atm_by_expiration["ATM IV - Current Mean ATM IV"] = (
        atm_by_expiration["ATM IV"] - current_mean_atm_iv
    )
    return atm_by_expiration


def plot_current_atm_iv_spread_view(
    current_atm_iv: pd.DataFrame,
    *,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot current ATM IV by DTE and the spread to the current term-structure mean."""
    required_columns = {
        "Expiration Date",
        "DTE",
        "ATM IV",
        "Current Mean ATM IV",
        "ATM IV - Current Mean ATM IV",
    }
    if current_atm_iv is None or current_atm_iv.empty:
        raise ValueError("No current ATM IV observations are available.")
    missing_columns = sorted(required_columns - set(current_atm_iv.columns))
    if missing_columns:
        raise ValueError(
            "Current ATM IV snapshot is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = current_atm_iv.copy()
    frame["Expiration Date"] = pd.to_datetime(frame["Expiration Date"], errors="coerce")
    frame["DTE"] = pd.to_numeric(frame["DTE"], errors="coerce")
    frame["ATM IV"] = pd.to_numeric(frame["ATM IV"], errors="coerce")
    frame["Current Mean ATM IV"] = pd.to_numeric(
        frame["Current Mean ATM IV"], errors="coerce"
    )
    frame["ATM IV - Current Mean ATM IV"] = pd.to_numeric(
        frame["ATM IV - Current Mean ATM IV"], errors="coerce"
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "Expiration Date",
            "DTE",
            "ATM IV",
            "Current Mean ATM IV",
            "ATM IV - Current Mean ATM IV",
        ]
    )
    frame = frame.sort_values(["DTE", "Expiration Date"])
    if frame.empty:
        raise ValueError("No valid current ATM IV observations are available.")

    mean_atm_iv = float(frame["Current Mean ATM IV"].iloc[0])
    title_prefix = f"{ticker_label} " if ticker_label else ""
    hover_data = np.column_stack(
        [
            frame["Expiration Date"].dt.strftime("%Y-%m-%d"),
            frame.get("ATM Strike", pd.Series(np.nan, index=frame.index)),
            frame.get("ATM Call IV", pd.Series(np.nan, index=frame.index)),
            frame.get("ATM Put IV", pd.Series(np.nan, index=frame.index)),
            frame.get("Contracts Used", pd.Series(np.nan, index=frame.index)),
        ]
    )
    spread_values = frame["ATM IV - Current Mean ATM IV"]
    spread_colors = np.where(spread_values.ge(0), "#DC2626", "#16A34A")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            "Current ATM IV by DTE",
            "Current ATM IV Minus Current Mean ATM IV",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["DTE"],
            y=frame["ATM IV"],
            customdata=hover_data,
            mode="lines+markers",
            name="Current ATM IV",
            line=dict(color="#2563EB", width=3),
            marker=dict(color="#2563EB", size=7),
            hovertemplate=(
                "DTE: %{x:.0f}<br>"
                "Expiration: %{customdata[0]}<br>"
                "ATM IV: %{y:.2%}<br>"
                "ATM Strike: %{customdata[1]:,.2f}<br>"
                "Call IV: %{customdata[2]:.2%}<br>"
                "Put IV: %{customdata[3]:.2%}<br>"
                "Contracts Used: %{customdata[4]:.0f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["DTE"],
            y=np.full(len(frame), mean_atm_iv),
            mode="lines",
            name="Current Mean ATM IV",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
            hovertemplate="Current Mean ATM IV: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame["DTE"],
            y=spread_values,
            customdata=hover_data,
            name="ATM IV - Current Mean",
            marker=dict(color=spread_colors),
            hovertemplate=(
                "DTE: %{x:.0f}<br>"
                "Expiration: %{customdata[0]}<br>"
                "Spread: %{y:+.2%}<br>"
                "ATM IV: %{customdata[2]:.2%} call / %{customdata[3]:.2%} put"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color="#6B7280", row=2, col=1)
    fig.update_yaxes(title_text="ATM IV", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Spread", tickformat="+.0%", row=2, col=1)
    fig.update_xaxes(title_text="Days Till Expiration", row=2, col=1)
    fig.update_layout(
        title=f"{title_prefix}Current ATM IV Term Structure vs Mean",
        template=template,
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(l=70, r=35, t=105, b=65),
    )
    return fig


def plot_historical_atm_iv_summary_view(
    daily_summary: pd.DataFrame,
    *,
    expiration_history: pd.DataFrame | None = None,
    target_dtes: Sequence[int] | None = None,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot mean ATM IV against a dropdown-selected target DTE."""
    required_columns = {
        "as_of_date",
        "Mean ATM IV",
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
    frame["Mean ATM IV"] = pd.to_numeric(frame["Mean ATM IV"], errors="coerce")
    frame = frame[
        pd.to_numeric(frame["Expirations Used"], errors="coerce").ge(3)
    ].copy()
    frame = frame.dropna(subset=["Mean ATM IV"])
    if frame.empty:
        raise ValueError(
            "At least three valid expirations per date are required to compare "
            "selected-DTE ATM IV with the fixed-tenor mean benchmark."
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
            "Selected DTE ATM IV Minus Fixed-Tenor Mean",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["as_of_date"],
            y=frame["Mean ATM IV"],
            customdata=summary_hover_data,
            mode="lines+markers",
            name="Fixed-Tenor Mean ATM IV",
            line=dict(color="#F59E0B", width=3, dash="dash"),
            marker=dict(color="#F59E0B", size=6, symbol="diamond"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>Fixed-Tenor Mean ATM IV: %{y:.2%}<br>"
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
            fixed_tenor_column = _fixed_tenor_iv_column(target_dte)
            if fixed_tenor_column in frame.columns:
                selected_dte_frame = frame[
                    [
                        "as_of_date",
                        fixed_tenor_column,
                        "Mean ATM IV",
                        "Expirations Used",
                    ]
                ].rename(columns={fixed_tenor_column: "Selected ATM IV"})
                selected_dte_frame["Selected ATM IV"] = pd.to_numeric(
                    selected_dte_frame["Selected ATM IV"], errors="coerce"
                )
                selected_dte_frame = selected_dte_frame.dropna(
                    subset=["Selected ATM IV"]
                )
                selected_dte_frame["Expiration Date"] = "Interpolated"
                selected_dte_frame["Actual DTE"] = target_dte
            else:
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
                            "Expirations Used",
                        ]
                    ],
                    on="as_of_date",
                    how="inner",
                )

            if selected_dte_frame.empty:
                continue

            selected_dte_frame["Selected - Mean"] = (
                selected_dte_frame["Selected ATM IV"]
                - selected_dte_frame["Mean ATM IV"]
            )
            selected_hover_data = np.column_stack(
                [
                    selected_dte_frame["Expiration Date"].apply(
                        lambda value: (
                            value.strftime("%Y-%m-%d")
                            if hasattr(value, "strftime")
                            else str(value)
                        )
                    ),
                    selected_dte_frame["Actual DTE"].round().astype(int),
                ]
            )
            target_trace_starts[target_dte] = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=selected_dte_frame["as_of_date"],
                    y=selected_dte_frame["Selected ATM IV"],
                    customdata=selected_hover_data,
                    mode="lines+markers",
                    name=f"{target_dte} DTE ATM IV",
                    visible=False,
                    line=dict(color="#2563EB", width=3),
                    marker=dict(color="#2563EB", size=6, symbol="circle"),
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
                    visible=False,
                    line=dict(color="#E11D48", width=2.4),
                    marker=dict(color="#E11D48", size=6, symbol="triangle-up"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Actual DTE: "
                        "%{customdata[1]}<br>Selected - Mean: %{y:.2%}"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    if not target_trace_starts:
        raise ValueError("No fixed-tenor ATM IV observations are available to plot.")

    default_target_dte = 30 if 30 in target_trace_starts else next(iter(target_trace_starts))
    trace_start = target_trace_starts[default_target_dte]
    for trace_index in range(trace_start, trace_start + 2):
        fig.data[trace_index].visible = True

    dropdown_buttons = []
    for target_dte, trace_start in target_trace_starts.items():
        visibility = [True] + [False] * (len(fig.data) - 1)
        visibility[trace_start : trace_start + 2] = [True, True]
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
        dropdown_buttons[-1]["args"][1]["title"] = (
            f"{ticker_label + ' ' if ticker_label else ''}"
            f"Historical ATM IV - {target_dte} DTE vs Mean"
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
        updatemenus=[
            dict(
                active=list(target_trace_starts).index(default_target_dte),
                buttons=dropdown_buttons,
                direction="down",
                x=0,
                y=1.16,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(t=130),
    )
    if default_target_dte is not None:
        fig.update_layout(
            title=f"{title_prefix}Historical ATM IV - {default_target_dte} DTE vs Mean"
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


def _prepare_close_price_series(
    price_history: pd.DataFrame | pd.Series,
    *,
    price_column: str = "Close",
) -> pd.Series:
    """Return a clean daily close series indexed by normalized dates."""
    if isinstance(price_history, pd.Series):
        series = price_history.copy()
    else:
        if price_column not in price_history.columns:
            raise ValueError(f"price_history is missing required column: {price_column}")
        series = price_history[price_column].copy()

    series = pd.to_numeric(series, errors="coerce")
    index = pd.to_datetime(series.index, errors="coerce", utc=True)
    series.index = index.tz_convert(None).normalize()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series.gt(0)].sort_index()
    return series.groupby(level=0).last()


def _prepare_ohlc_price_frame(
    price_history: pd.DataFrame,
    *,
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
) -> pd.DataFrame:
    """Return a clean daily OHLC frame indexed by normalized dates."""
    if price_history is None or not isinstance(price_history, pd.DataFrame):
        raise ValueError("price_history must be a DataFrame with OHLC price columns.")

    column_map = {
        open_column: "Open",
        high_column: "High",
        low_column: "Low",
        close_column: "Close",
    }
    missing_columns = sorted(set(column_map) - set(price_history.columns))
    if missing_columns:
        raise ValueError(
            "price_history is missing required OHLC columns: "
            + ", ".join(missing_columns)
        )

    frame = price_history[list(column_map)].rename(columns=column_map).copy()
    for column in ["Open", "High", "Low", "Close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame.index = index.tz_convert(None).normalize()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Open", "High", "Low", "Close"]
    )
    frame = frame[(frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return frame.sort_index().groupby(level=0).last()


def _realized_vol_for_window(
    log_returns: pd.Series,
    as_of_date: pd.Timestamp,
    trading_days: int,
    *,
    direction: str,
    trading_days_per_year: float,
) -> float:
    if trading_days <= 1:
        return np.nan

    if direction == "trailing":
        window = log_returns.loc[:as_of_date].tail(trading_days)
    elif direction == "forward":
        window = log_returns.loc[log_returns.index > as_of_date].head(trading_days)
    else:
        raise ValueError("direction must be 'trailing' or 'forward'.")

    if len(window) < trading_days:
        return np.nan
    return float(window.std() * np.sqrt(trading_days_per_year))


def _rolling_iv_rank(
    series: pd.Series,
    *,
    lookback: int | None,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if lookback is None:
        rolling_min = values.expanding(min_periods=min_periods).min()
        rolling_max = values.expanding(min_periods=min_periods).max()
    else:
        rolling_min = values.rolling(window=lookback, min_periods=min_periods).min()
        rolling_max = values.rolling(window=lookback, min_periods=min_periods).max()

    value_range = rolling_max - rolling_min
    rank = (values - rolling_min) / value_range
    return rank.where(value_range.gt(0))


def _rolling_iv_percentile(
    series: pd.Series,
    *,
    lookback: int | None,
    min_periods: int,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def _percentile_rank(window_values: np.ndarray) -> float:
        finite_values = window_values[np.isfinite(window_values)]
        if finite_values.size < min_periods:
            return np.nan
        current_value = finite_values[-1]
        return float(np.mean(finite_values <= current_value))

    if lookback is None:
        return values.expanding(min_periods=min_periods).apply(
            _percentile_rank,
            raw=True,
        )
    return values.rolling(window=lookback, min_periods=min_periods).apply(
        _percentile_rank,
        raw=True,
    )


def _axis_reference(axis_prefix: str, row: int) -> str:
    return axis_prefix if row == 1 else f"{axis_prefix}{row}"


def _high_percentile_region_shapes(
    dates: pd.Series,
    percentiles: pd.Series,
    *,
    rows: Sequence[int],
    threshold: float = 0.5,
    fillcolor: str = "rgba(245, 158, 11, 0.12)",
) -> list[dict[str, object]]:
    """Build vertical background regions where IV Percentile is above a threshold."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "percentile": pd.to_numeric(percentiles, errors="coerce"),
        }
    ).dropna(subset=["date", "percentile"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        return []

    frame["is_high"] = frame["percentile"].gt(threshold)
    if not frame["is_high"].any():
        return []

    date_diffs = frame["date"].diff().dropna()
    fallback_width = (
        date_diffs.median()
        if not date_diffs.empty and pd.notna(date_diffs.median())
        else pd.Timedelta(days=1)
    )
    if fallback_width <= pd.Timedelta(0):
        fallback_width = pd.Timedelta(days=1)

    shapes: list[dict[str, object]] = []
    high_values = frame["is_high"].to_numpy(dtype=bool)
    date_values = frame["date"].tolist()
    run_start: int | None = None
    for index, is_high in enumerate(high_values):
        if is_high and run_start is None:
            run_start = index
        is_last = index == len(high_values) - 1
        if run_start is not None and ((not is_high) or is_last):
            run_end = index if is_high and is_last else index - 1
            x0 = date_values[run_start]
            x1 = (
                date_values[run_end + 1]
                if run_end + 1 < len(date_values)
                else date_values[run_end] + fallback_width
            )
            for row in rows:
                shapes.append(
                    {
                        "type": "rect",
                        "xref": _axis_reference("x", row),
                        "yref": f"{_axis_reference('y', row)} domain",
                        "x0": x0,
                        "x1": x1,
                        "y0": 0,
                        "y1": 1,
                        "fillcolor": fillcolor,
                        "line": {"width": 0},
                        "layer": "below",
                    }
                )
            run_start = None
    return shapes


def _iv_premium_reference_shapes(*, has_rank_panels: bool) -> list[dict[str, object]]:
    shapes: list[dict[str, object]] = [
        {
            "type": "line",
            "xref": "paper",
            "yref": "y2",
            "x0": 0,
            "x1": 1,
            "y0": 0,
            "y1": 0,
            "line": {"color": "#9CA3AF", "dash": "dot", "width": 1},
            "layer": "above",
        }
    ]
    if has_rank_panels:
        for row in (3, 4):
            shapes.append(
                {
                    "type": "line",
                    "xref": "paper",
                    "yref": _axis_reference("y", row),
                    "x0": 0,
                    "x1": 1,
                    "y0": 0.5,
                    "y1": 0.5,
                    "line": {"color": "#F59E0B", "dash": "dot", "width": 1},
                    "layer": "above",
                }
            )
    return shapes


def build_historical_iv_premium_history(
    iv_history: pd.DataFrame,
    price_history: pd.DataFrame | pd.Series,
    *,
    target_dtes: Sequence[int | float] | None = None,
    price_column: str = "Close",
    trading_days_per_year: float = 252.0,
    calendar_days_per_year: float = 365.25,
    iv_rank_lookback: int | None = 252,
    iv_rank_min_periods: int = 20,
) -> pd.DataFrame:
    """
    Compare fixed-tenor ATM IV with trailing and forward realized volatility.

    Ex-ante premium uses trailing realized volatility available as of each date.
    Ex-post premium uses forward realized volatility after each date.
    """
    if iv_history is None or iv_history.empty:
        return pd.DataFrame()
    if "as_of_date" not in iv_history.columns:
        raise ValueError("iv_history is missing required column: as_of_date")

    frame = iv_history.copy()
    frame["as_of_date"] = (
        pd.to_datetime(frame["as_of_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    frame = frame.dropna(subset=["as_of_date"]).sort_values("as_of_date")
    if "Mean ATM IV" in frame.columns:
        frame["Mean ATM IV"] = pd.to_numeric(frame["Mean ATM IV"], errors="coerce")

    close_prices = _prepare_close_price_series(
        price_history,
        price_column=price_column,
    )
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_returns.empty:
        return pd.DataFrame()

    if "Underlying Close" in frame.columns:
        frame = frame.drop(columns=["Underlying Close"])
    close_frame = (
        close_prices.rename("Underlying Close")
        .rename_axis("as_of_date")
        .reset_index()
        .sort_values("as_of_date")
    )
    frame = pd.merge_asof(
        frame.sort_values("as_of_date"),
        close_frame,
        on="as_of_date",
        direction="backward",
    )

    target_values = _normalize_target_dtes(target_dtes)
    trailing_rv_columns: list[str] = []
    forward_rv_columns: list[str] = []
    ex_ante_columns: list[str] = []
    ex_post_columns: list[str] = []
    implied_move_pct_columns: list[str] = []
    implied_move_columns: list[str] = []
    implied_lower_columns: list[str] = []
    implied_upper_columns: list[str] = []
    realized_move_pct_columns: list[str] = []
    realized_move_columns: list[str] = []
    realized_lower_columns: list[str] = []
    realized_upper_columns: list[str] = []
    forward_realized_move_pct_columns: list[str] = []
    forward_realized_move_columns: list[str] = []
    forward_realized_lower_columns: list[str] = []
    forward_realized_upper_columns: list[str] = []

    for target_dte in target_values:
        target_label = _format_target_dte(target_dte)
        iv_column = _fixed_tenor_iv_column(target_dte)
        if iv_column not in frame.columns:
            continue

        trading_days = max(
            2,
            int(round(float(target_dte) * trading_days_per_year / calendar_days_per_year)),
        )
        trailing_rv_column = f"{target_label} DTE Trailing RV"
        forward_rv_column = f"{target_label} DTE Forward RV"
        ex_ante_column = f"{target_label} DTE Ex-Ante Premium"
        ex_post_column = f"{target_label} DTE Ex-Post Premium"
        variance_ex_ante_column = f"{target_label} DTE Ex-Ante Variance Premium"
        variance_ex_post_column = f"{target_label} DTE Ex-Post Variance Premium"
        iv_rank_column = f"{target_label} DTE IV Rank"
        iv_percentile_column = f"{target_label} DTE IV Percentile"
        implied_move_pct_column = f"{target_label} DTE Implied Move %"
        implied_move_column = f"{target_label} DTE Implied Move"
        implied_lower_column = f"{target_label} DTE Implied Lower Price"
        implied_upper_column = f"{target_label} DTE Implied Upper Price"
        realized_move_pct_column = f"{target_label} DTE Realized Move %"
        realized_move_column = f"{target_label} DTE Realized Move"
        realized_lower_column = f"{target_label} DTE Realized Lower Price"
        realized_upper_column = f"{target_label} DTE Realized Upper Price"
        forward_realized_move_pct_column = (
            f"{target_label} DTE Forward Realized Move %"
        )
        forward_realized_move_column = f"{target_label} DTE Forward Realized Move"
        forward_realized_lower_column = (
            f"{target_label} DTE Forward Realized Lower Price"
        )
        forward_realized_upper_column = (
            f"{target_label} DTE Forward Realized Upper Price"
        )

        frame[iv_column] = pd.to_numeric(frame[iv_column], errors="coerce")
        frame[implied_move_pct_column] = frame[iv_column] * np.sqrt(
            float(target_dte) / calendar_days_per_year
        )
        frame[implied_move_column] = (
            frame["Underlying Close"] * frame[implied_move_pct_column]
        )
        frame[implied_lower_column] = frame["Underlying Close"] - frame[
            implied_move_column
        ]
        frame[implied_upper_column] = frame["Underlying Close"] + frame[
            implied_move_column
        ]
        frame[iv_rank_column] = _rolling_iv_rank(
            frame[iv_column],
            lookback=iv_rank_lookback,
            min_periods=iv_rank_min_periods,
        )
        frame[iv_percentile_column] = _rolling_iv_percentile(
            frame[iv_column],
            lookback=iv_rank_lookback,
            min_periods=iv_rank_min_periods,
        )
        frame[trailing_rv_column] = frame["as_of_date"].apply(
            lambda date: _realized_vol_for_window(
                log_returns,
                date,
                trading_days,
                direction="trailing",
                trading_days_per_year=trading_days_per_year,
            )
        )
        frame[forward_rv_column] = frame["as_of_date"].apply(
            lambda date: _realized_vol_for_window(
                log_returns,
                date,
                trading_days,
                direction="forward",
                trading_days_per_year=trading_days_per_year,
            )
        )
        frame[realized_move_pct_column] = frame[trailing_rv_column] * np.sqrt(
            float(trading_days) / float(trading_days_per_year)
        )
        frame[realized_move_column] = (
            frame["Underlying Close"] * frame[realized_move_pct_column]
        )
        frame[realized_lower_column] = frame["Underlying Close"] - frame[
            realized_move_column
        ]
        frame[realized_upper_column] = frame["Underlying Close"] + frame[
            realized_move_column
        ]
        frame[forward_realized_move_pct_column] = frame[forward_rv_column] * np.sqrt(
            float(trading_days) / float(trading_days_per_year)
        )
        frame[forward_realized_move_column] = (
            frame["Underlying Close"] * frame[forward_realized_move_pct_column]
        )
        frame[forward_realized_lower_column] = frame["Underlying Close"] - frame[
            forward_realized_move_column
        ]
        frame[forward_realized_upper_column] = frame["Underlying Close"] + frame[
            forward_realized_move_column
        ]
        frame[ex_ante_column] = frame[iv_column] - frame[trailing_rv_column]
        frame[ex_post_column] = frame[iv_column] - frame[forward_rv_column]
        frame[variance_ex_ante_column] = frame[iv_column].pow(2) - frame[
            trailing_rv_column
        ].pow(2)
        frame[variance_ex_post_column] = frame[iv_column].pow(2) - frame[
            forward_rv_column
        ].pow(2)

        trailing_rv_columns.append(trailing_rv_column)
        forward_rv_columns.append(forward_rv_column)
        ex_ante_columns.append(ex_ante_column)
        ex_post_columns.append(ex_post_column)
        implied_move_pct_columns.append(implied_move_pct_column)
        implied_move_columns.append(implied_move_column)
        implied_lower_columns.append(implied_lower_column)
        implied_upper_columns.append(implied_upper_column)
        realized_move_pct_columns.append(realized_move_pct_column)
        realized_move_columns.append(realized_move_column)
        realized_lower_columns.append(realized_lower_column)
        realized_upper_columns.append(realized_upper_column)
        forward_realized_move_pct_columns.append(forward_realized_move_pct_column)
        forward_realized_move_columns.append(forward_realized_move_column)
        forward_realized_lower_columns.append(forward_realized_lower_column)
        forward_realized_upper_columns.append(forward_realized_upper_column)

    if trailing_rv_columns:
        frame["Mean Trailing RV"] = frame[trailing_rv_columns].mean(axis=1)
        frame["Mean Forward RV"] = frame[forward_rv_columns].mean(axis=1)
        frame["Mean Ex-Ante Premium"] = frame[ex_ante_columns].mean(axis=1)
        frame["Mean Ex-Post Premium"] = frame[ex_post_columns].mean(axis=1)
        frame["Mean Implied Move %"] = frame[implied_move_pct_columns].mean(axis=1)
        frame["Mean Implied Move"] = frame[implied_move_columns].mean(axis=1)
        frame["Mean Implied Lower Price"] = frame[implied_lower_columns].mean(axis=1)
        frame["Mean Implied Upper Price"] = frame[implied_upper_columns].mean(axis=1)
        frame["Mean Realized Move %"] = frame[realized_move_pct_columns].mean(axis=1)
        frame["Mean Realized Move"] = frame[realized_move_columns].mean(axis=1)
        frame["Mean Realized Lower Price"] = frame[realized_lower_columns].mean(axis=1)
        frame["Mean Realized Upper Price"] = frame[realized_upper_columns].mean(axis=1)
        frame["Mean Forward Realized Move %"] = frame[
            forward_realized_move_pct_columns
        ].mean(axis=1)
        frame["Mean Forward Realized Move"] = frame[
            forward_realized_move_columns
        ].mean(axis=1)
        frame["Mean Forward Realized Lower Price"] = frame[
            forward_realized_lower_columns
        ].mean(axis=1)
        frame["Mean Forward Realized Upper Price"] = frame[
            forward_realized_upper_columns
        ].mean(axis=1)
        if "Mean ATM IV" in frame.columns:
            frame["Mean IV Rank"] = _rolling_iv_rank(
                frame["Mean ATM IV"],
                lookback=iv_rank_lookback,
                min_periods=iv_rank_min_periods,
            )
            frame["Mean IV Percentile"] = _rolling_iv_percentile(
                frame["Mean ATM IV"],
                lookback=iv_rank_lookback,
                min_periods=iv_rank_min_periods,
            )
            frame["Mean Ex-Ante Variance Premium"] = frame["Mean ATM IV"].pow(2) - frame[
                "Mean Trailing RV"
            ].pow(2)
            frame["Mean Ex-Post Variance Premium"] = frame["Mean ATM IV"].pow(2) - frame[
                "Mean Forward RV"
            ].pow(2)

    return frame.reset_index(drop=True)


def plot_historical_iv_premium_view(
    premium_history: pd.DataFrame,
    *,
    target_dtes: Sequence[int | float] | None = None,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot fixed-tenor IV, realized volatility, and IV premium history."""
    required_columns = {"as_of_date"}
    if premium_history is None or premium_history.empty:
        raise ValueError("No historical IV premium observations are available.")
    missing_columns = sorted(required_columns - set(premium_history.columns))
    if missing_columns:
        raise ValueError(
            "Historical IV premium data is missing columns: " + ", ".join(missing_columns)
        )

    frame = premium_history.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    target_values = _normalize_target_dtes(target_dtes)
    target_specs: list[dict[str, str | int | float]] = []
    if {
        "Mean ATM IV",
        "Mean Trailing RV",
        "Mean Forward RV",
        "Mean Ex-Ante Premium",
        "Mean Ex-Post Premium",
    }.issubset(frame.columns):
        target_specs.append(
            {
                "label": "Mean",
                "iv": "Mean ATM IV",
                "trailing_rv": "Mean Trailing RV",
                "forward_rv": "Mean Forward RV",
                "ex_ante": "Mean Ex-Ante Premium",
                "ex_post": "Mean Ex-Post Premium",
                "rank": "Mean IV Rank",
                "percentile": "Mean IV Percentile",
                "underlying": "Underlying Close",
                "implied_move_pct": "Mean Implied Move %",
                "implied_move": "Mean Implied Move",
                "implied_lower": "Mean Implied Lower Price",
                "implied_upper": "Mean Implied Upper Price",
                "realized_move_pct": "Mean Realized Move %",
                "realized_move": "Mean Realized Move",
                "realized_lower": "Mean Realized Lower Price",
                "realized_upper": "Mean Realized Upper Price",
            }
        )

    for target_dte in target_values:
        target_label = _format_target_dte(target_dte)
        spec = {
            "label": f"{target_label} DTE",
            "iv": _fixed_tenor_iv_column(target_dte),
            "trailing_rv": f"{target_label} DTE Trailing RV",
            "forward_rv": f"{target_label} DTE Forward RV",
            "ex_ante": f"{target_label} DTE Ex-Ante Premium",
            "ex_post": f"{target_label} DTE Ex-Post Premium",
            "rank": f"{target_label} DTE IV Rank",
            "percentile": f"{target_label} DTE IV Percentile",
            "underlying": "Underlying Close",
            "implied_move_pct": f"{target_label} DTE Implied Move %",
            "implied_move": f"{target_label} DTE Implied Move",
            "implied_lower": f"{target_label} DTE Implied Lower Price",
            "implied_upper": f"{target_label} DTE Implied Upper Price",
            "realized_move_pct": f"{target_label} DTE Realized Move %",
            "realized_move": f"{target_label} DTE Realized Move",
            "realized_lower": f"{target_label} DTE Realized Lower Price",
            "realized_upper": f"{target_label} DTE Realized Upper Price",
        }
        if {
            str(spec["iv"]),
            str(spec["trailing_rv"]),
            str(spec["forward_rv"]),
            str(spec["ex_ante"]),
            str(spec["ex_post"]),
        }.issubset(frame.columns):
            target_specs.append(spec)

    if not target_specs:
        raise ValueError("No fixed-DTE IV premium columns are available to plot.")

    has_rank_panels = any(
        {str(spec["rank"]), str(spec["percentile"])}.issubset(frame.columns)
        for spec in target_specs
    )
    has_implied_move_panel = any(
        {
            str(spec["underlying"]),
            str(spec["implied_move_pct"]),
            str(spec["implied_move"]),
            str(spec["implied_lower"]),
            str(spec["implied_upper"]),
        }.issubset(frame.columns)
        for spec in target_specs
    )
    has_realized_move_panel = any(
        {
            str(spec["underlying"]),
            str(spec["realized_move_pct"]),
            str(spec["realized_move"]),
            str(spec["realized_lower"]),
            str(spec["realized_upper"]),
        }.issubset(frame.columns)
        for spec in target_specs
    )
    subplot_titles = [
        "Implied vs Realized Volatility",
        "IV Premium (IV - Realized Volatility)",
    ]
    if has_rank_panels:
        subplot_titles.extend(["IV Rank", "IV Percentile"])
    implied_move_row = (
        len(subplot_titles) + 1
        if has_implied_move_panel or has_realized_move_panel
        else None
    )
    if implied_move_row is not None:
        subplot_titles.append("Implied and Realized Vol Price Range")

    row_count = len(subplot_titles)
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055 if row_count > 2 else 0.09,
        row_heights=[1.0 / row_count] * row_count,
        subplot_titles=tuple(subplot_titles),
    )

    reference_shapes = _iv_premium_reference_shapes(has_rank_panels=has_rank_panels)
    trace_groups: dict[str, tuple[int, int, list[bool | str]]] = {}
    shape_groups: dict[str, list[dict[str, object]]] = {}
    for spec in target_specs:
        label = str(spec["label"])
        clean_columns = [
            "as_of_date",
            str(spec["iv"]),
            str(spec["trailing_rv"]),
            str(spec["forward_rv"]),
            str(spec["ex_ante"]),
            str(spec["ex_post"]),
        ]
        show_rank_panels = has_rank_panels and {
            str(spec["rank"]),
            str(spec["percentile"]),
        }.issubset(frame.columns)
        if show_rank_panels:
            clean_columns.extend([str(spec["rank"]), str(spec["percentile"])])
        show_implied_move_panel = implied_move_row is not None and {
            str(spec["underlying"]),
            str(spec["implied_move_pct"]),
            str(spec["implied_move"]),
            str(spec["implied_lower"]),
            str(spec["implied_upper"]),
        }.issubset(frame.columns)
        show_realized_move_panel = implied_move_row is not None and {
            str(spec["underlying"]),
            str(spec["realized_move_pct"]),
            str(spec["realized_move"]),
            str(spec["realized_lower"]),
            str(spec["realized_upper"]),
        }.issubset(frame.columns)
        if show_implied_move_panel:
            clean_columns.extend(
                [
                    str(spec["underlying"]),
                    str(spec["implied_move_pct"]),
                    str(spec["implied_move"]),
                    str(spec["implied_lower"]),
                    str(spec["implied_upper"]),
                ]
            )
        if show_realized_move_panel:
            clean_columns.extend(
                [
                    str(spec["underlying"]),
                    str(spec["realized_move_pct"]),
                    str(spec["realized_move"]),
                    str(spec["realized_lower"]),
                    str(spec["realized_upper"]),
                ]
            )
        clean_columns = list(dict.fromkeys(clean_columns))
        clean = frame[clean_columns].copy()
        for column in clean.columns:
            if column != "as_of_date":
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan)

        start_index = len(fig.data)
        default_visibility: list[bool | str] = []
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[str(spec["iv"])],
                mode="lines",
                name=f"{label} ATM IV",
                visible=False,
                line=dict(color="#2563EB", width=3),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>ATM IV: %{y:.2%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        default_visibility.append(True)
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[str(spec["trailing_rv"])],
                mode="lines",
                name=f"{label} trailing RV",
                visible=False,
                line=dict(color="#F59E0B", width=2.4, dash="dash"),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Trailing RV: %{y:.2%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        default_visibility.append(True)
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[str(spec["forward_rv"])],
                mode="lines",
                name=f"{label} forward RV",
                visible=False,
                line=dict(color="#14B8A6", width=2.4, dash="dot"),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Forward RV: %{y:.2%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        default_visibility.append("legendonly")
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[str(spec["ex_ante"])],
                mode="lines",
                name=f"{label} ex-ante premium",
                visible=False,
                line=dict(color="#E11D48", width=2.4),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Ex-ante premium: %{y:.2%}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        default_visibility.append(True)
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[str(spec["ex_post"])],
                mode="lines",
                name=f"{label} ex-post premium",
                visible=False,
                line=dict(color="#A855F7", width=2.4, dash="dash"),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Ex-post premium: %{y:.2%}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        default_visibility.append("legendonly")
        if show_rank_panels:
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["rank"])],
                    mode="lines",
                    name=f"{label} IV Rank",
                    visible=False,
                    line=dict(color="#2563EB", width=2.4),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>IV Rank: %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
            default_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["percentile"])],
                    mode="lines",
                    name=f"{label} IV Percentile",
                    visible=False,
                    line=dict(color="#F59E0B", width=2.4),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>IV Percentile: %{y:.1%}<extra></extra>"
                    ),
                ),
                row=4,
                col=1,
            )
            default_visibility.append(True)
        if show_implied_move_panel and implied_move_row is not None:
            implied_move_hover_data = np.column_stack(
                [
                    clean[str(spec["implied_move"])],
                    clean[str(spec["implied_move_pct"])],
                    clean[str(spec["implied_lower"])],
                    clean[str(spec["implied_upper"])],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["implied_upper"])],
                    customdata=implied_move_hover_data,
                    mode="lines",
                    name=f"{label} implied upper price",
                    visible=False,
                    line=dict(color="#16A34A", width=2.2),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Upper price: %{y:,.2f}<br>"
                        "Implied move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                ),
                row=implied_move_row,
                col=1,
            )
            default_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["implied_lower"])],
                    customdata=implied_move_hover_data,
                    mode="lines",
                    name=f"{label} implied lower price",
                    visible=False,
                    fill="tonexty",
                    fillcolor="rgba(22, 163, 74, 0.13)",
                    line=dict(color="#DC2626", width=2.2),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Lower price: %{y:,.2f}<br>"
                        "Implied move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                ),
                row=implied_move_row,
                col=1,
            )
            default_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["underlying"])],
                    customdata=implied_move_hover_data,
                    mode="lines",
                    name=f"{label} underlying close",
                    visible=False,
                    line=dict(color="#111827", width=2),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Close: %{y:,.2f}<br>"
                        "Lower / Upper: %{customdata[2]:,.2f} / "
                        "%{customdata[3]:,.2f}<extra></extra>"
                    ),
                ),
                row=implied_move_row,
                col=1,
            )
            default_visibility.append(True)

        if show_realized_move_panel and implied_move_row is not None:
            realized_move_hover_data = np.column_stack(
                [
                    clean[str(spec["realized_move"])],
                    clean[str(spec["realized_move_pct"])],
                    clean[str(spec["realized_lower"])],
                    clean[str(spec["realized_upper"])],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["realized_upper"])],
                    customdata=realized_move_hover_data,
                    mode="lines",
                    name=f"{label} realized upper price",
                    visible=False,
                    line=dict(color="#22D3EE", width=2.1, dash="dot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Realized upper price: %{y:,.2f}<br>"
                        "Realized-vol move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                ),
                row=implied_move_row,
                col=1,
            )
            default_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=clean["as_of_date"],
                    y=clean[str(spec["realized_lower"])],
                    customdata=realized_move_hover_data,
                    mode="lines",
                    name=f"{label} realized lower price",
                    visible=False,
                    line=dict(color="#EC4899", width=2.1, dash="dot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Realized lower price: %{y:,.2f}<br>"
                        "Realized-vol move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                ),
                row=implied_move_row,
                col=1,
            )
            default_visibility.append(True)

        trace_groups[label] = (start_index, len(fig.data), default_visibility)
        percentile_shape_rows = (
            (1, 2, implied_move_row) if implied_move_row is not None else (1, 2)
        )
        percentile_region_shapes = (
            _high_percentile_region_shapes(
                clean["as_of_date"],
                clean[str(spec["percentile"])],
                rows=percentile_shape_rows,
            )
            if show_rank_panels
            else []
        )
        shape_groups[label] = [*percentile_region_shapes, *reference_shapes]

    default_label = (
        "30 DTE"
        if "30 DTE" in trace_groups
        else "Mean"
        if "Mean" in trace_groups
        else next(iter(trace_groups))
    )
    start, stop, default_visibility = trace_groups[default_label]
    for trace_index, trace_visibility in zip(
        range(start, stop),
        default_visibility,
        strict=False,
    ):
        fig.data[trace_index].visible = trace_visibility

    dropdown_buttons = []
    for label, (start, stop, group_visibility) in trace_groups.items():
        visibility = [False] * len(fig.data)
        visibility[start:stop] = group_visibility
        dropdown_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{ticker_label + ' ' if ticker_label else ''}"
                            f"Historical IV Premium - {label}"
                        ),
                        "shapes": shape_groups.get(label, reference_shapes),
                    },
                ],
            )
        )

    title_prefix = f"{ticker_label} " if ticker_label else ""
    fig.update_layout(
        title=f"{title_prefix}Historical IV Premium - {default_label}",
        template=template,
        height=max(780, 270 * row_count),
        hovermode="x unified",
        shapes=shape_groups.get(default_label, reference_shapes),
        updatemenus=[
            dict(
                active=list(trace_groups).index(default_label),
                buttons=dropdown_buttons,
                direction="down",
                x=0,
                y=1.16,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=130),
    )
    fig.update_xaxes(title_text="Date", row=row_count, col=1)
    fig.update_yaxes(
        title_text="Annualized Volatility",
        tickformat=".1%",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Premium (vol pts)",
        tickformat="+.1%",
        row=2,
        col=1,
    )
    if has_rank_panels:
        fig.update_yaxes(
            title_text="IV Rank",
            tickformat=".0%",
            range=[0, 1],
            row=3,
            col=1,
        )
        fig.update_yaxes(
            title_text="IV Percentile",
            tickformat=".0%",
            range=[0, 1],
            row=4,
            col=1,
        )
    if implied_move_row is not None:
        fig.update_yaxes(
            title_text="Price",
            tickformat=",.2f",
            row=implied_move_row,
            col=1,
        )
    return fig


def plot_historical_implied_move_candlestick_view(
    premium_history: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    target_dtes: Sequence[int | float] | None = None,
    ticker_label: str | None = None,
    open_column: str = "Open",
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
    template: str = "plotly_white",
    show_implied_move_shading: bool = True,
) -> go.Figure:
    """Overlay historical candlesticks with fixed-DTE implied-move price ranges."""
    if premium_history is None or premium_history.empty:
        raise ValueError("No historical implied-move observations are available.")
    if "as_of_date" not in premium_history.columns:
        raise ValueError("premium_history is missing required column: as_of_date")

    ohlc = (
        _prepare_ohlc_price_frame(
            price_history,
            open_column=open_column,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
        )
        .rename_axis("as_of_date")
        .reset_index()
    )
    frame = premium_history.copy()
    frame["as_of_date"] = (
        pd.to_datetime(frame["as_of_date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    frame = frame.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    target_specs: list[dict[str, str]] = []
    for target_dte in _normalize_target_dtes(target_dtes):
        target_label = _format_target_dte(target_dte)
        spec = {
            "label": f"{target_label} DTE",
            "move_pct": f"{target_label} DTE Implied Move %",
            "move": f"{target_label} DTE Implied Move",
            "lower": f"{target_label} DTE Implied Lower Price",
            "upper": f"{target_label} DTE Implied Upper Price",
            "realized_move_pct": f"{target_label} DTE Realized Move %",
            "realized_move": f"{target_label} DTE Realized Move",
            "realized_lower": f"{target_label} DTE Realized Lower Price",
            "realized_upper": f"{target_label} DTE Realized Upper Price",
            "forward_realized_move_pct": (
                f"{target_label} DTE Forward Realized Move %"
            ),
            "forward_realized_move": f"{target_label} DTE Forward Realized Move",
            "forward_realized_lower": (
                f"{target_label} DTE Forward Realized Lower Price"
            ),
            "forward_realized_upper": (
                f"{target_label} DTE Forward Realized Upper Price"
            ),
            "percentile": f"{target_label} DTE IV Percentile",
        }
        if {
            spec["move_pct"],
            spec["move"],
            spec["lower"],
            spec["upper"],
        }.issubset(frame.columns):
            target_specs.append(spec)

    if not target_specs:
        raise ValueError("No fixed-DTE implied-move price columns are available to plot.")

    merged = ohlc.merge(frame, on="as_of_date", how="inner")
    if merged.empty:
        raise ValueError(
            "No overlapping dates are available between price history and implied moves."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=merged["as_of_date"],
            open=merged["Open"],
            high=merged["High"],
            low=merged["Low"],
            close=merged["Close"],
            name=f"{ticker_label or 'Underlying'} candlestick",
            increasing_line_color="#16A34A",
            decreasing_line_color="#DC2626",
            increasing_fillcolor="rgba(22, 163, 74, 0.55)",
            decreasing_fillcolor="rgba(220, 38, 38, 0.55)",
        )
    )

    trace_groups: dict[str, tuple[int, int, list[bool | str]]] = {}
    for spec in target_specs:
        label = spec["label"]
        realized_columns = [
            spec["realized_move_pct"],
            spec["realized_move"],
            spec["realized_lower"],
            spec["realized_upper"],
        ]
        has_realized_range = set(realized_columns).issubset(merged.columns)
        forward_realized_columns = [
            spec["forward_realized_move_pct"],
            spec["forward_realized_move"],
            spec["forward_realized_lower"],
            spec["forward_realized_upper"],
        ]
        has_forward_realized_range = set(forward_realized_columns).issubset(
            merged.columns
        )
        for column in [
            spec["move_pct"],
            spec["move"],
            spec["lower"],
            spec["upper"],
            *[column for column in realized_columns if column in merged.columns],
            *[
                column
                for column in forward_realized_columns
                if column in merged.columns
            ],
        ]:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        has_percentile = spec["percentile"] in merged.columns
        if has_percentile:
            merged[spec["percentile"]] = pd.to_numeric(
                merged[spec["percentile"]],
                errors="coerce",
            )

        hover_data = np.column_stack(
            [
                merged[spec["move"]],
                merged[spec["move_pct"]],
                merged[spec["lower"]],
                merged[spec["upper"]],
            ]
        )
        start_index = len(fig.data)
        group_visibility: list[bool | str] = []
        lower_trace_style: dict[str, str] = {}
        if show_implied_move_shading:
            lower_trace_style = {
                "fill": "tonexty",
                "fillcolor": "rgba(37, 99, 235, 0.12)",
            }
        fig.add_trace(
            go.Scatter(
                x=merged["as_of_date"],
                y=merged[spec["upper"]],
                customdata=hover_data,
                mode="lines",
                name=f"{label} implied upper price",
                visible=False,
                line=dict(color="#2563EB", width=2.2),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Upper price: %{y:,.2f}<br>"
                    "Implied move: %{customdata[0]:,.2f} "
                    "(%{customdata[1]:.2%})<extra></extra>"
                ),
            )
        )
        group_visibility.append(True)
        fig.add_trace(
            go.Scatter(
                x=merged["as_of_date"],
                y=merged[spec["lower"]],
                customdata=hover_data,
                mode="lines",
                name=f"{label} implied lower price",
                visible=False,
                line=dict(color="#F59E0B", width=2.2),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Lower price: %{y:,.2f}<br>"
                    "Implied move: %{customdata[0]:,.2f} "
                    "(%{customdata[1]:.2%})<extra></extra>"
                ),
                **lower_trace_style,
            )
        )
        group_visibility.append(True)
        if has_realized_range:
            realized_hover_data = np.column_stack(
                [
                    merged[spec["realized_move"]],
                    merged[spec["realized_move_pct"]],
                    merged[spec["realized_lower"]],
                    merged[spec["realized_upper"]],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=merged[spec["realized_upper"]],
                    customdata=realized_hover_data,
                    mode="lines",
                    name=f"{label} realized upper price",
                    visible=False,
                    line=dict(color="#22D3EE", width=2.1, dash="dot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Realized upper price: %{y:,.2f}<br>"
                        "Realized-vol move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                )
            )
            group_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=merged[spec["realized_lower"]],
                    customdata=realized_hover_data,
                    mode="lines",
                    name=f"{label} realized lower price",
                    visible=False,
                    line=dict(color="#EC4899", width=2.1, dash="dot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>Realized lower price: %{y:,.2f}<br>"
                        "Realized-vol move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                )
            )
            group_visibility.append(True)
        if has_forward_realized_range:
            forward_realized_hover_data = np.column_stack(
                [
                    merged[spec["forward_realized_move"]],
                    merged[spec["forward_realized_move_pct"]],
                    merged[spec["forward_realized_lower"]],
                    merged[spec["forward_realized_upper"]],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=merged[spec["forward_realized_upper"]],
                    customdata=forward_realized_hover_data,
                    mode="lines",
                    name=f"{label} forward RV upper price",
                    visible=False,
                    line=dict(color="#0F766E", width=2.0, dash="dashdot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Forward RV upper price: %{y:,.2f}<br>"
                        "Forward RV move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                )
            )
            group_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=merged[spec["forward_realized_lower"]],
                    customdata=forward_realized_hover_data,
                    mode="lines",
                    name=f"{label} forward RV lower price",
                    visible=False,
                    line=dict(color="#7C3AED", width=2.0, dash="dashdot"),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Forward RV lower price: %{y:,.2f}<br>"
                        "Forward RV move: %{customdata[0]:,.2f} "
                        "(%{customdata[1]:.2%})<extra></extra>"
                    ),
                )
            )
            group_visibility.append(True)
        if show_implied_move_shading and has_percentile:
            high_percentile_mask = merged[spec["percentile"]].gt(0.5)
            high_upper = merged[spec["upper"]].where(high_percentile_mask)
            high_lower = merged[spec["lower"]].where(high_percentile_mask)
            high_hover_data = np.column_stack(
                [
                    merged[spec["move"]],
                    merged[spec["move_pct"]],
                    merged[spec["percentile"]],
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=high_upper,
                    customdata=high_hover_data,
                    mode="lines",
                    name=f"{label} high-IV upper",
                    visible=False,
                    showlegend=False,
                    line=dict(color="rgba(245, 158, 11, 0)", width=0),
                    hoverinfo="skip",
                )
            )
            group_visibility.append(True)
            fig.add_trace(
                go.Scatter(
                    x=merged["as_of_date"],
                    y=high_lower,
                    customdata=high_hover_data,
                    mode="lines",
                    name=f"{label} IV percentile > 50%",
                    visible=False,
                    fill="tonexty",
                    fillcolor="rgba(245, 158, 11, 0.28)",
                    line=dict(color="rgba(245, 158, 11, 0)", width=0),
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>IV percentile: "
                        "%{customdata[2]:.1%}<br>Implied move: "
                        "%{customdata[0]:,.2f} (%{customdata[1]:.2%})"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            group_visibility.append(True)
        trace_groups[label] = (start_index, len(fig.data), group_visibility)

    default_label = (
        "30 DTE"
        if "30 DTE" in trace_groups
        else next(iter(trace_groups))
    )
    start, stop, default_visibility = trace_groups[default_label]
    for trace_index, trace_visibility in zip(
        range(start, stop),
        default_visibility,
        strict=False,
    ):
        fig.data[trace_index].visible = trace_visibility

    dropdown_buttons = []
    for label, (start, stop, group_visibility) in trace_groups.items():
        visibility = [True] + [False] * (len(fig.data) - 1)
        visibility[start:stop] = group_visibility
        dropdown_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{ticker_label + ' ' if ticker_label else ''}"
                            f"Historical Candlestick With Implied and Forward RV Move - {label}"
                        )
                    },
                ],
            )
        )

    title_prefix = f"{ticker_label} " if ticker_label else ""
    fig.update_layout(
        title=(
            f"{title_prefix}Historical Candlestick With Implied and Forward RV Move - "
            f"{default_label}"
        ),
        template=template,
        height=760,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        updatemenus=[
            dict(
                active=list(trace_groups).index(default_label),
                buttons=dropdown_buttons,
                direction="down",
                x=0,
                y=1.12,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=120),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", tickformat=",.2f")
    return fig


def plot_historical_iv_rank_percentile_view(
    premium_history: pd.DataFrame,
    *,
    target_dtes: Sequence[int | float] | None = None,
    ticker_label: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot IV Rank and IV Percentile for fixed-tenor ATM IV history."""
    if premium_history is None or premium_history.empty:
        raise ValueError("No historical IV rank/percentile observations are available.")
    if "as_of_date" not in premium_history.columns:
        raise ValueError("premium_history is missing required column: as_of_date")

    frame = premium_history.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    target_values = _normalize_target_dtes(target_dtes)
    target_specs: list[dict[str, str]] = []
    if {"Mean IV Rank", "Mean IV Percentile"}.issubset(frame.columns):
        target_specs.append(
            {
                "label": "Mean",
                "rank": "Mean IV Rank",
                "percentile": "Mean IV Percentile",
            }
        )

    for target_dte in target_values:
        target_label = _format_target_dte(target_dte)
        rank_column = f"{target_label} DTE IV Rank"
        percentile_column = f"{target_label} DTE IV Percentile"
        if {rank_column, percentile_column}.issubset(frame.columns):
            target_specs.append(
                {
                    "label": f"{target_label} DTE",
                    "rank": rank_column,
                    "percentile": percentile_column,
                }
            )

    if not target_specs:
        raise ValueError("No IV Rank or IV Percentile columns are available to plot.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.5, 0.5],
        subplot_titles=("IV Rank", "IV Percentile"),
    )

    trace_groups: dict[str, tuple[int, int]] = {}
    for spec in target_specs:
        label = spec["label"]
        clean = frame[["as_of_date", spec["rank"], spec["percentile"]]].copy()
        clean[spec["rank"]] = pd.to_numeric(clean[spec["rank"]], errors="coerce")
        clean[spec["percentile"]] = pd.to_numeric(
            clean[spec["percentile"]],
            errors="coerce",
        )
        clean = clean.replace([np.inf, -np.inf], np.nan)

        start_index = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[spec["rank"]],
                mode="lines",
                name=f"{label} IV Rank",
                visible=False,
                line=dict(color="#2563EB", width=2.8),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>IV Rank: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=clean["as_of_date"],
                y=clean[spec["percentile"]],
                mode="lines",
                name=f"{label} IV Percentile",
                visible=False,
                line=dict(color="#F59E0B", width=2.8),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>IV Percentile: %{y:.1%}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        trace_groups[label] = (start_index, len(fig.data))

    default_label = (
        "30 DTE"
        if "30 DTE" in trace_groups
        else "Mean"
        if "Mean" in trace_groups
        else next(iter(trace_groups))
    )
    start, stop = trace_groups[default_label]
    for trace_index in range(start, stop):
        fig.data[trace_index].visible = True

    dropdown_buttons = []
    for label, (start, stop) in trace_groups.items():
        visibility = [False] * len(fig.data)
        visibility[start:stop] = [True] * (stop - start)
        dropdown_buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {
                        "title": (
                            f"{ticker_label + ' ' if ticker_label else ''}"
                            f"Historical IV Rank and Percentile - {label}"
                        )
                    },
                ],
            )
        )

    title_prefix = f"{ticker_label} " if ticker_label else ""
    fig.update_layout(
        title=f"{title_prefix}Historical IV Rank and Percentile - {default_label}",
        template=template,
        height=700,
        hovermode="x unified",
        updatemenus=[
            dict(
                active=list(trace_groups).index(default_label),
                buttons=dropdown_buttons,
                direction="down",
                x=0,
                y=1.16,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(t=130),
    )
    fig.add_hline(y=0.5, line_color="#F59E0B", line_dash="dot", row=1, col=1)
    fig.add_hline(y=0.5, line_color="#F59E0B", line_dash="dot", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="IV Rank", tickformat=".0%", range=[0, 1], row=1, col=1)
    fig.update_yaxes(
        title_text="IV Percentile",
        tickformat=".0%",
        range=[0, 1],
        row=2,
        col=1,
    )
    return fig

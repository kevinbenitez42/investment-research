"""Stacked monthly, weekly, and daily seasonality view."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._shared import finalize_dark_figure


DEFAULT_BAR_COLOR = "#3B82F6"
HIGHLIGHT_BAR_COLOR = "#F59E0B"
MEDIAN_COLOR = "#EF4444"
CURRENT_YEAR_COLOR = "#22C55E"


@dataclass(frozen=True)
class _SeasonalityPanel:
    frequency: str
    frequency_label: str
    period_mean: pd.Series
    period_median: pd.Series
    current_returns: pd.Series | None
    current_period_label: str | None


def _coerce_return_series(data) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("seasonality data must be a Series or single-column DataFrame.")
        data = data.iloc[:, 0]
    elif not isinstance(data, pd.Series):
        raise TypeError("seasonality data must be a pandas Series or single-column DataFrame.")

    series = data.copy().dropna().sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    return series.astype(float)


def _coerce_as_of(as_of, series_values) -> pd.Timestamp:
    if as_of is not None:
        return pd.Timestamp(as_of)

    latest_dates = [
        series.index.max()
        for series in series_values
        if isinstance(series, pd.Series) and not series.empty
    ]
    if latest_dates:
        return pd.Timestamp(max(latest_dates))
    return pd.Timestamp.today().normalize()


def _prepare_monthly_panel(data: pd.Series, as_of: pd.Timestamp) -> _SeasonalityPanel:
    periods = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    period_mean = data.groupby(data.index.month).mean().reindex(range(1, 13))
    period_median = data.groupby(data.index.month).median().reindex(range(1, 13))
    period_mean.index = periods
    period_median.index = periods

    current_returns = None
    current_year_returns = data.loc[data.index.year == as_of.year]
    if not current_year_returns.empty:
        current_returns = current_year_returns.groupby(current_year_returns.index.month).last().reindex(range(1, 13))
        current_returns.index = periods

    return _SeasonalityPanel(
        frequency="monthly",
        frequency_label="Month",
        period_mean=period_mean,
        period_median=period_median,
        current_returns=current_returns,
        current_period_label=as_of.strftime("%b"),
    )


def _prepare_weekly_panel(data: pd.Series, as_of: pd.Timestamp) -> _SeasonalityPanel:
    weekly_data = data.to_frame(name="Return")
    weekly_data["Month_Num"] = weekly_data.index.month
    weekly_data["Month_Name"] = weekly_data.index.strftime("%b")
    weekly_data["Week_of_Month"] = weekly_data.index.to_series().apply(lambda value: (value.day - 1) // 7 + 1)
    weekly_data["Period_Label"] = weekly_data["Month_Name"] + " / Week " + weekly_data["Week_of_Month"].astype(str)
    weekly_data["Period_Num"] = weekly_data["Month_Num"] * 10 + weekly_data["Week_of_Month"]

    period_stats = (
        weekly_data.groupby(["Period_Num", "Period_Label"])["Return"]
        .agg(["mean", "median"])
        .reset_index()
        .sort_values("Period_Num")
    )

    period_mean = period_stats.set_index("Period_Label")["mean"]
    period_median = period_stats.set_index("Period_Label")["median"]

    current_period_num = as_of.month * 10 + ((as_of.day - 1) // 7 + 1)
    current_period_labels = period_stats.loc[
        period_stats["Period_Num"] == current_period_num,
        "Period_Label",
    ].values
    current_period_label = current_period_labels[0] if len(current_period_labels) else None

    current_returns = None
    weekly_data_current_year = weekly_data.loc[weekly_data.index.year == as_of.year]
    if not weekly_data_current_year.empty:
        current_returns = weekly_data_current_year.set_index("Period_Label")["Return"]
        current_returns = current_returns[~current_returns.index.duplicated(keep="last")].reindex(period_mean.index)

    return _SeasonalityPanel(
        frequency="weekly",
        frequency_label="Month / Week of Month",
        period_mean=period_mean,
        period_median=period_median,
        current_returns=current_returns,
        current_period_label=current_period_label,
    )


def _window_around_current_period(series: pd.Series, current_period_label: str | None, window_size: int) -> pd.Series:
    if current_period_label not in series.index:
        return series

    current_idx = series.index.get_loc(current_period_label)
    start_idx = current_idx - window_size
    end_idx = current_idx + window_size + 1
    if start_idx < 0:
        return pd.concat([series.iloc[start_idx:], series.iloc[:end_idx]])
    if end_idx > len(series):
        return pd.concat([series.iloc[start_idx:], series.iloc[: end_idx - len(series)]])
    return series.iloc[start_idx:end_idx]


def _prepare_daily_panel(data: pd.Series, as_of: pd.Timestamp, *, window_size: int) -> _SeasonalityPanel:
    periods = sorted(data.index.strftime("%m/%d").unique())
    current_period_label = as_of.strftime("%m/%d")
    if current_period_label not in periods:
        current_period_label = None

    period_mean = data.groupby(data.index.strftime("%m-%d")).mean()
    period_median = data.groupby(data.index.strftime("%m-%d")).median()
    period_mean.index = periods
    period_median.index = periods

    period_mean = _window_around_current_period(period_mean, current_period_label, window_size)
    period_median = _window_around_current_period(period_median, current_period_label, window_size)

    current_returns = None
    current_year_returns = data.loc[data.index.year == as_of.year].copy()
    if not current_year_returns.empty:
        current_year_returns.index = current_year_returns.index.strftime("%m/%d")
        current_returns = current_year_returns.reindex(period_mean.index)

    return _SeasonalityPanel(
        frequency="daily",
        frequency_label="Day (MM/DD)",
        period_mean=period_mean,
        period_median=period_median,
        current_returns=current_returns,
        current_period_label=current_period_label,
    )


def _bar_colors(panel: _SeasonalityPanel) -> list[str]:
    if panel.period_mean.empty:
        return []

    if panel.current_period_label in panel.period_mean.index:
        return [
            HIGHLIGHT_BAR_COLOR if period == panel.current_period_label else DEFAULT_BAR_COLOR
            for period in panel.period_mean.index
        ]

    fallback_label = panel.period_mean.index[-1]
    return [
        HIGHLIGHT_BAR_COLOR if period == fallback_label else DEFAULT_BAR_COLOR
        for period in panel.period_mean.index
    ]


def _median_stems(period_median: pd.Series) -> tuple[list, list]:
    stem_x = []
    stem_y = []
    for period, value in zip(period_median.index, period_median.values):
        if pd.isna(value):
            continue
        stem_x.extend([period, period, None])
        stem_y.extend([0, value, None])
    return stem_x, stem_y


def _add_panel_traces(fig: go.Figure, panel: _SeasonalityPanel, *, row: int, as_of: pd.Timestamp) -> None:
    showlegend = row == 1
    if panel.period_mean.empty:
        fig.add_annotation(
            text=f"No {panel.frequency} seasonality data available.",
            x=0.5,
            xref=f"x{row} domain" if row > 1 else "x domain",
            y=0.5,
            yref=f"y{row} domain" if row > 1 else "y domain",
            showarrow=False,
            row=row,
            col=1,
        )
        return

    fig.add_trace(
        go.Bar(
            x=panel.period_mean.index,
            y=panel.period_mean.values,
            name="Mean Return",
            marker_color=_bar_colors(panel),
            hovertemplate="Mean: %{y:.4f}<extra></extra>",
            legendgroup="mean",
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )

    median_stem_x, median_stem_y = _median_stems(panel.period_median)
    fig.add_trace(
        go.Scatter(
            x=median_stem_x,
            y=median_stem_y,
            mode="lines",
            name="Median Stem",
            line=dict(color=MEDIAN_COLOR, width=2),
            hoverinfo="skip",
            showlegend=False,
            legendgroup="median",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=panel.period_median.index,
            y=panel.period_median.values,
            mode="markers",
            name="Median Return",
            marker=dict(size=8, color=MEDIAN_COLOR),
            hovertemplate="Median: %{y:.4f}<extra></extra>",
            legendgroup="median",
            showlegend=showlegend,
        ),
        row=row,
        col=1,
    )

    if panel.current_returns is not None and panel.current_returns.notna().any():
        current_returns = panel.current_returns.reindex(panel.period_mean.index)
        fig.add_trace(
            go.Scatter(
                x=current_returns.index,
                y=current_returns.values,
                mode="lines+markers",
                name=f"{as_of.year} Return",
                line=dict(color=CURRENT_YEAR_COLOR, width=2),
                marker=dict(size=8, color=CURRENT_YEAR_COLOR, symbol="diamond"),
                hovertemplate="Current Year: %{y:.4f}<extra></extra>",
                legendgroup="current_year",
                showlegend=showlegend,
            ),
            row=row,
            col=1,
        )

    fig.add_hline(
        y=0,
        line_color="rgba(148, 163, 184, 0.55)",
        line_dash="dash",
        line_width=1,
        row=row,
        col=1,
    )


def plot_seasonality_stack_view(
    *,
    monthly_returns,
    weekly_returns,
    daily_returns,
    ticker_label="Asset",
    as_of=None,
    daily_window_size=30,
    template="plotly_dark",
):
    """Compose monthly, weekly, and daily return seasonality panels."""
    monthly_series = _coerce_return_series(monthly_returns)
    weekly_series = _coerce_return_series(weekly_returns)
    daily_series = _coerce_return_series(daily_returns)
    as_of = _coerce_as_of(as_of, (monthly_series, weekly_series, daily_series))

    panels = [
        _prepare_monthly_panel(monthly_series, as_of),
        _prepare_weekly_panel(weekly_series, as_of),
        _prepare_daily_panel(daily_series, as_of, window_size=int(daily_window_size)),
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.27, 0.31, 0.42],
        subplot_titles=(
            "Monthly Seasonality",
            "Weekly Seasonality",
            "Daily Seasonality",
        ),
    )

    for row, panel in enumerate(panels, start=1):
        _add_panel_traces(fig, panel, row=row, as_of=as_of)
        fig.update_xaxes(
            title_text=panel.frequency_label,
            tickangle=-45,
            type="category",
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="Return", row=row, col=1)

    fig.update_layout(
        title=f"{ticker_label} Return Seasonality",
        template=template,
        height=1200,
        bargap=0.12,
        hovermode="x unified",
        legend=dict(
            title="Metrics",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return finalize_dark_figure(fig)

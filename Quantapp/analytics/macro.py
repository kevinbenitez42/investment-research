"""Frequency-aware transformations used by country macro dashboards."""

from __future__ import annotations

import pandas as pd


def transform_macro_series(
    series: pd.Series,
    view: str = "level",
    *,
    periods: int | None = None,
    window: int | None = None,
    observations_per_year: int | None = None,
) -> pd.Series:
    """Transform a macro series without assuming a particular frequency."""
    values = pd.to_numeric(series, errors="coerce").sort_index().dropna()
    if view == "level":
        return values
    if view == "change":
        if periods is None:
            raise ValueError("periods is required for change")
        if periods < 1:
            raise ValueError("periods must be at least 1")
        return values.diff(periods).dropna()
    if view in {"yoy", "currency_strength"}:
        if periods is None:
            raise ValueError(f"periods is required for {view}")
        if periods < 1:
            raise ValueError("periods must be at least 1")
        return values.pct_change(periods, fill_method=None).mul(100).dropna()
    if view == "inverse_currency_strength":
        if periods is None:
            raise ValueError("periods is required for inverse_currency_strength")
        if periods < 1:
            raise ValueError("periods must be at least 1")
        return values.rdiv(1).pct_change(periods, fill_method=None).mul(100).dropna()
    if view == "annualized_change":
        if periods is None:
            raise ValueError("periods is required for annualized_change")
        if periods < 1 or observations_per_year is None or observations_per_year < 1:
            raise ValueError("positive periods and observations_per_year are required")
        return ((values / values.shift(periods)) ** (observations_per_year / periods) - 1).mul(100).dropna()
    if view == "rolling_mean":
        if window is None:
            raise ValueError("window is required for rolling_mean")
        if window < 1:
            raise ValueError("window must be at least 1")
        return values.rolling(window).mean().dropna()
    raise ValueError(f"Unsupported macro view: {view}")


__all__ = ["transform_macro_series"]

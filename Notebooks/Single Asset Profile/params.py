"""Shared parameter defaults for the Single Asset Profile notebooks."""

from __future__ import annotations

from copy import deepcopy


TIMEFRAME_PROFILES = {
    "swing": {"short": 3, "mid": 9, "long": 21},
    "position": {"short": 21, "mid": 50, "long": 200},
    "structural": {"short": 200, "mid": 500, "long": 1000},
}


_BASE_SINGLE_ASSET_NOTEBOOK_PARAMS = {
    "ticker_str": "UNH",
    "interval": "1d",
    "risk_free_ticker": "^IRX",
}


_SINGLE_ASSET_PROFILE_PARAMS = {
    **_BASE_SINGLE_ASSET_NOTEBOOK_PARAMS,
    "period": "20y",
    "benchmark_tickers": ["SPY", "XLV"],
    "length_of_plots": 20,
    "trading_strategy": "position",
    "var_position_value": None,
}


_FACTOR_ATTRIBUTION_PARAMS = {
    **_BASE_SINGLE_ASSET_NOTEBOOK_PARAMS,
    "analysis_period": "max",
    "rolling_window": 252,
    "ff_verbose": True,
}


_OPTIONS_PRICING_PARAMS = {
    **_BASE_SINGLE_ASSET_NOTEBOOK_PARAMS,
    "period": "10y",
    "risk_free_rate": 0.02 / 252,
    "time_frame_week": 7,
    "time_frame_short": 21,
    "time_frame_mid": 50,
    "time_frame_long": 200,
}


def _clone(mapping: dict) -> dict:
    """Return a defensive copy so each notebook can safely mutate local values."""
    return deepcopy(mapping)


def get_single_asset_profile_params() -> dict:
    """Return shared defaults for the single-asset profile notebooks."""
    return _clone(_SINGLE_ASSET_PROFILE_PARAMS)


def get_factor_attribution_params() -> dict:
    """Return shared defaults for the factor-attribution notebook."""
    return _clone(_FACTOR_ATTRIBUTION_PARAMS)


def get_options_pricing_params() -> dict:
    """Return shared defaults for the options-pricing notebook."""
    return _clone(_OPTIONS_PRICING_PARAMS)


def resolve_time_frame_map(strategy: str) -> dict[str, int]:
    """Validate a strategy name and return the matching timeframe profile."""
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in TIMEFRAME_PROFILES:
        raise ValueError(
            f"Invalid trading_strategy '{strategy}'. "
            f"Expected one of: {list(TIMEFRAME_PROFILES.keys())}"
        )
    return deepcopy(TIMEFRAME_PROFILES[normalized_strategy])

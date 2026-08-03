"""Parameters shared by single-asset analysis notebooks across research folders.

Edit ``COMMON_SINGLE_ASSET_PARAMS`` when the Single Asset research notebooks should
use a different asset or common history configuration. Every parameter that
is not universal remains notebook-local.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


COMMON_SINGLE_ASSET_PARAMS = {
    "ticker_str": "NVDA",
    "interval": "1d",
    "period": "20y",
}


def get_single_asset_params(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an isolated copy of the common Single Asset parameters."""
    params = deepcopy(COMMON_SINGLE_ASSET_PARAMS)
    if overrides:
        unknown = sorted(set(overrides).difference(params))
        if unknown:
            raise KeyError(f"Unknown common Single Asset parameter(s): {unknown}")
        params.update(deepcopy(dict(overrides)))
    return params

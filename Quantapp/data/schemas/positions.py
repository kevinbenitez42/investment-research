"""Position data contracts."""

from __future__ import annotations

SCHWAB_OPTION_SYMBOL_PATTERN_TEXT = (
    r"^(?P<underlying>[A-Z]{1,6})(?P<expiration>\d{6})(?P<option_type>[CP])(?P<strike>\d{8})$"
)

SCHWAB_OPTION_POSITION_COLUMNS = [
    "symbol",
    "underlying",
    "expiration",
    "days_to_expiration",
    "option_type",
    "strike",
    "underlying_symbol",
    "put_call",
    "long_quantity",
    "short_quantity",
    "net_quantity",
    "average_price",
    "directional_value",
]

SCHWAB_NET_DIRECTION_COLUMNS = [
    "directional_value",
    "sentiment",
    "sign",
]

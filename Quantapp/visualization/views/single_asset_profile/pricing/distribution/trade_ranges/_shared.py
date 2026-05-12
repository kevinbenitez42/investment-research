"""Shared helpers for trade-range views."""

from __future__ import annotations

HEADER_TOP_MARGIN = 150
HEADER_TITLE_Y = 0.97
HEADER_MENU_Y = 1.08


def header_margin(top=None):
    return dict(t=HEADER_TOP_MARGIN if top is None else int(top))


def header_title(text):
    return dict(
        text=str(text),
        x=0.5,
        xanchor="center",
        y=HEADER_TITLE_Y,
        yanchor="top",
    )


def dropdown_menu(
    *,
    buttons,
    x,
    active=None,
    y=None,
    direction="down",
    showactive=True,
    xanchor="left",
    yanchor="top",
    **overrides,
):
    menu = dict(
        type="dropdown",
        buttons=buttons,
        direction=direction,
        showactive=showactive,
        x=x,
        xanchor=xanchor,
        y=HEADER_MENU_Y if y is None else y,
        yanchor=yanchor,
    )
    if active is not None:
        menu["active"] = active
    menu.update(overrides)
    return menu


def coerce_positive_int(value):
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def preferred_numeric_window(options, preferred=200):
    normalized = []
    seen = set()
    for option in options:
        coerced = coerce_positive_int(option)
        if coerced is None or coerced in seen:
            continue
        normalized.append(coerced)
        seen.add(coerced)

    if not normalized:
        return None
    if preferred in seen:
        return preferred
    return max(normalized)

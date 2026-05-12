"""Shared visualization utilities used across higher-level views."""

from .display import (
    PLOTLY_NOTEBOOK_CONFIG,
    PLOTLY_NOTEBOOK_RENDERERS,
    configure_plotly_notebook_renderers,
    plotly_notebook_config,
    show_plotly_figure,
)
from .theme import (
    PLOTLY_DARK_AXIS,
    PLOTLY_DARK_BACKGROUND,
    PLOTLY_DARK_GRID,
    PLOTLY_DARK_MENU_BG,
    PLOTLY_DARK_TEXT,
    PLOTLY_DARK_THEME,
    apply_plotly_dark_theme,
    coerce_dark_foreground,
)

__all__ = [
    "PLOTLY_DARK_AXIS",
    "PLOTLY_DARK_BACKGROUND",
    "PLOTLY_DARK_GRID",
    "PLOTLY_DARK_MENU_BG",
    "PLOTLY_DARK_TEXT",
    "PLOTLY_DARK_THEME",
    "PLOTLY_NOTEBOOK_CONFIG",
    "PLOTLY_NOTEBOOK_RENDERERS",
    "apply_plotly_dark_theme",
    "coerce_dark_foreground",
    "configure_plotly_notebook_renderers",
    "plotly_notebook_config",
    "show_plotly_figure",
]

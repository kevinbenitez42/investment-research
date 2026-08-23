"""Notebook display helpers for Plotly figures."""

from __future__ import annotations

import plotly.io as pio

from .theme import apply_plotly_dark_theme


PLOTLY_NOTEBOOK_CONFIG = {"responsive": True, "scrollZoom": True}
PLOTLY_NOTEBOOK_RENDERERS = ("plotly_mimetype", "notebook", "notebook_connected", "jupyterlab")


def plotly_notebook_config(config=None):
    """Return the shared notebook Plotly config merged with optional overrides."""
    merged_config = PLOTLY_NOTEBOOK_CONFIG.copy()
    if config:
        merged_config.update(config)
    return merged_config


def configure_plotly_notebook_renderers(*, config=None, renderer_names=PLOTLY_NOTEBOOK_RENDERERS):
    """Apply the shared notebook Plotly config to known notebook renderers."""
    renderer_config = plotly_notebook_config(config)
    for renderer_name in renderer_names:
        try:
            pio.renderers[renderer_name].config = renderer_config.copy()
        except Exception:
            pass
    return renderer_config


def show_plotly_figure(fig, *, config=None, apply_theme=True, **layout_kwargs):
    """Show a Plotly figure with Quantapp's notebook defaults."""
    fig.update_layout(autosize=True, **layout_kwargs)
    if apply_theme:
        apply_plotly_dark_theme(fig)
    fig.show(config=plotly_notebook_config(config))

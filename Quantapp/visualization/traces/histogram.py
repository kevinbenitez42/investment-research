"""Histogram trace builders used by composed visualization views."""

from __future__ import annotations

import plotly.graph_objects as go


def build_optimal_window_histogram_trace(values, *, nbins):
    """Build the optimal-window histogram trace."""
    return go.Histogram(
        x=values,
        nbinsx=nbins,
        name="Optimal Window Size Distribution",
        showlegend=False,
    )

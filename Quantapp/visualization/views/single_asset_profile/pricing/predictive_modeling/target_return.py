"""Predictive-model target-return views."""

from __future__ import annotations

import plotly.graph_objects as go


def plot_forward_return_target_view(
    model_frame,
    *,
    ticker_label="Asset",
    target_column="next_return_3d",
    target_horizon_days=3,
    template="plotly_dark",
):
    """Compose the forward-return target used to derive the classifier label."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=model_frame.index,
            y=model_frame[target_column] * 100,
            name=f"Forward {target_horizon_days}-day return",
            line={"color": "#60a5fa", "width": 2},
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        title=f"{ticker_label} forward {target_horizon_days}-day return target",
        template=template,
        xaxis_title="Date",
        yaxis_title="Forward return (%)",
        height=500,
    )
    return fig

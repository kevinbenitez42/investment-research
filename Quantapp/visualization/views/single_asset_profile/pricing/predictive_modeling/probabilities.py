"""Predictive-model probability comparison views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_prediction_probability_view(
    probability_frame,
    *,
    thresholds,
    default_threshold=0.50,
    ticker_label="Asset",
    probability_column="predicted_probability_up_3d",
    realized_column="target_up_3d",
    target_horizon_days=3,
    template="plotly_dark",
):
    """Compose predicted probabilities against realized binary outcomes."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=probability_frame.index,
            y=probability_frame[probability_column] * 100,
            name=f"Predicted probability of higher close in {target_horizon_days} days",
            line={"color": "#34d399", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=probability_frame.index,
            y=probability_frame[realized_column] * 100,
            name="Realized outcome (0 or 100)",
            mode="markers",
            marker={"color": "#fbbf24", "size": 7, "opacity": 0.7},
        )
    )
    for threshold in thresholds:
        is_default = np.isclose(threshold, default_threshold)
        fig.add_hline(
            y=threshold * 100,
            line_dash="dot",
            line_color="#cbd5e1" if is_default else "#64748b",
            opacity=0.65 if is_default else 0.25,
        )

    threshold_text = ", ".join(f"{threshold * 100:.1f}%" for threshold in thresholds)
    fig.update_layout(
        title=(
            f"{ticker_label} logistic regression probability of a higher close in "
            f"{target_horizon_days} trading days<br><sup>Decision thresholds shown: {threshold_text}</sup>"
        ),
        template=template,
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        height=500,
    )
    return fig

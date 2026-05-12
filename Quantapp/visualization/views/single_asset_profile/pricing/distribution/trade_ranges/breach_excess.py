"""Trade-range excess breach-rate heatmap view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _build_trade_range_breach_heatmap_payload(summary_frame, confidence, supported_horizons, windows):
    confidence_frame = summary_frame.loc[summary_frame["confidence"].eq(float(confidence))].copy()
    confidence_frame["date_label"] = confidence_frame["latest_calibration_date"].map(
        lambda value: pd.Timestamp(value).strftime("%Y-%m-%d") if pd.notna(value) else ""
    )

    excess_rows = []
    customdata_rows = []
    for window in windows:
        window_frame = confidence_frame.loc[
            confidence_frame["window"].eq(int(window))
        ].set_index("horizon").reindex(supported_horizons)

        excess_row = window_frame["excess_breach_rate"].mul(100.0).tolist()
        actual_row = window_frame["actual_breach_rate"].mul(100.0).tolist()
        expected_row = window_frame["expected_breach_rate"].mul(100.0).tolist()
        date_row = window_frame["date_label"].tolist()

        excess_rows.append(excess_row)
        customdata_rows.append(
            [
                [actual, expected, excess, date_label]
                for actual, expected, excess, date_label in zip(
                    actual_row,
                    expected_row,
                    excess_row,
                    date_row,
                )
            ]
        )

    return np.asarray(excess_rows, dtype=float), np.asarray(customdata_rows, dtype=object)


def _build_breach_excess_heatmap_trace(
    *,
    z,
    x,
    y,
    customdata,
    confidence_label,
):
    trace = go.Heatmap(
        z=z,
        x=x,
        y=y,
        visible=True,
    )
    trace.coloraxis = "coloraxis"
    trace.customdata = customdata
    trace.xgap = 1
    trace.ygap = 1
    trace.hovertemplate = (
        "Holding Horizon: %{x}-session"
        "<br>Session Lookback: %{y}"
        f"<br>Confidence: {confidence_label}"
        "<br>Actual Either-Side Breach Rate: %{customdata[0]:.2f}%"
        "<br>Expected Either-Side Breach Rate: %{customdata[1]:.2f}%"
        "<br>Excess Either-Side Breach Rate: %{customdata[2]:.2f} pts"
        "<br>Latest Calibration Date: %{customdata[3]}"
        "<extra></extra>"
    )
    return trace


def plot_trade_range_breach_excess_view(
    summary_frame,
    *,
    supported_horizons,
    windows,
    confidence_levels,
    confidence_labels,
    ticker_label="Asset",
    target_end,
    template="plotly_dark",
):
    """Compose the empirical excess breach-rate heatmap view."""
    excess_values = summary_frame["excess_breach_rate"].dropna()
    color_limit = float(excess_values.abs().max() * 100.0) if not excess_values.empty else 1.0
    color_limit = max(color_limit, 1.0)
    window_labels = [f"{int(window)}-session" for window in windows]
    tick_step = 20 if target_end > 120 else 10
    tick_values = [
        horizon
        for horizon in supported_horizons
        if horizon in (1, target_end) or horizon % tick_step == 0
    ]

    fig = make_subplots(
        rows=len(confidence_levels),
        cols=1,
        subplot_titles=tuple(
            f"{confidence_labels[float(confidence)]} Either-Side Excess Breach Rate"
            for confidence in confidence_levels
        ),
        vertical_spacing=0.12,
    )

    for confidence_index, confidence in enumerate(confidence_levels, start=1):
        z_values, customdata_values = _build_trade_range_breach_heatmap_payload(
            summary_frame,
            confidence,
            supported_horizons,
            windows,
        )
        fig.add_trace(
            _build_breach_excess_heatmap_trace(
                z=z_values,
                x=supported_horizons,
                y=window_labels,
                customdata=customdata_values,
                confidence_label=confidence_labels[float(confidence)],
            ),
            row=confidence_index,
            col=1,
        )

        fig.update_xaxes(
            title_text="Holding Horizon (Sessions)",
            tickmode="array",
            tickvals=tick_values,
            row=confidence_index,
            col=1,
        )
        fig.update_yaxes(
            title_text="Session Lookback Window",
            categoryorder="array",
            categoryarray=window_labels,
            autorange="reversed",
            row=confidence_index,
            col=1,
        )

    fig.update_layout(
        title=(
            f"{ticker_label} Empirical Either-Side Breach-Rate Calibration "
            f"(Holding Horizons 1-{target_end} Sessions; "
            f'Lookbacks {" / ".join(str(window) for window in windows)})'
        ),
        template=template,
        height=980,
        margin={"t": 120, "r": 40, "b": 70, "l": 90},
        coloraxis={
            "colorscale": "RdBu",
            "cmin": -color_limit,
            "cmax": color_limit,
            "cmid": 0.0,
            "colorbar": {"title": "Excess Breach Rate (pts)"},
        },
    )
    return fig

"""Trade-range average breach-rate view."""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from Quantapp.visualization.traces.line import build_horizontal_level_trace, build_line_trace


def _build_average_breach_trace(
    *,
    x,
    y,
    window,
    confidence,
    color,
    showlegend,
    customdata,
):
    return build_line_trace(
        x=x,
        y=y,
        mode="lines",
        name=f"{int(window)}-session lookback",
        showlegend=showlegend,
        color=color,
        width=2.25,
        customdata=customdata,
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}"
            f"<br>Session Lookback: {int(window)}-session"
            f"<br>Confidence: {float(confidence):.0%}"
            "<br>Average Either-Side Breach Rate: %{y:.2%}"
            "<br>Expected Either-Side Breach Rate: %{customdata[0]:.2%}"
            "<br>Average Excess Breach Rate: %{customdata[1]:.2%}"
            "<br>Horizons Averaged: %{customdata[2]}"
            "<extra></extra>"
        ),
    )


def _build_expected_rate_trace(x_ref, *, expected_rate, confidence):
    return build_horizontal_level_trace(
        x_ref,
        y_value=expected_rate,
        name=f"{float(confidence):.0%} Expected Rate",
        color="#94a3b8",
        width=1.5,
        dash="dot",
    )


def plot_trade_range_breach_average_view(
    average_panel,
    count_panel,
    *,
    horizons,
    windows,
    confidence_levels,
    window_colors,
    ticker_label="Asset",
    template="plotly_dark",
):
    """Compose the average rolling breach-rate view across holding horizons."""
    subplot_titles = tuple(
        f"{float(confidence):.0%} Average Either-Side Rolling Breach Rate Across Horizons"
        for confidence in confidence_levels
    )
    fig = make_subplots(
        rows=len(confidence_levels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10 if len(confidence_levels) > 1 else 0.12,
        subplot_titles=subplot_titles,
    )

    for row_index, confidence in enumerate(confidence_levels, start=1):
        expected_rate = float(min(1.0, 2.0 * (1.0 - float(confidence))))
        row_x_ref = pd.Index([])

        for window in windows:
            column_key = (float(confidence), int(window))
            if column_key not in average_panel.columns:
                continue

            average_series = average_panel[column_key].dropna().sort_index()
            if average_series.empty:
                continue
            row_x_ref = average_series.index
            horizon_count = count_panel[column_key].reindex(average_series.index).astype(int)
            customdata = np.column_stack(
                [
                    np.full(len(average_series.index), expected_rate, dtype=float),
                    average_series.to_numpy(dtype=float) - expected_rate,
                    horizon_count.to_numpy(dtype=int),
                ]
            )

            fig.add_trace(
                _build_average_breach_trace(
                    x=average_series.index,
                    y=average_series,
                    window=window,
                    confidence=confidence,
                    color=window_colors.get(int(window), "#e2e8f0"),
                    showlegend=row_index == 1,
                    customdata=customdata,
                ),
                row=row_index,
                col=1,
            )

        if len(row_x_ref) > 0:
            fig.add_trace(
                _build_expected_rate_trace(
                    row_x_ref,
                    expected_rate=expected_rate,
                    confidence=confidence,
                ),
                row=row_index,
                col=1,
            )
            fig.add_annotation(
                x=0.99,
                y=expected_rate,
                xref=f"x{row_index} domain" if row_index > 1 else "x domain",
                yref=f"y{row_index}" if row_index > 1 else "y",
                text=f"Expected {expected_rate:.2%}",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=10, color="#94a3b8"),
            )

        fig.update_yaxes(title_text="Average Rolling Breach Rate", tickformat=".2%", row=row_index, col=1)

    fig.update_xaxes(title_text="Date", row=len(confidence_levels), col=1)
    fig.update_layout(
        title=(
            f"{ticker_label} Average Either-Side Rolling Breach Rate Across Holding Horizons "
            f"(Horizons {min(horizons)}-{max(horizons)}; "
            f'Lookbacks {" / ".join(str(window) for window in windows)})'
        ),
        template=template,
        height=360 * len(confidence_levels) + 120,
        margin={"t": 120, "r": 40, "b": 70, "l": 95},
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return fig

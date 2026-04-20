from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_vix_fix_bands(
    vix_fix_series: pd.Series,
    *,
    stdev_values: tuple[float, ...] | list[float] = (-0.5, 0.5, 1.5, 3),
    num_years: int = 5,
    title: str = "VIX Fix with Mean and Standard Deviations",
) -> go.Figure:
    """Build the Block 7 VIX Fix figure with mean, standard deviation, and shading bands."""
    if not isinstance(vix_fix_series, pd.Series):
        raise TypeError("vix_fix_series must be a pandas Series.")

    data_series = vix_fix_series.dropna().sort_index()
    if data_series.empty:
        raise ValueError("vix_fix_series must contain at least one non-null value.")

    if not isinstance(data_series.index, pd.DatetimeIndex):
        data_series.index = pd.to_datetime(data_series.index)

    if num_years <= 0:
        raise ValueError("num_years must be a positive integer.")

    zoom_start = data_series.index[-1] - pd.DateOffset(years=num_years)
    zoom_data = data_series.loc[data_series.index >= zoom_start]
    if zoom_data.empty:
        zoom_data = data_series

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_series.index,
            y=data_series,
            mode="lines",
            name="VIX Fix",
            line=dict(color="yellow"),
        )
    )

    mean_val = data_series.mean()
    std_val = data_series.std()

    fig.add_hline(
        y=mean_val,
        line_color="white",
        line_dash="dash",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="bottom right",
    )

    for stdev in stdev_values:
        sd_line = mean_val + (stdev * std_val)
        fig.add_hline(
            y=sd_line,
            line_color="white",
            line_dash="dot",
            annotation_text=f"{stdev} SD: {sd_line:.2f}",
            annotation_position="bottom right",
        )

    shade_colors = [
        "rgba(0, 255, 0, 0.3)",
        "rgba(255, 255, 0, 0.5)",
        "rgba(255, 0, 0, 0.7)",
    ]
    stdev_values_sorted = sorted(stdev_values)

    for index in range(len(stdev_values_sorted) - 1):
        lower_stdev = stdev_values_sorted[index]
        upper_stdev = stdev_values_sorted[index + 1]
        y0 = mean_val + (lower_stdev * std_val)
        y1 = mean_val + (upper_stdev * std_val)
        color = shade_colors[index] if index < len(shade_colors) else "rgba(255, 0, 0, 0.7)"

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=data_series.index.min(),
            y0=y0,
            x1=data_series.index.max(),
            y1=y1,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        height=800,
        xaxis=dict(range=[zoom_start, data_series.index[-1]]),
        yaxis=dict(range=[zoom_data.min(), zoom_data.max()]),
    )

    return fig

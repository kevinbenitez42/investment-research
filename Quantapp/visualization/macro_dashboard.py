"""Reusable Plotly figures for country macroeconomic notebooks."""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_TIMEFRAMES = (30, 20, 10, 5)


def _trace_date_range(fig):
    dates = []
    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None or len(x) == 0:
            continue
        values = pd.to_datetime(x, errors="coerce", utc=True)
        values = values[~values.isna()].tz_localize(None)
        if len(values):
            dates.extend((values.min(), values.max()))
    return (min(dates), max(dates)) if dates else (None, None)


def add_cycle_bands(fig, cycle_series, fillcolor="LightGray", opacity=0.28):
    """Add cycle/recession rectangles to all subplots in a single layout update."""
    if cycle_series is None or cycle_series.dropna().empty:
        return fig
    figure_start, figure_end = _trace_date_range(fig)
    if figure_start is None:
        return fig

    cycle = cycle_series.copy()
    cycle.index = pd.to_datetime(cycle.index)
    cycle = cycle.sort_index().fillna(0).astype(float)
    prior = cycle.loc[cycle.index < figure_start].tail(1)
    cycle = pd.concat([
        prior,
        cycle.loc[(cycle.index >= figure_start) & (cycle.index <= figure_end)],
    ])

    intervals = []
    start = None
    for timestamp, value in cycle.items():
        if value >= 0.5 and start is None:
            start = max(timestamp, figure_start)
        elif value < 0.5 and start is not None:
            intervals.append((start, min(timestamp, figure_end)))
            start = None
    if start is not None:
        intervals.append((start, figure_end))

    axis_numbers = sorted(
        int(name[5:] or 1) for name in fig.layout if name.startswith("xaxis")
    )
    shapes = list(fig.layout.shapes or ())
    for x0, x1 in intervals:
        for number in axis_numbers:
            suffix = "" if number == 1 else str(number)
            shapes.append(dict(
                type="rect",
                xref=f"x{suffix}",
                yref=f"y{suffix} domain",
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                fillcolor=fillcolor,
                opacity=opacity,
                layer="below",
                line_width=0,
            ))
    fig.update_layout(shapes=shapes)
    return fig


def add_timeframe_dropdown(fig, default_years=20, timeframes=DEFAULT_TIMEFRAMES):
    """Add one date-range control that updates every subplot x-axis."""
    _, end_date = _trace_date_range(fig)
    if end_date is None:
        return fig
    axis_names = sorted(
        (name for name in fig.layout if name.startswith("xaxis")),
        key=lambda name: int(name[5:] or 1),
    )
    buttons = []
    for years in timeframes:
        start = end_date - pd.DateOffset(years=years)
        relayout = {}
        for axis in axis_names:
            relayout[f"{axis}.range"] = [start, end_date]
            relayout[f"{axis}.autorange"] = False
        buttons.append(dict(label=f"{years}Y", method="relayout", args=[relayout]))
    active = timeframes.index(default_years) if default_years in timeframes else 0
    fig.update_layout(updatemenus=[dict(
        type="dropdown",
        direction="down",
        active=active,
        buttons=buttons,
        x=1,
        xanchor="right",
        y=1.14,
        yanchor="top",
    )])
    fig.update_xaxes(range=[end_date - pd.DateOffset(years=default_years), end_date])
    return fig


def build_country_macro_figures(config, series_by_key, cycle_series=None):
    """Build one vertically stacked figure for each configured macro section."""
    sections = OrderedDict()
    for indicator in config.indicators:
        series = series_by_key.get(indicator.key)
        if series is not None and not series.dropna().empty:
            series = series.dropna()
            cutoff = series.index.max() - pd.DateOffset(years=max(DEFAULT_TIMEFRAMES))
            sections.setdefault(indicator.section, []).append(
                (indicator, series.loc[series.index >= cutoff])
            )

    figures = OrderedDict()
    for section, entries in sections.items():
        fig = make_subplots(
            rows=len(entries),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=min(0.08, 0.20 / max(len(entries), 1)),
            subplot_titles=[indicator.label for indicator, _ in entries],
        )
        for row, (indicator, series) in enumerate(entries, start=1):
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series,
                name=indicator.label,
                mode="lines",
                showlegend=False,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>%{y:,.2f}"
                    f" {indicator.unit}<extra>{indicator.label}</extra>"
                ),
            ), row=row, col=1)
            if indicator.zero_line:
                fig.add_hline(y=0, line_color="gray", line_width=1, row=row, col=1)
            fig.update_yaxes(title_text=indicator.unit, row=row, col=1)
        fig.update_layout(
            title=f"{config.name}: {section}",
            template="plotly_dark",
            height=max(420, 285 * len(entries)),
            hovermode="x unified",
            margin=dict(l=95, r=35, t=105, b=50),
        )
        fig.update_xaxes(title_text="Date", row=len(entries), col=1)
        add_cycle_bands(fig, cycle_series)
        add_timeframe_dropdown(fig, default_years=config.default_years)
        figures[section] = fig
    return figures


__all__ = [
    "add_cycle_bands",
    "add_timeframe_dropdown",
    "build_country_macro_figures",
]

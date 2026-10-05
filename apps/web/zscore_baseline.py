"""Normalization and display rules shared by the momentum dashboard callbacks."""

from contextvars import ContextVar
from collections import defaultdict

import numpy as np
import pandas as pd


BASELINE = ContextVar("momentum_zscore_baseline", default=("full", 252, "3y"))


def baseline_key():
    mode, days, lookback = BASELINE.get()
    return mode, days if mode == "trailing" else None, lookback if mode == "visible" else None


def normalize(values, offsets, *, playback=False):
    """Use sample standard deviation; playback baselines end at each observation.

    Trailing N means N trading observations, including the current observation.
    Visible lookback uses the dashboard's calendar interval. Static charts use
    a fixed baseline ending at the last observation; playback uses past data only.
    Undefined scores (insufficient history or constant data) remain missing.
    """
    values = values.replace([np.inf, -np.inf], np.nan).sort_index()
    mode, days, lookback = BASELINE.get()
    offset = offsets.get(lookback) if mode == "visible" else None
    if playback:
        if mode == "trailing":
            reference = values.rolling(days, min_periods=2)
        elif offset is not None:
            # Calendar offsets (including leap years) must match the Lookback UI.
            from pandas.api.indexers import VariableOffsetWindowIndexer
            indexer = VariableOffsetWindowIndexer(index=values.index, offset=offset)
            reference = values.rolling(indexer, min_periods=2, closed="both")
        else:
            reference = values.expanding(min_periods=2)
        mean, std = reference.mean(), reference.std()
    else:
        reference = values
        if mode == "trailing":
            reference = values.iloc[-days:]
        elif offset is not None and not values.empty:
            reference = values.loc[values.index >= values.index[-1] - offset]
        mean, std = reference.mean(), reference.std()
    if np.isscalar(std):
        std = np.nan if std == 0 else std
    else:
        std = std.replace(0.0, np.nan)
    return (values - mean) / std


def finish_zscore_figure(figure):
    """Label metric levels descriptively and fit z-score axes to visible data."""
    labels = {"Accumulate": "Low", "Liquidate": "High", "Neutral": "Typical"}
    for annotation in figure.layout.annotations or ():
        if annotation.text in labels:
            # Spread panels historically reversed the trading labels. Its sign,
            # not the old trading instruction, determines Low versus High.
            label = labels[annotation.text]
            if label != "Typical" and isinstance(annotation.y, (float, int)):
                label = "Low" if annotation.y < 0 else "High"
            annotation.text = label
        elif annotation.text and ("Bullish Neutral" in annotation.text or "Bearish Neutral" in annotation.text):
            annotation.visible = False

    traces_by_axis = defaultdict(list)
    for trace in figure.data:
        if trace.visible not in (False, "legendonly"):
            traces_by_axis[getattr(trace, "yaxis", None) or "y"].append(trace)

    shapes_by_axis = defaultdict(list)
    for shape in figure.layout.shapes or ():
        if shape.visible is not False:
            shapes_by_axis[shape.yref].append(shape)

    for axis_name in figure.layout:
        if not axis_name.startswith("yaxis"):
            continue
        axis = figure.layout[axis_name]
        title = (axis.title.text or "").lower()
        if "z-score" not in title and "z score" not in title:
            continue
        if "derivative" in title:
            continue
        axis_ref = "y" + axis_name[5:]
        low, high = -2.0, 2.0
        for shape in shapes_by_axis[axis_ref]:
            for value in (shape.y0, shape.y1):
                if isinstance(value, (float, int)) and np.isfinite(value):
                    low, high = min(low, value), max(high, value)
        for trace in traces_by_axis[axis_ref]:
            y = getattr(trace, "y", None)
            x = getattr(trace, "x", None)
            if y is None or x is None or len(x) == 0 or len(y) != len(x):
                continue
            numeric = pd.to_numeric(pd.Series(y), errors="coerce")
            xaxis_name = "xaxis" + (getattr(trace, "xaxis", None) or "x")[1:]
            x_range = figure.layout[xaxis_name].range
            if x_range is not None:
                xs = pd.Series(x)
                if pd.api.types.is_numeric_dtype(xs.dtype):
                    try:
                        bounds = float(x_range[0]), float(x_range[1])
                    except (TypeError, ValueError):
                        # Plotly also accepts epoch milliseconds on date axes.
                        # Date bounds must never be coerced with float().
                        xs = pd.to_datetime(xs, unit="ms", errors="coerce", utc=True)
                        bounds = pd.to_datetime(x_range, utc=True)
                    mask = xs.between(*bounds)
                else:
                    xs = pd.to_datetime(xs, errors="coerce", utc=True)
                    mask = xs.between(pd.to_datetime(x_range[0], utc=True), pd.to_datetime(x_range[1], utc=True))
                numeric = numeric[mask]
            finite = numeric[np.isfinite(numeric)]
            if not finite.empty:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
        padding = (high - low) * 0.08
        axis.update(range=[low - padding, high + padding], autorange=False)
    return figure

"""Trade-range probability cone view."""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import build_visibility_mask
from ._shared import coerce_positive_int, dropdown_menu, header_margin, header_title, preferred_numeric_window

def plot_trade_range_probability_cone(
    cone_context,
    ticker_label="Asset",
    template="plotly_dark",
):
    """
    Plot a close-based two-sided probability cone with long and short tail markers
    for same-day trade planning.
    """
    required_keys = {
        "session_date",
        "window",
        "effective_window",
        "anchor_price",
        "latest_price",
        "sample_returns",
        "interval_confidence_levels",
        "tail_confidence_levels",
        "intervals",
        "long_tail_levels",
        "short_tail_levels",
        "median_return",
        "median_price",
    }
    if not isinstance(cone_context, Mapping):
        raise TypeError("cone_context must be a mapping.")

    raw_contexts_by_window = cone_context.get("cone_contexts_by_window")
    window_options = cone_context.get("windows")
    default_window = coerce_positive_int(
        cone_context.get("default_window", cone_context.get("window"))
    )

    if isinstance(raw_contexts_by_window, Mapping) and raw_contexts_by_window:
        contexts_by_window = {}
        for key, context in raw_contexts_by_window.items():
            coerced_window = coerce_positive_int(key)
            if coerced_window is None:
                continue
            contexts_by_window[coerced_window] = context

        if window_options is None:
            window_options = list(contexts_by_window.keys())
        else:
            try:
                window_options = [int(window) for window in window_options]
            except Exception as exc:
                raise ValueError("cone_context windows must be iterable integers.") from exc
            window_options = [window for window in window_options if window in contexts_by_window]

        if not window_options:
            raise ValueError("cone_context does not contain any valid probability-cone windows.")
        if default_window not in window_options:
            default_window = preferred_numeric_window(window_options) or window_options[0]
    else:
        missing = [key for key in required_keys if key not in cone_context]
        if missing:
            raise ValueError(f"cone_context missing required keys: {missing}")
        default_window = int(cone_context["window"])
        window_options = [default_window]
        contexts_by_window = {default_window: cone_context}

    default_context = contexts_by_window[default_window]
    missing = [key for key in required_keys if key not in default_context]
    if missing:
        raise ValueError(f"cone_context missing required keys: {missing}")

    def _trade_range_labels(context):
        horizon_sessions = max(1, int(context.get("horizon_sessions", 1)))
        if horizon_sessions == 1:
            return {
                "panel_title": lambda confidence: (
                    f"Today Prior-Close-Anchored {confidence:.0%} Projected Close Range (Close-to-Close)"
                ),
                "distribution_title": "Trailing Close-to-Close Return Distribution",
                "header_title": "Two-Sided Close-to-Close Trade Range Cone From Prior Close",
                "annotation_basis": "completed close-to-close sessions",
                "path_ticktext": ["Prior Close", "Projected Close"],
                "path_axis_title": "Close Path",
                "return_axis_title": "Close-to-Close Return",
                "histogram_name": "Close-to-Close Returns",
                "range_name": "Projected Close Range",
                "lower_price_label": "Lower Close",
                "upper_price_label": "Upper Close",
                "median_name": "Median Close",
                "anchor_name": "Prior Close",
                "anchor_text_prefix": "Prior Close",
                "show_latest_price": True,
                "latest_name": "Latest Price So Far",
                "latest_text_prefix": "Last",
                "tail_basis": "Close-to-Close",
            }

        horizon_label = f"{horizon_sessions}-Session"
        horizon_lower = f"{horizon_sessions}-session"
        return {
            "panel_title": lambda confidence: (
                f"Reference-Close-Anchored {confidence:.0%} Projected Close Range ({horizon_label} Horizon)"
            ),
            "distribution_title": f"Trailing {horizon_label} Forward Return Distribution",
            "header_title": f"Two-Sided {horizon_label} Trade Range Cone From Reference Close",
            "annotation_basis": f"completed {horizon_lower} close-based forward returns",
            "path_ticktext": ["Reference Close", "Projected Close"],
            "path_axis_title": "Holding-Period Close Path",
            "return_axis_title": f"{horizon_label} Forward Return",
            "histogram_name": f"{horizon_label} Returns",
            "range_name": f"{horizon_label} Projected Exit Range",
            "lower_price_label": "Lower Exit Price",
            "upper_price_label": "Upper Exit Price",
            "median_name": "Median Exit Close",
            "anchor_name": "Reference Close",
            "anchor_text_prefix": "Reference Close",
            "show_latest_price": False,
            "latest_name": "Current Close",
            "latest_text_prefix": "Current",
            "tail_basis": horizon_label,
        }

    default_labels = _trade_range_labels(default_context)

    interval_levels = sorted(default_context["interval_confidence_levels"], reverse=True)
    tail_levels = sorted(default_context["tail_confidence_levels"])
    panel_confidence_levels = sorted(set(interval_levels).intersection(tail_levels))
    if not panel_confidence_levels:
        panel_confidence_levels = sorted(set(interval_levels).union(tail_levels))

    distribution_row = len(panel_confidence_levels) + 1
    distribution_height = 0.32
    panel_height = (1.0 - distribution_height) / len(panel_confidence_levels)
    row_heights = [panel_height] * len(panel_confidence_levels) + [distribution_height]
    subplot_titles = tuple(
        [
            default_labels["panel_title"](confidence)
            for confidence in panel_confidence_levels
        ]
        + [default_labels["distribution_title"]]
    )

    fig = make_subplots(
        rows=distribution_row,
        cols=1,
        vertical_spacing=0.08,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )
    subplot_title_annotations = [copy.deepcopy(annotation) for annotation in fig.layout.annotations]

    interval_fill_colors = [
        "rgba(59, 130, 246, 0.18)",
        "rgba(34, 197, 94, 0.24)",
        "rgba(245, 158, 11, 0.30)",
        "rgba(239, 68, 68, 0.34)",
    ]
    interval_line_colors = [
        "rgba(59, 130, 246, 0.72)",
        "rgba(34, 197, 94, 0.84)",
        "rgba(245, 158, 11, 0.88)",
        "rgba(239, 68, 68, 0.90)",
    ]

    global_price_min = None
    global_price_max = None
    for window in window_options:
        context = contexts_by_window[window]
        missing = [key for key in required_keys if key not in context]
        if missing:
            raise ValueError(f"cone_context for window {window} missing required keys: {missing}")

        window_interval_levels = sorted(context["interval_confidence_levels"], reverse=True)
        window_tail_levels = sorted(context["tail_confidence_levels"])
        window_panel_levels = sorted(set(window_interval_levels).intersection(window_tail_levels))
        if not window_panel_levels:
            window_panel_levels = sorted(set(window_interval_levels).union(window_tail_levels))
        if window_panel_levels != panel_confidence_levels:
            raise ValueError("All cone windows must share the same confidence-level layout.")

        interval_map = context["intervals"]
        long_tail_map = context["long_tail_levels"]
        short_tail_map = context["short_tail_levels"]
        anchor_price = float(context["anchor_price"])
        latest_price = float(context["latest_price"])
        median_price = float(context["median_price"])

        candidate_prices = [anchor_price, latest_price, median_price]
        for confidence in window_interval_levels:
            band = interval_map[confidence]
            candidate_prices.extend([band["lower_price"], band["upper_price"]])
        for confidence in window_tail_levels:
            long_tail = long_tail_map[confidence]
            short_tail = short_tail_map[confidence]
            candidate_prices.extend(
                [
                    long_tail["var_price"],
                    long_tail["expected_shortfall_price"],
                    short_tail["var_price"],
                    short_tail["expected_shortfall_price"],
                ]
            )

        valid_prices = [float(price) for price in candidate_prices if np.isfinite(price)]
        if not valid_prices:
            raise ValueError(
                f"cone_context for window {window} does not contain any finite price levels to plot."
            )
        window_min = min(valid_prices)
        window_max = max(valid_prices)
        global_price_min = window_min if global_price_min is None else min(global_price_min, window_min)
        global_price_max = window_max if global_price_max is None else max(global_price_max, window_max)

    if global_price_min is None or global_price_max is None:
        raise ValueError("cone_context does not contain any finite price levels to plot.")

    default_anchor_price = float(default_context["anchor_price"])
    price_padding = max((global_price_max - global_price_min) * 0.10, default_anchor_price * 0.01)
    price_axis_min = global_price_min - price_padding
    price_axis_max = global_price_max + price_padding

    def _axis_ref(row_idx, axis_name):
        return axis_name if row_idx == 1 else f"{axis_name}{row_idx}"

    def _window_title(context):
        requested_window = int(context["window"])
        title_date = pd.Timestamp(context["session_date"]).strftime("%Y-%m-%d")
        labels = _trade_range_labels(context)
        return header_title(
            f"{ticker_label} {labels['header_title']} "
            f"({requested_window}-Session Lookback, {title_date})"
        )

    def _window_annotation(context):
        labels = _trade_range_labels(context)
        return dict(
            x=0.0,
            y=1.12,
            xref="paper",
            yref="paper",
            text=(
                f"Using the last {int(context['effective_window'])} {labels['annotation_basis']}. "
                f"Entry anchor: {float(context['anchor_price']):,.2f}. "
                "Red markers show long-risk floors; purple markers show short-risk ceilings."
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=12, color="rgba(226, 232, 240, 0.92)"),
        )

    def _window_shapes(context):
        shapes = []
        anchor_price = float(context["anchor_price"])
        median_return = float(context["median_return"])
        long_tail_map = context["long_tail_levels"]
        short_tail_map = context["short_tail_levels"]

        for row_idx in range(1, distribution_row):
            xref = _axis_ref(row_idx, "x")
            yref = _axis_ref(row_idx, "y")
            shapes.append(
                dict(
                    type="line",
                    xref=xref,
                    yref=f"{yref} domain",
                    x0=1.0,
                    x1=1.0,
                    y0=0.0,
                    y1=1.0,
                    line=dict(dash="dot", color="rgba(248, 250, 252, 0.40)", width=1),
                )
            )
            shapes.append(
                dict(
                    type="line",
                    xref=xref,
                    yref=yref,
                    x0=-0.10,
                    x1=1.20,
                    y0=anchor_price,
                    y1=anchor_price,
                    line=dict(dash="dot", color="rgba(34, 197, 94, 0.45)", width=1),
                )
            )

        xref = _axis_ref(distribution_row, "x")
        yref = _axis_ref(distribution_row, "y")
        shapes.append(
            dict(
                type="line",
                xref=xref,
                yref=f"{yref} domain",
                x0=median_return * 100.0,
                x1=median_return * 100.0,
                y0=0.0,
                y1=1.0,
                line=dict(dash="dash", color="#f8fafc", width=2),
            )
        )

        for idx, confidence in enumerate(tail_levels):
            long_tail = long_tail_map[confidence]
            short_tail = short_tail_map[confidence]
            long_color = "#ef4444" if idx == 0 else "#991b1b"
            short_color = "#a855f7" if idx == 0 else "#6d28d9"

            for value, color, dash in (
                (long_tail["var_return"] * 100.0, long_color, "dash"),
                (long_tail["expected_shortfall_return"] * 100.0, long_color, "dot"),
                (short_tail["var_return"] * 100.0, short_color, "dash"),
                (short_tail["expected_shortfall_return"] * 100.0, short_color, "dot"),
            ):
                shapes.append(
                    dict(
                        type="line",
                        xref=xref,
                        yref=f"{yref} domain",
                        x0=value,
                        x1=value,
                        y0=0.0,
                        y1=1.0,
                        line=dict(dash=dash, color=color, width=2),
                    )
                )

        return shapes

    window_annotations = {}
    window_shapes = {}
    traces_per_window = None

    for window in window_options:
        visible = window == default_window
        trace_start = len(fig.data)
        context = contexts_by_window[window]
        labels = _trade_range_labels(context)
        interval_map = context["intervals"]
        long_tail_map = context["long_tail_levels"]
        short_tail_map = context["short_tail_levels"]
        sample_returns = pd.Series(context["sample_returns"]).dropna()

        if sample_returns.empty:
            raise ValueError(
                f"cone_context sample_returns for window {window} must contain at least one return observation."
            )

        anchor_price = float(context["anchor_price"])
        latest_price = float(context["latest_price"])
        median_price = float(context["median_price"])
        effective_window = int(context["effective_window"])

        def add_projected_close_row(target_row, confidence, *, show_shared_legend):
            band = interval_map.get(confidence)
            style_index = panel_confidence_levels.index(confidence)
            fill_color = interval_fill_colors[style_index % len(interval_fill_colors)]
            line_color = interval_line_colors[style_index % len(interval_line_colors)]

            if band is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[0.0, 1.0, 1.0, 0.0, 0.0],
                        y=[anchor_price, band["upper_price"], band["lower_price"], anchor_price, anchor_price],
                        mode="lines",
                        line=dict(color=line_color, width=1.5),
                        fill="toself",
                        fillcolor=fill_color,
                        name=f"{confidence:.0%} {labels['range_name']}",
                        hovertemplate=(
                            f"{confidence:.0%} {labels['range_name']}"
                            f"<br>{labels['lower_price_label']}: %{{customdata[0]:,.2f}}"
                            f"<br>{labels['upper_price_label']}: %{{customdata[1]:,.2f}}"
                            "<br>Lower Return: %{customdata[2]:.4%}"
                            "<br>Upper Return: %{customdata[3]:.4%}"
                            "<extra></extra>"
                        ),
                        customdata=[
                            [
                                band["lower_price"],
                                band["upper_price"],
                                band["lower_return"],
                                band["upper_return"],
                            ]
                        ] * 5,
                        showlegend=True,
                        legendgroup=f"range-{confidence:.0%}",
                        visible=visible,
                    ),
                    row=target_row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1.0, 1.0],
                        y=[band["lower_price"], band["upper_price"]],
                        mode="markers",
                        name=f"{confidence:.0%} Range Edges",
                        marker=dict(color=line_color, size=8, symbol="diamond"),
                        hovertemplate=(
                            f"{confidence:.0%} Range Edge"
                            "<br>Close Price: %{y:,.2f}"
                            "<br>Return Threshold: %{customdata[0]:.4%}"
                            "<extra></extra>"
                        ),
                        customdata=[[band["lower_return"]], [band["upper_return"]]],
                        showlegend=False,
                        legendgroup=f"range-{confidence:.0%}",
                        visible=visible,
                    ),
                    row=target_row,
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[anchor_price, median_price],
                    mode="lines+markers",
                    name=labels["median_name"],
                    line=dict(color="#f8fafc", width=2, dash="dash"),
                    marker=dict(size=8, color="#f8fafc"),
                    hovertemplate=f"{labels['median_name']}: %{{y:,.2f}}<extra></extra>",
                    showlegend=show_shared_legend,
                    legendgroup="median-close",
                    visible=visible,
                ),
                row=target_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[0.0],
                    y=[anchor_price],
                    mode="markers+text",
                    name=labels["anchor_name"],
                    marker=dict(size=10, color="#22c55e", symbol="diamond"),
                    text=[f"{labels['anchor_text_prefix']} {anchor_price:,.2f}"],
                    textposition="top left",
                    hovertemplate=f"{labels['anchor_name']}: %{{y:,.2f}}<extra></extra>",
                    showlegend=show_shared_legend,
                    legendgroup="session-open",
                    visible=visible,
                ),
                row=target_row,
                col=1,
            )
            if labels["show_latest_price"]:
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[latest_price],
                        mode="markers+text",
                        name=labels["latest_name"],
                        marker=dict(size=9, color="#38bdf8", symbol="circle"),
                        text=[f"{labels['latest_text_prefix']} {latest_price:,.2f}"],
                        textposition="middle right",
                        hovertemplate=f"{labels['latest_name']}: %{{y:,.2f}}<extra></extra>",
                        showlegend=show_shared_legend,
                        legendgroup="latest-session-price",
                        visible=visible,
                    ),
                    row=target_row,
                    col=1,
                )

        for row_idx, confidence in enumerate(panel_confidence_levels, start=1):
            add_projected_close_row(row_idx, confidence, show_shared_legend=row_idx == 1)

            long_tail = long_tail_map.get(confidence)
            short_tail = short_tail_map.get(confidence)
            style_index = panel_confidence_levels.index(confidence)
            long_color = "#ef4444" if style_index == 0 else "#991b1b"
            short_color = "#a855f7" if style_index == 0 else "#6d28d9"

            if long_tail is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[long_tail["var_price"]],
                        mode="markers+text",
                        name=f"{confidence:.0%} {labels['tail_basis']} Long VaR Floor",
                        marker=dict(size=11, color=long_color, symbol="triangle-down"),
                        text=[f"{confidence:.0%} Long VaR Floor {long_tail['var_price']:,.2f}"],
                        textposition="middle right",
                        hovertemplate=(
                            f"{confidence:.0%} {labels['tail_basis']} Long VaR Floor"
                            "<br>Close Price: %{y:,.2f}"
                            f"<br>Return Threshold: {long_tail['var_return']:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=row_idx,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[long_tail["expected_shortfall_price"]],
                        mode="markers+text",
                        name=f"{confidence:.0%} {labels['tail_basis']} Long CVaR Floor",
                        marker=dict(size=10, color=long_color, symbol="x"),
                        text=[f"{confidence:.0%} Long CVaR Floor {long_tail['expected_shortfall_price']:,.2f}"],
                        textposition="middle right",
                        hovertemplate=(
                            f"{confidence:.0%} {labels['tail_basis']} Long CVaR Floor"
                            "<br>Close Price: %{y:,.2f}"
                            f"<br>Expected Shortfall: {long_tail['expected_shortfall_return']:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=row_idx,
                    col=1,
                )
            if short_tail is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[short_tail["var_price"]],
                        mode="markers+text",
                        name=f"{confidence:.0%} {labels['tail_basis']} Short VaR Ceiling",
                        marker=dict(size=11, color=short_color, symbol="triangle-up"),
                        text=[f"{confidence:.0%} Short VaR Ceiling {short_tail['var_price']:,.2f}"],
                        textposition="middle right",
                        hovertemplate=(
                            f"{confidence:.0%} {labels['tail_basis']} Short VaR Ceiling"
                            "<br>Close Price: %{y:,.2f}"
                            f"<br>Return Threshold: {short_tail['var_return']:.4%}"
                            "<extra></extra>"
                        ),
                        visible=visible,
                    ),
                    row=row_idx,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[1.0],
                        y=[short_tail["expected_shortfall_price"]],
                        mode="markers+text",
                        name=f"{confidence:.0%} {labels['tail_basis']} Short CVaR Ceiling",
                        marker=dict(size=10, color=short_color, symbol="x"),
                        text=[f"{confidence:.0%} Short CVaR Ceiling {short_tail['expected_shortfall_price']:,.2f}"],
                        textposition="middle right",
                        hovertemplate=(
                            f"{confidence:.0%} {labels['tail_basis']} Short CVaR Ceiling"
                            "<br>Close Price: %{y:,.2f}"
                            f"<br>Expected Shortfall: {short_tail['expected_shortfall_return']:.4%}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        visible=visible,
                    ),
                    row=row_idx,
                    col=1,
                )

        histogram_values = sample_returns.mul(100.0)
        fig.add_trace(
            go.Histogram(
                x=histogram_values,
                nbinsx=min(max(effective_window // 8, 20), 60),
                name=labels["histogram_name"],
                marker_color="rgba(59, 130, 246, 0.65)",
                opacity=0.85,
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
                showlegend=False,
                visible=visible,
            ),
            row=distribution_row,
            col=1,
        )

        added_traces = len(fig.data) - trace_start
        if traces_per_window is None:
            traces_per_window = added_traces

        window_shapes[window] = _window_shapes(context)
        window_annotations[window] = subplot_title_annotations + [_window_annotation(context)]

    if traces_per_window is None or traces_per_window <= 0:
        raise ValueError("Unable to build cone traces from the supplied probability payload.")

    for row_idx in range(1, distribution_row):
        fig.update_xaxes(
            row=row_idx,
            col=1,
            range=[-0.10, 1.20],
            tickmode="array",
            tickvals=[0.0, 1.0],
            ticktext=default_labels["path_ticktext"],
            title_text=default_labels["path_axis_title"],
        )
        fig.update_yaxes(
            row=row_idx,
            col=1,
            title_text="Price",
            range=[price_axis_min, price_axis_max],
            tickprefix="$",
        )

    fig.update_xaxes(
        row=distribution_row,
        col=1,
        title_text=default_labels["return_axis_title"],
        ticksuffix="%",
    )
    fig.update_yaxes(row=distribution_row, col=1, title_text="Count")

    updatemenus = []
    if len(window_options) > 1:
        total_traces = len(fig.data)
        buttons = []
        for idx, window in enumerate(window_options):
            visibility = build_visibility_mask(
                total_traces=total_traces,
                active_window_index=idx,
                traces_per_window=traces_per_window,
                constant_trace_indices=[],
            )
            buttons.append(
                dict(
                    label=str(window),
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": _window_title(contexts_by_window[window]),
                            "shapes": window_shapes[window],
                            "annotations": window_annotations[window],
                        },
                    ],
                )
            )
        updatemenus = [
            dropdown_menu(
                buttons=buttons,
                x=0.0,
                active=window_options.index(default_window),
            )
        ]

    fig.update_layout(
        updatemenus=updatemenus,
        title=_window_title(default_context),
        annotations=window_annotations[default_window],
        shapes=window_shapes[default_window],
        height=1450,
        margin=header_margin(),
        template=template,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.08,
    )
    return fig


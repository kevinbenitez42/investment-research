"""Trade-range history profile view."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import build_visibility_mask
from ._shared import coerce_positive_int, dropdown_menu, header_margin, header_title, preferred_numeric_window

def plot_trade_range_history_profile(
    history_context,
    ticker_label="Asset",
    template="plotly_dark",
):
    """
    Plot an ex-ante historical long/short trade-range profile using close-to-close
    returns and prior-session information only.
    """
    required_keys = {
        "interval_confidence_levels",
        "tail_confidence_levels",
        "session_returns",
        "session_open",
        "session_close",
    }
    if not isinstance(history_context, Mapping):
        raise TypeError("history_context must be a mapping.")
    missing = [key for key in required_keys if key not in history_context]
    if missing:
        raise ValueError(f"history_context missing required keys: {missing}")

    metrics_by_window = history_context.get("metrics_by_window")
    window_options = history_context.get("windows")
    default_window = history_context.get("default_window", history_context.get("window"))

    if isinstance(metrics_by_window, Mapping) and metrics_by_window:
        if window_options is None:
            window_options = list(metrics_by_window.keys())
        else:
            try:
                window_options = [int(window) for window in window_options]
            except Exception as exc:
                raise ValueError("history_context windows must be iterable integers.") from exc
            window_options = [window for window in window_options if window in metrics_by_window]

        if not window_options:
            raise ValueError("history_context does not contain any valid rolling windows.")
        if default_window not in window_options:
            default_window = preferred_numeric_window(window_options) or window_options[0]
    else:
        metrics_by_confidence = history_context.get("metrics_by_confidence")
        if not isinstance(metrics_by_confidence, Mapping) or not metrics_by_confidence:
            raise ValueError(
                "history_context metrics_by_confidence must be a non-empty mapping."
            )
        default_window = int(history_context["window"])
        window_options = [default_window]
        metrics_by_window = {default_window: metrics_by_confidence}

    interval_levels = sorted(history_context["interval_confidence_levels"], reverse=True)
    tail_levels = sorted(history_context["tail_confidence_levels"])
    session_returns = pd.Series(history_context["session_returns"]).dropna()
    session_close = pd.Series(history_context["session_close"]).dropna()
    if session_returns.empty or session_close.empty:
        raise ValueError("history_context does not contain enough session data to plot.")

    horizon_sessions = max(1, int(history_context.get("horizon_sessions", 1)))
    if horizon_sessions == 1:
        history_labels = {
            "return_name": "Close-to-Close Return",
            "returns_panel": "Close-to-Close Returns with Two-Sided Tail Thresholds",
            "forecast_panel": "Rolling Close-to-Close Tail Forecasts: Long Floors and Short Ceilings",
            "breach_panel": "Rolling Close-to-Close Breach Rates vs Expected",
            "header_basis": "Historical Two-Sided Close-to-Close Trade Range Profile",
        }
    else:
        horizon_label = f"{horizon_sessions}-Session Forward"
        history_labels = {
            "return_name": f"{horizon_label} Return",
            "returns_panel": f"{horizon_label} Returns with Two-Sided Tail Thresholds",
            "forecast_panel": f"Rolling {horizon_label} Tail Forecasts: Long Floors and Short Ceilings",
            "breach_panel": f"Rolling {horizon_label} Breach Rates vs Expected",
            "header_basis": f"Historical Two-Sided {horizon_label} Trade Range Profile",
        }

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.33, 0.27, 0.40],
        subplot_titles=(
            history_labels["returns_panel"],
            history_labels["forecast_panel"],
            history_labels["breach_panel"],
        ),
    )

    interval_fill_colors = {
        0.95: "rgba(34, 197, 94, 0.18)",
        0.99: "rgba(59, 130, 246, 0.15)",
    }
    interval_line_colors = {
        0.95: "rgba(34, 197, 94, 0.82)",
        0.99: "rgba(59, 130, 246, 0.82)",
    }
    long_colors = {
        0.95: "#ef4444",
        0.99: "#991b1b",
    }
    short_colors = {
        0.95: "#a855f7",
        0.99: "#6d28d9",
    }

    index_candidates = [session_returns.index]
    traces_per_window = None
    return_colors = np.where(
        session_returns >= 0,
        "rgba(34, 197, 94, 0.45)",
        "rgba(239, 68, 68, 0.45)",
    ).tolist()

    for window in window_options:
        visible = window == default_window
        trace_start = len(fig.data)
        metrics_by_confidence = metrics_by_window.get(window, {})

        fig.add_trace(
            go.Bar(
                x=session_returns.index,
                y=session_returns,
                name=history_labels["return_name"],
                marker_color=return_colors,
                opacity=0.75,
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}"
                    f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                    "<extra></extra>"
                ),
                showlegend=False,
                visible=visible,
            ),
            row=1,
            col=1,
        )

        for confidence in tail_levels:
            metric_set = metrics_by_confidence.get(confidence, {})
            lower_var_return = metric_set.get("lower_var_return", pd.Series(dtype=float)).dropna()
            upper_var_return = metric_set.get("upper_var_return", pd.Series(dtype=float)).dropna()
            lower_es_return = metric_set.get(
                "lower_expected_shortfall_return",
                pd.Series(dtype=float),
            ).dropna()
            upper_es_return = metric_set.get(
                "upper_expected_shortfall_return",
                pd.Series(dtype=float),
            ).dropna()
            lower_breaches = metric_set.get("lower_breaches", pd.Series(dtype=float)).dropna()
            upper_breaches = metric_set.get("upper_breaches", pd.Series(dtype=float)).dropna()

            lower_breach_index = (
                lower_breaches.index[lower_breaches.astype(bool)]
                if not lower_breaches.empty
                else pd.Index([])
            )
            upper_breach_index = (
                upper_breaches.index[upper_breaches.astype(bool)]
                if not upper_breaches.empty
                else pd.Index([])
            )
            lower_breach_returns = session_returns.reindex(lower_breach_index).dropna()
            upper_breach_returns = session_returns.reindex(upper_breach_index).dropna()

            fig.add_trace(
                go.Scatter(
                    x=lower_var_return.index,
                    y=lower_var_return,
                    mode="lines",
                    name=f"{confidence:.0%} Long VaR Return",
                    line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Long VaR Return"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Return Threshold: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_var_return.index,
                    y=upper_var_return,
                    mode="lines",
                    name=f"{confidence:.0%} Short VaR Return",
                    line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Short VaR Return"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Return Threshold: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=lower_breach_returns.index,
                    y=lower_breach_returns,
                    mode="markers",
                    name=f"{confidence:.0%} Long Breaches",
                    marker=dict(
                        color=long_colors.get(confidence, "#ef4444"),
                        size=7,
                        symbol="x",
                    ),
                    hovertemplate=(
                        f"{confidence:.0%} Long Breach"
                        "<br>Date: %{x|%Y-%m-%d}"
                        f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_breach_returns.index,
                    y=upper_breach_returns,
                    mode="markers",
                    name=f"{confidence:.0%} Short Breaches",
                    marker=dict(
                        color=short_colors.get(confidence, "#a855f7"),
                        size=7,
                        symbol="x",
                    ),
                    hovertemplate=(
                        f"{confidence:.0%} Short Breach"
                        "<br>Date: %{x|%Y-%m-%d}"
                        f"<br>{history_labels['return_name']}: %{{y:.4%}}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=lower_var_return.index,
                    y=lower_var_return,
                    mode="lines",
                    name=f"{confidence:.0%} Long VaR Floor",
                    line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Long VaR Floor"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Forecast Threshold: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=lower_es_return.index,
                    y=lower_es_return,
                    mode="lines",
                    name=f"{confidence:.0%} Long CVaR Floor",
                    line=dict(
                        color=long_colors.get(confidence, "#ef4444"),
                        width=2,
                        dash="dot",
                    ),
                    hovertemplate=(
                        f"{confidence:.0%} Long CVaR Floor"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Expected Shortfall: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_var_return.index,
                    y=upper_var_return,
                    mode="lines",
                    name=f"{confidence:.0%} Short VaR Ceiling",
                    line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Short VaR Ceiling"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Forecast Threshold: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_es_return.index,
                    y=upper_es_return,
                    mode="lines",
                    name=f"{confidence:.0%} Short CVaR Ceiling",
                    line=dict(
                        color=short_colors.get(confidence, "#a855f7"),
                        width=2,
                        dash="dot",
                    ),
                    hovertemplate=(
                        f"{confidence:.0%} Short CVaR Ceiling"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Expected Shortfall: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                ),
                row=2,
                col=1,
            )

            for series in (
                lower_var_return,
                upper_var_return,
                lower_es_return,
                upper_es_return,
                lower_breach_returns,
                upper_breach_returns,
            ):
                if not series.empty:
                    index_candidates.append(series.index)

        for confidence in tail_levels:
            metric_set = metrics_by_confidence.get(confidence, {})
            lower_breach_rate = metric_set.get(
                "lower_rolling_breach_rate",
                pd.Series(dtype=float),
            ).dropna()
            upper_breach_rate = metric_set.get(
                "upper_rolling_breach_rate",
                pd.Series(dtype=float),
            ).dropna()
            either_side_breach_rate = metric_set.get(
                "either_side_rolling_breach_rate",
                pd.Series(dtype=float),
            ).dropna()

            fig.add_trace(
                go.Scatter(
                    x=lower_breach_rate.index,
                    y=lower_breach_rate,
                    mode="lines",
                    name=f"{confidence:.0%} Long Breach Rate",
                    line=dict(color=long_colors.get(confidence, "#ef4444"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Long Breach Rate"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Breach Rate: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    visible=visible,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_breach_rate.index,
                    y=upper_breach_rate,
                    mode="lines",
                    name=f"{confidence:.0%} Short Breach Rate",
                    line=dict(color=short_colors.get(confidence, "#a855f7"), width=2),
                    hovertemplate=(
                        f"{confidence:.0%} Short Breach Rate"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Breach Rate: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    visible=visible,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=either_side_breach_rate.index,
                    y=either_side_breach_rate,
                    mode="lines",
                    name=f"{confidence:.0%} Either-Side Breach Rate",
                    line=dict(
                        color="#f59e0b" if confidence == 0.95 else "#b45309",
                        width=2,
                    ),
                    hovertemplate=(
                        f"{confidence:.0%} Either-Side Breach Rate"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Breach Rate: %{y:.4%}"
                        "<extra></extra>"
                    ),
                    visible=visible,
                ),
                row=3,
                col=1,
            )

            for series in (lower_breach_rate, upper_breach_rate, either_side_breach_rate):
                if not series.empty:
                    index_candidates.append(series.index)

        added_traces = len(fig.data) - trace_start
        if traces_per_window is None:
            traces_per_window = added_traces

    for confidence in tail_levels:
        expected_tail_rate = 1.0 - confidence
        reference_color = "#94a3b8" if confidence == 0.95 else "#64748b"
        fig.add_hline(
            y=expected_tail_rate,
            row=3,
            col=1,
            line_dash="dot",
            line_color=reference_color,
            line_width=1.5,
        )
        fig.add_annotation(
            x=0.99,
            y=expected_tail_rate,
            xref="x3 domain",
            yref="y3",
            text=f"{confidence:.0%} Expected Long / Short {expected_tail_rate:.2%}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color=reference_color),
        )
        either_side_expected_rate = min(1.0, 2.0 * expected_tail_rate)
        fig.add_hline(
            y=either_side_expected_rate,
            row=3,
            col=1,
            line_dash="dashdot",
            line_color=reference_color,
            line_width=1.5,
        )
        fig.add_annotation(
            x=0.99,
            y=either_side_expected_rate,
            xref="x3 domain",
            yref="y3",
            text=f"{confidence:.0%} Expected Either-Side {either_side_expected_rate:.2%}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color=reference_color),
        )

    fig.add_hline(
        y=0,
        row=1,
        col=1,
        line_dash="dash",
        line_color="rgba(248, 250, 252, 0.40)",
        line_width=1,
    )
    fig.add_hline(
        y=0,
        row=2,
        col=1,
        line_dash="dash",
        line_color="rgba(248, 250, 252, 0.40)",
        line_width=1,
    )
    fig.update_yaxes(title_text=history_labels["return_name"], tickformat=".2%", row=1, col=1)
    fig.update_yaxes(title_text="Forecast Return Threshold", tickformat=".2%", row=2, col=1)
    fig.update_yaxes(title_text="Breach Rate", tickformat=".2%", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    if traces_per_window is None or traces_per_window <= 0:
        raise ValueError("Unable to build trade-range traces from the supplied history payload.")

    non_empty_indices = [index for index in index_candidates if len(index) > 0]
    if non_empty_indices:
        global_start = min(index[0] for index in non_empty_indices)
        global_end = max(index[-1] for index in non_empty_indices)
        default_start = max(global_start, global_end - pd.DateOffset(years=3))
        fig.update_xaxes(range=[default_start, global_end])

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
                        "title": header_title(
                            f"{ticker_label} {history_labels['header_basis']} ({window}-Session Lookback)"
                        )
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dropdown_menu(
                buttons=buttons,
                x=0.0,
                active=window_options.index(default_window),
            )
        ],
        title=header_title(
            f"{ticker_label} {history_labels['header_basis']} ({default_window}-Session Lookback)"
        ),
        height=1325,
        margin=header_margin(top=170),
        template=template,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


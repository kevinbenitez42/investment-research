"""Volatility-model comparison views composed from line traces."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from Quantapp.visualization.figure_helpers import build_time_range_buttons
from Quantapp.visualization.traces.line import build_line_trace

HEADER_TOP_MARGIN = 150
HEADER_TITLE_Y = 0.97
HEADER_MENU_Y = 1.08


def _header_margin(top=None):
    return dict(t=HEADER_TOP_MARGIN if top is None else int(top))


def _header_title(text):
    return dict(
        text=str(text),
        x=0.5,
        xanchor="center",
        y=HEADER_TITLE_Y,
        yanchor="top",
    )


def _dropdown_menu(
    *,
    buttons,
    x,
    active=None,
    y=None,
    direction="down",
    showactive=True,
    xanchor="left",
    yanchor="top",
    **overrides,
):
    menu = dict(
        type="dropdown",
        buttons=buttons,
        direction=direction,
        showactive=showactive,
        x=x,
        xanchor=xanchor,
        y=HEADER_MENU_Y if y is None else y,
        yanchor=yanchor,
    )
    if active is not None:
        menu["active"] = active
    menu.update(overrides)
    return menu


def _smooth_annualized_model_vol(model_vol, window):
    return np.sqrt(pd.Series(model_vol).pow(2).rolling(int(window)).mean())


def _build_baseline_volatility_trace(baseline_vol, *, window, visible):
    return build_line_trace(
        x=baseline_vol.index,
        y=baseline_vol,
        mode="lines",
        name=f"Close-to-Close ({window}-day)",
        color="#1f77b4",
        width=2,
        visible=visible,
        showlegend=True,
        legendgroup="close-to-close-baseline",
    )


def _build_volatility_trace(
    series,
    *,
    name,
    color,
    dash=None,
    width=2,
    opacity=None,
    visible=True,
    showlegend=False,
    legendgroup=None,
):
    return build_line_trace(
        x=series.index,
        y=series,
        mode="lines",
        name=name,
        color=color,
        width=width,
        dash=dash,
        opacity=opacity,
        visible=visible,
        showlegend=showlegend,
        legendgroup=legendgroup,
    )


def _add_model_family_traces(
    fig,
    *,
    annualized_model_vols,
    volatility_model_specs,
    baseline_vol,
    window,
    visible,
):
    for model_name, _, color, dash in volatility_model_specs:
        model_vol = pd.Series(annualized_model_vols[model_name])
        smoothed_model_vol = _smooth_annualized_model_vol(model_vol, window)
        model_spread = (model_vol - baseline_vol).dropna()
        smoothed_model_spread = (smoothed_model_vol - baseline_vol).dropna()

        fig.add_trace(
            _build_volatility_trace(
                model_vol,
                name=f"Annualized {model_name}",
                color=color,
                dash=dash,
                visible=visible,
                showlegend=True,
                legendgroup=model_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            _build_volatility_trace(
                smoothed_model_vol,
                name=f"{model_name} Smoothed ({window}-day)",
                color=color,
                dash="longdash",
                width=3,
                opacity=0.9,
                visible=visible,
                showlegend=True,
                legendgroup=f"{model_name}-smoothed",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            _build_volatility_trace(
                model_spread,
                name=f"{model_name} Spread",
                color=color,
                dash=dash,
                visible=visible,
                legendgroup=f"{model_name}-spread",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            _build_volatility_trace(
                smoothed_model_spread,
                name=f"{model_name} Smoothed Spread",
                color=color,
                dash="longdash",
                width=3,
                opacity=0.9,
                visible=visible,
                legendgroup=f"{model_name}-smoothed-spread",
            ),
            row=3,
            col=1,
        )


def _add_rolling_realized_traces(
    fig,
    *,
    rolling_realized_vol_map,
    rolling_realized_vol_specs,
    baseline_vol,
    window,
    visible,
):
    for label, _, color, dash in rolling_realized_vol_specs:
        realized_vol = pd.Series(rolling_realized_vol_map[label])
        fig.add_trace(
            _build_volatility_trace(
                realized_vol,
                name=f"{label} ({window}-day)",
                color=color,
                dash=dash,
                visible=visible,
                legendgroup=f"rolling-realized-{label}",
            ),
            row=2,
            col=1,
        )

        if label != "Close-to-Close":
            realized_spread = (realized_vol - baseline_vol).dropna()
            fig.add_trace(
                _build_volatility_trace(
                    realized_spread,
                    name=f"{label} Spread",
                    color=color,
                    dash=dash,
                    visible=visible,
                    legendgroup=f"rolling-realized-spread-{label}",
                ),
                row=3,
                col=1,
            )


def _add_ewma_realized_traces(
    fig,
    *,
    ewma_realized_vol_map,
    ewma_realized_vol_specs,
    baseline_vol,
    window,
    visible,
):
    for label, _, color, dash in ewma_realized_vol_specs:
        ewma_vol = pd.Series(ewma_realized_vol_map[label])
        fig.add_trace(
            _build_volatility_trace(
                ewma_vol,
                name=f"{label} ({window}-day)",
                color=color,
                dash=dash,
                opacity=0.9,
                visible=visible,
                legendgroup=f"ewma-realized-{label}",
            ),
            row=2,
            col=1,
        )

        ewma_spread = (ewma_vol - baseline_vol).dropna()
        fig.add_trace(
            _build_volatility_trace(
                ewma_spread,
                name=f"{label} Spread",
                color=color,
                dash=dash,
                opacity=0.9,
                visible=visible,
                legendgroup=f"ewma-realized-spread-{label}",
            ),
            row=3,
            col=1,
        )


def _add_realized_minus_ewma_traces(
    fig,
    *,
    rolling_realized_vol_map,
    rolling_realized_vol_specs,
    ewma_realized_vol_map,
    ewma_realized_vol_specs,
    window,
    visible,
):
    for (rolling_label, _, color, _), (ewma_label, _, _, _) in zip(
        rolling_realized_vol_specs,
        ewma_realized_vol_specs,
    ):
        realized_minus_ewma = (
            pd.Series(rolling_realized_vol_map[rolling_label])
            - pd.Series(ewma_realized_vol_map[ewma_label])
        ).dropna()
        fig.add_trace(
            _build_volatility_trace(
                realized_minus_ewma,
                name=f"{rolling_label} Minus {ewma_label} ({window}-day)",
                color=color,
                width=3,
                opacity=0.9,
                visible=visible,
                legendgroup=f"realized-minus-ewma-{rolling_label}",
            ),
            row=4,
            col=1,
        )


def plot_volatility_model_comparison_view(
    annualized_model_vols,
    term_plot_data,
    *,
    volatility_model_specs,
    rolling_realized_vol_specs,
    ewma_realized_vol_specs,
    term_order,
    default_term,
    time_frame_map,
    ticker_label="Asset",
    template="plotly_dark",
):
    """Compose the volatility model comparison figure."""
    if not isinstance(annualized_model_vols, Mapping) or not annualized_model_vols:
        raise ValueError("annualized_model_vols must be a non-empty mapping.")
    if not isinstance(term_plot_data, Mapping) or not term_plot_data:
        raise ValueError("term_plot_data must be a non-empty mapping.")

    term_order = [term for term in term_order if term in term_plot_data]
    if not term_order:
        raise ValueError("No valid term order entries were provided for the volatility-model view.")
    if default_term not in term_order:
        default_term = term_order[0]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Annualized GARCH Family vs Close-to-Close Volatility (Raw + Smoothed)",
            "Annualized Rolling and EWMA Volatility Estimators",
            "Spread vs Close-to-Close Annualized Volatility",
            "Annualized Rolling Minus EWMA Volatility Estimators",
        ),
    )

    term_trace_bounds = {}
    term_ranges = {}
    for term in term_order:
        payload = term_plot_data[term]
        window = int(payload["window"])
        rolling_realized_vol_map = payload["rolling_realized_vol_map"]
        ewma_realized_vol_map = payload["ewma_realized_vol_map"]
        baseline_vol = pd.Series(rolling_realized_vol_map["Close-to-Close"])

        visible = term == default_term
        term_trace_start = len(fig.data)

        fig.add_trace(
            _build_baseline_volatility_trace(
                baseline_vol,
                window=window,
                visible=visible,
            ),
            row=1,
            col=1,
        )
        _add_model_family_traces(
            fig,
            annualized_model_vols=annualized_model_vols,
            volatility_model_specs=volatility_model_specs,
            baseline_vol=baseline_vol,
            window=window,
            visible=visible,
        )
        _add_rolling_realized_traces(
            fig,
            rolling_realized_vol_map=rolling_realized_vol_map,
            rolling_realized_vol_specs=rolling_realized_vol_specs,
            baseline_vol=baseline_vol,
            window=window,
            visible=visible,
        )
        _add_ewma_realized_traces(
            fig,
            ewma_realized_vol_map=ewma_realized_vol_map,
            ewma_realized_vol_specs=ewma_realized_vol_specs,
            baseline_vol=baseline_vol,
            window=window,
            visible=visible,
        )
        _add_realized_minus_ewma_traces(
            fig,
            rolling_realized_vol_map=rolling_realized_vol_map,
            rolling_realized_vol_specs=rolling_realized_vol_specs,
            ewma_realized_vol_map=ewma_realized_vol_map,
            ewma_realized_vol_specs=ewma_realized_vol_specs,
            window=window,
            visible=visible,
        )

        term_trace_bounds[term] = (term_trace_start, len(fig.data))
        term_ranges[term] = payload.get("term_range")

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.60)",
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.60)",
        row=4,
        col=1,
    )

    total_traces = len(fig.data)
    term_buttons = []
    for term in term_order:
        visibility = [False] * total_traces
        start, end = term_trace_bounds[term]
        for trace_index in range(start, end):
            visibility[trace_index] = True

        layout_updates = {
            "title": _header_title(
                f"{ticker_label} Volatility Models and Estimators ({time_frame_map[term]}-Day)"
            )
        }
        term_range = term_ranges.get(term)
        for axis_index in range(1, 5):
            axis_name = "xaxis" if axis_index == 1 else f"xaxis{axis_index}"
            if term_range is None:
                layout_updates[f"{axis_name}.autorange"] = True
            else:
                layout_updates[f"{axis_name}.range"] = term_range

        term_buttons.append(
            dict(
                label=f"{str(term).title()} ({time_frame_map[term]})",
                method="update",
                args=[{"visible": visibility}, layout_updates],
            )
        )

    available_ranges = [date_range for date_range in term_ranges.values() if date_range is not None]
    static_header_annotations = [
        dict(
            text="Volatility term",
            x=0.0,
            xref="paper",
            y=1.115,
            yref="paper",
            showarrow=False,
            xanchor="left",
        )
    ]
    updatemenus = [
        _dropdown_menu(
            buttons=term_buttons,
            x=0.0,
            active=term_order.index(default_term),
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor="rgba(148, 163, 184, 0.45)",
            font=dict(size=11),
        )
    ]

    if available_ranges:
        global_start = min(date_range[0] for date_range in available_ranges)
        global_end = max(date_range[1] for date_range in available_ranges)
        default_range = term_ranges[default_term] or [global_start, global_end]
        for row in range(1, 5):
            fig.update_xaxes(range=default_range, row=row, col=1)
        updatemenus.append(
            _dropdown_menu(
                buttons=build_time_range_buttons(global_start, global_end, axis_count=4),
                x=0.28,
                active=2,
                bgcolor="rgba(15, 23, 42, 0.95)",
                bordercolor="rgba(148, 163, 184, 0.45)",
                font=dict(size=11),
            )
        )
        static_header_annotations.append(
            dict(
                text="View timeframe",
                x=0.28,
                xref="paper",
                y=1.115,
                yref="paper",
                showarrow=False,
                xanchor="left",
            )
        )

    fig.update_yaxes(title_text="Annualized Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Spread vs Close-to-Close", row=3, col=1)
    fig.update_yaxes(title_text="Rolling Minus EWMA", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_layout(
        template=template,
        height=1450,
        margin=_header_margin(top=205),
        legend=dict(
            x=0.01,
            y=0.995,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(15, 23, 42, 0.35)",
        ),
        hovermode="x unified",
        title=_header_title(
            f"{ticker_label} Volatility Models and Estimators ({time_frame_map[default_term]}-Day)"
        ),
        updatemenus=updatemenus,
        annotations=list(fig.layout.annotations) + static_header_annotations,
    )
    return fig

"""Fixed-payout trade-range backtest view."""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from Quantapp.visualization.traces.line import build_line_trace
from ._shared import dropdown_menu, header_margin, header_title


def _format_fixed_payout_label(risk_dollars, reward_dollars):
    return f"Risk ${float(risk_dollars):,.0f} / Reward ${float(reward_dollars):,.0f}"


def _format_fixed_payout_short_label(risk_dollars, reward_dollars):
    return f"{float(risk_dollars):,.0f}/{float(reward_dollars):,.0f}"


def _build_fixed_payout_rolling_trace(*, trade_frame, strategy_label, payout_short_label, visible):
    return build_line_trace(
        x=trade_frame.index,
        y=trade_frame["Rolling PnL"],
        mode="lines",
        name=strategy_label,
        legendgroup=strategy_label,
        showlegend=True,
        visible=visible,
        customdata=np.column_stack(
            [
                trade_frame["Trade PnL"],
                trade_frame["Rolling Win Rate"],
                trade_frame["Cumulative PnL"],
            ]
        ),
        hovertemplate=(
            f"Payout: {payout_short_label}"
            "<br>Date: %{x|%Y-%m-%d}"
            "<br>Rolling PnL: $%{y:,.2f}"
            "<br>Trade PnL: $%{customdata[0]:,.2f}"
            "<br>Rolling Win Rate: %{customdata[1]:.2%}"
            "<br>Cumulative PnL: $%{customdata[2]:,.2f}"
            "<extra></extra>"
        ),
    )


def _build_fixed_payout_cumulative_trace(*, trade_frame, strategy_label, payout_short_label, visible):
    return build_line_trace(
        x=trade_frame.index,
        y=trade_frame["Cumulative PnL"],
        mode="lines",
        name=strategy_label,
        legendgroup=strategy_label,
        showlegend=False,
        visible=visible,
        customdata=np.column_stack(
            [
                trade_frame["Trade PnL"],
                trade_frame["Rolling PnL"],
                trade_frame["Drawdown"],
            ]
        ),
        hovertemplate=(
            f"Payout: {payout_short_label}"
            "<br>Date: %{x|%Y-%m-%d}"
            "<br>Cumulative PnL: $%{y:,.2f}"
            "<br>Trade PnL: $%{customdata[0]:,.2f}"
            "<br>Rolling PnL: $%{customdata[1]:,.2f}"
            "<br>Drawdown: $%{customdata[2]:,.2f}"
            "<extra></extra>"
        ),
    )


def plot_fixed_payout_strategy_backtest_view(
    strategy_profiles_by_payout,
    *,
    payout_options,
    default_payout,
    rolling_pnl_window,
    ticker_label="Asset",
    interval_floor_label="95% Long Interval Floor",
    template="plotly_dark",
):
    """Compose the fixed-payout strategy backtest view."""
    payout_options = list(payout_options)
    default_payout_label = _format_fixed_payout_label(*default_payout)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.48, 0.52],
        subplot_titles=(
            f"{int(rolling_pnl_window)}-Trade Rolling PnL",
            "Cumulative PnL",
        ),
    )

    trace_indices_by_payout = {}
    for risk_dollars, reward_dollars in payout_options:
        payout_label = _format_fixed_payout_label(risk_dollars, reward_dollars)
        payout_short_label = _format_fixed_payout_short_label(risk_dollars, reward_dollars)
        strategy_profiles = strategy_profiles_by_payout[payout_label]
        trace_indices_by_payout[payout_label] = []
        is_default_payout = payout_label == default_payout_label

        for profile in strategy_profiles:
            trade_frame = pd.DataFrame(profile["trade_frame"]).copy()
            strategy_label = str(profile["label"])

            rolling_trace_index = len(fig.data)
            fig.add_trace(
                _build_fixed_payout_rolling_trace(
                    trade_frame=trade_frame,
                    strategy_label=strategy_label,
                    payout_short_label=payout_short_label,
                    visible=is_default_payout,
                ),
                row=1,
                col=1,
            )
            trace_indices_by_payout[payout_label].append(rolling_trace_index)

            cumulative_trace_index = len(fig.data)
            fig.add_trace(
                _build_fixed_payout_cumulative_trace(
                    trade_frame=trade_frame,
                    strategy_label=strategy_label,
                    payout_short_label=payout_short_label,
                    visible=is_default_payout,
                ),
                row=2,
                col=1,
            )
            trace_indices_by_payout[payout_label].append(cumulative_trace_index)

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(248, 250, 252, 0.45)",
        line_width=1,
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(248, 250, 252, 0.45)",
        line_width=1,
        row=2,
        col=1,
    )

    total_trace_count = len(fig.data)
    dropdown_buttons = []
    for risk_dollars, reward_dollars in payout_options:
        payout_label = _format_fixed_payout_label(risk_dollars, reward_dollars)
        visible_mask = [False] * total_trace_count
        for trace_index in trace_indices_by_payout[payout_label]:
            visible_mask[trace_index] = True

        dropdown_buttons.append(
            dict(
                label=_format_fixed_payout_short_label(risk_dollars, reward_dollars),
                method="update",
                args=[
                    {"visible": visible_mask},
                    {
                        "title": header_title(
                            f"{ticker_label} Fixed Payout Strategy vs {interval_floor_label} ({payout_label})"
                        )
                    },
                ],
            )
        )

    fig.update_layout(
        title=header_title(
            f"{ticker_label} Fixed Payout Strategy vs {interval_floor_label} ({default_payout_label})"
        ),
        height=950,
        margin=header_margin(),
        template=template,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        updatemenus=[
            dropdown_menu(
                buttons=dropdown_buttons,
                x=1.0,
                xanchor="right",
                active=payout_options.index(default_payout),
                y=1.18,
                bgcolor="rgba(15, 23, 42, 0.95)",
                bordercolor="rgba(148, 163, 184, 0.45)",
                font=dict(size=11),
            )
        ],
        annotations=list(fig.layout.annotations)
        + [
            dict(
                x=1.0,
                y=1.205,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                text="Risk / Reward",
                font=dict(size=11, color="rgba(226, 232, 240, 0.92)"),
            )
        ],
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(
        title_text=f"{int(rolling_pnl_window)}-Trade Rolling PnL",
        tickprefix="$",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Cumulative PnL", tickprefix="$", row=2, col=1)
    return fig

"""Portfolio relative strength visualization views."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_z_score_diff_dropdown(z_scores_map: Mapping[str, pd.Series], *, title_metric: str = "Return") -> go.Figure:
    diff_frames = {}
    global_zmin = np.inf
    global_zmax = -np.inf

    for label, z_scores in z_scores_map.items():
        diff = pd.DataFrame(
            z_scores.values[:, None] - z_scores.values,
            index=z_scores.index,
            columns=z_scores.index,
        ).astype(float)
        diff_frames[label] = diff
        global_zmin = min(global_zmin, diff.min().min())
        global_zmax = max(global_zmax, diff.max().max())

    fig = go.Figure()
    labels = list(diff_frames.keys())
    for idx, label in enumerate(labels):
        diff = diff_frames[label]
        fig.add_trace(
            go.Heatmap(
                z=diff.values,
                x=diff.columns,
                y=diff.index,
                zmin=global_zmin,
                zmax=global_zmax,
                colorscale="RdBu",
                colorbar=dict(title="Z-Score Diff"),
                text=diff.round(2),
                hovertemplate="%{y} vs %{x}<br>Diff: %{z:.2f}<extra></extra>",
                visible=idx == 0,
            )
        )

    buttons = []
    for idx, label in enumerate(labels):
        visibility = [i == idx for i in range(len(labels))]
        buttons.append(
            dict(
                label=f"{label}-Day",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{title_metric} Pairwise Z-Score Differences ({label}-day)"},
                ],
            )
        )

    fig.update_layout(
        title=f"{title_metric} Pairwise Z-Score Differences ({labels[0]}-day)" if labels else f"{title_metric} Pairwise Z-Score Differences",
        template="plotly_dark",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.2,
                yanchor="top",
            )
        ],
        height=700,
    )
    fig.update_xaxes(title_text="Assets")
    fig.update_yaxes(title_text="Assets")
    return fig


def format_ticker(ticker, sign):
    return f"({ticker})" if sign < 0 else ticker


def format_snapshot_map(snapshot_map: Mapping[str, pd.Series], sign_series: pd.Series) -> dict[str, pd.Series]:
    formatted = {}
    for label, series in snapshot_map.items():
        formatted_series = series.copy()
        formatted_series.index = [format_ticker(ticker, sign_series.loc[ticker]) for ticker in formatted_series.index]
        formatted[label] = formatted_series
    return formatted


def highlight_signed_bars(index, sign_series: pd.Series, default_color="#636EFA", highlight_color="#F59E0B"):
    colors = []
    for ticker in index:
        raw_ticker = ticker.strip("()") if isinstance(ticker, str) else ticker
        sign_value = sign_series.reindex([raw_ticker]).fillna(1.0).iloc[0]
        colors.append(highlight_color if sign_value < 0 else default_color)
    return colors


def plot_benchmark_snapshot_zscores(
    *,
    windows_signed: Mapping[str, pd.Series],
    windows_unsigned: Mapping[str, pd.Series],
    windows_benchmark_minus_assets_signed: Mapping[str, pd.Series],
    windows_benchmark_minus_assets_unsigned: Mapping[str, pd.Series],
    sign_series: pd.Series,
    benchmark_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Asset Sharpe Ratio Z-Scores (Signed)",
            "Asset Sharpe Ratio Z-Scores (Unsigned)",
            f"{benchmark_label} - Asset Sharpe Spread Z-Score (Signed)",
            f"{benchmark_label} - Asset Sharpe Spread Z-Score (Unsigned)",
        ),
    )
    window_labels = list(windows_signed.keys())
    for idx, window in enumerate(window_labels):
        z_signed = windows_signed[window]
        z_unsigned = windows_unsigned[window]
        z_diff_signed = windows_benchmark_minus_assets_signed[window]
        z_diff_unsigned = windows_benchmark_minus_assets_unsigned[window]
        fig.add_trace(go.Bar(x=z_signed.index, y=z_signed.values, name=f"{window}-Day Signed", showlegend=False, visible=idx == 0), row=1, col=1)
        fig.add_trace(
            go.Bar(
                x=z_unsigned.index,
                y=z_unsigned.values,
                marker_color=highlight_signed_bars(z_unsigned.index, sign_series),
                name=f"{window}-Day Unsigned",
                showlegend=False,
                visible=idx == 0,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=z_diff_signed.index,
                y=z_diff_signed.values,
                name=f"{window}-Day {benchmark_label} - Asset Spread",
                showlegend=False,
                visible=idx == 0,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=z_diff_unsigned.index,
                y=z_diff_unsigned.values,
                marker_color=highlight_signed_bars(z_diff_unsigned.index, sign_series),
                name=f"{window}-Day {benchmark_label} - Asset Spread",
                showlegend=False,
                visible=idx == 0,
            ),
            row=2,
            col=2,
        )

    for row in (1, 2):
        for col in (1, 2):
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=row, col=col)
            for level, color in [(1, "green"), (2, "orange"), (3, "purple")]:
                fig.add_hline(y=level, line_dash="dot", line_color=color, row=row, col=col)
                fig.add_hline(y=-level, line_dash="dot", line_color=color, row=row, col=col)
            fig.add_hrect(y0=-1, y1=1, fillcolor="lightgreen", opacity=0.3, layer="below", line_width=0, row=row, col=col)
            fig.add_hrect(y0=-2, y1=2, fillcolor="lightyellow", opacity=0.3, layer="below", line_width=0, row=row, col=col)
            fig.add_hrect(y0=-3, y1=3, fillcolor="lightcoral", opacity=0.3, layer="below", line_width=0, row=row, col=col)

    buttons = []
    for idx, window in enumerate(window_labels):
        visibility = [False] * (4 * len(window_labels))
        base = 4 * idx
        visibility[base : base + 4] = [True, True, True, True]
        buttons.append(
            dict(
                label=f"{window}-Day",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Sharpe Z-Scores + {benchmark_label} Sharpe Spread Z-Scores - {window}-Day"},
                ],
            )
        )

    fig.update_layout(
        title=f"Sharpe Z-Scores + {benchmark_label} Sharpe Spread Z-Scores - 21-Day",
        template="plotly_dark",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.15,
                yanchor="top",
            )
        ],
        height=950,
    )
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="Assets", row=row, col=col)
            fig.update_yaxes(title_text="Z-Score", row=row, col=col)
    return fig

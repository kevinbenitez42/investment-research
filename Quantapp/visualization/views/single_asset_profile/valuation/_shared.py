"""Shared valuation view helpers."""

from __future__ import annotations

import plotly.graph_objects as go

RESPONSIVE_FIGURE_CONFIG = {"responsive": True, "displaylogo": False}


def apply_standard_figure_layout(
    fig: go.Figure,
    title: str,
    height: int,
    *,
    bottom_margin: int = 140,
) -> None:
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_dark",
        paper_bgcolor="#020817",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#0f172a",
            "font": {"color": "#e2e8f0", "size": 13},
            "namelength": -1,
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "x": 0,
            "xanchor": "left",
            "bgcolor": "rgba(2, 8, 23, 0.6)",
        },
        autosize=True,
        height=height,
        margin={"l": 60, "r": 30, "t": 120, "b": bottom_margin},
    )


def apply_price_history_layout(
    fig: go.Figure,
    title: str,
    *,
    yaxis_title: str = "USD per share",
    height: int = 680,
) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#020817",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#0f172a", "font_color": "#e2e8f0"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        autosize=True,
        height=height,
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False, automargin=True)


def apply_subplot_x_grid(fig: go.Figure, rows: int, cols: int = 1) -> None:
    for row_number in range(1, rows + 1):
        for col_number in range(1, cols + 1):
            fig.update_xaxes(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.18)",
                zeroline=False,
                automargin=True,
                row=row_number,
                col=col_number,
            )


def apply_dark_yaxis(
    fig: go.Figure,
    *,
    title_text: str,
    row: int,
    col: int = 1,
    tickformat: str | None = None,
    zeroline: bool = False,
) -> None:
    fig.update_yaxes(
        title_text=title_text,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=zeroline,
        zerolinecolor="rgba(226, 232, 240, 0.35)" if zeroline else None,
        automargin=True,
        tickformat=tickformat,
        row=row,
        col=col,
    )


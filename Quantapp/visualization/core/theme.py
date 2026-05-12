"""Shared Plotly theme helpers."""

from __future__ import annotations


PLOTLY_DARK_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#05070b",
    "plot_bgcolor": "#05070b",
    "font": dict(color="#e5e7eb"),
    "legend": dict(
        bgcolor="rgba(5, 7, 11, 0.55)",
        bordercolor="rgba(148, 163, 184, 0.20)",
        borderwidth=1,
    ),
    "hoverlabel": dict(
        bgcolor="#0f172a",
        bordercolor="#1f2937",
        font=dict(color="#f8fafc"),
    ),
}
PLOTLY_DARK_BACKGROUND = "#05070b"
PLOTLY_DARK_GRID = "rgba(148, 163, 184, 0.16)"
PLOTLY_DARK_AXIS = "rgba(148, 163, 184, 0.24)"
PLOTLY_DARK_TEXT = "rgba(226, 232, 240, 0.92)"
PLOTLY_DARK_MENU_BG = "rgba(9, 14, 24, 0.95)"


def coerce_dark_foreground(color):
    """Convert hard-coded dark foreground colors to readable dark-theme colors."""
    if not isinstance(color, str):
        return color

    normalized = color.replace(" ", "").lower()
    if normalized in {
        "black",
        "#000",
        "#000000",
        "rgb(0,0,0)",
        "rgba(0,0,0,0.85)",
        "rgba(0,0,0,1)",
    }:
        return "rgba(241, 245, 249, 0.88)"
    return color


def apply_plotly_dark_theme(fig):
    """Apply the shared Quantapp dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_DARK_THEME)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PLOTLY_DARK_GRID,
        zeroline=False,
        showline=True,
        linecolor=PLOTLY_DARK_AXIS,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PLOTLY_DARK_GRID,
        zeroline=False,
        showline=True,
        linecolor=PLOTLY_DARK_AXIS,
    )
    fig.update_scenes(
        bgcolor=PLOTLY_DARK_BACKGROUND,
        xaxis=dict(
            showbackground=True,
            backgroundcolor=PLOTLY_DARK_BACKGROUND,
            gridcolor=PLOTLY_DARK_GRID,
            zerolinecolor=PLOTLY_DARK_AXIS,
        ),
        yaxis=dict(
            showbackground=True,
            backgroundcolor=PLOTLY_DARK_BACKGROUND,
            gridcolor=PLOTLY_DARK_GRID,
            zerolinecolor=PLOTLY_DARK_AXIS,
        ),
        zaxis=dict(
            showbackground=True,
            backgroundcolor=PLOTLY_DARK_BACKGROUND,
            gridcolor=PLOTLY_DARK_GRID,
            zerolinecolor=PLOTLY_DARK_AXIS,
        ),
    )

    for trace in fig.data:
        trace_line = getattr(trace, "line", None)
        trace_line_color = getattr(trace_line, "color", None) if trace_line is not None else None
        if trace_line_color is not None:
            trace.line.color = coerce_dark_foreground(trace_line_color)

    for shape in fig.layout.shapes or []:
        shape_line = getattr(shape, "line", None)
        shape_line_color = getattr(shape_line, "color", None) if shape_line is not None else None
        if shape_line_color is not None:
            shape.line.color = coerce_dark_foreground(shape_line_color)

    for annotation in fig.layout.annotations or []:
        font_payload = annotation.font.to_plotly_json() if annotation.font else {}
        current_color = font_payload.get("color")
        if current_color is None:
            font_payload["color"] = PLOTLY_DARK_TEXT
        else:
            font_payload["color"] = coerce_dark_foreground(current_color)
        annotation.font = font_payload

    for menu in fig.layout.updatemenus or []:
        menu.bgcolor = PLOTLY_DARK_MENU_BG
        menu.bordercolor = PLOTLY_DARK_AXIS
        menu.borderwidth = 1
        menu.font = dict(color="#e5e7eb", size=12)

    return fig

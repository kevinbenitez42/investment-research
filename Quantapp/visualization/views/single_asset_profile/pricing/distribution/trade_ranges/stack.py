"""Stacked trade-range history and cone view."""

from __future__ import annotations

import copy

import pandas as pd
from plotly.subplots import make_subplots

from ._shared import dropdown_menu, header_margin, header_title


def _axis_index_from_ref(axis_ref, axis_letter):
    if axis_ref in (None, axis_letter):
        return 1
    if isinstance(axis_ref, str) and axis_ref.startswith(axis_letter):
        digits = "".join(ch for ch in axis_ref[1:] if ch.isdigit())
        return int(digits) if digits else 1
    return 1


def _shift_axis_ref(axis_ref, row_offset):
    if not isinstance(axis_ref, str) or axis_ref == "paper":
        return axis_ref
    suffix = " domain" if axis_ref.endswith(" domain") else ""
    base = axis_ref[:-7] if suffix else axis_ref
    if not base or base[0] not in ("x", "y"):
        return axis_ref
    axis_letter = base[0]
    digits = base[1:]
    axis_index = int(digits) if digits else 1
    shifted_index = axis_index + row_offset
    shifted_base = axis_letter if shifted_index == 1 else f"{axis_letter}{shifted_index}"
    return shifted_base + suffix


def _figure_row_count(fig):
    row_count = max(
        (_axis_index_from_ref(getattr(trace, "yaxis", None), "y") for trace in fig.data),
        default=0,
    )
    if row_count <= 0:
        raise ValueError("Trade-range figure does not contain any subplot traces to stack.")
    return row_count


def _subplot_titles_from_figure(fig, count):
    titles = [
        str(getattr(annotation, "text", ""))
        for annotation in list(getattr(fig.layout, "annotations", []))[:count]
    ]
    if len(titles) != count:
        raise ValueError("Trade-range figure is missing expected subplot titles.")
    return tuple(titles)


def _to_json(item):
    payload = item.to_plotly_json() if hasattr(item, "to_plotly_json") else dict(item)
    return copy.deepcopy(payload)


def _shift_shapes(shapes, row_offset):
    shifted_shapes = []
    for shape in shapes:
        shape_payload = _to_json(shape)
        shape_payload["xref"] = _shift_axis_ref(shape_payload.get("xref"), row_offset)
        shape_payload["yref"] = _shift_axis_ref(shape_payload.get("yref"), row_offset)
        shifted_shapes.append(shape_payload)
    return shifted_shapes


def _shift_annotations(annotations, row_offset):
    shifted_annotations = []
    for annotation in annotations:
        annotation_payload = _to_json(annotation)
        annotation_payload["xref"] = _shift_axis_ref(annotation_payload.get("xref"), row_offset)
        annotation_payload["yref"] = _shift_axis_ref(annotation_payload.get("yref"), row_offset)
        shifted_annotations.append(annotation_payload)
    return shifted_annotations


def _window_title(base_title, window_label):
    suffix = "-Session Lookback)"
    marker = base_title.rfind("(")
    suffix_index = base_title.rfind(suffix)
    if marker != -1 and suffix_index != -1 and marker < suffix_index:
        return f"{base_title[:marker + 1]}{window_label}{suffix}"
    return base_title


def _layout_axis(fig, axis_letter, axis_index):
    axis_name = f"{axis_letter}axis" if axis_index == 1 else f"{axis_letter}axis{axis_index}"
    return getattr(fig.layout, axis_name, None)


def _coerce_layout_value(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _axis_ref(axis_letter, axis_index):
    return axis_letter if axis_index == 1 else f"{axis_letter}{axis_index}"


def _datetime_bounds(fig):
    bounds = []
    for trace in fig.data:
        x_values = getattr(trace, "x", None)
        if x_values is None:
            continue
        try:
            x_index = pd.to_datetime(pd.Index(x_values), errors="coerce")
        except Exception:
            continue
        if len(x_index) == 0:
            continue
        x_index = x_index[~pd.isna(x_index)]
        if len(x_index) == 0:
            continue
        bounds.append((x_index.min(), x_index.max()))
    if not bounds:
        return None, None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def _copy_axis_layout(combined_fig, source_fig, source_row, target_row):
    source_xaxis = _layout_axis(source_fig, "x", source_row)
    if source_xaxis is not None:
        x_kwargs = {}
        for attr in ("range", "tickmode", "tickvals", "ticktext", "ticksuffix", "tickprefix"):
            value = getattr(source_xaxis, attr, None)
            if value is not None:
                x_kwargs[attr] = _coerce_layout_value(value)
        title_value = getattr(getattr(source_xaxis, "title", None), "text", None)
        if title_value is not None:
            x_kwargs["title_text"] = title_value
        if x_kwargs:
            combined_fig.update_xaxes(row=target_row, col=1, **x_kwargs)

    source_yaxis = _layout_axis(source_fig, "y", source_row)
    if source_yaxis is not None:
        y_kwargs = {}
        for attr in ("range", "tickprefix", "tickformat", "ticksuffix"):
            value = getattr(source_yaxis, attr, None)
            if value is not None:
                y_kwargs[attr] = _coerce_layout_value(value)
        title_value = getattr(getattr(source_yaxis, "title", None), "text", None)
        if title_value is not None:
            y_kwargs["title_text"] = title_value
        if y_kwargs:
            combined_fig.update_yaxes(row=target_row, col=1, **y_kwargs)


def _history_timeframe_buttons(global_start, global_end, history_row_count):
    def _timeframe_range(years=None):
        start = global_start if years is None else max(global_start, global_end - pd.DateOffset(years=years))
        layout_updates = {}
        for axis_index in range(1, history_row_count + 1):
            axis_name = "xaxis" if axis_index == 1 else f"xaxis{axis_index}"
            layout_updates[f"{axis_name}.range"] = [start, global_end]
        return layout_updates

    return [
        dict(label="1Y", method="relayout", args=[_timeframe_range(1)]),
        dict(label="2Y", method="relayout", args=[_timeframe_range(2)]),
        dict(label="3Y", method="relayout", args=[_timeframe_range(3)]),
        dict(label="5Y", method="relayout", args=[_timeframe_range(5)]),
        dict(label="10Y", method="relayout", args=[_timeframe_range(10)]),
        dict(label="Full", method="relayout", args=[_timeframe_range(None)]),
    ]


def plot_trade_range_stack_view(
    history_fig,
    cone_fig,
    *,
    title_text,
    template="plotly_dark",
):
    """Compose the stacked trade-range history and cone figure."""
    history_row_count = _figure_row_count(history_fig)
    cone_row_count = _figure_row_count(cone_fig)
    total_rows = history_row_count + cone_row_count

    history_row_heights = [0.14, 0.12, 0.11, 0.11, 0.11][:history_row_count]
    if len(history_row_heights) < history_row_count:
        history_row_heights.extend([0.11] * (history_row_count - len(history_row_heights)))
    cone_height_budget = max(0.20, 1.0 - sum(history_row_heights))
    if cone_row_count <= 1:
        cone_row_heights = [cone_height_budget]
    else:
        cone_panel_count = cone_row_count - 1
        cone_distribution_height = cone_height_budget * 0.32
        cone_panel_height = (cone_height_budget - cone_distribution_height) / cone_panel_count
        cone_row_heights = [cone_panel_height] * cone_panel_count + [cone_distribution_height]

    combined_fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        row_heights=history_row_heights + cone_row_heights,
        subplot_titles=(
            _subplot_titles_from_figure(history_fig, history_row_count)
            + _subplot_titles_from_figure(cone_fig, cone_row_count)
        ),
    )
    subplot_title_annotations = [copy.deepcopy(annotation).to_plotly_json() for annotation in combined_fig.layout.annotations]

    for trace in history_fig.data:
        target_row = _axis_index_from_ref(getattr(trace, "yaxis", None), "y")
        stacked_trace = copy.deepcopy(trace)
        stacked_trace.xaxis = None
        stacked_trace.yaxis = None
        combined_fig.add_trace(stacked_trace, row=target_row, col=1)

    for trace in cone_fig.data:
        target_row = history_row_count + _axis_index_from_ref(getattr(trace, "yaxis", None), "y")
        stacked_trace = copy.deepcopy(trace)
        stacked_trace.xaxis = None
        stacked_trace.yaxis = None
        combined_fig.add_trace(stacked_trace, row=target_row, col=1)

    history_static_shapes = _shift_shapes(history_fig.layout.shapes, 0)
    history_extra_annotations = _shift_annotations(history_fig.layout.annotations[history_row_count:], 0)
    cone_default_shapes = _shift_shapes(cone_fig.layout.shapes, history_row_count)
    cone_default_annotations = _shift_annotations(cone_fig.layout.annotations[cone_row_count:], history_row_count)

    for source_row in range(1, history_row_count + 1):
        _copy_axis_layout(combined_fig, history_fig, source_row, source_row)
    for source_row in range(1, cone_row_count + 1):
        _copy_axis_layout(combined_fig, cone_fig, source_row, history_row_count + source_row)

    if history_row_count > 1:
        history_anchor_ref = _axis_ref("x", history_row_count)
        for row in range(1, history_row_count):
            combined_fig.update_xaxes(matches=history_anchor_ref, row=row, col=1)

    cone_path_row_count = max(0, cone_row_count - 1)
    if cone_path_row_count > 1:
        cone_anchor_row = history_row_count + cone_path_row_count
        cone_anchor_ref = _axis_ref("x", cone_anchor_row)
        for row in range(history_row_count + 1, cone_anchor_row):
            combined_fig.update_xaxes(matches=cone_anchor_ref, row=row, col=1)

    updatemenus = []
    static_header_annotations = []
    history_menu = history_fig.layout.updatemenus[0] if getattr(history_fig.layout, "updatemenus", None) else None
    cone_menu = cone_fig.layout.updatemenus[0] if getattr(cone_fig.layout, "updatemenus", None) else None
    active_index = 0
    initial_title_text = title_text

    if history_menu is not None and cone_menu is not None:
        history_buttons = list(history_menu.buttons)
        cone_buttons = list(cone_menu.buttons)
        if len(history_buttons) == len(cone_buttons) and history_buttons:
            active_index = int(history_menu.active) if history_menu.active is not None else 0
            active_index = max(0, min(active_index, len(history_buttons) - 1))
            combined_buttons = []

            for history_button, cone_button in zip(history_buttons, cone_buttons):
                history_label = str(history_button.label)
                cone_label = str(cone_button.label)
                if history_label != cone_label:
                    raise ValueError("History and cone dropdown labels do not align for stacked trade-range figure.")

                history_args = history_button.args if history_button.args is not None else []
                cone_args = cone_button.args if cone_button.args is not None else []
                history_visibility = list(history_args[0].get("visible", [])) if history_args else []
                cone_visibility = list(cone_args[0].get("visible", [])) if cone_args else []
                if not history_visibility:
                    history_visibility = [trace.visible if trace.visible is not None else True for trace in history_fig.data]
                if not cone_visibility:
                    cone_visibility = [trace.visible if trace.visible is not None else True for trace in cone_fig.data]

                cone_layout_updates = cone_args[1] if len(cone_args) > 1 else {}
                cone_shapes = _shift_shapes(
                    cone_layout_updates.get("shapes", cone_fig.layout.shapes),
                    history_row_count,
                )
                cone_annotations = _shift_annotations(
                    cone_layout_updates.get("annotations", cone_fig.layout.annotations)[cone_row_count:],
                    history_row_count,
                )
                combined_buttons.append(
                    dict(
                        label=history_label,
                        method="update",
                        args=[
                            {"visible": history_visibility + cone_visibility},
                            {
                                "title": header_title(_window_title(title_text, history_label)),
                                "shapes": history_static_shapes + cone_shapes,
                                "annotations": subplot_title_annotations + history_extra_annotations + static_header_annotations + cone_annotations,
                            },
                        ],
                    )
                )

            initial_title_text = _window_title(title_text, str(history_buttons[active_index].label))
            updatemenus.append(
                dropdown_menu(
                    buttons=combined_buttons,
                    x=0.0,
                    active=active_index,
                )
            )

    history_global_start, history_global_end = _datetime_bounds(history_fig)
    if history_global_start is not None and history_global_end is not None:
        timeframe_menu_x = 0.18 if updatemenus else 0.0
        updatemenus.append(
            dropdown_menu(
                buttons=_history_timeframe_buttons(history_global_start, history_global_end, history_row_count),
                x=timeframe_menu_x,
                active=2,
            )
        )
        static_header_annotations.append(
            dict(
                text="View timeframe",
                x=timeframe_menu_x,
                xref="paper",
                y=1.115,
                yref="paper",
                showarrow=False,
                xanchor="left",
            )
        )

    combined_fig.update_layout(
        title=header_title(initial_title_text),
        height=2575 if cone_row_count <= 2 else 2875,
        margin=header_margin(top=205),
        template=template,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        bargap=0.08,
        updatemenus=updatemenus,
        shapes=history_static_shapes + cone_default_shapes,
        annotations=subplot_title_annotations + history_extra_annotations + static_header_annotations + cone_default_annotations,
    )
    return combined_fig

"""Visualization utilities."""

from importlib import import_module

_LAZY_EXPORTS = {
    "Plotter": (".plotter", "Plotter"),
    "PieChartPlotter": (".pie_chart_plotter", "PieChartPlotter"),
    "BarChartPlotter": (".bar_chart_plotter", "BarChartPlotter"),
    "HeatmapPlotter": (".heatmap_plotter", "HeatmapPlotter"),
    "add_mean_reference_line": (".figure_helpers", "add_mean_reference_line"),
    "add_std_annotations": (".figure_helpers", "add_std_annotations"),
    "add_zone_annotation": (".figure_helpers", "add_zone_annotation"),
    "add_horizontal_zone": (".figure_helpers", "add_horizontal_zone"),
    "add_horizontal_zone_trace": (".figure_helpers", "add_horizontal_zone_trace"),
    "build_time_range_buttons": (".figure_helpers", "build_time_range_buttons"),
    "build_detail_visibility_mask": (".figure_helpers", "build_detail_visibility_mask"),
    "build_visibility_mask": (".figure_helpers", "build_visibility_mask"),
    "add_sigma_reference_lines": (".figure_helpers", "add_sigma_reference_lines"),
}

__all__ = [
    "Plotter",
    "PieChartPlotter",
    "BarChartPlotter",
    "HeatmapPlotter",
    "add_mean_reference_line",
    "add_std_annotations",
    "add_zone_annotation",
    "add_horizontal_zone",
    "add_horizontal_zone_trace",
    "build_time_range_buttons",
    "build_detail_visibility_mask",
    "build_visibility_mask",
    "add_sigma_reference_lines",
]


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

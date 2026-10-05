"""Offline integration checks for tab routing and deferred diagnostics."""

import ast
import copy
import inspect
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from types import SimpleNamespace
import unittest

from dash import Dash, Input, Output, State, no_update
import plotly.graph_objects as go

from apps.web.zscore_baseline import BASELINE
from test_momentum_zscore_baseline import ROOT, dashboard_functions


def source_tree():
    return ast.parse((ROOT / "apps/web/momentum_dashboard.py").read_text(encoding="utf-8"))


class TabLoadingTests(unittest.TestCase):
    def test_composite_tabs_hide_shared_chart_container(self):
        ns = dashboard_functions("_toggle_risk_compounding_v2_view", namespace={})
        toggle = ns["_toggle_risk_compounding_v2_view"]
        for tab in ("risk_compounding_v2", "volatility_efficiency", "window_diagnostics", "drawdown"):
            combined, shared, snapshot = toggle(tab)
            self.assertEqual(combined["display"], "flex" if tab == "risk_compounding_v2" else "none")
            self.assertEqual(shared["display"], "block" if tab == "drawdown" else "none")
            self.assertEqual(snapshot["display"], "block" if tab == "window_diagnostics" else "none")

    def test_volatility_tab_preserves_mounted_chart_dimensions(self):
        ns = dashboard_functions("_update_momentum_efficiency_dashboard", namespace={
            "no_update": no_update,
            "block18_tab_config": {"volatility_efficiency": {}},
            "_block18_normalize_ratio_type": lambda value: value,
            "_block18_normalize_benchmark_selection": lambda value: value,
        })
        callback = ns["_update_momentum_efficiency_dashboard"]
        values = dict.fromkeys(inspect.signature(callback).parameters)
        values.update(active_tab="volatility_efficiency", window=20)
        figure, style, status, animate = callback(**values)
        self.assertIs(figure, no_update)
        self.assertIs(style, no_update)
        self.assertIn("Volatility", status)
        self.assertFalse(animate)

    def test_registered_router_refreshes_composite_tabs_and_has_one_graph_writer(self):
        app = Dash(__name__)
        calls = []
        ns = dashboard_functions("_block18_baseline_callback", "_block18_dashboard_outputs", "_block18_route_dashboard_graph", namespace={
            "Input": Input, "Output": Output, "State": State, "no_update": no_update,
            "block18_dash_app": app, "_zscore_baseline": BASELINE,
            "block18_metric_graph_specs": [("volatility", "Volatility", "1"), ("skew", "Skew", "2")],
            "block18_update_button_loading_style": {}, "block18_update_button_style": {},
            "_update_risk_compounding_v2_view": lambda *args: calls.append(args) or ("risk",) * 4,
            "_update_volatility_efficiency_panels": lambda *args: calls.append(args) or ("panel",) * 2,
        })
        tree = source_tree()
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_update_momentum_efficiency_dashboard")
        # Retain the real signature and decorators; replace only data-heavy work.
        stub = copy.deepcopy(node)
        stub.body = ast.parse("return (ratio_type, playback_position, window, False)").body
        exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), "router", "exec"), ns)
        registration = next(iter(app.callback_map.values()))
        callback = registration["callback"].__wrapped__
        inputs = registration["inputs"]
        self.assertIn("shared-display-range-dropdown", [item["id"] for item in inputs])
        self.assertIn("diagnostics-asof-slider", [item["id"] for item in inputs])
        for tab in ("drawdown", "risk_compounding_v2", "volatility_efficiency", "window_diagnostics"):
            settings = {
                "momentum-efficiency-view-tabs": tab, "shared-ratio-dropdown": "volatility",
                "momentum-efficiency-rendered-tab": tab,
                "shared-display-range-dropdown": "1y", "diagnostics-asof-slider": 123,
                "shared-window-input": 20, "shared-zscore-baseline": "trailing", "shared-zscore-days": 60,
            }
            result = callback(*[settings.get(item["id"]) for item in inputs + registration["state"]])
            self.assertEqual(len(result), 14)
            self.assertEqual(result[-1], tab)
            if tab == "window_diagnostics":
                self.assertEqual(result[:7], (no_update, no_update, "volatility", 123, 20, False, False))
            else:
                self.assertEqual(result[:7], ("volatility", 123, no_update, no_update, 20, False, False))
            if tab == "risk_compounding_v2":
                self.assertEqual(result[7:11], ("risk",) * 4)
            if tab == "volatility_efficiency":
                self.assertEqual(result[11:13], ("panel",) * 2)
                self.assertEqual(calls[-1], (tab, None, "1y", 20))
        writers = []
        for function in tree.body:
            if isinstance(function, ast.FunctionDef):
                for decorator in function.decorator_list:
                    for call in ast.walk(decorator):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Output":
                            if [getattr(arg, "value", None) for arg in call.args[:2]] == ["momentum-efficiency-tab-graph", "figure"]:
                                writers.append(function.name)
        self.assertEqual(writers, ["_update_momentum_efficiency_dashboard"])

    def test_horizon_snapshot_uses_an_isolated_graph(self):
        ns = dashboard_functions("_block18_route_dashboard_graph", namespace={
            "no_update": no_update,
        })
        destination = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

        @ns["_block18_route_dashboard_graph"]
        def callback(active_tab, rendered_tab):
            return destination, {"height": "600px"}, "snapshot", False, active_tab

        for previous in ("historical_surface_3d", "drawdown", "risk_compounding_v2"):
            result = callback("window_diagnostics", previous)
            self.assertIs(result[0], no_update)
            self.assertIs(result[1], no_update)
            self.assertIs(result[2], destination)
            self.assertEqual(result[3], {"height": "600px"})
            self.assertEqual(result[-1], "window_diagnostics")
        result = callback("drawdown", "window_diagnostics")
        self.assertIs(result[0], destination)
        self.assertIs(result[2], no_update)

    def test_playback_plan_reads_ratio_from_signature(self):
        normalized = []
        ns = dashboard_functions("_block18_register_functional_playback_plan", namespace={
            "_block18_normalize_ratio_type": lambda value: normalized.append(value) or value,
            "_block18_ratio_label": lambda value: value.title(),
            "_block18_diagnostics_playback_titles": lambda value: (),
            "block18_default_ratio_type": "sharpe",
            "block18_functional_playback_plan_cache": {},
            "ticker_str": "ASSET",
            "store_bounded": lambda *args, **kwargs: None,
        })
        signature = (("full", 252, "3y"), "sortino", ("SPY",), (3, 850), "b_spline", 25.0, True, 0)
        self.assertFalse(ns["_block18_register_functional_playback_plan"](go.Figure(), signature))
        self.assertEqual(normalized, ["sortino"])

    def test_playback_does_not_patch_a_previous_tabs_figure(self):
        node = next(n for n in source_tree().body
                    if isinstance(n, ast.FunctionDef) and n.name == "_update_momentum_efficiency_dashboard")
        node.decorator_list = []
        # Exercise the real playback branch, stopping before the heavy full render.
        end = next(i for i, statement in enumerate(node.body)
                   if isinstance(statement, ast.If) and "rendered_tab" in ast.unparse(statement.test))
        node.body = node.body[:end + 1] + ast.parse("return 'full render'").body
        calls = []
        ns = {
            "no_update": no_update,
            "block18_tab_config": {"window_diagnostics": {}},
            "ctx": SimpleNamespace(triggered_id="diagnostics-asof-slider"),
            "_block18_normalize_ratio_type": lambda value: value,
            "_block18_normalize_benchmark_selection": lambda value: value,
            "_patch_functional_horizon_playback_frame": lambda *args: calls.append(args) or "patch",
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "router", "exec"), ns)
        callback = ns[node.name]
        values = dict.fromkeys(inspect.signature(callback).parameters)
        values.update(active_tab="window_diagnostics", rendered_tab="drawdown")
        self.assertEqual(callback(**values), "full render")
        self.assertEqual(calls, [])
        values["rendered_tab"] = "window_diagnostics"
        self.assertEqual(callback(**values)[0], "patch")
        self.assertEqual(len(calls), 1)

    def test_baseline_and_lookback_changes_stop_playback(self):
        context = SimpleNamespace(triggered_id=None)
        ns = dashboard_functions("_update_diagnostics_playback", "_block18_next_playback_state", namespace={
            "ctx": context, "no_update": no_update,
            "block18_default_playback_position": 100, "block18_playback_controls_style": {},
            "_block18_playback_date": lambda position: (None, position),
        })
        function = ns["_update_diagnostics_playback"]
        for trigger in ("shared-zscore-baseline", "shared-zscore-days", "shared-display-range-dropdown", "shared-update-button"):
            context.triggered_id = trigger
            values = dict.fromkeys(inspect.signature(function).parameters)
            values.update(active_tab="window_diagnostics", playing=True, position=50)
            result = function(**values)
            self.assertEqual(result[:3], (False, True, "Play"))

    def test_diagnostics_load_only_requested_symbols_once_under_concurrency(self):
        calls = []
        ns = dashboard_functions("_ensure_momentum_diagnostics_ratio", namespace={
            "ticker_str": "ASSET", "normalize_momentum_ratio_type": lambda value: value,
            "_momentum_diagnostics_lock": RLock(),
            "momentum_diagnostics_contexts_by_ratio": {},
            "momentum_diagnostics_display_contexts_by_ratio": {},
            "momentum_diagnostics_sources": {"ASSET": "asset", "SPY": "spy", "QQQ": "qqq"},
            "build_momentum_diagnostics_context": lambda close, **kwargs: calls.append(close) or {"close": close},
            "coerce_momentum_diagnostics_context": lambda context: dict(context),
        })
        ensure = ns["_ensure_momentum_diagnostics_ratio"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: ensure("volatility", ["SPY"]), range(2)))
        self.assertEqual(calls, ["asset", "spy"])
        self.assertEqual(set(results[0][0]), {"ASSET", "SPY"})
        ensure("volatility", ["QQQ"])
        self.assertEqual(calls, ["asset", "spy", "qqq"])

    def test_failed_display_preparation_does_not_publish_partial_context(self):
        def fail(context):
            raise ValueError("invalid display context")
        ns = dashboard_functions("_ensure_momentum_diagnostics_ratio", namespace={
            "ticker_str": "ASSET", "normalize_momentum_ratio_type": lambda value: value,
            "_momentum_diagnostics_lock": RLock(),
            "momentum_diagnostics_contexts_by_ratio": {},
            "momentum_diagnostics_display_contexts_by_ratio": {},
            "momentum_diagnostics_sources": {"ASSET": "asset"},
            "build_momentum_diagnostics_context": lambda *args, **kwargs: {},
            "coerce_momentum_diagnostics_context": fail,
        })
        with self.assertRaises(ValueError):
            ns["_ensure_momentum_diagnostics_ratio"]("volatility")
        self.assertEqual(ns["momentum_diagnostics_contexts_by_ratio"]["volatility"], {})


if __name__ == "__main__":
    unittest.main()

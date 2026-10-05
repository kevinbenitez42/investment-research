"""Offline regressions; do not launch the notebook or fetch market data."""

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State

from apps.web.dashboard_cache import store_bounded
from apps.web.zscore_baseline import BASELINE, baseline_key, finish_zscore_figure, normalize


ROOT = Path(__file__).resolve().parents[1]
OFFSETS = {"full": None, "1y": pd.DateOffset(years=1), "3y": pd.DateOffset(years=3)}


def dashboard_functions(*names, namespace):
    """Load pure dashboard helpers without executing its data-loading startup."""
    path = ROOT / "apps/web/momentum_dashboard.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.token = BASELINE.set(("full", 252, "3y"))
        self.series = pd.Series(np.linspace(0.1, 0.4, 800), index=pd.bdate_range("2020-01-01", periods=800))
        self.series.iloc[10] = 12.0

    def tearDown(self):
        BASELINE.reset(self.token)

    def test_static_baselines_and_full_history_display_independence(self):
        expected = (self.series - self.series.mean()) / self.series.std()
        pd.testing.assert_series_equal(normalize(self.series, OFFSETS), expected)
        BASELINE.set(("full", 20, "1y"))
        pd.testing.assert_series_equal(normalize(self.series, OFFSETS), expected)
        for mode in ("visible", "trailing"):
            BASELINE.set((mode, 60, "1y"))
            reference = self.series.iloc[-60:] if mode == "trailing" else self.series.loc[self.series.index[-1] - OFFSETS["1y"]:]
            expected = (self.series - reference.mean()) / reference.std()
            pd.testing.assert_series_equal(normalize(self.series, OFFSETS), expected)

    def test_playback_matches_static_baseline_at_each_asof_without_future_data(self):
        for mode in ("full", "visible", "trailing"):
            BASELINE.set((mode, 60, "1y"))
            frame = self.series.to_frame("volatility")
            playback = normalize(frame, OFFSETS, playback=True)
            for position in (100, 400, 799):
                history = self.series.iloc[:position + 1]
                expected = normalize(history, OFFSETS).iloc[-1]
                self.assertAlmostEqual(playback.iloc[position, 0], expected, places=7)
            shortened = normalize(frame.iloc[:401], OFFSETS, playback=True)
            pd.testing.assert_frame_equal(playback.iloc[:401], shortened)

    def test_undefined_scores_and_missing_values(self):
        constant = self.series * 0 + 1
        self.assertTrue(normalize(constant, OFFSETS).isna().all())
        self.assertTrue(normalize(constant.to_frame(), OFFSETS, playback=True).isna().all().all())
        self.assertTrue(normalize(self.series.iloc[:1], OFFSETS).isna().all())

    def test_visible_axis_ignores_offscreen_and_hidden_extremes(self):
        fig = go.Figure(go.Scatter(x=self.series.index, y=self.series, name="Volatility Z-Score"))
        fig.add_trace(go.Scatter(x=self.series.index, y=self.series * 100, visible="legendonly"))
        fig.update_layout(yaxis_title="Volatility Z-Score", xaxis_range=[self.series.index[-50], self.series.index[-1]])
        fig.add_annotation(text="Accumulate", y=-1.5)
        fig.add_annotation(text="Liquidate", y=1.5)
        fig.add_annotation(text="Neutral", y=0)
        finish_zscore_figure(fig)
        self.assertEqual(tuple(fig.layout.yaxis.range), (-2.32, 2.32))
        self.assertEqual([a.text for a in fig.layout.annotations], ["Low", "High", "Typical"])
        fig.update_xaxes(range=[self.series.index[0], self.series.index[-1]])
        finish_zscore_figure(fig)
        self.assertGreater(fig.layout.yaxis.range[1], 12)

    def test_empty_numeric_trace_on_date_axis_does_not_convert_timestamps_to_float(self):
        dates = pd.date_range("2025-01-01", periods=3)
        figure = go.Figure(go.Scatter(x=dates, y=[100, 0, 1]))
        figure.add_trace(go.Scatter(x=np.array([], dtype=float), y=np.array([], dtype=float)))
        figure.update_layout(yaxis_title="Z-Score", xaxis_range=[dates[1], dates[-1]])
        finish_zscore_figure(figure)
        self.assertEqual(tuple(figure.layout.yaxis.range), (-2.32, 2.32))

    def test_epoch_milliseconds_with_timestamp_bounds(self):
        dates = pd.date_range("2025-01-01", periods=3)
        milliseconds = [date.timestamp() * 1000 for date in dates]
        figure = go.Figure(go.Scatter(x=milliseconds, y=[100, 0, 1]))
        figure.update_layout(yaxis_title="Z-Score", xaxis_type="date", xaxis_range=[dates[1], dates[-1]])
        finish_zscore_figure(figure)
        self.assertEqual(tuple(figure.layout.yaxis.range), (-2.32, 2.32))

    def test_numeric_horizon_axis_still_uses_numeric_bounds(self):
        figure = go.Figure(go.Scatter(x=[3, 20, 60], y=[100, 0, 1]))
        figure.update_layout(yaxis_title="Z-Score", xaxis_range=[20, 60])
        finish_zscore_figure(figure)
        self.assertEqual(tuple(figure.layout.yaxis.range), (-2.32, 2.32))

    def test_playback_cache_separates_baselines_and_references_are_fixed(self):
        namespace = dashboard_functions(
            "_block18_prepare_window_diagnostics_playback",
            "_block18_diagnostics_asof_row",
            "_block18_window_diagnostics_trace_updates",
            namespace={
                "pd": pd, "np": np, "ticker_str": "TEST",
                "store_bounded": store_bounded,
                "_block18_playback_date": lambda position: (self.series.index[position], position),
                "_block18_add_spline_updates": lambda updates, *args: updates,
                "_block18_add_horizon_derivative_updates": lambda updates, *args: updates,
                "_block18_add_horizon_extrema_updates": lambda updates, *args: updates,
                "_block18_add_horizon_transition_updates": lambda updates, *args: updates,
                "_block18_normalize_ratio_type": lambda value: value,
                "_block18_ratio_label": lambda value: value.title(),
                "_ensure_momentum_diagnostics_ratio": lambda *args: None,
                "_zscore_baseline_key": baseline_key, "_normalize_zscore": normalize,
                "block18_display_range_offsets": OFFSETS,
                "block18_window_diagnostics_playback_cache": {},
                "momentum_diagnostics_contexts_by_ratio": {
                    "volatility": {"TEST": {"ratio_table": self.series.to_frame(20)}},
                },
            },
        )
        build = namespace["_block18_prepare_window_diagnostics_playback"]
        full = build("volatility")["TEST"]
        BASELINE.set(("trailing", 60, "1y"))
        trailing = build("volatility")["TEST"]
        self.assertNotAlmostEqual(full["ratio_zscore"].iloc[-1, 0], trailing["ratio_zscore"].iloc[-1, 0])
        self.assertIsInstance(trailing["ratio_reference_mean"], pd.Series)
        self.assertIsInstance(trailing["ratio_reference_std"], pd.Series)
        self.assertTrue((trailing["ratio_reference_mean"] == 0).all())
        self.assertTrue((trailing["ratio_reference_std"] == 1).all())
        updates, _, _ = namespace["_block18_window_diagnostics_trace_updates"](
            400, [], ratio_type="volatility",
        )
        self.assertEqual(updates["Historical Mean Volatility Z-Score"].tolist(), [0.0])
        self.assertEqual(updates["Historical Volatility Z-Score +2 Std Dev"].tolist(), [2.0])
        self.assertEqual(updates["Historical Volatility Z-Score -2 Std Dev"].tolist(), [-2.0])
        self.assertAlmostEqual(updates["Current Volatility Z-Score"].iloc[0], trailing["ratio_zscore"].iloc[400, 0])

    def test_comparison_view_has_consistent_zones_and_reference_lines(self):
        from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency.benchmark_zscore_detail import plot_benchmark_zscore_detail
        score = normalize(self.series, OFFSETS)
        payload = {"SPY": {"custom": {
            "asset": score, "benchmark": score, "asset_ratio": self.series,
            "benchmark_ratio": self.series, "ratio_spread": score,
        }}}
        figure = plot_benchmark_zscore_detail(payload, ["SPY"], {"custom": 20}, ratio_label="Volatility")
        finish_zscore_figure(figure)
        labels = [annotation.text for annotation in figure.layout.annotations]
        self.assertEqual(labels.count("Low"), 2)
        self.assertEqual(labels.count("Typical"), 2)
        self.assertEqual(labels.count("High"), 2)
        for axis in ("y", "y2"):
            references = {
                float(trace.y[0]) for trace in figure.data
                if trace.yaxis == axis and len(trace.y) and len(set(trace.y)) == 1
            }
            self.assertEqual(references, {-2.0, -1.0, 0.0, 1.0, 2.0})

    def test_playback_patch_updates_profile_axis(self):
        from dash import Patch
        namespace = dashboard_functions("_block18_patch_functional_horizon_playback", namespace={
            "Patch": Patch, "np": np, "pd": pd,
            "block18_functional_playback_plan_cache": {"test": {
                "trace_targets": {"Current Volatility Z-Score": [{
                    "index": 0, "horizons": [20, 60], "yaxis": "y", "visible": True,
                }]}, "segment_targets": [], "annotation_targets": [],
            }},
            "_block18_window_diagnostics_trace_updates": lambda *args: (
                {"Current Volatility Z-Score": pd.Series([0.2, 8.0], index=[20, 60])},
                pd.Timestamp("2026-01-01"), 1,
            ),
            "_block18_values_at_horizons": lambda values, horizons: values.reindex(horizons).tolist(),
            "_block18_normalize_diagnostics_window_range": lambda value: value,
        })
        patch, _ = namespace["_block18_patch_functional_horizon_playback"]("test", 1, [], [20, 60])
        updates = patch.to_plotly_json()["operations"]
        axis_range = next(item["params"]["value"] for item in updates if item["location"] == ["layout", "yaxis", "range"])
        self.assertGreater(axis_range[1], 8)
        self.assertLess(axis_range[0], -2)

    def test_dash_callback_argument_order_and_context_reset(self):
        # Exercise Dash registration and the actual wrapped callback, with both
        # an existing lookback Input and an injected lookback State.
        for has_lookback in (True, False):
            app = Dash(__name__)
            namespace = dashboard_functions("_block18_baseline_callback", namespace={
                "Input": Input, "Output": Output, "State": State,
                "block18_dash_app": app, "_zscore_baseline": BASELINE,
            })
            dependencies = [Output("result", "children"), Input("metric", "value")]
            if has_lookback:
                dependencies.append(Input("shared-display-range-dropdown", "value"))
            dependencies.append(State("window", "value"))
            namespace["_block18_baseline_callback"](*dependencies)(lambda *args: (args, BASELINE.get()))
            registered = next(iter(app.callback_map.values()))
            callback = registered["callback"].__wrapped__
            args = ["volatility"] + (["1y"] if has_lookback else []) + ["trailing", 60, 20] + ([] if has_lookback else ["1y"])
            original, baseline = callback(*args)
            self.assertEqual(original, ("volatility", "1y", 20) if has_lookback else ("volatility", 20))
            self.assertEqual(baseline, ("trailing", 60, "1y"))
            self.assertEqual(BASELINE.get(), ("full", 252, "3y"))


class CacheTests(unittest.TestCase):
    def test_capacity_evicts_only_oldest_entry(self):
        cache = {"oldest": 1, "recent": 2}
        store_bounded(cache, "new", 3, limit=2)
        self.assertEqual(cache, {"recent": 2, "new": 3})

    def test_replacement_keeps_other_entries_and_refreshes_order(self):
        cache = {"first": 1, "second": 2}
        store_bounded(cache, "first", 10, limit=2)
        self.assertEqual(list(cache), ["second", "first"])
        store_bounded(cache, "third", 3, limit=2)
        self.assertEqual(cache, {"first": 10, "third": 3})

    def test_invalid_limit_does_not_mutate_cache(self):
        cache = {"existing": 1}
        with self.assertRaises(ValueError):
            store_bounded(cache, "existing", 2, limit=0)
        self.assertEqual(cache, {"existing": 1})


if __name__ == "__main__":
    unittest.main()

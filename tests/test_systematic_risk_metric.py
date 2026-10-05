"""Check selected-metric routing without fetching market data."""

import ast
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.web.dashboard_cache import store_bounded
from test_momentum_zscore_baseline import ROOT, dashboard_functions


class SystematicRiskMetricTests(unittest.TestCase):
    def test_metric_values_labels_and_beta_panels_follow_selection(self):
        dates = pd.bdate_range("2025-01-01", periods=80)
        prices = pd.Series(100 * np.exp(np.cumsum(0.001 + np.sin(np.arange(80)) * 0.01)), index=dates)
        calls = []
        metrics = {
            "volatility": pd.Series(np.linspace(0.1, 0.3, 80), index=dates),
            "sharpe": pd.Series(np.sin(np.arange(80) / 5), index=dates),
            "treynor::SPY": pd.Series(np.cos(np.arange(80) / 7), index=dates),
        }
        labels = {"volatility": "Volatility", "sharpe": "Sharpe", "treynor::SPY": "Treynor vs SPY"}
        def rolling(close, window, metric):
            calls.append((window, metric))
            return metrics[metric]
        ns = dashboard_functions("_block18_systematic_risk_figure", namespace={
            "pd": pd, "np": np, "go": go, "make_subplots": make_subplots,
            "ticker_str": "TEST", "annualization_factor": 252,
            "asset_history": prices.to_frame("Close"),
            "benchmark_data": {"SPY": prices.to_frame("Close"), "QQQ": (prices * 2).to_frame("Close")},
            "risk_free_daily_rate": pd.Series(0.0001, index=dates),
            "_block18_validate_window": int,
            "_block18_normalize_benchmark_selection": lambda value: value,
            "_block18_normalize_ratio_type": lambda value: value,
            "_block18_ratio_label": labels.__getitem__,
            "block18_benchmark_profile_palette": ["blue", "orange"],
            "rolling_ratio_series": rolling,
            "zscore_or_empty": lambda values: (values - values.mean()) / values.std(),
        })
        for metric, values in metrics.items():
            figure = ns["_block18_systematic_risk_figure"](20, ["SPY", "QQQ"], metric)
            top = [trace for trace in figure.data if getattr(trace, "yaxis", None) == "y"]
            self.assertEqual(len(top), 1)
            np.testing.assert_allclose(top[0].y, (values - values.mean()) / values.std())
            np.testing.assert_allclose(np.asarray(top[0].customdata)[:, 0], values)
            self.assertIn(labels[metric], top[0].name)
            self.assertIn(labels[metric], figure.layout.title.text)
            betas = [trace for trace in figure.data if getattr(trace, "yaxis", None) == "y2"]
            self.assertEqual(len(betas), 2)
            for trace in betas:
                np.testing.assert_allclose(np.asarray(trace.y)[19:], 1.0)
            table = next(trace for trace in figure.data if trace.type == "table")
            self.assertIn(f"Raw {labels[metric]}", table.header.values)
            self.assertEqual(table.cells.values[4][0], f"{values.iloc[-1]:.3f}")
        self.assertEqual(calls, [(20, key) for key in metrics])

    def test_metric_is_forwarded_and_cache_does_not_reuse_other_metric(self):
        tree = ast.parse((ROOT / "apps/web/momentum_dashboard.py").read_text(encoding="utf-8"))
        names = {"_block18_build_tab_figure", "_block18_cached_tab_figure"}
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                for default in node.args.defaults:
                    if isinstance(default, ast.Name):
                        ns[default.id] = None
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "block18_ratio_dependent_tabs" for t in node.targets):
                ns["block18_ratio_dependent_tabs"] = ast.literal_eval(node.value)
        calls = []
        def build(window, benchmarks, metric):
            calls.append(metric)
            return go.Figure(go.Scatter(y=[1], name=metric))
        ns.update({"go": go, "store_bounded": store_bounded,
                   "_block18_normalize_ratio_type": lambda value: value,
                   "_zscore_baseline_key": lambda: ("full", None, None),
                   "block18_base_figure_cache": {}, "_block18_systematic_risk_figure": build})
        dashboard_functions(*names, namespace=ns)
        cached = ns["_block18_cached_tab_figure"]
        for metric in ("sharpe", "volatility", "sharpe"):
            figure = cached("systematic_risk", 20, "1y", ["SPY"], ratio_type=metric)
            self.assertEqual(figure.data[0].name, metric)
        self.assertEqual(calls, ["sharpe", "volatility"])


if __name__ == "__main__":
    unittest.main()

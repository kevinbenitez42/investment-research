"""The compact controls preserve every benchmark-qualified metric key."""
import ast
import unittest

from dash import Dash, Input, Output, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apps.web.metric_comparison import metric_keys, combine_reference_figures
from test_momentum_zscore_baseline import ROOT, dashboard_functions


class MetricSelectionTests(unittest.TestCase):
    def setUp(self):
        self.benchmarks = ['SPY', 'Consumer Staples - Sector', 'Personal Care Products - Sub-Industry']
        self.resolve = dashboard_functions('_resolve_metric_selection', namespace={
            'benchmark_order': self.benchmarks, 'no_update': no_update, 'metric_keys': metric_keys,
        })['_resolve_metric_selection']

    def test_all_metric_reference_combinations(self):
        for metric in ['correlation', 'information', 'appraisal', 'treynor']:
            for reference in self.benchmarks:
                key, style = self.resolve(metric, reference)
                self.assertEqual(key, f'{metric}::{reference}')
                self.assertEqual(style['display'], 'block')

    def test_sharpe_sortino_ignore_retained_reference(self):
        for metric in ['sharpe', 'sortino']:
            for reference in [None, *self.benchmarks]:
                self.assertEqual(self.resolve(metric, reference), (metric, {'display': 'none'}))

    def test_missing_reference_does_not_send_invalid_metric_to_charts(self):
        key, style = self.resolve('information', None)
        self.assertIs(key, no_update)
        self.assertEqual(style['display'], 'block')

    def test_multiple_references_preserved_without_duplicates(self):
        value, _ = self.resolve('information', ['SPY', self.benchmarks[1], 'SPY'])
        self.assertEqual(value, ['information::SPY', f'information::{self.benchmarks[1]}'])

    def test_overlay_preserves_both_reference_series(self):
        figures = [go.Figure(go.Scatter(x=[1,2], y=y, name='Asset')) for y in ([1,2], [10,20])]
        result = combine_reference_figures(figures, ['SPY','Sector'])
        self.assertEqual(len(result.data), 2)
        self.assertIn('SPY', result.data[0].name)
        self.assertIn('Sector', result.data[1].name)
        self.assertEqual(list(result.data[1].y), [10,20])
        self.assertTrue(result.layout.yaxis.autorange)

    def test_passage_tables_and_surfaces_have_separate_domains(self):
        figure = make_subplots(rows=2, cols=1, specs=[[{'type':'xy'}],[{'type':'table'}]])
        figure.add_trace(go.Scatter(x=[1,2],y=[1,2]),row=1,col=1)
        figure.add_trace(go.Table(header={'values':['Events']}, cells={'values':[[3]]}),row=2,col=1)
        result = combine_reference_figures([figure,figure], ['SPY','Sector'])
        self.assertEqual(len(result.data), 4)
        self.assertNotEqual(result.data[0].yaxis, result.data[2].yaxis)
        self.assertGreater(result.data[1].domain.y[0], result.data[3].domain.y[1])
        surface = go.Figure(go.Surface(z=[[1,2],[3,4]]))
        result = combine_reference_figures([surface,surface], ['SPY','Sector'])
        self.assertNotEqual(result.data[0].scene, result.data[1].scene)
        self.assertGreater(result.layout.scene.domain.y[0], result.layout.scene2.domain.y[1])

    def test_risk_router_renders_every_selected_reference(self):
        calls = []
        def risk(*args):
            calls.append(args[3])
            figure = go.Figure(go.Scatter(x=[1], y=[len(calls)], name=args[3]))
            return (figure, figure, figure, figure)
        ns = dashboard_functions('_block18_dashboard_outputs', namespace={
            'go':go, 'no_update':no_update, 'metric_keys':metric_keys,
            'combine_reference_figures':combine_reference_figures,
            'block18_metric_graph_specs':[], 'block18_ratio_dependent_tabs':{'risk'},
            '_block18_ratio_label':lambda key:key,
            '_update_risk_compounding_v2_view':risk,
        })
        def main(active_tab, ratio_type, rendered_tab, _n_clicks, display_range_value,
                 selected_benchmarks, first_passage_start_sign, first_passage_start_condition,
                 first_passage_threshold, first_passage_end_sign, first_passage_end_condition,
                 first_passage_end_threshold, window):
            return no_update, no_update, ratio_type, False
        import inspect
        arguments = dict.fromkeys(inspect.signature(main).parameters)
        arguments.update(active_tab='risk_compounding_v2', ratio_type=['information::SPY','information::Sector'])
        result = ns['_block18_dashboard_outputs'](main)(**arguments)
        self.assertEqual(calls, ['information::SPY','information::Sector'])
        self.assertEqual(len(result[4].data), 2)
        self.assertEqual(len(result[7].data), 1)

    def test_six_options_and_registered_callback(self):
        tree = ast.parse((ROOT/'apps/web/momentum_dashboard.py').read_text(encoding='utf-8'))
        options = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'block18_ratio_options' for t in n.targets))
        ns = {'benchmark_order': self.benchmarks}
        exec(compile(ast.Module(body=[options], type_ignores=[]), '<options>', 'exec'), ns)
        self.assertEqual(len(ns['block18_ratio_options']), 6)
        ns['benchmark_order'] = []
        exec(compile(ast.Module(body=[options], type_ignores=[]), '<options>', 'exec'), ns)
        self.assertTrue(all(o['disabled'] for o in ns['block18_ratio_options'][2:]))
        app = Dash(__name__)
        ns.update(block18_dash_app=app, Input=Input, Output=Output, no_update=no_update)
        callback = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_resolve_metric_selection')
        exec(compile(ast.Module(body=[callback], type_ignores=[]), '<callback>', 'exec'), ns)
        self.assertEqual(len(app.callback_map), 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Input':
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'shared-ratio-dropdown':
                    self.assertEqual(node.args[1].value, 'data')


if __name__ == '__main__':
    unittest.main()

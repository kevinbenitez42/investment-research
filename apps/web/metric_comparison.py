"""Compose separately calculated reference metrics without averaging them."""
import copy
import json
import re

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


def metric_keys(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    return list(dict.fromkeys(item for item in values if item))


def combine_reference_figures(figures, labels):
    """Overlay compatible panels, retaining reference names and table rows."""
    if any(trace.type in {'surface', 'heatmap', 'table', 'scatter3d'}
           for figure in figures for trace in figure.data):
        return stack_reference_figures(figures, labels)
    result = go.Figure(figures[0])
    result.data = ()
    seen = set()
    tables = {}
    dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
    for reference_index, (figure, label) in enumerate(zip(figures, labels)):
        for original in figure.data:
            trace = copy.deepcopy(original)
            identity = json.dumps(trace.to_plotly_json(), cls=PlotlyJSONEncoder, sort_keys=True)
            if trace.type == 'table':
                # Preserve each reference's passage statistics in the same table.
                key = json.dumps(trace.domain.to_plotly_json(), sort_keys=True)
                columns = [list(column) for column in trace.cells.values]
                if columns:
                    columns[0] = [f'{label} · {value}' for value in columns[0]]
                if key in tables:
                    table = tables[key]
                    table.cells.values = [list(a) + list(b) for a, b in zip(table.cells.values, columns)]
                else:
                    trace.cells.values = columns
                    result.add_trace(trace)
                    tables[key] = result.data[-1]
                continue
            # Compounding traces do not depend on the metric reference.
            if identity in seen:
                continue
            seen.add(identity)
            trace.name = f'{trace.name or trace.type} | {label}'
            if 'legendgroup' in trace._valid_props:
                trace.legendgroup = f'{label}::{trace.legendgroup or trace.name}'
            if trace.type in {'scatter', 'scattergl'} and reference_index:
                trace.line.dash = dashes[reference_index % len(dashes)]
            result.add_trace(trace)
    # Keep all threshold/event annotations, but do not repeat subplot headings.
    for field in ('shapes', 'annotations'):
        items, identities = [], set()
        for figure_index, figure in enumerate(figures):
            for item in getattr(figure.layout, field) or ():
                payload = item.to_plotly_json()
                if (field == 'annotations' and figure_index and
                        payload.get('xref') == 'paper' and payload.get('yref') == 'paper'):
                    continue
                identity = json.dumps(payload, cls=PlotlyJSONEncoder, sort_keys=True)
                if identity not in identities:
                    identities.add(identity)
                    items.append(payload)
        setattr(result.layout, field, items)
    result.layout.title.text = 'Reference comparison: ' + ' | '.join(labels)
    # Restyle menus contain trace indices from individual figures; global Dash
    # controls remain authoritative for a composed comparison.
    result.layout.updatemenus = ()
    result.layout.showlegend = True
    result.update_yaxes(autorange=True)
    return result


def stack_reference_figures(figures, labels):
    """Give tables and opaque heatmaps/surfaces separate panels in one figure."""
    result = go.Figure()
    count = len(figures)
    counters = {}
    annotations, shapes = [], []
    for i, (figure, label) in enumerate(zip(figures, labels)):
        layout = figure.layout.to_plotly_json()
        for trace in figure.data:
            if trace.type in {'surface', 'scatter3d'}:
                layout.setdefault(trace.scene or 'scene', {})
            if trace.type in {'scatter', 'scattergl', 'bar', 'histogram', 'heatmap'}:
                layout.setdefault('xaxis' + (trace.xaxis or 'x')[1:], {})
                layout.setdefault('yaxis' + (trace.yaxis or 'y')[1:], {})
        low, high = (count - i - 1) / count, (count - i) / count - 0.035 / count
        def y(value):
            return low + float(value) * (high - low)
        mapping = {}
        for key in layout:
            match = re.fullmatch(r'(xaxis|yaxis|scene|coloraxis)(\d*)', key)
            if match:
                prefix = match[1]
                counters[prefix] = counters.get(prefix, 0) + 1
                mapping[key] = prefix + (str(counters[prefix]) if counters[prefix] > 1 else '')
        def ref(value):
            if not isinstance(value, str):
                return value
            base, *tail = value.split(' ')
            key = ('xaxis' + base[1:]) if re.fullmatch(r'x\d*', base) else (
                ('yaxis' + base[1:]) if re.fullmatch(r'y\d*', base) else base)
            translated = mapping.get(key, key)
            if translated.startswith('xaxis'):
                translated = 'x' + translated[5:]
            elif translated.startswith('yaxis'):
                translated = 'y' + translated[5:]
            return ' '.join([translated, *tail])
        for old, new in mapping.items():
            spec = copy.deepcopy(layout[old])
            if old.startswith('xaxis'):
                spec.setdefault('anchor', 'y' + old[5:])
            elif old.startswith('yaxis'):
                spec.setdefault('anchor', 'x' + old[5:])
            if old.startswith('yaxis'):
                spec['domain'] = [y(v) for v in spec.get('domain', [0, 1])]
            if old.startswith('scene'):
                domain = spec.setdefault('domain', {})
                domain['y'] = [y(v) for v in domain.get('y', [0, 1])]
            for key in ('anchor', 'matches', 'overlaying', 'scaleanchor'):
                if key in spec:
                    spec[key] = ref(spec[key])
            if 'colorbar' in spec:
                spec['colorbar']['y'] = (low + high) / 2
                spec['colorbar']['len'] = high - low
            result.layout[new] = spec
        for original in figure.data:
            trace = original.to_plotly_json()
            for key in ('xaxis', 'yaxis', 'scene', 'coloraxis'):
                if key in trace:
                    trace[key] = ref(trace[key])
            if trace['type'] in {'scatter', 'scattergl', 'bar', 'histogram', 'heatmap'}:
                trace.setdefault('xaxis', ref('x'))
                trace.setdefault('yaxis', ref('y'))
            if trace['type'] in {'surface', 'scatter3d'}:
                trace.setdefault('scene', ref('scene'))
            if 'domain' in trace or trace['type'] == 'table':
                domain = trace.setdefault('domain', {})
                domain['y'] = [y(v) for v in domain.get('y', [0, 1])]
            if 'colorbar' in trace:
                trace['colorbar'].update(y=(low + high)/2, len=high-low)
            trace['name'] = f"{trace.get('name', trace['type'])} | {label}"
            result.add_trace(trace)
        for field, destination in [('annotations', annotations), ('shapes', shapes)]:
            for item in layout.get(field, []):
                item = copy.deepcopy(item)
                item['xref'] = ref(item.get('xref', 'paper' if field == 'annotations' else 'x'))
                item['yref'] = ref(item.get('yref', 'paper' if field == 'annotations' else 'y'))
                if item['yref'] == 'paper':
                    for key in ('y', 'y0', 'y1'):
                        if key in item:
                            item[key] = y(item[key])
                destination.append(item)
        annotations.append(dict(text=label, x=0, y=high, xref='paper', yref='paper',
                                xanchor='left', yanchor='bottom', showarrow=False))
    result.update_layout(template=figures[0].layout.template,
                         height=sum(f.layout.height or 600 for f in figures),
                         title='Reference comparison', annotations=annotations, shapes=shapes,
                         margin=dict(l=80, r=60, t=100, b=50), showlegend=True)
    return result

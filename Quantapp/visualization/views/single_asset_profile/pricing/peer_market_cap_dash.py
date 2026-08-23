"""Dash helpers for peer market-cap treemap exploration."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import pandas as pd


TABLE_COLUMNS = [
    "Symbol",
    "Market Cap",
    "Market-Cap Weight",
    "Sector",
    "Industry",
    "Industry Group",
    "Sub-Industry",
]


def _format_market_cap_label(value) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def _format_weight_label(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2%}"


def create_peer_market_cap_dash_app(
    *,
    target_symbol: str,
    treemap_tables: Mapping[str, pd.DataFrame],
    company_tables: Mapping[str, pd.DataFrame],
    filter_labels: Iterable[str],
    treemap_trace_builder: Callable,
    treemap_nodes: pd.DataFrame,
):
    """Create a Dash app that filters the company table from treemap clicks.

    The app consumes already-built notebook tables. It does not fetch or query
    any market-cap data.
    """

    try:
        from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html
        from plotly import graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Dash is not installed in this kernel. Install the dashboard extra, "
            "for example `pip install -e .[dashboard]`, then rerun this cell."
        ) from exc

    filter_labels = list(filter_labels)
    if not filter_labels:
        raise ValueError("At least one non-empty treemap filter is required.")

    node_metadata = treemap_nodes.drop_duplicates("Node ID").set_index("Node ID")
    filter_options = [{"label": label, "value": label} for label in filter_labels]
    default_filter = filter_options[0]["value"]

    def node_style(node_id: str):
        if node_id in node_metadata.index:
            return (
                str(node_metadata.at[node_id, "Color"]),
                str(node_metadata.at[node_id, "Text Color"]),
            )
        return "#0f172a", "#e2e8f0"

    def level_node_id(row: pd.Series, column_id: str) -> str:
        sector = str(row["Sector"])
        industry_group = str(row["Industry Group"])
        industry = str(row["Industry"])
        sub_industry = str(row["Sub-Industry"])
        node_ids = {
            "Sector": sector,
            "Industry Group": f"{sector} > {industry_group}",
            "Industry": f"{sector} > {industry_group} > {industry}",
            "Sub-Industry": f"{sector} > {industry_group} > {industry} > {sub_industry}",
        }
        return node_ids[column_id]

    def treemap_figure(filter_label: str):
        filtered_nodes = treemap_tables.get(filter_label, pd.DataFrame())
        if filtered_nodes.empty:
            figure = go.Figure()
            figure.add_annotation(text="No loaded market-cap data", showarrow=False)
        else:
            figure = go.Figure(treemap_trace_builder(filtered_nodes, filter_label, visible=True))
        figure.update_layout(
            title=f"{target_symbol} Sector Peer Treemap - {filter_label}",
            template="plotly_dark",
            paper_bgcolor="#0b0f14",
            plot_bgcolor="#0b0f14",
            font=dict(color="#f8fafc"),
            margin=dict(t=64, r=16, b=16, l=16),
            height=800,
            clickmode="event+select",
            uirevision=filter_label,
        )
        return figure

    def selection_parts(selected_node_id: str | None) -> list[str]:
        if not selected_node_id:
            return []
        return [part.strip() for part in str(selected_node_id).split(" > ") if part.strip()]

    def selected_table_column(selected_node_id: str | None) -> str | None:
        parts = selection_parts(selected_node_id)
        if not parts:
            return None
        level_columns = ["Sector", "Industry Group", "Industry", "Sub-Industry", "Symbol"]
        return level_columns[min(len(parts), len(level_columns)) - 1]

    def filter_company_table(company_table: pd.DataFrame, selected_node_id: str | None):
        filtered_table = company_table.copy()
        parts = selection_parts(selected_node_id)
        if not parts:
            return filtered_table, "Showing all companies in the selected capitalization filter."

        if len(parts) >= 1:
            filtered_table = filtered_table.loc[filtered_table["Sector"].astype(str).eq(parts[0])]
        if len(parts) >= 2:
            filtered_table = filtered_table.loc[filtered_table["Industry Group"].astype(str).eq(parts[1])]
        if len(parts) >= 3:
            filtered_table = filtered_table.loc[filtered_table["Industry"].astype(str).eq(parts[2])]
        if len(parts) >= 4:
            filtered_table = filtered_table.loc[filtered_table["Sub-Industry"].astype(str).eq(parts[3])]
        if len(parts) >= 5:
            filtered_table = filtered_table.loc[filtered_table["Symbol"].astype(str).eq(parts[4])]

        selected_label = parts[-1]
        selected_level = ["Sector", "Industry Group", "Industry", "Sub-Industry", "Company"][
            min(len(parts), 5) - 1
        ]
        return filtered_table, f"Filtered by {selected_level}: {selected_label}"

    def table_records_and_styles(filter_label: str, selected_node_id: str | None = None):
        company_table = company_tables.get(filter_label, pd.DataFrame()).copy()
        if company_table.empty:
            return [], [], "No companies with loaded market caps for this filter."

        company_table, selection_text = filter_company_table(company_table, selected_node_id)
        if company_table.empty:
            return [], [], f"{selection_text}; no companies remain in this capitalization filter."

        company_table["Latest Market Cap"] = pd.to_numeric(company_table["Latest Market Cap"], errors="coerce")
        company_table = company_table.dropna(subset=["Latest Market Cap"])
        company_table = company_table.loc[company_table["Latest Market Cap"].gt(0)].copy()
        if company_table.empty:
            return [], [], f"{selection_text}; no companies with positive market caps remain in this filter."

        selected_total_market_cap = company_table["Latest Market Cap"].sum()
        company_table["Market-Cap Weight"] = (
            company_table["Latest Market Cap"] / selected_total_market_cap
            if selected_total_market_cap > 0
            else pd.NA
        )
        company_table["Market Cap Label"] = company_table["Latest Market Cap"].map(_format_market_cap_label)
        company_table["Market-Cap Weight Label"] = company_table["Market-Cap Weight"].map(_format_weight_label)

        company_table = company_table.sort_values("Latest Market Cap", ascending=False).reset_index(drop=True)
        display_table = company_table.rename(
            columns={
                "Market Cap Label": "Market Cap",
                "Market-Cap Weight Label": "Market-Cap Weight",
            }
        )[TABLE_COLUMNS].copy()
        styles = [
            {"if": {"state": "active"}, "border": "1px solid #f8fafc"},
            {"if": {"state": "selected"}, "border": "1px solid #f8fafc"},
        ]
        selected_column = selected_table_column(selected_node_id)
        selected_parts = selection_parts(selected_node_id)
        for row_index, row in company_table.iterrows():
            company_fill = str(row.get("Color", "#0f172a"))
            company_text = str(row.get("Text Color", "#e2e8f0"))
            for column_id in ["Symbol", "Market Cap", "Market-Cap Weight"]:
                styles.append(
                    {
                        "if": {"row_index": row_index, "column_id": column_id},
                        "backgroundColor": company_fill,
                        "color": company_text,
                    }
                )
            for column_id in ["Sector", "Industry", "Industry Group", "Sub-Industry"]:
                level_fill, level_text = node_style(level_node_id(row, column_id))
                styles.append(
                    {
                        "if": {"row_index": row_index, "column_id": column_id},
                        "backgroundColor": level_fill,
                        "color": level_text,
                    }
                )

            if selected_column == "Symbol":
                for column_id in TABLE_COLUMNS:
                    styles.append(
                        {
                            "if": {"row_index": row_index, "column_id": column_id},
                            "border": "2px solid #f8fafc",
                            "fontWeight": "700",
                        }
                    )
            elif selected_column and selected_parts:
                styles.append(
                    {
                        "if": {"row_index": row_index, "column_id": selected_column},
                        "border": "2px solid #f8fafc",
                        "fontWeight": "700",
                    }
                )

        total_market_cap_label = _format_market_cap_label(selected_total_market_cap)
        summary = (
            f"{selection_text} Showing {len(display_table):,} companies. "
            f"Weight base: {total_market_cap_label}."
        )
        return display_table.to_dict("records"), styles, summary

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        id="dash-market-cap-filter",
                        options=filter_options,
                        value=default_filter,
                        clearable=False,
                        style={"width": "260px", "color": "#111827"},
                    ),
                    html.Button(
                        "Clear selection",
                        id="dash-clear-selection",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#1e293b",
                            "border": "1px solid #334155",
                            "color": "#f8fafc",
                            "height": "38px",
                            "padding": "0 14px",
                        },
                    ),
                    html.Div(id="dash-selection-summary", style={"color": "#cbd5e1", "paddingTop": "8px"}),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "center", "marginBottom": "10px"},
            ),
            dcc.Store(id="dash-selected-node"),
            dcc.Graph(
                id="dash-market-cap-treemap-graph",
                figure=treemap_figure(default_filter),
                config={"responsive": True, "displaylogo": False},
                style={"height": "800px"},
            ),
            dash_table.DataTable(
                id="dash-company-table",
                columns=[{"name": column, "id": column} for column in TABLE_COLUMNS],
                data=[],
                page_action="none",
                fixed_rows={"headers": True},
                style_table={"height": "520px", "overflowY": "auto", "border": "1px solid #1e293b"},
                style_header={
                    "backgroundColor": "#111827",
                    "color": "#f8fafc",
                    "border": "1px solid #334155",
                    "fontWeight": "600",
                    "textAlign": "left",
                },
                style_cell={
                    "backgroundColor": "#0f172a",
                    "color": "#e2e8f0",
                    "border": "1px solid #1e293b",
                    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
                    "fontSize": "12px",
                    "padding": "11px 9px",
                    "textAlign": "left",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Symbol"}, "width": "9%"},
                    {"if": {"column_id": "Market Cap"}, "width": "11%"},
                    {"if": {"column_id": "Market-Cap Weight"}, "width": "11%"},
                    {"if": {"column_id": "Sector"}, "width": "14%"},
                    {"if": {"column_id": "Industry"}, "width": "19%"},
                    {"if": {"column_id": "Industry Group"}, "width": "18%"},
                    {"if": {"column_id": "Sub-Industry"}, "width": "18%"},
                ],
            ),
        ],
        style={"backgroundColor": "#0b0f14", "padding": "16px"},
    )

    @app.callback(
        Output("dash-market-cap-treemap-graph", "figure"),
        Input("dash-market-cap-filter", "value"),
    )
    def update_treemap(filter_label: str):
        return treemap_figure(filter_label)

    @app.callback(
        Output("dash-selected-node", "data"),
        Input("dash-market-cap-treemap-graph", "clickData"),
        Input("dash-clear-selection", "n_clicks"),
        Input("dash-market-cap-filter", "value"),
        State("dash-selected-node", "data"),
    )
    def update_selected_node(click_data, clear_clicks, filter_label, selected_node_id):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if triggered in {"dash-clear-selection", "dash-market-cap-filter"}:
            return None
        if triggered == "dash-market-cap-treemap-graph" and click_data and click_data.get("points"):
            clicked_node_id = click_data["points"][0].get("id")
            if clicked_node_id == selected_node_id:
                return None
            return clicked_node_id
        return None

    @app.callback(
        Output("dash-company-table", "data"),
        Output("dash-company-table", "style_data_conditional"),
        Output("dash-selection-summary", "children"),
        Input("dash-market-cap-filter", "value"),
        Input("dash-selected-node", "data"),
    )
    def update_company_table(filter_label: str, selected_node_id: str | None):
        return table_records_and_styles(filter_label, selected_node_id)

    return app

from __future__ import annotations

import os

from dash import Dash, Input, Output, State, dcc, html

from Quantapp.workflows import RiskAnalysisConfig, build_risk_analysis_dashboard_payload


DEFAULT_CONFIG = RiskAnalysisConfig()
GRAPH_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

app = Dash(__name__, suppress_callback_exceptions=True, title="Risk Analysis Dashboard")
server = app.server


def _render_summary_cards(cards: list[dict[str, str]]) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(card["label"], className="summary-card__label"),
                    html.Div(card["value"], className="summary-card__value"),
                    html.Div(card["meta"], className="summary-card__meta"),
                ],
                className="summary-card",
            )
            for card in cards
        ],
        className="summary-grid",
    )


def _render_section(section: dict[str, object]) -> html.Div:
    notes = section.get("notes") or []
    cards = section.get("cards") or []
    return html.Div(
        [
            html.Div(
                [html.P(note, className="section-note") for note in notes],
                className="section-notes",
            )
            if notes
            else None,
            html.Div([_render_graph_card(card) for card in cards], className="graph-grid"),
        ],
        className="section-panel",
    )


def _render_graph_card(card: dict[str, object]) -> html.Section:
    figure = card["figure"]
    figure.update_layout(autosize=True)
    return html.Section(
        [
            html.Div(card["title"], className="graph-card__title"),
            dcc.Graph(figure=figure, config=GRAPH_CONFIG, responsive=True, className="graph-card__graph"),
        ],
        className="graph-card",
    )


def _render_warnings(warnings: list[str]) -> html.Div | None:
    if not warnings:
        return None
    return html.Div(
        [html.Div("Notes", className="warning-panel__title"), html.Ul([html.Li(item) for item in warnings])],
        className="warning-panel",
    )


def _build_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Single Asset Risk Dashboard", className="hero__title"),
                            html.P(
                                "Prototype ideas in the notebook, then use this app to explore the same sections in a responsive dashboard layout.",
                                className="hero__copy",
                            ),
                        ],
                        className="hero",
                    ),
                    html.Div(
                        [
                            _control("Ticker", dcc.Input(id="ticker-input", type="text", value=DEFAULT_CONFIG.ticker_str, debounce=True)),
                            _control("Benchmarks", dcc.Input(id="benchmark-input", type="text", value=", ".join(DEFAULT_CONFIG.benchmark_tickers), debounce=True)),
                            _control(
                                "Strategy",
                                dcc.Dropdown(
                                    id="strategy-input",
                                    options=[
                                        {"label": "Swing", "value": "swing"},
                                        {"label": "Position", "value": "position"},
                                        {"label": "Structural", "value": "structural"},
                                    ],
                                    value=DEFAULT_CONFIG.trading_strategy,
                                    clearable=False,
                                ),
                            ),
                            _control(
                                "History",
                                dcc.Dropdown(
                                    id="period-input",
                                    options=[{"label": label, "value": value} for label, value in [("5 Years", "5y"), ("10 Years", "10y"), ("20 Years", "20y"), ("Max", "max")]],
                                    value=DEFAULT_CONFIG.period,
                                    clearable=False,
                                ),
                            ),
                            _control("Interval", dcc.Dropdown(id="interval-input", options=[{"label": "Daily", "value": "1d"}], value=DEFAULT_CONFIG.interval, clearable=False)),
                            _control("Risk-Free", dcc.Input(id="risk-free-input", type="text", value=DEFAULT_CONFIG.risk_free_ticker, debounce=True)),
                            _control("Treasury Years", dcc.Input(id="treasury-years-input", type="number", min=1, step=1, value=DEFAULT_CONFIG.length_of_plots)),
                            html.Button("Run Analysis", id="run-analysis", n_clicks=0, className="run-button"),
                        ],
                        className="control-panel",
                    ),
                ],
                className="shell__top",
            ),
            dcc.Loading(html.Div(id="dashboard-content", className="dashboard-content"), type="default"),
        ],
        className="app-shell",
    )


def _control(label: str, component) -> html.Div:
    return html.Div([html.Label(label, className="control__label"), component], className="control")


app.layout = _build_layout()


@app.callback(
    Output("dashboard-content", "children"),
    Input("run-analysis", "n_clicks"),
    State("ticker-input", "value"),
    State("benchmark-input", "value"),
    State("strategy-input", "value"),
    State("period-input", "value"),
    State("interval-input", "value"),
    State("risk-free-input", "value"),
    State("treasury-years-input", "value"),
)
def render_dashboard(_, ticker, benchmarks, strategy, period, interval, risk_free_ticker, length_of_plots):
    config = RiskAnalysisConfig(
        ticker_str=str(ticker or DEFAULT_CONFIG.ticker_str).strip().upper(),
        benchmark_tickers=[item.strip().upper() for item in str(benchmarks or "").split(",") if item.strip()],
        trading_strategy=str(strategy or DEFAULT_CONFIG.trading_strategy).strip().lower(),
        period=str(period or DEFAULT_CONFIG.period),
        interval=str(interval or DEFAULT_CONFIG.interval),
        risk_free_ticker=str(risk_free_ticker or DEFAULT_CONFIG.risk_free_ticker).strip().upper(),
        length_of_plots=int(length_of_plots or DEFAULT_CONFIG.length_of_plots),
    )

    try:
        payload = build_risk_analysis_dashboard_payload(config)
    except Exception as exc:
        return html.Div(
            [
                html.Div("Dashboard build failed", className="error-panel__title"),
                html.Pre(str(exc), className="error-panel__body"),
            ],
            className="error-panel",
        )

    tabs = dcc.Tabs(
        value=payload["sections"][0]["id"] if payload["sections"] else None,
        children=[
            dcc.Tab(label=section["label"], value=section["id"], children=_render_section(section), className="dashboard-tab", selected_className="dashboard-tab dashboard-tab--selected")
            for section in payload["sections"]
        ],
        className="dashboard-tabs",
    )

    return html.Div(
        [
            html.H2(payload["title"], className="dashboard-title"),
            _render_summary_cards(payload["summary_cards"]),
            _render_warnings(payload["warnings"]),
            tabs,
        ]
    )


if __name__ == "__main__":
    app.run(
        debug=os.getenv("DASH_DEBUG", "1") == "1",
        host=os.getenv("DASH_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8050")),
    )

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator
import yfinance as yf

from Quantapp.analytics.helper import Helper
from Quantapp.analytics.models import Models
from Quantapp.analytics.momentum_analytics import MomentumAnalytics
from Quantapp.analytics.risk_distribution_analytics import RiskDistributionAnalytics
from Quantapp.analytics.risk_relative_analytics import RiskRelativeAnalytics
from Quantapp.analytics.rolling import Rolling
from Quantapp.analytics.series_transforms import SeriesTransforms
from Quantapp.data import (
    MacroDataClient,
    align_series_to_common_index,
    load_benchmark_data,
    normalize_benchmark_tickers,
)
from Quantapp.visualization import (
    BarChartPlotter,
    CandleStickPlotter,
    LineChartPlotter,
    Plotter,
    build_time_range_buttons,
)


TIMEFRAME_PROFILES = {
    "swing": {"short": 3, "mid": 9, "long": 21},
    "position": {"short": 21, "mid": 50, "long": 200},
    "structural": {"short": 200, "mid": 500, "long": 1000},
}


@dataclass(slots=True)
class RiskAnalysisConfig:
    ticker_str: str = "SLV"
    benchmark_tickers: tuple[str, ...] | list[str] = ("SPY",)
    period: str = "20y"
    interval: str = "1d"
    risk_free_ticker: str = "^IRX"
    length_of_plots: int = 20
    trading_strategy: str = "position"


def build_risk_analysis_dashboard_payload(config: RiskAnalysisConfig) -> dict[str, Any]:
    helper = Helper()
    rolling = Rolling()
    series_transforms = SeriesTransforms()
    momentum_analytics = MomentumAnalytics()
    risk_relative_analytics = RiskRelativeAnalytics()
    risk_distribution_analytics = RiskDistributionAnalytics()
    qp = Plotter()
    model = Models()
    line_chart_plotter = LineChartPlotter()
    candle_stick_plotter = CandleStickPlotter()
    bar_chart_plotter = BarChartPlotter()

    ticker_str = str(config.ticker_str).strip().upper()
    if not ticker_str:
        raise ValueError("ticker_str is required.")

    benchmark_tickers = _coerce_benchmark_tickers(config.benchmark_tickers)
    strategy = str(config.trading_strategy).strip().lower()
    if strategy not in TIMEFRAME_PROFILES:
        raise ValueError(
            f"Invalid trading_strategy '{config.trading_strategy}'. "
            f"Expected one of: {list(TIMEFRAME_PROFILES)}"
        )

    time_frame_map = TIMEFRAME_PROFILES[strategy]
    warnings: list[str] = []
    sections: list[dict[str, Any]] = []

    ticker = yf.Ticker(ticker_str).history(period=config.period, interval=config.interval)
    vix = yf.Ticker("^VIX").history(period=config.period, interval=config.interval)
    risk_free_proxy = yf.Ticker(config.risk_free_ticker).history(period=config.period, interval=config.interval)

    ticker = helper.simplify_datetime_index(ticker)
    vix = helper.simplify_datetime_index(vix)
    risk_free_proxy = helper.simplify_datetime_index(risk_free_proxy)

    if ticker.empty or "Close" not in ticker:
        raise ValueError(f"No price history available for {ticker_str}.")
    if risk_free_proxy.empty or "Close" not in risk_free_proxy:
        raise ValueError(f"No risk-free history available for {config.risk_free_ticker}.")

    normalized_benchmarks = normalize_benchmark_tickers(benchmark_tickers, ticker_str)
    benchmark_data, skipped_benchmarks = load_benchmark_data(
        normalized_benchmarks,
        config.period,
        config.interval,
        helper,
    )
    if skipped_benchmarks:
        warnings.append(f"Skipped benchmarks with no data: {', '.join(skipped_benchmarks)}")

    _, ticker, vix, benchmark_data = align_series_to_common_index(ticker, vix, benchmark_data)

    risk_free_annual_yield = risk_free_proxy["Close"].dropna().sort_index().div(100)
    risk_free_daily_rate = ((1 + risk_free_annual_yield) ** (1 / 252) - 1).shift(1)
    risk_free_daily_rate = risk_free_daily_rate.reindex(ticker.index).ffill()

    ticker_returns = {
        frequency: series_transforms.returns(ticker, frequency=frequency)
        for frequency in ("monthly", "weekly", "daily")
    }

    summary_cards = _build_summary_cards(
        ticker=ticker,
        ticker_str=ticker_str,
        benchmark_data=benchmark_data,
        strategy=strategy,
        time_frame_map=time_frame_map,
    )

    overview_cards = _build_overview_cards(
        ticker=ticker,
        ticker_str=ticker_str,
        qp=qp,
        rolling=rolling,
        line_chart_plotter=line_chart_plotter,
        risk_distribution_analytics=risk_distribution_analytics,
        candle_stick_plotter=candle_stick_plotter,
        ticker_returns=ticker_returns,
        bar_chart_plotter=bar_chart_plotter,
    )
    sections.append({"id": "overview", "label": "Overview", "notes": [], "cards": overview_cards})

    treasury_cards, treasury_notes, treasury_warnings = _build_treasury_cards(
        ticker_str=ticker_str,
        benchmark_tickers=benchmark_tickers,
        interval=config.interval,
        length_of_plots=int(config.length_of_plots),
        helper=helper,
    )
    warnings.extend(treasury_warnings)
    if treasury_cards:
        sections.append({"id": "treasury", "label": "Treasury", "notes": treasury_notes, "cards": treasury_cards})

    momentum_cards = _build_momentum_cards(
        ticker=ticker,
        ticker_str=ticker_str,
        momentum_analytics=momentum_analytics,
        line_chart_plotter=line_chart_plotter,
        risk_free_daily_rate=risk_free_daily_rate,
    )
    sections.append({"id": "momentum", "label": "Momentum", "notes": [], "cards": momentum_cards})

    relative_cards, relative_notes, relative_warnings = _build_relative_risk_cards(
        ticker=ticker,
        ticker_str=ticker_str,
        benchmark_data=benchmark_data,
        time_frame_map=time_frame_map,
        risk_free_daily_rate=risk_free_daily_rate,
        rolling=rolling,
        risk_relative_analytics=risk_relative_analytics,
        risk_distribution_analytics=risk_distribution_analytics,
        line_chart_plotter=line_chart_plotter,
    )
    warnings.extend(relative_warnings)
    sections.append({"id": "relative-risk", "label": "Relative Risk", "notes": relative_notes, "cards": relative_cards})

    factor_cards, factor_notes, factor_warnings = _build_factor_cards(
        ticker_str=ticker_str,
        model=model,
        qp=qp,
    )
    warnings.extend(factor_warnings)
    if factor_cards:
        sections.append({"id": "factors", "label": "Factors", "notes": factor_notes, "cards": factor_cards})

    return {
        "title": f"{ticker_str} Risk Analysis Dashboard",
        "summary_cards": summary_cards,
        "warnings": warnings,
        "sections": [section for section in sections if section["cards"]],
    }


def _coerce_benchmark_tickers(values: tuple[str, ...] | list[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [item.strip().upper() for item in values.split(",") if item.strip()]
    return [str(item).strip().upper() for item in values if str(item).strip()]


def _build_summary_cards(
    *,
    ticker: pd.DataFrame,
    ticker_str: str,
    benchmark_data: dict[str, pd.DataFrame],
    strategy: str,
    time_frame_map: dict[str, int],
) -> list[dict[str, str]]:
    last_close = ticker["Close"].iloc[-1]
    latest_date = ticker.index[-1]
    first_date = ticker.index[0]
    benchmark_value = ", ".join(benchmark_data.keys()) if benchmark_data else "None"
    timeframe_value = " / ".join(str(time_frame_map[key]) for key in ("short", "mid", "long"))

    return [
        {"label": "Ticker", "value": ticker_str, "meta": f"{first_date:%Y-%m-%d} to {latest_date:%Y-%m-%d}"},
        {"label": "Last Close", "value": f"{last_close:,.2f}", "meta": f"As of {latest_date:%Y-%m-%d}"},
        {"label": "Benchmarks", "value": benchmark_value, "meta": f"{len(benchmark_data)} loaded"},
        {"label": "Strategy", "value": strategy.title(), "meta": f"Windows: {timeframe_value} days"},
        {"label": "Rows", "value": f"{len(ticker.index):,}", "meta": "Aligned daily observations"},
    ]


def _build_overview_cards(
    *,
    ticker: pd.DataFrame,
    ticker_str: str,
    qp: Plotter,
    rolling: Rolling,
    line_chart_plotter: LineChartPlotter,
    risk_distribution_analytics: RiskDistributionAnalytics,
    candle_stick_plotter: CandleStickPlotter,
    ticker_returns: dict[str, pd.Series],
    bar_chart_plotter: BarChartPlotter,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    candlestick_period = "2y"
    candlestick_view_options = [
        ("3 Months", pd.DateOffset(months=3)),
        ("6 Months", pd.DateOffset(months=6)),
        ("1 Year", pd.DateOffset(years=1)),
        ("2 Years", pd.DateOffset(years=2)),
        ("Max", None),
    ]
    candlestick_view_data = ticker.last(candlestick_period)
    candlestick_view_end = candlestick_view_data.index.max()
    candlestick_view_start = candlestick_view_data.index.min()

    def build_candlestick_range(offset=None):
        start = candlestick_view_start if offset is None else max(candlestick_view_start, candlestick_view_end - offset)
        span_days = max((candlestick_view_end - start).days, 1)
        padding_days = max(10, int(span_days * 0.08))
        return [start, candlestick_view_end + pd.Timedelta(days=padding_days)]

    candlestick_fig = candle_stick_plotter.create_candlestick_fig(
        ticker,
        period=candlestick_period,
        bollinger_window=50,
        title="Candlestick with 50-Period Bollinger Bands",
    )
    candlestick_fig.update_layout(
        height=1000,
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=label, method="relayout", args=[{"xaxis.range": build_candlestick_range(offset)}])
                    for label, offset in candlestick_view_options
                ],
                direction="down",
                showactive=True,
                active=2,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
    )
    candlestick_fig.update_xaxes(range=build_candlestick_range(pd.DateOffset(years=1)))
    candlestick_fig.add_annotation(
        text="View timeframe",
        x=0.0,
        xref="paper",
        y=1.145,
        yref="paper",
        showarrow=False,
        xanchor="left",
    )
    cards.append({"title": "Regime Candlestick", "figure": candlestick_fig})

    percentage_drop_fig = qp.plot_percentage_drop(
        ticker,
        n=int(252 / 2),
        title=f"{ticker_str} Percentage Drop from Highest Peak",
        show=False,
    )
    cards.append({"title": "Peak Pullback", "figure": percentage_drop_fig})

    drawdown_context = risk_distribution_analytics.build_risk_distribution_context(
        close_series=ticker["Close"],
        windows=[21, 50, 200],
        default_window=200,
    )
    drawdown_fig = line_chart_plotter.plot_rolling_max_drawdown(
        metrics_by_window=drawdown_context["metrics_by_window"],
        window_options=drawdown_context["windows"],
        default_window=drawdown_context["default_window"],
        ticker_label=ticker_str,
    )
    cards.append({"title": "Rolling Max Drawdown", "figure": drawdown_fig})

    vix_fix_fig = qp.plot_series_with_stdev_bands(
        rolling.vix_fix(ticker),
        title="VIX Fix with Mean and Standard Deviations",
        stdev_values=[-0.5, 0.5, 1.5, 3],
        show=False,
    )
    cards.append({"title": "VIX Fix Proxy", "figure": vix_fix_fig})

    cards.append({"title": "Monthly Seasonality", "figure": bar_chart_plotter.create_seasonality_fig(ticker_returns["monthly"], f"Monthly Seasonality: {ticker_str}", "monthly")})
    cards.append({"title": "Weekly Seasonality", "figure": bar_chart_plotter.create_seasonality_fig(ticker_returns["weekly"], f"Weekly Seasonality: {ticker_str}", "weekly")})
    cards.append({"title": "Daily Seasonality", "figure": bar_chart_plotter.create_seasonality_fig(ticker_returns["daily"], f"Daily Seasonality: {ticker_str}", "daily")})

    return cards


def _build_momentum_cards(
    *,
    ticker: pd.DataFrame,
    ticker_str: str,
    momentum_analytics: MomentumAnalytics,
    line_chart_plotter: LineChartPlotter,
    risk_free_daily_rate: pd.Series,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    momentum_diagnostics_context = momentum_analytics.build_momentum_window_diagnostics_context(
        close_series=ticker["Close"],
        window_sizes=list(range(3, 201)),
        highlight_windows=(7, 21, 50, 200),
        surface_years=10,
        risk_free_rate=risk_free_daily_rate,
    )
    momentum_diagnostic_figs = line_chart_plotter.plot_momentum_window_diagnostics(
        diagnostics_context=momentum_diagnostics_context,
        ticker_label=ticker_str,
    )

    title_map = {
        "optimal_window": "Optimal Momentum Window",
        "optimal_window_histogram": "Optimal Window Histogram",
        "sharpe_mean_median": "Sharpe Mean vs Median",
        "volatility_mean_median": "Volatility Mean vs Median",
        "sharpe_surface_3d": "Sharpe Surface",
    }
    for key, title in title_map.items():
        cards.append({"title": title, "figure": momentum_diagnostic_figs[key]})

    zscore_fig = line_chart_plotter.plot_momentum_zscore_comparison(
        zscore_data=momentum_analytics.momentum_zscore_map(
            ticker["Close"],
            window_pairs={
                "21 vs 50": (21, 50),
                "50 vs 200": (50, 200),
                "200 vs 400": (200, 400),
            },
        ),
        ticker_label=ticker_str,
        default_label="200 vs 400",
        default_time_label="3 Years",
        sigma_levels=(0.5, 1.0, 1.5),
    )
    zscore_fig.update_layout(height=850)
    cards.append({"title": "Momentum Z-Score", "figure": zscore_fig})

    return cards


def _build_relative_risk_cards(
    *,
    ticker: pd.DataFrame,
    ticker_str: str,
    benchmark_data: dict[str, pd.DataFrame],
    time_frame_map: dict[str, int],
    risk_free_daily_rate: pd.Series,
    rolling: Rolling,
    risk_relative_analytics: RiskRelativeAnalytics,
    risk_distribution_analytics: RiskDistributionAnalytics,
    line_chart_plotter: LineChartPlotter,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    notes: list[str] = []
    cards: list[dict[str, Any]] = []

    risk_context = risk_relative_analytics.build_sharpe_sortino_context(
        analytics=rolling,
        asset_close=ticker["Close"],
        time_frame_map=time_frame_map,
        benchmark_data=benchmark_data,
        risk_free_rate=risk_free_daily_rate,
    )

    term_config_map = risk_context["term_config_map"]
    long_window = time_frame_map.get("long")
    default_term_label = f"{long_window}-day" if long_window is not None else max(
        term_config_map,
        key=lambda label: int(str(label).split("-")[0]),
    )
    sharpe_fig = line_chart_plotter.plot_sharpe_sortino_comparison(
        term_config_map=term_config_map,
        ticker_label=ticker_str,
        default_label=default_term_label,
    )
    cards.append({"title": "Sharpe vs Sortino", "figure": sharpe_fig})

    distribution_context = risk_distribution_analytics.build_risk_distribution_context(
        close_series=ticker["Close"],
        windows=[21, 50, 200],
        default_window=200,
    )
    distribution_fig = line_chart_plotter.plot_distribution_shape_zscores(
        metrics_by_window=distribution_context["metrics_by_window"],
        window_options=distribution_context["windows"],
        default_window=distribution_context["default_window"],
        ticker_label=ticker_str,
    )
    cards.append({"title": "Distribution Shape", "figure": distribution_fig})

    try:
        cards.append(
            {
                "title": "Volatility Model Stack",
                "figure": _build_volatility_model_figure(
                    ticker=ticker,
                    ticker_str=ticker_str,
                    time_frame_map=time_frame_map,
                    rolling=rolling,
                ),
            }
        )
    except ImportError:
        warnings.append("Volatility models section skipped because the 'arch' package is not installed.")

    if risk_context["benchmark_order"]:
        benchmark_plot_payload = risk_relative_analytics.build_benchmark_plot_payload(
            asset_sharpe_map=risk_context["asset_sharpe_map"],
            asset_component_map=risk_context["asset_component_map"],
            benchmark_metrics=risk_context["benchmark_metrics"],
            spread_plot_data=risk_context["spread_plot_data"],
            time_frame_map=time_frame_map,
        )
        default_term = "long" if "long" in time_frame_map else max(time_frame_map, key=time_frame_map.get)
        cards.append(
            {
                "title": "Benchmark Spread Summary",
                "figure": line_chart_plotter.plot_multi_benchmark_sharpe_spread_summary(
                    summary_zscore_map=benchmark_plot_payload["summary_zscore_map"],
                    time_frame_map=time_frame_map,
                    ticker_label=ticker_str,
                    default_term=default_term,
                ),
            }
        )
        cards.append(
            {
                "title": "Benchmark Decomposition Detail",
                "figure": line_chart_plotter.plot_benchmark_zscore_detail(
                    detail_zscore_map=benchmark_plot_payload["detail_zscore_map"],
                    benchmark_order=benchmark_plot_payload["benchmark_order"],
                    time_frame_map=time_frame_map,
                    ticker_label=ticker_str,
                    default_benchmark=benchmark_plot_payload["default_benchmark"],
                    default_term=default_term,
                ),
            }
        )
    else:
        notes.append("No benchmark data was available for benchmark-relative plots.")

    return cards, notes, warnings


def _build_factor_cards(
    *,
    ticker_str: str,
    model: Models,
    qp: Plotter,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    cards: list[dict[str, Any]] = []
    notes: list[str] = []
    warnings: list[str] = []

    try:
        ff_proxy = model.run_ff5_proxy_analysis(
            asset_ticker=ticker_str,
            period="max",
            interval="1d",
            window=252,
            auto_window=True,
            verbose=False,
        )
        factor_figs = qp.plot_rolling_regression(
            ff_proxy["rolling_results"],
            ticker_str,
            ff_proxy["factor_returns_ff5"],
            show=False,
        )
        idio_fig = qp.plot_idiosyncratic_risk(
            ff_proxy["rolling_results"],
            ticker_str,
            show=False,
        )
    except Exception as exc:
        warnings.append(f"Factor model section skipped: {exc}")
        return cards, notes, warnings

    cards.append({"title": "Rolling Alpha", "figure": factor_figs["alpha"]})
    cards.append({"title": "Rolling Betas", "figure": factor_figs["betas"]})
    cards.append({"title": "Rolling R-Squared", "figure": factor_figs["r_squared"]})
    cards.append({"title": "Idiosyncratic Risk", "figure": idio_fig})
    notes.append("Factor analysis uses ETF proxy factors rather than direct Fama-French downloads.")

    return cards, notes, warnings


def _build_treasury_cards(
    *,
    ticker_str: str,
    benchmark_tickers: list[str],
    interval: str,
    length_of_plots: int,
    helper: Helper,
    comparison_windows: list[int] | None = None,
    include_curve_comparison: bool = True,
    include_historical_yields: bool = True,
    include_deannualized_curves: bool = True,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    notes: list[str] = []
    cards: list[dict[str, Any]] = []

    fred_key = _load_fred_key()
    if not fred_key:
        warnings.append("Treasury section skipped because FRED_API_KEY is not configured.")
        return cards, notes, warnings

    treasury_client = MacroDataClient(fred_key=fred_key)
    treasury_start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=length_of_plots)).strftime("%Y-%m-%d")
    treasury_yields = treasury_client.get_historical_treasury_yields(start_date=treasury_start_date).sort_index().dropna(how="all")

    maturity_to_years = {
        "1M": 1 / 12,
        "3M": 3 / 12,
        "6M": 6 / 12,
        "1Y": 1,
        "2Y": 2,
        "3Y": 3,
        "5Y": 5,
        "7Y": 7,
        "10Y": 10,
        "20Y": 20,
        "30Y": 30,
    }
    bill_maturities = {"1M", "3M", "6M", "1Y"}
    ordered_maturities = [maturity for maturity in maturity_to_years if maturity in treasury_yields.columns]
    latest_curve_frame = treasury_yields[ordered_maturities].ffill().dropna(how="all")

    asset_history = yf.Ticker(ticker_str).history(period="max", interval=interval)
    asset_history = helper.simplify_datetime_index(asset_history)
    asset_close = asset_history["Close"].dropna().sort_index()
    curve_benchmark_tickers = normalize_benchmark_tickers(benchmark_tickers, ticker_str)
    benchmark_close_map: dict[str, pd.Series] = {}
    skipped_curve_benchmarks: list[str] = []
    for benchmark_symbol in curve_benchmark_tickers:
        benchmark_history = yf.Ticker(benchmark_symbol).history(period="max", interval=interval)
        benchmark_history = helper.simplify_datetime_index(benchmark_history)
        if benchmark_history.empty or "Close" not in benchmark_history:
            skipped_curve_benchmarks.append(benchmark_symbol)
            continue
        benchmark_close = benchmark_history["Close"].dropna().sort_index()
        if benchmark_close.empty:
            skipped_curve_benchmarks.append(benchmark_symbol)
            continue
        benchmark_close_map[benchmark_symbol] = benchmark_close

    if latest_curve_frame.empty:
        warnings.append("Treasury section skipped because Treasury yield history came back empty.")
        return cards, notes, warnings
    if asset_close.empty:
        warnings.append(f"Treasury section skipped because no price history was available for {ticker_str}.")
        return cards, notes, warnings

    comparison_end_candidates = [latest_curve_frame.index.max(), asset_close.index.max()]
    comparison_end_candidates.extend(series.index.max() for series in benchmark_close_map.values())
    comparison_end_date = min(comparison_end_candidates)
    latest_curve_frame = latest_curve_frame.loc[:comparison_end_date]
    asset_close = asset_close.loc[:comparison_end_date]
    benchmark_close_map = {symbol: series.loc[:comparison_end_date] for symbol, series in benchmark_close_map.items()}

    if skipped_curve_benchmarks:
        warnings.append(
            "Skipped yield-curve benchmark overlays with no data: "
            + ", ".join(skipped_curve_benchmarks)
        )

    latest_curve_date = latest_curve_frame.index[-1]
    latest_curve = latest_curve_frame.iloc[-1].dropna()
    curve_positions = list(range(len(latest_curve.index)))
    curve_tick_text = latest_curve.index.tolist()
    curve_year_values = np.array([maturity_to_years[maturity] for maturity in latest_curve.index], dtype=float)
    curve_year_min = float(curve_year_values.min())
    curve_year_max = float(curve_year_values.max())
    curve_region_end = len(curve_positions) - 0.5
    bill_note_boundary = latest_curve.index.get_loc("1Y") + 0.5 if {"1Y", "2Y"}.issubset(set(latest_curve.index)) else None
    bill_region_mid = (bill_note_boundary - 0.5) / 2 if bill_note_boundary is not None else len(curve_positions) / 4
    note_bond_region_mid = (
        (bill_note_boundary + curve_region_end) / 2 if bill_note_boundary is not None else len(curve_positions) * 0.75
    )
    bill_note_boundary_years = (
        (maturity_to_years["1Y"] + maturity_to_years["2Y"]) / 2 if {"1Y", "2Y"}.issubset(set(latest_curve.index)) else None
    )
    _ = curve_year_min, curve_year_max, bill_note_boundary_years
    maturity_date_offsets = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "2Y": pd.DateOffset(years=2),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5),
        "7Y": pd.DateOffset(years=7),
        "10Y": pd.DateOffset(years=10),
        "20Y": pd.DateOffset(years=20),
        "30Y": pd.DateOffset(years=30),
    }

    def deannualize_treasury_yield(yield_pct, maturity):
        years = maturity_to_years[maturity]
        yield_decimal = yield_pct / 100
        if maturity in bill_maturities:
            return (((1 + yield_decimal) ** years) - 1) * 100
        return (((1 + yield_decimal / 2) ** (2 * years)) - 1) * 100

    def holding_period_asset_return(price_series, end_date, maturity):
        start_target = end_date - maturity_date_offsets[maturity]
        start_series = price_series.loc[:start_target]
        if start_series.empty:
            return np.nan
        start_price = start_series.iloc[-1]
        end_price = price_series.loc[:end_date].iloc[-1]
        if start_price <= 0 or end_price <= 0:
            return np.nan
        return ((end_price / start_price) - 1) * 100

    def annualize_asset_return(price_series, end_date, maturity):
        years = maturity_to_years[maturity]
        holding_period_return = holding_period_asset_return(price_series, end_date, maturity)
        if pd.isna(holding_period_return):
            return np.nan
        gross_return = 1 + (holding_period_return / 100)
        if gross_return <= 0:
            return np.nan
        return ((gross_return ** (1 / years)) - 1) * 100

    def build_interpolated_curve_data(curve_series):
        if curve_series is None or curve_series.empty or len(curve_series) < 2:
            return None
        x_values = np.array([maturity_to_years[maturity] for maturity in curve_series.index], dtype=float)
        y_values = curve_series.values.astype(float)
        sort_order = np.argsort(x_values)
        x_values = x_values[sort_order]
        y_values = y_values[sort_order]
        labels = [curve_series.index[idx] for idx in sort_order]
        x_plot_values = np.array([curve_tick_text.index(label) for label in labels], dtype=float)
        interpolator = PchipInterpolator(x_values, y_values)
        x_smooth = np.linspace(float(x_values.min()), float(x_values.max()), 300)
        y_smooth = interpolator(x_smooth)
        x_plot_smooth = np.interp(x_smooth, x_values, x_plot_values)
        return {
            "x_plot_values": x_plot_values,
            "y_values": y_values,
            "x_plot_smooth": x_plot_smooth,
            "y_smooth": y_smooth,
            "labels": labels,
        }

    def interpolate_curve_yield(curve_series, target_years):
        if curve_series is None or curve_series.empty:
            return np.nan
        valid_curve = curve_series.dropna()
        if valid_curve.empty:
            return np.nan
        x_values = np.array([maturity_to_years[maturity] for maturity in valid_curve.index], dtype=float)
        y_values = valid_curve.values.astype(float)
        sort_order = np.argsort(x_values)
        x_values = x_values[sort_order]
        y_values = y_values[sort_order]
        if len(x_values) == 1:
            return float(y_values[0])
        if target_years <= float(x_values.min()):
            return float(y_values[0])
        if target_years >= float(x_values.max()):
            return float(y_values[-1])
        interpolator = PchipInterpolator(x_values, y_values)
        interpolated_value = interpolator(float(target_years))
        return float(interpolated_value)

    def annualized_yield_to_holding_period_return(yield_pct, years):
        if pd.isna(yield_pct):
            return np.nan
        yield_decimal = yield_pct / 100
        return (((1 + yield_decimal) ** years) - 1) * 100

    def build_historical_excess_return_series(price_series, treasury_frame, maturity):
        treasury_locked_yield_series = treasury_frame[maturity].ffill().dropna()
        if treasury_locked_yield_series.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        years = maturity_to_years[maturity]
        offset = maturity_date_offsets[maturity]
        candidate_dates = price_series.index.intersection(treasury_locked_yield_series.index)
        holding_period_excess_values = []
        annualized_excess_values = []
        valid_dates = []
        for end_date in candidate_dates:
            start_target = end_date - offset
            start_price_series = price_series.loc[:start_target]
            start_yield_series = treasury_locked_yield_series.loc[:start_target]
            if start_price_series.empty or start_yield_series.empty:
                continue
            start_price = start_price_series.iloc[-1]
            end_price = price_series.loc[:end_date].iloc[-1]
            locked_yield = start_yield_series.iloc[-1]
            if start_price <= 0 or end_price <= 0 or pd.isna(locked_yield):
                continue
            asset_holding_period_return = ((end_price / start_price) - 1) * 100
            treasury_holding_period_return = deannualize_treasury_yield(locked_yield, maturity)
            gross_asset_return = 1 + (asset_holding_period_return / 100)
            asset_annualized_return = np.nan if gross_asset_return <= 0 else ((gross_asset_return ** (1 / years)) - 1) * 100
            holding_period_excess_values.append(asset_holding_period_return - treasury_holding_period_return)
            annualized_excess_values.append(asset_annualized_return - locked_yield)
            valid_dates.append(end_date)
        return (
            pd.Series(holding_period_excess_values, index=valid_dates, dtype=float),
            pd.Series(annualized_excess_values, index=valid_dates, dtype=float),
        )

    def build_window_locked_treasury_excess_frame(price_series, treasury_frame, window_days):
        if price_series is None or price_series.empty or len(price_series) <= window_days:
            return pd.DataFrame()
        years = window_days / 252
        result_rows = []
        result_dates = []
        price_index = price_series.index
        for end_idx in range(window_days, len(price_series)):
            end_date = price_index[end_idx]
            start_date = price_index[end_idx - window_days]
            start_price = price_series.iloc[end_idx - window_days]
            end_price = price_series.iloc[end_idx]
            if start_price <= 0 or end_price <= 0:
                continue
            treasury_history = treasury_frame.loc[:start_date].ffill().dropna(how="all")
            if treasury_history.empty:
                continue
            locked_curve = treasury_history.iloc[-1].dropna()
            locked_yield = interpolate_curve_yield(locked_curve, years)
            if pd.isna(locked_yield):
                continue
            asset_return = ((end_price / start_price) - 1) * 100
            treasury_return = annualized_yield_to_holding_period_return(locked_yield, years)
            result_rows.append(
                {
                    "asset_return": asset_return,
                    "treasury_return": treasury_return,
                    "spread": asset_return - treasury_return,
                    "locked_yield": locked_yield,
                }
            )
            result_dates.append(end_date)
        if not result_rows:
            return pd.DataFrame()
        return pd.DataFrame(result_rows, index=result_dates, dtype=float)

    def locked_treasury_yield(treasury_frame, end_date, maturity):
        treasury_locked_yield_series = treasury_frame[maturity].ffill().dropna()
        if treasury_locked_yield_series.empty:
            return np.nan
        start_target = end_date - maturity_date_offsets[maturity]
        start_yield_series = treasury_locked_yield_series.loc[:start_target]
        if start_yield_series.empty:
            return np.nan
        return float(start_yield_series.iloc[-1])

    def locked_treasury_holding_period_return(treasury_frame, end_date, maturity):
        locked_yield = locked_treasury_yield(treasury_frame, end_date, maturity)
        if pd.isna(locked_yield):
            return np.nan
        return deannualize_treasury_yield(locked_yield, maturity)

    ex_post_annualized_treasury_curve = pd.Series(
        {maturity: locked_treasury_yield(latest_curve_frame, latest_curve_date, maturity) for maturity in latest_curve.index}
    ).dropna()
    ex_post_holding_treasury_curve = pd.Series(
        {
            maturity: locked_treasury_holding_period_return(latest_curve_frame, latest_curve_date, maturity)
            for maturity in latest_curve.index
        }
    ).dropna()
    ex_post_annualized_positions = [curve_tick_text.index(maturity) for maturity in ex_post_annualized_treasury_curve.index]
    ex_post_holding_positions = [curve_tick_text.index(maturity) for maturity in ex_post_holding_treasury_curve.index]

    asset_annualized_curve = pd.Series(
        {maturity: annualize_asset_return(asset_close, latest_curve_date, maturity) for maturity in latest_curve.index}
    ).dropna()
    asset_annualized_curve = asset_annualized_curve.loc[
        asset_annualized_curve.index.intersection(ex_post_annualized_treasury_curve.index)
    ]
    annualized_asset_curve_positions = [curve_tick_text.index(maturity) for maturity in asset_annualized_curve.index]
    annualized_excess_curve = asset_annualized_curve - ex_post_annualized_treasury_curve.loc[asset_annualized_curve.index]
    annualized_gap_x, annualized_gap_y = [], []
    for maturity in asset_annualized_curve.index:
        position = curve_tick_text.index(maturity)
        annualized_gap_x.extend([position, position, None])
        annualized_gap_y.extend([ex_post_annualized_treasury_curve[maturity], asset_annualized_curve[maturity], None])

    asset_holding_period_curve = pd.Series(
        {maturity: holding_period_asset_return(asset_close, latest_curve_date, maturity) for maturity in latest_curve.index}
    ).dropna()
    asset_holding_period_curve = asset_holding_period_curve.loc[
        asset_holding_period_curve.index.intersection(ex_post_holding_treasury_curve.index)
    ]
    asset_curve_positions = [curve_tick_text.index(maturity) for maturity in asset_holding_period_curve.index]
    holding_period_excess_curve = asset_holding_period_curve - ex_post_holding_treasury_curve.loc[asset_holding_period_curve.index]
    holding_period_gap_x, holding_period_gap_y = [], []
    for maturity in asset_holding_period_curve.index:
        position = curve_tick_text.index(maturity)
        holding_period_gap_x.extend([position, position, None])
        holding_period_gap_y.extend([ex_post_holding_treasury_curve[maturity], asset_holding_period_curve[maturity], None])

    benchmark_curve_payloads = {}
    benchmark_color_sequence = ["#16a34a", "#9333ea", "#d97706", "#dc2626", "#0891b2", "#7c3aed"]
    for idx, (benchmark_symbol, benchmark_close) in enumerate(benchmark_close_map.items()):
        benchmark_annualized_curve = pd.Series(
            {maturity: annualize_asset_return(benchmark_close, latest_curve_date, maturity) for maturity in latest_curve.index}
        ).dropna()
        benchmark_annualized_curve = benchmark_annualized_curve.loc[
            benchmark_annualized_curve.index.intersection(ex_post_annualized_treasury_curve.index)
        ]
        benchmark_holding_curve = pd.Series(
            {maturity: holding_period_asset_return(benchmark_close, latest_curve_date, maturity) for maturity in latest_curve.index}
        ).dropna()
        benchmark_holding_curve = benchmark_holding_curve.loc[
            benchmark_holding_curve.index.intersection(ex_post_holding_treasury_curve.index)
        ]
        benchmark_curve_payloads[benchmark_symbol] = {
            "color": benchmark_color_sequence[idx % len(benchmark_color_sequence)],
            "annualized_curve": benchmark_annualized_curve,
            "annualized_positions": [curve_tick_text.index(maturity) for maturity in benchmark_annualized_curve.index],
            "annualized_excess_curve": benchmark_annualized_curve - ex_post_annualized_treasury_curve.loc[benchmark_annualized_curve.index],
            "holding_curve": benchmark_holding_curve,
            "holding_positions": [curve_tick_text.index(maturity) for maturity in benchmark_holding_curve.index],
            "holding_excess_curve": benchmark_holding_curve - ex_post_holding_treasury_curve.loc[benchmark_holding_curve.index],
        }

    annualized_curve_interp = build_interpolated_curve_data(ex_post_annualized_treasury_curve)
    deannualized_curve_interp = build_interpolated_curve_data(ex_post_holding_treasury_curve)
    asset_annualized_curve_interp = build_interpolated_curve_data(asset_annualized_curve)
    asset_holding_curve_interp = build_interpolated_curve_data(asset_holding_period_curve)
    for benchmark_payload in benchmark_curve_payloads.values():
        benchmark_payload["annualized_interp"] = build_interpolated_curve_data(benchmark_payload["annualized_curve"])
        benchmark_payload["holding_interp"] = build_interpolated_curve_data(benchmark_payload["holding_curve"])

    annualized_label_ceiling = ex_post_annualized_treasury_curve.max()
    if not asset_annualized_curve.empty:
        annualized_label_ceiling = max(annualized_label_ceiling, asset_annualized_curve.max())
    for benchmark_payload in benchmark_curve_payloads.values():
        if not benchmark_payload["annualized_curve"].empty:
            annualized_label_ceiling = max(annualized_label_ceiling, benchmark_payload["annualized_curve"].max())
    annualized_label_y = annualized_label_ceiling * 0.96

    deannualized_label_ceiling = ex_post_holding_treasury_curve.max()
    if not asset_holding_period_curve.empty:
        deannualized_label_ceiling = max(deannualized_label_ceiling, asset_holding_period_curve.max())
    for benchmark_payload in benchmark_curve_payloads.values():
        if not benchmark_payload["holding_curve"].empty:
            deannualized_label_ceiling = max(deannualized_label_ceiling, benchmark_payload["holding_curve"].max())
    deannualized_label_y = deannualized_label_ceiling * 0.96

    historical_excess_maturities = [maturity for maturity in ["3M", "1Y", "5Y", "10Y", "30Y"] if maturity in ordered_maturities]
    curve_overlay_label = f"{ticker_str} + Benchmarks" if benchmark_curve_payloads else ticker_str
    annualized_curve_title = f"Ex Post Annualized Treasury Hurdle vs {curve_overlay_label} ({latest_curve_date:%Y-%m-%d})"
    annualized_interp_title = f"Interpolated Ex Post Annualized Curves ({latest_curve_date:%Y-%m-%d})"
    holding_curve_title = f"Ex Post Treasury Holding Return vs {curve_overlay_label} ({latest_curve_date:%Y-%m-%d})"
    holding_interp_title = f"Interpolated Ex Post Holding-Period Curves ({latest_curve_date:%Y-%m-%d})"

    if include_historical_yields and include_deannualized_curves:
        treasury_fig = make_subplots(
            rows=3,
            cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
            row_heights=[0.44, 0.28, 0.28],
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
            subplot_titles=(
                "Historical Treasury Yields",
                annualized_curve_title,
                holding_curve_title,
                annualized_interp_title,
                holding_interp_title,
            ),
        )
        historical_panel = (1, 1)
        annualized_panel = (2, 1)
        holding_panel = (2, 2)
        annualized_interp_panel = (3, 1)
        holding_interp_panel = (3, 2)
        figure_height = 1450
        figure_title = "US Treasury Yield History with Ex Post Matched-Horizon Return Overlays"
    elif include_historical_yields:
        treasury_fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.44, 0.28, 0.28],
            vertical_spacing=0.1,
            subplot_titles=(
                "Historical Treasury Yields",
                annualized_curve_title,
                annualized_interp_title,
            ),
        )
        historical_panel = (1, 1)
        annualized_panel = (2, 1)
        holding_panel = None
        annualized_interp_panel = (3, 1)
        holding_interp_panel = None
        figure_height = 1350
        figure_title = "US Treasury Yield History with Ex Post Annualized Return Overlays"
    elif include_deannualized_curves:
        treasury_fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.5, 0.5],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            subplot_titles=(
                annualized_curve_title,
                holding_curve_title,
                annualized_interp_title,
                holding_interp_title,
            ),
        )
        historical_panel = None
        annualized_panel = (1, 1)
        holding_panel = (1, 2)
        annualized_interp_panel = (2, 1)
        holding_interp_panel = (2, 2)
        figure_height = 1150
        figure_title = "Ex Post Treasury Opportunity-Cost Curves"
    else:
        treasury_fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.52, 0.48],
            vertical_spacing=0.12,
            subplot_titles=(
                annualized_curve_title,
                annualized_interp_title,
            ),
        )
        historical_panel = None
        annualized_panel = (1, 1)
        holding_panel = None
        annualized_interp_panel = (2, 1)
        holding_interp_panel = None
        figure_height = 980
        figure_title = "Ex Post Annualized Treasury Opportunity-Cost Curves"

    if historical_panel is not None:
        historical_row, historical_col = historical_panel
        for maturity in ordered_maturities:
            series = treasury_yields[maturity].dropna()
            if series.empty:
                continue
            treasury_fig.add_trace(
                go.Scatter(x=series.index, y=series, mode="lines", name=maturity, line=dict(width=1.6)),
                row=historical_row,
                col=historical_col,
            )

    annualized_row, annualized_col = annualized_panel
    if not ex_post_annualized_treasury_curve.empty:
        treasury_fig.add_trace(
            go.Scatter(
                x=ex_post_annualized_positions,
                y=ex_post_annualized_treasury_curve.values.tolist(),
                mode="lines+markers",
                name="Locked Treasury Yield (Ex Post)",
                line=dict(color="#0f172a", width=3),
                marker=dict(size=9, color="#0f766e"),
                customdata=np.array(ex_post_annualized_treasury_curve.index),
                hovertemplate="Maturity=%{customdata}<br>Locked Treasury annualized yield=%{y:.2f}%<extra></extra>",
                showlegend=False,
            ),
            row=annualized_row,
            col=annualized_col,
        )
    if annualized_gap_x:
        treasury_fig.add_trace(
            go.Scatter(
                x=annualized_gap_x,
                y=annualized_gap_y,
                mode="lines",
                name="Annualized Excess Return Gap",
                line=dict(color="rgba(37, 99, 235, 0.45)", width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=annualized_row,
            col=annualized_col,
        )
    if not asset_annualized_curve.empty:
        treasury_fig.add_trace(
            go.Scatter(
                x=annualized_asset_curve_positions,
                y=asset_annualized_curve.values.tolist(),
                mode="lines+markers",
                name=f"{ticker_str} Annualized Return",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=9, color="#1d4ed8", symbol="diamond"),
                customdata=np.column_stack((asset_annualized_curve.index, annualized_excess_curve.values)),
                hovertemplate="Maturity=%{customdata[0]}<br>" + f"{ticker_str} annualized return=%{{y:.2f}}%<br>" + "Excess vs locked Treasury=%{customdata[1]:.2f}%<extra></extra>",
            ),
            row=annualized_row,
            col=annualized_col,
        )
    for benchmark_symbol, benchmark_payload in benchmark_curve_payloads.items():
        if benchmark_payload["annualized_curve"].empty:
            continue
        treasury_fig.add_trace(
            go.Scatter(
                x=benchmark_payload["annualized_positions"],
                y=benchmark_payload["annualized_curve"].values.tolist(),
                mode="lines+markers",
                name=f"{benchmark_symbol} Annualized Return",
                legendgroup=benchmark_symbol,
                line=dict(color=benchmark_payload["color"], width=2.4, dash="dash"),
                marker=dict(size=8, color=benchmark_payload["color"], symbol="circle-open"),
                customdata=np.column_stack((benchmark_payload["annualized_curve"].index, benchmark_payload["annualized_excess_curve"].values)),
                hovertemplate="Maturity=%{customdata[0]}<br>" + f"{benchmark_symbol} annualized return=%{{y:.2f}}%<br>" + "Excess vs locked Treasury=%{customdata[1]:.2f}%<extra></extra>",
            ),
            row=annualized_row,
            col=annualized_col,
        )

    annualized_interp_row, annualized_interp_col = annualized_interp_panel
    _add_interpolated_curve_traces(
        fig=treasury_fig,
        row=annualized_interp_row,
        col=annualized_interp_col,
        treasury_interp=annualized_curve_interp,
        asset_interp=asset_annualized_curve_interp,
        benchmark_payloads=benchmark_curve_payloads,
        benchmark_key="annualized_interp",
        treasury_name="Interpolated Locked Treasury Yield Curve",
        treasury_hover="Maturity=%{text}<br>Locked Treasury annualized yield=%{y:.2f}%<extra></extra>",
        asset_label=ticker_str,
    )

    if holding_panel is not None:
        holding_row, holding_col = holding_panel
        if not ex_post_holding_treasury_curve.empty:
            treasury_fig.add_trace(
                go.Scatter(
                    x=ex_post_holding_positions,
                    y=ex_post_holding_treasury_curve.values.tolist(),
                    mode="lines+markers",
                    name="Locked Treasury Holding Return (Ex Post)",
                    line=dict(color="#7c2d12", width=3),
                    marker=dict(size=9, color="#ea580c"),
                    customdata=np.array(ex_post_holding_treasury_curve.index),
                    hovertemplate="Maturity=%{customdata}<br>Locked Treasury holding-period return=%{y:.2f}%<extra></extra>",
                    showlegend=False,
                ),
                row=holding_row,
                col=holding_col,
            )
        for benchmark_symbol, benchmark_payload in benchmark_curve_payloads.items():
            if benchmark_payload["holding_curve"].empty:
                continue
            treasury_fig.add_trace(
                go.Scatter(
                    x=benchmark_payload["holding_positions"],
                    y=benchmark_payload["holding_curve"].values.tolist(),
                    mode="lines+markers",
                    name=f"{benchmark_symbol} Holding-Period Return",
                    legendgroup=benchmark_symbol,
                    line=dict(color=benchmark_payload["color"], width=2.4, dash="dash"),
                    marker=dict(size=8, color=benchmark_payload["color"], symbol="circle-open"),
                    customdata=np.column_stack((benchmark_payload["holding_curve"].index, benchmark_payload["holding_excess_curve"].values)),
                    hovertemplate="Maturity=%{customdata[0]}<br>" + f"{benchmark_symbol} holding-period return=%{{y:.2f}}%<br>" + "Excess vs locked Treasury=%{customdata[1]:.2f}%<extra></extra>",
                ),
                row=holding_row,
                col=holding_col,
            )
        if holding_period_gap_x:
            treasury_fig.add_trace(
                go.Scatter(
                    x=holding_period_gap_x,
                    y=holding_period_gap_y,
                    mode="lines",
                    name="Holding-Period Excess Return Gap",
                    line=dict(color="rgba(37, 99, 235, 0.45)", width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=holding_row,
                col=holding_col,
            )
        if not asset_holding_period_curve.empty:
            treasury_fig.add_trace(
                go.Scatter(
                    x=asset_curve_positions,
                    y=asset_holding_period_curve.values.tolist(),
                    mode="lines+markers",
                    name=f"{ticker_str} Holding-Period Return",
                    line=dict(color="#2563eb", width=3),
                    marker=dict(size=9, color="#1d4ed8", symbol="diamond"),
                    customdata=np.column_stack((asset_holding_period_curve.index, holding_period_excess_curve.values)),
                    hovertemplate="Maturity=%{customdata[0]}<br>" + f"{ticker_str} holding-period return=%{{y:.2f}}%<br>" + "Excess vs locked Treasury=%{customdata[1]:.2f}%<extra></extra>",
                ),
                row=holding_row,
                col=holding_col,
            )

    if holding_interp_panel is not None:
        holding_interp_row, holding_interp_col = holding_interp_panel
        _add_interpolated_curve_traces(
            fig=treasury_fig,
            row=holding_interp_row,
            col=holding_interp_col,
            treasury_interp=deannualized_curve_interp,
            asset_interp=asset_holding_curve_interp,
            benchmark_payloads=benchmark_curve_payloads,
            benchmark_key="holding_interp",
            treasury_name="Interpolated Locked Treasury Holding Curve",
            treasury_hover="Maturity=%{text}<br>Locked Treasury holding-period return=%{y:.2f}%<extra></extra>",
            asset_label=ticker_str,
            treasury_line_color="#7c2d12",
            treasury_marker_color="#ea580c",
        )

    if bill_note_boundary is not None:
        shaded_panels = [annualized_panel, annualized_interp_panel]
        label_specs = [
            (*annualized_panel, bill_region_mid, annualized_label_y, "Bills"),
            (*annualized_panel, note_bond_region_mid, annualized_label_y, "Notes/Bonds"),
            (*annualized_interp_panel, bill_region_mid, annualized_label_y, "Bills"),
            (*annualized_interp_panel, note_bond_region_mid, annualized_label_y, "Notes/Bonds"),
        ]
        if holding_panel is not None and holding_interp_panel is not None:
            shaded_panels.extend([holding_panel, holding_interp_panel])
            label_specs.extend(
                [
                    (*holding_panel, bill_region_mid, deannualized_label_y, "Bills"),
                    (*holding_panel, note_bond_region_mid, deannualized_label_y, "Notes/Bonds"),
                    (*holding_interp_panel, bill_region_mid, deannualized_label_y, "Bills"),
                    (*holding_interp_panel, note_bond_region_mid, deannualized_label_y, "Notes/Bonds"),
                ]
            )

        for row, col in shaded_panels:
            treasury_fig.add_vrect(
                x0=bill_note_boundary,
                x1=curve_region_end,
                fillcolor="rgba(148, 163, 184, 0.18)",
                line_width=0,
                row=row,
                col=col,
            )

        for row, col, x, y, text in label_specs:
            treasury_fig.add_annotation(
                x=x,
                y=y,
                text=text,
                showarrow=False,
                font=dict(size=11, color="#334155"),
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(148, 163, 184, 0.5)",
                row=row,
                col=col,
            )

    if historical_panel is not None:
        historical_row, historical_col = historical_panel
        treasury_fig.update_xaxes(title_text="Date", row=historical_row, col=historical_col)
        treasury_fig.update_yaxes(title_text="Yield (%)", row=historical_row, col=historical_col)

    maturity_axis_specs = [
        (*annualized_panel, "Annualized Return / Yield (%)"),
        (*annualized_interp_panel, "Annualized Return / Yield (%)"),
    ]
    if holding_panel is not None and holding_interp_panel is not None:
        maturity_axis_specs.extend(
            [
                (*holding_panel, "Holding-Period Return (%)"),
                (*holding_interp_panel, "Holding-Period Return (%)"),
            ]
        )

    for row, col, y_title in maturity_axis_specs:
        treasury_fig.update_xaxes(
            title_text="Maturity",
            tickmode="array",
            tickvals=curve_positions,
            ticktext=curve_tick_text,
            range=[-0.5, curve_region_end],
            row=row,
            col=col,
        )
        treasury_fig.update_yaxes(title_text=y_title, row=row, col=col)

    treasury_fig.update_layout(
        title=figure_title,
        template="plotly_white",
        height=figure_height,
        legend_title_text="Series",
        hovermode="x unified",
    )
    cards.append({"title": "Treasury Curve Comparison", "figure": treasury_fig})
    if not include_curve_comparison:
        cards = [card for card in cards if card["title"] != "Treasury Curve Comparison"]

    if comparison_windows:
        window_excess_frames: dict[int, pd.DataFrame] = {}
        historical_color_sequence = px.colors.qualitative.Bold
        normalized_windows = [int(window) for window in comparison_windows if int(window) > 0]
        for window_days in normalized_windows:
            window_frame = build_window_locked_treasury_excess_frame(
                price_series=asset_close,
                treasury_frame=latest_curve_frame,
                window_days=window_days,
            )
            if not window_frame.empty:
                window_excess_frames[window_days] = window_frame

        if window_excess_frames:
            historical_excess_fig = go.Figure()
            for idx, window_days in enumerate(normalized_windows):
                window_frame = window_excess_frames.get(window_days)
                if window_frame is None or window_frame.empty:
                    continue
                color = historical_color_sequence[idx % len(historical_color_sequence)]
                historical_excess_fig.add_trace(
                    go.Scatter(
                        x=window_frame.index,
                        y=window_frame["spread"].values,
                        mode="lines",
                        name=f"{window_days}d Spread",
                        line=dict(color=color, width=2.3),
                        customdata=np.column_stack(
                            (
                                window_frame["asset_return"].values,
                                window_frame["treasury_return"].values,
                                window_frame["locked_yield"].values,
                            )
                        ),
                        hovertemplate=(
                            "Date=%{x|%Y-%m-%d}<br>"
                            + f"{window_days}d excess return=%{{y:.2f}}%<br>"
                            + "Asset return=%{customdata[0]:.2f}%<br>"
                            + "Locked Treasury return=%{customdata[1]:.2f}%<br>"
                            + "Locked Treasury annualized yield=%{customdata[2]:.2f}%<extra></extra>"
                        ),
                    )
                )
            historical_excess_fig.add_hline(y=0, line_color="#475569", line_width=1, line_dash="dot")
            historical_excess_fig.update_yaxes(title_text="Excess Return (%)")
            historical_excess_fig.update_xaxes(title_text="Date")
            historical_excess_fig.update_layout(
                title=f"{ticker_str} Historical Excess Returns vs Locked Treasury ({', '.join(f'{window}d' for window in normalized_windows)} Windows)",
                template="plotly_white",
                height=620,
                legend_title_text="Window",
                hovermode="x unified",
            )
            cards.append({"title": "Historical Treasury Excess", "figure": historical_excess_fig})

            for window_days in normalized_windows:
                window_frame = window_excess_frames.get(window_days)
                if window_frame is None or window_frame.empty:
                    continue
                latest_window = window_frame.iloc[-1]
                latest_window_date = window_frame.index[-1]
                beat_bond_text = "Yes" if latest_window["spread"] > 0 else "No"
                notes.append(
                    f"As of {latest_window_date:%Y-%m-%d}, {ticker_str}'s {window_days}-day return was "
                    f"{latest_window['asset_return']:.2f}% versus {latest_window['treasury_return']:.2f}% "
                    f"from a locked Treasury hurdle, for a spread of {latest_window['spread']:.2f}%. "
                    f"Did {ticker_str} beat the Treasury? {beat_bond_text}."
                )
    else:
        historical_holding_excess_map = {}
        historical_annualized_excess_map = {}
        historical_color_sequence = px.colors.qualitative.Bold
        for maturity in historical_excess_maturities:
            holding_excess_series, annualized_excess_series = build_historical_excess_return_series(
                price_series=asset_close,
                treasury_frame=latest_curve_frame,
                maturity=maturity,
            )
            if not holding_excess_series.empty:
                historical_holding_excess_map[maturity] = holding_excess_series
            if not annualized_excess_series.empty:
                historical_annualized_excess_map[maturity] = annualized_excess_series

        if historical_holding_excess_map or historical_annualized_excess_map:
            historical_excess_fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f"{ticker_str} Historical Holding-Period Excess Return vs Locked Treasury",
                    f"{ticker_str} Historical Annualized Excess Return vs Locked Treasury",
                ),
            )
            for idx, maturity in enumerate(historical_excess_maturities):
                color = historical_color_sequence[idx % len(historical_color_sequence)]
                holding_series = historical_holding_excess_map.get(maturity)
                annualized_series = historical_annualized_excess_map.get(maturity)
                if holding_series is not None and not holding_series.empty:
                    historical_excess_fig.add_trace(
                        go.Scatter(x=holding_series.index, y=holding_series.values, mode="lines", name=maturity, legendgroup=maturity, line=dict(color=color, width=2.2)),
                        row=1,
                        col=1,
                    )
                if annualized_series is not None and not annualized_series.empty:
                    historical_excess_fig.add_trace(
                        go.Scatter(x=annualized_series.index, y=annualized_series.values, mode="lines", name=maturity, legendgroup=maturity, line=dict(color=color, width=2.2, dash="dot"), showlegend=False),
                        row=2,
                        col=1,
                    )
            historical_excess_fig.add_hline(y=0, line_color="#475569", line_width=1, line_dash="dot", row=1, col=1)
            historical_excess_fig.add_hline(y=0, line_color="#475569", line_width=1, line_dash="dot", row=2, col=1)
            historical_excess_fig.update_yaxes(title_text="Excess Return (%)", row=1, col=1)
            historical_excess_fig.update_yaxes(title_text="Excess Return (%)", row=2, col=1)
            historical_excess_fig.update_xaxes(title_text="Date", row=2, col=1)
            historical_excess_fig.update_layout(
                title=f"{ticker_str} Historical Excess Returns vs Locked Treasury Benchmarks",
                template="plotly_white",
                height=900,
                legend_title_text="Maturity",
                hovermode="x unified",
            )
            cards.append({"title": "Historical Treasury Excess", "figure": historical_excess_fig})

        if "3M" in treasury_yields.columns:
            asset_vs_bond_3m = pd.concat([asset_close.rename("Asset_Close"), treasury_yields[["3M"]]], axis=1, join="inner").dropna()
            asset_vs_bond_3m["asset_trailing_3m_return"] = asset_vs_bond_3m["Asset_Close"].pct_change(63)
            asset_vs_bond_3m["bond_locked_3m_return"] = (1 + asset_vs_bond_3m["3M"].shift(63) / 100) ** (3 / 12) - 1
            asset_vs_bond_3m["spread"] = asset_vs_bond_3m["asset_trailing_3m_return"] - asset_vs_bond_3m["bond_locked_3m_return"]
            asset_vs_bond_3m = asset_vs_bond_3m.dropna()
            if not asset_vs_bond_3m.empty:
                latest_asset_vs_bond = asset_vs_bond_3m.iloc[-1]
                latest_asset_vs_bond_date = asset_vs_bond_3m.index[-1]
                beat_bond_text = "Yes" if latest_asset_vs_bond["spread"] > 0 else "No"
                notes.append(
                    f"As of {latest_asset_vs_bond_date:%Y-%m-%d}, {ticker_str} returned "
                    f"{latest_asset_vs_bond['asset_trailing_3m_return']:.2%} over the last 3 months versus "
                    f"{latest_asset_vs_bond['bond_locked_3m_return']:.2%} from a locked 3-month Treasury. "
                    f"Did {ticker_str} beat the bond? {beat_bond_text}."
                )

    return cards, notes, warnings


def _add_interpolated_curve_traces(
    *,
    fig: go.Figure,
    row: int,
    col: int,
    treasury_interp: dict[str, Any] | None,
    asset_interp: dict[str, Any] | None,
    benchmark_payloads: dict[str, dict[str, Any]],
    benchmark_key: str,
    treasury_name: str,
    treasury_hover: str,
    asset_label: str,
    treasury_line_color: str = "#0f172a",
    treasury_marker_color: str = "#0f766e",
) -> None:
    if treasury_interp is not None:
        fig.add_trace(
            go.Scatter(
                x=treasury_interp["x_plot_smooth"],
                y=treasury_interp["y_smooth"],
                mode="lines",
                name=treasury_name,
                line=dict(color=treasury_line_color, width=3),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=treasury_interp["x_plot_values"],
                y=treasury_interp["y_values"],
                mode="markers",
                marker=dict(size=7, color=treasury_marker_color),
                text=treasury_interp["labels"],
                hovertemplate=treasury_hover,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    if asset_interp is not None:
        fig.add_trace(
            go.Scatter(
                x=asset_interp["x_plot_smooth"],
                y=asset_interp["y_smooth"],
                mode="lines",
                line=dict(color="#2563eb", width=3),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=asset_interp["x_plot_values"],
                y=asset_interp["y_values"],
                mode="markers",
                marker=dict(size=7, color="#1d4ed8", symbol="diamond"),
                text=asset_interp["labels"],
                hovertemplate="Maturity=%{text}<br>" + f"{asset_label} return=%{{y:.2f}}%<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    for benchmark_symbol, benchmark_payload in benchmark_payloads.items():
        benchmark_interp = benchmark_payload.get(benchmark_key)
        if benchmark_interp is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=benchmark_interp["x_plot_smooth"],
                y=benchmark_interp["y_smooth"],
                mode="lines",
                line=dict(color=benchmark_payload["color"], width=2.2, dash="dash"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_interp["x_plot_values"],
                y=benchmark_interp["y_values"],
                mode="markers",
                marker=dict(size=6, color=benchmark_payload["color"], symbol="circle-open"),
                text=benchmark_interp["labels"],
                hovertemplate="Maturity=%{text}<br>" + f"{benchmark_symbol} return=%{{y:.2f}}%<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _load_fred_key() -> str | None:
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        return fred_key
    if not os.path.exists(".env"):
        return None
    with open(".env", "r", encoding="utf-8") as env_file:
        for line in env_file:
            if line.startswith("FRED_API_KEY="):
                return line.strip().split("=", 1)[1]
    return None


def _build_volatility_model_figure(
    *,
    ticker: pd.DataFrame,
    ticker_str: str,
    time_frame_map: dict[str, int],
    rolling: Rolling,
) -> go.Figure:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("The 'arch' package is required for volatility model comparisons.") from exc

    close_series = ticker["Close"]
    ohlc_frame = ticker[["Open", "High", "Low", "Close"]].copy()
    returns = close_series.pct_change()
    garch_input = returns.dropna() * 100

    log_hl = np.log(ohlc_frame["High"] / ohlc_frame["Low"])
    log_ho = np.log(ohlc_frame["High"] / ohlc_frame["Open"])
    log_lo = np.log(ohlc_frame["Low"] / ohlc_frame["Open"])
    log_co = np.log(ohlc_frame["Close"] / ohlc_frame["Open"])
    log_oc = np.log(ohlc_frame["Open"] / ohlc_frame["Close"].shift(1))
    log_hc = np.log(ohlc_frame["High"] / ohlc_frame["Close"])
    log_lc = np.log(ohlc_frame["Low"] / ohlc_frame["Close"])

    garman_klass_variance = 0.5 * (log_hl ** 2) - ((2 * np.log(2)) - 1) * (log_co ** 2)
    parkinson_variance = (log_hl ** 2) / (4 * np.log(2))
    rs_variance = (log_hc * log_ho) + (log_lc * log_lo)

    volatility_model_specs = [
        ("GARCH(1,1)", dict(vol="GARCH", p=1, q=1, o=0), "#111111", "solid"),
        ("EGARCH(1,1)", dict(vol="EGARCH", p=1, o=1, q=1), "#d62728", "dash"),
        ("GJR-GARCH(1,1)", dict(vol="GARCH", p=1, o=1, q=1), "#2ca02c", "dot"),
    ]
    rolling_realized_vol_specs = [
        ("Close-to-Close", "close-to-close", "#1f77b4", "solid"),
        ("Parkinson", "parkinson", "#9467bd", "dash"),
        ("Yang-Zhang", "yang-zhang", "#ff7f0e", "dot"),
        ("Garman-Klass", "garman-klass", "#8c564b", "dashdot"),
        ("Rogers-Satchell", "rogers-satchell", "#17becf", "longdash"),
    ]
    ewma_realized_vol_specs = [
        ("EWMA Close-to-Close", "close-to-close", "#1f77b4", "longdashdot"),
        ("EWMA Parkinson", "parkinson", "#9467bd", "longdashdot"),
        ("EWMA Yang-Zhang", "yang-zhang", "#ff7f0e", "longdashdot"),
        ("EWMA Garman-Klass", "garman-klass", "#8c564b", "longdashdot"),
        ("EWMA Rogers-Satchell", "rogers-satchell", "#17becf", "longdashdot"),
    ]

    annualized_model_vols = {}
    for model_name, model_kwargs, _, _ in volatility_model_specs:
        model_fit = arch_model(garch_input, mean="Zero", dist="normal", rescale=False, **model_kwargs).fit(disp="off")
        annualized_model_vols[model_name] = (model_fit.conditional_volatility / 100.0) * np.sqrt(252)

    def compute_rolling_realized_vol_series(window, method):
        if method == "close-to-close":
            return returns.rolling(window).std() * np.sqrt(252)
        return rolling.volatility(ohlc_frame, windows=(window,), method=method).iloc[:, 0]

    def compute_ewma_realized_vol_series(window, method):
        alpha = 2.0 / (window + 1.0)
        if method == "close-to-close":
            return np.sqrt(returns.pow(2).ewm(alpha=alpha, adjust=False, min_periods=window).mean() * 252)
        if method == "parkinson":
            return np.sqrt(parkinson_variance.ewm(alpha=alpha, adjust=False, min_periods=window).mean().clip(lower=0) * 252)
        if method == "garman-klass":
            return np.sqrt(garman_klass_variance.ewm(alpha=alpha, adjust=False, min_periods=window).mean().clip(lower=0) * 252)
        if method == "rogers-satchell":
            return np.sqrt(rs_variance.ewm(alpha=alpha, adjust=False, min_periods=window).mean().clip(lower=0) * 252)
        if window < 2:
            return pd.Series(np.nan, index=ohlc_frame.index)
        k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
        overnight_variance = log_oc.ewm(alpha=alpha, adjust=False, min_periods=window).var(bias=False)
        open_to_close_variance = log_co.ewm(alpha=alpha, adjust=False, min_periods=window).var(bias=False)
        rs_component = rs_variance.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
        yz_variance = (overnight_variance + (k * open_to_close_variance) + ((1 - k) * rs_component)).clip(lower=0)
        return np.sqrt(yz_variance * 252)

    def smooth_annualized_model_vol(model_vol, window):
        return np.sqrt(model_vol.pow(2).rolling(window).mean())

    volatility_term_order = [term for term in time_frame_map if time_frame_map.get(term) is not None]
    default_vol_term = "long" if "long" in volatility_term_order else max(volatility_term_order, key=lambda term: int(time_frame_map[term]))

    vol_model_fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Annualized GARCH Family vs Close-to-Close Volatility (Raw + Smoothed)",
            "Annualized Rolling and EWMA Volatility Estimators",
            "Spread vs Close-to-Close Annualized Volatility",
            "Annualized Rolling Minus EWMA Volatility Estimators",
        ),
    )
    term_trace_bounds = {}
    term_ranges = {}

    for term in volatility_term_order:
        window = int(time_frame_map[term])
        rolling_realized_vol_map = {label: compute_rolling_realized_vol_series(window, method) for label, method, _, _ in rolling_realized_vol_specs}
        ewma_realized_vol_map = {label: compute_ewma_realized_vol_series(window, method) for label, method, _, _ in ewma_realized_vol_specs}
        baseline_vol = rolling_realized_vol_map["Close-to-Close"]
        visible = term == default_vol_term
        term_trace_start = len(vol_model_fig.data)

        vol_model_fig.add_trace(
            go.Scatter(x=baseline_vol.index, y=baseline_vol, mode="lines", name=f"Close-to-Close ({window}-day)", line=dict(color="#1f77b4", width=2), visible=visible, showlegend=visible, legendgroup="close-to-close-baseline"),
            row=1,
            col=1,
        )

        for model_name, _, color, dash in volatility_model_specs:
            model_vol = annualized_model_vols[model_name]
            smoothed_model_vol = smooth_annualized_model_vol(model_vol, window)
            model_spread = (model_vol - baseline_vol).dropna()
            smoothed_model_spread = (smoothed_model_vol - baseline_vol).dropna()

            vol_model_fig.add_trace(go.Scatter(x=model_vol.index, y=model_vol, mode="lines", name=f"Annualized {model_name}", line=dict(color=color, width=2, dash=dash), visible=visible, showlegend=visible, legendgroup=model_name), row=1, col=1)
            vol_model_fig.add_trace(go.Scatter(x=smoothed_model_vol.index, y=smoothed_model_vol, mode="lines", name=f"{model_name} Smoothed ({window}-day)", line=dict(color=color, width=3, dash="longdash"), opacity=0.9, visible=visible, showlegend=visible, legendgroup=f"{model_name}-smoothed"), row=1, col=1)
            vol_model_fig.add_trace(go.Scatter(x=model_spread.index, y=model_spread, mode="lines", name=f"{model_name} Spread", line=dict(color=color, width=2, dash=dash), visible=visible, showlegend=False, legendgroup=f"{model_name}-spread"), row=3, col=1)
            vol_model_fig.add_trace(go.Scatter(x=smoothed_model_spread.index, y=smoothed_model_spread, mode="lines", name=f"{model_name} Smoothed Spread", line=dict(color=color, width=3, dash="longdash"), opacity=0.9, visible=visible, showlegend=False, legendgroup=f"{model_name}-smoothed-spread"), row=3, col=1)

        for label, _, color, dash in rolling_realized_vol_specs:
            realized_vol = rolling_realized_vol_map[label]
            vol_model_fig.add_trace(go.Scatter(x=realized_vol.index, y=realized_vol, mode="lines", name=f"{label} ({window}-day)", line=dict(color=color, width=2, dash=dash), visible=visible, showlegend=False, legendgroup=f"rolling-realized-{label}"), row=2, col=1)
            if label != "Close-to-Close":
                realized_spread = (realized_vol - baseline_vol).dropna()
                vol_model_fig.add_trace(go.Scatter(x=realized_spread.index, y=realized_spread, mode="lines", name=f"{label} Spread", line=dict(color=color, width=2, dash=dash), visible=visible, showlegend=False, legendgroup=f"rolling-realized-spread-{label}"), row=3, col=1)

        for label, method, color, dash in ewma_realized_vol_specs:
            ewma_vol = ewma_realized_vol_map[label]
            vol_model_fig.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, mode="lines", name=f"{label} ({window}-day)", line=dict(color=color, width=2, dash=dash), opacity=0.9, visible=visible, showlegend=False, legendgroup=f"ewma-realized-{label}"), row=2, col=1)
            ewma_spread = (ewma_vol - baseline_vol).dropna()
            vol_model_fig.add_trace(go.Scatter(x=ewma_spread.index, y=ewma_spread, mode="lines", name=f"{label} Spread", line=dict(color=color, width=2, dash=dash), opacity=0.9, visible=visible, showlegend=False, legendgroup=f"ewma-realized-spread-{label}"), row=3, col=1)

        for (rolling_label, _, color, _), (ewma_label, _, _, _) in zip(rolling_realized_vol_specs, ewma_realized_vol_specs):
            realized_minus_ewma = (rolling_realized_vol_map[rolling_label] - ewma_realized_vol_map[ewma_label]).dropna()
            vol_model_fig.add_trace(go.Scatter(x=realized_minus_ewma.index, y=realized_minus_ewma, mode="lines", name=f"{rolling_label} Minus {ewma_label} ({window}-day)", line=dict(color=color, width=3), opacity=0.9, visible=visible, showlegend=False, legendgroup=f"realized-minus-ewma-{rolling_label}"), row=4, col=1)

        term_trace_bounds[term] = (term_trace_start, len(vol_model_fig.data))
        non_empty_series = [
            series
            for series in (
                [series.dropna() for series in rolling_realized_vol_map.values()]
                + [series.dropna() for series in ewma_realized_vol_map.values()]
                + [model_series.dropna() for model_series in annualized_model_vols.values()]
            )
            if not series.empty
        ]
        if non_empty_series:
            max_index = max(series.index.max() for series in non_empty_series)
            min_index = min(series.index.min() for series in non_empty_series)
            term_ranges[term] = [max(min_index, max_index - pd.DateOffset(years=3)), max_index]
        else:
            term_ranges[term] = None

    vol_model_fig.add_hline(y=0, line_dash="dot", line_color="rgba(120, 120, 120, 0.8)", row=3, col=1)
    vol_model_fig.add_hline(y=0, line_dash="dot", line_color="rgba(120, 120, 120, 0.8)", row=4, col=1)

    vol_buttons = []
    total_traces = len(vol_model_fig.data)
    for term in volatility_term_order:
        visibility = [False] * total_traces
        start, end = term_trace_bounds[term]
        for trace_idx in range(start, end):
            visibility[trace_idx] = True
        layout_updates = {
            "title": {"text": f"{ticker_str} Volatility Models and Estimators ({time_frame_map[term]}-Day)", "x": 0.5, "xanchor": "center", "y": 0.97, "yanchor": "top"},
            "yaxis": {"title": "Annualized Volatility"},
            "yaxis2": {"title": "Annualized Volatility"},
            "yaxis3": {"title": "Spread vs Close-to-Close"},
            "yaxis4": {"title": "Rolling Minus EWMA"},
        }
        if term_ranges.get(term) is not None:
            for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
                layout_updates[axis_name] = {"range": term_ranges[term]}
        vol_buttons.append(dict(label=f"{term.title()} ({time_frame_map[term]})", method="update", args=[{"visible": visibility}, layout_updates]))

    available_ranges = [date_range for date_range in term_ranges.values() if date_range is not None]
    if available_ranges:
        global_start = min(date_range[0] for date_range in available_ranges)
        global_end = max(date_range[1] for date_range in available_ranges)
        default_range = term_ranges[default_vol_term] or [global_start, global_end]
        for row in (1, 2, 3, 4):
            vol_model_fig.update_xaxes(range=default_range, row=row, col=1)
        time_range_menu = dict(type="dropdown", buttons=build_time_range_buttons(global_start, global_end, axis_count=4), x=0.28, xanchor="left", y=1.08, yanchor="top", showactive=True)
    else:
        time_range_menu = None

    vol_model_fig.update_layout(
        template="plotly_white",
        height=1450,
        margin=dict(t=150),
        legend=dict(x=0.01, y=0.99),
        yaxis_title="Annualized Volatility",
        yaxis2_title="Annualized Volatility",
        yaxis3_title="Spread vs Close-to-Close",
        yaxis4_title="Rolling Minus EWMA",
        title=dict(text=f"{ticker_str} Volatility Models and Estimators ({time_frame_map[default_vol_term]}-Day)", x=0.5, xanchor="center", y=0.97, yanchor="top"),
        updatemenus=[dict(type="dropdown", buttons=vol_buttons, x=0.0, xanchor="left", y=1.08, yanchor="top", showactive=True, active=volatility_term_order.index(default_vol_term))] + ([time_range_menu] if time_range_menu is not None else []),
    )
    return vol_model_fig

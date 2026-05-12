"""Portfolio performance summary views."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go

from Quantapp.analytics.series_utils import calculate_zscore


def zscore_for_plot(series) -> pd.Series:
    cleaned = pd.Series(series).dropna().sort_index()
    if cleaned.empty:
        return pd.Series(dtype=float)
    zscore_series = calculate_zscore(cleaned)
    if zscore_series.isna().all():
        return pd.Series(0.0, index=cleaned.index)
    return zscore_series.dropna()


def add_zscore_style_guides(fig: go.Figure) -> None:
    fig.add_hrect(y0=-2, y1=-1, fillcolor="rgba(0, 128, 0, 0.30)", line_width=0, layer="below")
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(211, 211, 211, 0.18)", line_width=0, layer="below")
    fig.add_hrect(y0=1, y1=2, fillcolor="rgba(180, 0, 0, 0.30)", line_width=0, layer="below")
    fig.add_hline(y=0, line=dict(color="rgba(255, 255, 255, 0.80)", width=1))
    for level in (0.5, 1, 1.5, 2):
        fig.add_hline(y=level, line=dict(color="rgba(220, 220, 220, 0.55)", width=1, dash="dot"))
        fig.add_hline(y=-level, line=dict(color="rgba(220, 220, 220, 0.55)", width=1, dash="dot"))


def plot_equity_curve(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    *,
    benchmark_label: str = "SPY",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity, mode="lines", name="Portfolio Equity Curve"))
    fig.add_trace(
        go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity,
            mode="lines",
            name=f"Benchmark ({benchmark_label}) Equity Curve",
        )
    )
    fig.update_layout(
        title="Portfolio vs Benchmark Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity Curve",
        template="plotly_dark",
    )
    return fig


def plot_rolling_sharpe_zscore(
    portfolio_sharpe_by_window: Mapping[int, pd.Series],
    benchmark_sharpe_by_window: Mapping[int, pd.Series],
    *,
    benchmark_label: str = "SPY",
    default_window: int = 200,
) -> go.Figure:
    windows = list(portfolio_sharpe_by_window.keys())
    if default_window not in windows and windows:
        default_window = windows[-1]

    fig = go.Figure()
    for window in windows:
        visible = window == default_window
        portfolio_z = zscore_for_plot(portfolio_sharpe_by_window[window])
        benchmark_z = zscore_for_plot(benchmark_sharpe_by_window[window])
        fig.add_trace(
            go.Scatter(
                x=portfolio_z.index,
                y=portfolio_z,
                mode="lines",
                name=f"Portfolio Rolling Sharpe Z-Score ({window} days)",
                visible=visible,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_z.index,
                y=benchmark_z,
                mode="lines",
                name=f"Benchmark ({benchmark_label}) Rolling Sharpe Z-Score ({window} days)",
                line=dict(dash="dot"),
                visible=visible,
            )
        )

    add_zscore_style_guides(fig)
    buttons = []
    for index, window in enumerate(windows):
        visible = [False] * (2 * len(windows))
        visible[index * 2] = True
        visible[index * 2 + 1] = True
        buttons.append(
            dict(
                args=[{"visible": visible}, {"title": f"Portfolio vs Benchmark Rolling Sharpe Z-Score ({window} days)"}],
                label=f"{window} Days",
                method="update",
            )
        )

    fig.update_layout(
        title=f"Portfolio vs Benchmark Rolling Sharpe Z-Score ({default_window} days)",
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_dark",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                active=windows.index(default_window) if default_window in windows else 0,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    return fig


def plot_rolling_correlation(
    correlation_by_window: Mapping[int, pd.Series],
    *,
    benchmark_label: str = "SPY",
) -> go.Figure:
    windows = list(correlation_by_window.keys())
    fig = go.Figure()
    for index, window in enumerate(windows):
        series = correlation_by_window[window]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=f"Portfolio Rolling Correlation ({window} days)",
                visible=index == 0,
            )
        )

    buttons = []
    for index, window in enumerate(windows):
        visible = [False] * len(windows)
        visible[index] = True
        buttons.append(dict(args=[{"visible": visible}], label=f"{window} Days", method="update"))

    fig.update_layout(
        title=f"Portfolio Rolling Correlation vs Benchmark ({benchmark_label})",
        xaxis_title="Date",
        yaxis_title="Rolling Correlation",
        template="plotly_dark",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
    )
    fig.add_hline(y=0, line=dict(color="Red", width=2, dash="dash"))
    return fig


def _range_start(series: pd.Series, trading_days: int):
    if series.empty:
        return None
    offset = min(len(series), trading_days)
    return series.index[-offset]


def plot_rolling_sortino(
    portfolio_sortino: pd.Series,
    asset_sortino_by_ticker: Mapping[str, pd.Series] | None = None,
    *,
    portfolio_name: str = "Portfolio",
    default_trading_days: int = 756,
) -> go.Figure:
    portfolio_sortino = pd.Series(portfolio_sortino).dropna().sort_index()
    fig = go.Figure()

    if portfolio_sortino.empty:
        fig.update_layout(
            title=f"Rolling Sortino Ratio of {portfolio_name}",
            xaxis_title="Date",
            yaxis_title="Rolling Sortino Ratio",
            legend_title="Ticker",
            template="plotly_dark",
            height=1000,
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=portfolio_sortino.index,
            y=portfolio_sortino,
            mode="lines",
            name=f"{portfolio_name} Sortino",
            line=dict(width=4, dash="dash"),
        )
    )

    mean_val = portfolio_sortino.mean()
    std_val = portfolio_sortino.std()
    center_x = portfolio_sortino.index[len(portfolio_sortino) // 2]

    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="Green",
        line_width=2,
        annotation_text="Mean",
        annotation_position="top left",
    )

    if pd.notna(std_val) and std_val != 0:
        std_levels = {
            1: {"color": "yellow", "opacity": 0.5},
            2: {"color": "LightCoral", "opacity": 0.5},
        }
        for level in range(-3, 4):
            if level == 0:
                continue
            fig.add_hline(
                y=mean_val + level * std_val,
                line_dash="dash",
                line_color="Red",
                line_width=2,
                annotation_text=f"{level:+d} sigma",
                annotation_position="top left",
            )

        for level, style in std_levels.items():
            fig.add_shape(
                type="rect",
                x0=portfolio_sortino.index.min(),
                x1=portfolio_sortino.index.max(),
                y0=mean_val + level * std_val,
                y1=mean_val + (level + 1) * std_val,
                line=dict(color="Red", width=2, dash="dash"),
                fillcolor=style["color"],
                opacity=style["opacity"],
            )
            fig.add_shape(
                type="rect",
                x0=portfolio_sortino.index.min(),
                x1=portfolio_sortino.index.max(),
                y0=mean_val - (level + 1) * std_val,
                y1=mean_val - level * std_val,
                line=dict(color="Red", width=2, dash="dash"),
                fillcolor=style["color"],
                opacity=style["opacity"],
            )
            fig.add_annotation(
                x=center_x,
                y=mean_val + (level + 0.5) * std_val,
                text=f"Portfolio +{level} sigma to +{level + 1} sigma",
                showarrow=False,
                yshift=10,
            )
            fig.add_annotation(
                x=center_x,
                y=mean_val - (level + 0.5) * std_val,
                text=f"Portfolio -{level + 1} sigma to -{level} sigma",
                showarrow=False,
                yshift=10,
            )

    fig.add_annotation(
        x=center_x,
        y=mean_val,
        text=f"Portfolio Mean: {mean_val:.2f}",
        showarrow=False,
        yshift=10,
    )

    for ticker, rolling_sortino in (asset_sortino_by_ticker or {}).items():
        rolling_sortino = pd.Series(rolling_sortino).dropna().sort_index()
        if rolling_sortino.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rolling_sortino.index,
                y=rolling_sortino,
                mode="lines",
                name=f"{ticker} Sortino",
                line=dict(width=2),
            )
        )

    fig.add_hline(y=0, line=dict(color="Red", width=2))

    end = portfolio_sortino.index[-1]
    range_options = [
        (252, "1 Year"),
        (756, "3 Years"),
        (1260, "5 Years"),
        (2520, "10 Years"),
    ]
    buttons = [
        dict(
            args=[{"xaxis.range": [_range_start(portfolio_sortino, trading_days), end]}],
            label=label,
            method="relayout",
        )
        for trading_days, label in range_options
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.25,
                yanchor="top",
                active=1,
            )
        ]
    )
    fig.update_xaxes(range=[_range_start(portfolio_sortino, default_trading_days), end])
    fig.update_layout(
        title=f"Rolling Sortino Ratio of {portfolio_name}",
        xaxis_title="Date",
        yaxis_title="Rolling Sortino Ratio",
        legend_title="Ticker",
        template="plotly_dark",
        height=1000,
    )
    return fig

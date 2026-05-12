"""Rolling factor-regression views."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _zero_line_shape(index):
    return dict(
        type="line",
        x0=index[0],
        y0=0,
        x1=index[-1],
        y1=0,
        line=dict(color="white", width=1, dash="dash"),
    )


def _plot_alpha(rolling_results, ticker_label):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_results.index,
            y=rolling_results["alpha"],
            mode="lines",
            name="Alpha",
        )
    )
    fig.add_shape(_zero_line_shape(rolling_results.index))

    mean_alpha = rolling_results["alpha"].mean()
    std_alpha = rolling_results["alpha"].std()
    fig.add_hline(
        y=mean_alpha,
        line_color="white",
        line_dash="dot",
        annotation_text=f"Mean: {mean_alpha:.2f}",
        annotation_position="bottom right",
    )
    for multiplier in [1, 1.5, 2, 3]:
        fig.add_hline(
            y=mean_alpha + multiplier * std_alpha,
            line_color="red",
            line_dash="dash",
            annotation_text=f"+{multiplier} sigma: {mean_alpha + multiplier * std_alpha:.2f}",
            annotation_position="top right",
        )
        fig.add_hline(
            y=mean_alpha - multiplier * std_alpha,
            line_color="green",
            line_dash="dash",
            annotation_text=f"-{multiplier} sigma: {mean_alpha - multiplier * std_alpha:.2f}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"{ticker_label} Rolling Alpha",
        xaxis_title="Date",
        yaxis_title="Alpha",
        template="plotly_dark",
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
            tickangle=-45,
            showgrid=True,
            zeroline=False,
        ),
    )
    return fig


def _plot_betas(rolling_results, ticker_label, factor_returns):
    factors_to_plot = [factor for factor in factor_returns.columns if factor != "RF"]
    num_factors = len(factors_to_plot)
    fig = make_subplots(
        rows=max(num_factors, 1),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{factor} Beta" for factor in factors_to_plot],
    )

    for row_idx, factor in enumerate(factors_to_plot, start=1):
        fig.add_trace(
            go.Scatter(
                x=rolling_results.index,
                y=rolling_results[f"{factor}_beta"],
                mode="lines",
                name=f"{factor} Beta",
            ),
            row=row_idx,
            col=1,
        )
        baseline = 1 if factor == "Mkt-RF" else 0
        fig.add_hline(
            y=baseline,
            row=row_idx,
            col=1,
            line=dict(color="white", dash="dash"),
            annotation_text=f"Baseline: {baseline}",
            annotation_position="bottom right",
        )

    beta_row_height = 220
    beta_height = min(max(520, beta_row_height * max(num_factors, 1) + 80), 1200)
    fig.update_layout(
        title=f"{ticker_label} Rolling Betas",
        template="plotly_dark",
        height=beta_height,
        showlegend=False,
    )
    return fig


def _plot_r_squared(rolling_results, ticker_label):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_results.index,
            y=rolling_results["r_squared"],
            mode="lines",
            name="R-Squared",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_results.index,
            y=rolling_results["adj_r_squared"],
            mode="lines",
            name="Adjusted R-Squared",
        )
    )
    mean_r_squared = rolling_results["r_squared"].mean()
    fig.add_hline(
        y=mean_r_squared,
        line_color="white",
        line_dash="dot",
        annotation_text=f"Mean: {mean_r_squared:.2f}",
        annotation_position="bottom right",
    )
    fig.add_shape(_zero_line_shape(rolling_results.index))
    fig.update_layout(
        title=f"{ticker_label} Rolling R-Squared",
        xaxis_title="Date",
        yaxis_title="R-Squared",
        template="plotly_dark",
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
            tickangle=-45,
            showgrid=True,
            zeroline=False,
        ),
    )
    return fig


def plot_rolling_regression_view(rolling_results, ticker_label, factor_returns):
    """Compose rolling alpha, beta, and R-squared factor-regression figures."""
    return {
        "alpha": _plot_alpha(rolling_results, ticker_label),
        "betas": _plot_betas(rolling_results, ticker_label, factor_returns),
        "r_squared": _plot_r_squared(rolling_results, ticker_label),
    }

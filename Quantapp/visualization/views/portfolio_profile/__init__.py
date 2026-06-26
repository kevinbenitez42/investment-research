"""Portfolio Profile notebook visualization views."""

from .performance_structure import (
    build_option_expiration_pl_figure,
    build_option_max_loss_by_underlying_figure,
    build_option_profit_loss_extremes_table,
    display_option_expiration_pl_view,
    format_snapshot_map,
    plot_benchmark_snapshot_zscores,
    plot_equity_curve,
    plot_options_expiration_ladder,
    plot_rolling_correlation,
    plot_rolling_portfolio_allocation,
    plot_rolling_portfolio_allocation_stacked,
    plot_rolling_sharpe_zscore,
    plot_rolling_sortino,
    plot_z_score_diff_dropdown,
)

__all__ = [
    "build_option_expiration_pl_figure",
    "build_option_max_loss_by_underlying_figure",
    "build_option_profit_loss_extremes_table",
    "display_option_expiration_pl_view",
    "format_snapshot_map",
    "plot_benchmark_snapshot_zscores",
    "plot_equity_curve",
    "plot_options_expiration_ladder",
    "plot_rolling_correlation",
    "plot_rolling_portfolio_allocation",
    "plot_rolling_portfolio_allocation_stacked",
    "plot_rolling_sharpe_zscore",
    "plot_rolling_sortino",
    "plot_z_score_diff_dropdown",
]

"""Inline options-pricing dashboard extracted from notebook Blocks 1-15A.

The research notebook executes this module and retains Blocks 16 onward.
"""


# %% [extracted notebook cell 1]
# Block 1: Notebook Description
#Notebook description

#This notebook is being used to evaluate the techinical market conditions of a single asset and assess
#the appropriate strategy to take in order to maximize returns.


# %% [extracted notebook cell 2]
# Block 2: Imports and Project Bootstrap
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
# Seed the local package import when the notebook starts in a subfolder.
for _project_root_candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
    if (_project_root_candidate / "Quantapp" / "project.py").exists():
        if str(_project_root_candidate) not in sys.path:
            sys.path.insert(0, str(_project_root_candidate))
        break
else:
    raise RuntimeError("Could not locate the project root containing Quantapp.")

from Quantapp.project import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()
SINGLE_ASSET_DIRECTORY = PROJECT_ROOT / "Research" / "Single Asset"
if str(SINGLE_ASSET_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SINGLE_ASSET_DIRECTORY))
from _params import get_single_asset_params


from Quantapp.data import yf as qa_yf
from Quantapp.visualization import (
    BarChartPlotter,
    Plotter,
    build_time_range_buttons,
)
from Quantapp.visualization.views.single_asset_profile.pricing.options_pricing import (
    build_current_atm_iv_by_expiration,
    build_historical_atm_iv_history,
    build_historical_iv_premium_history,
    build_atm_implied_move_term_structure,
    plot_atm_iv_realized_view,
    plot_atm_implied_move_term_structure_view,
    plot_gbm_paths_view,
    plot_current_atm_iv_spread_view,
    plot_historical_atm_iv_summary_view,
    plot_historical_implied_move_candlestick_view,
    plot_historical_iv_premium_view,
    plot_historical_iv_rank_percentile_view,
    plot_implied_volatility_by_strike_view,
    plot_iv_minus_realized_by_strike_view,
    plot_median_iv_minus_realized_view,
    plot_open_interest_overview_view,
    plot_open_interest_implied_move_ranges_view,
    plot_option_chain_table_view,
    plot_svi_surface_view,
)
from Quantapp.visualization.views.single_asset_profile.pricing.distribution import (
    plot_distribution_shape_zscores_view,
    plot_fixed_payout_strategy_backtest_view,
    plot_trade_range_history_profile,
    plot_trade_range_probability_cone,
    plot_trade_range_breach_average_view,
    plot_trade_range_breach_excess_view,
    plot_trade_range_stack_view,
    plot_volatility_model_comparison_view,
)
from scipy.stats import kurtosis, skew

from Quantapp.analytics import Helper, SeriesTransforms
from Quantapp.analytics import compute
from Quantapp.analytics.series_utils import (
    calculate_historical_var_metrics,
    calculate_textbook_rolling_max_drawdown,
    calculate_zscore,
    coerce_close_series,
    gini_coefficient,
)
from Quantapp.data import (
    MacroDataClient,
    align_series_to_common_index,
    get_current_options_chain,
    get_historical_options_eod_panel,
    get_market_history,
    load_benchmark_data,
    normalize_benchmark_tickers,
)
from Quantapp.secrets import load_project_env, require_secret

load_project_env()

warnings.filterwarnings("ignore")
logger = logging.getLogger("yfinance")

# Use one dark visual system for every Plotly figure in this notebook.
NOTEBOOK_PLOT_TEMPLATE = 'plotly_dark'
NOTEBOOK_PLOT_BACKGROUND = '#111827'
NOTEBOOK_PLOT_GRID = '#374151'
pio.templates.default = NOTEBOOK_PLOT_TEMPLATE

def apply_notebook_plot_theme(fig):
    fig.update_layout(
        template=NOTEBOOK_PLOT_TEMPLATE,
        paper_bgcolor=NOTEBOOK_PLOT_BACKGROUND,
        plot_bgcolor=NOTEBOOK_PLOT_BACKGROUND,
        font=dict(color='#E5E7EB'),
        legend=dict(bgcolor='rgba(17, 24, 39, 0.75)'),
    )
    fig.update_xaxes(gridcolor=NOTEBOOK_PLOT_GRID, zerolinecolor='#6B7280')
    fig.update_yaxes(gridcolor=NOTEBOOK_PLOT_GRID, zerolinecolor='#6B7280')

    # Plotly table cells use explicit fills, so the base template cannot recolor them.
    table_traces = [trace for trace in fig.data if trace.type == 'table']
    table_header_colors = ('#0F766E', '#9F1239')
    for table_number, trace in enumerate(table_traces):
        dark_cell_colors = [
            [
                'rgba(34, 197, 94, 0.45)' if '144, 238, 144' in str(color) else '#111827'
                for color in column_colors
            ]
            for column_colors in trace.cells.fill.color
        ]
        trace.update(
            header=dict(
                fill_color=table_header_colors[table_number % len(table_header_colors)],
                font=dict(color='#F9FAFB', size=11),
            ),
            cells=dict(
                fill_color=dark_cell_colors,
                font=dict(color='#E5E7EB'),
            ),
        )

    return fig

INSANE_SPIKE_SMOOTHING_CONFIG = {
    'price_columns': ('Open', 'High', 'Low', 'Close', 'Adj Close'),
    'rolling_window': 63,
    'absolute_log_return_threshold': np.log1p(0.35),
    'endpoint_absolute_log_return_threshold': np.log1p(0.75),
    'robust_z_threshold': 12.0,
    'endpoint_robust_z_threshold': 20.0,
    'interpolation_limit': 3,
}

def _robust_centered_mad_zscore(series, window):
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    min_periods = max(10, int(window) // 4)
    rolling_median = values.rolling(window, center=True, min_periods=min_periods).median()
    rolling_mad = values.sub(rolling_median).abs().rolling(
        window,
        center=True,
        min_periods=min_periods,
    ).median()
    scale = (1.4826 * rolling_mad).replace(0, np.nan)
    return values.sub(rolling_median).div(scale).replace([np.inf, -np.inf], np.nan)

def _isolated_price_spike_mask(
    series,
    *,
    rolling_window=None,
    absolute_log_return_threshold=None,
    endpoint_absolute_log_return_threshold=None,
    robust_z_threshold=None,
    endpoint_robust_z_threshold=None,
):
    config = INSANE_SPIKE_SMOOTHING_CONFIG
    rolling_window = int(rolling_window or config['rolling_window'])
    absolute_log_return_threshold = float(
        absolute_log_return_threshold or config['absolute_log_return_threshold']
    )
    endpoint_absolute_log_return_threshold = float(
        endpoint_absolute_log_return_threshold
        or config['endpoint_absolute_log_return_threshold']
    )
    robust_z_threshold = float(robust_z_threshold or config['robust_z_threshold'])
    endpoint_robust_z_threshold = float(
        endpoint_robust_z_threshold or config['endpoint_robust_z_threshold']
    )

    clean = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    log_values = np.log(clean.where(clean > 0))
    inbound_return = log_values.diff()
    outbound_return = log_values.shift(-1).sub(log_values)
    inbound_z = _robust_centered_mad_zscore(inbound_return, rolling_window)
    outbound_z = _robust_centered_mad_zscore(outbound_return, rolling_window)

    inbound_extreme = (
        inbound_return.abs().gt(absolute_log_return_threshold)
        | inbound_z.abs().gt(robust_z_threshold)
    )
    outbound_extreme = (
        outbound_return.abs().gt(absolute_log_return_threshold)
        | outbound_z.abs().gt(robust_z_threshold)
    )
    isolated_reversal = (
        inbound_extreme
        & outbound_extreme
        & inbound_return.notna()
        & outbound_return.notna()
        & np.sign(inbound_return).ne(np.sign(outbound_return))
    )

    endpoint_spike = (
        outbound_return.isna()
        & inbound_return.notna()
        & (
            inbound_return.abs().gt(endpoint_absolute_log_return_threshold)
            | inbound_z.abs().gt(endpoint_robust_z_threshold)
        )
    )
    return (isolated_reversal | endpoint_spike).fillna(False)

def smooth_insane_price_spikes(
    price_frame,
    *,
    label=None,
    price_columns=None,
    rolling_window=None,
    absolute_log_return_threshold=None,
    endpoint_absolute_log_return_threshold=None,
    robust_z_threshold=None,
    endpoint_robust_z_threshold=None,
    interpolation_limit=None,
):
    if price_frame is None or price_frame.empty:
        return price_frame, pd.DataFrame(columns=['column', 'spike_count'])

    config = INSANE_SPIKE_SMOOTHING_CONFIG
    price_columns = tuple(price_columns or config['price_columns'])
    interpolation_limit = int(interpolation_limit or config['interpolation_limit'])
    smoothed = price_frame.copy()
    report_rows = []
    interpolation_method = 'time' if isinstance(smoothed.index, pd.DatetimeIndex) else 'linear'

    for column in price_columns:
        if column not in smoothed.columns:
            continue
        original = pd.to_numeric(smoothed[column], errors='coerce')
        spike_mask = _isolated_price_spike_mask(
            original,
            rolling_window=rolling_window,
            absolute_log_return_threshold=absolute_log_return_threshold,
            endpoint_absolute_log_return_threshold=endpoint_absolute_log_return_threshold,
            robust_z_threshold=robust_z_threshold,
            endpoint_robust_z_threshold=endpoint_robust_z_threshold,
        )
        if not bool(spike_mask.any()):
            continue

        cleaned = original.mask(spike_mask)
        cleaned = cleaned.interpolate(
            method=interpolation_method,
            limit=interpolation_limit,
            limit_direction='both',
        ).ffill().bfill()
        smoothed[column] = cleaned.where(original.notna(), original)
        report_rows.append({'column': column, 'spike_count': int(spike_mask.sum())})

    if {'Open', 'High', 'Low', 'Close'}.issubset(smoothed.columns):
        smoothed['High'] = pd.concat(
            [smoothed['High'], smoothed['Open'], smoothed['Close']],
            axis=1,
        ).max(axis=1)
        smoothed['Low'] = pd.concat(
            [smoothed['Low'], smoothed['Open'], smoothed['Close']],
            axis=1,
        ).min(axis=1)

    spike_report = pd.DataFrame(report_rows)
    if label and not spike_report.empty:
        total_spikes = int(spike_report['spike_count'].sum())
        adjusted_columns = ', '.join(spike_report['column'].astype(str))
        print(
            f'Smoothed {total_spikes} isolated price spike(s) in {label} '
            f'across: {adjusted_columns}'
        )
    return smoothed, spike_report

def smooth_insane_series_spikes(
    series,
    *,
    label=None,
    rolling_window=63,
    robust_z_threshold=8.0,
    interpolation_limit=5,
    floor=None,
    ceiling=None,
):
    values = pd.to_numeric(pd.Series(series).copy(), errors='coerce').replace([np.inf, -np.inf], np.nan)
    values = values.mask(values.lt(floor)) if floor is not None else values
    values = values.mask(values.gt(ceiling)) if ceiling is not None else values
    robust_z = _robust_centered_mad_zscore(values, int(rolling_window))
    spike_mask = robust_z.abs().gt(float(robust_z_threshold)).fillna(False)
    if ceiling is not None:
        spike_mask = spike_mask | pd.to_numeric(pd.Series(series), errors='coerce').gt(float(ceiling)).fillna(False)

    if not bool(spike_mask.any()):
        report = pd.DataFrame(columns=['series', 'spike_count'])
        values.name = getattr(series, 'name', None)
        return values, report

    interpolation_method = 'time' if isinstance(values.index, pd.DatetimeIndex) else 'linear'
    smoothed = values.mask(spike_mask).interpolate(
        method=interpolation_method,
        limit=int(interpolation_limit),
        limit_direction='both',
    ).ffill().bfill()
    if floor is not None:
        smoothed = smoothed.clip(lower=float(floor))
    if ceiling is not None:
        smoothed = smoothed.clip(upper=float(ceiling))
    smoothed.name = getattr(series, 'name', None)
    report = pd.DataFrame([
        {'series': label or getattr(series, 'name', 'series'), 'spike_count': int(spike_mask.sum())}
    ])
    if label:
        print(f'Smoothed {int(spike_mask.sum())} insane spike(s) in {label}.')
    return smoothed, report


# %% [extracted notebook cell 3]
# Block 3: Shared and Options-Specific Parameters

pricing_params = get_single_asset_params()
TIMEFRAME_PROFILES = {
    "swing": {"short": 3, "mid": 9, "long": 21},
    "position": {"short": 21, "mid": 50, "long": 200},
    "structural": {"short": 200, "mid": 500, "long": 1000},
}

def resolve_time_frame_map(strategy):
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in TIMEFRAME_PROFILES:
        raise ValueError(f"Invalid trading_strategy: {strategy}")
    return dict(TIMEFRAME_PROFILES[normalized_strategy])

trading_strategy = "position"
time_frame_map = resolve_time_frame_map(trading_strategy)
options_params = {
    **pricing_params,
    "risk_free_ticker": "^IRX",
    "benchmark_tickers": ["SPY"],
    "trading_strategy": trading_strategy,
    "length_of_plots": 20,
    "var_position_value": None,
    "risk_free_rate": 0.02 / 252,
    "time_frame_week": 7,
    "time_frame_short": time_frame_map["short"],
    "time_frame_mid": time_frame_map["mid"],
    "time_frame_long": time_frame_map["long"],
}

ticker_str = options_params["ticker_str"]
interval = options_params["interval"]
period = options_params["period"]
risk_free_ticker = options_params["risk_free_ticker"]
risk_free_rate = options_params["risk_free_rate"]
time_frame_week = options_params["time_frame_week"]
time_frame_short = options_params["time_frame_short"]
time_frame_mid = options_params["time_frame_mid"]
time_frame_long = options_params["time_frame_long"]
benchmark_tickers = list(options_params["benchmark_tickers"])
trading_strategy = options_params["trading_strategy"]
length_of_plots = options_params["length_of_plots"]
var_position_value = options_params["var_position_value"]

options_params


# %% [extracted notebook cell 4]
# Block 4: TODO Notes
#take all compuation functions and put them in a separate file

#simplify the date x axis on the percent drawdown chart

#default the zoom range to a comfortable range, and create a dropdown to select the time range for Volatility section

#properly label and annoate the garch models

#Remove the VIX charting, its redudnant now that we have the volatility models


# %% [extracted notebook cell 5]
# Block 5: Shared Clients

qp = Plotter()
qe = MacroDataClient()
helper = Helper()
barChartPlotter = BarChartPlotter()
series_transforms = SeriesTransforms()


# %% [extracted notebook cell 6]
# Block 6: Underlying Data Load
#Load data: underlying data
print(f"Loading data for {ticker_str} with period {period} and interval {interval}")
ticker_handle = qa_yf.Ticker(ticker_str)  # Used for options metadata/fallbacks through Quantapp.data.

market_history = get_market_history(
    symbols=[ticker_str],
    period=period,
    interval=interval,
    provider="yfinance",
    align=False,
)
ticker = market_history.get(str(ticker_str).strip().upper(), pd.DataFrame())

if ticker.empty:
    raise ValueError(f"No underlying price history returned for {ticker_str}.")

ticker_raw = ticker.copy()
ticker, ticker_spike_smoothing_report = smooth_insane_price_spikes(
    ticker_raw,
    label=ticker_str,
)
market_history[str(ticker_str).strip().upper()] = ticker

price_series = ticker['Close'].dropna()
log_returns = np.log(price_series / price_series.shift(1)).dropna()
spot_price = float(price_series.iloc[-1])

rolling_vol_window = min(len(log_returns), 252)
if rolling_vol_window < 2:
    raise ValueError(f"Not enough history to compute volatility for {ticker_str}.")

annualized_vol = log_returns.iloc[-rolling_vol_window:].std() * np.sqrt(252)


# %% [extracted notebook cell 7]
# Block 7: Expiration Dates
#Load data: Expiration Dates
print(f"Loading options expiration dates for {ticker_str}")
options_expiration_dates = pd.DataFrame(ticker_handle.options, columns=['Expiration Date'])

if options_expiration_dates.empty:
    raise ValueError(f"No listed option expirations returned for {ticker_str}.")

options_expiration_dates = options_expiration_dates.sort_values('Expiration Date').reset_index(drop=True)
options_expiration_dates['Date Till Expiration'] = (
    pd.to_datetime(options_expiration_dates['Expiration Date']) - pd.Timestamp.today().normalize()
).dt.days


# %% [extracted notebook cell 8]
# Block 8: Current Options Chain Snapshot
# Retrieve the current options chain snapshot.

print(f"Loading options chain for {ticker_str}")

call_contract_chain = {}
put_contract_chain = {}
drop_columns = ['lastTradeDate', 'contractSize', 'currency', 'percentChange', 'change']
today = pd.Timestamp.today().normalize()
# Always define a usable underlying price before selecting the options provider.
# The yfinance compatibility fallback does not consistently expose an
# ``underlying_price`` attribute on its option-chain result.
underlying_price = float(spot_price)


try:
    massive_chain = get_current_options_chain(ticker_str, fallback_underlying_price=spot_price)
    massive_chain_df = massive_chain.chain
    underlying_price = massive_chain.underlying_price
    call_contract_chain = massive_chain.calls_by_expiration
    put_contract_chain = massive_chain.puts_by_expiration
    expirations = massive_chain.expirations
    options_expiration_dates = pd.DataFrame({'Expiration Date': expirations})
    options_expiration_dates['Date Till Expiration'] = (
        pd.to_datetime(options_expiration_dates['Expiration Date']) - today
    ).dt.days

    first_expiration_date = expirations[0]
    call_contracts = call_contract_chain[first_expiration_date].copy()
    put_contracts = put_contract_chain[first_expiration_date].copy()
    underlying_data = {'regularMarketPrice': underlying_price}
    first_expiration_chain = {'underlying': underlying_data, 'calls': call_contracts, 'puts': put_contracts}

    call_contract_chain_concat = massive_chain.calls
    put_contract_chain_concat = massive_chain.puts
    all_contracts_concat = pd.concat([call_contract_chain_concat, put_contract_chain_concat], ignore_index=True)

    print(f"Loaded options chain from Massive for {ticker_str}: {len(all_contracts_concat)} contracts")

except Exception as massive_error:
    print(f"Massive load failed ({massive_error}); falling back to Quantapp.data yfinance compatibility")

    first_expiration_date = options_expiration_dates['Expiration Date'].iloc[0]
    first_expiration_chain = ticker_handle.option_chain(first_expiration_date)
    underlying_data = first_expiration_chain.underlying
    fallback_underlying_price = pd.to_numeric(
        underlying_data.get('regularMarketPrice')
        if isinstance(underlying_data, dict) else np.nan,
        errors='coerce',
    )
    if np.isfinite(fallback_underlying_price) and fallback_underlying_price > 0:
        underlying_price = float(fallback_underlying_price)
    call_contracts = first_expiration_chain.calls.copy()
    put_contracts = first_expiration_chain.puts.copy()

    for expiration_date in options_expiration_dates['Expiration Date']:
        option_chain = ticker_handle.option_chain(expiration_date)
        days_till_expiration = (pd.to_datetime(expiration_date) - today).days

        call_df = option_chain.calls.drop(columns=drop_columns, errors='ignore').copy()
        put_df = option_chain.puts.drop(columns=drop_columns, errors='ignore').copy()

        for df, option_type in ((call_df, 'Call'), (put_df, 'Put')):
            df['Days Till Expiration'] = days_till_expiration
            df['Expiration Date'] = expiration_date
            df['bid-ask spread'] = df['ask'] - df['bid']
            df['Expiration day'] = pd.to_datetime(expiration_date).day
            df['Expiration day name'] = pd.to_datetime(expiration_date).strftime('%A')
            df['Type'] = option_type
            df['mid'] = (df['bid'] + df['ask']) / 2

        call_contract_chain[expiration_date] = call_df
        put_contract_chain[expiration_date] = put_df

    expirations = sorted(call_contract_chain.keys())
    call_contract_chain_concat = pd.concat(call_contract_chain.values(), ignore_index=True)
    put_contract_chain_concat = pd.concat(put_contract_chain.values(), ignore_index=True)
    all_contracts_concat = pd.concat([call_contract_chain_concat, put_contract_chain_concat], ignore_index=True)


# %% [extracted notebook cell 9]
# Block 9: Open Interest Overview

# Associate each upcoming earnings event with the listed expiration closest to it.
# The matched DTE is highlighted consistently on expiration-based plots.
def _upcoming_earnings_dates(ticker):
    today_normalized = pd.Timestamp.today().normalize()
    dates = pd.DatetimeIndex([])
    try:
        earnings = ticker.get_earnings_dates(limit=12)
        if earnings is not None and not earnings.empty:
            dates = pd.DatetimeIndex(pd.to_datetime(earnings.index, errors='coerce', utc=True)).tz_convert(None)
    except Exception:
        pass

    dates = pd.DatetimeIndex(dates.dropna()).normalize().unique().sort_values()
    future_dates = dates[dates >= today_normalized]
    if len(future_dates) == 0:
        try:
            calendar = ticker.get_calendar() or {}
            calendar_dates = calendar.get('Earnings Date', [])
            if not isinstance(calendar_dates, (list, tuple, pd.Series, pd.Index, np.ndarray)):
                calendar_dates = [calendar_dates]
            calendar_dates = pd.DatetimeIndex(
                pd.to_datetime(calendar_dates, errors='coerce', utc=True)
            ).tz_convert(None)
            calendar_dates = calendar_dates.dropna().normalize().unique().sort_values()
            future_dates = calendar_dates[calendar_dates >= today_normalized]
        except Exception as earnings_error:
            print(f'Upcoming earnings dates unavailable for {ticker_str}: {earnings_error}')
    return future_dates

upcoming_earnings_dates = _upcoming_earnings_dates(ticker_handle)
listed_expiration_dates = pd.DatetimeIndex(pd.to_datetime(expirations, errors='coerce')).dropna().normalize().sort_values()
earnings_expiration_map = {}
earnings_expiration_details = []
for earnings_date in upcoming_earnings_dates:
    if len(listed_expiration_dates):
        distance_days = np.abs((listed_expiration_dates - earnings_date).days)
        closest_position = int(np.argmin(distance_days))
        associated_expiration = listed_expiration_dates[closest_position]
        matched_dte = int((associated_expiration - pd.Timestamp.today().normalize()).days)
        match_distance = int(distance_days[closest_position])
        earnings_expiration_map.setdefault(associated_expiration, []).append(earnings_date)
        earnings_expiration_details.append({
            'Earnings Date': earnings_date,
            'Nearest Expiration': associated_expiration,
            'Matched DTE': matched_dte,
            'Calendar-Day Distance': match_distance,
        })
earnings_expiration_details = pd.DataFrame(earnings_expiration_details)

def _earnings_expiration_label(expiration):
    expiration_date = pd.Timestamp(expiration).normalize()
    return '<br><span style="color:#FDA4AF">◆ Earnings-nearest</span>' if expiration_date in earnings_expiration_map else ''

def mark_upcoming_earnings_contracts(figure, expiration_to_x=None):
    """Add an earnings summary and optional matched-DTE guide lines."""
    if not earnings_expiration_map:
        figure.add_annotation(
            x=1, y=1.04, xref='paper', yref='paper',
            xanchor='right', yanchor='bottom', showarrow=False, align='left',
            text=(
                f'<b>◇ Earnings schedule unavailable</b><br>'
                f'yfinance returned no future corporate earnings date for {ticker_str}. '
                'ETFs such as XLC do not have their own earnings announcement.'
            ),
            bgcolor='rgba(18, 20, 30, 0.96)', bordercolor='#64748B',
            borderwidth=1, borderpad=7, font=dict(color='#CBD5E1', size=10),
        )
        current_top_margin = figure.layout.margin.t or 80
        figure.update_layout(margin=dict(t=max(int(current_top_margin), 150)))
        return figure
    event_lines = []
    for expiration_date, earnings_dates in earnings_expiration_map.items():
        event_text = ', '.join(date.strftime('%b %d, %Y') for date in earnings_dates)
        matched_dte = int((expiration_date - pd.Timestamp.today().normalize()).days)
        distances = [abs(int((expiration_date - date).days)) for date in earnings_dates]
        distance_text = ', '.join(f'{distance}d away' for distance in distances)
        event_lines.append(
            f'<b>◆ Earnings {event_text}</b><br>'
            f'Nearest expiry {expiration_date:%b %d} · {matched_dte} DTE · {distance_text}'
        )
        if expiration_to_x is not None and expiration_date in expiration_to_x:
            marker_location = expiration_to_x[expiration_date]
            if isinstance(marker_location, dict):
                x0 = marker_location['band_start']
                x1 = marker_location['band_end']
                marker_x = marker_location['marker']
                category_shift = marker_location.get('category_shift', {})
            elif isinstance(marker_location, tuple):
                x0, x1 = marker_location
                marker_x = (x0 + x1) / 2
                category_shift = {}
            else:
                x0 = x1 = marker_x = marker_location
                category_shift = {'x0shift': -0.48, 'x1shift': 0.48}
            figure.add_shape(
                type='rect', x0=x0, x1=x1, y0=0, y1=1,
                xref='x', yref='paper', layer='below',
                fillcolor='rgba(244, 63, 94, 0.08)',
                line=dict(color='rgba(251, 113, 133, 0.38)', width=1, dash='dot'),
                **category_shift,
            )
            figure.add_shape(
                type='line', x0=marker_x, x1=marker_x,
                y0=0, y1=1, xref='x', yref='paper', layer='above',
                line=dict(color='rgba(251, 113, 133, 0.68)', width=2, dash='solid'),
            )
            figure.add_annotation(
                x=marker_x, y=0.98, xref='x', yref='paper',
                text=f'<b>Earnings warning</b><br>Pre-earnings window ≤ {matched_dte} DTE',
                showarrow=False, xanchor='right', yanchor='top', align='right',
                bgcolor='rgba(18, 20, 30, 0.86)',
                bordercolor='rgba(251, 113, 133, 0.50)', borderwidth=1, borderpad=4,
                font=dict(color='#FDA4AF', size=10),
            )
    figure.add_annotation(
        x=1, y=1.04, xref='paper', yref='paper', xanchor='right', yanchor='bottom',
        text='<br><br>'.join(event_lines), showarrow=False, align='left',
        bgcolor='rgba(18, 20, 30, 0.94)', bordercolor='rgba(251, 113, 133, 0.62)', borderwidth=1, borderpad=7,
        font=dict(color='#FFE4E6', size=10),
    )
    current_top_margin = figure.layout.margin.t or 80
    figure.update_layout(margin=dict(t=max(int(current_top_margin), 170)))
    return figure

# Compute totals
total_oi_calls = [call_contract_chain[exp]['openInterest'].sum() for exp in expirations]
total_oi_puts = [put_contract_chain[exp]['openInterest'].sum() for exp in expirations]
dte_list = [call_contract_chain[exp]['Days Till Expiration'].iloc[0] for exp in expirations]
x_tick_labels = [f'{exp}<br>{dte} DTE{_earnings_expiration_label(exp)}' for exp, dte in zip(expirations, dte_list)]
expiration_dates = pd.DatetimeIndex(pd.to_datetime(expirations)).normalize()
expiration_years = expiration_dates.year.tolist()

# Aggregate total OI (calls + puts)
total_oi_all = [c + p for c, p in zip(total_oi_calls, total_oi_puts)]
put_call_ratio = [
    (put_oi / call_oi) if call_oi else np.nan
    for call_oi, put_oi in zip(total_oi_calls, total_oi_puts)
]

fig = plot_open_interest_overview_view(
    expirations=expirations,
    total_oi_calls=total_oi_calls,
    total_oi_puts=total_oi_puts,
    total_oi_all=total_oi_all,
    dte_list=dte_list,
    put_call_ratio=put_call_ratio,
)
fig.update_xaxes(
    tickmode='array',
    tickvals=[str(exp) for exp in expirations],
    ticktext=x_tick_labels,
)
fig.update_xaxes(title_text='Option Expiration (DTE)', row=3, col=1)

# Shade each expiration year and label it relative to the current year.
current_year = pd.Timestamp.today().year
year_band_colors = [
    ('rgba(59, 130, 246, 0.14)', '#60A5FA'),
    ('rgba(52, 211, 153, 0.14)', '#34D399'),
    ('rgba(251, 191, 36, 0.14)', '#FBBF24'),
    ('rgba(192, 132, 252, 0.14)', '#C084FC'),
]
expiration_count = len(expirations)
open_interest_panel_refs = [
    ('x domain', 'y domain'),
    ('x2 domain', 'y2 domain'),
    ('x3 domain', 'y3 domain'),
]

for band_number, year in enumerate(dict.fromkeys(expiration_years)):
    year_positions = [i for i, expiration_year in enumerate(expiration_years) if expiration_year == year]
    first_position, last_position = min(year_positions), max(year_positions)
    fill_color, accent_color = year_band_colors[band_number % len(year_band_colors)]
    band_start = first_position / expiration_count
    band_end = (last_position + 1) / expiration_count

    for xref, yref in open_interest_panel_refs:
        fig.add_shape(
            type='rect',
            xref=xref,
            yref=yref,
            x0=band_start,
            x1=band_end,
            y0=0,
            y1=1,
            fillcolor=fill_color,
            line_width=0,
            layer='below',
        )

    if first_position > 0:
        boundary_x = first_position / expiration_count
        for xref, yref in open_interest_panel_refs:
            fig.add_shape(
                type='line',
                xref=xref,
                yref=yref,
                x0=boundary_x,
                x1=boundary_x,
                y0=0,
                y1=1,
                line=dict(color=accent_color, width=2, dash='dot'),
            )

    if year == current_year:
        year_context = 'Current year'
    elif year == current_year + 1:
        year_context = 'Next year'
    else:
        year_context = f'{year - current_year:+d} years'

    fig.add_annotation(
        x=(band_start + band_end) / 2,
        y=1,
        xref='x3 domain',
        yref='y3 domain',
        yshift=-6,
        text=f'<b>{year}</b><br><span style="font-size:10px">{year_context}</span>',
        showarrow=False,
        bordercolor=accent_color,
        borderwidth=1,
        bgcolor='rgba(17, 24, 39, 0.90)',
        font=dict(color=accent_color),
    )

# Mark standard quarterly expirations (third Friday of Mar/Jun/Sep/Dec).
expiration_positions = {date: position for position, date in enumerate(expiration_dates)}
quarterly_expirations = []
quarterly_year_months = sorted(
    {(date.year, date.month) for date in expiration_dates if date.month in (3, 6, 9, 12)}
)

for year, month in quarterly_year_months:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    first_friday = month_start + pd.Timedelta(days=(4 - month_start.weekday()) % 7)
    third_friday = first_friday + pd.Timedelta(weeks=2)
    quarterly_date = third_friday

    # If Friday is a market holiday, use the preceding Thursday when available.
    if quarterly_date not in expiration_positions:
        quarterly_date = third_friday - pd.Timedelta(days=1)

    if quarterly_date in expiration_positions:
        quarterly_expirations.append((expiration_positions[quarterly_date], quarterly_date))

for position, quarterly_date in quarterly_expirations:
    quarterly_x = (position + 0.5) / expiration_count
    for xref, yref in open_interest_panel_refs:
        fig.add_shape(
            type='line',
            xref=xref,
            yref=yref,
            x0=quarterly_x,
            x1=quarterly_x,
            y0=0,
            y1=1,
            line=dict(color='#F87171', width=3, dash='dash'),
        )
    fig.add_annotation(
        x=quarterly_x,
        y=1,
        xref='x domain',
        yref='y domain',
        xshift=7,
        yshift=-5,
        text=f'<b>Q{quarterly_date.quarter} {quarterly_date.year}</b>',
        textangle=-90,
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='#FCA5A5', size=10),
    )
open_interest_expiration_to_x = {
    expiration: {
        'band_start': str(expirations[0]),
        'band_end': str(expirations[position]),
        'marker': str(expirations[position]),
        'category_shift': {},
    }
    for position, expiration in enumerate(expiration_dates)
}
mark_upcoming_earnings_contracts(fig, open_interest_expiration_to_x)
apply_notebook_plot_theme(fig)
open_interest_overview_fig = fig


# %% [extracted notebook cell 10]
# Block 10: Open Interest and Put-Call Skew by Individual DTE

if not expirations:
    raise ValueError('No option expirations are available to plot.')

implied_move_spot = pd.to_numeric(underlying_price, errors='coerce')
if not np.isfinite(implied_move_spot) or implied_move_spot <= 0:
    implied_move_spot = float(spot_price)
else:
    implied_move_spot = float(implied_move_spot)

# The DTE selector shows one expiration at a time. Calls and puts share the
# upper panel; put IV minus call IV is plotted directly below.
open_interest_implied_move_fig = plot_open_interest_implied_move_ranges_view(
    call_contract_chain=call_contract_chain,
    put_contract_chain=put_contract_chain,
    expirations=expirations,
    spot_price=implied_move_spot,
)
mark_upcoming_earnings_contracts(open_interest_implied_move_fig)
apply_notebook_plot_theme(open_interest_implied_move_fig)
# Rendered in the tabbed Blocks 9-15 dashboard.

# Separate term structure: one ATM implied-move observation per expiration.
implied_move_by_expiration = build_atm_implied_move_term_structure(
    call_contract_chain,
    put_contract_chain,
    expirations,
    spot_price=implied_move_spot,
)
implied_move_term_structure_fig = plot_atm_implied_move_term_structure_view(
    implied_move_by_expiration,
    spot_price=implied_move_spot,
    ticker_label=ticker_str,
)
implied_move_expiration_to_x = {
    pd.Timestamp(expiration).normalize(): {
        'band_start': 0,
        'band_end': position,
        'marker': position,
    }
    for position, expiration in enumerate(
        pd.to_datetime(implied_move_by_expiration['Expiration Date'])
    )
}
mark_upcoming_earnings_contracts(
    implied_move_term_structure_fig,
    implied_move_expiration_to_x,
)
apply_notebook_plot_theme(implied_move_term_structure_fig)
# Rendered in the tabbed Blocks 9-15 dashboard.


# %% [extracted notebook cell 11]
# Block 11: ATM IV and Realized Volatility
#plot the ATM IV and Realized Volatility

# Define ATM IV fetcher
def get_atm_iv_for_expiration(expiration_date, contract_chain):
    contracts = contract_chain.get(expiration_date)
    if contracts is None or contracts.empty:
        return np.nan

    required_columns = {'strike', 'impliedVolatility'}
    if not required_columns.issubset(contracts.columns):
        return np.nan

    valid_contracts = contracts[['strike', 'impliedVolatility']].copy()
    valid_contracts['strike'] = pd.to_numeric(valid_contracts['strike'], errors='coerce')
    valid_contracts['impliedVolatility'] = pd.to_numeric(valid_contracts['impliedVolatility'], errors='coerce')
    valid_contracts = valid_contracts.dropna(subset=['strike', 'impliedVolatility'])
    if valid_contracts.empty:
        return np.nan

    idx = (valid_contracts['strike'] - spot_price).abs().idxmin()
    return float(valid_contracts.loc[idx, 'impliedVolatility'])

# Calculate realized vol for each expiration
def get_realized_vol_for_expiration(expiration_date):
    days = (pd.to_datetime(expiration_date) - pd.Timestamp.today().normalize()).days
    if 1 < days < len(log_returns):
        window_returns = log_returns.iloc[-days:]
        realized_vol = window_returns.std() * np.sqrt(252)
        return realized_vol
    return np.nan

atm_df = pd.DataFrame({'Expiration Date': expirations})
atm_df['ATM IV Call'] = atm_df['Expiration Date'].apply(
    lambda exp: get_atm_iv_for_expiration(exp, call_contract_chain)
 )
atm_df['ATM IV Put'] = atm_df['Expiration Date'].apply(
    lambda exp: get_atm_iv_for_expiration(exp, put_contract_chain)
 )
atm_df['Days Till Expiration'] = atm_df['Expiration Date'].apply(
    lambda d: (pd.to_datetime(d) - pd.Timestamp.today().normalize()).days
 )
atm_df['Realized Vol'] = atm_df['Expiration Date'].apply(get_realized_vol_for_expiration)
atm_df = atm_df.sort_values('Days Till Expiration').reset_index(drop=True)

# Compute spreads
atm_df['IV-RV Call'] = atm_df['ATM IV Call'] - atm_df['Realized Vol']
atm_df['IV-RV Put'] = atm_df['ATM IV Put'] - atm_df['Realized Vol']

# Calculate Put-Call IV Skew
atm_df['IV Skew'] = atm_df['ATM IV Put'] - atm_df['ATM IV Call']

fig = plot_atm_iv_realized_view(atm_df, ticker_label=ticker_str)
atm_dte_values = atm_df['Days Till Expiration'].tolist()
atm_x_positions = list(range(len(atm_df)))
atm_expiration_dates = pd.DatetimeIndex(pd.to_datetime(atm_df['Expiration Date'])).normalize()
atm_expiration_years = atm_expiration_dates.year.tolist()
atm_x_tick_labels = [
    f'{pd.to_datetime(expiration):%Y-%m-%d}<br>{int(dte)} DTE{_earnings_expiration_label(expiration)}'
    for expiration, dte in zip(atm_df['Expiration Date'], atm_dte_values)
]
atm_hover_data = list(zip(atm_df['Expiration Date'].astype(str), atm_dte_values))
for trace in fig.data:
    trace.update(
        x=atm_x_positions,
        customdata=atm_hover_data,
        hovertemplate=(
            'Expiration: %{customdata[0]}<br>'
            'DTE: %{customdata[1]} days<br>'
            'Value: %{y:.2%}<extra>%{fullData.name}</extra>'
        ),
    )
fig.update_xaxes(
    tickmode='array',
    tickvals=atm_x_positions,
    ticktext=atm_x_tick_labels,
    range=[-0.5, len(atm_x_positions) - 0.5],
    tickangle=-45,
    automargin=True,
)
fig.update_xaxes(title_text='Expiration Date (DTE)', row=3, col=1)

# Add the same year context used in Block 11 across all three panels.
atm_current_year = pd.Timestamp.today().year
atm_year_band_colors = [
    ('rgba(59, 130, 246, 0.14)', '#60A5FA'),
    ('rgba(52, 211, 153, 0.14)', '#34D399'),
    ('rgba(251, 191, 36, 0.14)', '#FBBF24'),
    ('rgba(192, 132, 252, 0.14)', '#C084FC'),
]

for band_number, year in enumerate(dict.fromkeys(atm_expiration_years)):
    year_positions = [
        i for i, expiration_year in enumerate(atm_expiration_years) if expiration_year == year
    ]
    first_position, last_position = min(year_positions), max(year_positions)
    fill_color, accent_color = atm_year_band_colors[band_number % len(atm_year_band_colors)]

    band_start = first_position - 0.5
    band_end = last_position + 0.5

    for row in (1, 2, 3):
        fig.add_vrect(
            x0=band_start,
            x1=band_end,
            fillcolor=fill_color,
            line_width=0,
            layer='below',
            row=row,
            col=1,
        )

    if first_position > 0:
        fig.add_vline(
            x=band_start,
            line_color=accent_color,
            line_width=2,
            line_dash='dot',
            row='all',
            col=1,
        )

    if year == atm_current_year:
        year_context = 'Current year'
    elif year == atm_current_year + 1:
        year_context = 'Next year'
    else:
        year_context = f'{year - atm_current_year:+d} years'

    fig.add_annotation(
        x=(first_position + last_position) / 2,
        y=1,
        xref='x3',
        yref='y3 domain',
        yshift=-6,
        text=f'<b>{year}</b><br><span style="font-size:10px">{year_context}</span>',
        showarrow=False,
        bordercolor=accent_color,
        borderwidth=1,
        bgcolor='rgba(17, 24, 39, 0.90)',
        font=dict(color=accent_color),
    )

# Mark standard quarterly expirations as in Block 11.
atm_expiration_x = dict(zip(atm_expiration_dates, atm_x_positions))
atm_quarterly_expirations = []
atm_quarterly_year_months = sorted(
    {(date.year, date.month) for date in atm_expiration_dates if date.month in (3, 6, 9, 12)}
)

for year, month in atm_quarterly_year_months:
    month_start = pd.Timestamp(year=year, month=month, day=1)
    first_friday = month_start + pd.Timedelta(days=(4 - month_start.weekday()) % 7)
    third_friday = first_friday + pd.Timedelta(weeks=2)
    quarterly_date = third_friday

    if quarterly_date not in atm_expiration_x:
        quarterly_date = third_friday - pd.Timedelta(days=1)

    if quarterly_date in atm_expiration_x:
        atm_quarterly_expirations.append((atm_expiration_x[quarterly_date], quarterly_date))

for position, quarterly_date in atm_quarterly_expirations:
    fig.add_vline(
        x=position,
        line_color='#F87171',
        line_width=3,
        line_dash='dash',
        row='all',
        col=1,
    )
    fig.add_annotation(
        x=position,
        y=1,
        xref='x',
        yref='y domain',
        xshift=7,
        yshift=-5,
        text=f'<b>Q{quarterly_date.quarter} {quarterly_date.year}</b>',
        textangle=-90,
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(color='#FCA5A5', size=10),
    )
atm_earnings_band_lookup = {
    expiration: {
        'band_start': 0,
        'band_end': position,
        'marker': position,
    }
    for expiration, position in atm_expiration_x.items()
}
mark_upcoming_earnings_contracts(fig, atm_earnings_band_lookup)
apply_notebook_plot_theme(fig)
atm_iv_realized_fig = fig


# %% [extracted notebook cell 12]
# Block 12: IV Minus Realized by Strike
#Plot IV - Realized Volatility by Strike for Calls and Puts

from importlib import reload
from Quantapp.visualization.views.single_asset_profile.pricing.options_pricing import iv_realized_by_strike as block19_iv_realized_views

block19_iv_realized_views = reload(block19_iv_realized_views)
plot_iv_minus_realized_by_strike_view = block19_iv_realized_views.plot_iv_minus_realized_by_strike_view

# Historical price data (must be a Series indexed by date, most recent last)
price_series = pd.to_numeric(ticker['Close'], errors='coerce').copy()
price_series.index = pd.to_datetime(price_series.index, errors='coerce', utc=True).tz_convert(None).normalize()
price_series = price_series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
price_series = price_series[price_series.gt(0)].groupby(level=0).last()
log_returns = np.log(price_series / price_series.shift(1)).dropna()

# Spot price for reference line
spot_price = float(price_series.iloc[-1])
today = pd.Timestamp.today().normalize()

IV_RV_MONEYNESS_CLUSTER_WIDTH = 0.01
IV_RV_DTE_CLUSTER_BINS = [
    (2, 7, '2-7 DTE'),
    (8, 14, '8-14 DTE'),
    (15, 30, '15-30 DTE'),
    (31, 60, '31-60 DTE'),
    (61, 90, '61-90 DTE'),
    (91, 180, '91-180 DTE'),
    (181, 365, '181-365 DTE'),
    (366, 730, '366-730 DTE'),
    (731, np.inf, '731+ DTE'),
]

def _block19_dte_cluster(days_till_expiration):
    days = int(days_till_expiration)
    for cluster_order, (lower_bound, upper_bound, label) in enumerate(IV_RV_DTE_CLUSTER_BINS):
        if lower_bound <= days <= upper_bound:
            return cluster_order, label
    return len(IV_RV_DTE_CLUSTER_BINS), f'{days} DTE'

def build_iv_minus_realized_by_strike(contract_chain):
    raw_cluster_frames = []
    for exp in sorted(contract_chain.keys()):
        if contract_chain[exp] is None or contract_chain[exp].empty:
            continue
        df_sorted = contract_chain[exp].copy()
        if not {'strike', 'impliedVolatility'}.issubset(df_sorted.columns):
            continue
        df_sorted['strike'] = pd.to_numeric(df_sorted['strike'], errors='coerce')
        df_sorted['impliedVolatility'] = pd.to_numeric(df_sorted['impliedVolatility'], errors='coerce')
        df_sorted = df_sorted.dropna(subset=['strike', 'impliedVolatility'])
        df_sorted = df_sorted[df_sorted['strike'].gt(0) & df_sorted['impliedVolatility'].gt(0)].copy()
        if df_sorted.empty:
            continue
        exp_date = pd.to_datetime(exp)
        days_till_exp = int((exp_date.normalize() - today).days)

        if days_till_exp < 2 or days_till_exp > len(log_returns):
            continue

        realized_vol_n = log_returns.rolling(window=days_till_exp).std().iloc[-1] * np.sqrt(252)
        if not np.isfinite(realized_vol_n):
            continue
        df_sorted["iv_minus_realized"] = df_sorted["impliedVolatility"] - realized_vol_n
        df_sorted["days_till_expiration"] = days_till_exp
        df_sorted["realized_vol"] = float(realized_vol_n)
        df_sorted["moneyness_pct"] = (df_sorted["strike"] / spot_price) - 1.0
        df_sorted["moneyness_mid_pct"] = (
            df_sorted["moneyness_pct"] / IV_RV_MONEYNESS_CLUSTER_WIDTH
        ).round() * IV_RV_MONEYNESS_CLUSTER_WIDTH
        dte_cluster_order, dte_cluster_label = _block19_dte_cluster(days_till_exp)
        df_sorted["dte_cluster_order"] = dte_cluster_order
        df_sorted["dte_cluster"] = dte_cluster_label
        raw_cluster_frames.append(
            df_sorted[
                [
                    "strike",
                    "impliedVolatility",
                    "iv_minus_realized",
                    "realized_vol",
                    "days_till_expiration",
                    "moneyness_mid_pct",
                    "dte_cluster_order",
                    "dte_cluster",
                ]
            ]
        )

    if not raw_cluster_frames:
        return {}

    raw_cluster_frame = pd.concat(raw_cluster_frames, ignore_index=True)
    clustered = (
        raw_cluster_frame.groupby(
            ["dte_cluster_order", "dte_cluster", "moneyness_mid_pct"],
            as_index=False,
            sort=True,
        )
        .agg(
            strike=("strike", "mean"),
            iv_minus_realized=("iv_minus_realized", "mean"),
            avg_iv=("impliedVolatility", "mean"),
            avg_realized_vol=("realized_vol", "mean"),
            days_till_expiration=("days_till_expiration", "mean"),
            dte_min=("days_till_expiration", "min"),
            dte_max=("days_till_expiration", "max"),
            contract_count=("iv_minus_realized", "size"),
        )
        .sort_values(["dte_cluster_order", "strike"])
        .reset_index(drop=True)
    )
    clustered["dte_contract_count"] = clustered.groupby("dte_cluster")["contract_count"].transform("sum")
    clustered["expiration_label"] = clustered.apply(
        lambda row: f"{row['dte_cluster']} avg ({int(row['dte_contract_count']):,} contracts)",
        axis=1,
    )

    clustered_by_dte = {}
    for _, cluster_frame in clustered.groupby(["dte_cluster_order", "dte_cluster"], sort=True):
        cluster_label = cluster_frame["expiration_label"].iloc[0]
        clustered_by_dte[cluster_label] = cluster_frame.sort_values("strike").reset_index(drop=True)
    return clustered_by_dte

call_iv_minus_realized_by_expiration = build_iv_minus_realized_by_strike(call_contract_chain)
put_iv_minus_realized_by_expiration = build_iv_minus_realized_by_strike(put_contract_chain)
block19_call_cluster_count = sum(len(frame) for frame in call_iv_minus_realized_by_expiration.values())
block19_put_cluster_count = sum(len(frame) for frame in put_iv_minus_realized_by_expiration.values())
block19_call_contract_count = sum(frame['contract_count'].sum() for frame in call_iv_minus_realized_by_expiration.values())
block19_put_contract_count = sum(frame['contract_count'].sum() for frame in put_iv_minus_realized_by_expiration.values())
print(
    'Clustered Block 19 IV-RV by DTE band and '
    f'{IV_RV_MONEYNESS_CLUSTER_WIDTH:.1%} moneyness bucket | '
    f'calls: {len(call_iv_minus_realized_by_expiration)} DTE bands, '
    f'{block19_call_cluster_count:,} strike buckets from {int(block19_call_contract_count):,} contracts | '
    f'puts: {len(put_iv_minus_realized_by_expiration)} DTE bands, '
    f'{block19_put_cluster_count:,} strike buckets from {int(block19_put_contract_count):,} contracts'
)

fig = plot_iv_minus_realized_by_strike_view(
    call_iv_minus_realized_by_expiration=call_iv_minus_realized_by_expiration,
    put_iv_minus_realized_by_expiration=put_iv_minus_realized_by_expiration,
    spot_price=spot_price,
)
mark_upcoming_earnings_contracts(fig)
apply_notebook_plot_theme(fig)
iv_minus_realized_by_strike_fig = fig


# %% [extracted notebook cell 13]
# Block 13: Put-Call IV Skew by DTE
# Plot put IV minus call IV by moneyness for each available expiration.

PUT_CALL_SKEW_MONEYNESS_LIMIT = 0.35
PUT_CALL_SKEW_MIN_SHARED_STRIKES = 3


def _block19b_resolve_dte(expiration, *frames):
    for frame in frames:
        if frame is None or frame.empty or 'Days Till Expiration' not in frame.columns:
            continue
        dte_values = pd.to_numeric(frame['Days Till Expiration'], errors='coerce').dropna()
        if not dte_values.empty:
            return int(round(float(dte_values.median())))
    return int((pd.to_datetime(expiration).normalize() - pd.Timestamp.today().normalize()).days)


def _block19b_prepare_iv_side(frame, side_name):
    required_columns = {'strike', 'impliedVolatility'}
    if frame is None or frame.empty or not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=['strike', f'{side_name}_iv'])

    prepared = frame[['strike', 'impliedVolatility']].copy()
    prepared['strike'] = pd.to_numeric(prepared['strike'], errors='coerce')
    prepared[f'{side_name}_iv'] = pd.to_numeric(prepared['impliedVolatility'], errors='coerce')
    prepared = prepared.drop(columns=['impliedVolatility']).dropna(subset=['strike', f'{side_name}_iv'])
    prepared = prepared[prepared['strike'].gt(0) & prepared[f'{side_name}_iv'].gt(0)]
    return prepared.groupby('strike', as_index=False)[f'{side_name}_iv'].mean()


def build_put_call_skew_by_dte(call_chain, put_chain, *, spot_price, moneyness_limit=None):
    spot = float(spot_price)
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError('A positive spot_price is required to compute put-call skew moneyness.')

    skew_by_dte = {}
    shared_expirations = sorted(set(call_chain.keys()) & set(put_chain.keys()), key=pd.to_datetime)

    for expiration in shared_expirations:
        call_frame = call_chain.get(expiration)
        put_frame = put_chain.get(expiration)
        dte = _block19b_resolve_dte(expiration, call_frame, put_frame)
        if dte < 0:
            continue

        calls = _block19b_prepare_iv_side(call_frame, 'call')
        puts = _block19b_prepare_iv_side(put_frame, 'put')
        skew_frame = calls.merge(puts, on='strike', how='inner')
        if skew_frame.empty:
            continue

        skew_frame['moneyness_pct'] = (skew_frame['strike'] / spot) - 1.0
        if moneyness_limit is not None:
            skew_frame = skew_frame[skew_frame['moneyness_pct'].abs().le(float(moneyness_limit))]
        if len(skew_frame) < PUT_CALL_SKEW_MIN_SHARED_STRIKES:
            continue

        expiration_date = pd.to_datetime(expiration).normalize()
        skew_frame['put_call_skew'] = skew_frame['put_iv'] - skew_frame['call_iv']
        skew_frame['expiration_date'] = expiration_date
        skew_frame['days_till_expiration'] = dte
        skew_frame = skew_frame.sort_values('moneyness_pct').reset_index(drop=True)
        skew_by_dte[f'{dte} DTE - {expiration_date:%Y-%m-%d}'] = skew_frame

    return skew_by_dte


put_call_skew_by_dte = build_put_call_skew_by_dte(
    call_contract_chain,
    put_contract_chain,
    spot_price=spot_price,
    moneyness_limit=PUT_CALL_SKEW_MONEYNESS_LIMIT,
)

if not put_call_skew_by_dte:
    raise ValueError('No shared call/put strikes were available to plot put-call skew by DTE.')

put_call_skew_fig = go.Figure()
put_call_skew_labels = list(put_call_skew_by_dte.keys())
put_call_skew_overlays = {}

def _block19b_chain_for_expiration(chain, expiration_date):
    for expiration_key, frame in chain.items():
        if pd.to_datetime(expiration_key).normalize() == expiration_date:
            return expiration_key, frame
    return None, None

def _block19b_max_oi_moneyness(frame):
    if frame is None or frame.empty or not {'strike', 'openInterest'}.issubset(frame.columns):
        return None
    values = frame[['strike', 'openInterest']].apply(pd.to_numeric, errors='coerce').dropna()
    if values.empty:
        return None
    max_row = values.loc[values['openInterest'].idxmax()]
    return (float(max_row['strike']) / float(spot_price)) - 1.0

for dte_index, (label, skew_frame) in enumerate(put_call_skew_by_dte.items()):
    hover_frame = skew_frame[
        ['strike', 'call_iv', 'put_iv', 'put_call_skew', 'expiration_date', 'days_till_expiration']
    ].copy()
    hover_frame['expiration_date'] = hover_frame['expiration_date'].dt.strftime('%Y-%m-%d')
    expiration_date = pd.to_datetime(skew_frame['expiration_date'].iloc[0]).normalize()
    expiration_key, call_expiration_chain = _block19b_chain_for_expiration(call_contract_chain, expiration_date)
    _, put_expiration_chain = _block19b_chain_for_expiration(put_contract_chain, expiration_date)
    overlay_shapes = [
        dict(type='line', x0=0, x1=0, y0=0, y1=1, xref='x', yref='paper', line=dict(color='#F87171', dash='dash', width=2)),
        dict(type='line', x0=0, x1=1, y0=0, y1=0, xref='paper', yref='y', line=dict(color='#CBD5E1', dash='dash')),
    ]
    overlay_annotations = []
    if expiration_key is not None and call_expiration_chain is not None and put_expiration_chain is not None:
        implied_move_row = build_atm_implied_move_term_structure(
            {expiration_key: call_expiration_chain},
            {expiration_key: put_expiration_chain},
            [expiration_key],
            spot_price=spot_price,
        )
        if not implied_move_row.empty:
            lower_moneyness = (float(implied_move_row.iloc[0]['Straddle Lower']) / float(spot_price)) - 1.0
            upper_moneyness = (float(implied_move_row.iloc[0]['Straddle Upper']) / float(spot_price)) - 1.0
            overlay_shapes.insert(0, dict(
                type='rect', x0=lower_moneyness, x1=upper_moneyness, y0=0, y1=1,
                xref='x', yref='paper', fillcolor='#F59E0B', opacity=0.18, layer='below',
                line=dict(color='#F59E0B', width=1, dash='dot'),
            ))
            overlay_annotations.extend([
                dict(x=lower_moneyness, y=1, xref='x', yref='paper', text='ATM lower', showarrow=False, xanchor='right', yanchor='bottom', font=dict(color='#FBBF24')),
                dict(x=upper_moneyness, y=1, xref='x', yref='paper', text='ATM upper', showarrow=False, xanchor='left', yanchor='bottom', font=dict(color='#FBBF24')),
            ])
        for marker_name, marker_color, marker_x in (
            ('Max Call OI', '#60A5FA', _block19b_max_oi_moneyness(call_expiration_chain)),
            ('Max Put OI', '#4ADE80', _block19b_max_oi_moneyness(put_expiration_chain)),
        ):
            if marker_x is not None:
                overlay_shapes.append(dict(type='line', x0=marker_x, x1=marker_x, y0=0, y1=1, xref='x', yref='paper', line=dict(color=marker_color, dash='dot', width=2)))
                overlay_annotations.append(dict(x=marker_x, y=0, xref='x', yref='paper', text=marker_name, showarrow=False, yanchor='bottom', textangle=-90, font=dict(color=marker_color)))
    put_call_skew_overlays[label] = (overlay_shapes, overlay_annotations)
    trace_visible = dte_index == 0
    put_call_skew_fig.add_trace(
        go.Scatter(
            x=skew_frame['moneyness_pct'],
            y=skew_frame['call_iv'],
            customdata=hover_frame.to_numpy(),
            mode='lines+markers',
            name='Call IV',
            line=dict(color='#60A5FA'),
            visible=trace_visible,
            hovertemplate=(
                'Moneyness: %{x:+.1%}<br>Call IV: %{y:.2%}<br>'
                'Strike: %{customdata[0]:,.2f}<br>Expiration: %{customdata[4]}<br>'
                'DTE: %{customdata[5]:.0f} days<extra>Call IV</extra>'
            ),
        )
    )
    put_call_skew_fig.add_trace(
        go.Scatter(
            x=skew_frame['moneyness_pct'],
            y=skew_frame['put_iv'],
            customdata=hover_frame.to_numpy(),
            mode='lines+markers',
            name='Put IV',
            line=dict(color='#4ADE80'),
            visible=trace_visible,
            hovertemplate=(
                'Moneyness: %{x:+.1%}<br>Put IV: %{y:.2%}<br>'
                'Strike: %{customdata[0]:,.2f}<br>Expiration: %{customdata[4]}<br>'
                'DTE: %{customdata[5]:.0f} days<extra>Put IV</extra>'
            ),
        )
    )
    put_call_skew_fig.add_trace(
        go.Scatter(
            x=skew_frame['moneyness_pct'],
            y=skew_frame['put_call_skew'],
            customdata=hover_frame.to_numpy(),
            mode='lines+markers',
            name='Put - Call IV',
            line=dict(color='#C084FC', dash='dash'),
            visible=trace_visible,
            opacity=0.85,
            hovertemplate=(
                'Moneyness: %{x:+.1%}<br>'
                'Put-call skew: %{y:+.2%}<br>'
                'Strike: %{customdata[0]:,.2f}<br>'
                'Call IV: %{customdata[1]:.2%}<br>'
                'Put IV: %{customdata[2]:.2%}<br>'
                'Expiration: %{customdata[4]}<br>'
                'DTE: %{customdata[5]:.0f} days'
                '<extra>%{fullData.name}</extra>'
            ),
        )
    )

put_call_skew_buttons = []
for trace_index, label in enumerate(put_call_skew_labels):
    visibility = [False] * (len(put_call_skew_labels) * 3)
    visibility[trace_index * 3:trace_index * 3 + 3] = [True, True, True]
    overlay_shapes, overlay_annotations = put_call_skew_overlays[label]
    put_call_skew_buttons.append(
        dict(
            label=label,
            method='update',
            args=[
                {'visible': visibility},
                {'title': f'{ticker_str} Put-Call IV Skew by Moneyness - {label}', 'shapes': overlay_shapes, 'annotations': overlay_annotations},
            ],
        )
    )

put_call_skew_fig.add_hline(y=0, line_color='#CBD5E1', line_dash='dash')
put_call_skew_fig.add_vline(x=0, line_color='#F87171', line_dash='dash')
put_call_skew_fig.update_layout(
    title=f'{ticker_str} Call IV, Put IV, and Put-Call Skew - {put_call_skew_labels[0]}',
    height=850,
    xaxis_title='Moneyness vs Spot',
    yaxis_title='Implied Volatility / Put IV - Call IV',
    hovermode='closest',
    legend_title_text='Series',
    shapes=put_call_skew_overlays[put_call_skew_labels[0]][0],
    annotations=put_call_skew_overlays[put_call_skew_labels[0]][1],
    updatemenus=[
        dict(
            active=0,
            buttons=put_call_skew_buttons,
            direction='down',
            x=0,
            y=1.12,
            xanchor='left',
            yanchor='top',
        )
    ],
)
put_call_skew_fig.update_xaxes(tickformat='+.0%')
put_call_skew_fig.update_yaxes(tickformat='+.1%')
mark_upcoming_earnings_contracts(put_call_skew_fig)
apply_notebook_plot_theme(put_call_skew_fig)
# Rendered in the tabbed Blocks 9-15 dashboard.


# %% [extracted notebook cell 14]
# Block 14: Median IV Minus Realized

today = pd.to_datetime("today")

# --- Helper function to calculate median IV - Realized Vol ---
def calc_median_iv_minus_realized(contract_chain, filter_func=None):
    median_dict = {}
    for exp in contract_chain.keys():
        df = contract_chain[exp].copy()
        if filter_func:
            df = df[filter_func(df)]
        if df.empty:
            continue
        exp_date = pd.to_datetime(exp)
        days_till_exp = (exp_date - today).days
        if days_till_exp < 2 or days_till_exp > len(log_returns):
            continue
        realized_vol_n = log_returns.rolling(window=days_till_exp).std().iloc[-1] * np.sqrt(252)
        median_val = (df['impliedVolatility'] - realized_vol_n).median()
        median_dict[exp_date] = median_val
    return median_dict

# --- Top subplot: All strikes ---
median_calls_all = calc_median_iv_minus_realized(call_contract_chain)
median_puts_all = calc_median_iv_minus_realized(put_contract_chain)

# --- Bottom subplot: OTM strikes ---
median_calls_otm = calc_median_iv_minus_realized(
    call_contract_chain,
    filter_func=lambda df: df['strike'] > spot_price
)
median_puts_otm = calc_median_iv_minus_realized(
    put_contract_chain,
    filter_func=lambda df: df['strike'] < spot_price
)

df_calls_all = pd.DataFrame({'Expiration': list(median_calls_all.keys()), 'Median': list(median_calls_all.values()), 'Type': 'Call'})
df_puts_all = pd.DataFrame({'Expiration': list(median_puts_all.keys()), 'Median': list(median_puts_all.values()), 'Type': 'Put'})
df_all = pd.concat([df_calls_all, df_puts_all])
df_all['DTE'] = (df_all['Expiration'] - today).dt.days
df_all = df_all.sort_values(['Expiration', 'Type']).reset_index(drop=True)

df_calls_otm = pd.DataFrame({'Expiration': list(median_calls_otm.keys()), 'Median': list(median_calls_otm.values()), 'Type': 'Call_OTM'})
df_puts_otm = pd.DataFrame({'Expiration': list(median_puts_otm.keys()), 'Median': list(median_puts_otm.values()), 'Type': 'Put_OTM'})
df_otm = pd.concat([df_calls_otm, df_puts_otm])
df_otm['DTE'] = (df_otm['Expiration'] - today).dt.days
df_otm = df_otm.sort_values(['Expiration', 'Type']).reset_index(drop=True)

fig = plot_median_iv_minus_realized_view(df_all, df_otm, today=today)

# Space expirations evenly while preserving their true dates and DTE values.
median_expiration_dates = pd.DatetimeIndex(
    pd.concat([df_all['Expiration'], df_otm['Expiration']], ignore_index=True).dropna().unique()
).normalize().sort_values()
median_x_positions = list(range(len(median_expiration_dates)))
median_expiration_to_x = dict(zip(median_expiration_dates, median_x_positions))
median_dte_values = [(expiration - today).days for expiration in median_expiration_dates]
median_x_tick_labels = [
    f'{expiration:%Y-%m-%d}<br>{int(dte)} DTE{_earnings_expiration_label(expiration)}'
    for expiration, dte in zip(median_expiration_dates, median_dte_values)
]

for trace in fig.data:
    trace_expirations = pd.DatetimeIndex(pd.to_datetime(list(trace.x))).normalize()
    trace_positions = [median_expiration_to_x[expiration] for expiration in trace_expirations]
    trace_hover_data = [
        (expiration.strftime('%Y-%m-%d'), int((expiration - today).days))
        for expiration in trace_expirations
    ]
    trace.update(
        x=trace_positions,
        customdata=trace_hover_data,
        hovertemplate=(
            'Expiration: %{customdata[0]}<br>'
            'DTE: %{customdata[1]} days<br>'
            'Median IV - Realized: %{y:.2%}<extra></extra>'
        ),
    )

if median_x_positions:
    fig.update_xaxes(
        type='linear',
        tickmode='array',
        tickvals=median_x_positions,
        ticktext=median_x_tick_labels,
        range=[-0.5, len(median_x_positions) - 0.5],
        tickangle=-45,
        automargin=True,
    )
    fig.update_xaxes(title_text='Expiration Date (DTE)', row=2, col=1)

    # Add the same year shading and expiration markers used in Block 13.
    median_current_year = pd.Timestamp.today().year
    median_expiration_years = median_expiration_dates.year.tolist()
    median_year_band_colors = [
        ('rgba(59, 130, 246, 0.14)', '#60A5FA'),
        ('rgba(52, 211, 153, 0.14)', '#34D399'),
        ('rgba(251, 191, 36, 0.14)', '#FBBF24'),
        ('rgba(192, 132, 252, 0.14)', '#C084FC'),
    ]

    for band_number, year in enumerate(dict.fromkeys(median_expiration_years)):
        year_positions = [
            i for i, expiration_year in enumerate(median_expiration_years) if expiration_year == year
        ]
        first_position, last_position = min(year_positions), max(year_positions)
        fill_color, accent_color = median_year_band_colors[
            band_number % len(median_year_band_colors)
        ]

        for row in (1, 2):
            fig.add_vrect(
                x0=first_position - 0.5,
                x1=last_position + 0.5,
                fillcolor=fill_color,
                line_width=0,
                layer='below',
                row=row,
                col=1,
            )

        if first_position > 0:
            fig.add_vline(
                x=first_position - 0.5,
                line_color=accent_color,
                line_width=2,
                line_dash='dot',
                row='all',
                col=1,
            )

        if year == median_current_year:
            year_context = 'Current year'
        elif year == median_current_year + 1:
            year_context = 'Next year'
        else:
            year_context = f'{year - median_current_year:+d} years'

        fig.add_annotation(
            x=(first_position + last_position) / 2,
            y=1,
            xref='x2',
            yref='y2 domain',
            yshift=-6,
            text=f'<b>{year}</b><br><span style="font-size:10px">{year_context}</span>',
            showarrow=False,
            bordercolor=accent_color,
            borderwidth=1,
            bgcolor='rgba(17, 24, 39, 0.90)',
            font=dict(color=accent_color),
        )

    median_quarterly_expirations = []
    median_quarterly_year_months = sorted(
        {(date.year, date.month) for date in median_expiration_dates if date.month in (3, 6, 9, 12)}
    )

    for year, month in median_quarterly_year_months:
        month_start = pd.Timestamp(year=year, month=month, day=1)
        first_friday = month_start + pd.Timedelta(days=(4 - month_start.weekday()) % 7)
        third_friday = first_friday + pd.Timedelta(weeks=2)
        quarterly_date = third_friday

        if quarterly_date not in median_expiration_to_x:
            quarterly_date = third_friday - pd.Timedelta(days=1)

        if quarterly_date in median_expiration_to_x:
            median_quarterly_expirations.append(
                (median_expiration_to_x[quarterly_date], quarterly_date)
            )

    for position, quarterly_date in median_quarterly_expirations:
        fig.add_vline(
            x=position,
            line_color='#F87171',
            line_width=3,
            line_dash='dash',
            row='all',
            col=1,
        )
        fig.add_annotation(
            x=position,
            y=1,
            xref='x',
            yref='y domain',
            xshift=7,
            yshift=-5,
            text=f'<b>Q{quarterly_date.quarter} {quarterly_date.year}</b>',
            textangle=-90,
            showarrow=False,
            xanchor='left',
            yanchor='top',
            font=dict(color='#FCA5A5', size=10),
        )
median_earnings_band_lookup = {
    expiration: {
        'band_start': 0,
        'band_end': position,
        'marker': position,
    }
    for expiration, position in median_expiration_to_x.items()
}
mark_upcoming_earnings_contracts(fig, median_earnings_band_lookup)
apply_notebook_plot_theme(fig)
median_iv_minus_realized_fig = fig


# %% [extracted notebook cell 15]
# Block 15: SVI Surface
#plot SVI surface for calls and puts
from scipy.optimize import minimize
import scipy.interpolate as interp

print(f"Spot price for {ticker_str}: {spot_price}")

def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def svi_objective(params, k, total_var):
    a, b, rho, m, sigma = params
    model_var = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sum((model_var - total_var) ** 2)

def fit_svi(k, total_var):
    x0 = [0.1, 0.1, 0.0, 0.0, 0.1]
    bounds = [(-1, 1), (1e-5, 5), (-0.999, 0.999), (-5, 5), (1e-5, 5)]
    res = minimize(svi_objective, x0, args=(k, total_var), bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else None

def get_realized_vol_surface(price_history, dtes):
    hist = price_history[['Close']].copy()
    hist['returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
    hist.dropna(inplace=True)

    rv_data = []
    for dte in sorted(set(dtes)):
        dte = int(dte)
        sub_ret = hist['returns'].iloc[-dte:]
        if len(sub_ret) >= dte * 0.8:
            realized_vol = np.std(sub_ret) * np.sqrt(252)
            rv_data.append((dte, realized_vol))
    return pd.DataFrame(rv_data, columns=['Days Till Expiration', 'Realized Vol'])

call_df = call_contract_chain_concat.copy()
call_df = call_df[call_df['impliedVolatility'].notna() & (call_df['impliedVolatility'] > 0)].copy()
call_df['T'] = call_df['Days Till Expiration'] / 365.0

put_df = put_contract_chain_concat.copy()
put_df = put_df[put_df['impliedVolatility'].notna() & (put_df['impliedVolatility'] > 0)].copy()
put_df['T'] = put_df['Days Till Expiration'] / 365.0

def fit_svi_surface(df, spot_price):
    unique_Ts = np.sort(df['T'].unique())
    svi_params_per_T = {}

    for T in unique_Ts:
        slice_df = df[df['T'] == T]
        if len(slice_df) < 5:
            continue
        K = slice_df['strike'].values
        iv = slice_df['impliedVolatility'].values
        total_var = iv ** 2 * T
        k = np.log(K / spot_price)
        params = fit_svi(k, total_var)
        if params is not None:
            svi_params_per_T[T] = params

    Ts = np.array(sorted(svi_params_per_T.keys()))
    params_array = np.array([svi_params_per_T[T] for T in Ts])
    param_interpolators = [
        interp.interp1d(Ts, params_array[:, i], kind='linear', fill_value='extrapolate')
        for i in range(5)
    ]

    strike_grid = np.linspace(df['strike'].min(), df['strike'].max(), 50)
    T_grid = np.linspace(df['T'].min(), df['T'].max(), 30)
    iv_surface = np.full((len(T_grid), len(strike_grid)), np.nan)

    for i, T in enumerate(T_grid):
        a, b, rho, m, sigma = [f(T) for f in param_interpolators]
        k_vals = np.log(strike_grid / spot_price)
        total_var = svi_total_variance(k_vals, a, b, rho, m, sigma)
        iv_surface[i, :] = np.sqrt(total_var / T)

    return strike_grid, T_grid, iv_surface

call_strike_grid, call_T_grid, call_iv_svi_surface = fit_svi_surface(call_df, spot_price)
put_strike_grid, put_T_grid, put_iv_svi_surface = fit_svi_surface(put_df, spot_price)

all_dtes = np.unique(np.concatenate([
    call_df['Days Till Expiration'].unique(),
    put_df['Days Till Expiration'].unique()
]))

# Get RV surface from the shared top-loaded price history
rv_df = get_realized_vol_surface(ticker, all_dtes)

fig = plot_svi_surface_view(
    call_df=call_df,
    put_df=put_df,
    call_strike_grid=call_strike_grid,
    call_t_grid=call_T_grid,
    call_iv_svi_surface=call_iv_svi_surface,
    put_strike_grid=put_strike_grid,
    put_t_grid=put_T_grid,
    put_iv_svi_surface=put_iv_svi_surface,
    realized_vol_surface_df=rv_df,
    spot_price=spot_price,
    ticker_label=ticker_str,
)
mark_upcoming_earnings_contracts(fig)
apply_notebook_plot_theme(fig)
svi_surface_fig = fig


# %% [extracted notebook cell 16]
# Block 15A: Tabbed yfinance Options Dashboard

import socket
from dash import Dash, Input, Output, dcc, html

options_yfinance_tab_figures = {
    "open_interest": ("Block 9 — Open Interest", [open_interest_overview_fig]),
    "dte_implied_move": (
        "Block 10 — DTE & Implied Move",
        [open_interest_implied_move_fig, implied_move_term_structure_fig],
    ),
    "atm_iv_realized": ("Block 11 — ATM IV vs Realized", [atm_iv_realized_fig]),
    "iv_rv_strike": ("Block 12 — IV-RV by Strike", [iv_minus_realized_by_strike_fig]),
    "put_call_skew": ("Block 13 — Put-Call Skew", [put_call_skew_fig]),
    "median_iv_premium": ("Block 14 — Median IV Premium", [median_iv_minus_realized_fig]),
    "svi_surface": ("Block 15 — SVI Surface", [svi_surface_fig]),
}

def _options_yfinance_dashboard_port(start=8091, stop=8120):
    for candidate_port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate_socket:
            try:
                candidate_socket.bind(("127.0.0.1", candidate_port))
            except OSError:
                continue
            return candidate_port
    raise RuntimeError("No open localhost port is available for the options tab dashboard.")

options_yfinance_dash_app = Dash("options-yfinance-tabs")
options_yfinance_dash_app.layout = html.Div(
    [
        dcc.Tabs(
            id="options-yfinance-tabs",
            value="open_interest",
            children=[
                dcc.Tab(label=label, value=tab_key)
                for tab_key, (label, _) in options_yfinance_tab_figures.items()
            ],
            colors={"border": "#334155", "primary": "#60a5fa", "background": "#111827"},
        ),
        html.Div(id="options-yfinance-tab-content"),
    ],
    style={"backgroundColor": "#0b0f14", "padding": "12px", "color": "#f8fafc"},
)

@options_yfinance_dash_app.callback(
    Output("options-yfinance-tab-content", "children"),
    Input("options-yfinance-tabs", "value"),
)
def _render_options_yfinance_tab(active_tab):
    label, figures = options_yfinance_tab_figures.get(
        active_tab, options_yfinance_tab_figures["open_interest"]
    )
    return html.Div(
        [
            dcc.Loading(
                type="circle",
                color="#60a5fa",
                children=dcc.Graph(
                    id=f"options-yfinance-{active_tab}-{figure_index}",
                    figure=figure,
                    config={"responsive": True, "displaylogo": False},
                    style={"height": f"{int(figure.layout.height or 900)}px"},
                ),
            )
            for figure_index, figure in enumerate(figures)
        ]
    )

options_yfinance_dash_app.run(
    host="127.0.0.1",
    port=_options_yfinance_dashboard_port(),
    debug=False,
    use_reloader=False,
    jupyter_mode="inline",
    jupyter_height=1900,
)

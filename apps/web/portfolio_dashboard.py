# %% [portfolio block 1]
# Code Block 1: Notebook description
#Notebook description
# This notebook evaluates the performance of a portfolio of assets on a buy-and-hold basis.
# It focuses on how multiple assets interact within the portfolio, rather than assessing a specific mechanical trading strategy.
# buy and hold is a good benchmark for any trading strategy, so it is important to evaluate the performance of a portfolio independently of
# trading strategies we may want to implement on the individual assets.

# %% [portfolio block 2]
# Code Block 2: Load Libraries
# Load Libraries
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from IPython.display import display as ipython_display

# Plotly discovers Pillow lazily while serializing callback responses. Import it
# completely before Dash starts so concurrent requests cannot observe a partially
# initialized PIL.Image module.
try:
    from PIL import Image as _PILImage

    _PILImage.Image
except ImportError:
    _PILImage = None

# Dashboard mode collects outputs into tabs instead of rendering each analysis inline.
def display(*_args, **_kwargs):
    return None
from schwab.auth import easy_client
import os
import sys
from pathlib import Path
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


from Quantapp.data import yf as qa_yf
from Quantapp.data import get_schwab_portfolio_snapshot
from Quantapp.data.adapters import SCHWAB_OPTION_SYMBOL_PATTERN

from Quantapp.data import MacroDataClient
from Quantapp.secrets import load_project_env, require_secret

load_project_env()

qe = MacroDataClient()

# %% [portfolio block 3]
# Code Block 3: Define functions and classes
#Define functions & Classes
#takes a dict of portfolio and their total amounts and directional (short or long value), converts dict to weightings instead of absolute values
def create_weighted_portfolio(portfolio):
    total = sum(abs(amount) for amount in portfolio.values())
    return {ticker: (amount / total) * (1 if direction == 'long' else -1)
            for (ticker, amount), direction in zip(portfolio.items(), ['long' if amount >= 0 else 'short' for amount in portfolio.values()])
    }
def create_weight_dict(portfolio):
    total = sum(abs(amount) for amount in portfolio.values())
    return {ticker: amount / total for ticker, amount in portfolio.items()}

def create_equal_weighted_dict(tickers):
    n = len(tickers)

    if n == 0:
        return {}

    equal_weight = 1 / n
    return {ticker: equal_weight for ticker in tickers}

def normalize_yf_ticker(ticker):
    if not isinstance(ticker, str):
        return ticker

    cleaned_ticker = ticker.strip()
    lookup_key = ''.join(character for character in cleaned_ticker.upper() if character.isalnum())
    class_share_yf_map = {
        'BRKB': 'BRK-B',
        'BFB': 'BF-B',
    }
    return class_share_yf_map.get(lookup_key, cleaned_ticker.replace('/', '-'))

def build_yf_ticker_map(tickers):
    return {ticker: normalize_yf_ticker(ticker) for ticker in tickers}

def zscore_series(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)

    return (series - mean) / std

def z_score(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)

    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)

def sharpe_annualized(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or np.isnan(std):
        return 0.0

    return (mean / std) * np.sqrt(252)

# %% [portfolio block 4]
# Code Block 4: Define parameters
#Define parameters
time_frame_week = 7
time_frame_short = 21
time_frame_mid = 50
time_frame_long = 200
selected_time_frame = time_frame_long
CLIENT_ID = require_secret("SCHWAB_CLIENT_ID")
APP_SECRET = require_secret("SCHWAB_APP_SECRET")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")

_token_path = os.getenv("SCHWAB_TOKEN_PATH")
TOKEN_PATH = Path(_token_path).expanduser() if _token_path else PROJECT_ROOT / "schwab_token.json"
if not TOKEN_PATH.is_absolute():
    TOKEN_PATH = PROJECT_ROOT / TOKEN_PATH
TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

#callback
period = '20y'
interval = '1d'
benchmark_str = 'SPY'

# %% [portfolio block 5]
# Code Block 5: Login to Schwab client
from schwab.auth import easy_client

client = easy_client(
    api_key=CLIENT_ID,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=str(TOKEN_PATH),
)

print(f"Schwab client ready. Token cache: {TOKEN_PATH}")

# %% [portfolio block 6]
# Code Block 6: Retrieve account and market data
# Schwab account retrieval and position normalization live in Quantapp.data.
portfolio_snapshot = get_schwab_portfolio_snapshot(client)

account_information = portfolio_snapshot.account_information
acct_map = portfolio_snapshot.account_numbers
acct_hash = portfolio_snapshot.account_hash
acct = portfolio_snapshot.account
positions = portfolio_snapshot.raw_positions
positions_df = portfolio_snapshot.option_positions
option_sentiment = portfolio_snapshot.option_sentiment
net_direction = portfolio_snapshot.net_direction
organized_positions = portfolio_snapshot.organized_positions
invested_symbols = portfolio_snapshot.invested_symbols
net_invested_amounts = portfolio_snapshot.net_invested_amounts
total_margin = portfolio_snapshot.total_margin
option_pattern = SCHWAB_OPTION_SYMBOL_PATTERN

# Retrieve core market data for benchmark and portfolio.
benchmark_symbol = benchmark_str if 'benchmark_str' in globals() else 'SPY'
benchmark_data = qa_yf.Ticker(benchmark_symbol).history(period=period, interval=interval)
invested_symbol_map = build_yf_ticker_map(invested_symbols)

if invested_symbol_map:
    portfolio_data = qa_yf.download(
        tickers=list(invested_symbol_map.values()),
        period=period,
        interval=interval,
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    portfolio_closing_prices = portfolio_data['Close']
    if isinstance(portfolio_closing_prices, pd.Series):
        portfolio_closing_prices = portfolio_closing_prices.to_frame(name=invested_symbols[0])
    else:
        portfolio_closing_prices = portfolio_closing_prices.rename(
            columns={yf_ticker: ticker for ticker, yf_ticker in invested_symbol_map.items()}
        )
else:
    portfolio_closing_prices = pd.DataFrame(index=benchmark_data.index)

benchmark_close = benchmark_data['Close']
portfolio_closing_prices.index = portfolio_closing_prices.index.tz_localize(None)
benchmark_close.index = benchmark_close.index.tz_localize(None)

net_direction
raw_prices = portfolio_closing_prices.copy()

# %% [portfolio block 7]
# Code Block 7: Schwab total value and cash position
# =========================
# 7) Schwab total value and cash position
# =========================
import importlib.util
import json
import sys


def _schwab_current_market_value(account_payload):
    securities_account = account_payload.get('securitiesAccount', {}) if isinstance(account_payload, dict) else {}
    balances = securities_account.get('currentBalances', {}) if isinstance(securities_account, dict) else {}
    if not isinstance(balances, dict):
        return None

    for key in ('liquidationValue', 'accountValue'):
        value = balances.get(key)
        if value is not None:
            return float(value)

    cash = balances.get('cashBalance')
    long_market_value = balances.get('longMarketValue')
    short_market_value = balances.get('shortMarketValue')
    if cash is not None or long_market_value is not None or short_market_value is not None:
        return float(cash or 0.0) + float(long_market_value or 0.0) + float(short_market_value or 0.0)
    return None


schwab_history_root = PROJECT_ROOT / 'csv_files' / 'schwab_api_raw'
schwab_market_value_run_dir = schwab_history_root / '20260701_212240'
if not schwab_market_value_run_dir.exists():
    schwab_market_value_run_dir = max(
        (path for path in schwab_history_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )

schwab_market_value_module_path = PROJECT_ROOT / 'scripts' / 'plot_schwab_equity_curve.py'
schwab_market_value_spec = importlib.util.spec_from_file_location(
    'schwab_market_value_curve',
    schwab_market_value_module_path,
)
schwab_market_value_module = importlib.util.module_from_spec(schwab_market_value_spec)
sys.modules[schwab_market_value_spec.name] = schwab_market_value_module
schwab_market_value_spec.loader.exec_module(schwab_market_value_module)

schwab_transactions_path = schwab_market_value_run_dir / 'account_1_combined_transactions.json'
schwab_transactions = json.loads(schwab_transactions_path.read_text(encoding='utf-8'))

schwab_current_market_value = _schwab_current_market_value(acct)
if schwab_current_market_value is None:
    raise ValueError('Could not find a current Schwab liquidation/account value in acct currentBalances.')

# Top row: keep the original Block 7 market-value proxy calculation.
schwab_realized_daily, schwab_realized_events, schwab_realized_unmatched = (
    schwab_market_value_module.build_realized_curve(schwab_transactions)
)
schwab_market_value_daily = schwab_market_value_module.add_realized_equity_proxy(
    schwab_realized_daily,
    ending_equity=schwab_current_market_value,
    equity_column='market_value',
)
schwab_external_cash_flow = schwab_market_value_module.build_external_cash_flow_daily(
    schwab_transactions,
)
schwab_daily_realized_pnl = schwab_market_value_daily['daily_pnl'].copy()
schwab_daily_realized_pnl.index = pd.to_datetime(schwab_daily_realized_pnl.index).normalize()
schwab_daily_realized_pnl = schwab_daily_realized_pnl.groupby(level=0).sum()
schwab_external_cash_flow_normalized = schwab_external_cash_flow.copy()
schwab_external_cash_flow_normalized.index = pd.to_datetime(schwab_external_cash_flow_normalized.index).normalize()
schwab_external_cash_flow_normalized = schwab_external_cash_flow_normalized.groupby(level=0).sum()
schwab_flow_pnl_index = schwab_daily_realized_pnl.index.union(
    schwab_external_cash_flow_normalized.index
).sort_values()
schwab_daily_realized_pnl = schwab_daily_realized_pnl.reindex(schwab_flow_pnl_index).fillna(0.0)
schwab_external_cash_flow_aligned = schwab_external_cash_flow_normalized.reindex(schwab_flow_pnl_index).fillna(0.0)
schwab_cumulative_flow_pnl = (
    schwab_daily_realized_pnl + schwab_external_cash_flow_aligned
).cumsum()
if not schwab_cumulative_flow_pnl.empty:
    schwab_cumulative_flow_pnl = pd.concat(
        [
            pd.Series(
                [0.0],
                index=[pd.Timestamp(schwab_cumulative_flow_pnl.index.min()) - pd.Timedelta(days=1)],
            ),
            schwab_cumulative_flow_pnl,
        ]
    )
schwab_intuitive_portfolio_daily = pd.DataFrame(
    {
        'cumulative_deposits_withdrawals_realized_pnl': schwab_cumulative_flow_pnl,
    }
)
# Bottom row: cleaned cash position from the position/cash reconstruction.
schwab_position_equity = schwab_market_value_module.build_position_equity_history(
    schwab_transactions,
    positions,
    acct,
    use_current_marks_for_open_positions=True,
)
schwab_position_equity_totals = schwab_position_equity['totals']
schwab_position_value_by_position = schwab_position_equity['position_value_by_position']
schwab_position_value_by_underlying = schwab_position_equity['position_value_by_group']
schwab_position_quantity_history = schwab_position_equity['quantity_history']
schwab_position_mark_history = schwab_position_equity['mark_history']
schwab_current_positions_frame = schwab_position_equity['current_positions']
schwab_total_account_value_raw = schwab_position_equity_totals['total_account_value']
(
    schwab_total_account_value,
    schwab_total_value_spike_mask,
) = schwab_market_value_module.despike_isolated_cash_position(
    schwab_total_account_value_raw,
    window=7,
    min_abs_spike=1_500,
    scale_multiplier=4,
    max_spike_span=3,
)
schwab_reconstructed_portfolio_daily = pd.DataFrame(
    {'reconstructed_total_account_value': schwab_total_account_value}
)

schwab_cash_raw = schwab_position_equity_totals['cash']
(
    schwab_cash_position,
    schwab_cash_spike_mask,
) = schwab_market_value_module.despike_isolated_cash_position(
    schwab_cash_raw,
    window=7,
    min_abs_spike=5_000,
    scale_multiplier=8,
    max_spike_span=3,
)
schwab_cash_spikes = pd.DataFrame(
    {
        'raw_cash': schwab_cash_raw.loc[schwab_cash_spike_mask],
        'cleaned_cash': schwab_cash_position.loc[schwab_cash_spike_mask],
    }
)
schwab_value_spikes = pd.DataFrame(
    {
        'raw_total_value': schwab_total_account_value_raw.loc[schwab_total_value_spike_mask],
        'cleaned_total_value': schwab_total_account_value.loc[schwab_total_value_spike_mask],
    }
)
schwab_risk_adjusted_window = time_frame_long if 'time_frame_long' in globals() else 200
schwab_portfolio_rolling_sharpe = schwab_market_value_module.rolling_sharpe_from_value(
    schwab_total_account_value,
    window=schwab_risk_adjusted_window,
    external_cash_flow=schwab_external_cash_flow,
)
schwab_portfolio_rolling_sortino = schwab_market_value_module.rolling_sortino_from_value(
    schwab_total_account_value,
    window=schwab_risk_adjusted_window,
    external_cash_flow=schwab_external_cash_flow,
)
schwab_latest_rolling_sharpe = (
    schwab_portfolio_rolling_sharpe.dropna().iloc[-1]
    if not schwab_portfolio_rolling_sharpe.dropna().empty
    else None
)
schwab_latest_rolling_sortino = (
    schwab_portfolio_rolling_sortino.dropna().iloc[-1]
    if not schwab_portfolio_rolling_sortino.dropna().empty
    else None
)

schwab_market_value_start = pd.Timestamp(schwab_reconstructed_portfolio_daily.index.min()).date()
schwab_market_value_end = pd.Timestamp(schwab_reconstructed_portfolio_daily.index.max()).date()

schwab_market_value_title = (
    f"Schwab Reconstructed Portfolio Value, Cash, and Risk Ratios "
    f"({schwab_market_value_start} to {schwab_market_value_end}, "
    f"ending ${schwab_current_market_value:,.2f})"
)

schwab_portfolio_market_value_fig = schwab_market_value_module.make_market_value_cash_stack_figure(
    schwab_reconstructed_portfolio_daily,
    schwab_cash_position,
    title=schwab_market_value_title,
    value_column='reconstructed_total_account_value',
    value_label='Estimated total account value',
    value_hover_label='Portfolio value',
    sharpe_series=schwab_portfolio_rolling_sharpe,
    sharpe_label=f'{schwab_risk_adjusted_window}-day cash-flow-adjusted Sharpe',
    sortino_series=schwab_portfolio_rolling_sortino,
    sortino_label=f'{schwab_risk_adjusted_window}-day cash-flow-adjusted Sortino',
)
# schwab_portfolio_market_value_fig is rendered in the tabbed dashboard below.
schwab_market_value_summary = pd.DataFrame(
    [
        {
            'first_value_date': schwab_market_value_start,
            'last_value_date': schwab_market_value_end,
            'starting_estimated_total_account_value': schwab_total_account_value.iloc[0],
            'ending_estimated_total_account_value': schwab_total_account_value.iloc[-1],
            'minimum_estimated_total_account_value': schwab_total_account_value.min(),
            'maximum_estimated_total_account_value': schwab_total_account_value.max(),
            'ending_current_schwab_account_value': schwab_current_market_value,
            'ending_reconstructed_cash_plus_positions_value': schwab_total_account_value_raw.iloc[-1],
            'min_reconstructed_cash_plus_positions_value': schwab_total_account_value_raw.min(),
            'max_reconstructed_cash_plus_positions_value': schwab_total_account_value_raw.max(),
            'ending_cumulative_deposits_withdrawals_realized_pnl': schwab_cumulative_flow_pnl.iloc[-1],
            'current_value_minus_cumulative_flow_pnl': schwab_current_market_value - schwab_cumulative_flow_pnl.iloc[-1],
            'net_external_cash_flow': schwab_external_cash_flow.sum(),
            'external_cash_flow_days': int(schwab_external_cash_flow.ne(0).sum()),
            'risk_adjusted_window': schwab_risk_adjusted_window,
            'latest_rolling_cash_flow_adjusted_sharpe': schwab_latest_rolling_sharpe,
            'latest_rolling_cash_flow_adjusted_sortino': schwab_latest_rolling_sortino,
            'ending_cash': schwab_cash_position.iloc[-1],
            'ending_position_market_value': schwab_position_equity_totals['position_market_value'].iloc[-1],
            'removed_cash_spike_points': int(schwab_cash_spike_mask.sum()),
            'removed_total_value_spike_points': int(schwab_total_value_spike_mask.sum()),
            'current_positions': len(schwab_current_positions_frame),
            'run_dir': str(schwab_market_value_run_dir),
        }
    ]
)
display(schwab_market_value_summary)

# %% [portfolio block 8]
# Code Block 8: DTE ladder
# =========================
# 8) Options expiration ladder
# =========================
from Quantapp.visualization.views.portfolio_profile.performance_structure import plot_options_expiration_ladder

options_expiration_ladder_fig = plot_options_expiration_ladder(positions_df)

# %% [portfolio block 9]
# Code Block 9: Option P/L at expiration and net cost basis
# Retrieve net cost basis for each option grouped by ticker
# net cost basis = net_quantity * average_price * 100 (per contract)
import importlib
from Quantapp.visualization.views.portfolio_profile.performance_structure import option_expiration_pl as option_expiration_pl_module

option_expiration_pl_module = importlib.reload(option_expiration_pl_module)
build_option_payoff_structure_frame = option_expiration_pl_module.build_option_payoff_structure_frame
build_option_profit_loss_extremes_table = option_expiration_pl_module.build_option_profit_loss_extremes_table
build_option_expiration_pl_dash_app = option_expiration_pl_module.build_option_expiration_pl_dash_app

# The unified portfolio dashboard owns the notebook output; keep this legacy hook inert.
display_option_expiration_pl_dash_app = lambda *_args, **_kwargs: None

if not positions_df.empty:
    positions_df['net_cost_basis'] = positions_df['net_quantity'] * positions_df['average_price'] * 100
    net_cost_basis = positions_df.groupby('underlying')['net_cost_basis'].sum().rename('net_cost_basis')
    # display(net_cost_basis.to_frame())
else:
    net_cost_basis = pd.Series(dtype=float, name='net_cost_basis')
    print("No options positions found.")

available_option_underlyings = (
    sorted(positions_df['underlying'].dropna().unique().tolist())
    if not positions_df.empty else []
)
benchmark_label_for_beta = benchmark_str if 'benchmark_str' in globals() else 'SPY'
portfolio_latest_prices = (
    portfolio_closing_prices.ffill().iloc[-1].dropna().to_dict()
    if isinstance(portfolio_closing_prices, pd.DataFrame) and not portfolio_closing_prices.empty
    else {}
)

def _ticker_lookup_key(ticker):
    return ''.join(character for character in str(ticker).upper() if character.isalnum())

benchmark_current_price = (
    float(benchmark_close.ffill().iloc[-1])
    if isinstance(benchmark_close, pd.Series) and not benchmark_close.dropna().empty
    else np.nan
)
benchmark_returns_for_beta = (
    benchmark_close.pct_change().dropna()
    if isinstance(benchmark_close, pd.Series)
    else pd.Series(dtype=float)
)
asset_returns_for_beta = (
    portfolio_closing_prices.pct_change()
    if isinstance(portfolio_closing_prices, pd.DataFrame) and not portfolio_closing_prices.empty
    else pd.DataFrame()
)
underlying_beta_map = {}

if not asset_returns_for_beta.empty and not benchmark_returns_for_beta.empty:
    for underlying_name in asset_returns_for_beta.columns:
        aligned_returns = pd.concat(
            [
                asset_returns_for_beta[underlying_name].rename('asset'),
                benchmark_returns_for_beta.rename('benchmark'),
            ],
            axis=1,
        ).dropna()

        if len(aligned_returns) < 2 or aligned_returns['benchmark'].var() == 0:
            underlying_beta_map[underlying_name] = 1.0
            continue

        beta_value = aligned_returns['asset'].cov(aligned_returns['benchmark']) / aligned_returns['benchmark'].var()

        if pd.notna(beta_value) and np.isfinite(beta_value):
            underlying_beta_map[underlying_name] = float(beta_value)
        else:
            underlying_beta_map[underlying_name] = 1.0
else:
    underlying_beta_map = {underlying_name: 1.0 for underlying_name in available_option_underlyings}

if available_option_underlyings:
    option_payoff_structure_frame = build_option_payoff_structure_frame(
        positions_df,
        portfolio_latest_prices=portfolio_latest_prices,
    )
    option_expiration_pl_app = display_option_expiration_pl_dash_app(
        positions_df,
        net_cost_basis=net_cost_basis,
        portfolio_latest_prices=portfolio_latest_prices,
        benchmark_current_price=benchmark_current_price,
        underlying_beta_map=underlying_beta_map,
        benchmark_label=benchmark_label_for_beta,
        payoff_structure_frame=option_payoff_structure_frame,
        jupyter_mode='inline',
        jupyter_height=3200,
        debug=False,
    )
    option_expiration_pl_view = None
    option_profit_loss_extremes_table = build_option_profit_loss_extremes_table(
        positions_df,
        portfolio_latest_prices=portfolio_latest_prices,
    )
    option_direction_sign = (
        option_profit_loss_extremes_table['direction']
        .map({'up': 1.0, 'down': -1.0, 'flat': 0.0, 'none': 0.0})
        .fillna(1.0)
        .rename('sign')
    )
    option_direction_sign_by_lookup_key = {
        _ticker_lookup_key(ticker): sign
        for ticker, sign in option_direction_sign.items()
    }
    # display(option_profit_loss_extremes_table)
else:
    option_payoff_structure_frame = pd.DataFrame(columns=['payoff_structure', 'payoff_bias']).rename_axis('ticker')
    option_expiration_pl_app = None
    option_profit_loss_extremes_table = pd.DataFrame(columns=['max_profit', 'max_loss', 'direction'])
    option_direction_sign = pd.Series(dtype=float, name='sign')
    option_direction_sign_by_lookup_key = {}

# %% [portfolio block 10]
# Code Block 10: Option Greek sensitivities
# Pull current option-chain IV/marks, then estimate per-leg Black-Scholes Greeks.
import importlib
from Quantapp.analytics import option_greeks as option_greeks_module
from Quantapp.data import get_current_options_chain

option_greeks_module = importlib.reload(option_greeks_module)
build_option_greek_sensitivity_frame = option_greeks_module.build_option_greek_sensitivity_frame
build_option_greeks_frame = option_greeks_module.build_option_greeks_frame
summarize_option_greeks = option_greeks_module.summarize_option_greeks

OPTION_GREEK_RISK_FREE_TICKER = '^IRX'
OPTION_GREEK_DEFAULT_ANNUAL_RATE = 0.02
OPTION_GREEK_DIVIDEND_YIELD_BY_UNDERLYING = {}
OPTION_GREEK_REFRESH_CHAINS = globals().get('OPTION_GREEK_REFRESH_CHAINS', False)
OPTION_GREEK_SCENARIO_POINTS = int(globals().get('OPTION_GREEK_SCENARIO_POINTS', 61))
OPTION_GREEK_DTE_BUCKETS = globals().get('OPTION_GREEK_DTE_BUCKETS', [('Total', None, None)])
OPTION_GREEK_CHAIN_CACHE = globals().get('OPTION_GREEK_CHAIN_CACHE', {})
OPTION_GREEK_PROVIDER_TICKER_ALIASES = {
    'BRKB': {
        'massive': ['BRK.B', 'BRK-B', 'BRK/B'],
        'yfinance': ['BRK-B', 'BRK.B', 'BRK/B'],
    },
    'BFB': {
        'massive': ['BF.B', 'BF-B', 'BF/B'],
        'yfinance': ['BF-B', 'BF.B', 'BF/B'],
    },
}


def _option_greek_ticker_lookup_key(ticker):
    return ''.join(character for character in str(ticker).upper() if character.isalnum())


def _dedupe_preserve_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value is None or str(value).strip() == '':
            continue
        value = str(value).strip()
        lookup_key = value.upper()
        if lookup_key in seen:
            continue
        seen.add(lookup_key)
        deduped.append(value)
    return deduped


def _provider_ticker_candidates(underlying, provider):
    raw_ticker = str(underlying).strip()
    lookup_key = _option_greek_ticker_lookup_key(raw_ticker)
    aliases = OPTION_GREEK_PROVIDER_TICKER_ALIASES.get(lookup_key, {})
    provider_aliases = aliases.get(provider, [])

    if provider == 'yfinance':
        fallback_ticker = normalize_yf_ticker(raw_ticker)
    else:
        fallback_ticker = raw_ticker

    return _dedupe_preserve_order([*provider_aliases, fallback_ticker, raw_ticker])


def _latest_annual_risk_free_rate(ticker=OPTION_GREEK_RISK_FREE_TICKER, fallback=OPTION_GREEK_DEFAULT_ANNUAL_RATE):
    try:
        rate_history = qa_yf.Ticker(ticker).history(period='10d', interval='1d')
        close_series = (
            rate_history['Close']
            if isinstance(rate_history, pd.DataFrame) and 'Close' in rate_history.columns
            else pd.Series(dtype=float)
        )
        rate_close = pd.to_numeric(close_series, errors='coerce').dropna()
        if not rate_close.empty:
            return max(float(rate_close.iloc[-1]) / 100.0, 0.0)
    except Exception as error:
        print(f'Risk-free rate load failed for {ticker}: {error}. Using {fallback:.2%}.')
    return float(fallback)


def _fallback_yfinance_current_option_chain(underlying, fallback_spot=np.nan):
    candidate_errors = {}
    for provider_ticker in _provider_ticker_candidates(underlying, 'yfinance'):
        try:
            ticker_handle = qa_yf.Ticker(provider_ticker)
            expirations = list(ticker_handle.options or [])
            if not expirations:
                raise ValueError('No listed option expirations returned.')

            today = pd.Timestamp.today().normalize()
            frames = []
            for expiration_date in expirations:
                option_chain = ticker_handle.option_chain(expiration_date)
                days_till_expiration = (pd.to_datetime(expiration_date) - today).days
                for raw_frame, option_type in ((option_chain.calls, 'Call'), (option_chain.puts, 'Put')):
                    frame = raw_frame.copy()
                    if frame.empty:
                        continue
                    frame['Type'] = option_type
                    frame['Expiration Date'] = expiration_date
                    frame['Days Till Expiration'] = days_till_expiration
                    frame['bid'] = pd.to_numeric(frame.get('bid'), errors='coerce')
                    frame['ask'] = pd.to_numeric(frame.get('ask'), errors='coerce')
                    frame['lastPrice'] = pd.to_numeric(frame.get('lastPrice'), errors='coerce')
                    frame['mid'] = (frame['bid'] + frame['ask']) / 2
                    frame['bid-ask spread'] = frame['ask'] - frame['bid']
                    frames.append(frame)

            if not frames:
                raise ValueError('No current option contracts returned.')
            return pd.concat(frames, ignore_index=True), fallback_spot, provider_ticker
        except Exception as error:
            candidate_errors[provider_ticker] = str(error)

    tried = ', '.join(candidate_errors.keys())
    raise ValueError(f'No yfinance option chain for {underlying}; tried {tried}. Details: {candidate_errors}')


def _load_current_option_chain_for_greeks(underlying, fallback_spot=np.nan):
    massive_candidate_errors = {}
    for provider_ticker in _provider_ticker_candidates(underlying, 'massive'):
        try:
            current_chain = get_current_options_chain(
                provider_ticker,
                fallback_underlying_price=fallback_spot,
            )
            source_label = 'massive' if provider_ticker == str(underlying).strip() else f'massive:{provider_ticker}'
            return current_chain.chain, current_chain.underlying_price, source_label
        except Exception as error:
            massive_candidate_errors[provider_ticker] = str(error)

    try:
        chain_frame, underlying_price, provider_ticker = _fallback_yfinance_current_option_chain(
            underlying,
            fallback_spot=fallback_spot,
        )
        print(
            f'{underlying}: Massive chain load failed for aliases {list(massive_candidate_errors)}; '
            f'used yfinance fallback {provider_ticker}.'
        )
        return chain_frame, underlying_price, f'yfinance:{provider_ticker}'
    except Exception as fallback_error:
        raise RuntimeError(
            f'Massive failed for aliases {massive_candidate_errors}; '
            f'yfinance fallback failed ({fallback_error})'
        ) from fallback_error


def _option_chain_type_key(value):
    normalized = str(value).strip().upper()
    if normalized in {'C', 'CALL'}:
        return 'C'
    if normalized in {'P', 'PUT'}:
        return 'P'
    return None


def _filter_chain_to_open_positions(chain_frame, underlying):
    if chain_frame is None or len(chain_frame) == 0:
        return chain_frame
    if 'positions_df' not in globals() or positions_df.empty:
        return chain_frame

    required_position_columns = {'underlying', 'expiration', 'option_type', 'strike'}
    if required_position_columns - set(positions_df.columns):
        return chain_frame

    underlying_key = _option_greek_ticker_lookup_key(underlying)
    scoped_positions = positions_df[
        positions_df['underlying'].map(_option_greek_ticker_lookup_key).eq(underlying_key)
    ].copy()
    if scoped_positions.empty:
        return chain_frame

    scoped_positions['_greek_filter_expiration_key'] = pd.to_datetime(
        scoped_positions['expiration'],
        errors='coerce',
    ).dt.normalize()
    scoped_positions['_greek_filter_option_type_key'] = scoped_positions['option_type'].map(_option_chain_type_key)
    scoped_positions['_greek_filter_strike_key'] = pd.to_numeric(
        scoped_positions['strike'],
        errors='coerce',
    ).round(6)
    position_keys = scoped_positions[
        [
            '_greek_filter_expiration_key',
            '_greek_filter_option_type_key',
            '_greek_filter_strike_key',
        ]
    ].dropna().drop_duplicates()
    if position_keys.empty:
        return chain_frame

    chain = pd.DataFrame(chain_frame).copy()
    chain['_greek_filter_expiration_key'] = pd.to_datetime(
        chain['Expiration Date'] if 'Expiration Date' in chain.columns else pd.Series(pd.NaT, index=chain.index),
        errors='coerce',
    ).dt.normalize()
    chain['_greek_filter_option_type_key'] = (
        chain['Type'] if 'Type' in chain.columns else pd.Series(np.nan, index=chain.index)
    ).map(_option_chain_type_key)
    chain['_greek_filter_strike_key'] = pd.to_numeric(
        chain['strike'] if 'strike' in chain.columns else pd.Series(np.nan, index=chain.index),
        errors='coerce',
    ).round(6)

    filtered = chain.merge(
        position_keys,
        how='inner',
        on=[
            '_greek_filter_expiration_key',
            '_greek_filter_option_type_key',
            '_greek_filter_strike_key',
        ],
    )
    helper_columns = [column for column in filtered.columns if column.startswith('_greek_filter_')]
    if filtered.empty:
        print(f'{underlying}: no chain rows matched open-position keys; using full chain for matching.')
        return chain_frame
    return filtered.drop(columns=helper_columns)


option_greek_annual_rate = _latest_annual_risk_free_rate()
option_current_chains_by_underlying = {}
option_spot_price_by_underlying = dict(portfolio_latest_prices)
option_chain_sources_by_underlying = {}
option_chain_load_errors = {}

for underlying in available_option_underlyings:
    fallback_spot = portfolio_latest_prices.get(underlying, np.nan)
    cache_key = _option_greek_ticker_lookup_key(underlying)
    try:
        if not OPTION_GREEK_REFRESH_CHAINS and cache_key in OPTION_GREEK_CHAIN_CACHE:
            cached_chain = OPTION_GREEK_CHAIN_CACHE[cache_key]
            chain_frame = pd.DataFrame(cached_chain['chain_frame']).copy()
            chain_spot = cached_chain.get('underlying_price', np.nan)
            chain_source = cached_chain.get('source', 'cache')
            load_action = 'reused cached'
        else:
            chain_frame, chain_spot, chain_source = _load_current_option_chain_for_greeks(
                underlying,
                fallback_spot=fallback_spot,
            )
            OPTION_GREEK_CHAIN_CACHE[cache_key] = {
                'chain_frame': pd.DataFrame(chain_frame).copy(),
                'underlying_price': chain_spot,
                'source': chain_source,
            }
            load_action = 'loaded'

        raw_contract_count = len(chain_frame)
        chain_frame = _filter_chain_to_open_positions(chain_frame, underlying)
        option_current_chains_by_underlying[underlying] = chain_frame
        option_chain_sources_by_underlying[underlying] = chain_source
        if pd.notna(chain_spot):
            option_spot_price_by_underlying[underlying] = float(chain_spot)
        print(
            f'{underlying}: {load_action} {raw_contract_count:,} contracts from {chain_source}; '
            f'using {len(chain_frame):,} contract rows for open-position matching.'
        )
    except Exception as error:
        option_chain_load_errors[underlying] = str(error)
        print(f'{underlying}: option chain load failed for Greeks: {error}')

if available_option_underlyings and option_current_chains_by_underlying:
    option_greeks_frame = build_option_greeks_frame(
        positions_df,
        option_current_chains_by_underlying,
        option_spot_price_by_underlying,
        annual_rate=option_greek_annual_rate,
        dividend_yield_by_underlying=OPTION_GREEK_DIVIDEND_YIELD_BY_UNDERLYING,
        as_of_date=pd.Timestamp.today().normalize(),
    )
    option_greeks_summary = summarize_option_greeks(option_greeks_frame)

    print(
        f'Option Greeks estimated with annual risk-free rate {option_greek_annual_rate:.2%}. '
        'Dividend yield defaults to 0 unless supplied in OPTION_GREEK_DIVIDEND_YIELD_BY_UNDERLYING.'
    )
    display(option_greeks_summary)

    option_greek_detail_columns = [
        'symbol',
        'underlying',
        'expiration',
        'option_type',
        'strike',
        'net_quantity',
        'spot',
        'chain_option_mark',
        'chain_implied_volatility',
        'delta',
        'gamma',
        'theta_per_day',
        'vega_per_vol_point',
        'rho_per_rate_point',
        'position_delta_shares',
        'position_delta_notional',
        'position_gamma_delta_per_dollar',
        'position_theta_per_day',
        'position_vega_per_vol_point',
        'position_rho_per_rate_point',
        'greek_status',
    ]
    option_greek_detail_columns = [
        column for column in option_greek_detail_columns
        if column in option_greeks_frame.columns
    ]
    display(
        option_greeks_frame[option_greek_detail_columns]
        .sort_values(['greek_status', 'underlying', 'expiration', 'strike'])
        .reset_index(drop=True)
    )

    option_greek_scenario_moves = np.linspace(-0.30, 0.30, max(3, OPTION_GREEK_SCENARIO_POINTS))
    option_greek_sensitivity_frame = build_option_greek_sensitivity_frame(
        option_greeks_frame,
        scenario_moves=option_greek_scenario_moves,
        dte_buckets=OPTION_GREEK_DTE_BUCKETS,
    )
    print(
        f'Greek sensitivity grid used {len(option_greek_scenario_moves):,} scenario points and '
        f'{len(OPTION_GREEK_DTE_BUCKETS):,} DTE bucket(s). '
        'Set OPTION_GREEK_SCENARIO_POINTS or OPTION_GREEK_DTE_BUCKETS before rerunning to change this.'
    )

    def _build_option_greek_sensitivity_figure(sensitivity_frame):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if sensitivity_frame.empty:
            return None
        if 'dte_bucket' not in sensitivity_frame.columns:
            raise ValueError(
                "Greek sensitivity frame is missing 'dte_bucket'. "
                "Rerun Code Block 10 so the reloaded option_greeks module rebuilds it."
            )

        greek_specs = [
            {
                'title': 'Delta<br><span style="font-size:11px">notional $</span>',
                'label': 'Delta',
                'column': 'position_delta_notional',
                'color': '#38BDF8',
            },
            {
                'title': 'Gamma<br><span style="font-size:11px">delta shares / $</span>',
                'label': 'Gamma',
                'column': 'position_gamma_delta_per_dollar',
                'color': '#A78BFA',
            },
            {
                'title': 'Theta<br><span style="font-size:11px">$ / day</span>',
                'label': 'Theta',
                'column': 'position_theta_per_day',
                'color': '#F59E0B',
            },
            {
                'title': 'Vega<br><span style="font-size:11px">$ / vol point</span>',
                'label': 'Vega',
                'column': 'position_vega_per_vol_point',
                'color': '#22C55E',
            },
            {
                'title': 'Rho<br><span style="font-size:11px">$ / rate point</span>',
                'label': 'Rho',
                'column': 'position_rho_per_rate_point',
                'color': '#F472B6',
            },
        ]
        desired_bucket_order = [bucket[0] for bucket in OPTION_GREEK_DTE_BUCKETS]
        bucket_values = sensitivity_frame['dte_bucket'].dropna().astype(str).unique().tolist()
        ordered_buckets = [
            *[bucket for bucket in desired_bucket_order if bucket in bucket_values],
            *sorted(bucket for bucket in bucket_values if bucket not in desired_bucket_order),
        ]
        underlying_values = sensitivity_frame['underlying'].dropna().astype(str).unique().tolist()
        ordered_underlyings = [
            *([underlying for underlying in ['TOTAL'] if underlying in underlying_values]),
            *sorted(underlying for underlying in underlying_values if underlying != 'TOTAL'),
        ]
        if not ordered_underlyings or not ordered_buckets:
            return None

        row_count = len(ordered_buckets)
        col_count = len(greek_specs)
        subplot_titles = [
            spec['title'] if row_index == 0 else ''
            for row_index in range(row_count)
            for spec in greek_specs
        ]
        fig = make_subplots(
            rows=row_count,
            cols=col_count,
            subplot_titles=subplot_titles,
            vertical_spacing=0.065,
            horizontal_spacing=0.045,
        )
        trace_groups = {}
        default_underlying = ordered_underlyings[0]

        for underlying_name in ordered_underlyings:
            scoped_underlying = sensitivity_frame[
                sensitivity_frame['underlying'].astype(str).eq(underlying_name)
            ].copy()
            trace_start = len(fig.data)
            visible = underlying_name == default_underlying

            for row_index, bucket_label in enumerate(ordered_buckets, start=1):
                scoped_bucket = scoped_underlying[
                    scoped_underlying['dte_bucket'].astype(str).eq(bucket_label)
                ].sort_values('scenario_move')

                for col_index, spec in enumerate(greek_specs, start=1):
                    fig.add_trace(
                        go.Scatter(
                            x=scoped_bucket['scenario_move'],
                            y=scoped_bucket[spec['column']],
                            mode='lines',
                            name=spec['label'],
                            line=dict(color=spec['color'], width=2.2),
                            visible=visible,
                            showlegend=row_index == 1,
                            hovertemplate=(
                                f'DTE: {bucket_label}<br>'
                                'Move: %{x:.1%}<br>'
                                f"{spec['label']}: " + '%{y:,.2f}<extra></extra>'
                            ),
                        ),
                        row=row_index,
                        col=col_index,
                    )
                    fig.add_vline(
                        x=0,
                        line_width=1,
                        line_dash='dash',
                        line_color='rgba(226, 232, 240, 0.55)',
                        row=row_index,
                        col=col_index,
                    )
                    fig.update_xaxes(
                        tickformat='.0%',
                        title_text='Spot move' if row_index == row_count else None,
                        row=row_index,
                        col=col_index,
                    )

            trace_groups[underlying_name] = (trace_start, len(fig.data))

        for row_index, bucket_label in enumerate(ordered_buckets, start=1):
            axis_number = (row_index - 1) * col_count + 1
            yaxis_key = 'yaxis' if axis_number == 1 else f'yaxis{axis_number}'
            y_domain = fig.layout[yaxis_key].domain
            fig.add_annotation(
                text=f'<b>{bucket_label}</b>',
                xref='paper',
                yref='paper',
                x=-0.035,
                y=sum(y_domain) / 2,
                xanchor='right',
                yanchor='middle',
                showarrow=False,
                font=dict(color='#E5E7EB', size=13),
            )

        buttons = []
        total_traces = len(fig.data)
        for underlying_name in ordered_underlyings:
            visible = [False] * total_traces
            trace_start, trace_end = trace_groups[underlying_name]
            for trace_idx in range(trace_start, trace_end):
                visible[trace_idx] = True
            buttons.append(
                {
                    'label': underlying_name,
                    'method': 'update',
                    'args': [
                        {'visible': visible},
                        {'title': f'Option Greek Sensitivity: {underlying_name}'},
                    ],
                }
            )

        fig.update_layout(
            title=f'Option Greek Sensitivity: {default_underlying}',
            template='plotly_dark',
            paper_bgcolor='#0F172A',
            plot_bgcolor='#111827',
            font=dict(color='#E5E7EB'),
            height=440 if row_count == 1 else max(820, 225 * row_count),
            hovermode='x unified',
            margin=dict(t=130, b=90, l=120, r=45),
            updatemenus=[
                {
                    'buttons': buttons,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0,
                    'xanchor': 'left',
                    'y': 1.11,
                    'yanchor': 'top',
                    'bgcolor': '#1F2937',
                    'bordercolor': '#475569',
                    'font': {'color': '#E5E7EB'},
                }
            ],
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.07,
                xanchor='left',
                x=0,
                bgcolor='rgba(15, 23, 42, 0.72)',
            ),
        )
        fig.update_xaxes(gridcolor='rgba(148, 163, 184, 0.22)', zerolinecolor='rgba(226, 232, 240, 0.55)')
        fig.update_yaxes(gridcolor='rgba(148, 163, 184, 0.22)', zerolinecolor='rgba(226, 232, 240, 0.55)')
        return fig

    option_greek_sensitivity_fig = _build_option_greek_sensitivity_figure(
        option_greek_sensitivity_frame
    )
    if option_greek_sensitivity_fig is not None:
        # option_greek_sensitivity_fig is rendered in the tabbed dashboard below.
        pass
    else:
        print('No valid Greek sensitivity curves could be built from the priced option legs.')
else:
    option_greeks_frame = pd.DataFrame()
    option_greeks_summary = pd.DataFrame()
    print('No current option chains were loaded, so option Greeks were not calculated.')

# %% [portfolio block 11]
# Code Block 11: Option directional exposure summary
# Spread-aware long/short/neutral view using delta, beta-adjusted delta, and scenario P/L.
OPTION_DIRECTION_SCENARIO_MOVES = [-0.10, -0.05, 0.0, 0.05, 0.10]
OPTION_DIRECTION_NEUTRAL_BAND = float(globals().get('OPTION_DIRECTION_NEUTRAL_BAND', 0.025))
OPTION_DIRECTION_STRONG_BAND = float(globals().get('OPTION_DIRECTION_STRONG_BAND', 0.10))
OPTION_PAYOFF_TAIL_BAND = float(globals().get('OPTION_PAYOFF_TAIL_BAND', 0.10))
OPTION_PAYOFF_FLAT_BAND = float(globals().get('OPTION_PAYOFF_FLAT_BAND', 0.05))


def _option_direction_lookup_key(ticker):
    return ''.join(character for character in str(ticker).upper() if character.isalnum())


def _option_direction_current_account_value():
    current_value = globals().get('schwab_current_market_value', np.nan)
    if pd.notna(current_value):
        return float(current_value)

    account_payload = globals().get('acct', {})
    securities_account = account_payload.get('securitiesAccount', {}) if isinstance(account_payload, dict) else {}
    balances = securities_account.get('currentBalances', {}) if isinstance(securities_account, dict) else {}
    if not isinstance(balances, dict):
        return np.nan

    for key in ('liquidationValue', 'accountValue'):
        value = balances.get(key)
        if value is not None:
            return float(value)

    cash = balances.get('cashBalance')
    long_market_value = balances.get('longMarketValue')
    short_market_value = balances.get('shortMarketValue')
    if cash is not None or long_market_value is not None or short_market_value is not None:
        return float(cash or 0.0) + float(long_market_value or 0.0) + float(short_market_value or 0.0)
    return np.nan


def _option_direction_beta_lookup():
    raw_beta_map = globals().get('underlying_beta_map', {})
    if not isinstance(raw_beta_map, dict):
        return {}

    beta_lookup = {}
    for ticker, beta in raw_beta_map.items():
        if pd.notna(beta) and np.isfinite(beta):
            beta_lookup[_option_direction_lookup_key(ticker)] = float(beta)
    return beta_lookup


def _option_direction_label(score):
    if pd.isna(score) or not np.isfinite(score):
        return 'Unknown'
    if score >= OPTION_DIRECTION_STRONG_BAND:
        return 'Strong Long'
    if score >= OPTION_DIRECTION_NEUTRAL_BAND:
        return 'Long'
    if score <= -OPTION_DIRECTION_STRONG_BAND:
        return 'Strong Short'
    if score <= -OPTION_DIRECTION_NEUTRAL_BAND:
        return 'Short'
    return 'Neutral'


def _first_valid_numeric(values):
    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
    return float(numeric_values.iloc[0]) if not numeric_values.empty else np.nan


def _build_option_direction_scenario_pnl(greeks_frame):
    if 'build_option_greek_sensitivity_frame' not in globals() or greeks_frame.empty:
        return pd.DataFrame()

    scenario_frame = build_option_greek_sensitivity_frame(
        greeks_frame,
        scenario_moves=np.array(OPTION_DIRECTION_SCENARIO_MOVES, dtype=float),
        dte_buckets=[('Total', None, None)],
    )
    if scenario_frame.empty:
        return pd.DataFrame()

    scenario_values = scenario_frame[
        scenario_frame['dte_bucket'].astype(str).eq('Total')
    ].copy()
    scenario_values['scenario_move'] = scenario_values['scenario_move'].round(4)
    value_grid = scenario_values.pivot_table(
        index='underlying',
        columns='scenario_move',
        values='scenario_position_value',
        aggfunc='sum',
    )
    if 0.0 not in value_grid.columns:
        return pd.DataFrame()

    base_value = value_grid[0.0]
    scenario_pnl = pd.DataFrame(index=value_grid.index)
    scenario_column_names = {
        -0.10: 'pnl_down_10pct',
        -0.05: 'pnl_down_5pct',
        0.05: 'pnl_up_5pct',
        0.10: 'pnl_up_10pct',
    }
    for scenario_move, column_name in scenario_column_names.items():
        rounded_move = round(float(scenario_move), 4)
        if rounded_move in value_grid.columns:
            scenario_pnl[column_name] = value_grid[rounded_move] - base_value
        else:
            scenario_pnl[column_name] = np.nan

    scenario_pnl['scenario_skew_5pct'] = scenario_pnl['pnl_up_5pct'] - scenario_pnl['pnl_down_5pct']
    scenario_pnl['scenario_convexity_5pct'] = scenario_pnl['pnl_up_5pct'] + scenario_pnl['pnl_down_5pct']
    scenario_pnl.index.name = 'ticker'
    return scenario_pnl


def _option_direction_price_for_underlying(underlying):
    price_map = globals().get('portfolio_latest_prices', {})
    if not isinstance(price_map, dict):
        return np.nan

    direct_price = price_map.get(underlying, np.nan)
    if pd.notna(direct_price):
        return float(direct_price)

    target_key = _option_direction_lookup_key(underlying)
    for ticker, price in price_map.items():
        if _option_direction_lookup_key(ticker) == target_key and pd.notna(price):
            return float(price)
    return np.nan


def _option_expiration_pl_at_prices(legs_frame, prices):
    prices = np.asarray(prices, dtype=float)
    total_pl = np.zeros(len(prices), dtype=float)

    for _, leg in legs_frame.iterrows():
        strike = float(leg['strike'])
        option_type = str(leg['option_type']).upper()
        net_quantity = float(leg['net_quantity'])
        average_price = float(leg['average_price'])
        intrinsic = (
            np.maximum(prices - strike, 0.0)
            if option_type == 'C'
            else np.maximum(strike - prices, 0.0)
        )
        total_pl += net_quantity * (intrinsic - average_price) * 100.0
    return total_pl


def _payoff_structure_label(*, low_pl, middle_pl, high_pl, best_low, best_high, max_pl, min_pl):
    pl_scale = max(abs(max_pl), abs(min_pl), 1.0)
    payoff_tolerance = pl_scale * OPTION_PAYOFF_FLAT_BAND
    payoff_range = max_pl - min_pl

    if payoff_range <= payoff_tolerance:
        return 'Flat / Neutral', 'neutral'

    middle_beats_tails = middle_pl >= max(low_pl, high_pl) + payoff_tolerance
    tails_beat_middle = min(low_pl, high_pl) >= middle_pl + payoff_tolerance

    if tails_beat_middle:
        return 'Long Vol / Tails', 'convex'
    if middle_beats_tails and not (best_low or best_high):
        return 'Neutral / Range', 'neutral'
    if best_high and not best_low:
        return 'Bullish', 'bullish'
    if best_low and not best_high:
        return 'Bearish', 'bearish'
    if high_pl >= low_pl + payoff_tolerance and high_pl >= middle_pl + payoff_tolerance:
        return 'Bullish', 'bullish'
    if low_pl >= high_pl + payoff_tolerance and low_pl >= middle_pl + payoff_tolerance:
        return 'Bearish', 'bearish'
    if middle_beats_tails:
        return 'Neutral / Range', 'neutral'
    return 'Mixed / Complex', 'mixed'


def _build_option_payoff_structure_frame(positions_frame):
    structure_columns = [
        'payoff_structure',
        'payoff_bias',
        'payoff_best_price',
        'payoff_best_pl',
        'payoff_worst_pl',
        'payoff_low_tail_pl',
        'payoff_middle_pl',
        'payoff_high_tail_pl',
    ]
    if positions_frame is None or positions_frame.empty:
        return pd.DataFrame(columns=structure_columns).rename_axis('ticker')

    required_columns = {'underlying', 'strike', 'option_type', 'net_quantity', 'average_price'}
    if required_columns - set(positions_frame.columns):
        return pd.DataFrame(columns=structure_columns).rename_axis('ticker')

    rows = []
    for underlying, legs_frame in positions_frame.groupby('underlying', dropna=False):
        legs = legs_frame.copy()
        for column in ('strike', 'net_quantity', 'average_price'):
            legs[column] = pd.to_numeric(legs[column], errors='coerce')
        legs['option_type'] = legs['option_type'].astype(str).str.upper().str[0]
        legs = legs.dropna(subset=['strike', 'option_type', 'net_quantity', 'average_price'])
        legs = legs[legs['option_type'].isin(['C', 'P'])]
        if legs.empty:
            continue

        strikes = legs['strike'].dropna().astype(float)
        current_price = _option_direction_price_for_underlying(underlying)
        upper_candidates = [float(strikes.max()) * 2.0]
        if pd.notna(current_price) and current_price > 0:
            upper_candidates.append(float(current_price) * 2.0)
        upper_price = max(candidate for candidate in upper_candidates if np.isfinite(candidate))
        if not np.isfinite(upper_price) or upper_price <= 0:
            continue

        grid_values = [np.linspace(0.0, upper_price, 501), strikes.to_numpy(dtype=float)]
        if pd.notna(current_price) and current_price >= 0:
            grid_values.append(np.array([float(current_price)]))
        price_grid = np.unique(np.concatenate(grid_values))
        total_pl = _option_expiration_pl_at_prices(legs, price_grid)
        finite_mask = np.isfinite(price_grid) & np.isfinite(total_pl)
        if not finite_mask.any():
            continue

        price_grid = price_grid[finite_mask]
        total_pl = total_pl[finite_mask]
        max_pl = float(np.nanmax(total_pl))
        min_pl = float(np.nanmin(total_pl))
        pl_scale = max(abs(max_pl), abs(min_pl), 1.0)
        payoff_tolerance = pl_scale * OPTION_PAYOFF_FLAT_BAND
        best_prices = price_grid[total_pl >= max_pl - payoff_tolerance]
        best_price = float((best_prices.min() + best_prices.max()) / 2.0)
        low_pl = float(total_pl[0])
        high_pl = float(total_pl[-1])
        middle_price = float(strikes.median()) if not strikes.empty else upper_price / 2.0
        middle_pl = float(np.interp(middle_price, price_grid, total_pl))
        tail_width = upper_price * OPTION_PAYOFF_TAIL_BAND
        best_low = bool(best_prices.min() <= tail_width)
        best_high = bool(best_prices.max() >= upper_price - tail_width)
        payoff_structure, payoff_bias = _payoff_structure_label(
            low_pl=low_pl,
            middle_pl=middle_pl,
            high_pl=high_pl,
            best_low=best_low,
            best_high=best_high,
            max_pl=max_pl,
            min_pl=min_pl,
        )

        rows.append(
            {
                'ticker': underlying,
                'payoff_structure': payoff_structure,
                'payoff_bias': payoff_bias,
                'payoff_best_price': best_price,
                'payoff_best_pl': max_pl,
                'payoff_worst_pl': min_pl,
                'payoff_low_tail_pl': low_pl,
                'payoff_middle_pl': middle_pl,
                'payoff_high_tail_pl': high_pl,
            }
        )

    if not rows:
        return pd.DataFrame(columns=structure_columns).rename_axis('ticker')
    return pd.DataFrame(rows).set_index('ticker').rename_axis('ticker')


option_direction_account_value = _option_direction_current_account_value()
option_direction_beta_lookup = _option_direction_beta_lookup()

if 'option_greeks_frame' not in globals() or option_greeks_frame.empty:
    option_direction_summary = pd.DataFrame()
    print('Run Code Block 10 first so option Greeks and scenario exposures are available.')
else:
    option_direction_positions = option_greeks_frame.copy()
    if 'greek_status' not in option_direction_positions.columns:
        option_direction_positions['greek_status'] = 'ok'

    numeric_columns = [
        'net_quantity',
        'spot',
        'position_market_value',
        'position_delta_shares',
        'position_delta_notional',
        'position_gamma_delta_per_dollar',
        'position_theta_per_day',
        'position_vega_per_vol_point',
        'position_rho_per_rate_point',
    ]
    for column in numeric_columns:
        if column not in option_direction_positions.columns:
            option_direction_positions[column] = np.nan
        option_direction_positions[column] = pd.to_numeric(option_direction_positions[column], errors='coerce')

    option_direction_positions['option_legs'] = 1
    option_direction_positions['priced_legs'] = option_direction_positions['greek_status'].eq('ok').astype(int)
    option_direction_positions['unmatched_legs'] = option_direction_positions['greek_status'].ne('ok').astype(int)
    option_direction_positions['gross_contracts'] = option_direction_positions['net_quantity'].abs()

    option_direction_summary = (
        option_direction_positions.groupby('underlying', dropna=False)
        .agg(
            option_legs=('option_legs', 'sum'),
            priced_legs=('priced_legs', 'sum'),
            unmatched_legs=('unmatched_legs', 'sum'),
            gross_contracts=('gross_contracts', 'sum'),
            spot_price=('spot', _first_valid_numeric),
            position_market_value=('position_market_value', 'sum'),
            delta_shares=('position_delta_shares', 'sum'),
            dollar_delta=('position_delta_notional', 'sum'),
            gamma_delta_per_dollar=('position_gamma_delta_per_dollar', 'sum'),
            theta_per_day=('position_theta_per_day', 'sum'),
            vega_per_vol_point=('position_vega_per_vol_point', 'sum'),
            rho_per_rate_point=('position_rho_per_rate_point', 'sum'),
        )
        .rename_axis('ticker')
    )
    no_priced_legs = option_direction_summary['priced_legs'].eq(0)
    exposure_columns = [
        'position_market_value',
        'delta_shares',
        'dollar_delta',
        'gamma_delta_per_dollar',
        'theta_per_day',
        'vega_per_vol_point',
        'rho_per_rate_point',
    ]
    option_direction_summary.loc[no_priced_legs, exposure_columns] = np.nan
    option_direction_summary['delta_contract_equiv'] = option_direction_summary['delta_shares'] / 100.0
    option_direction_summary['beta_to_benchmark'] = [
        option_direction_beta_lookup.get(_option_direction_lookup_key(ticker), 1.0)
        for ticker in option_direction_summary.index
    ]
    option_direction_summary.loc[no_priced_legs, 'beta_to_benchmark'] = np.nan
    option_direction_summary['beta_adjusted_dollar_delta'] = (
        option_direction_summary['dollar_delta'] * option_direction_summary['beta_to_benchmark']
    )

    total_row = option_direction_summary.sum(numeric_only=True).to_frame().T
    total_row.index = pd.Index(['TOTAL'], name='ticker')
    total_row[['spot_price', 'beta_to_benchmark']] = np.nan
    option_direction_summary = pd.concat([total_row, option_direction_summary])

    finite_account_value = pd.notna(option_direction_account_value) and option_direction_account_value > 0
    if finite_account_value:
        option_direction_summary['ticker_delta_pct_account'] = (
            option_direction_summary['dollar_delta'] / option_direction_account_value
        )
        option_direction_summary['market_delta_pct_account'] = (
            option_direction_summary['beta_adjusted_dollar_delta'] / option_direction_account_value
        )
    else:
        option_direction_summary['ticker_delta_pct_account'] = np.nan
        option_direction_summary['market_delta_pct_account'] = np.nan
    no_summary_priced_legs = option_direction_summary['priced_legs'].eq(0)
    option_direction_summary.loc[
        no_summary_priced_legs,
        ['ticker_delta_pct_account', 'market_delta_pct_account'],
    ] = np.nan

    option_direction_scenario_pnl = _build_option_direction_scenario_pnl(option_greeks_frame)
    option_direction_summary = option_direction_summary.join(option_direction_scenario_pnl, how='left')
    option_payoff_structure_frame = _build_option_payoff_structure_frame(
        globals().get('positions_df', pd.DataFrame())
    )
    option_direction_summary = option_direction_summary.join(option_payoff_structure_frame, how='left')
    if 'TOTAL' in option_direction_summary.index:
        option_direction_summary.loc['TOTAL', 'payoff_structure'] = 'Portfolio Mix'
        option_direction_summary.loc['TOTAL', 'payoff_bias'] = 'mixed'
    option_direction_summary['current_ticker_delta_direction'] = option_direction_summary['ticker_delta_pct_account'].map(
        _option_direction_label
    )
    option_direction_summary['current_market_delta_direction'] = option_direction_summary['market_delta_pct_account'].map(
        _option_direction_label
    )
    option_direction_summary['ticker_direction'] = option_direction_summary['current_ticker_delta_direction']
    option_direction_summary['market_direction'] = option_direction_summary['current_market_delta_direction']
    payoff_structure_sort_order = {
        'Portfolio Mix': 0,
        'Bullish': 1,
        'Bearish': 2,
        'Neutral / Range': 3,
        'Flat / Neutral': 4,
        'Long Vol / Tails': 5,
        'Mixed / Complex': 6,
    }
    option_direction_summary['_payoff_structure_sort'] = (
        option_direction_summary['payoff_structure'].map(payoff_structure_sort_order).fillna(99)
    )
    option_direction_summary['_absolute_market_delta_sort'] = (
        option_direction_summary['market_delta_pct_account'].abs().fillna(-1.0)
    )
    option_direction_summary['_ticker_sort'] = option_direction_summary.index.astype(str)
    option_direction_summary = option_direction_summary.sort_values(
        ['_payoff_structure_sort', 'payoff_structure', '_absolute_market_delta_sort', '_ticker_sort'],
        ascending=[True, True, False, True],
    ).drop(columns=['_payoff_structure_sort', '_absolute_market_delta_sort', '_ticker_sort'])

    print(
        'payoff_structure classifies the expiration payoff shape; current delta labels use delta as a percent of account value. '
        f'Neutral band: +/-{OPTION_DIRECTION_NEUTRAL_BAND:.1%}; '
        f'strong band: +/-{OPTION_DIRECTION_STRONG_BAND:.1%}.'
    )
    if finite_account_value:
        print(f'Account value used for sizing: ${option_direction_account_value:,.2f}.')
    else:
        print('Account value was unavailable, so percent-of-account direction scores are blank.')
    if not option_direction_beta_lookup:
        print('Beta map was unavailable, so beta-adjusted market delta defaults to raw dollar delta.')

    option_direction_display_columns = [
        'payoff_structure',
        'payoff_bias',
        'current_ticker_delta_direction',
        'current_market_delta_direction',
        'spot_price',
        'delta_shares',
        'delta_contract_equiv',
        'dollar_delta',
        'ticker_delta_pct_account',
        'beta_to_benchmark',
        'beta_adjusted_dollar_delta',
        'market_delta_pct_account',
        'pnl_down_10pct',
        'pnl_down_5pct',
        'pnl_up_5pct',
        'pnl_up_10pct',
        'scenario_skew_5pct',
        'scenario_convexity_5pct',
        'payoff_best_price',
        'payoff_best_pl',
        'payoff_worst_pl',
    ]
    option_direction_display_columns = [
        column for column in option_direction_display_columns
        if column in option_direction_summary.columns
    ]
    option_direction_formats = {
        'spot_price': '${:,.2f}',
        'delta_shares': '{:,.0f}',
        'delta_contract_equiv': '{:,.2f}',
        'dollar_delta': '${:,.0f}',
        'ticker_delta_pct_account': '{:.1%}',
        'beta_to_benchmark': '{:.2f}',
        'beta_adjusted_dollar_delta': '${:,.0f}',
        'market_delta_pct_account': '{:.1%}',
        'pnl_down_10pct': '${:,.0f}',
        'pnl_down_5pct': '${:,.0f}',
        'pnl_up_5pct': '${:,.0f}',
        'pnl_up_10pct': '${:,.0f}',
        'scenario_skew_5pct': '${:,.0f}',
        'scenario_convexity_5pct': '${:,.0f}',
        'payoff_best_price': '${:,.2f}',
        'payoff_best_pl': '${:,.0f}',
        'payoff_worst_pl': '${:,.0f}',
    }

    def _format_option_direction_table_value(value, format_spec=None):
        if pd.isna(value):
            return ''
        if format_spec is None:
            return str(value)
        try:
            return format_spec.format(value)
        except (TypeError, ValueError):
            return str(value)


    option_direction_table = option_direction_summary[option_direction_display_columns].reset_index()
    option_direction_table_columns = option_direction_table.columns.tolist()
    option_direction_column_labels = {
        'ticker': 'Ticker',
        'payoff_structure': 'Payoff Structure',
        'payoff_bias': 'Payoff Bias',
        'current_ticker_delta_direction': 'Current Ticker Delta',
        'current_market_delta_direction': 'Current Market Delta',
        'spot_price': 'Spot',
        'delta_shares': 'Delta Shares',
        'delta_contract_equiv': 'Delta Contract Eq.',
        'dollar_delta': 'Dollar Delta',
        'ticker_delta_pct_account': 'Ticker Delta % Acct',
        'beta_to_benchmark': 'Beta',
        'beta_adjusted_dollar_delta': 'Beta Adj. Dollar Delta',
        'market_delta_pct_account': 'Market Delta % Acct',
        'pnl_down_10pct': 'P/L -10%',
        'pnl_down_5pct': 'P/L -5%',
        'pnl_up_5pct': 'P/L +5%',
        'pnl_up_10pct': 'P/L +10%',
        'scenario_skew_5pct': '+5/-5 Skew',
        'scenario_convexity_5pct': '+5/-5 Convexity',
        'payoff_best_price': 'Best Price',
        'payoff_best_pl': 'Best Exp. P/L',
        'payoff_worst_pl': 'Worst Exp. P/L',
    }
    option_direction_table_values = [
        [
            _format_option_direction_table_value(value, option_direction_formats.get(column))
            for value in option_direction_table[column]
        ]
        for column in option_direction_table_columns
    ]
    option_payoff_structure_colors = {
        'Portfolio Mix': '#475569',
        'Bullish': '#166534',
        'Bearish': '#991B1B',
        'Neutral / Range': '#1D4ED8',
        'Flat / Neutral': '#334155',
        'Long Vol / Tails': '#6D28D9',
        'Mixed / Complex': '#92400E',
    }
    option_direction_row_colors = [
        option_payoff_structure_colors.get(str(payoff_structure), '#111827')
        for payoff_structure in option_direction_table['payoff_structure']
    ]
    option_direction_column_widths = [
        76 if column == 'ticker' else 140 if column in {'payoff_structure', 'payoff_bias'} else 118
        for column in option_direction_table_columns
    ]

    import plotly.graph_objects as go

    option_direction_table_fig = go.Figure(
        data=[
            go.Table(
                columnwidth=option_direction_column_widths,
                header=dict(
                    values=[
                        f'<b>{option_direction_column_labels.get(column, column)}</b>'
                        for column in option_direction_table_columns
                    ],
                    fill_color='#334155',
                    font=dict(color='#F8FAFC', size=12),
                    align='left',
                    height=32,
                ),
                cells=dict(
                    values=option_direction_table_values,
                    fill_color=[option_direction_row_colors] * len(option_direction_table_columns),
                    font=dict(color='#E5E7EB', size=11),
                    align='left',
                    height=28,
                ),
            )
        ]
    )
    option_direction_table_fig.update_layout(
        title='Option Directional Exposure by Payoff Structure',
        template='plotly_dark',
        paper_bgcolor='#0F172A',
        margin=dict(t=55, b=20, l=10, r=10),
        height=max(460, 92 + 28 * len(option_direction_table)),
    )
    # option_direction_table_fig is rendered in the tabbed dashboard below.
# %% [portfolio block 12]
# Code Block 12: Option leg coverage audit
# =========================
# 10) Verify every raw Schwab option leg is represented in positions_df
# =========================
raw_option_rows = []
regex_miss_rows = []

for position in positions:
    instrument = position.get('instrument', {})

    if instrument.get('assetType') != 'OPTION':
        continue

    raw_symbol = instrument.get('symbol', '')
    normalized_symbol = ''.join((raw_symbol or '').split())
    match = option_pattern.match(normalized_symbol)
    expiration = pd.NaT
    parsed_option_type = pd.NA
    parsed_strike = pd.NA

    if match:
        expiration = pd.to_datetime('20' + match.group('expiration'), format='%Y%m%d', errors='coerce')
        parsed_option_type = match.group('option_type')
        parsed_strike = int(match.group('strike')) / 1000

    row = {
        'raw_symbol': raw_symbol,
        'normalized_symbol': normalized_symbol,
        'underlying_symbol': instrument.get('underlyingSymbol'),
        'put_call': instrument.get('putCall'),
        'expiration': expiration,
        'parsed_option_type': parsed_option_type,
        'parsed_strike': parsed_strike,
        'long_quantity': position.get('longQuantity', 0.0),
        'short_quantity': position.get('shortQuantity', 0.0),
        'average_price': position.get('averagePrice', 0.0),
    }
    raw_option_rows.append(row)

    if not match:
        regex_miss_rows.append(dict(row, drop_reason='symbol_did_not_match_option_pattern'))

raw_option_df = pd.DataFrame(raw_option_rows)
parsed_option_df = positions_df.copy()

if raw_option_df.empty:
    # print('[Code Block 12] No raw option legs returned by Schwab for the selected account.')
    pass
else:
    raw_audit = (
        raw_option_df.groupby('normalized_symbol', dropna=False, as_index=False)
        .agg(
            raw_rows=('normalized_symbol', 'size'),
            raw_long_quantity=('long_quantity', 'sum'),
            raw_short_quantity=('short_quantity', 'sum'),
            raw_average_price=('average_price', 'first'),
            underlying_symbol=('underlying_symbol', 'first'),
            expiration=('expiration', 'first'),
            put_call=('put_call', 'first'),
        )
    )
    if parsed_option_df.empty:
        parsed_audit = pd.DataFrame(columns=[
            'symbol',
            'parsed_rows',
            'parsed_long_quantity',
            'parsed_short_quantity',
            'parsed_average_price',
        ])
    else:
        parsed_option_df['symbol'] = parsed_option_df['symbol'].astype(str)
        parsed_audit = (
            parsed_option_df.groupby('symbol', as_index=False)
            .agg(
                parsed_rows=('symbol', 'size'),
                parsed_long_quantity=('long_quantity', 'sum'),
                parsed_short_quantity=('short_quantity', 'sum'),
                parsed_average_price=('average_price', 'first'),
            )
        )
    audit = raw_audit.merge(parsed_audit, left_on='normalized_symbol', right_on='symbol', how='left')
    audit[['parsed_rows', 'parsed_long_quantity', 'parsed_short_quantity']] = audit[
        ['parsed_rows', 'parsed_long_quantity', 'parsed_short_quantity']
    ].fillna(0)
    audit['raw_rows'] = audit['raw_rows'].fillna(0)
    audit['row_match'] = audit['raw_rows'].astype(int) == audit['parsed_rows'].astype(int)
    audit['long_match'] = audit['raw_long_quantity'].round(6) == audit['parsed_long_quantity'].round(6)
    audit['short_match'] = audit['raw_short_quantity'].round(6) == audit['parsed_short_quantity'].round(6)
    audit['present_in_positions_df'] = audit[['row_match', 'long_match', 'short_match']].all(axis=1)
    raw_symbol_count = raw_option_df['normalized_symbol'].nunique()
    parsed_symbol_count = parsed_option_df['symbol'].nunique() if not parsed_option_df.empty else 0
    # print('[Code Block 12] Option Leg Coverage Audit')
    # print(f'- Accounts returned by Schwab: {len(acct_map):,}')
    # print(f'- Selected account hash: {acct_hash}')
    # print(f'- Raw option legs from Schwab: {len(raw_option_df):,}')
    # print(f'- Parsed option legs in positions_df: {len(parsed_option_df):,}')
    # print(f'- Distinct option symbols from Schwab: {raw_symbol_count:,}')
    # print(f'- Distinct option symbols in positions_df: {parsed_symbol_count:,}')
    # print(f'- Regex misses: {len(regex_miss_rows):,}')

    if audit['present_in_positions_df'].all():
        # print('Status: PASS - every raw Schwab option leg is represented in positions_df.')
        pass
    else:
        # print('Status: FAIL - some option legs are missing or quantities do not match.')
        # display(
        #     audit.loc[
        #         ~audit['present_in_positions_df'],
        #         [
        #             'underlying_symbol',
        #             'normalized_symbol',
        #             'expiration',
        #             'put_call',
        #             'raw_rows',
        #             'parsed_rows',
        #             'raw_long_quantity',
        #             'parsed_long_quantity',
        #             'raw_short_quantity',
        #             'parsed_short_quantity',
        #         ],
        #     ].sort_values(['underlying_symbol', 'expiration', 'normalized_symbol'])
        # )
        pass
    if regex_miss_rows:
        # print('Fix suggestion: some option symbols failed the OCC regex, so keep those legs using instrument fields even when regex parsing fails.')
        # display(pd.DataFrame(regex_miss_rows).sort_values(['underlying_symbol', 'raw_symbol']))
        pass

    if len(acct_map) > 1:
        # print('Fix suggestion: your code currently pulls acct_map[0]. If your broker UI is showing another account, choose the matching hash or loop through every account.')
        pass

# %% [portfolio block 13]
# Code Block 13: Asset-level signed and unsigned analytics
# =========================
# 11) Asset-level signed and unsigned analytics
# =========================
rolling_windows = (21, 50, 200)
benchmark_label = 'SPY'

def _rolling_sharpe(values, window):
    rolling_mean = values.rolling(window).mean()
    rolling_std = values.rolling(window).std(ddof=0)
    sharpe = rolling_mean.div(rolling_std).mul(np.sqrt(252))
    return sharpe.mask(rolling_std.eq(0), 0.0).replace([np.inf, -np.inf], np.nan)

def _apply_zscore(values):
    if isinstance(values, pd.Series):
        return z_score(values)

    return values.apply(z_score)

def _latest_sorted_snapshot(frame):
    valid = frame.dropna(how='all')

    if valid.empty:
        return pd.Series(dtype=float)

    return valid.iloc[-1].sort_values(ascending=False)

def _log_summary(label, values):
    if isinstance(values, pd.DataFrame):
        valid = values.dropna(how='all')

        if valid.empty:
            return f'- {label}: empty DataFrame'

        return (
            f'- {label}: DataFrame {valid.shape[0]:,} x {valid.shape[1]} '
            f'({valid.index.min():%Y-%m-%d} to {valid.index.max():%Y-%m-%d})'
        )
    valid = values.dropna()

    if valid.empty:
        return f'- {label}: empty Series'

    return f'- {label}: Series {valid.shape[0]:,} rows ({valid.index.min():%Y-%m-%d} to {valid.index.max():%Y-%m-%d})'

# Daily return streams: unsigned raw returns and signed tradable returns
daily_returns = portfolio_closing_prices.pct_change().dropna(how='all')
benchmark_daily_returns = benchmark_close.pct_change().dropna()
if 'option_direction_sign_by_lookup_key' not in globals():
    raise RuntimeError('Run Code Block 11 before this block so option direction signs are available.')

signs = pd.Series(
    {
        ticker: option_direction_sign_by_lookup_key.get(_ticker_lookup_key(ticker), 1.0)
        for ticker in portfolio_closing_prices.columns
    },
    dtype=float,
)
daily_returns_signed = daily_returns.mul(signs, axis=1)
benchmark_daily_returns_aligned = benchmark_daily_returns.reindex(daily_returns.index).dropna()
daily_returns_for_corr = daily_returns.reindex(benchmark_daily_returns_aligned.index)
daily_returns_signed_for_corr = daily_returns_signed.reindex(benchmark_daily_returns_aligned.index)
portfolio_daily_returns_for_corr = daily_returns_signed.mean(axis=1).reindex(benchmark_daily_returns_aligned.index)
# print(f'[Code Block 13] Calculated daily return streams for {len(portfolio_closing_prices.columns)} assets.')

# Assets / Rolling horizon returns (unsigned + signed)
asset_return_windows = {
    window: portfolio_closing_prices.pct_change(window).dropna(how='all')
    for window in rolling_windows
}
asset_signed_return_windows = {
    window: frame.mul(signs, axis=1)
    for window, frame in asset_return_windows.items()
}
asset_return_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_return_windows.items()
}
asset_signed_return_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_signed_return_windows.items()
}
portfolio_assets_rolling_returns_21 = asset_return_windows[21]
portfolio_assets_rolling_returns_50 = asset_return_windows[50]
portfolio_assets_rolling_returns_200 = asset_return_windows[200]
portfolio_assets_rolling_signed_returns_21 = asset_signed_return_windows[21]
portfolio_assets_rolling_signed_returns_50 = asset_signed_return_windows[50]
portfolio_assets_rolling_signed_returns_200 = asset_signed_return_windows[200]
portfolio_assets_rolling_returns_z_scores_21 = asset_return_z_windows[21]
portfolio_assets_rolling_returns_z_scores_50 = asset_return_z_windows[50]
portfolio_assets_rolling_returns_z_scores_200 = asset_return_z_windows[200]
portfolio_assets_rolling_signed_return_z_scores_21 = asset_signed_return_z_windows[21]
portfolio_assets_rolling_signed_return_z_scores_50 = asset_signed_return_z_windows[50]
portfolio_assets_rolling_signed_return_z_scores_200 = asset_signed_return_z_windows[200]
# print('[Code Block 13] Calculated asset rolling returns and signed return z-scores for 21/50/200-day windows.')

# Assets / Rolling Sharpe (unsigned + signed)
asset_sharpe_windows = {
    window: _rolling_sharpe(daily_returns, window)
    for window in rolling_windows
}
asset_signed_sharpe_windows = {
    window: _rolling_sharpe(daily_returns_signed, window)
    for window in rolling_windows
}
asset_sharpe_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_sharpe_windows.items()
}
asset_signed_sharpe_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_signed_sharpe_windows.items()
}
portfolio_assets_rolling_sharpe_21 = asset_sharpe_windows[21]
portfolio_assets_rolling_sharpe_50 = asset_sharpe_windows[50]
portfolio_assets_rolling_sharpe_200 = asset_sharpe_windows[200]
portfolio_assets_rolling_signed_sharpe_21 = asset_signed_sharpe_windows[21]
portfolio_assets_rolling_signed_sharpe_50 = asset_signed_sharpe_windows[50]
portfolio_assets_rolling_signed_sharpe_200 = asset_signed_sharpe_windows[200]
portfolio_assets_rolling_sharpe_z_scores_21 = asset_sharpe_z_windows[21]
portfolio_assets_rolling_sharpe_z_scores_50 = asset_sharpe_z_windows[50]
portfolio_assets_rolling_sharpe_z_scores_200 = asset_sharpe_z_windows[200]
portfolio_assets_rolling_signed_sharpe_z_scores_21 = asset_signed_sharpe_z_windows[21]
portfolio_assets_rolling_signed_sharpe_z_scores_50 = asset_signed_sharpe_z_windows[50]
portfolio_assets_rolling_signed_sharpe_z_scores_200 = asset_signed_sharpe_z_windows[200]
# print('[Code Block 13] Calculated vectorized asset rolling Sharpe series for 21/50/200-day windows.')

# Assets / Rolling correlation to benchmark (unsigned + signed)
asset_corr_windows = {
    window: daily_returns_for_corr.rolling(window).corr(benchmark_daily_returns_aligned)
    for window in rolling_windows
}
asset_signed_corr_windows = {
    window: daily_returns_signed_for_corr.rolling(window).corr(benchmark_daily_returns_aligned)
    for window in rolling_windows
}
asset_corr_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_corr_windows.items()
}
asset_signed_corr_z_windows = {
    window: _apply_zscore(frame)
    for window, frame in asset_signed_corr_windows.items()
}
portfolio_assets_rolling_correlation_21 = asset_corr_windows[21]
portfolio_assets_rolling_correlation_50 = asset_corr_windows[50]
portfolio_assets_rolling_correlation_200 = asset_corr_windows[200]
portfolio_assets_rolling_signed_correlation_21 = asset_signed_corr_windows[21]
portfolio_assets_rolling_signed_correlation_50 = asset_signed_corr_windows[50]
portfolio_assets_rolling_signed_correlation_200 = asset_signed_corr_windows[200]
portfolio_assets_rolling_correlation_z_scores_21 = asset_corr_z_windows[21]
portfolio_assets_rolling_correlation_z_scores_50 = asset_corr_z_windows[50]
portfolio_assets_rolling_correlation_z_scores_200 = asset_corr_z_windows[200]
portfolio_assets_rolling_signed_correlation_z_scores_21 = asset_signed_corr_z_windows[21]
portfolio_assets_rolling_signed_correlation_z_scores_50 = asset_signed_corr_z_windows[50]
portfolio_assets_rolling_signed_correlation_z_scores_200 = asset_signed_corr_z_windows[200]
# print(f'[Code Block 13] Calculated asset rolling correlations to {benchmark_label} for 21/50/200-day windows.')

# Assets / Latest z-scores by metric
latest_assets_return_z_scores_21 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_return_z_scores_21)
latest_assets_return_z_scores_50 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_return_z_scores_50)
latest_assets_return_z_scores_200 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_return_z_scores_200)
latest_assets_sharpe_z_scores_21 = _latest_sorted_snapshot(portfolio_assets_rolling_sharpe_z_scores_21)
latest_assets_sharpe_z_scores_50 = _latest_sorted_snapshot(portfolio_assets_rolling_sharpe_z_scores_50)
latest_assets_sharpe_z_scores_200 = _latest_sorted_snapshot(portfolio_assets_rolling_sharpe_z_scores_200)
latest_assets_signed_sharpe_z_scores_21 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_sharpe_z_scores_21)
latest_assets_signed_sharpe_z_scores_50 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_sharpe_z_scores_50)
latest_assets_signed_sharpe_z_scores_200 = _latest_sorted_snapshot(portfolio_assets_rolling_signed_sharpe_z_scores_200)

# Display labels: add parentheses only for signed series with negative direction
def format_ticker(ticker, sign):
    return f'({ticker})' if sign < 0 else ticker

latest_assets_return_z_scores_21.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_return_z_scores_21.index]
latest_assets_return_z_scores_50.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_return_z_scores_50.index]
latest_assets_return_z_scores_200.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_return_z_scores_200.index]
latest_assets_sharpe_z_scores_21.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_sharpe_z_scores_21.index]
latest_assets_sharpe_z_scores_50.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_sharpe_z_scores_50.index]
latest_assets_sharpe_z_scores_200.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_sharpe_z_scores_200.index]
latest_assets_signed_sharpe_z_scores_21.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_signed_sharpe_z_scores_21.index]
latest_assets_signed_sharpe_z_scores_50.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_signed_sharpe_z_scores_50.index]
latest_assets_signed_sharpe_z_scores_200.index = [format_ticker(t, signs.loc[t]) for t in latest_assets_signed_sharpe_z_scores_200.index]

# Portfolio aggregates (signed equal-weight daily rebalance)
portfolio_daily_returns = daily_returns_signed.mean(axis=1)
portfolio_equity = (1 + portfolio_daily_returns).cumprod()
portfolio_return_21 = portfolio_equity.pct_change(21).dropna()
portfolio_return_50 = portfolio_equity.pct_change(50).dropna()
portfolio_return_200 = portfolio_equity.pct_change(200).dropna()
portfolio_rolling_sharpe_21 = _rolling_sharpe(portfolio_daily_returns, 21).dropna()
portfolio_rolling_sharpe_50 = _rolling_sharpe(portfolio_daily_returns, 50).dropna()
portfolio_rolling_sharpe_200 = _rolling_sharpe(portfolio_daily_returns, 200).dropna()

# Benchmark analytics (unsigned benchmark series)
benchmark_equity = (1 + benchmark_daily_returns).cumprod()
benchmark_rolling_sharpe_21 = _rolling_sharpe(benchmark_daily_returns, 21).dropna()
benchmark_rolling_sharpe_50 = _rolling_sharpe(benchmark_daily_returns, 50).dropna()
benchmark_rolling_sharpe_200 = _rolling_sharpe(benchmark_daily_returns, 200).dropna()
portfolio_rolling_signed_correlation_21 = portfolio_daily_returns_for_corr.rolling(21).corr(benchmark_daily_returns_aligned).dropna()
portfolio_rolling_signed_correlation_50 = portfolio_daily_returns_for_corr.rolling(50).corr(benchmark_daily_returns_aligned).dropna()
portfolio_rolling_signed_correlation_200 = portfolio_daily_returns_for_corr.rolling(200).corr(benchmark_daily_returns_aligned).dropna()
# print(f'[Code Block 13] Calculated portfolio and benchmark aggregate analytics versus {benchmark_label}.')
calculation_log = [
    '[Code Block 13] Summary:',
    f'- Assets analyzed: {len(portfolio_closing_prices.columns)}',
    f'- Rolling windows: {", ".join(str(window) for window in rolling_windows)} trading days',
    _log_summary('Asset daily returns', daily_returns),
    _log_summary('Signed asset daily returns', daily_returns_signed),
    _log_summary('Signed asset rolling returns (200d)', portfolio_assets_rolling_signed_returns_200),
    _log_summary('Asset rolling Sharpe (200d)', portfolio_assets_rolling_sharpe_200),
    _log_summary(f'Asset rolling correlation to {benchmark_label} (200d)', portfolio_assets_rolling_signed_correlation_200),
    _log_summary('Portfolio equity', portfolio_equity),
    _log_summary('Portfolio rolling Sharpe (200d)', portfolio_rolling_sharpe_200),
    _log_summary(f'Portfolio rolling correlation to {benchmark_label} (200d)', portfolio_rolling_signed_correlation_200),
]
# print("\n".join(calculation_log))

# %% [portfolio block 14]
# Code Block 14: Plots
# =========================
# 12) Plots
# =========================
from Quantapp.visualization.views.portfolio_profile.performance_structure import (
    plot_equity_curve,
    plot_rolling_correlation,
    plot_rolling_sharpe_zscore,
)

portfolio_equity_fig = plot_equity_curve(
    portfolio_equity,
    benchmark_equity,
    benchmark_label=benchmark_str,
)
# portfolio_equity_fig is rendered in the tabbed dashboard below.
rolling_sharpe_fig = plot_rolling_sharpe_zscore(
    {
        21: portfolio_rolling_sharpe_21,
        50: portfolio_rolling_sharpe_50,
        200: portfolio_rolling_sharpe_200,
    },
    {
        21: benchmark_rolling_sharpe_21,
        50: benchmark_rolling_sharpe_50,
        200: benchmark_rolling_sharpe_200,
    },
    benchmark_label=benchmark_str,
    default_window=200,
)
# rolling_sharpe_fig is rendered in the tabbed dashboard below.
rolling_correlation_fig = plot_rolling_correlation(
    {
        21: portfolio_rolling_signed_correlation_21,
        50: portfolio_rolling_signed_correlation_50,
        200: portfolio_rolling_signed_correlation_200,
    },
    benchmark_label=benchmark_str,
)
# rolling_correlation_fig is rendered in the tabbed dashboard below.
# %% [portfolio block 15]
# Code Block 15: Relative price strength z-score plots
# Relative Price Strength Z-Score Plots
from Quantapp.analytics.series_utils import calculate_zscore
from Quantapp.visualization.views.portfolio_profile.performance_structure import (
    format_snapshot_map,
    plot_benchmark_snapshot_zscores,
)

def risk_adjusted_returns(data, windows, ratio_type='sharpe', risk_free_rate=0.0, annualization_factor=252):
    if isinstance(windows, (str, bytes)):
        raise ValueError("windows must be an integer or an iterable of integers")
    try:
        window_list = [int(window) for window in windows]
    except TypeError:
        window_list = [int(windows)]
    if not window_list or any(window <= 0 for window in window_list):
        raise ValueError("windows must contain positive integers")

    price_frame = data.to_frame(name=data.name or "price") if isinstance(data, pd.Series) else data
    if not isinstance(price_frame, pd.DataFrame):
        raise TypeError("data must be a pandas Series or DataFrame")

    returns = price_frame.pct_change()
    if isinstance(risk_free_rate, pd.Series):
        periodic_rate = risk_free_rate.astype(float).sort_index().reindex(returns.index).ffill()
    elif np.isscalar(risk_free_rate):
        periodic_rate = pd.Series((1.0 + float(risk_free_rate)) ** (1.0 / annualization_factor) - 1.0, index=returns.index)
    else:
        raise TypeError("risk_free_rate must be a scalar annual rate or a pandas Series")

    excess_returns = returns.sub(periodic_rate, axis=0)
    single_window = len(window_list) == 1
    single_series = price_frame.shape[1] == 1
    output = []
    for column in returns.columns:
        excess = excess_returns[column]
        for window in window_list:
            mean_excess = excess.rolling(window).mean()
            if ratio_type == 'sharpe':
                volatility = excess.rolling(window).std()
                ratio = np.sqrt(annualization_factor) * mean_excess / volatility
                ratio = ratio.where(volatility > 0)
            elif ratio_type == 'sortino':
                downside = excess.where(excess < 0, 0.0)
                downside_deviation = downside.rolling(window).apply(lambda values: np.sqrt((values**2).mean()), raw=True)
                ratio = np.sqrt(annualization_factor) * mean_excess / downside_deviation
            else:
                raise ValueError("Invalid ratio_type. Use 'sharpe' or 'sortino'.")

            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio.name = f"{ratio_type}_ratio_{window}" if single_window and single_series else f"{column}_{ratio_type}_{window}"
            output.append(ratio)
    return pd.concat(output, axis=1)

time_frame_map = {
    '21': time_frame_short,
    '50': time_frame_mid,
    '200': time_frame_long,
}
if '_ticker_lookup_key' not in globals():
    def _ticker_lookup_key(ticker):
        return ''.join(character for character in str(ticker).upper() if character.isalnum())

if 'option_direction_sign_by_lookup_key' not in globals():
    raise RuntimeError('Run Code Block 11 before this block so option direction signs are available.')

sign_series = pd.Series(
    {
        ticker: option_direction_sign_by_lookup_key.get(_ticker_lookup_key(ticker), 1.0)
        for ticker in portfolio_closing_prices.columns
    },
    dtype=float,
)

asset_frame = portfolio_closing_prices.dropna(how='all').sort_index()
benchmark_series = benchmark_close.dropna().sort_index()
common_index = asset_frame.index.intersection(benchmark_series.index)
asset_frame = asset_frame.reindex(common_index).dropna(how='all')
benchmark_series = benchmark_series.reindex(common_index).dropna()
signed_returns = asset_frame.pct_change().mul(sign_series, axis=1)

def latest_zscore_snapshot(metric_frame):
    if metric_frame.empty:
        return pd.Series(dtype=float)
    zscore_frame = metric_frame.apply(calculate_zscore)
    if zscore_frame.empty:
        return pd.Series(dtype=float)
    return zscore_frame.iloc[-1].dropna().sort_values(ascending=False)

def latest_spread_zscore_snapshot(asset_metric_frame, benchmark_metric):
    if asset_metric_frame.empty:
        return pd.Series(dtype=float)
    benchmark_aligned = benchmark_metric.reindex(asset_metric_frame.index)
    spread_frame = asset_metric_frame.apply(lambda column: benchmark_aligned - column, axis=0)
    return latest_zscore_snapshot(spread_frame)

def rolling_sharpe_from_returns(returns, window, annualization_factor=252):
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    ratio = np.sqrt(annualization_factor) * rolling_mean / rolling_std
    return ratio.where(rolling_std > 0).replace([np.inf, -np.inf], np.nan)

benchmark_snapshot = {
    'unsigned_asset_latest_zscores': {},
    'signed_asset_latest_zscores': {},
    'unsigned_spread_latest_zscores': {},
    'signed_spread_latest_zscores': {},
}

for term, window in time_frame_map.items():
    window = int(window)
    unsigned_sharpe = risk_adjusted_returns(
        asset_frame,
        windows=[window],
        ratio_type='sharpe',
    )
    if unsigned_sharpe.shape[1] == asset_frame.shape[1]:
        unsigned_sharpe.columns = asset_frame.columns

    benchmark_sharpe = risk_adjusted_returns(
        benchmark_series,
        windows=[window],
        ratio_type='sharpe',
    ).iloc[:, 0]
    signed_sharpe = rolling_sharpe_from_returns(signed_returns, window=window)

    benchmark_snapshot['unsigned_asset_latest_zscores'][term] = latest_zscore_snapshot(unsigned_sharpe)
    benchmark_snapshot['signed_asset_latest_zscores'][term] = latest_zscore_snapshot(signed_sharpe)
    benchmark_snapshot['unsigned_spread_latest_zscores'][term] = latest_spread_zscore_snapshot(
        unsigned_sharpe,
        benchmark_sharpe,
    )
    benchmark_snapshot['signed_spread_latest_zscores'][term] = latest_spread_zscore_snapshot(
        signed_sharpe,
        benchmark_sharpe,
    )
windows_signed = format_snapshot_map(
    benchmark_snapshot['signed_asset_latest_zscores'],
    sign_series,
)
windows_unsigned = benchmark_snapshot['unsigned_asset_latest_zscores']
windows_benchmark_minus_assets_signed = format_snapshot_map(
    benchmark_snapshot['signed_spread_latest_zscores'],
    sign_series,
)
windows_benchmark_minus_assets_unsigned = benchmark_snapshot['unsigned_spread_latest_zscores']

def set_dropdown_default(fig, target_label):
    if not fig.layout.updatemenus:
        return fig

    buttons = list(fig.layout.updatemenus[0].buttons)
    button_labels = [button.label for button in buttons]
    if target_label not in button_labels:
        return fig

    active_index = button_labels.index(target_label)
    active_button = buttons[active_index]
    visible = active_button.args[0].get('visible') if active_button.args else None
    if visible is not None:
        for trace, is_visible in zip(fig.data, visible):
            trace.visible = is_visible

    if len(active_button.args) > 1:
        title = active_button.args[1].get('title')
        if title:
            fig.update_layout(title=title)

    fig.layout.updatemenus[0].active = active_index
    return fig

snapshot_fig = plot_benchmark_snapshot_zscores(
    windows_signed=windows_signed,
    windows_unsigned=windows_unsigned,
    windows_benchmark_minus_assets_signed=windows_benchmark_minus_assets_signed,
    windows_benchmark_minus_assets_unsigned=windows_benchmark_minus_assets_unsigned,
    sign_series=sign_series,
    benchmark_label=benchmark_str,
)
snapshot_fig = set_dropdown_default(snapshot_fig, '200-Day')
# snapshot_fig is rendered in the tabbed dashboard below.
# %% [portfolio block 16]
# Code Block 16: Portfolio asset relationship clustering
# Compute raw-return correlations, distances, and hierarchical clustering inputs for Block 17.
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

relationship_returns = (
    portfolio_closing_prices
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .dropna(how='all')
)
relationship_returns = relationship_returns.loc[:, relationship_returns.notna().sum() >= 2]
relationship_returns = relationship_returns.loc[:, relationship_returns.std(skipna=True) > 0]

if relationship_returns.shape[1] < 2:
    raise ValueError('At least two assets with valid return history are required for clustering.')

asset_correlation = relationship_returns.corr().clip(-1.0, 1.0)
asset_correlation = asset_correlation.dropna(how='all').dropna(axis=1, how='all')
asset_correlation = asset_correlation.fillna(0.0)
np.fill_diagonal(asset_correlation.values, 1.0)

asset_distance = (2.0 * (1.0 - asset_correlation)).clip(lower=0.0)
np.fill_diagonal(asset_distance.values, 0.0)
condensed_asset_distance = squareform(asset_distance.values, checks=False)
asset_linkage = linkage(condensed_asset_distance, method='average')
asset_order = asset_correlation.index[leaves_list(asset_linkage)].tolist()
clustered_asset_correlation = asset_correlation.loc[asset_order, asset_order]
clustered_asset_distance = asset_distance.loc[asset_order, asset_order]

dendro = dendrogram(
    asset_linkage,
    labels=asset_correlation.index.tolist(),
    no_plot=True,
)

# %% [portfolio block 17]
# Code Block 17: Benchmark-residual asset relationship clustering
# Regress each asset's daily returns against the benchmark, then cluster residual correlations.
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from plotly.subplots import make_subplots
import plotly.graph_objects as go

minimum_residual_observations = 30
residual_benchmark_label = benchmark_label if 'benchmark_label' in globals() else benchmark_str

benchmark_residual_asset_returns = (
    portfolio_closing_prices
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .dropna(how='all')
)
benchmark_residual_benchmark_returns = (
    benchmark_close
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
benchmark_residual_benchmark_returns.name = residual_benchmark_label

common_residual_index = benchmark_residual_asset_returns.index.intersection(benchmark_residual_benchmark_returns.index)
benchmark_residual_asset_returns = benchmark_residual_asset_returns.reindex(common_residual_index)
benchmark_residual_benchmark_returns = benchmark_residual_benchmark_returns.reindex(common_residual_index)

def _benchmark_residualize(asset_returns, benchmark_returns, min_observations):
    residual_series_by_asset = {}
    regression_rows = []

    if benchmark_returns.std(skipna=True) == 0:
        raise ValueError('Benchmark returns must have non-zero variance for residual regression.')

    for ticker, asset_return_series in asset_returns.items():
        aligned_returns = pd.concat(
            [asset_return_series, benchmark_returns],
            axis=1,
            keys=['asset', 'benchmark'],
        ).dropna()

        if aligned_returns.shape[0] < min_observations:
            continue
        if aligned_returns['asset'].std(skipna=True) == 0 or aligned_returns['benchmark'].std(skipna=True) == 0:
            continue

        benchmark_values = aligned_returns['benchmark'].to_numpy(dtype=float)
        asset_values = aligned_returns['asset'].to_numpy(dtype=float)
        regression_matrix = np.column_stack([np.ones(len(aligned_returns)), benchmark_values])
        alpha, beta = np.linalg.lstsq(regression_matrix, asset_values, rcond=None)[0]
        fitted_values = alpha + beta * benchmark_values
        residual_series = pd.Series(
            asset_values - fitted_values,
            index=aligned_returns.index,
            name=ticker,
        )

        if residual_series.std(skipna=True) == 0:
            continue

        residual_series_by_asset[ticker] = residual_series
        regression_rows.append(
            {
                'Ticker': ticker,
                'Alpha': alpha,
                'Beta': beta,
                'Observations': aligned_returns.shape[0],
                'Residual Volatility': residual_series.std(ddof=0),
            }
        )

    residual_frame = pd.DataFrame(residual_series_by_asset).dropna(how='all')
    regression_stats = pd.DataFrame(regression_rows)
    if not regression_stats.empty:
        regression_stats = regression_stats.set_index('Ticker').sort_index()

    return residual_frame, regression_stats

benchmark_residual_returns, benchmark_residual_regression_stats = _benchmark_residualize(
    benchmark_residual_asset_returns,
    benchmark_residual_benchmark_returns,
    minimum_residual_observations,
)
benchmark_residual_returns = benchmark_residual_returns.loc[
    :,
    benchmark_residual_returns.notna().sum() >= minimum_residual_observations,
]

if benchmark_residual_returns.shape[1] < 2:
    raise ValueError(
        'At least two assets with valid benchmark-residual return history are required for clustering.'
    )

benchmark_residual_correlation = benchmark_residual_returns.corr(
    min_periods=minimum_residual_observations
).clip(-1.0, 1.0)
benchmark_residual_correlation = benchmark_residual_correlation.dropna(how='all').dropna(axis=1, how='all')
valid_residual_assets = benchmark_residual_correlation.index.intersection(benchmark_residual_correlation.columns)
benchmark_residual_correlation = benchmark_residual_correlation.loc[valid_residual_assets, valid_residual_assets]

if benchmark_residual_correlation.shape[1] < 2:
    raise ValueError(
        'At least two assets with overlapping benchmark-residual return history are required for clustering.'
    )

benchmark_residual_correlation = benchmark_residual_correlation.fillna(0.0)
np.fill_diagonal(benchmark_residual_correlation.values, 1.0)

benchmark_residual_distance = (2.0 * (1.0 - benchmark_residual_correlation)).clip(lower=0.0)
np.fill_diagonal(benchmark_residual_distance.values, 0.0)
condensed_residual_distance = squareform(benchmark_residual_distance.values, checks=False)
benchmark_residual_linkage = linkage(condensed_residual_distance, method='average')
benchmark_residual_order = benchmark_residual_correlation.index[leaves_list(benchmark_residual_linkage)].tolist()
clustered_benchmark_residual_correlation = benchmark_residual_correlation.loc[
    benchmark_residual_order,
    benchmark_residual_order,
]
clustered_benchmark_residual_distance = benchmark_residual_distance.loc[
    benchmark_residual_order,
    benchmark_residual_order,
]

benchmark_residual_dendro = dendrogram(
    benchmark_residual_linkage,
    labels=benchmark_residual_correlation.index.tolist(),
    no_plot=True,
)

required_raw_clustering_vars = [
    'dendro',
    'clustered_asset_correlation',
    'clustered_asset_distance',
]
missing_raw_clustering_vars = [name for name in required_raw_clustering_vars if name not in globals()]
if missing_raw_clustering_vars:
    raise RuntimeError('Run Code Block 16 before Code Block 17 so raw-return clustering inputs are available.')

relationship_comparison_fig = make_subplots(
    rows=2,
    cols=2,
    row_heights=[0.32, 0.68],
    column_widths=[0.5, 0.5],
    vertical_spacing=0.08,
    horizontal_spacing=0.08,
    subplot_titles=(
        'Raw Return Dendrogram',
        f'Benchmark-Residual Dendrogram ({residual_benchmark_label})',
        'Raw Return Correlation Heatmap',
        'Benchmark-Residual Correlation Heatmap',
    ),
)

for icoord, dcoord in zip(dendro['icoord'], dendro['dcoord']):
    relationship_comparison_fig.add_trace(
        go.Scatter(
            x=icoord,
            y=dcoord,
            mode='lines',
            line=dict(color='#38BDF8', width=1.5),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1,
        col=1,
    )

for icoord, dcoord in zip(benchmark_residual_dendro['icoord'], benchmark_residual_dendro['dcoord']):
    relationship_comparison_fig.add_trace(
        go.Scatter(
            x=icoord,
            y=dcoord,
            mode='lines',
            line=dict(color='#A78BFA', width=1.5),
            hoverinfo='skip',
            showlegend=False,
        ),
        row=1,
        col=2,
    )

raw_leaf_x = [5 + 10 * idx for idx in range(len(dendro['ivl']))]
residual_leaf_x = [5 + 10 * idx for idx in range(len(benchmark_residual_dendro['ivl']))]
relationship_comparison_fig.update_xaxes(
    tickmode='array',
    tickvals=raw_leaf_x,
    ticktext=dendro['ivl'],
    tickangle=-35,
    row=1,
    col=1,
)
relationship_comparison_fig.update_xaxes(
    tickmode='array',
    tickvals=residual_leaf_x,
    ticktext=benchmark_residual_dendro['ivl'],
    tickangle=-35,
    row=1,
    col=2,
)
relationship_comparison_fig.update_yaxes(title_text='Distance', row=1, col=1)
relationship_comparison_fig.update_yaxes(title_text='Distance', row=1, col=2)

relationship_comparison_fig.add_trace(
    go.Heatmap(
        z=clustered_asset_correlation.values,
        x=clustered_asset_correlation.columns,
        y=clustered_asset_correlation.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title='Raw Correlation', x=0.46, y=0.31, len=0.56),
        customdata=clustered_asset_distance.values,
        hovertemplate=(
            'Asset X: %{x}<br>'
            'Asset Y: %{y}<br>'
            'Raw Correlation: %{z:.3f}<br>'
            'Distance: %{customdata:.3f}<extra></extra>'
        ),
    ),
    row=2,
    col=1,
)
relationship_comparison_fig.add_trace(
    go.Heatmap(
        z=clustered_benchmark_residual_correlation.values,
        x=clustered_benchmark_residual_correlation.columns,
        y=clustered_benchmark_residual_correlation.index,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title='Residual Correlation', x=1.02, y=0.31, len=0.56),
        customdata=clustered_benchmark_residual_distance.values,
        hovertemplate=(
            'Asset X: %{x}<br>'
            'Asset Y: %{y}<br>'
            'Residual Correlation: %{z:.3f}<br>'
            'Distance: %{customdata:.3f}<extra></extra>'
        ),
    ),
    row=2,
    col=2,
)
relationship_comparison_fig.update_xaxes(title_text='Assets', tickangle=-35, row=2, col=1)
relationship_comparison_fig.update_xaxes(title_text='Assets', tickangle=-35, row=2, col=2)
relationship_comparison_fig.update_yaxes(title_text='Assets', autorange='reversed', row=2, col=1)
relationship_comparison_fig.update_yaxes(title_text='Assets', autorange='reversed', row=2, col=2)

relationship_comparison_fig.update_layout(
    title=f'Portfolio Asset Relationship Clustering: Raw Returns vs {residual_benchmark_label} Residuals',
    template='plotly_dark',
    height=1050,
    margin=dict(l=70, r=70, t=100, b=120),
)
# relationship_comparison_fig is rendered in the tabbed dashboard below.

# %% [dashboard]
# Unified tabbed portfolio dashboard
from dash import Dash, dash_table, dcc, html
from datetime import date, datetime
import socket


def _dashboard_graph(figure, graph_id, height=760):
    if figure is None:
        return html.Div("No data is available for this view.", style={"padding": "28px", "color": "#94A3B8"})
    return dcc.Graph(
        id=graph_id,
        figure=figure,
        config={"displaylogo": False, "responsive": True},
        style={"height": f"{height}px"},
    )


def _dashboard_table(frame, table_id, page_size=15):
    if frame is None:
        frame = pd.DataFrame()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.copy().reset_index()
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: value.isoformat()
            if isinstance(value, (pd.Timestamp, pd.Timedelta, date, datetime))
            else value.item()
            if isinstance(value, np.generic)
            else str(value)
            if isinstance(value, Path)
            else value
        )
    return dash_table.DataTable(
        id=table_id,
        data=frame.to_dict("records"),
        columns=[{"name": column, "id": column} for column in frame.columns],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#1E293B", "color": "#F8FAFC", "fontWeight": "700", "border": "1px solid #334155"},
        style_cell={
            "backgroundColor": "#0F172A", "color": "#E2E8F0", "border": "1px solid #334155",
            "padding": "8px", "fontFamily": "Arial, sans-serif", "fontSize": "12px",
            "textAlign": "left", "minWidth": "110px", "maxWidth": "280px", "whiteSpace": "normal",
        },
    )


def _dashboard_section(title, children):
    return html.Div([html.H3(title, style={"margin": "8px 0 14px"}), *children], style={"padding": "18px 8px"})


if available_option_underlyings:
    # Use the original option P/L Dash app as the parent application. This keeps
    # its complete layout and every registered callback intact inside our tab.
    portfolio_dashboard_app = build_option_expiration_pl_dash_app(
        positions_df,
        net_cost_basis=net_cost_basis,
        portfolio_latest_prices=portfolio_latest_prices,
        benchmark_current_price=benchmark_current_price,
        underlying_beta_map=underlying_beta_map,
        benchmark_label=benchmark_label_for_beta,
        payoff_structure_frame=option_payoff_structure_frame,
        app_name=__name__,
        component_prefix="portfolio-option-expiration-pl",
    )
    option_expiration_pl_panel = portfolio_dashboard_app.layout
else:
    portfolio_dashboard_app = Dash(__name__)
    option_expiration_pl_panel = html.Div(
        "No option positions are available for the expiration P/L view.",
        style={"padding": "28px", "color": "#94A3B8"},
    )

portfolio_dashboard_app.title = "Portfolio Monitoring, Performance & Risk"
portfolio_dashboard_app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Portfolio Monitoring, Performance & Risk", style={"margin": "0", "fontSize": "28px"}),
                html.Div(f"Schwab portfolio analytics benchmarked against {benchmark_str}", style={"color": "#94A3B8", "marginTop": "6px"}),
            ],
            style={"padding": "22px 24px", "borderBottom": "1px solid #334155"},
        ),
        dcc.Tabs(
            id="portfolio-view-tabs",
            value="overview",
            colors={"border": "#334155", "primary": "#38BDF8", "background": "#0F172A"},
            children=[
                dcc.Tab(label="Overview", value="overview", children=[
                    _dashboard_section("Account value, cash & risk ratios", [
                        _dashboard_graph(schwab_portfolio_market_value_fig, "portfolio-market-value-graph", 980),
                        _dashboard_table(schwab_market_value_summary, "portfolio-market-value-summary", 10),
                    ])
                ]),
                dcc.Tab(label="Options", value="options", children=[
                    _dashboard_section("Expiration ladder", [
                        _dashboard_graph(options_expiration_ladder_fig, "portfolio-options-dte-graph", 760)
                    ]),
                    _dashboard_section("Greek sensitivities", [
                        _dashboard_graph(globals().get("option_greek_sensitivity_fig"), "portfolio-option-greeks-graph", 490),
                    ]),
                    _dashboard_section("P/L at expiration", [
                        option_expiration_pl_panel,
                        _dashboard_table(option_profit_loss_extremes_table, "portfolio-options-extremes-table"),
                    ]),
                    _dashboard_section("Directional exposure", [
                        _dashboard_graph(globals().get("option_direction_table_fig"), "portfolio-option-direction-graph", 760)
                    ]),
                ]),
                dcc.Tab(label="Performance & Risk", value="performance", children=[
                    _dashboard_section("Buy-and-hold performance", [
                        _dashboard_graph(portfolio_equity_fig, "portfolio-equity-graph", 760),
                        _dashboard_graph(rolling_sharpe_fig, "portfolio-sharpe-graph", 760),
                        _dashboard_graph(rolling_correlation_fig, "portfolio-correlation-graph", 760),
                    ])
                ]),
                dcc.Tab(label="Relative Strength", value="relative-strength", children=[
                    _dashboard_section("Latest rolling risk-adjusted z-scores", [
                        _dashboard_graph(snapshot_fig, "portfolio-relative-strength-graph", 900)
                    ])
                ]),
                dcc.Tab(label="Relationships", value="relationships", children=[
                    _dashboard_section("Raw and benchmark-residual clustering", [
                        _dashboard_graph(relationship_comparison_fig, "portfolio-relationships-graph", 1120),
                        _dashboard_table(benchmark_residual_regression_stats, "portfolio-residual-regression-table"),
                    ])
                ]),
            ],
        ),
    ],
    style={"backgroundColor": "#020617", "color": "#E2E8F0", "minHeight": "100vh", "fontFamily": "Arial, sans-serif", "padding": "0 14px 28px"},
)


def _portfolio_dashboard_available_port(host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind((host, 0))
        return int(port_socket.getsockname()[1])


portfolio_dashboard_app.run(
    host="127.0.0.1",
    port=_portfolio_dashboard_available_port(),
    debug=False,
    use_reloader=False,
    jupyter_mode="inline",
    jupyter_height=3200,
)

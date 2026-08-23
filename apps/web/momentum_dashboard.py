"""Inline Dash application extracted from Momentum & Risk-Adjusted Performance.ipynb.

Run this module from the research notebook so Dash renders in that notebook's output.
"""


# %% [notebook block 1]
# Block 1: notebook description and analysis objective

#This notebook is being used to evaluate momentum, efficiency, relative performance, and factor exposure for a single asset.
#Original Risk Analysis blocks included here: 14-21.


# %% [notebook block 2]
# Block 2: import libraries and initialize analytics services
import logging
import warnings
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display
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

from Quantapp.visualization import (
    Plotter,
    )
from Quantapp.visualization.core import (
    configure_plotly_notebook_renderers,
    )
from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency import (
    plot_benchmark_zscore_detail,
    plot_candlestick_drawdown_recovery_view,
    plot_momentum_zscore_comparison,
    plot_momentum_window_diagnostics_grid_view,
    plot_rolling_correlation_view,
    plot_seasonality_stack_view,
    plot_sharpe_sortino_comparison,
    plot_sharpe_surface_view,
    plot_sharpe_zscore_heatmap_view,
    )
from Quantapp.analytics import compute
from Quantapp.analytics import (
    Metric,
    SeriesTransforms,
    )
from Quantapp.data import GICSDataClient, build_gics_peer_frames, get_market_history

warnings.filterwarnings("ignore")
logger = logging.getLogger("yfinance")
metric = Metric()
series_transforms = SeriesTransforms()

CORRELATION_METRIC_PREFIX = "correlation::"
BENCHMARK_RELATIVE_METRIC_PREFIXES = ("appraisal", "treynor", "information")


def _is_correlation_metric(metric_type):
    return str(metric_type).strip().lower().startswith(CORRELATION_METRIC_PREFIX)


def _correlation_metric_symbol(metric_type):
    return str(metric_type).strip()[len(CORRELATION_METRIC_PREFIX):].strip()


def _is_benchmark_relative_metric(metric_type):
    metric_type = str(metric_type).strip().lower()
    return any(
        metric_type.startswith(f"{prefix}::")
        and bool(metric_type.split("::", 1)[1].strip())
        for prefix in BENCHMARK_RELATIVE_METRIC_PREFIXES
    )


def _benchmark_relative_metric_parts(metric_type):
    metric_name, benchmark_symbol = str(metric_type).strip().split("::", 1)
    return metric_name.lower(), benchmark_symbol.strip()


def risk_adjusted_returns(
    data, windows, ratio_type='sharpe', risk_free_rate=0.0,
    annualization_factor=252, benchmark_prices=None,
):
    ratio_type = str(ratio_type).strip().lower()
    if ratio_type not in {"sharpe", "sortino", "return", "volatility", "downside_volatility"} and not _is_correlation_metric(ratio_type) and not _is_benchmark_relative_metric(ratio_type):
        raise ValueError(
            "Invalid ratio_type. Use 'sharpe', 'sortino', 'return', 'volatility', "
            "or 'downside_volatility', or use a benchmark-relative metric."
        )
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
    benchmark_returns = None
    if _is_correlation_metric(ratio_type) or _is_benchmark_relative_metric(ratio_type):
        if benchmark_prices is None:
            raise ValueError("benchmark_prices is required for a benchmark-relative metric.")
        benchmark_returns = pd.Series(benchmark_prices).pct_change(fill_method=None)
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
        downside_squared = (
            excess.where(excess < 0, 0.0).pow(2)
            if ratio_type == "sortino"
            else None
        )
        for window in window_list:
            mean_excess = excess.rolling(window).mean()
            if ratio_type == 'sharpe':
                volatility = excess.rolling(window).std()
                ratio = np.sqrt(annualization_factor) * mean_excess / volatility
                ratio = ratio.where(volatility > 0)
            elif ratio_type == "sortino":
                downside_deviation = downside_squared.rolling(window).mean().pow(0.5)
                ratio = np.sqrt(annualization_factor) * mean_excess / downside_deviation
                ratio = ratio.where(downside_deviation > 0)
            elif ratio_type == "return":
                ratio = (
                    (1.0 + returns[column]).rolling(window).apply(np.prod, raw=True)
                    ** (annualization_factor / window)
                    - 1.0
                )
            elif ratio_type == "volatility":
                ratio = np.sqrt(annualization_factor) * returns[column].rolling(window).std()
            elif ratio_type == "downside_volatility":
                downside_squared_returns = returns[column].where(
                    returns[column] < 0, 0.0
                ).pow(2)
                ratio = np.sqrt(
                    annualization_factor * downside_squared_returns.rolling(window).mean()
                )
            elif _is_benchmark_relative_metric(ratio_type):
                metric_name, _ = _benchmark_relative_metric_parts(ratio_type)
                aligned_benchmark = benchmark_returns.reindex(returns.index)
                benchmark_excess = aligned_benchmark - periodic_rate
                asset_excess = excess
                if metric_name == "information":
                    active_return = returns[column] - aligned_benchmark
                    tracking_error = active_return.rolling(window).std()
                    ratio = (
                        np.sqrt(annualization_factor)
                        * active_return.rolling(window).mean()
                        / tracking_error
                    ).where(tracking_error > 0)
                else:
                    covariance = asset_excess.rolling(window).cov(benchmark_excess)
                    benchmark_variance = benchmark_excess.rolling(window).var()
                    beta = covariance.div(
                        benchmark_variance.where(benchmark_variance > 0)
                    )
                    if metric_name == "treynor":
                        stable_beta = beta.where(beta.abs() >= 0.05)
                        ratio = annualization_factor * mean_excess / stable_beta
                    else:
                        alpha = (
                            mean_excess
                            - beta * benchmark_excess.rolling(window).mean()
                        )
                        asset_variance = asset_excess.rolling(window).var()
                        residual_variance = (
                            asset_variance
                            + beta.pow(2) * benchmark_variance
                            - 2.0 * beta * covariance
                        ).clip(lower=0.0)
                        residual_risk = residual_variance.pow(0.5)
                        ratio = (
                            np.sqrt(annualization_factor) * alpha / residual_risk
                        ).where(residual_risk > 0)
            else:
                ratio = returns[column].rolling(window).corr(benchmark_returns)

            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio.name = f"{ratio_type}_ratio_{window}" if single_window and single_series else f"{column}_{ratio_type}_{window}"
            output.append(ratio)
    return pd.concat(output, axis=1)


# %% [notebook block 3]
# Block 3: initialize plotting helpers and Momentum & Efficiency display theme

import plotly.graph_objects as go

qp = Plotter()

# This is intentionally notebook-local: importing Quantapp.visualization does not apply this theme.
configure_plotly_notebook_renderers()

CURRENT_VALUE_REFERENCE_META_KEY = "quantapp_current_value_reference"
CURRENT_VALUE_REFERENCE_LINE_STYLE = dict(color="rgba(250, 204, 21, 0.82)", width=2, dash="dash")

def _is_current_value_reference_trace(trace):
    meta = getattr(trace, "meta", None)
    if isinstance(meta, dict) and meta.get(CURRENT_VALUE_REFERENCE_META_KEY):
        return True
    return str(getattr(trace, "name", "")).endswith(" Current Value")

def _current_value_reference_visibility(source_visible):
    return False if source_visible is False else "legendonly"

def _numeric_trace_y(trace):
    y_values = getattr(trace, "y", None)
    if y_values is None:
        return None, None
    numeric_y = pd.to_numeric(pd.Series(list(y_values)), errors="coerce")
    numeric_array = numeric_y.to_numpy(dtype=float)
    finite_mask = np.isfinite(numeric_array)
    return numeric_array, finite_mask

def _is_line_trace_for_current_value(trace):
    if _is_current_value_reference_trace(trace):
        return False
    if getattr(trace, "type", None) not in {"scatter", "scattergl"}:
        return False
    if "lines" not in str(getattr(trace, "mode", "")):
        return False
    if getattr(trace, "hoverinfo", None) == "skip":
        return False
    fill = getattr(trace, "fill", None)
    if fill not in (None, "none"):
        return False

    numeric_y, finite_mask = _numeric_trace_y(trace)
    if numeric_y is None or finite_mask.sum() < 2:
        return False
    return np.unique(numeric_y[finite_mask]).size > 1

def _trace_current_value_reference_payload(trace):
    numeric_y, finite_mask = _numeric_trace_y(trace)
    if numeric_y is None or finite_mask.sum() < 2:
        return None

    x_values = getattr(trace, "x", None)
    if x_values is None:
        x_values = list(range(len(numeric_y)))
    else:
        x_values = list(x_values)
    if len(x_values) != len(numeric_y):
        x_values = list(range(len(numeric_y)))

    valid_x = [x_value for x_value, is_valid in zip(x_values, finite_mask) if is_valid]
    current_value = numeric_y[finite_mask][-1]
    return valid_x[0], valid_x[-1], current_value

def _extend_visibility_buttons_for_current_value_lines(fig, source_indices):
    if not source_indices:
        return
    original_trace_count = len(fig.data) - len(source_indices)
    for menu in fig.layout.updatemenus or []:
        for button in menu.buttons or []:
            args = list(button.args or [])
            if not args or not isinstance(args[0], dict) or "visible" not in args[0]:
                continue
            visible = list(args[0]["visible"])
            if len(visible) != original_trace_count:
                continue
            args[0]["visible"] = visible + [
                _current_value_reference_visibility(visible[source_index])
                for source_index in source_indices
            ]
            button.args = tuple(args)

def _style_current_value_reference_lines(fig):
    for trace in fig.data:
        if _is_current_value_reference_trace(trace):
            trace.update(line=CURRENT_VALUE_REFERENCE_LINE_STYLE.copy(), visible="legendonly", showlegend=True)

def _add_current_value_reference_lines(fig):
    if any(_is_current_value_reference_trace(trace) for trace in fig.data):
        _style_current_value_reference_lines(fig)
        return fig

    source_indices = []
    for source_index, trace in enumerate(list(fig.data)):
        if not _is_line_trace_for_current_value(trace):
            continue
        payload = _trace_current_value_reference_payload(trace)
        if payload is None:
            continue
        x_start, x_end, current_value = payload
        reference_trace = dict(
            x=[x_start, x_end],
            y=[current_value, current_value],
            mode="lines",
            name=f"{getattr(trace, 'name', '') or 'Series'} Current Value",
            line=CURRENT_VALUE_REFERENCE_LINE_STYLE.copy(),
            hoverinfo="skip",
            showlegend=True,
            visible=_current_value_reference_visibility(getattr(trace, "visible", True)),
            meta={CURRENT_VALUE_REFERENCE_META_KEY: True},
        )
        xaxis = getattr(trace, "xaxis", None)
        yaxis = getattr(trace, "yaxis", None)
        if xaxis:
            reference_trace["xaxis"] = xaxis
        if yaxis:
            reference_trace["yaxis"] = yaxis
        fig.add_trace(go.Scatter(**reference_trace))
        source_indices.append(source_index)

    _extend_visibility_buttons_for_current_value_lines(fig, source_indices)
    return fig

if not hasattr(go.Figure, "_quantapp_original_show"):
    go.Figure._quantapp_original_show = go.Figure.show

def _quantapp_show_with_current_value_lines(self, *args, **kwargs):
    _add_current_value_reference_lines(self)
    return go.Figure._quantapp_original_show(self, *args, **kwargs)

go.Figure.show = _quantapp_show_with_current_value_lines


# %% [notebook block 4]
# Block 4: load shared parameters and set notebook-specific controls

pricing_params = get_single_asset_params()
# Accept the common class-share spellings used in portfolio exports and user
# input.  Yahoo Finance uses a hyphen (for example, BRK-B), while users often
# enter BRK/B, BRK\B, or BRK.B.
ticker_str = (
    str(pricing_params["ticker_str"])
    .strip()
    .upper()
    .replace("/", "-")
    .replace("\\", "-")
    .replace(".", "-")
)
if not ticker_str:
    raise ValueError("ticker_str must contain a ticker symbol.")
interval = pricing_params["interval"]
period = pricing_params["period"]

vix_str = "^VIX"
risk_free_ticker = "^IRX"
# SPY is the normal broad-market benchmark, but it cannot also serve as its
# own comparison series: the role-assignment step below intentionally removes
# the analyzed asset from ``benchmark_data``.  Use the underlying S&P 500 index
# when SPY itself is the asset so benchmark-dependent dashboard blocks remain
# populated and the comparison is economically meaningful.
benchmark_tickers = ["^GSPC"] if str(ticker_str).strip().upper() == "SPY" else ["SPY"]
include_factor_peer_index = True
auto_build_factor_peer_index = False  # Load cached GICS indexes without building a large peer basket during startup.
factor_peer_index_max_symbols = 40  # Limit peer downloads while retaining a broad peer basket.
factor_peer_index_source_ticker = None  # None = current ticker. For an ETF, optionally set a representative stock whose Factor peer index should be used.
peer_index_cache_dir = PROJECT_ROOT / "company_data" / "factor_peer_indexes"
time_frame_map = {"short": 21, "mid": 50, "long": 200}
selected_time_frames = [21, 50, 200]
default_window = 200
length_of_plots = 20
var_position_value = None


# %% [notebook block 5]
# Block 4A: Direct yfinance Price Retrieval Check
# Run this after Block 4 when you want to verify raw Yahoo Finance prices.

import yfinance as yf

run_direct_yfinance_check = False
direct_yfinance_symbols = list(dict.fromkeys([
    ticker_str,
    vix_str,
    risk_free_ticker,
    *benchmark_tickers,
]))

def _direct_yfinance_flatten_columns(frame):
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    standard_price_columns = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    for level in range(frame.columns.nlevels):
        level_values = pd.Index(frame.columns.get_level_values(level))
        if standard_price_columns.intersection(set(level_values.astype(str))):
            flattened = frame.copy()
            flattened.columns = level_values
            return flattened.loc[:, ~flattened.columns.duplicated()]

    flattened = frame.copy()
    flattened.columns = ["_".join(str(part) for part in column if str(part)) for column in frame.columns.to_flat_index()]
    return flattened

def _direct_yfinance_history(symbol):
    frame = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        frame = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
    frame = _direct_yfinance_flatten_columns(frame)
    if not frame.empty:
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None).normalize()
        frame = frame[~frame.index.isna()].sort_index()
    return frame

direct_yfinance_price_history = (
    {
        symbol: _direct_yfinance_history(symbol)
        for symbol in direct_yfinance_symbols
    }
    if run_direct_yfinance_check
    else {}
)

direct_yfinance_price_summary = pd.DataFrame(
    [
        {
            "Symbol": symbol,
            "Rows": len(frame),
            "Valid Close Rows": int(pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not frame.empty else 0,
            "Start": frame.index.min() if not frame.empty else pd.NaT,
            "End": frame.index.max() if not frame.empty else pd.NaT,
            "Last Close": pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce").dropna().iloc[-1] if not frame.empty and not pd.to_numeric(frame.get("Close", pd.Series(dtype=float)), errors="coerce").dropna().empty else np.nan,
        }
        for symbol, frame in direct_yfinance_price_history.items()
    ]
)

#display(direct_yfinance_price_summary)
#direct_yfinance_price_history.get(ticker_str, pd.DataFrame()).tail()


# %% [notebook block 6]
# Block 5: fetch market history and assign notebook roles
requested_symbols = [
    ticker_str,
    vix_str,
    risk_free_ticker,
    *benchmark_tickers,
]

asset_histories = get_market_history(
    symbols=requested_symbols,
    period=period,
    interval=interval,
    provider="yfinance",
    # Keep histories unaligned here so a sparse proxy such as ^IRX cannot collapse
    # the asset and benchmark histories to only their shared dates.
    align=False,
)

asset_history           = asset_histories.get(ticker_str, pd.DataFrame())
vix_history             = asset_histories.get(vix_str, pd.DataFrame())
risk_free_proxy_history = asset_histories.get(risk_free_ticker, pd.DataFrame())

benchmark_data = {
    symbol: frame
    for symbol, frame in asset_histories.items()
    if symbol not in {ticker_str, vix_str, risk_free_ticker}
}

factor_gics_index_suffixes = {
    "Sector": "sector",
    "Industry Group": "industry_group",
    "Industry": "industry",
    "Sub-Industry": "sub_industry",
}

def factor_index_display_label(cached, default_label):
    """Prefer descriptive GICS labels over target-ticker cache labels."""
    cached_name = cached.get("GICS Name", pd.Series(dtype="object")).dropna()
    cached_level = cached.get("GICS Level", pd.Series(dtype="object")).dropna()
    if not cached_name.empty and not cached_level.empty:
        return f"{str(cached_name.iloc[-1]).strip()} - {str(cached_level.iloc[-1]).strip()}"
    cached_label = cached.get("Benchmark Label", pd.Series(dtype="object")).dropna()
    return str(cached_label.iloc[-1]) if not cached_label.empty else default_label


def load_factor_index_cache(cache_path, default_label):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None, None
    cached = pd.read_csv(cache_path)
    required_columns = {"Date", "Close"}
    if not required_columns.issubset(cached.columns):
        raise ValueError(f"Factor-index cache is missing columns {sorted(required_columns - set(cached.columns))}: {cache_path}")

    dates = pd.to_datetime(cached["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    close = pd.to_numeric(cached["Close"], errors="coerce")
    index_frame = pd.DataFrame({"Close": close.to_numpy()}, index=dates)
    index_frame = index_frame.loc[~index_frame.index.isna()].dropna(subset=["Close"])
    index_frame = index_frame.loc[~index_frame.index.duplicated(keep="last")].sort_index()
    if not asset_history.empty:
        index_frame = index_frame.loc[asset_history.index.min():asset_history.index.max()]

    label = factor_index_display_label(cached, default_label)
    cached_level = cached.get("GICS Level", pd.Series(dtype="object")).dropna()
    level = str(cached_level.iloc[-1]) if not cached_level.empty else None
    return index_frame, (label, level)

def load_factor_peer_index(target_symbol, cache_dir):
    safe_symbol = str(target_symbol).replace(".", "_").replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_symbol}_peer_index.csv"
    peer_frame, metadata = load_factor_index_cache(cache_path, f"{target_symbol} Peer Index")
    return peer_frame, cache_path, metadata

def build_missing_factor_peer_index(target_symbol, cache_dir, max_symbols=40):
    """Build and cache an equal-weight GICS sub-industry peer index."""
    company_cache_path = PROJECT_ROOT / "company_data" / "gics_companies.csv"
    if company_cache_path.exists():
        companies = pd.read_csv(company_cache_path)
    else:
        gics_client = GICSDataClient(save_path=PROJECT_ROOT)
        companies = gics_client.retrieve_companies()
        company_cache_path.parent.mkdir(parents=True, exist_ok=True)
        companies.to_csv(company_cache_path, index=False)

    peer_context = build_gics_peer_frames(
        target_symbol, companies=companies, symbol_label="YFinance Symbol"
    )
    peer_rows = peer_context.frame("Sub-Industry")
    peer_symbols = list(dict.fromkeys(peer_rows["Normalized Symbol"].dropna().astype(str)))
    peer_symbols = [symbol for symbol in peer_symbols if symbol != peer_context.target_symbol][:max_symbols]
    if len(peer_symbols) < 2:
        raise ValueError(
            f"At least two GICS sub-industry peers are required for {target_symbol}; found {len(peer_symbols)}."
        )

    peer_histories = get_market_history(
        symbols=peer_symbols, period=period, interval=interval, provider="yfinance", align=False
    )
    peer_closes = []
    for symbol in peer_symbols:
        history = peer_histories.get(symbol, pd.DataFrame())
        if history is not None and not history.empty and "Close" in history.columns:
            peer_closes.append(pd.to_numeric(history["Close"], errors="coerce").rename(symbol))
    if len(peer_closes) < 2:
        raise ValueError(f"Usable price history was available for only {len(peer_closes)} peers.")

    peer_prices = pd.concat(peer_closes, axis=1).sort_index()
    peer_returns = peer_prices.pct_change(fill_method=None)
    valid_counts = peer_returns.notna().sum(axis=1)
    index_returns = peer_returns.mean(axis=1, skipna=True).where(valid_counts >= 2).dropna()
    if index_returns.empty:
        raise ValueError(f"Peer histories for {target_symbol} did not have overlapping return dates.")
    peer_index = (100.0 * (1.0 + index_returns).cumprod()).rename("Close")
    first_date = peer_prices.index[peer_prices.index < peer_index.index[0]]
    if len(first_date):
        peer_index = pd.concat([pd.Series([100.0], index=first_date[-1:], name="Close"), peer_index])

    safe_symbol = str(target_symbol).replace(".", "_").replace("/", "_")
    cache_path = Path(cache_dir) / f"{safe_symbol}_peer_index.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    peer_label = f"{peer_context.target_row['Sub-Industry']} - Sub-Industry"
    export_frame = peer_index.to_frame()
    export_frame["Target Symbol"] = target_symbol
    export_frame["Benchmark Label"] = peer_label
    export_frame["GICS Level"] = "Sub-Industry"
    export_frame["GICS Name"] = peer_context.target_row["Sub-Industry"]
    export_frame["Weighting"] = "Equal Weight"
    export_frame["Constituent Count"] = len(peer_closes)
    export_frame.to_csv(cache_path, index_label="Date")
    return cache_path, peer_label, len(peer_closes)

def load_factor_gics_indexes(target_symbol, cache_dir):
    safe_symbol = str(target_symbol).replace(".", "_").replace("/", "_")
    loaded_indexes = {}
    cache_paths = {}
    for gics_level, file_suffix in factor_gics_index_suffixes.items():
        cache_path = Path(cache_dir) / f"{safe_symbol}_{file_suffix}_index.csv"
        cache_paths[gics_level] = cache_path
        index_frame, metadata = load_factor_index_cache(
            cache_path, f"{target_symbol} {gics_level} Index"
        )
        if index_frame is not None:
            label, stored_level = metadata
            loaded_indexes[label] = {
                "frame": index_frame,
                "level": stored_level or gics_level,
                "path": cache_path,
            }
    return loaded_indexes, cache_paths

if include_factor_peer_index:
    factor_peer_lookup_symbol = factor_peer_index_source_ticker or ticker_str
    factor_gics_benchmarks, factor_gics_cache_paths = load_factor_gics_indexes(
        factor_peer_lookup_symbol, peer_index_cache_dir
    )
    if factor_gics_benchmarks:
        for factor_index_label, factor_index_payload in factor_gics_benchmarks.items():
            factor_index_frame = factor_index_payload["frame"]
            if factor_index_frame.empty:
                print(f"Factor index cache has no dates overlapping the asset history: {factor_index_payload['path']}")
                continue
            benchmark_data[factor_index_label] = factor_index_frame
            print(
                f"Loaded benchmark {factor_index_label} ({factor_index_payload['level']})"
                + (f" for ETF/asset {ticker_str}" if factor_peer_lookup_symbol != ticker_str else "")
                + f" from {factor_index_payload['path']}"
            )
        missing_gics_levels = [
            level for level, path in factor_gics_cache_paths.items() if not path.exists()
        ]
        if missing_gics_levels:
            print(f"Missing Factor GICS index caches: {missing_gics_levels}. Rerun Factor Analysis block 5.")
    else:
        factor_peer_frame, factor_peer_cache_path, factor_peer_metadata = load_factor_peer_index(
            factor_peer_lookup_symbol, peer_index_cache_dir
        )
        if factor_peer_frame is not None and not factor_peer_frame.empty:
            factor_peer_label, factor_peer_level = factor_peer_metadata
            benchmark_data[factor_peer_label] = factor_peer_frame
            print(
                f"Loaded legacy benchmark {factor_peer_label}"
                + (f" ({factor_peer_level})" if factor_peer_level else "")
                + f" from {factor_peer_cache_path}"
            )
        elif auto_build_factor_peer_index:
            try:
                built_path, built_label, constituent_count = build_missing_factor_peer_index(
                    factor_peer_lookup_symbol,
                    peer_index_cache_dir,
                    max_symbols=factor_peer_index_max_symbols,
                )
                factor_peer_frame, _, factor_peer_metadata = load_factor_peer_index(
                    factor_peer_lookup_symbol, peer_index_cache_dir
                )
                if factor_peer_frame is None or factor_peer_frame.empty:
                    raise ValueError("The generated peer index has no dates overlapping the asset history.")
                benchmark_data[built_label] = factor_peer_frame
                print(
                    f"Built and loaded {built_label} from {constituent_count} GICS peers; "
                    f"cached at {built_path}"
                )
            except Exception as exc:
                print(
                    f"Could not automatically build a peer index for {factor_peer_lookup_symbol}: {exc}. "
                    f"Using {benchmark_tickers} only. For an ETF, set "
                    "factor_peer_index_source_ticker to a representative company."
                )
        else:
            print(
                f"Peer index not found for {factor_peer_lookup_symbol}; using {benchmark_tickers} only."
            )

#loaded_benchmark_tickers = list(benchmark_data)
#analysis_index = asset_history.index


# %% [notebook block 7]
# Block 7: derive analysis series from normalized market data

try:
    risk_free_daily_rate = series_transforms.annualized_yield_to_periodic_rate(
        risk_free_proxy_history,
        annualization_factor=252,
        input_is_percent=True,
        lag_periods=1,
        reference_index=asset_history.index,
    )
except (TypeError, ValueError):
    risk_free_daily_rate = pd.Series(0.0, index=asset_history.index)

risk_free_daily_rate = pd.Series(risk_free_daily_rate, index=asset_history.index).replace([np.inf, -np.inf], np.nan)
if risk_free_daily_rate.dropna().empty:
    if isinstance(risk_free_proxy_history, pd.DataFrame) and "Close" in risk_free_proxy_history:
        latest_risk_free_yield = pd.to_numeric(risk_free_proxy_history["Close"], errors="coerce").dropna()
    else:
        latest_risk_free_yield = pd.Series(dtype=float)

    if latest_risk_free_yield.empty:
        risk_free_daily_rate = pd.Series(0.0, index=asset_history.index)
        print("Risk-free proxy unavailable; using 0.00% annual risk-free rate fallback.")
    else:
        fallback_annual_rate = float(latest_risk_free_yield.iloc[-1]) / 100.0
        fallback_daily_rate = (1.0 + fallback_annual_rate) ** (1.0 / 252) - 1.0
        risk_free_daily_rate = pd.Series(fallback_daily_rate, index=asset_history.index)
        print(
            f"Risk-free proxy has sparse aligned data; using latest {risk_free_ticker} "
            f"yield as a constant fallback: {fallback_annual_rate:.2%} annual."
        )
else:
    risk_free_daily_rate = risk_free_daily_rate.ffill().bfill()


ticker_monthly_data = series_transforms.resample(asset_history, frequency="monthly")
ticker_weekly_data = series_transforms.resample(asset_history, frequency="weekly")
ticker_daily_data = series_transforms.resample(asset_history, frequency="daily")

ticker_monthly_returns = ticker_monthly_data["Close"].pct_change(fill_method=None).dropna()
ticker_weekly_returns = ticker_weekly_data["Close"].pct_change(fill_method=None).dropna()
ticker_daily_returns = ticker_daily_data["Close"].pct_change(fill_method=None).dropna()


# %% [notebook block 8]
# Block 12: plot stacked rolling Sharpe z-score heatmaps for 1-200 day windows plus cross-window summaries

heatmap_windows = list(range(1, 201))

def heatmap_zscore(series):
    clean = pd.Series(series).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    std = clean.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=clean.index)
    return (clean - clean.mean()) / std

def rolling_sharpe_frame(close):
    return risk_adjusted_returns(
        close.dropna().sort_index(),
        windows=heatmap_windows,
        ratio_type="sharpe",
        risk_free_rate=risk_free_daily_rate,
    ).set_axis(heatmap_windows, axis=1)

asset_sharpe_frame = rolling_sharpe_frame(asset_history["Close"])

asset_sharpe_zscore_frame = asset_sharpe_frame.apply(heatmap_zscore)

benchmark_sharpe_zscore_frames = {}
benchmark_spread_zscore_frames = {}
for symbol, benchmark_frame in benchmark_data.items():
    benchmark_sharpe_frame = rolling_sharpe_frame(benchmark_frame["Close"])
    benchmark_sharpe_zscore_frames[symbol] = benchmark_sharpe_frame.apply(heatmap_zscore)
    benchmark_spread = benchmark_sharpe_frame - asset_sharpe_frame
    benchmark_spread_zscore_frames[symbol] = benchmark_spread.apply(heatmap_zscore)


#display(benchmark_spread_zscore_frames)
fig_block12_sharpe_zscore_heatmap = plot_sharpe_zscore_heatmap_view(
    asset_sharpe_zscore_frame=asset_sharpe_zscore_frame,
    benchmark_sharpe_zscore_frames=benchmark_sharpe_zscore_frames,
    benchmark_spread_zscore_frames=benchmark_spread_zscore_frames,
    ticker_label=ticker_str,
)
fig = fig_block12_sharpe_zscore_heatmap


# %% [notebook block 9]
# Block 11: compute rolling Sharpe windows, momentum histograms, and volatility

block11_min_window = 3
block11_fallback_max_window = 400
annualization_factor = 252

import yfinance as yf

def block11_current_option_chain_dtes(symbol, *, min_dte=0, max_dte=None):
    try:
        expiration_values = yf.Ticker(symbol).options or []
    except Exception as error:
        print(f"Option-chain expirations unavailable for {symbol}: {error}")
        return pd.DataFrame(columns=["Expiration Date", "DTE"])

    expiration_dates = pd.to_datetime(list(expiration_values), errors="coerce")
    expiration_dates = pd.DatetimeIndex(expiration_dates).dropna().normalize()
    if expiration_dates.empty:
        return pd.DataFrame(columns=["Expiration Date", "DTE"])

    today = pd.Timestamp.today().normalize()
    dte_frame = pd.DataFrame({"Expiration Date": expiration_dates})
    dte_frame["DTE"] = (dte_frame["Expiration Date"] - today).dt.days.astype(int)
    dte_frame = dte_frame[dte_frame["DTE"] >= int(min_dte)]
    if max_dte is not None:
        dte_frame = dte_frame[dte_frame["DTE"] <= int(max_dte)]
    return dte_frame.drop_duplicates("DTE").sort_values("DTE").reset_index(drop=True)

def block11_add_option_dte_vlines(fig, dte_frame):
    if dte_frame.empty:
        print(
            f"No listed {ticker_str} option-chain DTEs fall inside the "
            f"{min(window_sizes)}-{max(window_sizes)} day Block 11 window range."
        )
        return fig

    marker_rows = [
        (1, 2), (2, 1), (2, 2),
        (3, 1), (4, 1), (5, 1),
    ]
    annotation_rows = {(1, 2), (2, 1), (2, 2), (3, 1)}
    for marker_index, marker in dte_frame.iterrows():
        dte = int(marker["DTE"])
        expiration_date = pd.Timestamp(marker["Expiration Date"])
        for subplot_row, subplot_col in marker_rows:
            vline_kwargs = dict(
                x=dte,
                line_color="rgba(250, 204, 21, 0.58)",
                line_width=1,
                line_dash="dot",
                opacity=0.72,
                row=subplot_row,
                col=subplot_col,
            )
            if (subplot_row, subplot_col) in annotation_rows:
                vline_kwargs.update(
                    annotation_text=f"Exact {dte} DTE<br>{expiration_date:%Y-%m-%d}",
                    annotation_position="top left",
                    annotation_font_size=10,
                    annotation_font_color="rgba(250, 204, 21, 0.92)",
                )
            fig.add_vline(**vline_kwargs)
    return fig

block11_option_chain_dtes = block11_current_option_chain_dtes(
    ticker_str,
    min_dte=block11_min_window,
    max_dte=None,
)
if block11_option_chain_dtes.empty:
    block11_max_window = block11_fallback_max_window
    block11_window_limit_source = "fallback because the option chain was unavailable"
    print(
        f"No current option-chain expirations were available for {ticker_str}; "
        f"using the {block11_fallback_max_window}-day diagnostics fallback."
    )
else:
    block11_max_window = int(block11_option_chain_dtes["DTE"].max())
    block11_window_limit_source = f"latest listed {ticker_str} option DTE"
window_sizes = list(range(block11_min_window, block11_max_window + 1))
print(
    f"Block 11 diagnostics windows: {block11_min_window}-{block11_max_window} days "
    f"({block11_window_limit_source})."
)

MOMENTUM_RATIO_TYPES = (
    "sharpe", "sortino", "return", "volatility", "downside_volatility"
)
MOMENTUM_RATIO_LABELS = {
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "return": "Return",
    "volatility": "Volatility",
    "downside_volatility": "Downside Volatility",
}


def normalize_momentum_ratio_type(ratio_type):
    ratio_type = str(ratio_type or "sharpe").strip()
    if _is_correlation_metric(ratio_type):
        benchmark_symbol = _correlation_metric_symbol(ratio_type)
        if not benchmark_symbol:
            raise ValueError("A correlation metric must identify a benchmark.")
        return f"{CORRELATION_METRIC_PREFIX}{benchmark_symbol}"
    if _is_benchmark_relative_metric(ratio_type):
        metric_name, benchmark_symbol = _benchmark_relative_metric_parts(ratio_type)
        return f"{metric_name}::{benchmark_symbol}"
    ratio_type = ratio_type.lower()
    if ratio_type not in MOMENTUM_RATIO_TYPES and not _is_correlation_metric(ratio_type):
        raise ValueError(
            "ratio_type must be 'sharpe', 'sortino', 'return', 'volatility', "
            "or 'downside_volatility', or identify a benchmark for Appraisal, "
            "Treynor, or Information ratio."
        )
    return ratio_type


def momentum_ratio_label(ratio_type):
    ratio_type = normalize_momentum_ratio_type(ratio_type)
    if _is_correlation_metric(ratio_type):
        return f"Correlation vs {_correlation_metric_symbol(ratio_type)}"
    if _is_benchmark_relative_metric(ratio_type):
        metric_name, benchmark_symbol = _benchmark_relative_metric_parts(ratio_type)
        return f"{metric_name.title()} vs {benchmark_symbol}"
    return MOMENTUM_RATIO_LABELS[ratio_type]


def _momentum_diagnostics_base(close):
    momentum_close = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    aligned_risk_free_daily_rate = pd.Series(risk_free_daily_rate, index=momentum_close.index).ffill().bfill()
    returns = momentum_close.pct_change()
    excess_returns = returns - aligned_risk_free_daily_rate
    volatility_df = np.sqrt(annualization_factor) * compute.rolling_windows(
        excess_returns,
        metric=pd.Series.std,
        windows=window_sizes,
    )
    return momentum_close, aligned_risk_free_daily_rate, volatility_df


def _momentum_diagnostics_context_from_base(
    momentum_close,
    aligned_risk_free_daily_rate,
    volatility_df,
    *,
    ratio_type,
    highlight_windows=(),
):
    ratio_type = normalize_momentum_ratio_type(ratio_type)
    benchmark_prices = None
    if _is_correlation_metric(ratio_type) or _is_benchmark_relative_metric(ratio_type):
        benchmark_symbol = (
            _correlation_metric_symbol(ratio_type)
            if _is_correlation_metric(ratio_type)
            else _benchmark_relative_metric_parts(ratio_type)[1]
        )
        benchmark_frame = benchmark_data.get(benchmark_symbol)
        if benchmark_frame is None or "Close" not in benchmark_frame:
            raise ValueError(f"No benchmark Close history is available for {benchmark_symbol}.")
        benchmark_prices = benchmark_frame["Close"]
    ratio_table = risk_adjusted_returns(
        momentum_close,
        windows=window_sizes,
        ratio_type=ratio_type,
        risk_free_rate=aligned_risk_free_daily_rate,
        annualization_factor=annualization_factor,
        benchmark_prices=benchmark_prices,
    ).set_axis(window_sizes, axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna(how="all").copy()

    context = {
        "ratio_type": ratio_type,
        "ratio_label": momentum_ratio_label(ratio_type),
        "ratio_table": ratio_table,
        f"{ratio_type}_table": ratio_table,
        # Compatibility alias for the existing visualization schema. For a
        # Sortino context this contains Sortino values by design.
        "sharpe_table": ratio_table,
        "volatility_df": volatility_df,
        "window_sizes": tuple(window_sizes),
        "highlight_windows": tuple(highlight_windows),
    }
    return context


def build_momentum_diagnostics_context(
    close, *, ratio_type="sharpe", highlight_windows=()
):
    """Build one full-horizon risk-adjusted diagnostics context."""
    base = _momentum_diagnostics_base(close)
    return _momentum_diagnostics_context_from_base(
        *base,
        ratio_type=ratio_type,
        highlight_windows=highlight_windows,
    )


def build_momentum_diagnostics_contexts(
    close, *, ratio_types=MOMENTUM_RATIO_TYPES, highlight_windows=()
):
    """Build multiple ratio contexts while computing shared volatility once."""
    if isinstance(ratio_types, str):
        ratio_types = [ratio_types]
    normalized_ratio_types = tuple(dict.fromkeys(
        normalize_momentum_ratio_type(ratio_type) for ratio_type in ratio_types
    ))
    if not normalized_ratio_types:
        raise ValueError("ratio_types must contain at least one supported ratio.")
    base = _momentum_diagnostics_base(close)
    return {
        ratio_type: _momentum_diagnostics_context_from_base(
            *base,
            ratio_type=ratio_type,
            highlight_windows=highlight_windows,
        )
        for ratio_type in normalized_ratio_types
    }

from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency._shared import coerce_momentum_diagnostics_context
from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency.momentum_window_diagnostics_grid import (
    HORIZON_SEGMENT_LINE_STYLES,
    HORIZON_SEGMENT_REGIONS,
    _horizon_derivative_series,
)
from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency.sharpe_zscore_heatmap import (
    MEAN_PANEL_REFERENCE_LEVELS,
    MEAN_PANEL_REFERENCE_LINE_COLOR,
    MEAN_PANEL_ZONE_FILLS,
)

def block11_axis_range(series_collection, *, padding=0.12, min_span=1.0):
    finite_chunks = []
    for values in series_collection:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite_values = numeric[np.isfinite(numeric)]
        if finite_values.size > 0:
            finite_chunks.append(finite_values)
    if not finite_chunks:
        return None
    values = np.concatenate(finite_chunks)
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    if span <= 0:
        midpoint = (lower + upper) / 2.0
        span = min_span
        lower = midpoint - span / 2.0
        upper = midpoint + span / 2.0
    axis_padding = max(span * padding, min_span * 0.05)
    return [lower - axis_padding, upper + axis_padding]

def block11_reference_band_series(mean_by_window, std_by_window):
    mean_series = pd.Series(mean_by_window)
    std_series = pd.Series(std_by_window)
    return [
        mean_series + sign * std_series * level
        for level in (1, 2)
        for sign in (1, -1)
    ]

def block11_add_horizon_derivative_traces(fig, series, *, symbol, base_name, color, dash, rows):
    for order, row in zip((1, 2), rows):
        derivative = _horizon_derivative_series(series.index, series.values, order=order)
        derivative_label = "First Derivative" if order == 1 else "Second Derivative"
        units = "Z-score / 20 horizon days" if order == 1 else "Z-score / (20 horizon days)^2"
        fig.add_trace(
            go.Scatter(
                x=derivative.index,
                y=derivative.values,
                mode="lines",
                name=f"{base_name} — {derivative_label}",
                line=dict(color=color, width=2.0, dash=dash),
                showlegend=False,
                hovertemplate=(
                    f"Benchmark / index: {symbol}<br>"
                    "Lookback horizon: %{x} trading days<br>"
                    + derivative_label + ": %{y:.3f} " + units + "<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )

momentum_diagnostics_contexts_by_ratio = {
    ratio_type: {} for ratio_type in MOMENTUM_RATIO_TYPES
}
momentum_diagnostics_sources = {ticker_str: asset_history["Close"]}
momentum_diagnostics_sources.update({
    symbol: benchmark_frame["Close"]
    for symbol, benchmark_frame in benchmark_data.items()
    if isinstance(benchmark_frame, pd.DataFrame) and "Close" in benchmark_frame
})
for symbol, close in momentum_diagnostics_sources.items():
    # Only the default metric is needed during startup. Other metrics are
    # calculated and cached when the user first selects them.
    symbol_contexts = build_momentum_diagnostics_contexts(
        close, ratio_types=("sharpe",)
    )
    for ratio_type, diagnostics_context in symbol_contexts.items():
        momentum_diagnostics_contexts_by_ratio[ratio_type][symbol] = diagnostics_context

# Original names remain Sharpe aliases so downstream notebook cells and the
# current dashboard keep working while ratio-aware renderers are introduced.
momentum_diagnostics_contexts = momentum_diagnostics_contexts_by_ratio["sharpe"]

# Preserve the original single-asset names for downstream notebook cells.
momentum_diagnostics_context = momentum_diagnostics_contexts[ticker_str]
sharpe_table = momentum_diagnostics_context["sharpe_table"]
volatility_df = momentum_diagnostics_context["volatility_df"]

momentum_diagnostics_display_contexts = {
    symbol: coerce_momentum_diagnostics_context(context)
    for symbol, context in momentum_diagnostics_contexts.items()
}
momentum_diagnostics_display_contexts_by_ratio = {
    "sharpe": momentum_diagnostics_display_contexts,
    **{
        ratio_type: {
            symbol: coerce_momentum_diagnostics_context(context)
            for symbol, context in ratio_contexts.items()
        }
        for ratio_type, ratio_contexts in momentum_diagnostics_contexts_by_ratio.items()
        if ratio_type != "sharpe"
    },
}


def _ensure_momentum_diagnostics_ratio(ratio_type):
    """Build one metric's asset/benchmark contexts once, on first use."""
    ratio_type = normalize_momentum_ratio_type(ratio_type)
    ratio_contexts = momentum_diagnostics_contexts_by_ratio.setdefault(
        ratio_type, {}
    )
    display_contexts = momentum_diagnostics_display_contexts_by_ratio.setdefault(
        ratio_type, {}
    )
    for symbol, close in momentum_diagnostics_sources.items():
        if symbol in ratio_contexts:
            continue
        context = build_momentum_diagnostics_context(
            close, ratio_type=ratio_type
        )
        ratio_contexts[symbol] = context
        display_contexts[symbol] = coerce_momentum_diagnostics_context(context)
    return ratio_contexts, display_contexts

fig_momentum_window_diagnostics_grid = plot_momentum_window_diagnostics_grid_view(
    diagnostics_context=momentum_diagnostics_context,
    ticker_label=ticker_str,
)

block11_benchmark_colors = ["#f97316", "#22c55e", "#facc15", "#ef4444", "#ec4899", "#f8fafc"]
block11_benchmark_dashes = ["dash", "dot", "longdash", "dashdot", "solid"]

for benchmark_index, (symbol, display_context) in enumerate(momentum_diagnostics_display_contexts.items()):
    if symbol == ticker_str:
        continue
    color = block11_benchmark_colors[benchmark_index % len(block11_benchmark_colors)]
    dash = block11_benchmark_dashes[benchmark_index % len(block11_benchmark_dashes)]

    benchmark_current_sharpe_zscore = display_context["current_sharpe_zscore"].dropna()
    if not benchmark_current_sharpe_zscore.empty:
        fig_momentum_window_diagnostics_grid.add_trace(
            go.Scatter(
                x=benchmark_current_sharpe_zscore.index,
                y=benchmark_current_sharpe_zscore.values,
                mode="lines+markers",
                name=f"{symbol} Current Sharpe Z-Score",
                line=dict(color=color, width=2.2, dash=dash),
                marker=dict(size=4),
                hovertemplate=(
                    "Benchmark: " + symbol + "<br>"
                    "Window: %{x} day(s)<br>"
                    "Current Sharpe Z-Score: %{y:.2f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        block11_add_horizon_derivative_traces(
            fig_momentum_window_diagnostics_grid,
            benchmark_current_sharpe_zscore,
            symbol=symbol,
            base_name=f"{symbol} Current Sharpe Z-Score",
            color=color,
            dash=dash,
            rows=(4, 5),
        )

asset_display_context = momentum_diagnostics_display_contexts[ticker_str]
row3_range_series = [
    asset_display_context["current_sharpe_zscore"],
    asset_display_context["sharpe_zscore_mean_by_window"],
    *block11_reference_band_series(
        asset_display_context["sharpe_zscore_mean_by_window"],
        asset_display_context["sharpe_zscore_std_by_window"],
    ),
    *[
        context["current_sharpe_zscore"]
        for symbol, context in momentum_diagnostics_display_contexts.items()
        if symbol != ticker_str
    ],
    pd.Series([-2.0, 2.0]),
]
fig_momentum_window_diagnostics_grid.update_yaxes(range=block11_axis_range(row3_range_series), row=3, col=1)
block11_add_option_dte_vlines(fig_momentum_window_diagnostics_grid, block11_option_chain_dtes)
fig = fig_momentum_window_diagnostics_grid


# %% [notebook block 10]
# Block 8: compute rolling arithmetic/geometric mean returns with MAD-score details

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency._shared import (
    finalize_dark_figure,
    header_margin,
    header_title,
    trace_datetime_bounds,
)

configured_rolling_mean_windows = globals().get(
    "selected_time_frames",
    [globals().get("default_window", 200)],
)
if isinstance(configured_rolling_mean_windows, (int, float, np.integer)):
    configured_rolling_mean_windows = [configured_rolling_mean_windows]

rolling_mean_windows = []
for window in configured_rolling_mean_windows:
    try:
        window = int(window)
    except (TypeError, ValueError):
        continue
    if window > 0 and window not in rolling_mean_windows:
        rolling_mean_windows.append(window)
if not rolling_mean_windows:
    rolling_mean_windows = [200]

preferred_rolling_mean_window = int(globals().get("default_window", 200))
rolling_mean_window = (
    preferred_rolling_mean_window
    if preferred_rolling_mean_window in rolling_mean_windows
    else 200
    if 200 in rolling_mean_windows
    else max(rolling_mean_windows)
)

def rolling_arithmetic_label(window):
    return f"{int(window)}D Arithmetic Mean"

def rolling_geometric_label(window):
    return f"{int(window)}D Geometric Mean"

compounding_efficiency_label = "Compounding Efficiency"
volatility_drag_label = "Volatility Drag"

def compounding_efficiency_from_means(arithmetic_mean, geometric_mean):
    return (1.0 + geometric_mean).div(1.0 + arithmetic_mean).replace([np.inf, -np.inf], np.nan)

def volatility_drag_from_means(arithmetic_mean, geometric_mean):
    return arithmetic_mean - geometric_mean

def rolling_metric_mad_score(series):
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    median = clean.median()
    mad = (clean - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(0.0, index=clean.index)
    return ((clean - median) / (1.4826 * mad)).dropna()

def metric_detail_array(value_series, mad_score_series, index):
    detail_frame = pd.concat(
        {
            "value": pd.Series(value_series),
            "mad_score": pd.Series(mad_score_series),
        },
        axis=1,
    ).reindex(index)
    return detail_frame[["value", "mad_score"]].to_numpy()

def mad_score_annotation_text(metric_label, metric_symbol, latex_formula, function_name):
    return (
        f"<b>{metric_label}</b><br>"
        f"${metric_symbol}_t = {latex_formula}$<br>"
        rf"$m_t = \frac{{{metric_symbol}_t - \operatorname{{median}}({metric_symbol})}}{{1.4826 \cdot \operatorname{{MAD}}({metric_symbol})}}$<br>"
        rf"$\operatorname{{MAD}}({metric_symbol}) = \operatorname{{median}}(|{metric_symbol} - \operatorname{{median}}({metric_symbol})|)$<br>"
        f"<span style=\"font-size:10px\">{function_name}</span>"
    )

def add_formula_annotation(fig, text, y_position):
    fig.add_annotation(
        text=text,
        x=0.01,
        y=y_position,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        bgcolor="rgba(15, 23, 42, 0.72)",
        bordercolor="rgba(148, 163, 184, 0.34)",
        borderwidth=1,
        font=dict(size=11, color="rgba(226, 232, 240, 0.95)"),
    )

def block8_dropdown_menu(buttons, x, active=0):
    return dict(
        type="dropdown",
        buttons=buttons,
        direction="down",
        showactive=True,
        active=active,
        x=x,
        xanchor="left",
        y=1.09,
        yanchor="top",
        bgcolor="rgba(15, 23, 42, 0.92)",
        bordercolor="rgba(148, 163, 184, 0.35)",
        font=dict(color="rgba(226, 232, 240, 0.96)"),
    )

def block8_time_range_buttons(global_start, global_end, axis_count):
    def make_range(years=None):
        start = global_start if years is None else max(global_start, global_end - pd.DateOffset(years=years))
        return {
            ("xaxis.range" if axis_idx == 1 else f"xaxis{axis_idx}.range"): [start, global_end]
            for axis_idx in range(1, axis_count + 1)
        }

    return [
        dict(label="10 Years", method="relayout", args=[make_range(10)]),
        dict(label="5 Years", method="relayout", args=[make_range(5)]),
        dict(label="3 Years", method="relayout", args=[make_range(3)]),
        dict(label="1 Year", method="relayout", args=[make_range(1)]),
        dict(label="All", method="relayout", args=[make_range(None)]),
    ]

def _block8_clean_return_series(series):
    return (
        pd.to_numeric(pd.Series(series), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

if "ticker_daily_returns" not in globals():
    if "ticker_daily_data" in globals() and isinstance(ticker_daily_data, pd.DataFrame) and "Close" in ticker_daily_data:
        ticker_daily_returns = ticker_daily_data["Close"].pct_change(fill_method=None).dropna()
    elif "asset_history" in globals() and isinstance(asset_history, pd.DataFrame) and "Close" in asset_history:
        ticker_daily_returns = asset_history["Close"].pct_change(fill_method=None).dropna()
    else:
        raise ValueError("Block 8 needs ticker_daily_returns from Block 7. Run Blocks 4, 5, and 7 first.")

rolling_mean_returns = _block8_clean_return_series(ticker_daily_returns)
if rolling_mean_returns.empty:
    asset_rows = len(asset_history) if "asset_history" in globals() and isinstance(asset_history, pd.DataFrame) else 0
    daily_rows = len(ticker_daily_data) if "ticker_daily_data" in globals() and isinstance(ticker_daily_data, pd.DataFrame) else 0
    raise ValueError(
        "Block 8 has no clean daily returns. This usually means Block 5 did not retrieve usable price data, "
        "Block 7 was not rerun after changing parameters, or the Close series is empty. "
        f"asset_history rows: {asset_rows:,}; ticker_daily_data rows: {daily_rows:,}. "
        "Run Blocks 4, 5, and 7 before Block 8."
    )

available_return_count = len(rolling_mean_returns)
requested_rolling_mean_windows = list(rolling_mean_windows)
rolling_mean_windows = [window for window in rolling_mean_windows if window <= available_return_count]
skipped_rolling_mean_windows = [
    window for window in requested_rolling_mean_windows if window > available_return_count
]
if not rolling_mean_windows:
    if available_return_count < 2:
        raise ValueError(
            "Block 8 needs at least two clean daily returns for a rolling mean plot. "
            f"Only {available_return_count:,} clean return row(s) are available. "
            "Run Blocks 4, 5, and 7 and confirm the selected ticker retrieved price history."
        )
    fallback_window = min(max(2, available_return_count), preferred_rolling_mean_window)
    rolling_mean_windows = [fallback_window]
    print(
        "Block 8: requested rolling windows "
        f"{requested_rolling_mean_windows} are larger than the {available_return_count:,} "
        f"available clean daily returns. Using {fallback_window}D instead."
    )
elif skipped_rolling_mean_windows:
    print(
        "Block 8: skipping rolling windows larger than the available clean daily returns: "
        f"{skipped_rolling_mean_windows}. Available return rows: {available_return_count:,}."
    )

if rolling_mean_window not in rolling_mean_windows:
    rolling_mean_window = (
        preferred_rolling_mean_window
        if preferred_rolling_mean_window in rolling_mean_windows
        else max(rolling_mean_windows)
    )


def rolling_mean_metric_frames(daily_returns, window):
    window = int(window)
    returns = pd.Series(daily_returns).dropna().sort_index()
    arithmetic_mean = returns.rolling(
        window,
        min_periods=window,
    ).mean()
    geometric_mean = np.expm1(
        np.log1p(returns).rolling(
            window,
            min_periods=window,
        ).mean()
    )
    compounding_efficiency = compounding_efficiency_from_means(arithmetic_mean, geometric_mean)
    volatility_drag = volatility_drag_from_means(arithmetic_mean, geometric_mean)
    arithmetic_label = rolling_arithmetic_label(window)
    geometric_label = rolling_geometric_label(window)

    values = pd.concat(
        {
            arithmetic_label: arithmetic_mean,
            geometric_label: geometric_mean,
            compounding_efficiency_label: compounding_efficiency,
            volatility_drag_label: volatility_drag,
        },
        axis=1,
    ).dropna(how="all")
    mad_scores = pd.concat(
        {
            arithmetic_label: rolling_metric_mad_score(arithmetic_mean),
            geometric_label: rolling_metric_mad_score(geometric_mean),
            compounding_efficiency_label: rolling_metric_mad_score(compounding_efficiency),
            volatility_drag_label: rolling_metric_mad_score(volatility_drag),
        },
        axis=1,
    ).dropna(how="all")
    return values, mad_scores

asset_rolling_mean_detail = {}
for window in rolling_mean_windows:
    values, mad_scores = rolling_mean_metric_frames(rolling_mean_returns, window)
    if not values.empty:
        asset_rolling_mean_detail[window] = {
            "values": values,
            "mad_scores": mad_scores,
        }
if not asset_rolling_mean_detail:
    raise ValueError(
        "Block 8 could not build rolling mean data after cleaning the return series. "
        f"Clean return rows: {available_return_count:,}; requested windows: {requested_rolling_mean_windows}; "
        f"usable windows: {rolling_mean_windows}. "
        "If this is zero or unexpectedly small, rerun Blocks 4, 5, and 7 and check that data retrieval returned a non-empty Close series."
    )
if rolling_mean_window not in asset_rolling_mean_detail:
    rolling_mean_window = next(iter(asset_rolling_mean_detail))

rolling_mean_comparison = asset_rolling_mean_detail[rolling_mean_window]["values"]
rolling_mean_mad_comparison = asset_rolling_mean_detail[rolling_mean_window]["mad_scores"]

benchmark_rolling_mean_detail = {window: {} for window in asset_rolling_mean_detail}
for benchmark_symbol, benchmark_frame in benchmark_data.items():
    if isinstance(benchmark_frame, pd.DataFrame):
        if "Close" not in benchmark_frame:
            continue
        benchmark_close = benchmark_frame["Close"]
    else:
        benchmark_close = pd.Series(benchmark_frame)

    benchmark_returns = benchmark_close.pct_change(fill_method=None)
    for window in asset_rolling_mean_detail:
        benchmark_values, benchmark_mad_scores = rolling_mean_metric_frames(benchmark_returns, window)
        if benchmark_values.empty:
            continue
        benchmark_rolling_mean_detail[window][benchmark_symbol] = {
            "values": benchmark_values,
            "mad_scores": benchmark_mad_scores,
        }

benchmark_overlay_order = []
for window_payload in benchmark_rolling_mean_detail.values():
    for benchmark_symbol in window_payload:
        if benchmark_symbol not in benchmark_overlay_order:
            benchmark_overlay_order.append(benchmark_symbol)
benchmark_line_dashes = ["dot", "dash", "longdash", "dashdot"]
benchmark_line_dash_map = {
    symbol: benchmark_line_dashes[index % len(benchmark_line_dashes)]
    for index, symbol in enumerate(benchmark_overlay_order)
}

rolling_mean_subplot_count = 2

fig_rolling_mean_comparison = make_subplots(
    rows=rolling_mean_subplot_count,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.50, 0.50],
    subplot_titles=(
        f"{compounding_efficiency_label} MAD Score",
        f"{volatility_drag_label} MAD Score",
    ),
)

def block8_plot_title(window):
    return f"{ticker_str}: {int(window)}-Day Compounding Efficiency and Volatility Drag MAD Scores vs Benchmarks"

def metric_trace_specs(window):
    return [
        (compounding_efficiency_label, 1, "#A3E635", "Efficiency", ".4f", True),
        (volatility_drag_label, 2, "#E879F9", "Volatility Drag", ".2%", True),
    ]

def add_metric_trace(
    name_prefix,
    values,
    mad_scores,
    metric_label,
    row,
    color,
    raw_label,
    raw_format,
    line_dash="solid",
    line_width=2.0,
    opacity=1.0,
    visible=True,
):
    series = mad_scores.get(metric_label, pd.Series(dtype=float)).dropna()
    if series.empty:
        return

    raw_value_template = "%{customdata[0]:" + raw_format + "}"
    hovertemplate = (
        "%{x|%Y-%m-%d}<br>"
        "MAD Score: %{y:.2f}<br>"
        f"{raw_label}: " + raw_value_template + "<extra></extra>"
    )
    fig_rolling_mean_comparison.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=f"{name_prefix} {metric_label} MAD Score",
            line=dict(color=color, width=line_width, dash=line_dash),
            opacity=opacity,
            visible=visible,
            customdata=metric_detail_array(
                values.get(metric_label, pd.Series(dtype=float)),
                mad_scores.get(metric_label, pd.Series(dtype=float)),
                series.index,
            ),
            hovertemplate=hovertemplate,
        ),
        row=row,
        col=1,
    )

window_trace_indices = {}
for window, asset_payload in asset_rolling_mean_detail.items():
    window_visible = window == rolling_mean_window
    trace_start = len(fig_rolling_mean_comparison.data)
    for metric_label, row, color, raw_label, raw_format, _ in metric_trace_specs(window):
        add_metric_trace(
            name_prefix=ticker_str,
            values=asset_payload["values"],
            mad_scores=asset_payload["mad_scores"],
            metric_label=metric_label,
            row=row,
            color=color,
            raw_label=raw_label,
            raw_format=raw_format,
            visible=window_visible,
        )

    for benchmark_symbol, benchmark_payload in benchmark_rolling_mean_detail.get(window, {}).items():
        for metric_label, row, color, raw_label, raw_format, _ in metric_trace_specs(window):
            add_metric_trace(
                name_prefix=benchmark_symbol,
                values=benchmark_payload["values"],
                mad_scores=benchmark_payload["mad_scores"],
                metric_label=metric_label,
                row=row,
                color=color,
                raw_label=raw_label,
                raw_format=raw_format,
                line_dash=benchmark_line_dash_map.get(benchmark_symbol, "dot"),
                line_width=1.4,
                opacity=0.82,
                visible=window_visible,
            )
    window_trace_indices[window] = list(range(trace_start, len(fig_rolling_mean_comparison.data)))

total_trace_count = len(fig_rolling_mean_comparison.data)
base_plot_title = block8_plot_title(rolling_mean_window)

def rolling_window_visibility(window):
    active_indices = set(window_trace_indices.get(window, []))
    return [trace_index in active_indices for trace_index in range(total_trace_count)]

rolling_window_order = list(asset_rolling_mean_detail)
rolling_window_dropdown_buttons = [
    dict(
        label=f"{int(window)} Days",
        method="update",
        args=[
            {"visible": rolling_window_visibility(window)},
            {"title": header_title(block8_plot_title(window))},
        ],
    )
    for window in rolling_window_order
]
rolling_window_dropdown_active = rolling_window_order.index(rolling_mean_window)

plot_x_indexes = [payload["mad_scores"].index for payload in asset_rolling_mean_detail.values()]
for window_payload in benchmark_rolling_mean_detail.values():
    plot_x_indexes.extend(
        payload["mad_scores"].index
        for payload in window_payload.values()
        if not payload["mad_scores"].empty
    )
plot_x_indexes = [index for index in plot_x_indexes if len(index) > 0]
plot_start = min((index.min() for index in plot_x_indexes), default=None)
plot_end = max((index.max() for index in plot_x_indexes), default=None)

block8_updatemenus = [
    block8_dropdown_menu(
        rolling_window_dropdown_buttons,
        x=0.0,
        active=rolling_window_dropdown_active,
    )
]
if plot_start is not None and plot_end is not None:
    default_start = max(plot_start, plot_end - pd.DateOffset(years=10))
    for subplot_row in range(1, rolling_mean_subplot_count + 1):
        fig_rolling_mean_comparison.update_xaxes(range=[default_start, plot_end], row=subplot_row, col=1)
    block8_updatemenus.append(
        block8_dropdown_menu(
            block8_time_range_buttons(plot_start, plot_end, rolling_mean_subplot_count),
            x=0.24,
            active=0,
        )
    )

for score_row in (1, 2):
    is_efficiency_row = score_row == 1
    lower_zone_color = "rgba(34, 197, 94, 0.16)" if is_efficiency_row else "rgba(239, 68, 68, 0.16)"
    upper_zone_color = "rgba(239, 68, 68, 0.16)" if is_efficiency_row else "rgba(34, 197, 94, 0.16)"
    neutral_lower_bound = -1 if is_efficiency_row else -0.5
    neutral_upper_bound = 0.5 if is_efficiency_row else 1
    lower_zone_bounds = (-2, -1) if is_efficiency_row else (-1, -0.5)
    upper_zone_bounds = (0.5, 1) if is_efficiency_row else (1, 2)
    positive_reference_levels = (0.5, 1) if is_efficiency_row else (1, 2)
    negative_reference_levels = (1, 2) if is_efficiency_row else (0.5, 1)
    fig_rolling_mean_comparison.add_hrect(
        y0=neutral_lower_bound,
        y1=neutral_upper_bound,
        fillcolor="rgba(148, 163, 184, 0.12)",
        line_width=0,
        layer="below",
        row=score_row,
        col=1,
    )
    fig_rolling_mean_comparison.add_hrect(
        y0=lower_zone_bounds[0],
        y1=lower_zone_bounds[1],
        fillcolor=lower_zone_color,
        line_width=0,
        layer="below",
        row=score_row,
        col=1,
    )
    fig_rolling_mean_comparison.add_hrect(
        y0=upper_zone_bounds[0],
        y1=upper_zone_bounds[1],
        fillcolor=upper_zone_color,
        line_width=0,
        layer="below",
        row=score_row,
        col=1,
    )
    fig_rolling_mean_comparison.add_hline(
        y=0,
        line_dash="solid",
        line_color="rgba(226, 232, 240, 0.70)",
        row=score_row,
        col=1,
    )
    for sigma_level in positive_reference_levels:
        fig_rolling_mean_comparison.add_hline(
            y=sigma_level,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.55)",
            row=score_row,
            col=1,
        )
    for sigma_level in negative_reference_levels:
        fig_rolling_mean_comparison.add_hline(
            y=-sigma_level,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.55)",
            row=score_row,
            col=1,
        )

score_zone_annotations = [
    (1, "Efficient Compounding", 0.75, "rgba(255, 235, 235, 0.96)"),
    (1, "Poor Compounding", -1.5, "rgba(235, 255, 235, 0.96)"),
    (2, "High Drag", 1.5, "rgba(235, 255, 235, 0.96)"),
    (2, "Low Drag", -0.75, "rgba(255, 235, 235, 0.96)"),
]

for zone_row, zone_label, zone_y, zone_color in score_zone_annotations:
    fig_rolling_mean_comparison.add_annotation(
        x=0.5,
        y=zone_y,
        xref="x domain",
        yref="y",
        text=zone_label,
        showarrow=False,
        xanchor="center",
        yanchor="middle",
        font=dict(color=zone_color, size=14),
        row=zone_row,
        col=1,
    )

add_formula_annotation(
    fig_rolling_mean_comparison,
    mad_score_annotation_text(
        compounding_efficiency_label,
        "CE",
        r"\frac{1 + g_t}{1 + a_t}",
        "compounding_efficiency_from_means(arithmetic_mean, geometric_mean)",
    ),
    y_position=0.86,
)
add_formula_annotation(
    fig_rolling_mean_comparison,
    mad_score_annotation_text(
        volatility_drag_label,
        "VD",
        r"a_t - g_t",
        "volatility_drag_from_means(arithmetic_mean, geometric_mean)",
    ),
    y_position=0.38,
)

fig_rolling_mean_comparison.update_layout(
    title=header_title(base_plot_title),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=header_margin(top=165),
    updatemenus=block8_updatemenus,
    height=1125,
)
fig_rolling_mean_comparison.update_yaxes(title_text="MAD Score", tickformat=".2f", range=[-6, 2], row=1, col=1)
fig_rolling_mean_comparison.update_yaxes(title_text="MAD Score", tickformat=".2f", range=[-2, 6], row=2, col=1)
fig_rolling_mean_comparison.update_xaxes(title_text="Date", row=2, col=1)
fig_rolling_mean_comparison = finalize_dark_figure(fig_rolling_mean_comparison)
# Block 8 rows are rendered inside Block 18.


# %% [notebook block 11]
# Block 10: stack candlestick, drawdown comparison, and rolling recovery time
# Change selected_time_frames in Block 4 to a list like [21, 50, 200], then rerun this cell.

drawdown_recovery_by_window = {}

for window in selected_time_frames:

    rolling_peak = compute.rolling(
        asset_history['Close'],
        metric=pd.Series.max,
        window=window,
        min_periods=1,
    )

    underwater_series = asset_history['Close'].div(rolling_peak).sub(1.0).dropna()

    drawdown_series = compute.rolling(
        asset_history['Close'],
        metric=metric.textbook_window_drawdown,
        window=window,
        dropna=False,
    ).dropna()

    recovery_series = compute.rolling(
        asset_history['Close'],
        metric=metric.window_recovery_time,
        window=window,
        dropna=False,
    ).dropna()

    drawdown_recovery_by_window[window] = {
        'underwater': underwater_series,
        'max_drawdown': drawdown_series,
        'recovery_time': recovery_series,
    }

fig_block10_drawdown_recovery = plot_candlestick_drawdown_recovery_view(
    price_frame=asset_history,
    drawdown_recovery_by_window=drawdown_recovery_by_window,
    ticker_label=ticker_str,
    candlestick_period=period,
    default_timeframe_label='10 Years',
)
fig = fig_block10_drawdown_recovery


# %% [notebook block 12]
# Block 13: visualize monthly and quarterly seasonality patterns

ticker_quarterly_data = series_transforms.resample(asset_history, frequency="quarterly")
ticker_quarterly_returns = ticker_quarterly_data["Close"].pct_change(fill_method=None).dropna()


def _seasonality_metric_series(frequency, metric_type):
    """Calculate one selected metric for every calendar month or quarter."""
    metric_type = normalize_momentum_ratio_type(metric_type)
    daily_returns = asset_history["Close"].pct_change(fill_method=None)
    aligned_risk_free = risk_free_daily_rate.reindex(daily_returns.index).ffill().bfill()
    if _is_correlation_metric(metric_type):
        benchmark_symbol = _correlation_metric_symbol(metric_type)
        benchmark_frame = benchmark_data.get(benchmark_symbol)
        if benchmark_frame is None or "Close" not in benchmark_frame:
            raise ValueError(f"No benchmark Close history is available for {benchmark_symbol}.")
        benchmark_returns = benchmark_frame["Close"].pct_change(fill_method=None)
        observations = pd.concat(
            [daily_returns.rename("asset"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
    else:
        benchmark_returns = None
        observations = (
        daily_returns
        if metric_type in {"return", "volatility", "downside_volatility"}
        else daily_returns - aligned_risk_free
        )

    def period_metric(values):
        if _is_correlation_metric(metric_type):
            values = pd.DataFrame(values).dropna()
            return values["asset"].corr(values["benchmark"]) if len(values) >= 2 else np.nan
        values = pd.Series(values).dropna()
        if values.empty:
            return np.nan
        if metric_type == "return":
            return (1.0 + values).prod() - 1.0
        if len(values) < 2:
            return np.nan
        if metric_type == "volatility":
            return np.sqrt(annualization_factor) * values.std()
        if metric_type == "downside_volatility":
            return np.sqrt(
                annualization_factor * values.where(values < 0, 0.0).pow(2).mean()
            )
        mean_return = values.mean()
        if metric_type == "sharpe":
            denominator = values.std()
        else:
            denominator = values.where(values < 0, 0.0).pow(2).mean() ** 0.5
        return np.sqrt(annualization_factor) * mean_return / denominator if denominator > 0 else np.nan

    if _is_correlation_metric(metric_type):
        return observations.resample(frequency).apply(period_metric).dropna()
    return observations.resample(frequency).apply(period_metric).dropna()


def _block18_seasonality_figure(metric_type="sharpe"):
    metric_type = normalize_momentum_ratio_type(metric_type)
    metric_label = momentum_ratio_label(metric_type)
    return plot_seasonality_stack_view(
        monthly_returns=_seasonality_metric_series("ME", metric_type),
        quarterly_returns=_seasonality_metric_series("QE", metric_type),
        ticker_label=ticker_str,
        as_of=asset_history.index.max(),
        metric_label=metric_label,
    )

fig_ticker_seasonality_stack = plot_seasonality_stack_view(
    monthly_returns=ticker_monthly_returns,
    quarterly_returns=ticker_quarterly_returns,
    ticker_label=ticker_str,
    as_of=asset_history.index.max(),
)


# %% [notebook block 13]
# Block 14: compute Sharpe/Sortino ratios and spreads

from Quantapp.analytics.series_utils import calculate_zscore

asset_close = asset_history['Close'].dropna().sort_index()
annualization_factor = 252
risk_time_frame_map = {str(term): int(window) for term, window in time_frame_map.items()}
selected_windows = sorted(dict.fromkeys(int(window) for window in selected_time_frames))
selected_time_frame_map = {}

for window in selected_windows:
    term_key = next(
        (term for term, mapped_window in risk_time_frame_map.items() if mapped_window == window),
        f"selected_{window}",
    )
    selected_time_frame_map[term_key] = window

risk_time_frame_map.update(selected_time_frame_map)

def rolling_ratio_series(close, window, ratio_type):
    benchmark_prices = None
    normalized_ratio_type = normalize_momentum_ratio_type(ratio_type)
    if _is_correlation_metric(normalized_ratio_type):
        benchmark_symbol = _correlation_metric_symbol(normalized_ratio_type)
        benchmark_prices = benchmark_data[benchmark_symbol]["Close"]
    elif _is_benchmark_relative_metric(normalized_ratio_type):
        _, benchmark_symbol = _benchmark_relative_metric_parts(normalized_ratio_type)
        benchmark_prices = benchmark_data[benchmark_symbol]["Close"]
    ratio_frame = risk_adjusted_returns(
        close.dropna().sort_index(),
        windows=[window],
        ratio_type=normalized_ratio_type,
        risk_free_rate=risk_free_daily_rate,
        benchmark_prices=benchmark_prices,
    )
    return ratio_frame.iloc[:, 0]

def rolling_risk_components(close, window):
    close = close.dropna().sort_index()
    if isinstance(risk_free_daily_rate, pd.Series):
        periodic_risk_free_rate = risk_free_daily_rate.astype(float).sort_index().reindex(close.index).ffill()
    else:
        periodic_risk_free_rate = (1.0 + float(risk_free_daily_rate)) ** (1.0 / annualization_factor) - 1.0
    excess_returns = close.pct_change() - periodic_risk_free_rate
    rolling_mean = excess_returns.rolling(window).mean()
    rolling_std = excess_returns.rolling(window).std()
    sharpe_ratio = np.sqrt(annualization_factor) * rolling_mean / rolling_std
    return {
        "annualized_excess_return": annualization_factor * rolling_mean,
        "annualized_volatility": np.sqrt(annualization_factor) * rolling_std,
        "sharpe_ratio": sharpe_ratio.where(rolling_std > 0).replace([np.inf, -np.inf], np.nan),
    }

def zscore_or_empty(series):
    clean = pd.Series(series).dropna()
    return calculate_zscore(clean).dropna() if not clean.empty else pd.Series(dtype=float)

asset_sharpe_map = {}
asset_sortino_map = {}
asset_component_map = {}

for term, window in risk_time_frame_map.items():
    asset_sharpe_map[term] = rolling_ratio_series(asset_close, window, "sharpe")
    asset_sortino_map[term] = rolling_ratio_series(asset_close, window, "sortino")
    asset_component_map[term] = rolling_risk_components(asset_close, window)

asset_sharpe_sortino_spread_map = {
    term: asset_sortino_map[term] - asset_sharpe_map[term]
    for term in risk_time_frame_map
}

benchmark_metrics = {}
for symbol, benchmark_frame in benchmark_data.items():
    benchmark_close = benchmark_frame["Close"] if isinstance(benchmark_frame, pd.DataFrame) else benchmark_frame
    benchmark_close = benchmark_close.dropna().sort_index()
    benchmark_metrics[symbol] = {}

    for term, window in risk_time_frame_map.items():
        benchmark_components = rolling_risk_components(benchmark_close, window)
        benchmark_sharpe = benchmark_components["sharpe_ratio"]
        benchmark_metrics[symbol][term] = {
            "spread": benchmark_close.pct_change(window) - asset_close.pct_change(window),
            "annualized_excess_return": benchmark_components["annualized_excess_return"],
            "annualized_volatility": benchmark_components["annualized_volatility"],
            "sharpe_ratio": benchmark_sharpe,
            "sharpe_spread": benchmark_sharpe - asset_sharpe_map[term],
        }

benchmark_order = list(benchmark_metrics)
default_benchmark = "SPY" if "SPY" in benchmark_order else (benchmark_order[0] if benchmark_order else None)
spread_plot_data = {
    term: {symbol: benchmark_metrics[symbol][term]["sharpe_spread"] for symbol in benchmark_order}
    for term in risk_time_frame_map
}

term_config_map = {}
for term, window in risk_time_frame_map.items():
    label = f"{window}-day"
    sharpe = asset_sharpe_map[term]
    sortino = asset_sortino_map[term]
    spread = asset_sharpe_sortino_spread_map[term]
    term_config_map[label] = {
        "sharpe": sharpe,
        "sortino": sortino,
        "spread": spread,
        "sharpe_zscore": zscore_or_empty(sharpe),
        "sortino_zscore": zscore_or_empty(sortino),
        "spread_zscore": zscore_or_empty(spread),
        "time_frame": window,
        "term_key": term,
    }

selected_term_config_map = {
    f"{window}-day": term_config_map[f"{window}-day"]
    for window in selected_time_frame_map.values()
    if f"{window}-day" in term_config_map
}


# %% [notebook block 14]
# Block 15: plot rolling correlation of the asset versus benchmarks

asset_daily_returns = asset_history['Close'].pct_change(fill_method=None)
correlation_term_order = [term for term in time_frame_map if time_frame_map.get(term) is not None]
correlation_benchmark_order = benchmark_order if benchmark_order else list(benchmark_data.keys())
rolling_correlation_map = {}

for term in correlation_term_order:
    window = int(time_frame_map[term])
    term_series_map = {}

    for symbol in correlation_benchmark_order:
        benchmark_frame = benchmark_data.get(symbol)
        if benchmark_frame is None or 'Close' not in benchmark_frame:
            continue

        benchmark_daily_returns = benchmark_frame['Close'].pct_change(fill_method=None)
        aligned_returns = pd.concat(
            [
                asset_daily_returns.rename('asset'),
                benchmark_daily_returns.rename(symbol),
            ],
            axis=1,
        ).dropna()
        if aligned_returns.empty:
            continue

        rolling_correlation_series = aligned_returns['asset'].rolling(window).corr(aligned_returns[symbol]).dropna()
        if rolling_correlation_series.empty:
            continue

        term_series_map[symbol] = rolling_correlation_series

    if term_series_map:
        rolling_correlation_map[term] = term_series_map

rolling_correlation_fig = plot_rolling_correlation_view(
    rolling_correlation_map=rolling_correlation_map,
    time_frame_map=time_frame_map,
    term_order=correlation_term_order,
    benchmark_order=correlation_benchmark_order,
    ticker_label=ticker_str,
)


# %% [notebook block 15]
# Block 16: Upside Efficiency
# Change selected_time_frames in Block 4 to a list like [21, 50, 200], then rerun the notebook.

fig_block16_upside_efficiency = plot_sharpe_sortino_comparison(
    term_config_map=selected_term_config_map,
    ticker_label=ticker_str,
)
fig = fig_block16_upside_efficiency


# %% [notebook block 16]
# Block 18: combine risk-adjusted return and benchmark plots
# Requires the current kernel session to have fresh outputs from Blocks 2, 7, and 14.
# Enter the rolling window below, then press Update to rebuild the decomposition.

import copy
import socket

from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

# Keep the notebook dashboard visually consistent without depending on a web-font
# download. Inter is used when installed, with modern native UI fonts as fallbacks.
BLOCK18_FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def _block18_apply_typography(figure):
    """Apply the dashboard typeface to all Plotly-rendered text."""
    figure.update_layout(font={"family": BLOCK18_FONT_FAMILY})
    return figure
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import BSpline, UnivariateSpline
from scipy.signal import find_peaks
from scipy.stats import norm as scipy_normal
from statsmodels.tsa.stattools import acf as statsmodels_acf
try:
    from arch import arch_model as block18_arch_model
except ImportError:
    block18_arch_model = None
from Quantapp.analytics import compute
from Quantapp.analytics.series_utils import (
    calculate_textbook_rolling_max_drawdown,
    coerce_close_series,
    gini_coefficient,
)
from Quantapp.visualization.views.single_asset_profile.pricing.distribution import (
    plot_distribution_shape_zscores_view,
)
from Quantapp.visualization.views.single_asset_profile.pricing.momentum_efficiency._shared import (
    finalize_dark_figure,
    header_margin,
    header_title,
    trace_datetime_bounds,
)

def benchmark_zscore_for_plot(series):
    clean = pd.Series(series).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    zscore_series = calculate_zscore(clean)
    if zscore_series.isna().all():
        return pd.Series(0.0, index=clean.index)
    return zscore_series.dropna()

def _block18_close_series(frame_or_series, label):
    if isinstance(frame_or_series, pd.DataFrame):
        if "Close" not in frame_or_series:
            raise ValueError(f"{label} is missing a Close column.")
        close = frame_or_series["Close"]
    else:
        close = pd.Series(frame_or_series)
    close = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if close.empty:
        raise ValueError(f"{label} has no usable Close values.")
    return close

def _block18_validate_window(value):
    try:
        window = int(value)
    except (TypeError, ValueError):
        raise ValueError("Enter a whole-number rolling window, for example 21, 50, or 200.")
    if window < 2:
        raise ValueError("The Block 18 rolling window must be at least 2 days.")
    return window

def build_block18_decomposition_figure(
    window, benchmark_symbols=None, ratio_type="sharpe"
):
    window = _block18_validate_window(window)
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    requested_benchmark_symbols = [
        symbol for symbol in (benchmark_symbols or []) if symbol in benchmark_order
    ]
    if benchmark_symbols is not None and not requested_benchmark_symbols:
        return _block18_asset_only_risk_figure(window, ratio_type)
    if not benchmark_order:
        raise ValueError("No benchmark data available for benchmark comparison plots.")
    if not requested_benchmark_symbols:
        requested_benchmark_symbols = [default_benchmark or benchmark_order[0]]
    benchmark_symbol = requested_benchmark_symbols[0]

    asset_close_for_window = _block18_close_series(asset_history, ticker_str)
    minimum_rows = window + 2
    if len(asset_close_for_window) < minimum_rows:
        raise ValueError(
            f"{ticker_str} only has {len(asset_close_for_window):,} price rows; "
            f"Block 18 needs at least {minimum_rows:,} rows for a {window}-day window."
        )
    term_key = "custom"
    block18_time_frame_map = {term_key: window}
    asset_ratio = rolling_ratio_series(asset_close_for_window, window, ratio_type)
    asset_components = rolling_risk_components(asset_close_for_window, window)
    benchmark_detail_payloads = {}
    skipped_benchmark_details = {}
    for candidate_symbol in requested_benchmark_symbols:
        try:
            candidate_close = _block18_close_series(benchmark_data.get(candidate_symbol), candidate_symbol)
        except ValueError as error:
            skipped_benchmark_details[candidate_symbol] = str(error)
            continue
        if len(candidate_close) < minimum_rows:
            skipped_benchmark_details[candidate_symbol] = (
                f"only {len(candidate_close):,} rows; needs {minimum_rows:,}"
            )
            continue

        candidate_components = rolling_risk_components(candidate_close, window)
        candidate_ratio = rolling_ratio_series(candidate_close, window, ratio_type)
        if _is_benchmark_relative_metric(ratio_type):
            metric_name, reference_symbol = _benchmark_relative_metric_parts(ratio_type)
            if candidate_symbol == reference_symbol and metric_name in {"appraisal", "information"}:
                # The reference benchmark has zero active return/alpha by
                # definition. Represent that baseline explicitly instead of
                # retaining the undefined 0/0 ratio.
                candidate_ratio = pd.Series(0.0, index=candidate_close.index)
        benchmark_detail_payloads[candidate_symbol] = {
            "asset": benchmark_zscore_for_plot(asset_ratio),
            "benchmark": benchmark_zscore_for_plot(candidate_ratio),
            "asset_ratio": asset_ratio.dropna(),
            "benchmark_ratio": candidate_ratio.dropna(),
            "asset_excess_return": asset_components.get("annualized_excess_return", pd.Series(dtype=float)).dropna(),
            "benchmark_excess_return": candidate_components.get("annualized_excess_return", pd.Series(dtype=float)).dropna(),
            "asset_volatility": asset_components.get("annualized_volatility", pd.Series(dtype=float)).dropna(),
            "benchmark_volatility": candidate_components.get("annualized_volatility", pd.Series(dtype=float)).dropna(),
            "ratio_spread": benchmark_zscore_for_plot(candidate_ratio - asset_ratio),
            "relative_spread": benchmark_zscore_for_plot(
                candidate_close.pct_change(window) - asset_close_for_window.pct_change(window)
            ),
        }

    plotted_benchmark_symbols = [
        symbol for symbol in requested_benchmark_symbols if symbol in benchmark_detail_payloads
    ]
    if not plotted_benchmark_symbols:
        raise ValueError("No benchmark has enough usable history for the selected rolling window.")
    if benchmark_symbol not in benchmark_detail_payloads:
        benchmark_symbol = plotted_benchmark_symbols[0]
    if skipped_benchmark_details:
        print(
            "Block 18 skipped benchmark overlays: "
            + "; ".join(f"{symbol}: {reason}" for symbol, reason in skipped_benchmark_details.items())
        )

    detail_zscore_map = {
        benchmark_symbol: {term_key: benchmark_detail_payloads[benchmark_symbol]}
    }

    detail_fig = plot_benchmark_zscore_detail(
        detail_zscore_map=detail_zscore_map,
        benchmark_order=[benchmark_symbol],
        time_frame_map=block18_time_frame_map,
        ticker_label=ticker_str,
        default_benchmark=benchmark_symbol,
        default_term=term_key,
        ratio_label=ratio_label,
    )

    overlay_colors = ["#f97316", "#a78bfa", "#14b8a6", "#facc15", "#fb7185"]
    overlay_dashes = ["dash", "dot", "longdash", "dashdot"]
    for overlay_index, overlay_symbol in enumerate(plotted_benchmark_symbols):
        if overlay_symbol == benchmark_symbol:
            continue
        overlay_payload = benchmark_detail_payloads[overlay_symbol]
        overlay_color = overlay_colors[overlay_index % len(overlay_colors)]
        overlay_dash = overlay_dashes[overlay_index % len(overlay_dashes)]
        overlay_zscore = overlay_payload["benchmark"].dropna()
        if not overlay_zscore.empty:
            detail_fig.add_trace(
                go.Scatter(
                    x=overlay_zscore.index,
                    y=overlay_zscore,
                    mode="lines",
                    name=f"{overlay_symbol} {window}-Day {ratio_label} Z-Score",
                    legendgroup=f"{overlay_symbol}-{term_key}",
                    line=dict(color=overlay_color, dash=overlay_dash, width=2.2),
                    showlegend=True,
                    hovertemplate=(
                        f"{overlay_symbol}<br>%{{x|%Y-%m-%d}}<br>"
                        f"{ratio_label} Z-Score: %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        overlay_spread = overlay_payload["ratio_spread"].dropna()
        if not overlay_spread.empty:
            detail_fig.add_trace(
                go.Scatter(
                    x=overlay_spread.index,
                    y=overlay_spread,
                    mode="lines",
                    name=(
                        f"{overlay_symbol} - {ticker_str} {window}-Day "
                        f"{ratio_label} Spread Z-Score"
                    ),
                    legendgroup=f"{overlay_symbol}-{term_key}",
                    line=dict(color=overlay_color, dash=overlay_dash, width=2.0),
                    showlegend=False,
                    hovertemplate=(
                        f"{overlay_symbol} - {ticker_str}<br>%{{x|%Y-%m-%d}}<br>Spread Z-Score: %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
    detail_fig.update_layout(updatemenus=[])
    rolling_mean_rows_fig = build_block18_rolling_mean_rows_figure(
        window, plotted_benchmark_symbols
    )
    return _block18_combine_detail_and_mad_rows(
        detail_fig,
        rolling_mean_rows_fig,
        window,
        plotted_benchmark_symbols,
        ratio_label=ratio_label,
    )

def _block18_rolling_metric_mad_score(series):
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    median = clean.median()
    mad = (clean - median).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(0.0, index=clean.index)
    return ((clean - median) / (1.4826 * mad)).dropna()

def _block18_metric_detail_array(value_series, mad_score_series, index):
    detail_frame = pd.concat(
        {
            "value": pd.Series(value_series),
            "mad_score": pd.Series(mad_score_series),
        },
        axis=1,
    ).reindex(index)
    return detail_frame[["value", "mad_score"]].to_numpy()

def _block18_rolling_mean_metric_frames(close_or_returns, window, *, values_are_returns=False):
    window = _block18_validate_window(window)
    series = pd.to_numeric(pd.Series(close_or_returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    returns = series if values_are_returns else series.pct_change(fill_method=None)
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(returns) < window:
        raise ValueError(f"Need at least {window:,} clean return rows; only {len(returns):,} are available.")
    arithmetic_mean = returns.rolling(window, min_periods=window).mean()
    geometric_mean = np.expm1(np.log1p(returns).rolling(window, min_periods=window).mean())
    compounding_efficiency = (1.0 + geometric_mean).div(1.0 + arithmetic_mean).replace([np.inf, -np.inf], np.nan)
    volatility_drag = arithmetic_mean - geometric_mean
    values = pd.concat(
        {
            "Compounding Efficiency": compounding_efficiency,
            "Volatility Drag": volatility_drag,
        },
        axis=1,
    ).dropna(how="all")
    mad_scores = pd.concat(
        {
            "Compounding Efficiency": _block18_rolling_metric_mad_score(compounding_efficiency),
            "Volatility Drag": _block18_rolling_metric_mad_score(volatility_drag),
        },
        axis=1,
    ).dropna(how="all")
    return values, mad_scores

def _block18_add_mad_trace(fig, name_prefix, values, mad_scores, metric_label, row, color, raw_label, raw_format, *, line_dash="solid", line_width=2.0, opacity=1.0):
    series = mad_scores.get(metric_label, pd.Series(dtype=float)).dropna()
    if series.empty:
        return
    raw_value_template = "%{customdata[0]:" + raw_format + "}"
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=f"{name_prefix} {metric_label} MAD Score",
            line=dict(color=color, width=line_width, dash=line_dash),
            opacity=opacity,
            customdata=_block18_metric_detail_array(
                values.get(metric_label, pd.Series(dtype=float)),
                mad_scores.get(metric_label, pd.Series(dtype=float)),
                series.index,
            ),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "MAD Score: %{y:.2f}<br>"
                f"{raw_label}: " + raw_value_template + "<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )

def _block18_add_mad_score_guides(fig):
    for score_row in (1, 2):
        is_efficiency_row = score_row == 1
        lower_zone_color = "rgba(34, 197, 94, 0.16)" if is_efficiency_row else "rgba(239, 68, 68, 0.16)"
        upper_zone_color = "rgba(239, 68, 68, 0.16)" if is_efficiency_row else "rgba(34, 197, 94, 0.16)"
        neutral_lower_bound = -1 if is_efficiency_row else -0.5
        neutral_upper_bound = 0.5 if is_efficiency_row else 1
        lower_zone_bounds = (-2, -1) if is_efficiency_row else (-1, -0.5)
        upper_zone_bounds = (0.5, 1) if is_efficiency_row else (1, 2)
        positive_reference_levels = (0.5, 1) if is_efficiency_row else (1, 2)
        negative_reference_levels = (1, 2) if is_efficiency_row else (0.5, 1)
        fig.add_hrect(y0=neutral_lower_bound, y1=neutral_upper_bound, fillcolor="rgba(148, 163, 184, 0.12)", line_width=0, layer="below", row=score_row, col=1)
        fig.add_hrect(y0=lower_zone_bounds[0], y1=lower_zone_bounds[1], fillcolor=lower_zone_color, line_width=0, layer="below", row=score_row, col=1)
        fig.add_hrect(y0=upper_zone_bounds[0], y1=upper_zone_bounds[1], fillcolor=upper_zone_color, line_width=0, layer="below", row=score_row, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(226, 232, 240, 0.70)", row=score_row, col=1)
        for sigma_level in positive_reference_levels:
            fig.add_hline(y=sigma_level, line_dash="dash", line_color="rgba(148, 163, 184, 0.55)", row=score_row, col=1)
        for sigma_level in negative_reference_levels:
            fig.add_hline(y=-sigma_level, line_dash="dash", line_color="rgba(148, 163, 184, 0.55)", row=score_row, col=1)

    for zone_row, zone_label, zone_y, zone_color in [
        (1, "Efficient Compounding", 0.75, "rgba(255, 235, 235, 0.96)"),
        (1, "Poor Compounding", -1.5, "rgba(235, 255, 235, 0.96)"),
        (2, "High Drag", 1.5, "rgba(235, 255, 235, 0.96)"),
        (2, "Low Drag", -0.75, "rgba(255, 235, 235, 0.96)"),
    ]:
        fig.add_annotation(
            x=0.5,
            y=zone_y,
            xref="x domain",
            yref="y",
            text=zone_label,
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(color=zone_color, size=13),
            row=zone_row,
            col=1,
        )

def build_block18_rolling_mean_rows_figure(window, benchmark_symbols=None):
    window = _block18_validate_window(window)
    benchmark_candidates = benchmark_order if benchmark_symbols is None else benchmark_symbols
    benchmark_symbols = [
        symbol for symbol in benchmark_candidates if symbol in benchmark_data
    ]
    if "ticker_daily_returns" in globals():
        asset_values, asset_mad_scores = _block18_rolling_mean_metric_frames(ticker_daily_returns, window, values_are_returns=True)
    else:
        asset_values, asset_mad_scores = _block18_rolling_mean_metric_frames(_block18_close_series(asset_history, ticker_str), window)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.50, 0.50],
        subplot_titles=("Compounding Efficiency MAD Score", "Volatility Drag MAD Score"),
    )
    metric_specs = [
        ("Compounding Efficiency", 1, "#A3E635", "Efficiency", ".4f"),
        ("Volatility Drag", 2, "#E879F9", "Volatility Drag", ".2%"),
    ]
    for metric_label, row, color, raw_label, raw_format in metric_specs:
        _block18_add_mad_trace(fig, ticker_str, asset_values, asset_mad_scores, metric_label, row, color, raw_label, raw_format)

    benchmark_line_dashes = ["dot", "dash", "longdash", "dashdot"]
    for benchmark_index, benchmark_symbol in enumerate(benchmark_symbols):
        benchmark_frame = benchmark_data[benchmark_symbol]
        if isinstance(benchmark_frame, pd.DataFrame) and "Close" in benchmark_frame:
            benchmark_close = benchmark_frame["Close"]
        else:
            benchmark_close = pd.Series(benchmark_frame)
        benchmark_values, benchmark_mad_scores = _block18_rolling_mean_metric_frames(benchmark_close, window)
        line_dash = benchmark_line_dashes[benchmark_index % len(benchmark_line_dashes)]
        for metric_label, row, color, raw_label, raw_format in metric_specs:
            _block18_add_mad_trace(
                fig,
                benchmark_symbol,
                benchmark_values,
                benchmark_mad_scores,
                metric_label,
                row,
                color,
                raw_label,
                raw_format,
                line_dash=line_dash,
                line_width=1.4,
                opacity=0.82,
            )

    plot_indexes = [trace.x for trace in fig.data if getattr(trace, "x", None) is not None and len(trace.x) > 0]
    plot_start = min((pd.Index(index).min() for index in plot_indexes), default=None)
    plot_end = max((pd.Index(index).max() for index in plot_indexes), default=None)
    if plot_start is not None and plot_end is not None:
        default_start = max(plot_start, plot_end - pd.DateOffset(years=10))
        for subplot_row in (1, 2):
            fig.update_xaxes(range=[default_start, plot_end], row=subplot_row, col=1)

    _block18_add_mad_score_guides(fig)
    fig.update_layout(
        title=header_title(f"{ticker_str}: {window}-Day Compounding Efficiency and Volatility Drag MAD Scores vs Benchmarks"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=header_margin(top=120),
        height=760,
    )
    fig.update_yaxes(title_text="MAD Score", tickformat=".2f", range=[-6, 2], row=1, col=1)
    fig.update_yaxes(title_text="MAD Score", tickformat=".2f", range=[-2, 6], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return finalize_dark_figure(fig)

def _block18_axis_ref_row(axis_ref):
    if axis_ref is None:
        return 1
    token = str(axis_ref).split()[0]
    if len(token) == 1:
        return 1
    try:
        return int(token[1:])
    except ValueError:
        return 1

def _block18_axis_ref(axis_letter, row, *, domain=False):
    suffix = "" if row == 1 else str(row)
    axis_ref = f"{axis_letter}{suffix}"
    return f"{axis_ref} domain" if domain else axis_ref

def _block18_target_row(source_row, row_offset, row_map):
    if row_map is None:
        return source_row + row_offset
    return row_map.get(source_row)

def _block18_remap_axis_ref(axis_ref, row_offset, row_map=None):
    if not isinstance(axis_ref, str):
        return axis_ref
    token = axis_ref.split()[0]
    if not token or token[0] not in {"x", "y"}:
        return axis_ref
    domain = axis_ref.endswith(" domain")
    target_row = _block18_target_row(_block18_axis_ref_row(token), row_offset, row_map)
    if target_row is None:
        return None
    return _block18_axis_ref(token[0], target_row, domain=domain)

def _block18_copy_subplot_traces(source_fig, target_fig, *, row_offset=0, row_map=None):
    for trace in source_fig.data:
        source_row = _block18_axis_ref_row(getattr(trace, "yaxis", None))
        target_row = _block18_target_row(source_row, row_offset, row_map)
        if target_row is None:
            continue
        target_fig.add_trace(copy.deepcopy(trace), row=target_row, col=1)

def _block18_copy_axis_annotations(source_fig, target_fig, *, row_offset=0, row_map=None):
    copied_annotations = []
    for annotation in source_fig.layout.annotations or []:
        annotation_payload = annotation.to_plotly_json()
        if annotation_payload.get("yref") == "paper":
            continue
        annotation_payload["xref"] = _block18_remap_axis_ref(annotation_payload.get("xref"), row_offset, row_map)
        annotation_payload["yref"] = _block18_remap_axis_ref(annotation_payload.get("yref"), row_offset, row_map)
        if annotation_payload["xref"] is None or annotation_payload["yref"] is None:
            continue
        copied_annotations.append(annotation_payload)
    if copied_annotations:
        existing_annotations = [
            annotation.to_plotly_json()
            for annotation in (target_fig.layout.annotations or [])
        ]
        target_fig.update_layout(
            annotations=existing_annotations + copied_annotations
        )

def _block18_copy_axis_shapes(source_fig, target_fig, *, row_offset=0, row_map=None):
    copied_shapes = []
    for shape in source_fig.layout.shapes or []:
        shape_payload = shape.to_plotly_json()
        shape_payload["xref"] = _block18_remap_axis_ref(shape_payload.get("xref"), row_offset, row_map)
        shape_payload["yref"] = _block18_remap_axis_ref(shape_payload.get("yref"), row_offset, row_map)
        if shape_payload["xref"] is None or shape_payload["yref"] is None:
            continue
        copied_shapes.append(shape_payload)
    if copied_shapes:
        existing_shapes = [
            shape.to_plotly_json() for shape in (target_fig.layout.shapes or [])
        ]
        target_fig.update_layout(shapes=existing_shapes + copied_shapes)


def _block18_rehome_trace(trace, target_fig, row, col):
    moved = copy.deepcopy(trace)
    moved.xaxis = None
    moved.yaxis = None
    target_fig.add_trace(moved, row=row, col=col)


def _block18_functional_horizon_summary_figure(
    source_fig, ratio_label=None, *, ratio_type=None
):
    """Extract the retained ratio/volatility summary row above the controls."""
    ratio_label = str(
        ratio_label or _block18_ratio_label(ratio_type or "sharpe")
    )
    summary = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Current vs Mean/Median {ratio_label} by Window",
            "Mean vs Median Volatility by Window",
        ),
        horizontal_spacing=0.09,
    )
    axis_targets = {"y3": (1, 1), "y4": (1, 2)}
    for trace in source_fig.data:
        target = axis_targets.get(getattr(trace, "yaxis", None) or "y")
        if target is not None:
            _block18_rehome_trace(trace, summary, *target)
    summary.update_xaxes(title_text="Momentum Window Size (Days)", row=1, col=1)
    summary.update_xaxes(title_text="Momentum Window Size (Days)", row=1, col=2)
    summary.update_yaxes(title_text=f"{ratio_label} Ratio", row=1, col=1)
    summary.update_yaxes(title_text="Annualized Volatility", row=1, col=2)
    summary.update_layout(
        title=f"{ticker_str} {ratio_label} & Volatility by Momentum Window",
        template="plotly_dark", height=480,
        margin=dict(t=110, r=35, b=55, l=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _block18_apply_typography(summary)


def _block18_functional_horizon_profile_figure(
    source_fig, horizon_range=None, ratio_label=None, *, ratio_type=None
):
    """Extract the selected ratio profile and derivative rows for display."""
    ratio_label = str(
        ratio_label or _block18_ratio_label(ratio_type or "sharpe")
    )
    profile = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        subplot_titles=(
            f"Current {ratio_label} Z-Score by Window",
            f"First Horizon Derivative of Current {ratio_label} Z-Score",
            f"Second Horizon Derivative of Current {ratio_label} Z-Score",
        ),
        row_heights=[0.52, 0.24, 0.24],
    )
    _block18_copy_subplot_traces(source_fig, profile, row_map={5: 1, 6: 2, 7: 3})
    _block18_copy_axis_shapes(source_fig, profile, row_map={5: 1, 6: 2, 7: 3})
    _block18_copy_axis_annotations(source_fig, profile, row_map={5: 1, 6: 2, 7: 3})
    profile.update_xaxes(title_text="Momentum Window Size (Days)", row=3, col=1)
    profile.update_yaxes(title_text=f"{ratio_label} Z-Score", row=1, col=1)
    profile.update_yaxes(title_text="Slope (Z / 20d)", row=2, col=1)
    profile.update_yaxes(title_text="Curvature (Z / 20d²)", row=3, col=1)
    selected_horizon_range = (
        _block18_normalize_diagnostics_window_range(horizon_range)
        if horizon_range is not None
        else source_fig.layout.xaxis5.range
    )
    if selected_horizon_range is not None:
        for profile_row in (1, 2, 3):
            profile.update_xaxes(
                range=list(selected_horizon_range),
                autorange=False,
                row=profile_row,
                col=1,
            )
    source_layout = source_fig.layout
    for source_axis, target_axis in (("yaxis5", "yaxis"), ("yaxis6", "yaxis2"), ("yaxis7", "yaxis3")):
        source_range = getattr(source_layout, source_axis).range
        if source_range is not None:
            profile.layout[target_axis].range = source_range
    profile.update_layout(
        title=f"{ticker_str} {ratio_label} Functional Horizon Profile",
        template="plotly_dark", height=1120,
        margin=dict(t=90, r=35, b=55, l=75),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _block18_apply_typography(profile)

def _block18_combine_detail_and_mad_rows(
    detail_fig, mad_rows_fig, window, benchmark_symbols, *, ratio_label="Sharpe"
):
    benchmark_label = " + ".join(benchmark_symbols)
    combined_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.36, 0.29, 0.35],
        subplot_titles=(
            "Risk-Adjusted Return Z-Score Comparison",
            f"{ratio_label} Spread Z-Score",
            "Volatility Drag MAD Score",
        ),
    )
    _block18_copy_subplot_traces(detail_fig, combined_fig, row_map={1: 1, 2: 2})
    _block18_copy_axis_annotations(detail_fig, combined_fig, row_map={1: 1, 2: 2})
    _block18_copy_axis_shapes(detail_fig, combined_fig, row_map={1: 1, 2: 2})
    _block18_copy_subplot_traces(mad_rows_fig, combined_fig, row_map={2: 3})
    _block18_copy_axis_annotations(mad_rows_fig, combined_fig, row_map={2: 3})
    _block18_copy_axis_shapes(mad_rows_fig, combined_fig, row_map={2: 3})

    combined_fig.update_yaxes(title_text=f"{ratio_label} Z-Score", row=1, col=1)
    combined_fig.update_yaxes(title_text="Spread Z-Score", row=2, col=1)
    combined_fig.update_yaxes(title_text="MAD Score", tickformat=".2f", range=[-2, 6], row=3, col=1)
    combined_fig.update_xaxes(title_text="Date", row=3, col=1)

    detail_start, detail_end = trace_datetime_bounds(combined_fig.data)
    if detail_start is not None and detail_end is not None:
        detail_default_start = max(detail_start, detail_end - pd.DateOffset(years=3))
        combined_fig.update_xaxes(range=[detail_default_start, detail_end])

    combined_fig.update_layout(
        title=header_title(
            f"{ticker_str} vs {benchmark_label} Risk-Adjusted Return and Compounding Diagnostics [{window}-Day]"
        ),
        hovermode="x unified",
        height=1610,
        margin=header_margin(top=150),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[],
    )
    return finalize_dark_figure(combined_fig)


def _block18_asset_only_risk_figure(window, ratio_type="sharpe"):
    """Build the three-panel risk view without silently adding a benchmark."""
    window = _block18_validate_window(window)
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    asset_close = _block18_close_series(asset_history, ticker_str)
    asset_ratio_zscore = benchmark_zscore_for_plot(
        rolling_ratio_series(asset_close, window, ratio_type)
    )
    mad_rows_figure = build_block18_rolling_mean_rows_figure(window, [])
    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.36, 0.29, 0.35],
        subplot_titles=(
            "Risk-Adjusted Return Z-Score",
            f"{ratio_label} Spread Z-Score",
            "Volatility Drag MAD Score",
        ),
    )
    figure.add_trace(
        go.Scatter(
            x=asset_ratio_zscore.index, y=asset_ratio_zscore,
            mode="lines", name=f"{ticker_str} {window}-Day {ratio_label} Z-Score",
            line={"color": "#60a5fa", "width": 2.2},
            hovertemplate=(
                f"{ticker_str}<br>%{{x|%Y-%m-%d}}<br>"
                f"{ratio_label} Z-Score: %{{y:.2f}}<extra></extra>"
            ),
        ), row=1, col=1,
    )
    _block18_copy_subplot_traces(mad_rows_figure, figure, row_map={2: 3})
    _block18_copy_axis_annotations(mad_rows_figure, figure, row_map={2: 3})
    _block18_copy_axis_shapes(mad_rows_figure, figure, row_map={2: 3})
    figure.add_hline(
        y=0, line_color="rgba(226, 232, 240, 0.70)", row=1, col=1
    )
    figure.add_annotation(
        text="Select a benchmark to calculate the ratio spread",
        showarrow=False, font={"color": "#94a3b8", "size": 13},
        bgcolor="rgba(15, 23, 42, 0.78)", bordercolor="#334155",
        borderwidth=1, borderpad=8, row=2, col=1,
    )
    figure.update_yaxes(title_text=f"{ratio_label} Z-Score", row=1, col=1)
    figure.update_yaxes(title_text="Spread Z-Score", row=2, col=1)
    figure.update_yaxes(
        title_text="MAD Score", tickformat=".2f", range=[-2, 6], row=3, col=1
    )
    figure.update_xaxes(title_text="Date", row=3, col=1)
    date_start, date_end = trace_datetime_bounds(figure.data)
    if date_start is not None and date_end is not None:
        figure.update_xaxes(
            range=[max(date_start, date_end - pd.DateOffset(years=3)), date_end]
        )
    figure.update_layout(
        title=header_title(
            f"{ticker_str} Risk-Adjusted Return and Compounding Diagnostics [{window}-Day]"
        ),
        hovermode="x unified", height=1610, margin=header_margin(top=150),
        template="plotly_dark", updatemenus=[],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return finalize_dark_figure(figure)

def _block18_error_figure(message, *, height=850):
    error_fig = go.Figure()
    error_fig.add_annotation(
        text=message,
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font=dict(color="#fca5a5", size=14),
    )
    error_fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        margin=dict(t=48, r=24, b=24, l=24),
    )
    return error_fig

def _block18_available_port(start=8060, stop=8090):
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No open localhost port found for the Block 18 Dash app.")

block18_default_window = int(globals().get("default_window", 200))
block18_default_display_range = "3y"
block18_default_ratio_type = "sharpe"
block18_ratio_options = [
    {"label": "Sharpe Ratio", "value": "sharpe"},
    {"label": "Sortino Ratio", "value": "sortino"},
    {"label": "Return", "value": "return"},
    {"label": "Volatility", "value": "volatility"},
    {"label": "Downside Volatility", "value": "downside_volatility"},
] + [
    {
        "label": f"Correlation vs {symbol}",
        "value": f"{CORRELATION_METRIC_PREFIX}{symbol}",
    }
    for symbol in benchmark_order
] + [
    {
        "label": f"{metric_name.title()} Ratio vs {symbol}",
        "value": f"{metric_name}::{symbol}",
    }
    for metric_name in BENCHMARK_RELATIVE_METRIC_PREFIXES
    for symbol in benchmark_order
]
block18_default_benchmark_selection = ["SPY"] if "SPY" in benchmark_order else (
    [default_benchmark] if default_benchmark else []
)


def _block18_normalize_ratio_type(ratio_type):
    return normalize_momentum_ratio_type(ratio_type or block18_default_ratio_type)


def _block18_ratio_label(ratio_type):
    return momentum_ratio_label(_block18_normalize_ratio_type(ratio_type))


block18_playback_dates = pd.DatetimeIndex([])
for _ratio_contexts in momentum_diagnostics_contexts_by_ratio.values():
    _asset_ratio_context = _ratio_contexts.get(ticker_str)
    if _asset_ratio_context is None:
        continue
    _ratio_dates = pd.DatetimeIndex(
        pd.to_datetime(
            pd.DataFrame(_asset_ratio_context["ratio_table"])
            .dropna(how="all")
            .index,
            errors="coerce",
            utc=True,
        )
    ).dropna().tz_convert(None).normalize()
    block18_playback_dates = block18_playback_dates.union(_ratio_dates)
block18_playback_dates = block18_playback_dates.unique().sort_values()
if block18_playback_dates.empty:
    raise ValueError("Block 18 playback needs at least one valid diagnostics date.")
block18_today = pd.Timestamp.today().normalize()
block18_default_playback_position = int(
    block18_playback_dates.searchsorted(block18_today, side="right") - 1
)
block18_default_playback_position = max(
    0, min(block18_default_playback_position, len(block18_playback_dates) - 1)
)
block18_playback_step = 5  # Advance five trading sessions per timer tick.
block18_continuous_playback_interval = 75
block18_playback_speed_options = [
    {"label": "Continuous (no hold)", "value": "continuous"},
    {"label": "Slow", "value": 750},
    {"label": "Normal", "value": 300},
    {"label": "Fast", "value": 120},
]
block18_default_playback_speed = "continuous"
block18_default_playback_interval = block18_continuous_playback_interval

def _block18_animation_options(speed_milliseconds):
    continuous = speed_milliseconds == "continuous"
    if continuous:
        frame_duration = block18_continuous_playback_interval
    else:
        try:
            frame_duration = max(100, int(speed_milliseconds))
        except (TypeError, ValueError):
            continuous = block18_default_playback_speed == "continuous"
            frame_duration = block18_default_playback_interval
    transition_duration = (
        frame_duration if continuous else max(80, frame_duration - 20)
    )
    return {
        "frame": {"duration": frame_duration, "redraw": False},
        "transition": {"duration": transition_duration, "easing": "linear"},
        "mode": "immediate",
        "fromcurrent": True,
    }
block18_playback_mark_positions = sorted(set(
    int(position) for position in np.linspace(0, len(block18_playback_dates) - 1, num=min(8, len(block18_playback_dates)))
))
block18_playback_marks = {
    position: block18_playback_dates[position].strftime("%Y-%m-%d")
    for position in block18_playback_mark_positions
}
block18_base_figure_cache = {}
block18_window_diagnostics_playback_cache = {}
block18_functional_playback_plan_cache = {}
block18_diagnostics_window_min = int(min(window_sizes))
block18_diagnostics_window_max = int(max(window_sizes))
block18_default_diagnostics_window_range = [
    block18_diagnostics_window_min, block18_diagnostics_window_max
]
block18_diagnostics_window_marks = {
    value: str(value)
    for value in sorted(set([
        block18_diagnostics_window_min,
        30, 60, 90, 120, 180, 200, 252, 300, 400, 540, 720,
        block18_diagnostics_window_max,
    ]))
    if block18_diagnostics_window_min <= value <= block18_diagnostics_window_max
}
block18_default_spline_toggle = "b_spline"
block18_default_raw_overlay_toggle = ["raw_overlay"]
block18_autocorrelation_lag_min = 1
block18_autocorrelation_lag_max = 10000
block18_default_autocorrelation_lag_range = [1, 60]
block18_default_autocorrelation_return_horizon = 21
block18_default_autocorrelation_sampling_mode = "overlapping"
block18_default_autocorrelation_comparison_lag = 1
block18_default_first_passage_threshold = 1.0
block18_default_first_passage_end_threshold = 0.0
block18_default_first_passage_start_condition = "above"
block18_default_first_passage_end_condition = "below"
block18_default_first_passage_start_sign = "both"
block18_default_first_passage_end_sign = "both"
block18_default_spline_strength = 25
block18_horizon_derivative_scale = 20.0
block18_extrema_min_prominence = 0.12
block18_extrema_prominence_fraction = 0.08
block18_asset_profile_color = "#60a5fa"
block18_benchmark_profile_palette = [
    "#f97316", "#22c55e", "#facc15", "#ef4444", "#ec4899", "#f8fafc"
]

block18_tab_style = {"backgroundColor": "#111827", "color": "#cbd5e1", "borderColor": "#334155"}
block18_selected_tab_style = {"backgroundColor": "#1f2937", "color": "#ffffff", "borderColor": "#60a5fa"}
block18_display_range_options = [
    {"label": "Full", "value": "full"},
    {"label": "10 Years", "value": "10y"},
    {"label": "5 Years", "value": "5y"},
    {"label": "3 Years", "value": "3y"},
    {"label": "2 Years", "value": "2y"},
    {"label": "1 Year", "value": "1y"},
    {"label": "6 Months", "value": "6m"},
    {"label": "3 Months", "value": "3m"},
]
block18_display_range_labels = {option["value"]: option["label"] for option in block18_display_range_options}
block18_display_range_offsets = {
    "full": None,
    "10y": pd.DateOffset(years=10),
    "5y": pd.DateOffset(years=5),
    "3y": pd.DateOffset(years=3),
    "2y": pd.DateOffset(years=2),
    "1y": pd.DateOffset(years=1),
    "6m": pd.DateOffset(months=6),
    "3m": pd.DateOffset(months=3),
}
block18_tab_config = {
    "historical_surface_3d": {"label": "Historical Ratio Surface", "height": 1200, "accent": "#06b6d4", "tab_color": "rgba(6, 182, 212, 0.16)", "selected_tab_color": "rgba(6, 182, 212, 0.30)"},
    "window_diagnostics": {"label": "Functional Horizon Profile", "height": 1900, "accent": "#f59e0b", "tab_color": "rgba(245, 158, 11, 0.16)", "selected_tab_color": "rgba(245, 158, 11, 0.30)"},
    "risk": {"label": "Risk & Compounding", "height": 1610, "accent": "#60a5fa", "tab_color": "rgba(96, 165, 250, 0.16)", "selected_tab_color": "rgba(96, 165, 250, 0.30)"},
    "risk_compounding_v2": {"label": "Risk & Compounding v.2", "height": 3700, "accent": "#818cf8", "tab_color": "rgba(129, 140, 248, 0.16)", "selected_tab_color": "rgba(129, 140, 248, 0.30)"},
    "kappa": {"label": "Kappa Profile", "height": 1450, "accent": "#a78bfa", "tab_color": "rgba(167, 139, 250, 0.16)", "selected_tab_color": "rgba(167, 139, 250, 0.30)"},
    "pezier_white": {"label": "Pézier–White", "height": 1500, "accent": "#10b981", "tab_color": "rgba(16, 185, 129, 0.16)", "selected_tab_color": "rgba(16, 185, 129, 0.30)"},
    "systematic_risk": {"label": "Systematic Risk", "height": 1550, "accent": "#eab308", "tab_color": "rgba(234, 179, 8, 0.16)", "selected_tab_color": "rgba(234, 179, 8, 0.30)"},
    "cornish_fisher": {"label": "Cornish–Fisher", "height": 1900, "accent": "#ec4899", "tab_color": "rgba(236, 72, 153, 0.16)", "selected_tab_color": "rgba(236, 72, 153, 0.30)"},
    "autocorrelation": {"label": "Time Dependency", "height": 1000, "accent": "#38bdf8", "tab_color": "rgba(56, 189, 248, 0.16)", "selected_tab_color": "rgba(56, 189, 248, 0.30)"},
    "first_passage": {"label": "First Passage", "height": 1550, "accent": "#f43f5e", "tab_color": "rgba(244, 63, 94, 0.16)", "selected_tab_color": "rgba(244, 63, 94, 0.30)"},
    "drawdown": {"label": "Price Chart", "height": 1650, "accent": "#f87171"},
    "seasonality": {"label": "Seasonality", "height": 850, "accent": "#34d399"},
    "correlation": {"label": "Rolling Correlation", "height": 850, "accent": "#38bdf8"},
    "volatility_efficiency": {"label": "Volatility, Efficiency & Distribution", "height": 2050, "accent": "#fb7185"},
}
block18_default_tab = "drawdown"
block18_tab_order = ["drawdown", "historical_surface_3d", "window_diagnostics", "risk", "risk_compounding_v2", "systematic_risk", "kappa", "pezier_white", "cornish_fisher", "autocorrelation", "first_passage", "seasonality", "correlation", "volatility_efficiency"]
block18_ratio_dependent_tabs = {
    "historical_surface_3d", "window_diagnostics", "risk", "seasonality"
}
block18_date_range_tabs = {"risk", "systematic_risk", "pezier_white", "cornish_fisher", "first_passage", "drawdown", "correlation", "volatility_efficiency"}
block18_native_axis_status = {
    "seasonality": "Native seasonal axes",
    "historical_surface_3d": "3D date/lookback-horizon axes",
    "window_diagnostics": "Native lookback-horizon/DTE axes",
}

def _block18_graph_height(figure, default_height=1125):
    height = getattr(getattr(figure, "layout", None), "height", None)
    try:
        return f"{int(height)}px"
    except (TypeError, ValueError):
        return f"{int(default_height)}px"


def _block18_add_history_range_controls(
    figure, axis_names, global_start, global_end, *, menu_y=1.075
):
    """Add the shared 1M-All history buttons and default date axes to six months."""
    global_start = pd.Timestamp(global_start)
    global_end = pd.Timestamp(global_end)
    range_options = [
        ("1M", pd.DateOffset(months=1)),
        ("3M", pd.DateOffset(months=3)),
        ("6M", pd.DateOffset(months=6)),
        ("1Y", pd.DateOffset(years=1)),
        ("3Y", pd.DateOffset(years=3)),
        ("All", None),
    ]
    buttons = []
    for label, offset in range_options:
        range_start = global_start if offset is None else max(global_start, global_end - offset)
        buttons.append(dict(
            label=label,
            method="relayout",
            args=[{f"{axis_name}.range": [range_start, global_end] for axis_name in axis_names}],
        ))
    default_start = max(global_start, global_end - pd.DateOffset(months=6))
    for axis_name in axis_names:
        figure.layout[axis_name].update(
            type="date", range=[default_start, global_end], autorange=False
        )
    existing_menus = list(figure.layout.updatemenus or ())
    existing_annotations = list(figure.layout.annotations or ())
    existing_menus.append(dict(
        type="buttons", direction="right", buttons=buttons, active=2,
        x=0.5, y=menu_y, xanchor="center", yanchor="bottom",
        bgcolor="#111827", bordercolor="#475569", borderwidth=1,
        font=dict(color="#e2e8f0", size=11),
        pad=dict(l=4, r=4, t=3, b=3),
    ))
    existing_annotations.append(dict(
        text="History range", x=0.5, y=menu_y + 0.04,
        xref="paper", yref="paper", showarrow=False,
        font=dict(color="#94a3b8", size=11),
    ))
    figure.update_layout(updatemenus=existing_menus, annotations=existing_annotations)
    return figure

def _block18_tab(value):
    tab_config = block18_tab_config[value]
    accent = tab_config["accent"]
    return dcc.Tab(
        label=tab_config["label"],
        value=value,
        style={
            **block18_tab_style,
            "backgroundColor": tab_config.get("tab_color", block18_tab_style["backgroundColor"]),
            "borderTop": f"4px solid {accent}",
            "borderBottom": "1px solid #334155",
        },
        selected_style={
            **block18_selected_tab_style,
            "backgroundColor": tab_config.get(
                "selected_tab_color", block18_selected_tab_style["backgroundColor"]
            ),
            "borderTop": f"4px solid {accent}",
            "boxShadow": f"inset 0 3px 0 {accent}",
        },
    )

def _block18_display_range_label(value):
    return block18_display_range_labels.get(value, block18_display_range_labels[block18_default_display_range])

def _block18_strip_figure_controls(
    figure, *, preserve_benchmark_selector=False, preserve_all_controls=False
):
    stripped = go.Figure(figure)
    if preserve_all_controls:
        return stripped
    if not preserve_benchmark_selector:
        stripped.layout.updatemenus = ()
        return stripped

    benchmark_labels = set(benchmark_order)
    benchmark_menus = []
    for menu in stripped.layout.updatemenus or []:
        button_labels = {str(button.label) for button in menu.buttons or []}
        if button_labels.intersection(benchmark_labels):
            benchmark_menus.append(menu)
    if benchmark_menus:
        stripped.layout.updatemenus = tuple(benchmark_menus)
    return stripped

def _block18_named_figure(figure_name, label, default_height=1125):
    figure = globals().get(figure_name)
    if figure is None:
        return _block18_error_figure(f"{label} figure is unavailable. Run the source cell first.", height=default_height)
    return figure

def _block18_apply_display_range(
    figure,
    display_range_value,
    *,
    preserve_benchmark_selector=False,
    as_of_date=None,
    show_playhead=False,
):
    figure = _block18_strip_figure_controls(
        figure, preserve_benchmark_selector=preserve_benchmark_selector
    )
    display_range_value = display_range_value if display_range_value in block18_display_range_offsets else block18_default_display_range
    date_start, date_end = trace_datetime_bounds(figure.data)
    if date_start is None or date_end is None:
        return figure

    effective_end = date_end
    if as_of_date is not None:
        requested_end = pd.Timestamp(as_of_date)
        if requested_end.tzinfo is not None:
            requested_end = requested_end.tz_convert(None)
        effective_end = min(date_end, requested_end)
        effective_end = max(date_start, effective_end)

    offset = block18_display_range_offsets[display_range_value]
    range_start = date_start if offset is None else max(date_start, effective_end - offset)
    for axis_name in figure.layout.to_plotly_json():
        if axis_name.startswith("xaxis"):
            figure.layout[axis_name].update(range=[range_start, effective_end])

    if show_playhead:
        figure.add_shape(
            type="line",
            x0=effective_end,
            x1=effective_end,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"color": "#facc15", "width": 2, "dash": "dash"},
            name="Playback as-of date",
        )
        figure.add_annotation(
            x=effective_end,
            y=1,
            xref="x",
            yref="paper",
            text=f"As of {effective_end:%Y-%m-%d}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": "#facc15", "size": 11},
            bgcolor="rgba(15, 23, 42, 0.80)",
        )
    return figure

def _block18_build_term_config_for_window(window):
    label = f"{window}-day"
    if label in term_config_map:
        return {label: term_config_map[label]}

    sharpe = rolling_ratio_series(asset_close, window, "sharpe")
    sortino = rolling_ratio_series(asset_close, window, "sortino")
    spread = sortino - sharpe
    return {
        label: {
            "sharpe": sharpe,
            "sortino": sortino,
            "spread": spread,
            "sharpe_zscore": zscore_or_empty(sharpe),
            "sortino_zscore": zscore_or_empty(sortino),
            "spread_zscore": zscore_or_empty(spread),
            "time_frame": window,
            "term_key": label,
        }
    }

def _block18_drawdown_recovery_for_window(window):
    rolling_peak = compute.rolling(
        asset_history["Close"],
        metric=pd.Series.max,
        window=window,
        min_periods=1,
    )
    underwater_series = asset_history["Close"].div(rolling_peak).sub(1.0).dropna()
    drawdown_series = compute.rolling(
        asset_history["Close"],
        metric=metric.textbook_window_drawdown,
        window=window,
        dropna=False,
    ).dropna()
    recovery_series = compute.rolling(
        asset_history["Close"],
        metric=metric.window_recovery_time,
        window=window,
        dropna=False,
    ).dropna()
    return {
        window: {
            "underwater": underwater_series,
            "max_drawdown": drawdown_series,
            "recovery_time": recovery_series,
        }
    }

def _block18_rolling_correlation_for_window(window, benchmark_symbols=None):
    asset_returns_for_window = asset_close.pct_change()
    term_label = f"{window}-day"
    term_series_map = {}
    benchmark_candidates = benchmark_order if benchmark_symbols is None else benchmark_symbols
    benchmark_symbols = [
        symbol for symbol in benchmark_candidates if symbol in benchmark_data
    ]
    for symbol in benchmark_symbols:
        benchmark_frame = benchmark_data[symbol]
        benchmark_close = benchmark_frame["Close"] if isinstance(benchmark_frame, pd.DataFrame) else benchmark_frame
        benchmark_returns = pd.Series(benchmark_close).dropna().sort_index().pct_change()
        aligned_returns = pd.concat(
            {"asset": asset_returns_for_window, symbol: benchmark_returns},
            axis=1,
        ).dropna()
        if aligned_returns.empty:
            continue

        rolling_correlation_series = aligned_returns["asset"].rolling(window).corr(aligned_returns[symbol]).dropna()
        if not rolling_correlation_series.empty:
            term_series_map[symbol] = rolling_correlation_series

    return {term_label: term_series_map} if term_series_map else {}

def _block18_benchmark_trace_owner(trace_name):
    trace_name = str(trace_name)
    for symbol in sorted(benchmark_order, key=len, reverse=True):
        if trace_name == symbol or trace_name.startswith(f"{symbol} "):
            return symbol
    return None

def _block18_filter_benchmark_traces(figure, selected_benchmarks):
    filtered = go.Figure(figure)
    filtered.data = tuple(
        trace
        for trace in filtered.data
        if (
            _block18_benchmark_trace_owner(getattr(trace, "name", "")) is None
            or _block18_benchmark_trace_owner(getattr(trace, "name", "")) in selected_benchmarks
        )
    )
    return filtered

def _block18_normalized_shannon_entropy(values, bins=5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < bins or np.ptp(values) == 0:
        return np.nan
    counts = np.histogram(values, bins=bins)[0]
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(bins))

def _block18_hurst_rs(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    candidate_sizes = np.array([8, 16, 32, 64, 128])
    sizes = candidate_sizes[candidate_sizes <= len(values) // 2]
    rs_points = []
    for size in sizes:
        ratios = []
        for start in range(0, len(values) - size + 1, size):
            chunk = values[start:start + size]
            std = chunk.std(ddof=1)
            if std > 0:
                path = np.cumsum(chunk - chunk.mean())
                ratios.append((path.max() - path.min()) / std)
        if ratios:
            rs_points.append((size, np.mean(ratios)))
    if len(rs_points) < 2:
        return np.nan
    x, y = np.asarray(rs_points, dtype=float).T
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])

block18_garch_volatility_cache = {}

def _block18_rolling_garch11_volatility(log_returns, window):
    if block18_arch_model is None:
        raise ImportError("GARCH(1,1) requires the 'arch' package in the dashboard environment.")
    window = _block18_validate_window(window)
    if window < 30:
        raise ValueError("Rolling GARCH(1,1) requires a fitting window of at least 30 returns.")
    returns = pd.to_numeric(pd.Series(log_returns), errors="coerce").replace([np.inf, -np.inf], np.nan).sort_index()
    clean_signature = returns.dropna()
    if len(clean_signature) < window:
        raise ValueError(f"Rolling GARCH(1,1) needs at least {window:,} clean returns.")
    cache_key = (window, len(clean_signature), clean_signature.index[-1], float(clean_signature.iloc[-1]))
    if cache_key not in block18_garch_volatility_cache:
        rolling_forecast = pd.Series(np.nan, index=returns.index, dtype=float, name="rolling_garch11_volatility")
        for origin_position in range(window - 1, len(returns)):
            estimation_sample = returns.iloc[origin_position - window + 1:origin_position + 1].dropna()
            if len(estimation_sample) != window:
                continue
            try:
                fitted = block18_arch_model(
                    estimation_sample * 100.0, mean="Constant", vol="GARCH", p=1, q=1,
                    dist="normal", rescale=False,
                ).fit(disp="off", show_warning=False)
                one_day_variance = float(fitted.forecast(horizon=1, method="analytic", reindex=False).variance.iloc[-1, 0])
                if np.isfinite(one_day_variance) and one_day_variance >= 0:
                    rolling_forecast.iloc[origin_position] = np.sqrt(one_day_variance * 252.0)
            except Exception:
                continue
        block18_garch_volatility_cache.clear()
        block18_garch_volatility_cache[cache_key] = rolling_forecast
    return block18_garch_volatility_cache[cache_key].copy()

def _block18_volatility_efficiency_figure(window):
    selected_window = _block18_validate_window(window)
    vol_window = selected_window
    efficiency_window = selected_window
    vol_of_vol_window = selected_window
    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    returns = np.log(close).diff().dropna()
    required_returns = efficiency_window + vol_of_vol_window - 1
    if len(returns) < required_returns:
        raise ValueError(f"{ticker_str} needs at least {required_returns:,} returns for nested {selected_window}-day volatility metrics.")
    volatility = returns.rolling(vol_window).std() * np.sqrt(252) * 100
    entropy = returns.rolling(efficiency_window).apply(
        _block18_normalized_shannon_entropy, raw=True, kwargs={"bins": 5}
    )
    autocorrelation = returns.rolling(efficiency_window).corr(returns.shift(1))
    hurst = returns.rolling(efficiency_window).apply(_block18_hurst_rs, raw=True)
    vol_of_vol = volatility.rolling(vol_of_vol_window).std()
    garch_volatility = _block18_rolling_garch11_volatility(returns, selected_window)
    metrics = pd.concat({
        "Volatility": volatility,
        "Entropy": entropy,
        "Autocorrelation": autocorrelation,
        "Hurst exponent": hurst,
        "Vol of vol": vol_of_vol,
    }, axis=1)
    titles = [
        f"Volatility ({vol_window}d annualized)",
        f"Normalized Entropy ({efficiency_window}d)",
        f"Lag-1 Autocorrelation ({efficiency_window}d)",
        f"Hurst Exponent ({efficiency_window}d)",
        f"Volatility of Volatility ({vol_of_vol_window}d)",
    ]
    colors = ["#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#fb7185"]
    figure = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=titles)
    for row, (column, color) in enumerate(zip(metrics.columns, colors), start=1):
        trace_name = f"{selected_window}-Day Trailing Realized Volatility" if column == "Volatility" else column
        figure.add_trace(go.Scatter(x=metrics.index, y=metrics[column], name=trace_name, mode="lines", line=dict(color=color, width=1.5), showlegend=column == "Volatility"), row=row, col=1)
        if row == 1:
            figure.add_trace(go.Scatter(x=garch_volatility.index, y=garch_volatility, name=f"Rolling GARCH(1,1), {selected_window}-Day Fit, 1-Day Forecast", mode="lines", line=dict(color="#f97316", width=1.5), showlegend=True), row=1, col=1)
    figure.add_hline(y=1.0, line_dash="dot", line_color="#64748b", row=2, col=1)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=3, col=1)
    figure.add_hline(y=0.5, line_dash="dot", line_color="#64748b", row=4, col=1)
    figure.update_yaxes(title_text="Annualized %", row=1, col=1)
    figure.update_yaxes(title_text="0-1", range=[0, 1.05], row=2, col=1)
    figure.update_yaxes(title_text="Correlation", range=[-1, 1], row=3, col=1)
    figure.update_yaxes(title_text="H", row=4, col=1)
    figure.update_yaxes(title_text="Percentage pts", row=5, col=1)
    figure.update_layout(title=f"{ticker_str} - Volatility & Market Efficiency", template="plotly_dark", height=1200, hovermode="x unified", showlegend=False, margin=dict(t=90, r=30, b=45, l=80))
    return figure

def _block18_return_distribution_figure(window):
    close = coerce_close_series(asset_history["Close"])
    distribution_window = _block18_validate_window(window)
    if len(close) <= distribution_window:
        raise ValueError(f"Return Distribution needs more than {distribution_window:,} clean price observations.")
    windows = [distribution_window]
    daily_returns = close.pct_change(fill_method=None).dropna()
    metrics_by_window = {}
    for distribution_window in windows:
        def rolling_quantile(level):
            return daily_returns.rolling(distribution_window).quantile(level).dropna()
        rolling_skew = daily_returns.rolling(distribution_window).skew().dropna()
        rolling_kurtosis = daily_returns.rolling(distribution_window).kurt().dropna()
        rolling_gini = compute.rolling(
            daily_returns, metric=gini_coefficient, window=distribution_window
        ).dropna()
        metrics_by_window[distribution_window] = {
            "daily_returns": daily_returns.copy(),
            "return_q10": rolling_quantile(0.10),
            "return_q25": rolling_quantile(0.25),
            "return_median": rolling_quantile(0.50),
            "return_q75": rolling_quantile(0.75),
            "return_q90": rolling_quantile(0.90),
            "max_drawdown": calculate_textbook_rolling_max_drawdown(close, window=distribution_window).dropna(),
            "skew_z": calculate_zscore(rolling_skew),
            "kurtosis_z": calculate_zscore(rolling_kurtosis),
            "gini_z": calculate_zscore(rolling_gini),
        }
    return plot_distribution_shape_zscores_view(
        metrics_by_window=metrics_by_window,
        window_options=windows,
        default_window=distribution_window,
        ticker_label=ticker_str,
        include_return_panel=False,
    )

def _block18_volatility_efficiency_distribution_figure(window):
    volatility_figure = _block18_volatility_efficiency_figure(window)
    distribution_figure = _block18_return_distribution_figure(window)
    subplot_titles = [
        f"Volatility ({int(window)}d annualized)",
        f"Normalized Entropy ({int(window)}d)",
        f"Lag-1 Autocorrelation ({int(window)}d)",
        f"Hurst Exponent ({int(window)}d)",
        f"Volatility of Volatility ({int(window)}d)",
        "Rolling Skew Z-Score",
        "Rolling Excess Kurtosis Z-Score",
        "Rolling Gini Coefficient Z-Score",
    ]
    combined = make_subplots(
        rows=6, cols=2, shared_xaxes=True, vertical_spacing=0.035, horizontal_spacing=0.08,
        specs=[
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        subplot_titles=subplot_titles, row_heights=[0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
    )
    volatility_positions = [(1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
    for trace, (row, col) in zip(volatility_figure.data, volatility_positions):
        combined.add_trace(copy.deepcopy(trace), row=row, col=col)
    distribution_axis_rows = {"x": 4, "x2": 5, "x3": 6}
    for trace in distribution_figure.data:
        distribution_trace = copy.deepcopy(trace)
        distribution_trace.showlegend = False
        combined.add_trace(distribution_trace, row=distribution_axis_rows.get(trace.xaxis, 6), col=1)
    # Keep the analytical calculations at full resolution, but cap browser payload size.
    # Plotly otherwise retransmits and redraws tens of thousands of points on every tab selection.
    block18_combined_max_points = 1500
    for trace in combined.data:
        point_count = len(trace.x) if getattr(trace, "x", None) is not None else 0
        if point_count <= block18_combined_max_points:
            continue
        sample_positions = np.unique(np.linspace(0, point_count - 1, block18_combined_max_points, dtype=int))
        trace.x = np.asarray(trace.x)[sample_positions]
        if getattr(trace, "y", None) is not None and len(trace.y) == point_count:
            trace.y = np.asarray(trace.y)[sample_positions]
        if getattr(trace, "customdata", None) is not None and len(trace.customdata) == point_count:
            trace.customdata = np.asarray(trace.customdata)[sample_positions]
    combined.update_yaxes(title_text="Annualized %", row=1, col=1)
    combined.update_yaxes(title_text="0-1", range=[0, 1.05], row=1, col=2)
    combined.update_yaxes(title_text="Correlation", range=[-1, 1], row=2, col=1)
    combined.update_yaxes(title_text="H", row=2, col=2)
    combined.update_yaxes(title_text="Percentage pts", row=3, col=1)
    combined.update_yaxes(title_text="Z-Score", row=4, col=1)
    combined.update_yaxes(title_text="Z-Score", row=5, col=1)
    combined.update_yaxes(title_text="Z-Score", row=6, col=1)
    combined.add_hline(y=1.0, line_dash="dot", line_color="#64748b", row=1, col=2)
    combined.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=2, col=1)
    combined.add_hline(y=0.5, line_dash="dot", line_color="#64748b", row=2, col=2)
    combined.update_layout(
        title=f"{ticker_str} - Volatility, Market Efficiency & Return Distribution ({int(window)}-Day Distribution Window)",
        template="plotly_dark", height=2050, hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        margin=dict(t=110, r=30, b=45, l=80), updatemenus=[],
    )
    return combined


def _block18_kappa_profile_figure(window):
    """Plot Kappa orders 1.0-4.0 for the selected trailing return window."""
    window = _block18_validate_window(window)
    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    returns = close.pct_change(fill_method=None).dropna()
    periodic_mar = (
        pd.to_numeric(risk_free_daily_rate, errors="coerce")
        .reindex(returns.index)
        .ffill()
        .bfill()
    )
    sample = pd.concat(
        {"return": returns, "mar": periodic_mar}, axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna().tail(window)
    if len(sample) < window:
        raise ValueError(
            f"Kappa Profile needs {window:,} aligned returns and risk-free rates; "
            f"only {len(sample):,} are available."
        )

    excess = sample["return"] - sample["mar"]
    shortfall = (-excess).clip(lower=0.0)
    mean_excess = float(excess.mean())
    centered_returns = sample["return"] - float(sample["return"].mean())
    orders = np.round(np.arange(1.0, 4.0 + 0.05, 0.1), 1)
    kappas = []
    downside_scales = []
    total_lp_ratios = []
    total_lp_scales = []
    for order in orders:
        downside_scale = float(shortfall.pow(order).mean() ** (1.0 / order))
        total_lp_scale = float(
            centered_returns.abs().pow(order).mean() ** (1.0 / order)
        )
        downside_scales.append(downside_scale)
        total_lp_scales.append(total_lp_scale)
        kappas.append(mean_excess / downside_scale if downside_scale > 0 else np.nan)
        total_lp_ratios.append(
            mean_excess / total_lp_scale if total_lp_scale > 0 else np.nan
        )

    kappa_frame = pd.DataFrame({
        "Order": orders,
        "Downside Kappa": kappas,
        "Downside Lp scale": downside_scales,
        "Total Lp ratio": total_lp_ratios,
        "Total Lp scale": total_lp_scales,
    })
    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.40, 0.60],
        vertical_spacing=0.10,
    )
    figure.add_trace(
        go.Scatter(
            x=kappa_frame["Order"],
            y=kappa_frame["Downside Kappa"],
            mode="lines+markers",
            name="Downside Kappa",
            line={"color": "#f97316", "width": 3},
            marker={"size": 7, "symbol": "circle", "color": "#f97316"},
            customdata=np.column_stack([
                kappa_frame["Downside Lp scale"],
                np.repeat(mean_excess, len(kappa_frame)),
            ]),
            hovertemplate=(
                "Kappa-%{x:.1f}: %{y:.4f}<br>"
                "Downside scale: %{customdata[0]:.6f}<br>"
                "Mean excess return: %{customdata[1]:.6f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=kappa_frame["Order"],
            y=kappa_frame["Total Lp ratio"],
            mode="lines+markers",
            name="Two-sided total-risk Lp",
            line={"color": "#22d3ee", "width": 3},
            marker={"size": 7, "symbol": "diamond", "color": "#22d3ee"},
            customdata=np.column_stack([
                kappa_frame["Total Lp scale"],
                np.repeat(mean_excess, len(kappa_frame)),
            ]),
            hovertemplate=(
                "Total L%{x:.1f} ratio: %{y:.4f}<br>"
                "Mean-centered Lp scale: %{customdata[0]:.6f}<br>"
                "Mean excess return: %{customdata[1]:.6f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    for conventional_order, label in ((1.0, "Kappa-1"), (2.0, "Kappa-2 / Sortino"), (3.0, "Kappa-3"), (4.0, "Kappa-4")):
        figure.add_vline(
            x=conventional_order,
            line_dash="dot",
            line_color="#64748b",
            annotation_text=label,
            annotation_position="top",
            row=1,
            col=1,
        )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", row=1, col=1)
    figure.add_trace(
        go.Table(
            header={
                "values": [
                    "Order (p)", "Downside Kappa", "Downside Lp scale",
                    "Total Lp ratio", "Total Lp scale",
                ],
                "fill_color": "#1e293b",
                "font": {"color": "#f8fafc", "size": 12},
                "align": "center",
            },
            cells={
                "values": [
                    [f"{value:.1f}" for value in kappa_frame["Order"]],
                    [f"{value:.4f}" if np.isfinite(value) else "N/A" for value in kappa_frame["Downside Kappa"]],
                    [f"{value:.6f}" for value in kappa_frame["Downside Lp scale"]],
                    [f"{value:.4f}" if np.isfinite(value) else "N/A" for value in kappa_frame["Total Lp ratio"]],
                    [f"{value:.6f}" for value in kappa_frame["Total Lp scale"]],
                ],
                "fill_color": "#0f172a",
                "font": {"color": "#e2e8f0", "size": 11},
                "align": "center",
                "height": 24,
            },
        ),
        row=2,
        col=1,
    )
    latest_mar = float(sample["mar"].iloc[-1])
    annualized_latest_mar = (1.0 + latest_mar) ** annualization_factor - 1.0
    figure.update_xaxes(title_text="Lp order (p = n)", dtick=0.1, row=1, col=1)
    figure.update_yaxes(title_text="Reward-to-risk ratio", row=1, col=1)
    figure.update_layout(
        title=(
            f"{ticker_str} Downside Kappa vs Total-Risk Lp Profile — {window}-Day Window<br>"
            f"<sup>Orders 1.0–4.0 in 0.1 steps; daily risk-free MAR "
            f"(latest annualized rate {annualized_latest_mar:.2%}); total risk is mean-centered; "
            "ratios are non-annualized</sup>"
        ),
        template="plotly_dark",
        height=1450,
        hovermode="x unified",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
        margin={"t": 105, "r": 35, "b": 35, "l": 75},
    )
    return figure


def _block18_pezier_white_figure(window):
    """Show rolling Sharpe attribution to skewness and excess kurtosis."""
    window = _block18_validate_window(window)
    if window < 4:
        raise ValueError("Pézier–White analysis requires a window of at least 4 returns.")

    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    returns = close.pct_change(fill_method=None).dropna()
    periodic_mar = (
        pd.to_numeric(risk_free_daily_rate, errors="coerce")
        .reindex(returns.index)
        .ffill()
        .bfill()
    )
    excess = (returns - periodic_mar).replace([np.inf, -np.inf], np.nan)
    rolling_mean = excess.rolling(window, min_periods=window).mean()
    rolling_std = excess.rolling(window, min_periods=window).std(ddof=1)
    periodic_sharpe = rolling_mean.div(rolling_std.where(rolling_std > 0))
    rolling_skew = returns.rolling(window, min_periods=window).skew()
    rolling_excess_kurtosis = returns.rolling(window, min_periods=window).kurt()

    skew_adjusted_periodic = periodic_sharpe * (
        1.0 + (rolling_skew / 6.0) * periodic_sharpe
    )
    fully_adjusted_periodic = periodic_sharpe * (
        1.0
        + (rolling_skew / 6.0) * periodic_sharpe
        - (rolling_excess_kurtosis / 24.0) * periodic_sharpe.pow(2)
    )
    annualizer = np.sqrt(annualization_factor)
    profile = pd.concat(
        {
            "Sharpe": periodic_sharpe * annualizer,
            "Skew-adjusted": skew_adjusted_periodic * annualizer,
            "Pézier–White": fully_adjusted_periodic * annualizer,
            "Skew contribution": (skew_adjusted_periodic - periodic_sharpe) * annualizer,
            "Kurtosis contribution": (fully_adjusted_periodic - skew_adjusted_periodic) * annualizer,
            "Skewness": rolling_skew,
            "Excess kurtosis": rolling_excess_kurtosis,
        },
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    latest = profile.dropna().tail(1)
    if latest.empty:
        raise ValueError(
            f"No complete Pézier–White estimates are available for the {window}-day window."
        )
    latest_date = pd.Timestamp(latest.index[-1])
    latest_values = latest.iloc[-1]

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.46, 0.27, 0.27],
        vertical_spacing=0.055,
        subplot_titles=[
            "Rolling Sharpe comparison — baseline, skew-adjusted, and full Pézier–White",
            "Higher-moment contribution to annualized Sharpe",
            f"Latest {window}-day assessment ({latest_date:%Y-%m-%d})",
        ],
    )
    performance_styles = {
        "Sharpe": ("#94a3b8", "dot"),
        "Skew-adjusted": ("#f59e0b", "dash"),
        "Pézier–White": ("#10b981", "solid"),
    }
    for column, (color, dash) in performance_styles.items():
        figure.add_trace(
            go.Scatter(
                x=profile.index,
                y=profile[column],
                name=column,
                mode="lines",
                line={"color": color, "width": 2.4, "dash": dash},
                customdata=np.column_stack([
                    profile["Skewness"], profile["Excess kurtosis"]
                ]),
                hovertemplate=(
                    f"{column}: %{{y:.3f}}<br>"
                    "Skewness: %{customdata[0]:.3f}<br>"
                    "Excess kurtosis: %{customdata[1]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    contribution_styles = {
        "Skew contribution": "#38bdf8",
        "Kurtosis contribution": "#f97316",
    }
    for column, color in contribution_styles.items():
        figure.add_trace(
            go.Scatter(
                x=profile.index,
                y=profile[column],
                name=column,
                mode="lines",
                line={"color": color, "width": 2},
                fill="tozeroy",
                opacity=0.78,
                hovertemplate=f"{column}: %{{y:+.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    net_adjustment = float(latest_values["Pézier–White"] - latest_values["Sharpe"])
    assessment = (
        "Higher moments improve the Sharpe assessment"
        if net_adjustment > 0
        else "Higher moments reduce the Sharpe assessment"
        if net_adjustment < 0
        else "Higher moments leave the Sharpe assessment unchanged"
    )
    # Add axis-spanning shapes before the domain-based table trace. Older
    # Plotly releases inspect every existing trace while adding an hline and
    # incorrectly request ``xaxis`` from Table traces.
    for row in range(1, 3):
        figure.add_hline(
            y=0.0, line_dash="dot", line_color="#64748b", row=row, col=1
        )
    figure.add_trace(
        go.Table(
            columnwidth=[1.45, 0.85, 2.6],
            header={
                "values": ["Measure", "Latest value", "Investment interpretation"],
                "fill_color": "#1e293b",
                "font": {"color": "#f8fafc", "size": 12},
                "align": ["left", "center", "left"],
            },
            cells={
                "values": [
                    [
                        "Annualized Sharpe", "Skew-adjusted Sharpe", "Pézier–White adjusted Sharpe",
                        "Skewness", "Excess kurtosis", "Net higher-moment adjustment",
                    ],
                    [
                        f"{latest_values['Sharpe']:.3f}",
                        f"{latest_values['Skew-adjusted']:.3f}",
                        f"{latest_values['Pézier–White']:.3f}",
                        f"{latest_values['Skewness']:.3f}",
                        f"{latest_values['Excess kurtosis']:.3f}",
                        f"{net_adjustment:+.3f}",
                    ],
                    [
                        "Mean/volatility baseline",
                        "Baseline plus the third-moment effect",
                        "Baseline plus skewness and fourth-moment effects",
                        "Positive is favorable; negative indicates left-tail asymmetry",
                        "Positive indicates heavier-than-normal tails",
                        assessment,
                    ],
                ],
                "fill_color": "#0f172a",
                "font": {"color": "#e2e8f0", "size": 11},
                "align": ["left", "center", "left"],
                "height": 30,
            },
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="Annualized ratio", row=1, col=1)
    figure.update_yaxes(title_text="Sharpe points", row=2, col=1)
    figure.update_layout(
        title=(
            f"{ticker_str} Pézier–White Higher-Moment Sharpe Analysis — {window}-Day Window<br>"
            "<sup>Daily Sharpe is adjusted for skewness and excess kurtosis, then scaled by √252</sup>"
        ),
        template="plotly_dark",
        height=1500,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
        margin={"t": 115, "r": 35, "b": 35, "l": 80},
    )
    return figure


def _block18_systematic_risk_figure(window, selected_benchmarks):
    """Plot rolling benchmark-specific beta and Treynor ratios."""
    window = _block18_validate_window(window)
    selected_benchmarks = _block18_normalize_benchmark_selection(
        selected_benchmarks
    )
    if not selected_benchmarks:
        raise ValueError("Select at least one benchmark index for Systematic Risk.")

    asset_close_series = pd.to_numeric(
        asset_history["Close"], errors="coerce"
    ).dropna().sort_index()
    asset_returns = asset_close_series.pct_change(fill_method=None)
    treynor_by_benchmark = {}
    treynor_zscore_by_benchmark = {}
    beta_by_benchmark = {}
    annual_excess_by_benchmark = {}
    latest_rows = []
    minimum_beta_magnitude = 0.05

    for symbol in selected_benchmarks:
        benchmark_frame = benchmark_data.get(symbol)
        if benchmark_frame is None:
            continue
        benchmark_close = (
            benchmark_frame["Close"]
            if isinstance(benchmark_frame, pd.DataFrame)
            else benchmark_frame
        )
        benchmark_returns = pd.to_numeric(
            benchmark_close, errors="coerce"
        ).dropna().sort_index().pct_change(fill_method=None)
        aligned = pd.concat(
            {
                "asset": asset_returns,
                "benchmark": benchmark_returns,
                "mar": pd.to_numeric(risk_free_daily_rate, errors="coerce"),
            },
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)
        aligned["mar"] = aligned["mar"].ffill().bfill()
        aligned = aligned.dropna()
        if len(aligned) < window:
            continue

        rolling_covariance = aligned["asset"].rolling(
            window, min_periods=window
        ).cov(aligned["benchmark"])
        rolling_benchmark_variance = aligned["benchmark"].rolling(
            window, min_periods=window
        ).var(ddof=1)
        beta = rolling_covariance.div(
            rolling_benchmark_variance.where(rolling_benchmark_variance > 0)
        ).replace([np.inf, -np.inf], np.nan)
        annual_excess = (
            (aligned["asset"] - aligned["mar"])
            .rolling(window, min_periods=window)
            .mean()
            * annualization_factor
        )
        stable_beta = beta.where(beta.abs() >= minimum_beta_magnitude)
        treynor = annual_excess.div(stable_beta).replace(
            [np.inf, -np.inf], np.nan
        )
        treynor_zscore = zscore_or_empty(treynor)
        beta_by_benchmark[symbol] = beta
        annual_excess_by_benchmark[symbol] = annual_excess
        treynor_by_benchmark[symbol] = treynor
        treynor_zscore_by_benchmark[symbol] = treynor_zscore

        latest = pd.concat(
            {
                "Treynor": treynor,
                "Treynor z-score": treynor_zscore,
                "Beta": beta,
                "Annual excess": annual_excess,
            },
            axis=1,
        ).dropna().tail(1)
        if not latest.empty:
            latest_rows.append((symbol, latest.index[-1], latest.iloc[-1]))

    if not treynor_by_benchmark:
        raise ValueError(
            f"No selected benchmark has enough aligned history for a {window}-day window."
        )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.50, 0.27, 0.23],
        vertical_spacing=0.075,
        subplot_titles=[
            "Historical Treynor ratio z-scores by benchmark beta",
            "Rolling systematic beta by benchmark",
            "Latest systematic-risk assessment",
        ],
    )
    colors = block18_benchmark_profile_palette
    for position, symbol in enumerate(treynor_by_benchmark):
        color = colors[position % len(colors)]
        treynor = treynor_by_benchmark[symbol]
        treynor_zscore = treynor_zscore_by_benchmark[symbol]
        beta = beta_by_benchmark[symbol]
        figure.add_trace(
            go.Scatter(
                x=treynor_zscore.index,
                y=treynor_zscore,
                name=f"Treynor Z-Score vs {symbol}",
                mode="lines",
                line={"color": color, "width": 2.4},
                customdata=np.column_stack([
                    treynor.reindex(treynor_zscore.index),
                    beta.reindex(treynor_zscore.index),
                ]),
                hovertemplate=(
                    f"Treynor z-score vs {symbol}: %{{y:.3f}}<br>"
                    "Raw Treynor: %{customdata[0]:.3f}<br>"
                    "Rolling beta: %{customdata[1]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=beta.index,
                y=beta,
                name=f"Beta vs {symbol}",
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"Beta vs {symbol}: %{{y:.3f}}<extra></extra>",
                legendgroup=symbol,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Match the z-score regime shading used by Risk & Compounding. Add these
    # before the Table trace for compatibility with older Plotly.
    for lower, upper, color in (
        (-1.0, 1.0, "rgba(148, 163, 184, 0.12)"),
        (-2.0, -1.0, "rgba(0, 128, 0, 0.30)"),
        (1.0, 2.0, "rgba(180, 0, 0, 0.30)"),
    ):
        figure.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=color,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )
    figure.add_hline(
        y=0.0,
        line_dash="solid",
        line_color="rgba(226, 232, 240, 0.70)",
        row=1,
        col=1,
    )
    for sigma_level in (-2.0, -1.0, 1.0, 2.0):
        figure.add_hline(
            y=sigma_level,
            line_dash="dash",
            line_color="rgba(148, 163, 184, 0.55)",
            row=1,
            col=1,
        )
    for label, y_position, color in (
        ("Accumulate", -1.5, "rgba(235, 255, 235, 0.95)"),
        ("Neutral", 0.0, "rgba(235, 235, 235, 0.95)"),
        ("Liquidate", 1.5, "rgba(255, 235, 235, 0.95)"),
    ):
        figure.add_annotation(
            x=0.5,
            y=y_position,
            xref="x domain",
            yref="y",
            text=label,
            showarrow=False,
            font={"color": color, "size": 12},
            row=1,
            col=1,
        )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=2, col=1)
    figure.add_hline(y=1.0, line_dash="dash", line_color="#94a3b8", row=2, col=1)

    if latest_rows:
        table_values = [
            [row[0] for row in latest_rows],
            [pd.Timestamp(row[1]).strftime("%Y-%m-%d") for row in latest_rows],
            [f"{row[2]['Beta']:.3f}" for row in latest_rows],
            [f"{row[2]['Annual excess']:.2%}" for row in latest_rows],
            [f"{row[2]['Treynor']:.3f}" for row in latest_rows],
            [f"{row[2]['Treynor z-score']:.3f}" for row in latest_rows],
        ]
    else:
        table_values = [
            ["N/A"], ["N/A"], ["N/A"], ["N/A"], ["N/A"], ["N/A"]
        ]
    figure.add_trace(
        go.Table(
            header={
                "values": [
                    "Benchmark", "As of", "Rolling beta",
                    "Annualized excess return", "Raw Treynor", "Treynor z-score",
                ],
                "fill_color": "#1e293b",
                "font": {"color": "#f8fafc", "size": 12},
                "align": "center",
            },
            cells={
                "values": table_values,
                "fill_color": "#0f172a",
                "font": {"color": "#e2e8f0", "size": 11},
                "align": "center",
                "height": 30,
            },
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="Treynor z-score", row=1, col=1)
    figure.update_yaxes(title_text="Beta", row=2, col=1)
    figure.update_layout(
        title=(
            f"{ticker_str} Systematic Risk — {window}-Day Rolling Window<br>"
            f"<sup>Z-scores standardize each benchmark's historical Treynor series; "
            f"raw Treynor = annualized mean excess return ÷ benchmark beta; "
            f"ratios suppressed when |beta| &lt; {minimum_beta_magnitude:.2f}</sup>"
        ),
        template="plotly_dark",
        height=1550,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
        margin={"t": 115, "r": 35, "b": 35, "l": 85},
    )
    return figure


def _block18_cornish_fisher_figure(window):
    """Plot rolling Cornish-Fisher tail risk and modified Sharpe ratios."""
    window = _block18_validate_window(window)
    if window < 4:
        raise ValueError("Cornish–Fisher analysis requires a window of at least 4 returns.")

    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    returns = close.pct_change(fill_method=None).dropna()
    periodic_mar = (
        pd.to_numeric(risk_free_daily_rate, errors="coerce")
        .reindex(returns.index)
        .ffill()
        .bfill()
    )
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_excess_mean = (returns - periodic_mar).rolling(
        window, min_periods=window
    ).mean()
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=1)
    regular_sharpe = rolling_excess_mean.div(
        rolling_std.where(rolling_std > 0)
    ).replace([np.inf, -np.inf], np.nan)
    rolling_skew = returns.rolling(window, min_periods=window).skew()
    rolling_excess_kurtosis = returns.rolling(window, min_periods=window).kurt()
    confidence_levels = (0.95, 0.99)
    calculations = {}

    for confidence in confidence_levels:
        lower_tail_probability = 1.0 - confidence
        normal_quantile = float(scipy_normal.ppf(lower_tail_probability))
        adjusted_quantile = (
            normal_quantile
            + ((normal_quantile ** 2 - 1.0) / 6.0) * rolling_skew
            + ((normal_quantile ** 3 - 3.0 * normal_quantile) / 24.0)
            * rolling_excess_kurtosis
            - ((2.0 * normal_quantile ** 3 - 5.0 * normal_quantile) / 36.0)
            * rolling_skew.pow(2)
        )
        normal_var = -(rolling_mean + rolling_std * normal_quantile)
        modified_var = -(rolling_mean + rolling_std * adjusted_quantile)
        modified_sharpe = rolling_excess_mean.div(
            modified_var.where(modified_var > 0)
        ).replace([np.inf, -np.inf], np.nan)
        calculations[confidence] = {
            "normal_quantile": normal_quantile,
            "adjusted_quantile": adjusted_quantile,
            "normal_var": normal_var,
            "modified_var": modified_var,
            "modified_sharpe": modified_sharpe,
        }

    complete = pd.concat(
        {
            "skewness": rolling_skew,
            "excess_kurtosis": rolling_excess_kurtosis,
            **{
                f"modified_var_{confidence:.2f}": payload["modified_var"]
                for confidence, payload in calculations.items()
            },
        },
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if complete.empty:
        raise ValueError(
            f"No complete Cornish–Fisher estimates are available for the {window}-day window."
        )
    latest_date = pd.Timestamp(complete.index[-1])

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy", "secondary_y": True}],
            [{"type": "table"}],
        ],
        row_heights=[0.32, 0.29, 0.19, 0.20],
        vertical_spacing=0.055,
        subplot_titles=[
            "Rolling Cornish–Fisher modified Sharpe",
            "Normal VaR versus Cornish–Fisher modified VaR",
            "Higher-moment inputs driving the tail adjustment",
            f"Latest tail-risk assessment ({latest_date:%Y-%m-%d})",
        ],
    )
    confidence_colors = {0.95: "#22d3ee", 0.99: "#f97316"}
    figure.add_trace(
        go.Scatter(
            x=regular_sharpe.index,
            y=regular_sharpe,
            name="Regular Sharpe",
            mode="lines",
            line={"color": "#f8fafc", "width": 2.2, "dash": "dash"},
            hovertemplate="Regular periodic Sharpe: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for confidence, payload in calculations.items():
        confidence_label = f"{confidence:.0%}"
        color = confidence_colors[confidence]
        figure.add_trace(
            go.Scatter(
                x=payload["modified_sharpe"].index,
                y=payload["modified_sharpe"],
                name=f"{confidence_label} Modified Sharpe",
                mode="lines",
                line={"color": color, "width": 2.5},
                customdata=payload["modified_var"].reindex(
                    payload["modified_sharpe"].index
                ) * 100.0,
                hovertemplate=(
                    f"{confidence_label} modified Sharpe: %{{y:.4f}}<br>"
                    "Modified VaR: %{customdata:.3f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=payload["normal_var"].index,
                y=payload["normal_var"] * 100.0,
                name=f"{confidence_label} Normal VaR",
                mode="lines",
                line={"color": color, "width": 1.6, "dash": "dot"},
                hovertemplate=f"{confidence_label} normal VaR: %{{y:.3f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=payload["modified_var"].index,
                y=payload["modified_var"] * 100.0,
                name=f"{confidence_label} Modified VaR",
                mode="lines",
                line={"color": color, "width": 2.5},
                hovertemplate=f"{confidence_label} modified VaR: %{{y:.3f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=rolling_skew.index,
            y=rolling_skew,
            name="Skewness",
            mode="lines",
            line={"color": "#a78bfa", "width": 2},
            hovertemplate="Skewness: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=rolling_excess_kurtosis.index,
            y=rolling_excess_kurtosis,
            name="Excess kurtosis",
            mode="lines",
            line={"color": "#facc15", "width": 2},
            hovertemplate="Excess kurtosis: %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    # Add shapes before the domain-based table for older Plotly compatibility.
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=1, col=1)
    figure.add_hline(y=0.0, line_dash="dot", line_color="#64748b", row=3, col=1)

    table_rows = []
    latest_modified_vars = {}
    for confidence, payload in calculations.items():
        latest_frame = pd.concat(
            {
                "normal_var": payload["normal_var"],
                "modified_var": payload["modified_var"],
                "modified_sharpe": payload["modified_sharpe"],
                "adjusted_quantile": payload["adjusted_quantile"],
            },
            axis=1,
        ).dropna().tail(1)
        if latest_frame.empty:
            continue
        row = latest_frame.iloc[-1]
        latest_modified_vars[confidence] = float(row["modified_var"])
        table_rows.append(
            (
                f"{confidence:.0%}",
                float(payload["normal_quantile"]),
                float(row["adjusted_quantile"]),
                float(row["normal_var"]),
                float(row["modified_var"]),
                float(row["modified_sharpe"]),
            )
        )
    ordering_warning = (
        "Warning: adjusted 99% VaR is below adjusted 95% VaR"
        if (
            0.95 in latest_modified_vars
            and 0.99 in latest_modified_vars
            and latest_modified_vars[0.99] < latest_modified_vars[0.95]
        )
        else "Tail quantiles are correctly ordered"
    )
    latest_regular_sharpe = regular_sharpe.dropna().tail(1)
    latest_regular_sharpe_label = (
        f"{float(latest_regular_sharpe.iloc[-1]):.4f}"
        if not latest_regular_sharpe.empty
        else "N/A"
    )
    figure.add_trace(
        go.Table(
            columnwidth=[0.7, 0.9, 0.8, 0.9, 1.0, 1.1, 1.1, 2.1],
            header={
                "values": [
                    "Confidence", "Regular Sharpe", "Normal z", "Adjusted z",
                    "Normal VaR", "Modified VaR", "Modified Sharpe", "Diagnostic",
                ],
                "fill_color": "#1e293b",
                "font": {"color": "#f8fafc", "size": 12},
                "align": "center",
            },
            cells={
                "values": [
                    [row[0] for row in table_rows],
                    [latest_regular_sharpe_label for _ in table_rows],
                    [f"{row[1]:.3f}" for row in table_rows],
                    [f"{row[2]:.3f}" for row in table_rows],
                    [f"{row[3]:.3%}" for row in table_rows],
                    [f"{row[4]:.3%}" for row in table_rows],
                    [f"{row[5]:.4f}" for row in table_rows],
                    [ordering_warning for _ in table_rows],
                ],
                "fill_color": "#0f172a",
                "font": {"color": "#e2e8f0", "size": 11},
                "align": "center",
                "height": 32,
            },
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Excess return / modified VaR", row=1, col=1)
    figure.update_yaxes(title_text="One-day loss threshold (%)", row=2, col=1)
    figure.update_yaxes(title_text="Skewness", row=3, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Excess kurtosis", row=3, col=1, secondary_y=True)
    figure.update_layout(
        title=(
            f"{ticker_str} Cornish–Fisher Tail-Risk Analysis — {window}-Day Window<br>"
            "<sup>Modified VaR adjusts the normal one-day loss quantile for rolling skewness and excess kurtosis</sup>"
        ),
        template="plotly_dark",
        height=1900,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
        margin={"t": 120, "r": 90, "b": 35, "l": 90},
    )
    return figure


def _block18_autocorrelation_figure(
    window,
    ratio_type="sharpe",
    lag_range=None,
    return_horizon=block18_default_autocorrelation_return_horizon,
    sampling_mode=block18_default_autocorrelation_sampling_mode,
    comparison_lag=block18_default_autocorrelation_comparison_lag,
):
    """Plot return dependence and volatility-clustering ACF diagnostics."""
    window = _block18_validate_window(window)
    try:
        return_horizon = int(return_horizon)
    except (TypeError, ValueError):
        return_horizon = block18_default_autocorrelation_return_horizon
    if return_horizon < 1:
        raise ValueError("Return horizon must be at least 1 trading day.")
    sampling_mode = str(sampling_mode or "").strip().lower()
    if sampling_mode not in {"overlapping", "non_overlapping"}:
        sampling_mode = block18_default_autocorrelation_sampling_mode
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    metric_series = rolling_ratio_series(close, return_horizon, ratio_type).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(metric_series) < window:
        raise ValueError(
            f"Autocorrelation needs {window:,} clean {ratio_label} observations; "
            f"only {len(metric_series):,} are available at a {return_horizon}-day horizon."
        )
    metric_sample = metric_series.tail(window)
    if sampling_mode == "overlapping":
        metric_values = metric_sample
        sampling_label = "Overlapping"
    else:
        sample_positions = list(
            range(len(metric_sample) - 1, -1, -return_horizon)
        )[::-1]
        metric_values = metric_sample.iloc[sample_positions]
        sampling_label = "Non-overlapping"
    metric_values = metric_values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(metric_values) < 6:
        raise ValueError(
            f"Only {len(metric_values):,} usable {sampling_label.lower()} "
            f"{ratio_label} observations remain; at least 6 are required."
        )

    permitted_max_lag = max(
        1, min(block18_autocorrelation_lag_max, len(metric_values) - 1)
    )
    try:
        comparison_lag = int(comparison_lag)
    except (TypeError, ValueError):
        comparison_lag = block18_default_autocorrelation_comparison_lag
    comparison_lag = max(1, min(comparison_lag, permitted_max_lag))
    if not isinstance(lag_range, (list, tuple)) or len(lag_range) != 2:
        lag_range = block18_default_autocorrelation_lag_range
    try:
        first_lag, last_lag = sorted(int(value) for value in lag_range)
    except (TypeError, ValueError):
        first_lag, last_lag = block18_default_autocorrelation_lag_range
    first_lag = max(1, min(first_lag, permitted_max_lag))
    last_lag = max(first_lag, min(last_lag, permitted_max_lag))
    lags = np.arange(first_lag, last_lag + 1)
    effective_pairs = len(metric_values) - lags
    significance_bounds = 1.96 / np.sqrt(effective_pairs)
    diagnostic_series = {
        ratio_label: metric_values,
        f"Absolute {ratio_label}": metric_values.abs(),
        f"Squared {ratio_label}": metric_values.pow(2),
    }
    figure = make_subplots(
        rows=2,
        cols=3,
        shared_yaxes="rows",
        row_heights=[0.48, 0.52],
        vertical_spacing=0.12,
        horizontal_spacing=0.055,
        subplot_titles=[
            f"{ratio_label} — ACF",
            f"Absolute {ratio_label} — ACF",
            f"Squared {ratio_label} — ACF",
            f"{ratio_label} — current vs lag {comparison_lag}",
            f"Absolute {ratio_label} — current vs lag {comparison_lag}",
            f"Squared {ratio_label} — current vs lag {comparison_lag}",
        ],
    )
    diagnostic_colors = {
        ratio_label: "#38bdf8",
        f"Absolute {ratio_label}": "#a78bfa",
        f"Squared {ratio_label}": "#f97316",
    }
    significant_counts = {}
    maximum_absolute_correlation = float(np.max(significance_bounds))
    for col, (series_label, values_series) in enumerate(
        diagnostic_series.items(), start=1
    ):
        numeric_values = values_series.to_numpy(dtype=float)
        acf_values = statsmodels_acf(
            numeric_values, nlags=last_lag, fft=True, adjusted=False
        )[first_lag:last_lag + 1]
        finite_values = np.asarray(acf_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            maximum_absolute_correlation = max(
                maximum_absolute_correlation,
                float(np.max(np.abs(finite_values))),
            )
        significant = np.abs(acf_values) > significance_bounds
        adequate_pairs = effective_pairs >= 30
        significant_counts[series_label] = int((significant & adequate_pairs).sum())
        color = diagnostic_colors[series_label]
        bar_colors = np.where(
            ~adequate_pairs,
            "#334155",
            np.where(significant, color, "#64748b"),
        )
        customdata = np.column_stack([
            effective_pairs, significance_bounds, significant, adequate_pairs
        ])
        figure.add_trace(
            go.Bar(
                x=lags,
                y=acf_values,
                name=f"{series_label} ACF",
                marker_color=bar_colors,
                customdata=customdata,
                hovertemplate=(
                    f"{series_label} ACF lag %{{x}}: %{{y:.4f}}<br>"
                    "Effective pairs: %{customdata[0]:.0f}<br>"
                    "95% bound: ±%{customdata[1]:.4f}<br>"
                    "Outside band: %{customdata[2]}<br>"
                    "At least 30 pairs: %{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=lags,
                y=significance_bounds,
                mode="lines",
                line={"color": "rgba(226, 232, 240, 0.60)", "width": 1, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=lags,
                y=-significance_bounds,
                mode="lines",
                line={"color": "rgba(226, 232, 240, 0.60)", "width": 1, "dash": "dash"},
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.16)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        figure.add_hline(
            y=0.0,
            line_color="rgba(226, 232, 240, 0.70)",
            line_width=1,
            row=1,
            col=col,
        )
        comparison_multiplier = 1.0
        comparison_units = ""
        comparison_values = values_series * comparison_multiplier
        figure.add_trace(
            go.Scatter(
                x=comparison_values.index,
                y=comparison_values,
                name=f"{series_label} current",
                mode="lines",
                line={"color": color, "width": 1.8},
                hovertemplate=(
                    f"{series_label} current: %{{y:.4f}}{comparison_units}<extra></extra>"
                ),
            ),
            row=2,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=comparison_values.index,
                y=comparison_values.shift(comparison_lag),
                name=f"{series_label} lag {comparison_lag}",
                mode="lines",
                line={"color": "#f8fafc", "width": 1.5, "dash": "dash"},
                hovertemplate=(
                    f"{series_label} lag {comparison_lag}: "
                    f"%{{y:.4f}}{comparison_units}<extra></extra>"
                ),
            ),
            row=2,
            col=col,
        )

    y_axis_limit = min(
        1.0,
        max(0.20, 1.15 * maximum_absolute_correlation),
    )
    for col in range(1, 4):
        figure.update_yaxes(
            title_text="ACF" if col == 1 else None,
            range=[-y_axis_limit, y_axis_limit],
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="Lag", row=1, col=col)
        figure.update_xaxes(title_text="Date", row=2, col=col)
        figure.update_yaxes(
            title_text=ratio_label if col == 1 else None,
            row=2,
            col=col,
        )
    volatility_cluster_count = (
        significant_counts[f"Absolute {ratio_label}"]
        + significant_counts[f"Squared {ratio_label}"]
    )
    figure.update_layout(
        title=(
            f"{ticker_str} {ratio_label} Dependence — {window}-Observation History<br>"
            f"<sup>{sampling_label} {return_horizon}-day {ratio_label} "
            f"({len(metric_values)} observations); "
            f"lags {first_lag}–{last_lag}; lag-specific approximate 95% bounds; "
            f"comparison lag: {comparison_lag} return observations; "
            f"metric significant ACF lags: {significant_counts[ratio_label]}; "
            f"absolute+squared metric significant ACF lags: {volatility_cluster_count}; "
            "dark bars have fewer than 30 effective pairs</sup>"
        ),
        template="plotly_dark",
        height=1000,
        hovermode="x unified",
        showlegend=False,
        bargap=0.18,
        margin={"t": 105, "r": 30, "b": 45, "l": 70},
    )
    return figure


def _block18_compute_first_passage_events(
    standardized_series,
    threshold,
    end_threshold,
    start_condition,
    end_condition,
    start_sign,
    end_sign,
):
    """Return non-overlapping first-passage events for an explicit boundary rule."""
    series = pd.to_numeric(pd.Series(standardized_series), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna().sort_index()
    if len(series) < 3:
        return []
    threshold = abs(float(threshold))
    end_threshold = abs(float(end_threshold))
    start_condition = start_condition if start_condition in {"above", "below"} else "above"
    end_condition = end_condition if end_condition in {"above", "below"} else "below"
    start_sign = start_sign if start_sign in {"positive", "negative", "both"} else "both"
    end_sign = end_sign if end_sign in {"positive", "negative", "both"} else "both"

    values = series.to_numpy(dtype=float)
    dates = series.index
    events = []
    position = 1
    while position < len(values):
        previous_start_value = (
            abs(values[position - 1]) if start_sign == "both" else values[position - 1]
        )
        current_start_value = (
            abs(values[position]) if start_sign == "both" else values[position]
        )
        start_boundary = -threshold if start_sign == "negative" else threshold
        crossed = (
            previous_start_value <= start_boundary < current_start_value
            if start_condition == "above"
            else previous_start_value >= start_boundary > current_start_value
        )
        if not crossed:
            position += 1
            continue

        start_position = position
        direction = "upper" if values[start_position] >= 0 else "lower"
        recovery_position = None
        for candidate in range(start_position + 1, len(values)):
            if end_sign == "both" and end_threshold == 0 and end_condition == "below":
                completed = (
                    values[candidate] <= 0
                    if direction == "upper"
                    else values[candidate] >= 0
                )
            else:
                candidate_value = (
                    abs(values[candidate]) if end_sign == "both" else values[candidate]
                )
                end_boundary = -end_threshold if end_sign == "negative" else end_threshold
                completed = (
                    candidate_value >= end_boundary
                    if end_condition == "above"
                    else candidate_value <= end_boundary
                )
            if completed:
                recovery_position = candidate
                break

        end_position = recovery_position if recovery_position is not None else len(values) - 1
        start_date = dates[start_position]
        end_date = dates[end_position]
        events.append({
            "direction": direction,
            "start_position": start_position,
            "end_position": end_position,
            "start_date": start_date,
            "end_date": end_date,
            "start_zscore": values[start_position],
            "end_zscore": values[end_position],
            "duration_observations": (
                recovery_position - start_position
                if recovery_position is not None else np.nan
            ),
            "duration_calendar_days": (
                (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
                if recovery_position is not None else np.nan
            ),
            "completed": recovery_position is not None,
        })
        if recovery_position is None:
            break
        position = recovery_position + 1
    return events


def _block18_first_passage_figure(
    window, ratio_type="sharpe", threshold=block18_default_first_passage_threshold,
    end_threshold=block18_default_first_passage_end_threshold,
    start_condition=block18_default_first_passage_start_condition,
    end_condition=block18_default_first_passage_end_condition,
    start_sign=block18_default_first_passage_start_sign,
    end_sign=block18_default_first_passage_end_sign,
):
    """Measure time from an extreme metric crossing to its first mean return."""
    window = _block18_validate_window(window)
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = block18_default_first_passage_threshold
    if not np.isfinite(threshold):
        raise ValueError("The first-passage starting boundary must be finite.")
    try:
        end_threshold = float(end_threshold)
    except (TypeError, ValueError):
        end_threshold = block18_default_first_passage_end_threshold
    if not np.isfinite(end_threshold):
        raise ValueError("The first-passage ending boundary must be finite.")
    start_condition = start_condition if start_condition in {"above", "below"} else "above"
    end_condition = end_condition if end_condition in {"above", "below"} else "below"
    start_sign = start_sign if start_sign in {"positive", "negative", "both"} else "both"
    end_sign = end_sign if end_sign in {"positive", "negative", "both"} else "both"
    threshold = abs(threshold)
    end_threshold = abs(end_threshold)

    close = pd.to_numeric(asset_history["Close"], errors="coerce").dropna().sort_index()
    metric_series = rolling_ratio_series(close, window, ratio_type).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    metric_zscore = zscore_or_empty(metric_series)
    if len(metric_zscore) < 3:
        raise ValueError(
            f"Not enough {ratio_label} history is available for first-passage analysis."
        )

    all_events = _block18_compute_first_passage_events(
        metric_zscore, threshold, end_threshold,
        start_condition, end_condition, start_sign, end_sign,
    )
    upper_events = [event for event in all_events if event["direction"] == "upper"]
    lower_events = [event for event in all_events if event["direction"] == "lower"]
    figure = make_subplots(
        rows=4,
        cols=1,
        specs=[
            [{"type": "xy"}], [{"type": "xy"}],
            [{"type": "xy"}], [{"type": "table"}],
        ],
        row_heights=[0.43, 0.24, 0.16, 0.17],
        vertical_spacing=0.065,
        subplot_titles=[
            f"{ratio_label} z-score and conditional first passages",
            "Historical first-passage times and expanding indicators",
            "Completed first-passage duration distribution",
            "Excursion summary",
        ],
    )
    figure.add_trace(
        go.Scatter(
            x=metric_zscore.index,
            y=metric_zscore,
            name=f"{ratio_label} Z-Score",
            mode="lines",
            line={"color": "#38bdf8", "width": 1.8},
            hovertemplate=f"{ratio_label} z-score: %{{y:.3f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    event_styles = {
        "upper": ("#ef4444", "triangle-down", f"Crossed above {threshold:g}"),
        "lower": ("#22c55e", "triangle-up", f"Crossed below {threshold:g}"),
    }
    for direction, events in (("upper", upper_events), ("lower", lower_events)):
        color, symbol, label = event_styles[direction]
        if events:
            figure.add_trace(
                go.Scatter(
                    x=[event["start_date"] for event in events],
                    y=[event["start_zscore"] for event in events],
                    name=label,
                    mode="markers",
                    marker={"color": color, "size": 10, "symbol": symbol},
                    customdata=[
                        [event["duration_observations"], event["completed"]]
                        for event in events
                    ],
                    hovertemplate=(
                        f"{label}<br>Start z-score: %{{y:.3f}}<br>"
                        "Passage observations: %{customdata[0]}<br>"
                        "Completed: %{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        completed = [event for event in events if event["completed"]]
        if completed:
            figure.add_trace(
                go.Histogram(
                    x=[event["duration_observations"] for event in completed],
                    name=f"{direction.title()} excursions",
                    marker_color=color,
                    opacity=0.68,
                    hovertemplate="Passage time: %{x} observations<br>Count: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )

    completed_history = sorted(
        [event for event in all_events if event["completed"]],
        key=lambda event: event["end_date"],
    )
    if completed_history:
        history_dates = [event["end_date"] for event in completed_history]
        history_durations = pd.Series(
            [event["duration_observations"] for event in completed_history],
            index=pd.DatetimeIndex(history_dates),
            dtype=float,
        )
        history_colors = [
            "#ef4444" if event["direction"] == "upper" else "#22c55e"
            for event in completed_history
        ]
        figure.add_trace(
            go.Scatter(
                x=history_durations.index, y=history_durations,
                name="Completed passage time", mode="markers",
                marker={"color": history_colors, "size": 8, "opacity": 0.8},
                hovertemplate="Completed: %{x|%Y-%m-%d}<br>Passage time: %{y:.0f} observations<extra></extra>",
            ), row=2, col=1,
        )
        historical_indicators = (
            ("Expanding mean", history_durations.expanding().mean(), "#38bdf8"),
            ("Expanding median", history_durations.expanding().median(), "#fbbf24"),
            ("Expanding maximum", history_durations.expanding().max(), "#c084fc"),
        )
        for indicator_name, indicator_values, indicator_color in historical_indicators:
            figure.add_trace(
                go.Scatter(
                    x=indicator_values.index, y=indicator_values,
                    name=indicator_name, mode="lines",
                    line={"color": indicator_color, "width": 2},
                    hovertemplate=f"{indicator_name}: %{{y:.1f}} observations<extra></extra>",
                ), row=2, col=1,
            )

    # Add guides before the table trace for older Plotly compatibility.
    figure.add_hline(y=0.0, line_color="#f8fafc", line_width=1.4, row=1, col=1)
    if start_sign in {"positive", "both"}:
        figure.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", row=1, col=1)
    if start_sign in {"negative", "both"}:
        figure.add_hline(y=-threshold, line_dash="dash", line_color="#22c55e", row=1, col=1)
    if end_sign in {"positive", "both"}:
        figure.add_hline(y=end_threshold, line_dash="dot", line_color="#fbbf24", row=1, col=1)
    if end_sign in {"negative", "both"}:
        figure.add_hline(y=-end_threshold, line_dash="dot", line_color="#fbbf24", row=1, col=1)

    summary_rows = []
    summary_event_groups = [("Positive", upper_events), ("Negative", lower_events)]
    for direction, events in summary_event_groups:
        completed = [event for event in events if event["completed"]]
        durations = np.asarray(
            [event["duration_observations"] for event in completed], dtype=float
        )
        summary_rows.append([
            direction,
            len(events),
            len(completed),
            sum(not event["completed"] for event in events),
            f"{np.mean(durations):.1f}" if durations.size else "N/A",
            f"{np.median(durations):.1f}" if durations.size else "N/A",
            f"{np.max(durations):.0f}" if durations.size else "N/A",
        ])
    figure.add_trace(
        go.Table(
            header={
                "values": [
                    "Direction", "Crossings", "Completed", "Open",
                    "Mean time", "Median time", "Maximum time",
                ],
                "fill_color": "#1e293b",
                "font": {"color": "#f8fafc", "size": 12},
                "align": "center",
            },
            cells={
                "values": list(map(list, zip(*summary_rows))),
                "fill_color": "#0f172a",
                "font": {"color": "#e2e8f0", "size": 11},
                "align": "center",
                "height": 30,
            },
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Z-Score", row=1, col=1)
    figure.update_yaxes(title_text="Passage observations", row=2, col=1)
    figure.update_xaxes(title_text="Completion date", row=2, col=1)
    figure.update_xaxes(title_text="Passage time (metric observations)", row=3, col=1)
    figure.update_yaxes(title_text="Event count", row=3, col=1)
    sign_labels = {"positive": "+", "negative": "-", "both": "+/-"}
    condition_labels = {"above": ">", "below": "<"}
    figure.update_layout(
        title=(
            f"{ticker_str} {ratio_label} First-Passage Analysis — {window}-Day Metric Window<br>"
            f"<sup>Start ({sign_labels[start_sign]}, {condition_labels[start_condition]}, {threshold:g} sigma); "
            f"end ({sign_labels[end_sign]}, {condition_labels[end_condition]}, {end_threshold:g} sigma)</sup>"
        ),
        template="plotly_dark",
        height=1550,
        hovermode="x unified",
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1.0},
        margin={"t": 115, "r": 35, "b": 35, "l": 80},
    )
    return figure


def _block18_risk_first_passage_series(window, selected_benchmarks, ratio_type):
    """Return the ticker-owned series represented by the three risk panels."""
    window = _block18_validate_window(window)
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    asset_close = _block18_close_series(asset_history, ticker_str)
    asset_ratio = rolling_ratio_series(asset_close, window, ratio_type).dropna()

    requested = _block18_normalize_benchmark_selection(selected_benchmarks)
    benchmark_symbol = None
    benchmark_ratio = pd.Series(dtype=float)
    for candidate in requested:
        try:
            candidate_close = _block18_close_series(benchmark_data.get(candidate), candidate)
        except ValueError:
            continue
        if len(candidate_close) < window + 2:
            continue
        candidate_ratio = rolling_ratio_series(candidate_close, window, ratio_type).dropna()
        if _is_benchmark_relative_metric(ratio_type):
            metric_name, reference_symbol = _benchmark_relative_metric_parts(ratio_type)
            if candidate == reference_symbol and metric_name in {"appraisal", "information"}:
                candidate_ratio = pd.Series(0.0, index=candidate_close.index)
        benchmark_symbol = candidate
        benchmark_ratio = candidate_ratio
        break
    if requested and benchmark_symbol is None:
        raise ValueError("No selected benchmark has enough history for first-passage analysis.")

    if "ticker_daily_returns" in globals():
        _, asset_mad_scores = _block18_rolling_mean_metric_frames(
            ticker_daily_returns, window, values_are_returns=True
        )
    else:
        _, asset_mad_scores = _block18_rolling_mean_metric_frames(asset_close, window)
    volatility_drag = asset_mad_scores.get(
        "Volatility Drag", pd.Series(dtype=float)
    ).dropna()

    return [
        {
            "key": "risk_adjusted_return",
            "row": 1,
            "title": f"{ticker_str} {ratio_label} Z-Score",
            "series": benchmark_zscore_for_plot(asset_ratio),
        },
        {
            "key": "ratio_spread",
            "row": 2,
            "title": (
                f"{benchmark_symbol} - {ticker_str} {ratio_label} Spread Z-Score"
                if benchmark_symbol is not None
                else f"{ratio_label} Spread Z-Score (select a benchmark)"
            ),
            "series": (
                benchmark_zscore_for_plot(benchmark_ratio - asset_ratio)
                if benchmark_symbol is not None else pd.Series(dtype=float)
            ),
        },
        {
            "key": "volatility_drag",
            "row": 3,
            "title": f"{ticker_str} Volatility Drag MAD Score",
            "series": volatility_drag,
        },
    ]


def _block18_boundary_levels(sign, magnitude):
    magnitude = abs(float(magnitude))
    if sign == "positive":
        return [magnitude]
    if sign == "negative":
        return [-magnitude]
    return sorted(set([-magnitude, magnitude]))


def _block18_overlay_first_passage_on_risk_figure(
    risk_figure,
    metric_specs,
    threshold,
    end_threshold,
    start_condition,
    end_condition,
    start_sign,
    end_sign,
):
    """Overlay boundary rules and event markers on each original risk panel."""
    figure = go.Figure(risk_figure)
    for metric_spec in metric_specs:
        row = metric_spec["row"]
        metric_series = pd.to_numeric(
            pd.Series(metric_spec["series"]), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(metric_series) < 3:
            continue
        events = _block18_compute_first_passage_events(
            metric_series, threshold, end_threshold,
            start_condition, end_condition, start_sign, end_sign,
        )
        for level in _block18_boundary_levels(start_sign, threshold):
            figure.add_hline(
                y=level, row=row, col=1, line_dash="dash", line_width=1.5,
                line_color="rgba(244, 63, 94, 0.90)", layer="above",
            )
        for level in _block18_boundary_levels(end_sign, end_threshold):
            figure.add_hline(
                y=level, row=row, col=1, line_dash="dot", line_width=1.5,
                line_color="rgba(251, 191, 36, 0.90)", layer="above",
            )
        if not events:
            continue
        figure.add_trace(
            go.Scatter(
                x=[event["start_date"] for event in events],
                y=[event["start_zscore"] for event in events],
                mode="markers", showlegend=False,
                name=f"{metric_spec['title']} passage starts",
                marker={
                    "size": 9,
                    "symbol": "diamond",
                    "color": [
                        "#ef4444" if event["direction"] == "upper" else "#22c55e"
                        for event in events
                    ],
                    "line": {"color": "#f8fafc", "width": 0.7},
                },
                customdata=[
                    [event["completed"], event["duration_observations"]]
                    for event in events
                ],
                hovertemplate=(
                    "Passage start<br>%{x|%Y-%m-%d}<br>Score: %{y:.3f}<br>"
                    "Completed: %{customdata[0]}<br>Duration: %{customdata[1]} observations"
                    "<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        completed = [event for event in events if event["completed"]]
        if completed:
            figure.add_trace(
                go.Scatter(
                    x=[event["end_date"] for event in completed],
                    y=[event["end_zscore"] for event in completed],
                    mode="markers", showlegend=False,
                    name=f"{metric_spec['title']} passage completions",
                    marker={
                        "size": 8, "symbol": "x", "color": "#fbbf24",
                        "line": {"color": "#fef3c7", "width": 1},
                    },
                    customdata=[[event["duration_observations"]] for event in completed],
                    hovertemplate=(
                        "Passage complete<br>%{x|%Y-%m-%d}<br>Score: %{y:.3f}<br>"
                        "Duration: %{customdata[0]:.0f} observations<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )
    sign_labels = {"positive": "+", "negative": "-", "both": "+/-"}
    condition_labels = {"above": ">", "below": "<"}
    figure.update_layout(
        title=(
            f"{figure.layout.title.text}<br>"
            f"<sup>Integrated first passage: start "
            f"({sign_labels[start_sign]}, {condition_labels[start_condition]}, {abs(float(threshold)):g}); "
            f"end ({sign_labels[end_sign]}, {condition_labels[end_condition]}, {abs(float(end_threshold)):g}). "
            "Diamonds=start; gold crosses=completion.</sup>"
        ),
        margin={**figure.layout.margin.to_plotly_json(), "t": 180},
    )
    return figure


def _block18_first_passage_analytics_figure(
    standardized_series,
    metric_title,
    threshold,
    end_threshold,
    start_condition,
    end_condition,
    start_sign,
    end_sign,
):
    """Compact nonredundant passage history, distribution, and data table."""
    events = _block18_compute_first_passage_events(
        standardized_series, threshold, end_threshold,
        start_condition, end_condition, start_sign, end_sign,
    )
    upper_events = [event for event in events if event["direction"] == "upper"]
    lower_events = [event for event in events if event["direction"] == "lower"]
    completed_history = sorted(
        [event for event in events if event["completed"]],
        key=lambda event: event["end_date"],
    )
    figure = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "table"}]],
        column_widths=[0.48, 0.22, 0.30], horizontal_spacing=0.055,
        subplot_titles=(
            "Historical passage time and expanding indicators",
            "Duration distribution",
            "First-passage data",
        ),
    )
    if completed_history:
        completion_dates = pd.DatetimeIndex(
            [event["end_date"] for event in completed_history]
        )
        durations = pd.Series(
            [event["duration_observations"] for event in completed_history],
            index=completion_dates, dtype=float,
        )
        figure.add_trace(
            go.Scatter(
                x=durations.index, y=durations, mode="markers",
                name="Completed passages",
                marker={
                    "size": 7,
                    "color": [
                        "#ef4444" if event["direction"] == "upper" else "#22c55e"
                        for event in completed_history
                    ],
                },
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} observations<extra></extra>",
            ), row=1, col=1,
        )
        for name, values, color in (
            ("Expanding mean", durations.expanding().mean(), "#38bdf8"),
            ("Expanding median", durations.expanding().median(), "#fbbf24"),
            ("Expanding maximum", durations.expanding().max(), "#c084fc"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=values.index, y=values, mode="lines", name=name,
                    line={"color": color, "width": 2},
                    hovertemplate=f"{name}: %{{y:.1f}} observations<extra></extra>",
                ), row=1, col=1,
            )
    else:
        figure.add_annotation(
            text="No completed passages for this rule", showarrow=False,
            font={"color": "#94a3b8", "size": 12}, row=1, col=1,
        )

    for direction, direction_events, color in (
        ("Positive", upper_events, "#ef4444"),
        ("Negative", lower_events, "#22c55e"),
    ):
        completed = [event for event in direction_events if event["completed"]]
        if completed:
            figure.add_trace(
                go.Histogram(
                    x=[event["duration_observations"] for event in completed],
                    name=f"{direction} duration", marker_color=color, opacity=0.65,
                    hovertemplate="%{x} observations<br>Count: %{y}<extra></extra>",
                ), row=1, col=2,
            )

    summary_rows = []
    for direction, direction_events in (
        ("Positive", upper_events), ("Negative", lower_events)
    ):
        completed = [event for event in direction_events if event["completed"]]
        durations = np.asarray(
            [event["duration_observations"] for event in completed], dtype=float
        )
        summary_rows.append([
            direction,
            len(direction_events),
            len(completed),
            sum(not event["completed"] for event in direction_events),
            f"{np.mean(durations):.1f}" if durations.size else "N/A",
            f"{np.median(durations):.1f}" if durations.size else "N/A",
            f"{np.max(durations):.0f}" if durations.size else "N/A",
        ])
    figure.add_trace(
        go.Table(
            columnwidth=[1.15, 0.8, 0.9, 0.65, 0.8, 0.85, 0.75],
            header={
                "values": ["Side", "Events", "Done", "Open", "Mean", "Median", "Max"],
                "fill_color": "#1e293b", "align": "center",
                "font": {"color": "#f8fafc", "size": 10},
            },
            cells={
                "values": list(map(list, zip(*summary_rows))),
                "fill_color": "#0f172a", "align": "center", "height": 29,
                "font": {"color": "#e2e8f0", "size": 10},
            },
        ), row=1, col=3,
    )
    sign_labels = {"positive": "+", "negative": "-", "both": "+/-"}
    condition_labels = {"above": ">", "below": "<"}
    figure.update_xaxes(title_text="Completion date", row=1, col=1)
    figure.update_yaxes(title_text="Observations", row=1, col=1)
    figure.update_xaxes(title_text="Passage observations", row=1, col=2)
    figure.update_yaxes(title_text="Count", row=1, col=2)
    figure.update_layout(
        title=(
            f"{metric_title} First-Passage Analysis<br>"
            f"<sup>Start ({sign_labels[start_sign]}, {condition_labels[start_condition]}, {abs(float(threshold)):g}); "
            f"end ({sign_labels[end_sign]}, {condition_labels[end_condition]}, {abs(float(end_threshold)):g})</sup>"
        ),
        template="plotly_dark", height=610, barmode="overlay",
        hovermode="closest", margin={"t": 115, "r": 30, "b": 50, "l": 65},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return _block18_apply_typography(figure)


def _block18_build_tab_figure(
    active_tab, window, display_range_value, selected_benchmarks,
    ratio_type="sharpe", autocorrelation_lag_range=None,
    autocorrelation_return_horizon=block18_default_autocorrelation_return_horizon,
    autocorrelation_sampling_mode=block18_default_autocorrelation_sampling_mode,
    autocorrelation_comparison_lag=block18_default_autocorrelation_comparison_lag,
    first_passage_threshold=block18_default_first_passage_threshold,
    first_passage_end_threshold=block18_default_first_passage_end_threshold,
    first_passage_start_condition=block18_default_first_passage_start_condition,
    first_passage_end_condition=block18_default_first_passage_end_condition,
    first_passage_start_sign=block18_default_first_passage_start_sign,
    first_passage_end_sign=block18_default_first_passage_end_sign,
):
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    if active_tab == "volatility_efficiency":
        return _block18_volatility_efficiency_distribution_figure(window)
    if active_tab == "risk":
        return build_block18_decomposition_figure(
            window, selected_benchmarks, ratio_type
        )
    if active_tab == "systematic_risk":
        return _block18_systematic_risk_figure(window, selected_benchmarks)
    if active_tab == "kappa":
        return _block18_kappa_profile_figure(window)
    if active_tab == "pezier_white":
        return _block18_pezier_white_figure(window)
    if active_tab == "cornish_fisher":
        return _block18_cornish_fisher_figure(window)
    if active_tab == "autocorrelation":
        return _block18_autocorrelation_figure(
            window,
            ratio_type,
            autocorrelation_lag_range,
            autocorrelation_return_horizon,
            autocorrelation_sampling_mode,
            autocorrelation_comparison_lag,
        )
    if active_tab == "first_passage":
        return _block18_first_passage_figure(
            window, ratio_type, first_passage_threshold, first_passage_end_threshold,
            first_passage_start_condition, first_passage_end_condition,
            first_passage_start_sign, first_passage_end_sign,
        )
    if active_tab == "upside":
        term_label = f"{window}-day"
        return plot_sharpe_sortino_comparison(
            term_config_map=_block18_build_term_config_for_window(window),
            ticker_label=ticker_str,
            default_label=term_label,
        )
    if active_tab == "drawdown":
        return plot_candlestick_drawdown_recovery_view(
            price_frame=asset_history,
            drawdown_recovery_by_window=_block18_drawdown_recovery_for_window(window),
            ticker_label=ticker_str,
            candlestick_period=period,
            default_window=window,
            show_window_menu=False,
            default_timeframe_label=_block18_display_range_label(display_range_value),
        )
    if active_tab == "correlation":
        term_label = f"{window}-day"
        return plot_rolling_correlation_view(
            rolling_correlation_map=_block18_rolling_correlation_for_window(window, selected_benchmarks),
            time_frame_map={term_label: window},
            term_order=[term_label],
            benchmark_order=selected_benchmarks,
            ticker_label=ticker_str,
        )
    if active_tab == "seasonality":
        return _block18_seasonality_figure(ratio_type)
    if active_tab == "heatmap":
        return plot_sharpe_zscore_heatmap_view(
            asset_sharpe_zscore_frame=asset_sharpe_zscore_frame,
            benchmark_sharpe_zscore_frames={
                symbol: benchmark_sharpe_zscore_frames[symbol]
                for symbol in selected_benchmarks if symbol in benchmark_sharpe_zscore_frames
            },
            benchmark_spread_zscore_frames={
                symbol: benchmark_spread_zscore_frames[symbol]
                for symbol in selected_benchmarks if symbol in benchmark_spread_zscore_frames
            },
            benchmark_order=selected_benchmarks,
            default_benchmark=selected_benchmarks[0] if selected_benchmarks else None,
            ticker_label=ticker_str,
        )
    if active_tab == "historical_surface_3d":
        return _block18_historical_ratio_surface_3d(
            selected_benchmarks, ratio_type
        )
    if active_tab == "window_diagnostics":
        _block18_prepare_window_diagnostics_playback(ratio_type)
        return _block18_filter_benchmark_traces(
            _block18_ratio_diagnostics_source_figure(ratio_type),
            selected_benchmarks,
        )
    return _block18_error_figure("Select a valid Momentum & Efficiency view.", height=850)

def _block18_expected_factor_benchmarks():
    if not globals().get("include_factor_peer_index", False):
        return {}
    lookup_symbol = globals().get("factor_peer_index_source_ticker") or ticker_str
    safe_symbol = str(lookup_symbol).replace(".", "_").replace("/", "_")
    expected = {}
    for gics_level, file_suffix in factor_gics_index_suffixes.items():
        cache_path = Path(peer_index_cache_dir) / f"{safe_symbol}_{file_suffix}_index.csv"
        if not cache_path.exists():
            continue
        cached_header = pd.read_csv(cache_path, nrows=1)
        label = factor_index_display_label(
            cached_header, f"{lookup_symbol} {gics_level} Index"
        )
        expected[label] = cache_path
    if expected:
        return expected

    legacy_path = Path(peer_index_cache_dir) / f"{safe_symbol}_peer_index.csv"
    if legacy_path.exists():
        cached_header = pd.read_csv(legacy_path, nrows=1)
        label = factor_index_display_label(
            cached_header, f"{lookup_symbol} Peer Index"
        )
        expected[label] = legacy_path
    return expected

def _block18_validate_peer_state(active_tab, figure=None, required_benchmarks=None):
    expected_benchmarks = _block18_expected_factor_benchmarks()
    if not expected_benchmarks:
        return
    benchmark_candidates = (
        expected_benchmarks if required_benchmarks is None else required_benchmarks
    )
    required_labels = [
        label for label in benchmark_candidates
        if label in expected_benchmarks
    ]
    missing_labels = [label for label in required_labels if label not in benchmark_data]
    if missing_labels:
        raise RuntimeError(
            f"Factor GICS indexes exist but are not loaded in benchmark_data: {missing_labels}. "
            "Restart the kernel and Run All, or rerun Block 5 and every downstream source cell before Block 18."
        )
    if active_tab not in {"heatmap", "historical_surface_3d", "window_diagnostics"} or figure is None:
        return
    trace_names = [str(getattr(trace, "name", "")) for trace in figure.data]
    missing_trace_labels = [
        label for label in required_labels
        if not any(label in trace_name for trace_name in trace_names)
    ]
    if missing_trace_labels:
        source_block = "Block 12" if active_tab == "heatmap" else "Block 11"
        raise RuntimeError(
            f"Factor indexes {missing_trace_labels} are loaded but the cached "
            f"{block18_tab_config[active_tab]['label']} figure is stale. "
            f"Rerun {source_block}, then rerun Block 18."
        )

def _block18_normalize_benchmark_selection(selected_benchmarks):
    if isinstance(selected_benchmarks, str):
        selected_benchmarks = [selected_benchmarks]
    selected = [
        symbol for symbol in (selected_benchmarks or []) if symbol in benchmark_order
    ]
    return list(dict.fromkeys(selected))

def _block18_benchmark_selection_label(selected_benchmarks):
    return ", ".join(selected_benchmarks) if selected_benchmarks else "None"

def _block18_normalize_diagnostics_window_range(window_range):
    if not isinstance(window_range, (list, tuple)) or len(window_range) != 2:
        return block18_default_diagnostics_window_range.copy()
    try:
        lower, upper = sorted(int(value) for value in window_range)
    except (TypeError, ValueError):
        return block18_default_diagnostics_window_range.copy()
    lower = max(block18_diagnostics_window_min, lower)
    upper = min(block18_diagnostics_window_max, upper)
    if lower == upper and block18_diagnostics_window_min < block18_diagnostics_window_max:
        if upper < block18_diagnostics_window_max:
            upper += 1
        else:
            lower -= 1
    return [lower, max(lower, upper)]


def _block18_horizon_range_from_inputs(minimum_value, maximum_value):
    try:
        minimum_value = int(minimum_value)
    except (TypeError, ValueError):
        minimum_value = block18_default_diagnostics_window_range[0]
    try:
        maximum_value = int(maximum_value)
    except (TypeError, ValueError):
        maximum_value = block18_default_diagnostics_window_range[1]
    return _block18_normalize_diagnostics_window_range(
        [minimum_value, maximum_value]
    )


def _block18_visible_horizon_region(region_start, region_end, lower, upper):
    """Return the portion of a horizon region visible in the selected range."""
    left = max(float(lower), float(region_start))
    right = min(float(upper), float(region_end))
    return left, right, right > left


def _block18_segment_horizon_mask(horizons, segment_start, segment_end):
    """Assign horizons once using right-closed (start, end] segment bounds."""
    numeric_horizons = np.asarray(horizons, dtype=float)
    selected = np.isfinite(numeric_horizons)
    if segment_start is not None:
        selected &= numeric_horizons > float(segment_start)
    if segment_end is not None:
        selected &= numeric_horizons <= float(segment_end)
    return selected


def _block18_apply_diagnostics_window_range(figure, window_range):
    lower, upper = _block18_normalize_diagnostics_window_range(window_range)
    profile_axes = {"x5", "x6", "x7"}
    for trace in figure.data:
        if (getattr(trace, "xaxis", None) or "x") not in profile_axes:
            continue
        x_values = getattr(trace, "x", None)
        y_values = getattr(trace, "y", None)
        if x_values is None:
            continue
        numeric_x = pd.to_numeric(pd.Series(list(x_values)), errors="coerce").to_numpy(dtype=float)
        selected = np.isfinite(numeric_x) & (numeric_x >= lower) & (numeric_x <= upper)
        if len(selected) == 0:
            continue
        trace.x = np.asarray(x_values)[selected]
        if y_values is not None and len(y_values) == len(selected):
            trace.y = np.asarray(y_values)[selected]
        customdata = getattr(trace, "customdata", None)
        if customdata is not None and len(customdata) == len(selected):
            trace.customdata = np.asarray(customdata)[selected]
    annotation_prefix = "quantapp-horizon-region-label|"
    for annotation in figure.layout.annotations or []:
        annotation_name = str(getattr(annotation, "name", "") or "")
        if not annotation_name.startswith(annotation_prefix):
            continue
        try:
            region_start, region_end = (
                float(value)
                for value in annotation_name[len(annotation_prefix):].split("|", 1)
            )
        except (TypeError, ValueError):
            continue
        visible_left, visible_right, is_visible = _block18_visible_horizon_region(
            region_start, region_end, lower, upper
        )
        annotation.visible = is_visible
        if is_visible:
            annotation.x = (visible_left + visible_right) / 2.0
    for row in (3, 4, 5):
        figure.update_xaxes(range=[lower, upper], autorange=False, row=row, col=1)
    return figure, [lower, upper]


def _block18_apply_horizon_range(figure, active_tab, window_range):
    """Apply one shared horizon selection to the first three dashboard views."""
    lower, upper = _block18_normalize_diagnostics_window_range(window_range)
    if active_tab == "window_diagnostics":
        return _block18_apply_diagnostics_window_range(figure, [lower, upper])
    if active_tab == "heatmap":
        for trace in figure.data:
            if getattr(trace, "type", None) != "heatmap":
                continue
            y_values = getattr(trace, "y", None)
            z_values = getattr(trace, "z", None)
            if y_values is None or z_values is None:
                continue
            numeric_y = pd.to_numeric(pd.Series(list(y_values)), errors="coerce").to_numpy(dtype=float)
            selected = np.isfinite(numeric_y) & (numeric_y >= lower) & (numeric_y <= upper)
            z_array = np.asarray(z_values)
            if z_array.ndim == 2 and z_array.shape[0] == len(selected):
                trace.y = np.asarray(y_values)[selected]
                trace.z = z_array[selected, :]
                customdata = getattr(trace, "customdata", None)
                if customdata is not None:
                    custom_array = np.asarray(customdata)
                    if custom_array.shape[0] == len(selected):
                        trace.customdata = custom_array[selected, ...]
        for heatmap_row in (1, 3):
            figure.update_yaxes(range=[upper, lower], autorange=False, row=heatmap_row, col=1)
    elif active_tab == "historical_surface_3d":
        selected_surfaces = {}
        for trace in figure.data:
            meta = getattr(trace, "meta", None)
            meta = meta if isinstance(meta, dict) else {}
            trace_role = meta.get("quantapp_role")
            if trace_role == "horizon_boundary_plane":
                try:
                    boundary = float(meta.get("horizon_boundary"))
                except (TypeError, ValueError):
                    trace.visible = False
                else:
                    trace.visible = lower <= boundary <= upper
                continue
            if trace_role in {"horizon_region", "horizon_region_label"}:
                visible_left, visible_right, is_visible = _block18_visible_horizon_region(
                    meta.get("region_start"), meta.get("region_end"), lower, upper
                )
                trace.visible = is_visible
                if is_visible:
                    if trace_role == "horizon_region":
                        trace.x = [
                            [visible_left, visible_right],
                            [visible_left, visible_right],
                        ]
                        region_label = str(getattr(trace, "name", "Horizon Region")).removesuffix(
                            " Horizon Region"
                        )
                        trace.hovertemplate = (
                            f"{region_label}<br>"
                            f"Visible horizons: {visible_left:g}-{visible_right:g} "
                            "trading days<extra></extra>"
                        )
                    else:
                        trace.x = [(visible_left + visible_right) / 2.0]
                continue
            if getattr(trace, "type", None) != "surface":
                continue
            x_values = getattr(trace, "x", None)
            z_values = getattr(trace, "z", None)
            if x_values is None or z_values is None:
                continue
            x_array = np.asarray(x_values)
            z_array = np.asarray(z_values)
            if x_array.ndim != 1 or z_array.ndim != 2 or z_array.shape[1] != len(x_array):
                continue
            numeric_x = pd.to_numeric(pd.Series(x_array), errors="coerce").to_numpy(dtype=float)
            selected = np.isfinite(numeric_x) & (numeric_x >= lower) & (numeric_x <= upper)
            trace.x = x_array[selected]
            trace.z = z_array[:, selected]
            owner = meta.get("owner")
            if owner is not None and selected.any():
                selected_surfaces[str(owner)] = (
                    list(trace.y),
                    numeric_x,
                    np.asarray(z_array, dtype=float),
                    selected,
                )
            surfacecolor = getattr(trace, "surfacecolor", None)
            if surfacecolor is not None:
                color_array = np.asarray(surfacecolor)
                if color_array.ndim == 2 and color_array.shape[1] == len(selected):
                    trace.surfacecolor = color_array[:, selected]
        for trace in figure.data:
            meta = getattr(trace, "meta", None)
            meta = meta if isinstance(meta, dict) else {}
            owner = meta.get("owner")
            trace_role = meta.get("quantapp_role")
            if trace_role not in {
                "segment_historical_sharpe_level_outline",
                "segment_historical_sharpe_level",
            }:
                continue
            selected_surface = selected_surfaces.get(str(owner))
            if selected_surface is None:
                trace.visible = False
                continue
            dates, horizons, z_array, selected = selected_surface
            segment_selected = selected & _block18_segment_horizon_mask(
                horizons,
                meta.get("segment_start"),
                meta.get("segment_end"),
            )
            trace.visible = bool(segment_selected.any())
            if not trace.visible:
                continue
            visible_horizons = horizons[segment_selected]
            trace.x = dates
            trace.y = (
                pd.DataFrame(z_array[:, segment_selected])
                .mean(axis=1, skipna=True)
                .to_numpy(dtype=float)
            )
            if trace_role == "segment_historical_sharpe_level":
                ratio_label = str(meta.get("ratio_label") or "Sharpe")
                trace.hovertemplate = (
                    f"Series: {owner}<br>"
                    f"Segment: {meta.get('segment_label')}<br>"
                    f"Visible horizons: {visible_horizons.min():g}-"
                    f"{visible_horizons.max():g} trading days<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"Segment Mean {ratio_label} Z-Score: %{{y:.2f}}"
                    "<extra></extra>"
                )
        scene_number = 1
        while True:
            scene_name = "scene" if scene_number == 1 else f"scene{scene_number}"
            if scene_name not in figure.layout:
                break
            figure.layout[scene_name].xaxis.update(range=[upper, lower], autorange=False)
            scene_number += 1
    return figure, [lower, upper]

def _block18_spline_is_enabled(toggle_value):
    return _block18_spline_method(toggle_value) != "off"


def _block18_spline_method(toggle_value):
    """Normalize current and legacy smoothing-control values."""
    if isinstance(toggle_value, (list, tuple, set)):
        toggle_value = next(iter(toggle_value), "off")
    aliases = {"cubic_bspline": "b_spline", "b_spline": "b_spline", "p_spline": "p_spline"}
    return aliases.get(str(toggle_value or "off"), "off")


def _block18_spline_method_label(toggle_value):
    return {"b_spline": "Cubic B-spline", "p_spline": "P-spline", "off": "Off"}[
        _block18_spline_method(toggle_value)
    ]

def _block18_raw_overlay_is_enabled(toggle_value):
    if isinstance(toggle_value, str):
        toggle_value = [toggle_value]
    return "raw_overlay" in (toggle_value or [])

def _block18_normalize_spline_strength(strength):
    try:
        return max(0.0, min(float(strength), 100.0))
    except (TypeError, ValueError):
        return float(block18_default_spline_strength)

def _block18_ratio_label_from_trace_name(trace_name, default="Sharpe"):
    trace_name = str(trace_name)
    if trace_name.startswith("Current ") and trace_name.endswith(" Z-Score"):
        return trace_name[len("Current "):-len(" Z-Score")]
    for candidate in MOMENTUM_RATIO_TYPES:
        candidate_label = momentum_ratio_label(candidate)
        if candidate_label in trace_name:
            return candidate_label
    return str(default)

def _block18_is_current_horizon_profile(trace_name):
    trace_name = str(trace_name)
    if trace_name.startswith("Current ") and trace_name.endswith(" Z-Score"):
        return True
    current_ratio_names = {
        f"Current {momentum_ratio_label(ratio_type)} Z-Score"
        for ratio_type in MOMENTUM_RATIO_TYPES
    }
    if trace_name in current_ratio_names | {"Cross-Window Relative Z-Score"}:
        return True
    owner = _block18_benchmark_trace_owner(trace_name)
    return owner is not None and trace_name in {
        *(f"{owner} {name}" for name in current_ratio_names),
        f"{owner} Cross-Window Relative Z-Score",
    }

def _block18_spline_trace_name(raw_trace_name):
    return f"{raw_trace_name} — Smoothed Spline"

def _block18_is_spline_trace(trace_name):
    return str(trace_name).endswith("— Smoothed Spline")

def _block18_horizon_derivative_trace_name(raw_trace_name, order, use_spline):
    derivative_label = "First Derivative" if order == 1 else "Second Derivative"
    profile_label = "Smoothed Spline " if use_spline else ""
    return f"{raw_trace_name} — {profile_label}{derivative_label}"

def _block18_is_horizon_derivative_trace(trace_name):
    trace_name = str(trace_name)
    return trace_name.endswith("First Derivative") or trace_name.endswith("Second Derivative")

def _block18_horizon_extrema_trace_name(raw_trace_name, kind, use_spline):
    profile_label = "Smoothed Spline " if use_spline else "Raw "
    marker_label = "Peak Markers" if kind == "peak" else "Trough Markers"
    return f"{raw_trace_name} — {profile_label}{marker_label}"

def _block18_is_horizon_extrema_trace(trace_name):
    trace_name = str(trace_name)
    return trace_name.endswith("Peak Markers") or trace_name.endswith("Trough Markers")

def _block18_horizon_transition_trace_name(raw_trace_name, feature, use_spline):
    profile_label = "Smoothed Spline " if use_spline else "Raw "
    feature_label = (
        "Fastest Movement Markers"
        if feature == "movement" else "Fastest Rate-Change Markers"
    )
    return f"{raw_trace_name} — {profile_label}{feature_label}"

def _block18_horizon_transition_profile_trace_name(raw_trace_name, feature, use_spline):
    profile_label = "Smoothed Spline " if use_spline else "Raw "
    feature_label = (
        "Fastest Movement Profile Markers"
        if feature == "movement" else "Fastest Rate-Change Profile Markers"
    )
    return f"{raw_trace_name} — {profile_label}{feature_label}"

def _block18_is_horizon_transition_trace(trace_name):
    trace_name = str(trace_name)
    return (
        trace_name.endswith("Fastest Movement Markers")
        or trace_name.endswith("Fastest Rate-Change Markers")
        or trace_name.endswith("Fastest Movement Profile Markers")
        or trace_name.endswith("Fastest Rate-Change Profile Markers")
    )

def _block18_horizon_deceleration_trace_name(raw_trace_name, panel, use_spline):
    profile_label = "Smoothed Spline " if use_spline else "Raw "
    panel_label = "Profile" if panel == "profile" else "Slope"
    return f"{raw_trace_name} — {profile_label}{panel_label} Deceleration Leg"

def _block18_is_horizon_deceleration_trace(trace_name):
    return str(trace_name).endswith("Deceleration Leg")

def _block18_horizon_derivative_axes(trace_name, order):
    cross_window = str(trace_name).endswith("Cross-Window Relative Z-Score")
    if cross_window:
        return ("x9", "y9") if order == 1 else ("x10", "y10")
    return ("x6", "y6") if order == 1 else ("x7", "y7")

def _block18_horizon_profile_color(trace_name):
    owner = _block18_benchmark_trace_owner(trace_name)
    if owner is None:
        return block18_asset_profile_color
    owner_index = benchmark_order.index(owner) if owner in benchmark_order else 0
    return block18_benchmark_profile_palette[
        owner_index % len(block18_benchmark_profile_palette)
    ]

def _block18_horizon_profile_hovertemplate(trace_name, profile_kind):
    trace_name = str(trace_name)
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    metric_label = (
        "Cross-window relative Z-score"
        if trace_name.endswith("Cross-Window Relative Z-Score")
        else f"Current {_block18_ratio_label_from_trace_name(trace_name)} Z-score"
    )
    return (
        f"<b>{identity_label}</b><br>"
        f"Profile: {profile_kind}<br>"
        "Lookback horizon: %{x} trading days<br>"
        + metric_label + ": %{y:.2f}<extra></extra>"
    )

def _block18_horizon_derivative_hovertemplate(trace_name, order, use_spline):
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    derivative_label = "Slope" if order == 1 else "Curvature"
    profile_label = "Selected spline" if use_spline else "Raw finite difference"
    units = "Z-score / 20 horizon days" if order == 1 else "Z-score / (20 horizon days)^2"
    return (
        f"<b>{identity_label}</b><br>"
        f"Derivative source: {profile_label}<br>"
        "Lookback horizon: %{x} trading days<br>"
        + derivative_label + ": %{y:.3f} " + units + "<extra></extra>"
    )

def _block18_horizon_extrema_hovertemplate(trace_name, kind, use_spline):
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    turning_label = "Peak" if kind == "peak" else "Trough"
    curvature_label = "negative" if kind == "peak" else "positive"
    profile_label = "Selected spline" if use_spline else "Raw profile"
    return (
        f"<b>{identity_label}</b><br>"
        f"Turning point: {turning_label}<br>"
        f"Profile: {profile_label}<br>"
        "Lookback horizon: %{x} trading days<br>"
        "Z-score: %{y:.2f}<br>"
        f"Slope crosses zero; curvature is {curvature_label}<extra></extra>"
    )

def _block18_horizon_transition_hovertemplate(trace_name, feature, use_spline):
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    profile_label = "Selected spline" if use_spline else "Raw finite difference"
    if feature == "movement":
        feature_label = "Fastest curve movement between turning points"
        measure_label = "Slope"
        units = "Z-score / 20 horizon days"
    else:
        feature_label = "Fastest change in rate between turning points"
        measure_label = "Curvature"
        units = "Z-score / (20 horizon days)^2"
    return (
        f"<b>{identity_label}</b><br>"
        f"{feature_label}<br>"
        f"Derivative source: {profile_label}<br>"
        "Lookback horizon: %{x} trading days<br>"
        + measure_label + ": %{y:.3f} " + units + "<extra></extra>"
    )

def _block18_horizon_transition_profile_hovertemplate(trace_name, feature, use_spline):
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    feature_label = (
        "Fastest curve movement"
        if feature == "movement" else "Fastest change in rate"
    )
    profile_label = "Selected spline" if use_spline else "Raw profile"
    return (
        f"<b>{identity_label}</b><br>"
        f"Profile marker: {feature_label}<br>"
        f"Profile: {profile_label}<br>"
        "Lookback horizon: %{x} trading days<br>"
        "Profile Z-score: %{y:.2f}<extra></extra>"
    )

def _block18_horizon_deceleration_hovertemplate(trace_name, panel, use_spline):
    owner = _block18_benchmark_trace_owner(trace_name)
    identity_label = (
        f"Benchmark / index: {owner}"
        if owner is not None else f"Asset (stock / ETF): {ticker_str}"
    )
    profile_label = "Selected spline" if use_spline else "Raw finite difference"
    value_label = "Profile Z-score" if panel == "profile" else "Slope"
    units = "" if panel == "profile" else " Z-score / 20 horizon days"
    return (
        f"<b>{identity_label}</b><br>"
        "Deceleration leg: |slope| is decreasing<br>"
        f"Derivative source: {profile_label}<br>"
        "Lookback horizon: %{x} trading days<br>"
        + value_label + ": %{y:.3f}" + units + "<extra></extra>"
    )

def _block18_apply_horizon_profile_colors(figure):
    for trace in figure.data:
        trace_name = str(getattr(trace, "name", ""))
        if not _block18_is_current_horizon_profile(trace_name):
            continue
        if getattr(trace, "line", None) is not None:
            trace.line.color = _block18_horizon_profile_color(trace_name)
        trace.hovertemplate = _block18_horizon_profile_hovertemplate(trace_name, "Raw values")
    return figure

def _block18_cubic_bspline_series(series, strength):
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    x_values = pd.to_numeric(pd.Series(numeric.index), errors="coerce").to_numpy(dtype=float)
    y_values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    smoothed = np.full(len(numeric), np.nan, dtype=float)
    if finite.sum() < 4:
        smoothed[finite] = y_values[finite]
        return pd.Series(smoothed, index=numeric.index)

    x_fit = x_values[finite]
    y_fit = y_values[finite]
    strength = _block18_normalize_spline_strength(strength) / 100.0
    variance = float(np.nanvar(y_fit))
    smoothing_budget = strength * len(y_fit) * max(variance, 1e-8)
    try:
        spline = UnivariateSpline(x_fit, y_fit, k=3, s=smoothing_budget)
        evaluation_mask = finite & (x_values >= x_fit.min()) & (x_values <= x_fit.max())
        smoothed[evaluation_mask] = spline(x_values[evaluation_mask])
    except Exception:
        smoothed[finite] = y_fit
    return pd.Series(smoothed, index=numeric.index)


def _block18_pspline_series(series, strength):
    """Fit a cubic B-spline basis with a second-difference coefficient penalty."""
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    x_values = pd.to_numeric(pd.Series(numeric.index), errors="coerce").to_numpy(dtype=float)
    y_values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    smoothed = np.full(len(numeric), np.nan, dtype=float)
    if finite.sum() < 6:
        smoothed[finite] = y_values[finite]
        return pd.Series(smoothed, index=numeric.index)

    x_fit = x_values[finite]
    y_fit = y_values[finite]
    x_min, x_max = float(x_fit.min()), float(x_fit.max())
    if x_max <= x_min:
        smoothed[finite] = y_fit
        return pd.Series(smoothed, index=numeric.index)
    scaled_x = (x_fit - x_min) / (x_max - x_min)
    degree = 3
    basis_count = min(28, max(8, int(round(len(x_fit) / 8))))
    interior_count = max(0, basis_count - degree - 1)
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    try:
        design = BSpline.design_matrix(scaled_x, knots, degree, extrapolate=False).toarray()
        difference = np.diff(np.eye(design.shape[1]), n=2, axis=0)
        normalized_strength = _block18_normalize_spline_strength(strength) / 100.0
        penalty = 10.0 ** (-4.0 + 8.0 * normalized_strength)
        system = design.T @ design + penalty * (difference.T @ difference)
        coefficients = np.linalg.solve(system, design.T @ y_fit)
        smoothed[finite] = design @ coefficients
    except Exception:
        smoothed[finite] = y_fit
    return pd.Series(smoothed, index=numeric.index)


def _block18_smoothed_spline_series(series, strength, spline_toggle):
    if _block18_spline_method(spline_toggle) == "p_spline":
        return _block18_pspline_series(series, strength)
    return _block18_cubic_bspline_series(series, strength)

def _block18_horizon_derivative_series(
    series, strength, order, use_spline, spline_toggle="b_spline"
):
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    x_values = pd.to_numeric(pd.Series(numeric.index), errors="coerce").to_numpy(dtype=float)
    y_values = numeric.to_numpy(dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    derivative = np.full(len(numeric), np.nan, dtype=float)
    minimum_points = 4 if use_spline else max(3, order + 1)
    if finite.sum() < minimum_points:
        return pd.Series(derivative, index=numeric.index)

    x_fit = x_values[finite]
    y_fit = y_values[finite]
    try:
        if use_spline and _block18_spline_method(spline_toggle) == "p_spline":
            smoothed = _block18_pspline_series(numeric, strength)
            derivative_values = smoothed.loc[numeric.index[finite]].to_numpy(dtype=float)
            for _ in range(order):
                derivative_values = np.gradient(derivative_values, x_fit, edge_order=2)
        elif use_spline:
            normalized_strength = _block18_normalize_spline_strength(strength) / 100.0
            variance = float(np.nanvar(y_fit))
            smoothing_budget = normalized_strength * len(y_fit) * max(variance, 1e-8)
            fitted_spline = UnivariateSpline(x_fit, y_fit, k=3, s=smoothing_budget)
            derivative_values = fitted_spline.derivative(order)(x_fit)
        else:
            derivative_values = y_fit.copy()
            for _ in range(order):
                derivative_values = np.gradient(
                    derivative_values, x_fit, edge_order=2
                )
        derivative[finite] = derivative_values * (block18_horizon_derivative_scale ** order)
    except Exception:
        pass
    return pd.Series(derivative, index=numeric.index)

def _block18_horizon_extrema_series(
    profile_series, first_derivative, second_derivative, kind
):
    profile = pd.to_numeric(pd.Series(profile_series), errors="coerce")
    first = pd.to_numeric(pd.Series(first_derivative), errors="coerce").reindex(profile.index)
    second = pd.to_numeric(pd.Series(second_derivative), errors="coerce").reindex(profile.index)
    markers = pd.Series(np.nan, index=profile.index, dtype=float)
    finite = np.isfinite(profile.to_numpy(dtype=float))
    finite &= np.isfinite(first.to_numpy(dtype=float))
    finite &= np.isfinite(second.to_numpy(dtype=float))
    if finite.sum() < 5:
        return markers

    finite_positions = np.flatnonzero(finite)
    profile_values = profile.iloc[finite_positions].to_numpy(dtype=float)
    first_values = first.iloc[finite_positions].to_numpy(dtype=float)
    second_values = second.iloc[finite_positions].to_numpy(dtype=float)
    robust_range = float(
        np.nanpercentile(profile_values, 95) - np.nanpercentile(profile_values, 5)
    )
    prominence = max(
        block18_extrema_min_prominence,
        block18_extrema_prominence_fraction * max(robust_range, 0.0),
    )
    minimum_separation = max(2, int(round(len(profile_values) * 0.02)))
    direction = 1.0 if kind == "peak" else -1.0
    candidate_positions, _ = find_peaks(
        direction * profile_values,
        prominence=prominence,
        distance=minimum_separation,
    )
    for candidate in candidate_positions:
        if candidate <= 0 or candidate >= len(profile_values) - 1:
            continue
        if kind == "peak":
            derivative_crossing = first_values[candidate - 1] >= 0 >= first_values[candidate + 1]
            curvature_confirms = second_values[candidate] < 0
        else:
            derivative_crossing = first_values[candidate - 1] <= 0 <= first_values[candidate + 1]
            curvature_confirms = second_values[candidate] > 0
        if derivative_crossing and curvature_confirms:
            original_position = finite_positions[candidate]
            markers.iloc[original_position] = profile_values[candidate]
    return markers

def _block18_horizon_transition_marker_series(
    first_derivative, second_derivative, peak_markers, trough_markers, feature
):
    first = pd.to_numeric(pd.Series(first_derivative), errors="coerce")
    second = pd.to_numeric(pd.Series(second_derivative), errors="coerce").reindex(first.index)
    peaks = pd.to_numeric(pd.Series(peak_markers), errors="coerce").reindex(first.index)
    troughs = pd.to_numeric(pd.Series(trough_markers), errors="coerce").reindex(first.index)
    source = first if feature == "movement" else second
    markers = pd.Series(np.nan, index=first.index, dtype=float)
    turning_points = sorted(
        [(int(position), "peak") for position in np.flatnonzero(peaks.notna().to_numpy())]
        + [(int(position), "trough") for position in np.flatnonzero(troughs.notna().to_numpy())]
    )
    source_values = source.to_numpy(dtype=float)
    for (left, left_kind), (right, right_kind) in zip(turning_points, turning_points[1:]):
        if left_kind == right_kind or right - left <= 2:
            continue
        interior = np.arange(left + 1, right, dtype=int)
        finite_interior = interior[np.isfinite(source_values[interior])]
        if finite_interior.size == 0:
            continue
        selected = int(
            finite_interior[np.argmax(np.abs(source_values[finite_interior]))]
        )
        markers.iloc[selected] = source_values[selected]
    return markers

def _block18_project_transition_markers_to_profile(profile_series, transition_markers):
    profile = pd.to_numeric(pd.Series(profile_series), errors="coerce")
    transitions = pd.to_numeric(pd.Series(transition_markers), errors="coerce").reindex(profile.index)
    projected = pd.Series(np.nan, index=profile.index, dtype=float)
    selected = transitions.notna() & profile.notna()
    projected.loc[selected] = profile.loc[selected]
    return projected

def _block18_horizon_deceleration_leg_series(
    source_series, peak_markers, trough_markers, movement_markers
):
    source = pd.to_numeric(pd.Series(source_series), errors="coerce")
    peaks = pd.to_numeric(pd.Series(peak_markers), errors="coerce").reindex(source.index)
    troughs = pd.to_numeric(pd.Series(trough_markers), errors="coerce").reindex(source.index)
    movement = pd.to_numeric(pd.Series(movement_markers), errors="coerce").reindex(source.index)
    deceleration = pd.Series(np.nan, index=source.index, dtype=float)
    turning_points = sorted(
        [(int(position), "peak") for position in np.flatnonzero(peaks.notna().to_numpy())]
        + [(int(position), "trough") for position in np.flatnonzero(troughs.notna().to_numpy())]
    )
    movement_positions = np.flatnonzero(movement.notna().to_numpy())
    for (left, left_kind), (right, right_kind) in zip(turning_points, turning_points[1:]):
        if left_kind == right_kind:
            continue
        interval_movements = movement_positions[
            (movement_positions > left) & (movement_positions < right)
        ]
        if interval_movements.size == 0:
            continue
        maximum_speed_position = int(interval_movements[0])
        deceleration.iloc[maximum_speed_position:right] = (
            source.iloc[maximum_speed_position:right]
        )
    return deceleration

def _block18_add_spline_updates(trace_updates, spline_toggle, spline_strength):
    if not _block18_spline_is_enabled(spline_toggle):
        return trace_updates
    for trace_name, values in list(trace_updates.items()):
        if _block18_is_current_horizon_profile(trace_name):
            trace_updates[_block18_spline_trace_name(trace_name)] = _block18_smoothed_spline_series(
                values, spline_strength, spline_toggle
            )
    return trace_updates

def _block18_add_horizon_derivative_updates(
    trace_updates, spline_toggle, spline_strength
):
    use_spline = _block18_spline_is_enabled(spline_toggle)
    for trace_name, values in list(trace_updates.items()):
        if not _block18_is_current_horizon_profile(trace_name):
            continue
        for order in (1, 2):
            derivative_name = _block18_horizon_derivative_trace_name(
                trace_name, order, use_spline
            )
            trace_updates[derivative_name] = _block18_horizon_derivative_series(
                values, spline_strength, order, use_spline, spline_toggle
            )
    return trace_updates

def _block18_add_horizon_extrema_updates(
    trace_updates, spline_toggle
):
    use_spline = _block18_spline_is_enabled(spline_toggle)
    if not use_spline:
        return trace_updates
    raw_updates = [
        (trace_name, values) for trace_name, values in list(trace_updates.items())
        if _block18_is_current_horizon_profile(trace_name)
    ]
    for trace_name, raw_values in raw_updates:
        profile_name = _block18_spline_trace_name(trace_name) if use_spline else trace_name
        profile_values = trace_updates.get(profile_name, raw_values)
        first_derivative = trace_updates.get(
            _block18_horizon_derivative_trace_name(trace_name, 1, use_spline)
        )
        second_derivative = trace_updates.get(
            _block18_horizon_derivative_trace_name(trace_name, 2, use_spline)
        )
        if first_derivative is None or second_derivative is None:
            continue
        for kind in ("peak", "trough"):
            trace_updates[_block18_horizon_extrema_trace_name(
                trace_name, kind, use_spline
            )] = _block18_horizon_extrema_series(
                profile_values, first_derivative, second_derivative, kind
            )
    return trace_updates

def _block18_add_horizon_transition_updates(trace_updates, spline_toggle):
    use_spline = _block18_spline_is_enabled(spline_toggle)
    if not use_spline:
        return trace_updates
    raw_trace_names = [
        trace_name for trace_name in list(trace_updates)
        if _block18_is_current_horizon_profile(trace_name)
    ]
    for trace_name in raw_trace_names:
        profile_name = _block18_spline_trace_name(trace_name) if use_spline else trace_name
        profile_values = trace_updates.get(profile_name, trace_updates[trace_name])
        first_derivative = trace_updates.get(
            _block18_horizon_derivative_trace_name(trace_name, 1, use_spline)
        )
        second_derivative = trace_updates.get(
            _block18_horizon_derivative_trace_name(trace_name, 2, use_spline)
        )
        peak_markers = trace_updates.get(
            _block18_horizon_extrema_trace_name(trace_name, "peak", use_spline)
        )
        trough_markers = trace_updates.get(
            _block18_horizon_extrema_trace_name(trace_name, "trough", use_spline)
        )
        if any(value is None for value in (
            first_derivative, second_derivative, peak_markers, trough_markers
        )):
            continue
        for feature in ("movement",):
            transition_markers = _block18_horizon_transition_marker_series(
                first_derivative, second_derivative,
                peak_markers, trough_markers, feature,
            )
            trace_updates[_block18_horizon_transition_trace_name(
                trace_name, feature, use_spline
            )] = transition_markers
            trace_updates[_block18_horizon_transition_profile_trace_name(
                trace_name, feature, use_spline
            )] = _block18_project_transition_markers_to_profile(
                profile_values, transition_markers
            )
        movement_markers = trace_updates.get(
            _block18_horizon_transition_trace_name(
                trace_name, "movement", use_spline
            )
        )
        if movement_markers is not None:
            for panel, source_values in (
                ("profile", profile_values), ("slope", first_derivative)
            ):
                trace_updates[_block18_horizon_deceleration_trace_name(
                    trace_name, panel, use_spline
                )] = _block18_horizon_deceleration_leg_series(
                    source_values, peak_markers, trough_markers, movement_markers
                )
    return trace_updates

def _block18_configure_spline_traces(
    figure, spline_toggle, spline_strength, raw_overlay_toggle=None, *, compute_values=True
):
    figure = _block18_apply_horizon_profile_colors(figure)
    figure.data = tuple(
        trace for trace in figure.data
        if not _block18_is_spline_trace(getattr(trace, "name", ""))
        and not _block18_is_horizon_derivative_trace(getattr(trace, "name", ""))
        and not _block18_is_horizon_extrema_trace(getattr(trace, "name", ""))
        and not _block18_is_horizon_transition_trace(getattr(trace, "name", ""))
        and not _block18_is_horizon_deceleration_trace(getattr(trace, "name", ""))
    )
    use_spline = _block18_spline_is_enabled(spline_toggle)
    show_raw_overlay = _block18_raw_overlay_is_enabled(raw_overlay_toggle)
    raw_profile_traces = [
        trace for trace in figure.data
        if _block18_is_current_horizon_profile(getattr(trace, "name", ""))
    ]
    for raw_trace in raw_profile_traces:
        raw_series = pd.Series(
            list(raw_trace.y), index=pd.to_numeric(pd.Series(list(raw_trace.x)), errors="coerce")
        )
        if not use_spline:
            raw_trace.mode = "lines"
            raw_trace.visible = True
            raw_trace.opacity = 1.0
            raw_trace.showlegend = True
            raw_trace.line.width = max(float(raw_trace.line.width or 0.0), 2.2)
            raw_trace.line.dash = "solid"
        spline_line = raw_trace.line.to_plotly_json() if getattr(raw_trace, "line", None) is not None else {}
        profile_color = spline_line.get("color") or _block18_horizon_profile_color(raw_trace.name)
        active_profile = raw_series
        if use_spline:
            smoothed = (
                _block18_smoothed_spline_series(raw_series, spline_strength, spline_toggle)
                if compute_values else pd.Series(np.nan, index=raw_series.index)
            )
            active_profile = smoothed
            spline_line.update({
                "color": profile_color,
                "width": max(float(spline_line.get("width") or 2.0) + 1.5, 3.5),
                "dash": "solid",
            })
            if show_raw_overlay:
                raw_trace.visible = True
                raw_trace.line.color = profile_color
                raw_trace.line.width = 1.25
                raw_trace.line.dash = "solid"
                raw_trace.opacity = 0.40
                raw_trace.showlegend = False
            else:
                raw_trace.visible = False
                raw_trace.showlegend = False
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(smoothed),
                mode="lines",
                name=_block18_spline_trace_name(raw_trace.name),
                line=spline_line,
                opacity=1.0,
                showlegend=True,
                xaxis=raw_trace.xaxis,
                yaxis=raw_trace.yaxis,
                hovertemplate=_block18_horizon_profile_hovertemplate(
                    raw_trace.name, _block18_spline_method_label(spline_toggle)
                ),
            ))
        derivatives = {}
        for order in (1, 2):
            derivative = (
                _block18_horizon_derivative_series(
                    raw_series, spline_strength, order, use_spline, spline_toggle
                )
                if compute_values else pd.Series(np.nan, index=raw_series.index)
            )
            derivatives[order] = derivative
            derivative_xaxis, derivative_yaxis = _block18_horizon_derivative_axes(
                raw_trace.name, order
            )
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(derivative),
                mode="lines",
                name=_block18_horizon_derivative_trace_name(
                    raw_trace.name, order, use_spline
                ),
                line={"color": profile_color, "width": 2.4, "dash": "solid"},
                opacity=1.0,
                showlegend=False,
                xaxis=derivative_xaxis,
                yaxis=derivative_yaxis,
                hovertemplate=_block18_horizon_derivative_hovertemplate(
                    raw_trace.name, order, use_spline
                ),
            ))
        if not use_spline:
            continue
        extrema_by_kind = {}
        for kind in ("peak", "trough"):
            extrema = _block18_horizon_extrema_series(
                active_profile, derivatives[1], derivatives[2], kind
            )
            extrema_by_kind[kind] = extrema
            marker_symbol = "circle"
            marker_color = "#22c55e" if kind == "peak" else "#ef4444"
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(extrema),
                mode="markers",
                name=_block18_horizon_extrema_trace_name(
                    raw_trace.name, kind, use_spline
                ),
                marker={
                    "color": marker_color,
                    "size": 18,
                    "symbol": marker_symbol,
                    "line": {"color": "#0b0f14", "width": 3.5},
                },
                opacity=1.0,
                showlegend=False,
                cliponaxis=False,
                xaxis=raw_trace.xaxis,
                yaxis=raw_trace.yaxis,
                hovertemplate=_block18_horizon_extrema_hovertemplate(
                    raw_trace.name, kind, use_spline
                ),
            ))
        movement_markers = _block18_horizon_transition_marker_series(
            derivatives[1], derivatives[2],
            extrema_by_kind["peak"], extrema_by_kind["trough"], "movement",
        )
        first_derivative_xaxis, first_derivative_yaxis = (
            _block18_horizon_derivative_axes(raw_trace.name, 1)
        )
        for panel, source_values, panel_xaxis, panel_yaxis in (
            ("profile", active_profile, raw_trace.xaxis, raw_trace.yaxis),
            ("slope", derivatives[1], first_derivative_xaxis, first_derivative_yaxis),
        ):
            deceleration_leg = _block18_horizon_deceleration_leg_series(
                source_values, extrema_by_kind["peak"],
                extrema_by_kind["trough"], movement_markers,
            )
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(deceleration_leg),
                mode="lines",
                name=_block18_horizon_deceleration_trace_name(
                    raw_trace.name, panel, use_spline
                ),
                line={"color": profile_color, "width": 6.0, "dash": "dot"},
                opacity=0.90,
                showlegend=False,
                connectgaps=False,
                xaxis=panel_xaxis,
                yaxis=panel_yaxis,
                hovertemplate=_block18_horizon_deceleration_hovertemplate(
                    raw_trace.name, panel, use_spline
                ),
            ))
        transition_specs = {
            "movement": {"order": 1, "symbol": "diamond", "size": 16},
        }
        for feature, marker_spec in transition_specs.items():
            transition_markers = movement_markers
            derivative_xaxis, derivative_yaxis = _block18_horizon_derivative_axes(
                raw_trace.name, marker_spec["order"]
            )
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(transition_markers),
                mode="markers",
                name=_block18_horizon_transition_trace_name(
                    raw_trace.name, feature, use_spline
                ),
                marker={
                    "color": profile_color,
                    "size": marker_spec["size"],
                    "symbol": marker_spec["symbol"],
                    "line": {"color": "#0b0f14", "width": 3.5},
                },
                opacity=1.0,
                showlegend=False,
                cliponaxis=False,
                xaxis=derivative_xaxis,
                yaxis=derivative_yaxis,
                hovertemplate=_block18_horizon_transition_hovertemplate(
                    raw_trace.name, feature, use_spline
                ),
            ))
            profile_transition_markers = _block18_project_transition_markers_to_profile(
                active_profile, transition_markers
            )
            figure.add_trace(go.Scatter(
                x=list(raw_trace.x),
                y=_block18_json_values(profile_transition_markers),
                mode="markers",
                name=_block18_horizon_transition_profile_trace_name(
                    raw_trace.name, feature, use_spline
                ),
                marker={
                    "color": profile_color,
                    "size": marker_spec["size"],
                    "symbol": marker_spec["symbol"],
                    "line": {"color": "#0b0f14", "width": 3.5},
                },
                opacity=1.0,
                showlegend=False,
                cliponaxis=False,
                xaxis=raw_trace.xaxis,
                yaxis=raw_trace.yaxis,
                hovertemplate=_block18_horizon_transition_profile_hovertemplate(
                    raw_trace.name, feature, use_spline
                ),
            ))
    return figure


block18_ratio_diagnostics_figure_cache = {}


def _block18_ratio_diagnostics_source_figure(ratio_type):
    """Build the full five-row diagnostics source for one selected ratio."""
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    _ensure_momentum_diagnostics_ratio(ratio_type)
    if ratio_type in block18_ratio_diagnostics_figure_cache:
        return go.Figure(block18_ratio_diagnostics_figure_cache[ratio_type])

    ratio_label = _block18_ratio_label(ratio_type)
    diagnostics_contexts = momentum_diagnostics_contexts_by_ratio.get(
        ratio_type, {}
    )
    display_contexts = momentum_diagnostics_display_contexts_by_ratio.get(
        ratio_type, {}
    )
    asset_context = diagnostics_contexts.get(ticker_str)
    asset_display_context = display_contexts.get(ticker_str)
    if asset_context is None or asset_display_context is None:
        raise ValueError(f"No {ratio_label} diagnostics are available for {ticker_str}.")

    figure = plot_momentum_window_diagnostics_grid_view(
        diagnostics_context=asset_context,
        ticker_label=ticker_str,
        ratio_label=ratio_label,
    )
    for benchmark_index, (symbol, display_context) in enumerate(
        display_contexts.items()
    ):
        if symbol == ticker_str:
            continue
        color = block11_benchmark_colors[
            benchmark_index % len(block11_benchmark_colors)
        ]
        dash = block11_benchmark_dashes[
            benchmark_index % len(block11_benchmark_dashes)
        ]
        benchmark_current_zscore = display_context[
            "current_ratio_zscore"
        ].dropna()
        if benchmark_current_zscore.empty:
            continue
        base_name = f"{symbol} Current {ratio_label} Z-Score"
        figure.add_trace(
            go.Scatter(
                x=benchmark_current_zscore.index,
                y=benchmark_current_zscore.values,
                mode="lines+markers",
                name=base_name,
                line=dict(color=color, width=2.2, dash=dash),
                marker=dict(size=4),
                hovertemplate=(
                    f"Benchmark: {symbol}<br>"
                    "Window: %{x} day(s)<br>"
                    f"Current {ratio_label} Z-Score: %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        block11_add_horizon_derivative_traces(
            figure,
            benchmark_current_zscore,
            symbol=symbol,
            base_name=base_name,
            color=color,
            dash=dash,
            rows=(4, 5),
        )

    row3_range_series = [
        asset_display_context["current_ratio_zscore"],
        asset_display_context["ratio_zscore_mean_by_window"],
        *block11_reference_band_series(
            asset_display_context["ratio_zscore_mean_by_window"],
            asset_display_context["ratio_zscore_std_by_window"],
        ),
        *[
            context["current_ratio_zscore"]
            for symbol, context in display_contexts.items()
            if symbol != ticker_str
        ],
        pd.Series([-2.0, 2.0]),
    ]
    figure.update_yaxes(
        range=block11_axis_range(row3_range_series), row=3, col=1
    )
    block11_add_option_dte_vlines(figure, block11_option_chain_dtes)
    block18_ratio_diagnostics_figure_cache[ratio_type] = go.Figure(figure)
    return go.Figure(figure)


def _block18_functional_segment_level_hover(
    owner, segment_label, visible_min, visible_max, segment_level,
    ratio_label="Sharpe",
):
    level_text = (
        f"{float(segment_level):.2f}"
        if segment_level is not None and np.isfinite(segment_level)
        else "Unavailable"
    )
    return (
        f"Series: {owner}<br>"
        f"Segment: {segment_label}<br>"
        "Profile source: Raw values<br>"
        f"Visible horizons: {float(visible_min):g}-{float(visible_max):g} "
        "trading days<br>"
        f"Segment Mean {ratio_label} Z-Score: {level_text}<extra></extra>"
    )


def _block18_add_functional_segment_levels(figure, horizon_range):
    """Add one raw-profile mean level per owner inside each visible segment."""
    figure.data = tuple(
        trace
        for trace in figure.data
        if not (
            isinstance(getattr(trace, "meta", None), dict)
            and str(trace.meta.get("quantapp_role", "")).startswith(
                "functional_segment_level"
            )
        )
    )
    lower, upper = _block18_normalize_diagnostics_window_range(horizon_range)
    source_traces = [
        trace
        for trace in figure.data
        if (getattr(trace, "xaxis", None) or "x") == "x5"
        and _block18_is_current_horizon_profile(getattr(trace, "name", ""))
    ]
    if not source_traces:
        return figure

    level_specs = []
    for source_trace in source_traces:
        source_x = getattr(source_trace, "x", None)
        source_y = getattr(source_trace, "y", None)
        if source_x is None or source_y is None:
            continue
        x_values = pd.to_numeric(
            pd.Series(list(source_x)), errors="coerce"
        ).to_numpy(dtype=float)
        y_values = pd.to_numeric(
            pd.Series(list(source_y)), errors="coerce"
        ).to_numpy(dtype=float)
        if len(x_values) == 0 or len(x_values) != len(y_values):
            continue
        in_range = np.isfinite(x_values)
        in_range &= (x_values >= lower) & (x_values <= upper)
        raw_trace_name = str(source_trace.name)
        ratio_label = _block18_ratio_label_from_trace_name(raw_trace_name)
        owner = _block18_benchmark_trace_owner(raw_trace_name) or ticker_str
        raw_line = (
            source_trace.line.to_plotly_json()
            if getattr(source_trace, "line", None) is not None
            else {}
        )
        line_color = raw_line.get("color") or _block18_horizon_profile_color(
            raw_trace_name
        )
        for segment_index, (
            segment_label, segment_start, segment_end, _segment_color
        ) in enumerate(HORIZON_SEGMENT_REGIONS):
            segment_horizons = in_range & _block18_segment_horizon_mask(
                x_values, segment_start, segment_end
            )
            if not segment_horizons.any():
                continue
            visible_left = max(
                float(lower),
                float(segment_start) if segment_start is not None else float(lower),
            )
            visible_right = min(
                float(upper),
                float(segment_end) if segment_end is not None else float(upper),
            )
            if visible_right < visible_left:
                continue
            selected_values = segment_horizons & np.isfinite(y_values)
            segment_level = (
                float(np.mean(y_values[selected_values]))
                if selected_values.any()
                else None
            )
            visible_horizons = x_values[segment_horizons]
            level_specs.append(dict(
                owner=owner,
                raw_trace_name=raw_trace_name,
                ratio_label=ratio_label,
                segment_index=segment_index,
                segment_label=segment_label,
                segment_start=segment_start,
                segment_end=segment_end,
                visible_min=float(np.min(visible_horizons)),
                visible_max=float(np.max(visible_horizons)),
                left=visible_left,
                right=visible_right,
                level=segment_level,
                color=line_color,
                dash=HORIZON_SEGMENT_LINE_STYLES[segment_index][1],
                xaxis=source_trace.xaxis,
                yaxis=source_trace.yaxis,
            ))

    for level in level_specs:
        figure.add_trace(go.Scatter(
            x=[level["left"], level["right"]],
            y=[level["level"], level["level"]],
            mode="lines",
            name=f"{level['raw_trace_name']} — {level['segment_label']} Raw Segment Level",
            meta={
                "quantapp_role": "functional_segment_level",
                "owner": level["owner"],
                "segment_index": level["segment_index"],
                "segment_label": level["segment_label"],
                "segment_start": level["segment_start"],
                "segment_end": level["segment_end"],
                "source_trace_name": level["raw_trace_name"],
                "ratio_label": level["ratio_label"],
                "visible_min": level["visible_min"],
                "visible_max": level["visible_max"],
            },
            line=dict(color=level["color"], width=4.0, dash=level["dash"]),
            zorder=-1,
            opacity=1.0,
            hovertemplate=_block18_functional_segment_level_hover(
                level["owner"],
                level["segment_label"],
                level["visible_min"],
                level["visible_max"],
                level["level"],
                level["ratio_label"],
            ),
            showlegend=False,
            xaxis=level["xaxis"],
            yaxis=level["yaxis"],
        ))
    return figure


def _block18_prepare_window_diagnostics_playback(ratio_type="sharpe"):
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    _ensure_momentum_diagnostics_ratio(ratio_type)
    if ratio_type in block18_window_diagnostics_playback_cache:
        return block18_window_diagnostics_playback_cache[ratio_type]

    ratio_contexts = momentum_diagnostics_contexts_by_ratio.get(ratio_type, {})
    ratio_playback_contexts = {}
    for symbol, diagnostics_context in ratio_contexts.items():
        ratio_frame = pd.DataFrame(diagnostics_context["ratio_table"]).copy()
        ratio_frame = ratio_frame.apply(pd.to_numeric, errors="coerce").sort_index()
        ratio_frame.index = pd.to_datetime(
            ratio_frame.index, errors="coerce", utc=True
        ).tz_convert(None).normalize()
        ratio_frame = ratio_frame.loc[~ratio_frame.index.isna()]
        ratio_frame = ratio_frame.loc[
            ~ratio_frame.index.duplicated(keep="last")
        ]

        expanding_mean = ratio_frame.expanding(min_periods=2).mean()
        expanding_std = ratio_frame.expanding(min_periods=2).std().replace(
            0.0, np.nan
        )
        ratio_zscore = ratio_frame.sub(expanding_mean).div(expanding_std).astype(
            "float32"
        )
        playback_context = {
            "ratio_type": ratio_type,
            "ratio_label": _block18_ratio_label(ratio_type),
            "ratio_zscore": ratio_zscore,
        }
        if symbol == ticker_str:
            playback_context.update({
                "ratio_reference_mean": ratio_zscore.expanding(
                    min_periods=2
                ).mean().astype("float32"),
                "ratio_reference_std": ratio_zscore.expanding(
                    min_periods=2
                ).std().astype("float32"),
            })
        ratio_playback_contexts[symbol] = playback_context

    block18_window_diagnostics_playback_cache[ratio_type] = (
        ratio_playback_contexts
    )
    return ratio_playback_contexts


def _block18_surface_sample(frame, maximum_dates=360, recent_dates=126):
    """Keep 3D payloads responsive while preserving detail in the default 6M view."""
    surface_frame = pd.DataFrame(frame).apply(pd.to_numeric, errors="coerce").sort_index()
    surface_frame = surface_frame.loc[~surface_frame.index.duplicated(keep="last")]
    surface_frame = surface_frame.dropna(how="all").dropna(axis=1, how="all")
    if len(surface_frame) > maximum_dates:
        recent_count = min(int(recent_dates), maximum_dates, len(surface_frame))
        older_count = maximum_dates - recent_count
        recent_positions = np.arange(len(surface_frame) - recent_count, len(surface_frame))
        older_positions = (
            np.linspace(0, len(surface_frame) - recent_count - 1, older_count, dtype=int)
            if older_count > 0 else np.array([], dtype=int)
        )
        surface_frame = surface_frame.iloc[np.unique(np.concatenate([older_positions, recent_positions]))]
    return surface_frame


def _block18_historical_ratio_surface_3d(
    selected_benchmarks, ratio_type="sharpe"
):
    """Plot the selected historical risk-adjusted-ratio profile in 3D."""
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    _ensure_momentum_diagnostics_ratio(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    playback_contexts = _block18_prepare_window_diagnostics_playback(
        ratio_type
    )
    owners = [ticker_str] + [
        symbol for symbol in _block18_normalize_benchmark_selection(selected_benchmarks)
        if symbol in playback_contexts and symbol != ticker_str
    ]
    owners = [owner for owner in owners if owner in playback_contexts]
    if not owners:
        return _block18_error_figure(
            f"No historical {ratio_label} profile data are available.",
            height=850,
        )

    subplot_titles = []
    for owner in owners:
        subplot_titles.extend([
            f"{owner} — Historical {ratio_label} Z-Score Profile",
            f"{owner} — Historical {ratio_label} Level by Horizon Segment",
        ])
    figure = make_subplots(
        rows=len(owners) * 2,
        cols=1,
        specs=[
            spec
            for _ in owners
            for spec in ([{"type": "surface"}], [{"type": "xy"}])
        ],
        subplot_titles=subplot_titles,
        row_heights=[height for _ in owners for height in (0.72, 0.28)],
        vertical_spacing=min(0.055, 0.12 / max(1, len(owners) * 2 - 1)),
    )

    surface_date_bounds = []
    legend_segment_labels = set()
    for owner_number, owner in enumerate(owners, start=1):
        surface_row = owner_number * 2 - 1
        level_row = owner_number * 2
        context = playback_contexts[owner]
        surface_frame = _block18_surface_sample(context["ratio_zscore"])
        if surface_frame.empty:
            continue
        surface_date_bounds.append((surface_frame.index.min(), surface_frame.index.max()))
        figure.add_trace(
            go.Surface(
                x=pd.to_numeric(pd.Index(surface_frame.columns), errors="coerce"),
                y=list(pd.to_datetime(surface_frame.index)),
                z=surface_frame.to_numpy(dtype=float),
                name=f"{owner} {ratio_label} Z-Score",
                meta={
                    "quantapp_role": "historical_sharpe_surface",
                    "owner": owner,
                    "ratio_type": ratio_type,
                    "ratio_label": ratio_label,
                },
                colorscale="RdBu",
                reversescale=False,
                cmin=-4.0,
                cmax=4.0,
                cmid=0,
                showscale=owner_number == 1,
                colorbar=dict(
                    title=f"{ratio_label} Z-Score",
                    len=min(0.42, 0.78 / len(owners)),
                    x=1.01,
                    y=1.0 - ((owner_number - 0.5) / len(owners)),
                    thickness=12,
                ),
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "project_z": True,
                        "width": 1,
                    }
                },
                hovertemplate=(
                    f"Series: {owner}<br>"
                    "Date: %{y|%Y-%m-%d}<br>"
                    "Lookback: %{x:.0f} days<br>"
                    f"{ratio_label} Z-Score: %{{z:.2f}}<extra></extra>"
                ),
            ),
            row=surface_row,
            col=1,
        )
        surface_start = pd.Timestamp(surface_frame.index.min())
        surface_end = pd.Timestamp(surface_frame.index.max())
        surface_midpoint = surface_start + (surface_end - surface_start) / 2
        available_horizons = pd.to_numeric(
            pd.Index(surface_frame.columns), errors="coerce"
        ).to_numpy(dtype=float)
        finite_horizons = available_horizons[np.isfinite(available_horizons)]
        if finite_horizons.size:
            minimum_horizon = float(np.min(finite_horizons))
            maximum_horizon = float(np.max(finite_horizons))
        else:
            minimum_horizon = maximum_horizon = 0.0
        owner_segment_levels = []
        for segment_index, (
            region_label, region_start, region_end, region_color
        ) in enumerate(HORIZON_SEGMENT_REGIONS):
            left = max(
                minimum_horizon,
                float(region_start) if region_start is not None else minimum_horizon,
            )
            right = min(
                maximum_horizon,
                float(region_end) if region_end is not None else maximum_horizon,
            )
            if right <= left:
                continue
            figure.add_trace(
                go.Surface(
                    x=[[left, right], [left, right]],
                    y=[
                        [surface_start, surface_start],
                        [surface_end, surface_end],
                    ],
                    z=[[-4.23, -4.23], [-4.23, -4.23]],
                    surfacecolor=[[0, 0], [0, 0]],
                    colorscale=[[0, region_color], [1, region_color]],
                    cmin=0,
                    cmax=1,
                    opacity=0.26,
                    showscale=False,
                    showlegend=False,
                    name=f"{region_label} Horizon Region",
                    meta={
                        "quantapp_role": "horizon_region",
                        "region_start": left,
                        "region_end": right,
                        "owner": owner,
                    },
                    hovertemplate=(
                        f"{region_label}<br>"
                        f"Horizons: {left:g}-{right:g} trading days<extra></extra>"
                    ),
                ),
                row=surface_row,
                col=1,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[(left + right) / 2.0],
                    y=[surface_midpoint],
                    z=[-4.10],
                    mode="text",
                    text=[f"<b>{region_label}</b>"],
                    meta={
                        "quantapp_role": "horizon_region_label",
                        "region_start": left,
                        "region_end": right,
                        "owner": owner,
                    },
                    textfont=dict(color="#f8fafc", size=14),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=surface_row,
                col=1,
            )
            segment_mask = _block18_segment_horizon_mask(
                available_horizons, region_start, region_end
            )
            if segment_mask.any():
                segment_level = surface_frame.iloc[:, segment_mask].mean(
                    axis=1, skipna=True
                )
                line_color, line_dash = HORIZON_SEGMENT_LINE_STYLES[segment_index]
                owner_segment_levels.append(dict(
                    segment_index=segment_index,
                    label=region_label,
                    start=region_start,
                    end=region_end,
                    level=segment_level,
                    color=line_color,
                    dash=line_dash,
                ))

        for boundary_index in range(len(HORIZON_SEGMENT_REGIONS) - 1):
            left_label, _left_start, boundary, _left_color = (
                HORIZON_SEGMENT_REGIONS[boundary_index]
            )
            right_label = HORIZON_SEGMENT_REGIONS[boundary_index + 1][0]
            if boundary is None:
                continue
            boundary = float(boundary)
            if not (minimum_horizon < boundary < maximum_horizon):
                continue
            figure.add_trace(
                go.Surface(
                    x=[[boundary, boundary], [boundary, boundary]],
                    y=[
                        [surface_start, surface_end],
                        [surface_start, surface_end],
                    ],
                    z=[[-4.23, -4.23], [4.25, 4.25]],
                    surfacecolor=[[0, 0], [0, 0]],
                    colorscale=[[0, "#cbd5e1"], [1, "#cbd5e1"]],
                    cmin=0,
                    cmax=1,
                    opacity=0.18,
                    lighting=dict(
                        ambient=1.0,
                        diffuse=0.0,
                        specular=0.0,
                        roughness=1.0,
                        fresnel=0.0,
                    ),
                    showscale=False,
                    showlegend=False,
                    name=f"{boundary:g}-Day Horizon Boundary",
                    meta={
                        "quantapp_role": "horizon_boundary_plane",
                        "horizon_boundary": boundary,
                        "owner": owner,
                        "left_segment": left_label,
                        "right_segment": right_label,
                    },
                    hovertemplate=(
                        f"{left_label} / {right_label}<br>"
                        f"Boundary: {boundary:g} trading days<extra></extra>"
                    ),
                ),
                row=surface_row,
                col=1,
            )

        for segment in owner_segment_levels:
            common_meta = {
                "owner": owner,
                "ratio_type": ratio_type,
                "ratio_label": ratio_label,
                "segment_index": segment["segment_index"],
                "segment_label": segment["label"],
                "segment_start": segment["start"],
                "segment_end": segment["end"],
            }
            figure.add_trace(
                go.Scatter(
                    x=segment["level"].index,
                    y=segment["level"].values,
                    mode="lines",
                    name=f"{segment['label']} outline",
                    legendgroup=f"horizon-segment-{segment['segment_index']}",
                    meta={
                        **common_meta,
                        "quantapp_role": "segment_historical_sharpe_level_outline",
                    },
                    line=dict(color="#020617", width=6.0, dash=segment["dash"]),
                    opacity=0.95,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=level_row,
                col=1,
            )
        for segment in owner_segment_levels:
            common_meta = {
                "owner": owner,
                "ratio_type": ratio_type,
                "ratio_label": ratio_label,
                "segment_index": segment["segment_index"],
                "segment_label": segment["label"],
                "segment_start": segment["start"],
                "segment_end": segment["end"],
            }
            show_segment_legend = segment["label"] not in legend_segment_labels
            if show_segment_legend:
                legend_segment_labels.add(segment["label"])
            figure.add_trace(
                go.Scatter(
                    x=segment["level"].index,
                    y=segment["level"].values,
                    mode="lines",
                    name=segment["label"],
                    legendgroup=f"horizon-segment-{segment['segment_index']}",
                    legendrank=segment["segment_index"],
                    meta={
                        **common_meta,
                        "quantapp_role": "segment_historical_sharpe_level",
                    },
                    line=dict(
                        color=segment["color"], width=3.2, dash=segment["dash"]
                    ),
                    opacity=1.0,
                    hovertemplate=(
                        f"Series: {owner}<br>"
                        f"Segment: {segment['label']}<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        f"Segment Mean {ratio_label} Z-Score: %{{y:.2f}}"
                        "<extra></extra>"
                    ),
                    showlegend=show_segment_legend,
                ),
                row=level_row,
                col=1,
            )
        level_xref = "x domain" if owner_number == 1 else f"x{owner_number} domain"
        level_yref = "y" if owner_number == 1 else f"y{owner_number}"
        for zone_start, zone_end, zone_color, zone_name in (
            (-1, 1, MEAN_PANEL_ZONE_FILLS["neutral"], "Neutral"),
            (-2, -1, MEAN_PANEL_ZONE_FILLS["green"], "Low"),
            (1, 2, MEAN_PANEL_ZONE_FILLS["red"], "High"),
        ):
            figure.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=zone_start,
                y1=zone_end,
                xref=level_xref,
                yref=level_yref,
                fillcolor=zone_color,
                line_width=0,
                layer="below",
                name=f"{owner} {zone_name} {ratio_label} Zone",
                showlegend=False,
            )
        figure.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=0,
            y1=0,
            xref=level_xref,
            yref=level_yref,
            line=dict(color="rgba(226, 232, 240, 0.60)", width=1.3),
            layer="below",
        )
        for reference_level in MEAN_PANEL_REFERENCE_LEVELS:
            figure.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=reference_level,
                y1=reference_level,
                xref=level_xref,
                yref=level_yref,
                line=dict(
                    color=MEAN_PANEL_REFERENCE_LINE_COLOR,
                    width=1,
                    dash="dot",
                ),
                layer="below",
            )
        figure.update_yaxes(
            range=block11_axis_range(
                [
                    *(segment["level"] for segment in owner_segment_levels),
                    pd.Series([-2.0, 2.0]),
                ],
                padding=0.08,
                min_span=4.0,
            ),
            autorange=False,
            row=level_row,
            col=1,
        )

    if not surface_date_bounds:
        return _block18_error_figure(
            f"No finite historical {ratio_label} profiles are available.",
            height=850,
        )
    global_start = pd.Timestamp(min(start for start, _ in surface_date_bounds))
    global_end = pd.Timestamp(max(end for _, end in surface_date_bounds))
    default_start = max(global_start, global_end - pd.DateOffset(months=6))
    default_label_date = default_start + (global_end - default_start) * 0.75
    horizon_region_label_indices = [
        trace_index
        for trace_index, trace in enumerate(figure.data)
        if isinstance(getattr(trace, "meta", None), dict)
        and trace.meta.get("quantapp_role") == "horizon_region_label"
    ]
    for trace_index in horizon_region_label_indices:
        figure.data[trace_index].y = [default_label_date]

    scene_count = len(owners)
    for scene_number in range(1, scene_count + 1):
        scene_name = "scene" if scene_number == 1 else f"scene{scene_number}"
        figure.layout[scene_name].update(
            xaxis=dict(title="Lookback Window (Days)", autorange=True, gridcolor="#334155"),
            yaxis=dict(
                title="Date", type="date", gridcolor="#334155",
                range=[default_start, global_end],
            ),
            zaxis=dict(title="Z-Score", range=[-4.25, 4.25], gridcolor="#334155"),
            bgcolor="#0b0f14",
            camera=dict(eye=dict(x=1.45, y=-1.55, z=0.85)),
            aspectmode="manual",
            aspectratio=dict(x=1.15, y=1.65, z=0.75),
        )
    for owner_number in range(1, len(owners) + 1):
        level_row = owner_number * 2
        figure.update_xaxes(
            title_text="Date",
            type="date",
            range=[default_start, global_end],
            gridcolor="#334155",
            row=level_row,
            col=1,
        )
        figure.update_yaxes(
            title_text=f"Segment Mean {ratio_label} Z-Score",
            zeroline=False,
            gridcolor="#334155",
            row=level_row,
            col=1,
        )

    range_options = [
        ("1M", pd.DateOffset(months=1)),
        ("3M", pd.DateOffset(months=3)),
        ("6M", pd.DateOffset(months=6)),
        ("1Y", pd.DateOffset(years=1)),
        ("3Y", pd.DateOffset(years=3)),
        ("All", None),
    ]
    range_buttons = []
    for label, offset in range_options:
        range_start = global_start if offset is None else max(global_start, global_end - offset)
        range_label_date = range_start + (global_end - range_start) * 0.75
        relayout = {}
        for scene_number in range(1, scene_count + 1):
            scene_name = "scene" if scene_number == 1 else f"scene{scene_number}"
            relayout[f"{scene_name}.yaxis.range"] = [range_start, global_end]
            xaxis_name = "xaxis" if scene_number == 1 else f"xaxis{scene_number}"
            relayout[f"{xaxis_name}.range"] = [range_start, global_end]
        range_buttons.append(dict(
            label=label,
            method="update",
            args=[
                {"y": [[range_label_date] for _ in horizon_region_label_indices]},
                relayout,
                horizon_region_label_indices,
            ],
        ))

    figure.update_layout(
        title=(
            f"{ticker_str} Historical {ratio_label} Surface v.2 — "
            "Functional Horizon Profiles Through Time"
        ),
        template="plotly_dark",
        height=max(980, 790 * len(owners)),
        margin=dict(l=35, r=95, t=145, b=105),
        showlegend=True,
        legend=dict(
            title_text="Horizon segment",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.025,
            yanchor="top",
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(color="#f8fafc", size=12),
            groupclick="togglegroup",
        ),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            buttons=range_buttons,
            active=2,
            x=0.5,
            y=1.075,
            xanchor="center",
            yanchor="bottom",
            bgcolor="#111827",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
            pad=dict(l=4, r=4, t=3, b=3),
        )],
        annotations=list(figure.layout.annotations or ()) + [dict(
            text="History range",
            x=0.5,
            y=1.115,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#94a3b8", size=11),
        )],
    )
    return _block18_apply_typography(figure)

def _block18_diagnostics_asof_row(frame, as_of_date):
    if frame.empty:
        return pd.Series(dtype=float)
    row_position = frame.index.searchsorted(pd.Timestamp(as_of_date), side="right") - 1
    if row_position < 0:
        return pd.Series(index=frame.columns, dtype=float)
    return pd.to_numeric(frame.iloc[row_position], errors="coerce")

def _block18_window_diagnostics_trace_updates(
    playback_position, selected_benchmarks, spline_toggle=None,
    spline_strength=None, ratio_type="sharpe",
):
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    playback_contexts = _block18_prepare_window_diagnostics_playback(
        ratio_type
    )
    as_of_date, playback_position = _block18_playback_date(playback_position)
    asset_context = playback_contexts[ticker_str]

    current_ratio = _block18_diagnostics_asof_row(
        asset_context["ratio_zscore"], as_of_date
    )
    ratio_mean = _block18_diagnostics_asof_row(
        asset_context["ratio_reference_mean"], as_of_date
    )
    ratio_std = _block18_diagnostics_asof_row(
        asset_context["ratio_reference_std"], as_of_date
    )
    trace_updates = {
        f"Current {ratio_label} Z-Score": current_ratio,
        f"Historical Mean {ratio_label} Z-Score": ratio_mean,
    }
    for level in (1, 2):
        trace_updates[
            f"Historical {ratio_label} Z-Score +{level} Std Dev"
        ] = ratio_mean + level * ratio_std
        trace_updates[
            f"Historical {ratio_label} Z-Score -{level} Std Dev"
        ] = ratio_mean - level * ratio_std

    for symbol in selected_benchmarks:
        benchmark_context = playback_contexts.get(symbol)
        if benchmark_context is None:
            continue
        trace_updates[
            f"{symbol} Current {ratio_label} Z-Score"
        ] = _block18_diagnostics_asof_row(
            benchmark_context["ratio_zscore"], as_of_date
        )
    trace_updates = _block18_add_spline_updates(
        trace_updates, spline_toggle, spline_strength
    )
    trace_updates = _block18_add_horizon_derivative_updates(
        trace_updates, spline_toggle, spline_strength
    )
    trace_updates = _block18_add_horizon_extrema_updates(
        trace_updates, spline_toggle
    )
    trace_updates = _block18_add_horizon_transition_updates(
        trace_updates, spline_toggle
    )
    return trace_updates, as_of_date, playback_position

def _block18_json_values(series):
    return [None if pd.isna(value) else float(value) for value in pd.Series(series)]

def _block18_diagnostics_playback_titles(ratio_type="sharpe"):
    ratio_label = _block18_ratio_label(ratio_type)
    return (
        f"Current {ratio_label} Z-Score by Window",
        f"First Horizon Derivative of Current {ratio_label} Z-Score",
        f"Second Horizon Derivative of Current {ratio_label} Z-Score",
    )

def _block18_diagnostics_annotation_for_date(
    annotation_text, as_of_date, ratio_type="sharpe"
):
    annotation_text = str(annotation_text)
    for title in _block18_diagnostics_playback_titles(ratio_type):
        if annotation_text.startswith(title):
            return f"{title} ({as_of_date:%Y-%m-%d})"
    return None

def _block18_apply_window_diagnostics_asof(
    figure, playback_position, selected_benchmarks, spline_toggle=None, spline_strength=None,
    raw_overlay_toggle=None, ratio_type="sharpe",
):
    trace_updates, as_of_date, playback_position = _block18_window_diagnostics_trace_updates(
        playback_position, selected_benchmarks, spline_toggle, spline_strength,
        ratio_type,
    )
    # Create the requested trace structure without repeating the spline/feature
    # calculations already completed in ``trace_updates``.
    figure = _block18_configure_spline_traces(
        figure, spline_toggle, spline_strength, raw_overlay_toggle,
        compute_values=False,
    )
    for trace in figure.data:
        trace_name = str(getattr(trace, "name", ""))
        if trace_name in trace_updates:
            trace.y = _block18_json_values(trace_updates[trace_name])
    for annotation in figure.layout.annotations or []:
        annotation_text = str(getattr(annotation, "text", ""))
        updated_text = _block18_diagnostics_annotation_for_date(
            annotation_text, as_of_date, ratio_type
        )
        if updated_text is not None:
            annotation.text = updated_text
    return figure, as_of_date

def _block18_update_functional_profile_date_annotations(
    figure, as_of_date, ratio_type="sharpe"
):
    for annotation in figure.layout.annotations or []:
        updated_text = _block18_diagnostics_annotation_for_date(
            getattr(annotation, "text", ""), as_of_date, ratio_type
        )
        if updated_text is not None:
            annotation.text = updated_text
    return figure


def _block18_functional_playback_signature(
    selected_benchmarks, horizon_range, spline_toggle, spline_strength,
    raw_overlay_toggle, cache_version=0, ratio_type="sharpe",
):
    try:
        cache_version = int(cache_version or 0)
    except (TypeError, ValueError):
        cache_version = 0
    return (
        _block18_normalize_ratio_type(ratio_type),
        tuple(_block18_normalize_benchmark_selection(selected_benchmarks)),
        tuple(_block18_normalize_diagnostics_window_range(horizon_range)),
        _block18_spline_method(spline_toggle),
        float(_block18_normalize_spline_strength(spline_strength)),
        bool(_block18_raw_overlay_is_enabled(raw_overlay_toggle)),
        cache_version,
    )


def _block18_values_at_horizons(values, horizons):
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric.index = pd.to_numeric(pd.Series(numeric.index), errors="coerce").to_numpy()
    numeric = numeric.loc[~pd.isna(numeric.index)]
    numeric = numeric.loc[~numeric.index.duplicated(keep="last")]
    target_horizons = pd.to_numeric(
        pd.Series(list(horizons)), errors="coerce"
    ).to_numpy(dtype=float)
    return _block18_json_values(numeric.reindex(target_horizons))


def _block18_register_functional_playback_plan(figure, signature):
    """Cache the final three-row trace topology used by lightweight playback."""
    ratio_type = _block18_normalize_ratio_type(
        signature[0] if signature else block18_default_ratio_type
    )
    ratio_label = _block18_ratio_label(ratio_type)
    trace_targets = {}
    segment_targets = []
    for trace_index, trace in enumerate(figure.data):
        meta = getattr(trace, "meta", None)
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("quantapp_role") == "functional_segment_level":
            source_trace_name = meta.get("source_trace_name")
            owner = str(meta.get("owner") or ticker_str)
            if not source_trace_name:
                source_trace_name = (
                    f"Current {ratio_label} Z-Score"
                    if owner == ticker_str
                    else f"{owner} Current {ratio_label} Z-Score"
                )
            segment_targets.append({
                "index": trace_index,
                "source_trace_name": str(source_trace_name),
                "owner": owner,
                "segment_label": str(meta.get("segment_label") or "Segment"),
                "segment_start": meta.get("segment_start"),
                "segment_end": meta.get("segment_end"),
                "visible_min": meta.get("visible_min"),
                "visible_max": meta.get("visible_max"),
                "ratio_label": str(meta.get("ratio_label") or ratio_label),
            })
            continue
        trace_name = str(getattr(trace, "name", ""))
        trace_x = getattr(trace, "x", None)
        if not trace_name or trace_x is None:
            continue
        trace_targets.setdefault(trace_name, []).append({
            "index": trace_index,
            "horizons": list(trace_x),
        })

    annotation_targets = []
    for annotation_index, annotation in enumerate(figure.layout.annotations or []):
        annotation_text = str(getattr(annotation, "text", ""))
        for title in _block18_diagnostics_playback_titles(ratio_type):
            if annotation_text.startswith(title):
                annotation_targets.append((annotation_index, title))
                break

    if f"Current {ratio_label} Z-Score" not in trace_targets:
        block18_functional_playback_plan_cache.pop(signature, None)
        return False
    if len(block18_functional_playback_plan_cache) >= 16:
        block18_functional_playback_plan_cache.clear()
    block18_functional_playback_plan_cache[signature] = {
        "trace_targets": trace_targets,
        "segment_targets": segment_targets,
        "annotation_targets": annotation_targets,
    }
    return True


def _block18_patch_functional_horizon_playback(
    signature, playback_position, selected_benchmarks, horizon_range,
    spline_toggle=None, spline_strength=None, ratio_type="sharpe",
):
    """Patch only per-date values; structural controls still receive full figures."""
    plan = block18_functional_playback_plan_cache.get(signature)
    if plan is None:
        return None, None
    trace_updates, as_of_date, _ = _block18_window_diagnostics_trace_updates(
        playback_position, selected_benchmarks, spline_toggle, spline_strength,
        ratio_type,
    )
    patched_figure = Patch()
    for trace_name, targets in plan["trace_targets"].items():
        values = trace_updates.get(trace_name)
        if values is None:
            continue
        for target in targets:
            patched_figure["data"][target["index"]]["y"] = (
                _block18_values_at_horizons(values, target["horizons"])
            )

    lower, upper = _block18_normalize_diagnostics_window_range(horizon_range)
    for target in plan["segment_targets"]:
        raw_values = trace_updates.get(target["source_trace_name"])
        segment_level = None
        visible_min = target["visible_min"]
        visible_max = target["visible_max"]
        if raw_values is not None:
            numeric = pd.to_numeric(pd.Series(raw_values), errors="coerce")
            horizons = pd.to_numeric(
                pd.Series(numeric.index), errors="coerce"
            ).to_numpy(dtype=float)
            values = numeric.to_numpy(dtype=float)
            segment_horizons = np.isfinite(horizons)
            segment_horizons &= (horizons >= lower) & (horizons <= upper)
            segment_horizons &= _block18_segment_horizon_mask(
                horizons, target["segment_start"], target["segment_end"]
            )
            if segment_horizons.any():
                visible_horizons = horizons[segment_horizons]
                visible_min = float(np.min(visible_horizons))
                visible_max = float(np.max(visible_horizons))
            selected = segment_horizons & np.isfinite(values)
            if selected.any():
                segment_level = float(np.mean(values[selected]))
        patched_figure["data"][target["index"]]["y"] = [
            segment_level, segment_level
        ]
        patched_figure["data"][target["index"]]["hovertemplate"] = (
            _block18_functional_segment_level_hover(
                target["owner"], target["segment_label"],
                visible_min, visible_max, segment_level,
                target.get("ratio_label") or _block18_ratio_label(ratio_type),
            )
        )

    for annotation_index, title in plan["annotation_targets"]:
        patched_figure["layout"]["annotations"][annotation_index]["text"] = (
            f"{title} ({as_of_date:%Y-%m-%d})"
        )
    return patched_figure, as_of_date


def _block18_functional_horizon_status(
    selected_benchmarks, horizon_range, spline_toggle, spline_strength,
    raw_overlay_toggle, as_of_date, ratio_type="sharpe",
):
    lower, upper = _block18_normalize_diagnostics_window_range(horizon_range)
    status = (
        f"{block18_tab_config['window_diagnostics']['label']} | "
        f"{block18_native_axis_status['window_diagnostics']} | Benchmarks: "
        f"{_block18_benchmark_selection_label(selected_benchmarks)} | "
        f"Ratio: {_block18_ratio_label(ratio_type)} | "
        f"Windows: {lower}-{upper} days"
    )
    status += (
        f" | {_block18_spline_method_label(spline_toggle)}: "
        f"{_block18_normalize_spline_strength(spline_strength):.0f}%"
        if _block18_spline_is_enabled(spline_toggle)
        else " | Smoothing: Off"
    )
    if (
        _block18_spline_is_enabled(spline_toggle)
        and _block18_raw_overlay_is_enabled(raw_overlay_toggle)
    ):
        status += " | Raw overlay: On"
    return status + f" | As of: {pd.Timestamp(as_of_date):%Y-%m-%d}"

def _block18_playback_date(position):
    try:
        position = int(position)
    except (TypeError, ValueError):
        position = block18_default_playback_position
    position = max(0, min(position, len(block18_playback_dates) - 1))
    return pd.Timestamp(block18_playback_dates[position]), position

def _block18_playback_position_for_date(selected_date):
    try:
        selected_date = pd.Timestamp(selected_date)
        if pd.isna(selected_date):
            return block18_default_playback_position
        if selected_date.tzinfo is not None:
            selected_date = selected_date.tz_convert(None)
        selected_date = selected_date.normalize()
    except (TypeError, ValueError):
        return block18_default_playback_position
    position = int(block18_playback_dates.searchsorted(selected_date, side="right") - 1)
    return max(0, min(position, len(block18_playback_dates) - 1))

def _block18_cached_tab_figure(
    active_tab, window, display_range_value, selected_benchmarks,
    cache_version=0, ratio_type="sharpe", autocorrelation_lag_range=None,
    autocorrelation_return_horizon=block18_default_autocorrelation_return_horizon,
    autocorrelation_sampling_mode=block18_default_autocorrelation_sampling_mode,
    autocorrelation_comparison_lag=block18_default_autocorrelation_comparison_lag,
    first_passage_threshold=block18_default_first_passage_threshold,
    first_passage_end_threshold=block18_default_first_passage_end_threshold,
    first_passage_start_condition=block18_default_first_passage_start_condition,
    first_passage_end_condition=block18_default_first_passage_end_condition,
    first_passage_start_sign=block18_default_first_passage_start_sign,
    first_passage_end_sign=block18_default_first_passage_end_sign,
):
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    if active_tab == "volatility_efficiency":
        display_range_value = "full"
        selected_benchmarks = []
    cache_key = (
        active_tab,
        int(window),
        display_range_value,
        tuple(selected_benchmarks),
        ratio_type
        if active_tab in block18_ratio_dependent_tabs
        or active_tab in {"autocorrelation", "first_passage"}
        else None,
        tuple(autocorrelation_lag_range or ()) if active_tab == "autocorrelation" else None,
        int(autocorrelation_return_horizon or 1) if active_tab == "autocorrelation" else None,
        autocorrelation_sampling_mode if active_tab == "autocorrelation" else None,
        int(autocorrelation_comparison_lag or 1) if active_tab == "autocorrelation" else None,
        float(first_passage_threshold or block18_default_first_passage_threshold)
        if active_tab == "first_passage" else None,
        float(first_passage_end_threshold or 0.0)
        if active_tab == "first_passage" else None,
        first_passage_start_condition if active_tab == "first_passage" else None,
        first_passage_end_condition if active_tab == "first_passage" else None,
        first_passage_start_sign if active_tab == "first_passage" else None,
        first_passage_end_sign if active_tab == "first_passage" else None,
        int(cache_version or 0),
    )
    if cache_key not in block18_base_figure_cache:
        if len(block18_base_figure_cache) >= 24:
            block18_base_figure_cache.clear()
        block18_base_figure_cache[cache_key] = _block18_build_tab_figure(
            active_tab, window, display_range_value, selected_benchmarks,
            ratio_type,
            autocorrelation_lag_range,
            autocorrelation_return_horizon,
            autocorrelation_sampling_mode,
            autocorrelation_comparison_lag,
            first_passage_threshold,
            first_passage_end_threshold,
            first_passage_start_condition,
            first_passage_end_condition,
            first_passage_start_sign, first_passage_end_sign,
        )
    return go.Figure(block18_base_figure_cache[cache_key])

def _block18_next_playback_state(trigger_id, playing, position, active_tab):
    maximum = block18_default_playback_position
    _, position = _block18_playback_date(position)
    playing = bool(playing)

    if active_tab != "window_diagnostics":
        return False, True, "Play", position
    if trigger_id == "diagnostics-step-back-button":
        return False, True, "Play", max(0, position - 1)
    if trigger_id == "diagnostics-step-forward-button":
        next_position = min(position + 1, maximum)
        return False, True, "Replay" if next_position >= maximum else "Play", next_position
    if trigger_id == "diagnostics-playback-button":
        playing = not playing
        if playing and position >= maximum:
            position = 0
    elif trigger_id == "diagnostics-playback-interval" and playing:
        position = min(position + block18_playback_step, maximum)
        if position >= maximum:
            return False, True, "Replay", position

    return playing, not playing, "Pause" if playing else "Play", position

def _block18_render_dashboard_figure(
    active_tab,
    window_value,
    display_range_value,
    selected_benchmarks=None,
    ratio_type="sharpe",
    playback_position=None,
    diagnostics_window_range=None,
    spline_toggle=None,
    spline_strength=None,
    raw_overlay_toggle=None,
    autocorrelation_lag_range=None,
    autocorrelation_return_horizon=block18_default_autocorrelation_return_horizon,
    autocorrelation_sampling_mode=block18_default_autocorrelation_sampling_mode,
    autocorrelation_comparison_lag=block18_default_autocorrelation_comparison_lag,
    first_passage_threshold=block18_default_first_passage_threshold,
    first_passage_end_threshold=block18_default_first_passage_end_threshold,
    first_passage_start_condition=block18_default_first_passage_start_condition,
    first_passage_end_condition=block18_default_first_passage_end_condition,
    first_passage_start_sign=block18_default_first_passage_start_sign,
    first_passage_end_sign=block18_default_first_passage_end_sign,
    cache_version=0,
):
    active_tab = active_tab if active_tab in block18_tab_config else block18_default_tab
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    ratio_label = _block18_ratio_label(ratio_type)
    display_range_value = display_range_value if display_range_value in block18_display_range_offsets else block18_default_display_range
    selected_benchmarks = _block18_normalize_benchmark_selection(selected_benchmarks)
    label = block18_tab_config[active_tab]["label"]
    display_label = _block18_display_range_label(display_range_value)

    try:
        _block18_validate_peer_state(active_tab, required_benchmarks=selected_benchmarks)
        if active_tab in {"risk", "systematic_risk", "kappa", "pezier_white", "cornish_fisher", "autocorrelation", "first_passage", "drawdown", "correlation", "volatility_efficiency"}:
            window = _block18_validate_window(window_value)
            figure = _block18_cached_tab_figure(
                active_tab, window, display_range_value, selected_benchmarks,
                cache_version, ratio_type, autocorrelation_lag_range,
                autocorrelation_return_horizon, autocorrelation_sampling_mode,
                autocorrelation_comparison_lag, first_passage_threshold,
                first_passage_end_threshold, first_passage_start_condition,
                first_passage_end_condition, first_passage_start_sign,
                first_passage_end_sign
            )
            status = f"{label} | Window: {window:,} trading days | Display: {display_label}"
        else:
            figure = _block18_cached_tab_figure(
                active_tab, block18_default_window, display_range_value,
                selected_benchmarks, cache_version, ratio_type,
                autocorrelation_lag_range, autocorrelation_return_horizon,
                autocorrelation_sampling_mode, autocorrelation_comparison_lag,
                first_passage_threshold, first_passage_end_threshold
                , first_passage_start_condition, first_passage_end_condition
                , first_passage_start_sign, first_passage_end_sign
            )
            status = f"{label} | Display: {display_label}" if active_tab in block18_date_range_tabs else f"{label} | {block18_native_axis_status.get(active_tab, 'Native axes')}"
        _block18_validate_peer_state(active_tab, figure, selected_benchmarks)
        playback_date, playback_position = _block18_playback_date(playback_position)
        if active_tab == "window_diagnostics":
            figure, playback_date = _block18_apply_window_diagnostics_asof(
                figure, playback_position, selected_benchmarks, spline_toggle, spline_strength,
                raw_overlay_toggle, ratio_type,
            )
        if active_tab in block18_date_range_tabs:
            figure = _block18_apply_display_range(
                figure,
                display_range_value,
                preserve_benchmark_selector=active_tab == "heatmap",
            )
            if active_tab == "first_passage" and hasattr(figure.layout, "xaxis3"):
                # The third x-axis is a numeric duration histogram, not a date axis.
                figure.layout.xaxis3.update(autorange=True, range=None)
        else:
            figure = _block18_strip_figure_controls(
                figure,
                preserve_all_controls=active_tab == "historical_surface_3d",
            )
        if active_tab in {"heatmap", "historical_surface_3d", "window_diagnostics"}:
            figure, diagnostics_window_range = _block18_apply_horizon_range(
                figure, active_tab, diagnostics_window_range
            )
        if active_tab == "heatmap":
            history_start, history_end = trace_datetime_bounds(figure.data)
            if history_start is not None and history_end is not None:
                figure = _block18_add_history_range_controls(
                    figure,
                    ["xaxis", "xaxis2", "xaxis3", "xaxis4"],
                    history_start,
                    history_end,
                )
        if active_tab in {"heatmap", "historical_surface_3d", "window_diagnostics", "risk", "systematic_risk", "correlation"}:
            status += " | Benchmarks: " + _block18_benchmark_selection_label(selected_benchmarks)
        if active_tab in {"heatmap", "historical_surface_3d"}:
            status += f" | Horizons: {diagnostics_window_range[0]}-{diagnostics_window_range[1]} days"
        if active_tab == "window_diagnostics":
            figure = _block18_add_functional_segment_levels(
                figure, diagnostics_window_range
            )
            figure = _block18_functional_horizon_profile_figure(
                figure, diagnostics_window_range, ratio_type=ratio_type
            )
            figure = _block18_update_functional_profile_date_annotations(
                figure, playback_date, ratio_type
            )
            status = _block18_functional_horizon_status(
                selected_benchmarks, diagnostics_window_range,
                spline_toggle, spline_strength, raw_overlay_toggle,
                playback_date, ratio_type,
            )
        elif active_tab in block18_ratio_dependent_tabs:
            status += f" | Ratio: {ratio_label}"
        return _block18_apply_typography(figure), status
    except Exception as error:
        message = f"{label} could not render: {error}"
        return _block18_apply_typography(
            _block18_error_figure(message, height=block18_tab_config[active_tab]["height"])
        ), message

block18_initial_figure, block18_initial_status = _block18_render_dashboard_figure(
    block18_default_tab,
    block18_default_window,
    block18_default_display_range,
    block18_default_benchmark_selection,
    ratio_type=block18_default_ratio_type,
)

block18_update_button_style = {
    "height": "38px", "alignSelf": "end", "backgroundColor": "#2563eb",
    "border": "1px solid #1d4ed8", "borderRadius": "5px", "color": "#ffffff",
    "fontWeight": "600", "padding": "0 16px", "cursor": "pointer",
    "boxShadow": "0 1px 3px rgba(37, 99, 235, 0.35)", "transition": "all 150ms ease",
}
block18_update_button_loading_style = {
    **block18_update_button_style, "backgroundColor": "#0f766e", "border": "1px solid #2dd4bf",
    "color": "#ecfeff", "cursor": "progress", "boxShadow": "0 0 0 3px rgba(45, 212, 191, 0.22)",
    "transform": "translateY(1px)", "opacity": 0.92,
}

block18_controls = html.Div(
    [
        html.Div(
            [
                html.Div("Window", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                dcc.Input(
                    id="shared-window-input",
                    type="number",
                    min=2,
                    max=10000,
                    step=1,
                    value=block18_default_window,
                    debounce=True,
                    style={
                        "width": "130px",
                        "height": "36px",
                        "backgroundColor": "#111827",
                        "border": "1px solid #334155",
                        "color": "#f8fafc",
                        "padding": "0 10px",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    "Metric",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                ),
                dcc.Dropdown(
                    id="shared-ratio-dropdown",
                    options=block18_ratio_options,
                    value=block18_default_ratio_type,
                    clearable=False,
                    style={"width": "360px", "color": "#0f172a"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div("Lookback", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="shared-display-range-dropdown",
                    options=block18_display_range_options,
                    value=block18_default_display_range,
                    clearable=False,
                    style={"width": "150px", "color": "#0f172a"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div("Benchmark indices", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="benchmark-index-dropdown",
                    options=[{"label": symbol, "value": symbol} for symbol in benchmark_order],
                    value=block18_default_benchmark_selection,
                    multi=True,
                    clearable=False,
                    placeholder="Select benchmark indices",
                    style={"width": "520px", "color": "#0f172a"},
                ),
            ]
        ),
        html.Button(
            "Update",
            id="shared-update-button",
            n_clicks=0,
            style=block18_update_button_style,
        ),
        html.Div(
            id="shared-view-status",
            children=block18_initial_status,
            style={"color": "#cbd5e1", "alignSelf": "end", "paddingBottom": "9px"},
        ),
        html.Div(
            [
                html.Div(
                    "Horizon range (minimum–maximum days)",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                ),
                html.Div(
                    [
                        html.Div([
                            html.Div("Minimum days", style={"fontSize": "10px", "color": "#64748b"}),
                            dcc.Input(
                                id="historical-horizon-min-input", type="number",
                                min=block18_diagnostics_window_min,
                                max=block18_diagnostics_window_max, step=1,
                                value=block18_default_diagnostics_window_range[0],
                                debounce=True,
                                style={"width": "130px", "height": "36px", "padding": "0 10px"},
                            ),
                        ]),
                        html.Div([
                            html.Div("Maximum days", style={"fontSize": "10px", "color": "#64748b"}),
                            dcc.Input(
                                id="historical-horizon-max-input", type="number",
                                min=block18_diagnostics_window_min,
                                max=block18_diagnostics_window_max, step=1,
                                value=block18_default_diagnostics_window_range[1],
                                debounce=True,
                                style={"width": "130px", "height": "36px", "padding": "0 10px"},
                            ),
                        ]),
                    ],
                    style={"display": "flex", "gap": "12px", "alignItems": "end"},
                ),
            ],
            id="historical-horizon-range-control",
            style={"display": "block", "flex": "1 1 100%", "minWidth": "420px", "padding": "4px 8px 0"},
        ),
    ],
    style={
        "display": "flex",
        "gap": "12px",
        "alignItems": "stretch",
        "flexWrap": "wrap",
        "marginBottom": "12px",
    },
)

block18_initial_autocorrelation_lag_max = max(
    1, min(block18_autocorrelation_lag_max, block18_default_window - 1)
)
block18_autocorrelation_controls = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Metric horizon (trading days)",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                        ),
                        dcc.Input(
                            id="autocorrelation-return-horizon-input",
                            type="number",
                            min=1,
                            step=1,
                            value=block18_default_autocorrelation_return_horizon,
                            debounce=True,
                            style={"width": "150px", "height": "36px", "padding": "0 10px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            "Return sampling",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                        ),
                        dcc.RadioItems(
                            id="autocorrelation-sampling-mode",
                            options=[
                                {"label": " Non-overlapping", "value": "non_overlapping"},
                                {"label": " Overlapping", "value": "overlapping"},
                            ],
                            value=block18_default_autocorrelation_sampling_mode,
                            inline=True,
                            style={"display": "flex", "gap": "18px", "color": "#e2e8f0"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            "Current vs lagged comparison",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                        ),
                        dcc.Input(
                            id="autocorrelation-comparison-lag-input",
                            type="number",
                            min=1,
                            max=block18_initial_autocorrelation_lag_max,
                            step=1,
                            value=block18_default_autocorrelation_comparison_lag,
                            debounce=True,
                            style={"width": "150px", "height": "36px", "padding": "0 10px"},
                        ),
                    ]
                ),
            ],
            style={"display": "flex", "gap": "24px", "alignItems": "end", "marginBottom": "12px"},
        ),
        html.Div(
            "Displayed lag range",
            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "8px"},
        ),
        dcc.RangeSlider(
            id="autocorrelation-lag-range-slider",
            min=1,
            max=block18_initial_autocorrelation_lag_max,
            step=1,
            value=[
                1,
                min(
                    block18_default_autocorrelation_lag_range[1],
                    block18_initial_autocorrelation_lag_max,
                ),
            ],
            marks={
                lag: str(lag)
                for lag in (1, 5, 10, 20, 40, 60, 100, 150, 200)
                if lag <= block18_initial_autocorrelation_lag_max
            },
            allowCross=False,
            pushable=1,
            updatemode="mouseup",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ],
    id="autocorrelation-lag-range-control",
    style={
        "display": "none",
        "marginBottom": "14px",
        "padding": "10px 18px 18px",
        "border": "1px solid #1e293b",
        "borderRadius": "6px",
        "backgroundColor": "#0f172a",
    },
)

block18_first_passage_controls = html.Div(
    [
        html.Div(
            [
                html.Div(
                    "Starting boundary (standard deviations)",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                ),
                dcc.Dropdown(
                    id="first-passage-start-sign-dropdown",
                    className="first-passage-dropdown",
                    options=[
                        {"label": "+ Positive", "value": "positive"},
                        {"label": "- Negative", "value": "negative"},
                        {"label": "+/- Both", "value": "both"},
                    ],
                    value=block18_default_first_passage_start_sign,
                    clearable=False,
                    style={"width": "180px", "marginBottom": "6px", "color": "#0f172a"},
                ),
                dcc.Dropdown(
                    id="first-passage-start-condition-dropdown",
                    className="first-passage-dropdown",
                    options=[
                        {"label": "Greater than (>)", "value": "above"},
                        {"label": "Less than (<)", "value": "below"},
                    ],
                    value=block18_default_first_passage_start_condition,
                    clearable=False,
                    style={"width": "180px", "marginBottom": "6px", "color": "#0f172a"},
                ),
                dcc.Input(
                    id="first-passage-threshold-input",
                    type="number", step=0.1,
                    value=block18_default_first_passage_threshold,
                    debounce=True,
                    style={"width": "180px", "height": "36px", "padding": "0 10px"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    "Ending boundary (standard deviations; 0 = mean)",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                ),
                dcc.Dropdown(
                    id="first-passage-end-sign-dropdown",
                    className="first-passage-dropdown",
                    options=[
                        {"label": "+ Positive", "value": "positive"},
                        {"label": "- Negative", "value": "negative"},
                        {"label": "+/- Both", "value": "both"},
                    ],
                    value=block18_default_first_passage_end_sign,
                    clearable=False,
                    style={"width": "180px", "marginBottom": "6px", "color": "#0f172a"},
                ),
                dcc.Dropdown(
                    id="first-passage-end-condition-dropdown",
                    className="first-passage-dropdown",
                    options=[
                        {"label": "Less than (<)", "value": "below"},
                        {"label": "Greater than (>)", "value": "above"},
                    ],
                    value=block18_default_first_passage_end_condition,
                    clearable=False,
                    style={"width": "180px", "marginBottom": "6px", "color": "#0f172a"},
                ),
                dcc.Input(
                    id="first-passage-end-threshold-input",
                    type="number", step=0.1,
                    value=block18_default_first_passage_end_threshold,
                    debounce=True,
                    style={"width": "180px", "height": "36px", "padding": "0 10px"},
                ),
            ]
        ),
    ],
    id="first-passage-threshold-control",
    style={
        "display": "none",
        "marginBottom": "14px",
        "padding": "10px 18px 14px",
        "border": "1px solid #1e293b",
        "borderRadius": "6px",
        "backgroundColor": "#0f172a",
        "gap": "24px",
        "alignItems": "end",
    },
)

block18_playback_controls_style = {
    "display": "none",
    "gap": "14px",
    "alignItems": "stretch",
    "flexWrap": "wrap",
    "marginTop": "16px",
    "padding": "10px 12px",
    "border": "1px solid #1e293b",
    "borderRadius": "6px",
}

block18_playback_controls = html.Div(
    [
        html.Div(
            [
                html.Div("Functional horizon playback", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                html.Button(
                    "Play",
                    id="diagnostics-playback-button",
                    n_clicks=0,
                    style={
                        "height": "36px",
                        "width": "78px",
                        "backgroundColor": "#0f766e",
                        "border": "1px solid #14b8a6",
                        "color": "#ffffff",
                    },
                ),
            ]
        ),
        html.Div(
            [
                html.Div("Speed", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="diagnostics-playback-speed-dropdown",
                    options=block18_playback_speed_options,
                    value=block18_default_playback_speed,
                    clearable=False,
                    style={"width": "120px", "color": "#0f172a"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div("Jump to date", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                dcc.DatePickerSingle(
                    id="diagnostics-date-picker",
                    min_date_allowed=block18_playback_dates[0].date(),
                    max_date_allowed=block18_playback_dates[-1].date(),
                    date=block18_playback_dates[block18_default_playback_position].date(),
                    display_format="YYYY-MM-DD",
                    clearable=False,
                    first_day_of_week=0,
                    style={"fontSize": "13px"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div("As-of date (profile and derivative rows)", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                html.Div(
                    [
                        html.Button(
                            "←",
                            id="diagnostics-step-back-button",
                            n_clicks=0,
                            title="Previous trading date",
                            style={"width": "38px", "height": "32px", "fontSize": "20px", "backgroundColor": "#1e293b", "border": "1px solid #475569", "color": "#f8fafc"},
                        ),
                        html.Div(
                            dcc.Slider(
                                id="diagnostics-asof-slider",
                                min=0,
                                max=len(block18_playback_dates) - 1,
                                step=1,
                                value=block18_default_playback_position,
                                marks=block18_playback_marks,
                                included=False,
                                updatemode="drag",
                            ),
                            style={"flex": "1 1 auto", "minWidth": "300px", "padding": "0 8px"},
                        ),
                        html.Button(
                            "→",
                            id="diagnostics-step-forward-button",
                            n_clicks=0,
                            title="Next trading date",
                            style={"width": "38px", "height": "32px", "fontSize": "20px", "backgroundColor": "#1e293b", "border": "1px solid #475569", "color": "#f8fafc"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "width": "100%"},
                ),
            ],
            style={"flex": "1 1 680px", "minWidth": "420px"},
        ),
        html.Div(
            [
                html.Div(
                    "Lookback horizon range shown in profile and derivative rows",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                ),
                html.Div(
                    [
                        html.Div([
                            html.Div("Minimum days", style={"fontSize": "10px", "color": "#64748b"}),
                            dcc.Input(
                                id="diagnostics-horizon-min-input", type="number",
                                min=block18_diagnostics_window_min,
                                max=block18_diagnostics_window_max, step=1,
                                value=block18_default_diagnostics_window_range[0],
                                debounce=True,
                                style={"width": "130px", "height": "36px", "padding": "0 10px"},
                            ),
                        ]),
                        html.Div([
                            html.Div("Maximum days", style={"fontSize": "10px", "color": "#64748b"}),
                            dcc.Input(
                                id="diagnostics-horizon-max-input", type="number",
                                min=block18_diagnostics_window_min,
                                max=block18_diagnostics_window_max, step=1,
                                value=block18_default_diagnostics_window_range[1],
                                debounce=True,
                                style={"width": "130px", "height": "36px", "padding": "0 10px"},
                            ),
                        ]),
                    ],
                    style={"display": "flex", "gap": "12px", "alignItems": "end"},
                ),
            ],
            style={"flex": "1 1 100%", "minWidth": "420px", "padding": "4px 8px 0"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Spline curves & feature markers",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"},
                        ),
                        dcc.RadioItems(
                            id="diagnostics-spline-toggle",
                            options=[
                                {"label": " Off (hide spline + feature markers)", "value": "off"},
                                {"label": " B-spline", "value": "b_spline"},
                                {"label": " P-spline", "value": "p_spline"},
                            ],
                            value=block18_default_spline_toggle,
                            inline=True,
                            style={"color": "#e2e8f0", "whiteSpace": "nowrap", "display": "flex", "gap": "12px"},
                        ),
                        dcc.Checklist(
                            id="diagnostics-raw-overlay-toggle",
                            options=[{"label": " Overlay raw values", "value": "raw_overlay"}],
                            value=block18_default_raw_overlay_toggle,
                            inline=True,
                            style={"color": "#e2e8f0", "whiteSpace": "nowrap", "marginTop": "4px"},
                        ),
                    ],
                    style={"flex": "0 0 215px"},
                ),
                html.Div(
                    [
                        html.Div("Spline smoothing strength", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "4px"}),
                        dcc.Slider(
                            id="diagnostics-spline-strength-slider",
                            min=0,
                            max=100,
                            step=1,
                            value=block18_default_spline_strength,
                            marks={0: "Exact", 25: "Light", 50: "Medium", 75: "Strong", 100: "Max"},
                            updatemode="mouseup",
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    style={"flex": "1 1 520px", "minWidth": "360px"},
                ),
            ],
            style={"display": "flex", "gap": "18px", "alignItems": "center", "flex": "1 1 100%", "paddingTop": "8px"},
        ),
    ],
    id="diagnostics-playback-controls",
    style=block18_playback_controls_style,
)

block18_metric_graph_specs = [
    ("volatility", "Volatility & GARCH(1,1)", "1 / 1"),
    ("entropy", "Normalized Entropy", "1 / 2"),
    ("autocorrelation", "Lag-1 Autocorrelation", "2 / 1"),
    ("hurst", "Hurst Exponent", "2 / 2"),
    ("vol_of_vol", "Volatility of Volatility", "3 / 1 / 4 / 3"),
    ("skew", "Rolling Skew Z-Score", "4 / 1 / 5 / 3"),
    ("kurtosis", "Rolling Excess Kurtosis Z-Score", "5 / 1 / 6 / 3"),
    ("gini", "Rolling Gini Z-Score", "6 / 1 / 7 / 3"),
]

def _block18_metric_loading_panel(metric_key, title, grid_area):
    return html.Div(
        dcc.Loading(
            id=f"{metric_key}-loading",
            custom_spinner=html.Div(
                [
                    html.Div(className="quantapp-loading-spinner"),
                    html.Div(
                        f"Rendering {title}…",
                        style={"fontSize": "14px", "fontWeight": "600", "color": "#f8fafc"},
                    ),
                ],
                style={"textAlign": "center"},
            ),
            delay_show=75,
            delay_hide=100,
            parent_style={"minHeight": "260px", "backgroundColor": "#0b0f14"},
            children=dcc.Graph(
                id=f"volatility-efficiency-{metric_key}-graph",
                figure=_block18_error_figure(f"Loading {title}...", height=430),
                config={"responsive": True, "displaylogo": False},
                style={"height": "430px"},
            ),
        ),
        style={"gridArea": grid_area, "minWidth": 0},
    )

block18_volatility_efficiency_grid = html.Div(
    [_block18_metric_loading_panel(*spec) for spec in block18_metric_graph_specs],
    id="volatility-efficiency-plot-grid",
    style={
        "display": "none", "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
        "gridTemplateRows": "repeat(6, auto)", "gap": "14px", "marginTop": "12px",
    },
)

def _block18_combined_section(title, subtitle, graph_id, height, accent):
    return html.Section(
        [
            html.Div(
                [
                    html.Div(title, style={"fontSize": "18px", "fontWeight": "700", "color": "#f8fafc"}),
                    html.Div(subtitle, style={"fontSize": "12px", "color": "#94a3b8", "marginTop": "4px"}),
                ],
                style={
                    "padding": "15px 18px 12px",
                    "borderLeft": f"4px solid {accent}",
                    "background": "linear-gradient(90deg, rgba(30, 41, 59, 0.82), rgba(15, 23, 42, 0.45))",
                },
            ),
            dcc.Loading(
                custom_spinner=html.Div(
                    [
                        html.Div(className="quantapp-loading-spinner"),
                        html.Div(
                            f"Rendering {title}...",
                            style={"fontSize": "16px", "fontWeight": "600", "color": "#f8fafc"},
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
                delay_show=75,
                delay_hide=100,
                parent_style={"minHeight": f"{height}px", "backgroundColor": "#0b0f14"},
                children=dcc.Graph(
                    id=graph_id,
                    figure=_block18_error_figure(f"Loading {title}...", height=height),
                    config={"responsive": True, "displaylogo": False},
                    style={"height": f"{height}px"},
                ),
            ),
        ],
        style={
            "overflow": "hidden", "border": "1px solid #263449",
            "borderRadius": "10px", "backgroundColor": "#0b0f14",
            "boxShadow": "0 10px 30px rgba(0, 0, 0, 0.22)",
        },
    )


block18_risk_compounding_v2_view = html.Div(
    [
        _block18_combined_section(
            "Integrated Risk, Compounding & Passage Boundaries",
            "All three original panels with shared boundary guides, passage starts, and passage completions",
            "risk-compounding-v2-risk-graph",
            block18_tab_config["risk"]["height"],
            block18_tab_config["risk"]["accent"],
        ),
        _block18_combined_section(
            "Risk-Adjusted Return Passage Data",
            "Historical durations, expanding mean/median/maximum, distribution, and event summary",
            "risk-compounding-v2-risk-adjusted-passage-graph",
            610,
            block18_tab_config["first_passage"]["accent"],
        ),
        _block18_combined_section(
            "Ratio Spread Passage Data",
            "First-passage behavior of the selected benchmark-minus-asset risk-adjusted spread",
            "risk-compounding-v2-spread-passage-graph",
            610,
            "#a78bfa",
        ),
        _block18_combined_section(
            "Volatility Drag Passage Data",
            "First-passage behavior of the asset's volatility-drag MAD score",
            "risk-compounding-v2-volatility-drag-passage-graph",
            610,
            "#e879f9",
        ),
    ],
    id="risk-compounding-v2-view",
    style={"display": "none", "gap": "18px", "marginTop": "12px"},
)

block18_dash_app = Dash(__name__)


def _block18_loading_panel():
    return html.Div(
        [
            html.Div(className="quantapp-loading-spinner"),
            html.Div(
                "Rendering selected view…",
                style={"fontSize": "18px", "fontWeight": "600", "color": "#f8fafc"},
            ),
            html.Div(
                "Calculating metrics and preparing chart data",
                style={"fontSize": "13px", "color": "#94a3b8", "marginTop": "7px"},
            ),
        ],
        style={"textAlign": "center"},
    )


block18_dash_app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            :root {{ font-family: {BLOCK18_FONT_FAMILY}; }}
            html, body, #react-entry-point, button, input, select, textarea {{
                font-family: {BLOCK18_FONT_FAMILY};
            }}
            @keyframes quantapp-loading-spin {{
                to {{ transform: rotate(360deg); }}
            }}
            .quantapp-loading-spinner {{
                width: 46px;
                height: 46px;
                margin: 0 auto 18px;
                border: 4px solid #1e293b;
                border-top-color: #60a5fa;
                border-radius: 50%;
                animation: quantapp-loading-spin 0.8s linear infinite;
            }}
            #momentum-efficiency-tab-graph[data-dash-is-loading="true"] {{
                height: 420px !important;
                min-height: 420px !important;
                max-height: 420px !important;
            }}
            .first-passage-dropdown .Select-control,
            .first-passage-dropdown .Select-menu-outer {{
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }}
            .first-passage-dropdown .Select-value-label,
            .first-passage-dropdown .Select-placeholder,
            .first-passage-dropdown .Select-input input,
            .first-passage-dropdown .VirtualizedSelectOption {{
                color: #0f172a !important;
            }}
            .first-passage-dropdown .VirtualizedSelectFocusedOption {{
                background-color: #dbeafe !important;
                color: #0f172a !important;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""
block18_dash_app.layout = html.Div(
    [
        block18_controls,
        block18_autocorrelation_controls,
        block18_first_passage_controls,
        dcc.Interval(
            id="diagnostics-playback-interval",
            interval=block18_default_playback_interval,
            n_intervals=0,
            disabled=True,
        ),
        dcc.Store(id="diagnostics-playback-playing", data=False),
        dcc.Tabs(
            id="momentum-efficiency-view-tabs",
            value=block18_default_tab,
            children=[_block18_tab(tab_value) for tab_value in block18_tab_order],
            colors={"border": "#334155", "primary": "#60a5fa", "background": "#111827"},
        ),
        dcc.Graph(
            id="functional-horizon-summary-graph",
            figure=_block18_error_figure("Functional Horizon summary", height=480),
            config={"responsive": True, "displaylogo": False},
            style={"display": "none", "height": "480px"},
        ),
        block18_playback_controls,
        dcc.Loading(
            id="momentum-efficiency-graph-loading",
            custom_spinner=_block18_loading_panel(),
            delay_show=75,
            delay_hide=100,
            target_components={"momentum-efficiency-tab-graph": "figure"},
            # Keep the graph mounted beneath an opaque placeholder during normal
            # loads. Playback forces this loader to ``hide``, which leaves the
            # mounted graph visible while lightweight frame patches arrive.
            overlay_style={
                "visibility": "visible",
                "backgroundColor": "#0b0f14",
            },
            parent_style={
                "minHeight": "420px",
                "position": "relative",
                "backgroundColor": "#0b0f14",
            },
            children=dcc.Graph(
                id="momentum-efficiency-tab-graph",
                figure=block18_initial_figure,
                animate=False,
                animation_options=_block18_animation_options(block18_default_playback_speed),
                config={"responsive": True, "displaylogo": False},
                style={"height": _block18_graph_height(block18_initial_figure, 2150)},
            ),
        ),
        block18_risk_compounding_v2_view,
        block18_volatility_efficiency_grid,
    ],
    style={
        "backgroundColor": "#0b0f14", "padding": "12px", "color": "#f8fafc",
        "fontFamily": BLOCK18_FONT_FAMILY,
    },
)

block18_metric_panel_cache = {}


@block18_dash_app.callback(
    Output("risk-compounding-v2-view", "style"),
    Output("momentum-efficiency-graph-loading", "parent_style"),
    Input("momentum-efficiency-view-tabs", "value"),
)
def _toggle_risk_compounding_v2_view(active_tab):
    combined_active = active_tab == "risk_compounding_v2"
    return (
        {
            "display": "flex" if combined_active else "none",
            "flexDirection": "column", "gap": "18px", "marginTop": "12px",
        },
        {
            "display": "none" if combined_active else "block",
            "minHeight": "420px", "position": "relative",
            "backgroundColor": "#0b0f14",
        },
    )


@block18_dash_app.callback(
    Output("autocorrelation-lag-range-control", "style"),
    Output("autocorrelation-lag-range-slider", "max"),
    Output("autocorrelation-lag-range-slider", "marks"),
    Output("autocorrelation-comparison-lag-input", "max"),
    Input("momentum-efficiency-view-tabs", "value"),
    Input("shared-update-button", "n_clicks"),
    Input("autocorrelation-return-horizon-input", "value"),
    Input("autocorrelation-sampling-mode", "value"),
    State("shared-window-input", "value"),
)
def _update_autocorrelation_lag_control(
    active_tab, _update_clicks, return_horizon, sampling_mode, window,
):
    try:
        window = _block18_validate_window(window)
        return_horizon = max(1, int(return_horizon))
        sample_count = (
            window
            if sampling_mode == "overlapping"
            else window // return_horizon
        )
        permitted_max = max(
            1, min(block18_autocorrelation_lag_max, sample_count - 1)
        )
    except (TypeError, ValueError):
        permitted_max = block18_initial_autocorrelation_lag_max
    marks = {
        lag: str(lag)
        for lag in (
            1, 5, 10, 20, 40, 60, 100, 200, 500, 1000, 2500, 5000,
            permitted_max,
        )
        if lag <= permitted_max
    }
    style = {
        "display": "block" if active_tab == "autocorrelation" else "none",
        "marginBottom": "14px",
        "padding": "10px 18px 18px",
        "border": "1px solid #1e293b",
        "borderRadius": "6px",
        "backgroundColor": "#0f172a",
    }
    return style, permitted_max, marks, permitted_max


@block18_dash_app.callback(
    Output("first-passage-threshold-control", "style"),
    Input("momentum-efficiency-view-tabs", "value"),
)
def _toggle_first_passage_threshold(active_tab):
    return {
        "display": "flex"
        if active_tab in {"first_passage", "risk_compounding_v2"}
        else "none",
        "marginBottom": "14px",
        "padding": "10px 18px 14px",
        "border": "1px solid #1e293b",
        "borderRadius": "6px",
        "backgroundColor": "#0f172a",
        "gap": "24px",
        "alignItems": "end",
    }


@block18_dash_app.callback(
    Output("momentum-efficiency-graph-loading", "display"),
    Input("diagnostics-playback-playing", "data"),
    Input("momentum-efficiency-view-tabs", "value"),
)
def _suppress_loading_panel_during_playback(playing, active_tab):
    if active_tab == "window_diagnostics" and bool(playing):
        return "hide"
    return "auto"


@block18_dash_app.callback(
    Output("historical-horizon-range-control", "style"),
    Input("momentum-efficiency-view-tabs", "value"),
)
def _toggle_historical_horizon_range(active_tab):
    active_tab = active_tab if active_tab in block18_tab_config else block18_default_tab
    return {
        "display": "block" if active_tab == "historical_surface_3d" else "none",
        "flex": "1 1 100%",
        "minWidth": "420px",
        "padding": "4px 8px 0",
    }


@block18_dash_app.callback(
    Output("functional-horizon-summary-graph", "figure"),
    Output("functional-horizon-summary-graph", "style"),
    Input("momentum-efficiency-view-tabs", "value"),
    Input("shared-update-button", "n_clicks"),
    Input("shared-ratio-dropdown", "value"),
)
def _update_functional_horizon_summary(active_tab, _update_clicks, ratio_type):
    if active_tab != "window_diagnostics":
        return no_update, {"display": "none"}
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    source = _block18_ratio_diagnostics_source_figure(ratio_type)
    summary = _block18_functional_horizon_summary_figure(
        source, ratio_type=ratio_type
    )
    return summary, {"display": "block", "height": _block18_graph_height(summary, 480)}

def _block18_extract_metric_panel(source_figure, axis_name, title, yaxis_title, display_range_value, *, reference_level=None, showlegend=False):
    panel = go.Figure()
    for source_trace in source_figure.data:
        if (getattr(source_trace, "xaxis", None) or "x") != axis_name:
            continue
        trace = copy.deepcopy(source_trace)
        trace.xaxis = None
        trace.yaxis = None
        trace.showlegend = bool(showlegend and trace.showlegend is not False)
        panel.add_trace(trace)
    if reference_level is not None:
        panel.add_hline(y=reference_level, line_dash="dot", line_color="#64748b")
    panel.update_layout(
        title=title, template="plotly_dark", height=430, hovermode="x unified",
        showlegend=showlegend, margin=dict(t=65, r=25, b=45, l=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    panel.update_yaxes(title_text=yaxis_title)
    return _block18_apply_typography(
        _block18_apply_display_range(panel, display_range_value)
    )

def _block18_build_metric_panels(window, display_range_value, cache_version=0):
    window = _block18_validate_window(window)
    cache_key = (window, display_range_value, int(cache_version or 0))
    if cache_key in block18_metric_panel_cache:
        return tuple(go.Figure(figure) for figure in block18_metric_panel_cache[cache_key])
    volatility_figure = _block18_volatility_efficiency_figure(window)
    distribution_figure = _block18_return_distribution_figure(window)
    panel_specs = [
        (volatility_figure, "x", f"{ticker_str} {window}-Day Realized Volatility vs Rolling GARCH(1,1)", "Annualized %", None, True),
        (volatility_figure, "x2", f"Normalized Entropy ({window}d)", "0-1", 1.0, False),
        (volatility_figure, "x3", f"Lag-1 Autocorrelation ({window}d)", "Correlation", 0.0, False),
        (volatility_figure, "x4", f"Hurst Exponent ({window}d)", "H", 0.5, False),
        (volatility_figure, "x5", f"Volatility of Volatility ({window}d)", "Percentage pts", None, False),
        (distribution_figure, "x", f"Rolling Skew Z-Score ({window}d)", "Z-Score", None, False),
        (distribution_figure, "x2", f"Rolling Excess Kurtosis Z-Score ({window}d)", "Z-Score", None, False),
        (distribution_figure, "x3", f"Rolling Gini Z-Score ({window}d)", "Z-Score", None, False),
    ]
    panels = tuple(
        _block18_extract_metric_panel(
            source, axis_name, title, yaxis_title, display_range_value,
            reference_level=reference_level, showlegend=showlegend,
        )
        for source, axis_name, title, yaxis_title, reference_level, showlegend in panel_specs
    )
    if len(block18_metric_panel_cache) >= 12:
        block18_metric_panel_cache.clear()
    block18_metric_panel_cache[cache_key] = tuple(go.Figure(figure) for figure in panels)
    return panels

@block18_dash_app.callback(
    Output("volatility-efficiency-plot-grid", "style"),
    Input("momentum-efficiency-view-tabs", "value"),
)
def _toggle_volatility_efficiency_grid(active_tab):
    if active_tab != "volatility_efficiency":
        return {"display": "none"}
    return {
        "display": "grid",
        "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
        "gridTemplateRows": "repeat(6, auto)",
        "gap": "14px",
        "marginTop": "12px",
    }


@block18_dash_app.callback(
    Output("risk-compounding-v2-risk-graph", "figure"),
    Output("risk-compounding-v2-risk-adjusted-passage-graph", "figure"),
    Output("risk-compounding-v2-spread-passage-graph", "figure"),
    Output("risk-compounding-v2-volatility-drag-passage-graph", "figure"),
    Input("momentum-efficiency-view-tabs", "value"),
    Input("shared-update-button", "n_clicks"),
    Input("shared-display-range-dropdown", "value"),
    Input("shared-ratio-dropdown", "value"),
    Input("benchmark-index-dropdown", "value"),
    Input("first-passage-start-sign-dropdown", "value"),
    Input("first-passage-start-condition-dropdown", "value"),
    Input("first-passage-threshold-input", "value"),
    Input("first-passage-end-sign-dropdown", "value"),
    Input("first-passage-end-condition-dropdown", "value"),
    Input("first-passage-end-threshold-input", "value"),
    State("shared-window-input", "value"),
)
def _update_risk_compounding_v2_view(
    active_tab, update_clicks, display_range_value, ratio_type,
    selected_benchmarks, first_passage_start_sign,
    first_passage_start_condition, first_passage_threshold,
    first_passage_end_sign, first_passage_end_condition,
    first_passage_end_threshold, window,
):
    if active_tab != "risk_compounding_v2":
        return no_update, no_update, no_update, no_update
    risk_figure, _ = _block18_render_dashboard_figure(
        "risk", window, display_range_value, selected_benchmarks,
        ratio_type=ratio_type, cache_version=update_clicks,
    )
    try:
        metric_specs = _block18_risk_first_passage_series(
            window, selected_benchmarks, ratio_type
        )
        integrated_risk_figure = _block18_overlay_first_passage_on_risk_figure(
            risk_figure, metric_specs,
            first_passage_threshold, first_passage_end_threshold,
            first_passage_start_condition, first_passage_end_condition,
            first_passage_start_sign, first_passage_end_sign,
        )
        analytics_figures = []
        for metric_spec in metric_specs:
            analytics_figure = _block18_first_passage_analytics_figure(
                metric_spec["series"], metric_spec["title"],
                first_passage_threshold, first_passage_end_threshold,
                first_passage_start_condition, first_passage_end_condition,
                first_passage_start_sign, first_passage_end_sign,
            )
            analytics_figure = _block18_apply_display_range(
                analytics_figure, display_range_value
            )
            if hasattr(analytics_figure.layout, "xaxis2"):
                analytics_figure.layout.xaxis2.update(autorange=True, range=None)
            analytics_figures.append(analytics_figure)
    except Exception as error:
        integrated_risk_figure = risk_figure
        analytics_figures = [
            _block18_apply_typography(
                _block18_error_figure(
                    f"Integrated first-passage analysis could not render: {error}",
                    height=610,
                )
            )
            for _ in range(3)
        ]
    return integrated_risk_figure, *analytics_figures


@block18_dash_app.callback(
    *[Output(f"volatility-efficiency-{metric_key}-graph", "figure") for metric_key, _, _ in block18_metric_graph_specs],
    Input("momentum-efficiency-view-tabs", "value"),
    Input("shared-update-button", "n_clicks"),
    State("shared-display-range-dropdown", "value"),
    State("shared-window-input", "value"),
)
def _update_volatility_efficiency_panels(active_tab, update_clicks, display_range_value, window):
    if active_tab != "volatility_efficiency":
        return tuple([no_update] * len(block18_metric_graph_specs))
    try:
        panels = _block18_build_metric_panels(window, display_range_value, update_clicks)
    except Exception as error:
        panels = tuple(
            _block18_apply_typography(
                _block18_error_figure(f"{title} could not render: {error}", height=430)
            )
            for _, title, _ in block18_metric_graph_specs
        )
    return panels

@block18_dash_app.callback(
    Output("diagnostics-playback-interval", "interval"),
    Output("momentum-efficiency-tab-graph", "animation_options"),
    Input("diagnostics-playback-speed-dropdown", "value"),
)
def _update_diagnostics_playback_speed(speed_milliseconds):
    selected_speed = speed_milliseconds
    if selected_speed == "continuous":
        interval = block18_continuous_playback_interval
    else:
        try:
            interval = max(100, int(selected_speed))
        except (TypeError, ValueError):
            selected_speed = block18_default_playback_speed
            interval = block18_default_playback_interval
    return interval, _block18_animation_options(selected_speed)

@block18_dash_app.callback(
    Output("diagnostics-playback-playing", "data"),
    Output("diagnostics-playback-interval", "disabled"),
    Output("diagnostics-playback-button", "children"),
    Output("diagnostics-asof-slider", "value"),
    Output("diagnostics-playback-controls", "style"),
    Input("diagnostics-playback-button", "n_clicks"),
    Input("diagnostics-playback-interval", "n_intervals"),
    Input("diagnostics-step-back-button", "n_clicks"),
    Input("diagnostics-step-forward-button", "n_clicks"),
    Input("diagnostics-date-picker", "date"),
    Input("momentum-efficiency-view-tabs", "value"),
    Input("benchmark-index-dropdown", "value"),
    Input("shared-ratio-dropdown", "value"),
    Input("diagnostics-horizon-min-input", "value"),
    Input("diagnostics-horizon-max-input", "value"),
    Input("diagnostics-spline-toggle", "value"),
    Input("diagnostics-raw-overlay-toggle", "value"),
    Input("diagnostics-spline-strength-slider", "value"),
    State("diagnostics-playback-playing", "data"),
    State("diagnostics-asof-slider", "value"),
)
def _update_diagnostics_playback(
    _n_clicks, _n_intervals, _step_back_clicks, _step_forward_clicks,
    selected_date, active_tab, _selected_benchmarks, _ratio_type,
    _window_min, _window_max,
    _spline_toggle, _raw_overlay_toggle, _spline_strength, playing, position
):
    if ctx.triggered_id in {
        "benchmark-index-dropdown",
        "shared-ratio-dropdown",
        "diagnostics-horizon-min-input",
        "diagnostics-horizon-max-input",
        "diagnostics-spline-toggle",
        "diagnostics-raw-overlay-toggle",
        "diagnostics-spline-strength-slider",
        "diagnostics-date-picker",
    }:
        playing = False
    if ctx.triggered_id == "diagnostics-date-picker":
        position = _block18_playback_position_for_date(selected_date)
    next_playing, interval_disabled, button_label, next_position = _block18_next_playback_state(
        ctx.triggered_id, playing, position, active_tab
    )
    slider_value = (
        next_position
        if ctx.triggered_id in {
            "diagnostics-playback-button",
            "diagnostics-playback-interval",
            "diagnostics-step-back-button",
            "diagnostics-step-forward-button",
            "diagnostics-date-picker",
        }
        else no_update
    )
    controls_style = {
        **block18_playback_controls_style,
        "display": "flex" if active_tab == "window_diagnostics" else "none",
    }
    return next_playing, interval_disabled, button_label, slider_value, controls_style

@block18_dash_app.callback(
    Output("momentum-efficiency-tab-graph", "figure", allow_duplicate=True),
    Input("diagnostics-asof-slider", "value"),
    State("momentum-efficiency-view-tabs", "value"),
    State("benchmark-index-dropdown", "value"),
    State("diagnostics-horizon-min-input", "value"),
    State("diagnostics-horizon-max-input", "value"),
    State("diagnostics-spline-toggle", "value"),
    State("diagnostics-raw-overlay-toggle", "value"),
    State("diagnostics-spline-strength-slider", "value"),
    State("shared-update-button", "n_clicks"),
    State("shared-ratio-dropdown", "value"),
    prevent_initial_call=True,
)
def _patch_functional_horizon_playback_frame(
    playback_position, active_tab, selected_benchmarks,
    diagnostics_horizon_min, diagnostics_horizon_max,
    spline_toggle, raw_overlay_toggle, spline_strength,
    update_clicks, ratio_type,
):
    if active_tab != "window_diagnostics":
        return no_update
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    selected_benchmarks = _block18_normalize_benchmark_selection(selected_benchmarks)
    diagnostics_horizon_range = _block18_horizon_range_from_inputs(
        diagnostics_horizon_min, diagnostics_horizon_max
    )
    signature = _block18_functional_playback_signature(
        selected_benchmarks,
        diagnostics_horizon_range,
        spline_toggle,
        spline_strength,
        raw_overlay_toggle,
        update_clicks,
        ratio_type=ratio_type,
    )
    patched_figure, _ = _block18_patch_functional_horizon_playback(
        signature,
        playback_position,
        selected_benchmarks,
        diagnostics_horizon_range,
        spline_toggle,
        spline_strength,
        ratio_type=ratio_type,
    )
    return patched_figure if patched_figure is not None else no_update


@block18_dash_app.callback(
    Output("momentum-efficiency-tab-graph", "figure"),
    Output("momentum-efficiency-tab-graph", "style"),
    Output("shared-view-status", "children"),
    Output("momentum-efficiency-tab-graph", "animate"),
    Input("momentum-efficiency-view-tabs", "value"),
    Input("shared-update-button", "n_clicks"),
    Input("shared-display-range-dropdown", "value"),
    Input("shared-ratio-dropdown", "value"),
    Input("benchmark-index-dropdown", "value"),
    Input("historical-horizon-min-input", "value"),
    Input("historical-horizon-max-input", "value"),
    Input("diagnostics-horizon-min-input", "value"),
    Input("diagnostics-horizon-max-input", "value"),
    Input("diagnostics-spline-toggle", "value"),
    Input("diagnostics-raw-overlay-toggle", "value"),
    Input("diagnostics-spline-strength-slider", "value"),
    Input("autocorrelation-lag-range-slider", "value"),
    Input("autocorrelation-return-horizon-input", "value"),
    Input("autocorrelation-sampling-mode", "value"),
    Input("autocorrelation-comparison-lag-input", "value"),
    Input("first-passage-start-sign-dropdown", "value"),
    Input("first-passage-start-condition-dropdown", "value"),
    Input("first-passage-threshold-input", "value"),
    Input("first-passage-end-sign-dropdown", "value"),
    Input("first-passage-end-condition-dropdown", "value"),
    Input("first-passage-end-threshold-input", "value"),
    State("shared-window-input", "value"),
    State("diagnostics-asof-slider", "value"),
    running=[
        (Output("shared-update-button", "disabled"), True, False),
        (Output("shared-update-button", "children"), "Updating ...", "Update"),
        (Output("shared-update-button", "style"), block18_update_button_loading_style, block18_update_button_style),
    ],
)
def _update_momentum_efficiency_dashboard(
    active_tab, _n_clicks, display_range_value, ratio_type,
    selected_benchmarks,
    historical_horizon_min, historical_horizon_max,
    diagnostics_horizon_min, diagnostics_horizon_max,
    spline_toggle, raw_overlay_toggle, spline_strength,
    autocorrelation_lag_range, autocorrelation_return_horizon,
    autocorrelation_sampling_mode, autocorrelation_comparison_lag,
    first_passage_start_sign, first_passage_start_condition, first_passage_threshold,
    first_passage_end_sign, first_passage_end_condition,
    first_passage_end_threshold, window,
    playback_position,
):
    active_tab = active_tab if active_tab in block18_tab_config else block18_default_tab
    ratio_type = _block18_normalize_ratio_type(ratio_type)
    selected_benchmarks = _block18_normalize_benchmark_selection(selected_benchmarks)
    if active_tab == "risk_compounding_v2":
        try:
            combined_window = _block18_validate_window(window)
            window_label = f"{combined_window:,} trading days"
        except (TypeError, ValueError):
            window_label = str(window)
        status = (
            f"Risk & Compounding v.2 | Window: {window_label} | "
            f"Display: {_block18_display_range_label(display_range_value)} | "
            f"Ratio: {_block18_ratio_label(ratio_type)} | Benchmarks: "
            f"{_block18_benchmark_selection_label(selected_benchmarks)}"
        )
        return no_update, {"display": "none"}, status, False
    if active_tab == "volatility_efficiency":
        status = f"Volatility, Efficiency & Distribution | Window: {window} trading days | Updating individual panels"
        return no_update, {"display": "none"}, status, False
    historical_horizon_range = _block18_horizon_range_from_inputs(
        historical_horizon_min, historical_horizon_max
    )
    diagnostics_horizon_range = _block18_horizon_range_from_inputs(
        diagnostics_horizon_min, diagnostics_horizon_max
    )
    selected_horizon_range = (
        historical_horizon_range
        if active_tab == "historical_surface_3d"
        else diagnostics_horizon_range
    )
    functional_signature = None
    if active_tab == "window_diagnostics":
        functional_signature = _block18_functional_playback_signature(
            selected_benchmarks, diagnostics_horizon_range,
            spline_toggle, spline_strength, raw_overlay_toggle,
            _n_clicks,
            ratio_type=ratio_type,
        )
    figure, status = _block18_render_dashboard_figure(
        active_tab,
        window,
        display_range_value,
        selected_benchmarks,
        ratio_type=ratio_type,
        playback_position=playback_position,
        diagnostics_window_range=selected_horizon_range,
        spline_toggle=spline_toggle,
        spline_strength=spline_strength,
        raw_overlay_toggle=raw_overlay_toggle,
        autocorrelation_lag_range=autocorrelation_lag_range,
        autocorrelation_return_horizon=autocorrelation_return_horizon,
        autocorrelation_sampling_mode=autocorrelation_sampling_mode,
        autocorrelation_comparison_lag=autocorrelation_comparison_lag,
        first_passage_threshold=first_passage_threshold,
        first_passage_end_threshold=first_passage_end_threshold,
        first_passage_start_condition=first_passage_start_condition,
        first_passage_end_condition=first_passage_end_condition,
        first_passage_start_sign=first_passage_start_sign,
        first_passage_end_sign=first_passage_end_sign,
        cache_version=_n_clicks,
    )
    if active_tab == "window_diagnostics" and functional_signature is not None:
        _block18_register_functional_playback_plan(
            figure, functional_signature
        )
    default_height = block18_tab_config.get(active_tab, {}).get("height", 1125)
    return (
        figure,
        {"display": "block", "height": _block18_graph_height(figure, default_height)},
        status,
        False,  # Full tab/control updates must use Plotly.react, not animation.
    )

block18_dash_app.run(
    host="127.0.0.1",
    port=_block18_available_port(),
    debug=False,
    use_reloader=False,
    jupyter_mode="inline",
    jupyter_height=4400,
)

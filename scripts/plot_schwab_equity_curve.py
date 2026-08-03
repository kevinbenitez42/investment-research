"""Build a Schwab transaction-derived realized P/L equity curve.

This plots cumulative realized trading P/L plus dividends/interest from raw
Schwab transaction JSON files produced by ``fetch_schwab_history.py``.

The Schwab transaction endpoint does not provide historical net liquidation
snapshots, so this is not a mark-to-market account value curve. Open positions
are not marked to market.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RETURN_CASH_TYPES = {"DIVIDEND_OR_INTEREST"}
EXTERNAL_CASH_TYPES = {
    "CASH_RECEIPT",
    "CASH_DISBURSEMENT",
    "ELECTRONIC_FUND",
    "JOURNAL",
}
POSITION_TYPES = {"TRADE", "RECEIVE_AND_DELIVER"}
IGNORED_TYPES = {"SMA_ADJUSTMENT"}


@dataclass
class Lot:
    quantity: float
    cash_flow: float


def _normalized_security_key(instrument: dict[str, Any]) -> str:
    asset_type = str(instrument.get("assetType") or "UNKNOWN").upper()
    symbol = str(
        instrument.get("uniformSymbol")
        or instrument.get("symbol")
        or instrument.get("instrumentId")
        or "UNKNOWN"
    ).strip()
    if asset_type == "OPTION":
        symbol = "".join(symbol.split())
    else:
        symbol = symbol.upper()
    return f"{asset_type}:{symbol}"


def _security_label(instrument: dict[str, Any]) -> str:
    asset_type = str(instrument.get("assetType") or "UNKNOWN").upper()
    symbol = str(instrument.get("uniformSymbol") or instrument.get("symbol") or "UNKNOWN").strip()
    symbol = " ".join(symbol.split()) if asset_type == "OPTION" else symbol.upper()
    return f"{symbol} ({asset_type})"


def _security_group(instrument: dict[str, Any]) -> str:
    asset_type = str(instrument.get("assetType") or "UNKNOWN").upper()
    if asset_type == "OPTION":
        underlying = instrument.get("underlyingSymbol")
        if underlying:
            return str(underlying).upper()
        symbol = str(instrument.get("uniformSymbol") or instrument.get("symbol") or "").strip()
        compact = "".join(symbol.split())
        letters = []
        for char in compact:
            if char.isalpha():
                letters.append(char)
            else:
                break
        return "".join(letters) or "OPTION"
    return str(instrument.get("symbol") or instrument.get("uniformSymbol") or "UNKNOWN").upper()


def _unit_mark_from_item(item: dict[str, Any]) -> float | None:
    instrument = item.get("instrument") or {}
    amount = float(item.get("amount") or 0.0)
    price = item.get("price")
    cost = item.get("cost")
    if price is not None:
        price_value = abs(float(price))
        if instrument.get("assetType") == "OPTION":
            multiplier = float(instrument.get("optionPremiumMultiplier") or 100.0)
            return price_value * multiplier
        return price_value
    if amount and cost is not None:
        return abs(float(cost) / amount)
    return None


def _current_balance_value(account_payload: dict[str, Any], key: str) -> float | None:
    securities_account = account_payload.get("securitiesAccount", {}) if isinstance(account_payload, dict) else {}
    balances = securities_account.get("currentBalances", {}) if isinstance(securities_account, dict) else {}
    value = balances.get(key) if isinstance(balances, dict) else None
    return float(value) if value is not None else None


def current_liquidation_value(account_payload: dict[str, Any]) -> float | None:
    return _current_balance_value(account_payload, "liquidationValue") or _current_balance_value(
        account_payload,
        "accountValue",
    )


def current_cash_balance(account_payload: dict[str, Any]) -> float | None:
    return _current_balance_value(account_payload, "cashBalance")


def _latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No Schwab API raw run directories found under {root}")
    return max(candidates, key=lambda path: path.name)


def _parse_datetime(record: dict[str, Any]) -> pd.Timestamp:
    value = record.get("time") or record.get("tradeDate") or record.get("settlementDate")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp


def _instrument_key(instrument: dict[str, Any]) -> str:
    return str(
        instrument.get("uniformSymbol")
        or instrument.get("symbol")
        or instrument.get("instrumentId")
        or "UNKNOWN"
    )


def _instrument_label(instrument: dict[str, Any]) -> str:
    symbol = _instrument_key(instrument)
    asset_type = instrument.get("assetType") or "UNKNOWN"
    return f"{symbol} ({asset_type})"


def _signed_quantity(record_type: str, item: dict[str, Any]) -> float:
    amount = float(item.get("amount") or 0.0)
    return amount


def _security_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for item in record.get("transferItems") or []:
        if not isinstance(item, dict):
            continue
        instrument = item.get("instrument") or {}
        if instrument.get("assetType") != "CURRENCY":
            items.append(item)
    return items


def _fee_total(record: dict[str, Any]) -> float:
    total = 0.0
    for item in record.get("transferItems") or []:
        if not isinstance(item, dict):
            continue
        instrument = item.get("instrument") or {}
        if instrument.get("assetType") == "CURRENCY" and item.get("feeType"):
            total += float(item.get("cost") or 0.0)
    return total


def _allocated_cash_flows(record: dict[str, Any]) -> list[tuple[dict[str, Any], float]]:
    items = _security_items(record)
    if not items:
        return []

    fee_total = _fee_total(record)
    weights = [abs(float(item.get("cost") or 0.0)) for item in items]
    denominator = sum(weights)

    allocated = []
    for item, weight in zip(items, weights):
        security_cash = float(item.get("cost") or 0.0)
        if fee_total and denominator:
            fee = fee_total * (weight / denominator)
        elif fee_total:
            fee = fee_total / len(items)
        else:
            fee = 0.0
        allocated.append((item, security_cash + fee))
    return allocated


def _realize_against_lots(
    *,
    lots: deque[Lot],
    quantity: float,
    cash_flow: float,
) -> tuple[float, float, float]:
    realized = 0.0
    unmatched_quantity = 0.0
    remaining_quantity = quantity
    remaining_cash = cash_flow

    while abs(remaining_quantity) > 1e-9 and lots:
        lot = lots[0]
        if lot.quantity == 0 or lot.quantity * remaining_quantity >= 0:
            break

        close_quantity = min(abs(remaining_quantity), abs(lot.quantity))
        lot_fraction = close_quantity / abs(lot.quantity)
        trade_fraction = close_quantity / abs(remaining_quantity)

        lot_cash = lot.cash_flow * lot_fraction
        trade_cash = remaining_cash * trade_fraction
        realized += lot_cash + trade_cash

        lot.quantity -= (1 if lot.quantity > 0 else -1) * close_quantity
        lot.cash_flow -= lot_cash
        remaining_quantity -= (1 if remaining_quantity > 0 else -1) * close_quantity
        remaining_cash -= trade_cash

        if abs(lot.quantity) <= 1e-9:
            lots.popleft()

    if abs(remaining_quantity) > 1e-9 and not lots:
        unmatched_quantity = remaining_quantity

    return realized, remaining_quantity, remaining_cash if abs(remaining_quantity) > 1e-9 else 0.0


def build_realized_curve(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_records = sorted(records, key=lambda record: _parse_datetime(record))
    lots_by_instrument: dict[str, deque[Lot]] = defaultdict(deque)
    event_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for record in sorted_records:
        if not isinstance(record, dict):
            continue

        timestamp = _parse_datetime(record)
        if pd.isna(timestamp):
            continue
        record_type = record.get("type")
        date = timestamp.date()

        if record_type in RETURN_CASH_TYPES:
            amount = float(record.get("netAmount") or 0.0)
            if amount:
                event_rows.append(
                    {
                        "date": date,
                        "timestamp": timestamp.isoformat(),
                        "type": record_type,
                        "component": "income",
                        "instrument": "Cash income",
                        "realized_pnl": amount,
                        "activity_id": record.get("activityId"),
                    }
                )
            continue

        if record_type in EXTERNAL_CASH_TYPES or record_type in IGNORED_TYPES:
            continue

        if record_type not in POSITION_TYPES:
            continue

        for item, cash_flow in _allocated_cash_flows(record):
            instrument = item.get("instrument") or {}
            instrument_key = _instrument_key(instrument)
            label = _instrument_label(instrument)
            quantity = _signed_quantity(str(record_type), item)
            if abs(quantity) <= 1e-9 and abs(cash_flow) <= 1e-9:
                continue

            lots = lots_by_instrument[instrument_key]
            realized, remaining_quantity, remaining_cash = _realize_against_lots(
                lots=lots,
                quantity=quantity,
                cash_flow=cash_flow,
            )

            if abs(realized) > 1e-9:
                event_rows.append(
                    {
                        "date": date,
                        "timestamp": timestamp.isoformat(),
                        "type": record_type,
                        "component": "realized_trade",
                        "instrument": label,
                        "realized_pnl": realized,
                        "activity_id": record.get("activityId"),
                    }
                )

            if abs(remaining_quantity) > 1e-9:
                position_effect = item.get("positionEffect")
                if position_effect == "CLOSING":
                    unmatched_rows.append(
                        {
                            "date": date,
                            "timestamp": timestamp.isoformat(),
                            "type": record_type,
                            "instrument": label,
                            "quantity": remaining_quantity,
                            "cash_flow": remaining_cash,
                            "activity_id": record.get("activityId"),
                        }
                    )
                else:
                    lots.append(Lot(quantity=remaining_quantity, cash_flow=remaining_cash))

    events = pd.DataFrame(event_rows)
    if events.empty:
        daily = pd.DataFrame(columns=["realized_trade_pnl", "income", "daily_pnl", "cumulative_pnl"])
    else:
        daily = (
            events.pivot_table(
                index="date",
                columns="component",
                values="realized_pnl",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
            .rename_axis(columns=None)
        )
        for column in ("realized_trade", "income"):
            if column not in daily.columns:
                daily[column] = 0.0
        daily = daily.rename(columns={"realized_trade": "realized_trade_pnl"})
        daily["daily_pnl"] = daily["realized_trade_pnl"] + daily["income"]
        daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    unmatched = pd.DataFrame(unmatched_rows)
    return daily, events, unmatched


def build_external_cash_flow_daily(records: list[dict[str, Any]]) -> pd.Series:
    """Aggregate external deposits, withdrawals, and transfers by day."""
    cash_changes = defaultdict(float)
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("type") not in EXTERNAL_CASH_TYPES:
            continue
        timestamp = _parse_datetime(record)
        if pd.isna(timestamp):
            continue
        cash_changes[timestamp.date()] += float(record.get("netAmount") or 0.0)
    return pd.Series(cash_changes, dtype=float).sort_index()


def add_realized_equity_proxy(
    daily: pd.DataFrame,
    ending_equity: float,
    *,
    equity_column: str = "realized_equity_proxy",
) -> pd.DataFrame:
    """Anchor cumulative realized P/L to a known ending account value.

    This produces a value-like series that ends at ``ending_equity``. It is a
    proxy, not true mark-to-market historical net liquidation value.
    """
    daily = daily.copy()
    ending_pnl = float(daily["cumulative_pnl"].iloc[-1])
    daily[equity_column] = float(ending_equity) - ending_pnl + daily["cumulative_pnl"]
    return daily


def add_external_cash_equity_proxy(
    daily: pd.DataFrame,
    external_cash_flow: pd.Series,
    ending_equity: float,
    *,
    equity_column: str = "market_value_with_external_cash",
    cash_flow_column: str = "external_cash_flow",
) -> pd.DataFrame:
    """Add an ending-anchored value proxy that includes external cash activity."""
    daily = daily.copy()
    external_cash_flow = pd.Series(external_cash_flow, dtype=float).sort_index()
    combined_index = daily.index.union(external_cash_flow.index).sort_values()
    daily = daily.reindex(combined_index)

    for column in ("realized_trade_pnl", "income", "daily_pnl"):
        if column not in daily.columns:
            daily[column] = 0.0
        daily[column] = daily[column].fillna(0.0)

    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
    ending_realized_pnl = float(daily["cumulative_pnl"].iloc[-1])
    for existing_equity_column in ("realized_equity_proxy", "market_value"):
        if existing_equity_column in daily.columns:
            daily[existing_equity_column] = (
                float(ending_equity)
                - ending_realized_pnl
                + daily["cumulative_pnl"]
            )

    daily[cash_flow_column] = external_cash_flow.reindex(combined_index).fillna(0.0)
    daily["daily_pnl_with_external_cash"] = daily["daily_pnl"] + daily[cash_flow_column]
    daily["cumulative_pnl_with_external_cash"] = daily["daily_pnl_with_external_cash"].cumsum()
    ending_change = float(daily["cumulative_pnl_with_external_cash"].iloc[-1])
    daily[equity_column] = (
        float(ending_equity)
        - ending_change
        + daily["cumulative_pnl_with_external_cash"]
    )
    return daily


def rolling_sharpe_from_value(
    value: pd.Series,
    *,
    window: int = 200,
    external_cash_flow: pd.Series | None = None,
    annualization_factor: int = 252,
) -> pd.Series:
    """Calculate a rolling Sharpe ratio from a value series.

    When external cash flow is supplied, deposits are subtracted from value
    changes and withdrawals are added back before returns are calculated.
    """
    values = pd.Series(value, dtype=float).dropna().sort_index()
    values.index = pd.to_datetime(values.index).normalize()
    values = values.groupby(level=0).last()

    cash_flow = pd.Series(0.0, index=values.index)
    if external_cash_flow is not None:
        cash_flow = pd.Series(external_cash_flow, dtype=float).sort_index()
        cash_flow.index = pd.to_datetime(cash_flow.index).normalize()
        cash_flow = cash_flow.groupby(level=0).sum().reindex(values.index).fillna(0.0)

    adjusted_change = values.diff() - cash_flow
    starting_value = values.shift(1).abs()
    starting_value = starting_value.mask(starting_value == 0.0)
    returns = (adjusted_change / starting_value).dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=0)
    rolling_std = rolling_std.mask(rolling_std == 0.0)
    return (rolling_mean / rolling_std * (annualization_factor ** 0.5)).dropna()


def rolling_sortino_from_value(
    value: pd.Series,
    *,
    window: int = 200,
    external_cash_flow: pd.Series | None = None,
    annualization_factor: int = 252,
    target_return: float = 0.0,
) -> pd.Series:
    """Calculate a rolling Sortino ratio from a value series."""
    values = pd.Series(value, dtype=float).dropna().sort_index()
    values.index = pd.to_datetime(values.index).normalize()
    values = values.groupby(level=0).last()

    cash_flow = pd.Series(0.0, index=values.index)
    if external_cash_flow is not None:
        cash_flow = pd.Series(external_cash_flow, dtype=float).sort_index()
        cash_flow.index = pd.to_datetime(cash_flow.index).normalize()
        cash_flow = cash_flow.groupby(level=0).sum().reindex(values.index).fillna(0.0)

    adjusted_change = values.diff() - cash_flow
    starting_value = values.shift(1).abs()
    starting_value = starting_value.mask(starting_value == 0.0)
    returns = (adjusted_change / starting_value).dropna()
    excess_returns = returns - float(target_return)
    downside_returns = excess_returns.clip(upper=0.0)
    rolling_mean = excess_returns.rolling(window).mean()
    downside_deviation = downside_returns.pow(2).rolling(window).mean().pow(0.5)
    downside_deviation = downside_deviation.mask(downside_deviation == 0.0)
    return (rolling_mean / downside_deviation * (annualization_factor ** 0.5)).dropna()


def expanding_sharpe_from_value(
    value: pd.Series,
    *,
    external_cash_flow: pd.Series | None = None,
    annualization_factor: int = 252,
    min_periods: int = 2,
) -> pd.Series:
    """Calculate an inception-to-date Sharpe ratio at each date."""
    values = pd.Series(value, dtype=float).dropna().sort_index()
    values.index = pd.to_datetime(values.index).normalize()
    values = values.groupby(level=0).last()

    cash_flow = pd.Series(0.0, index=values.index)
    if external_cash_flow is not None:
        cash_flow = pd.Series(external_cash_flow, dtype=float).sort_index()
        cash_flow.index = pd.to_datetime(cash_flow.index).normalize()
        cash_flow = cash_flow.groupby(level=0).sum().reindex(values.index).fillna(0.0)

    adjusted_change = values.diff() - cash_flow
    starting_value = values.shift(1).abs()
    starting_value = starting_value.mask(starting_value == 0.0)
    returns = (adjusted_change / starting_value).dropna()
    expanding_mean = returns.expanding(min_periods=min_periods).mean()
    expanding_std = returns.expanding(min_periods=min_periods).std(ddof=0)
    expanding_std = expanding_std.mask(expanding_std == 0.0)
    return (expanding_mean / expanding_std * (annualization_factor ** 0.5)).dropna()


def expanding_sortino_from_value(
    value: pd.Series,
    *,
    external_cash_flow: pd.Series | None = None,
    annualization_factor: int = 252,
    target_return: float = 0.0,
    min_periods: int = 2,
) -> pd.Series:
    """Calculate an inception-to-date Sortino ratio at each date."""
    values = pd.Series(value, dtype=float).dropna().sort_index()
    values.index = pd.to_datetime(values.index).normalize()
    values = values.groupby(level=0).last()

    cash_flow = pd.Series(0.0, index=values.index)
    if external_cash_flow is not None:
        cash_flow = pd.Series(external_cash_flow, dtype=float).sort_index()
        cash_flow.index = pd.to_datetime(cash_flow.index).normalize()
        cash_flow = cash_flow.groupby(level=0).sum().reindex(values.index).fillna(0.0)

    adjusted_change = values.diff() - cash_flow
    starting_value = values.shift(1).abs()
    starting_value = starting_value.mask(starting_value == 0.0)
    returns = (adjusted_change / starting_value).dropna()
    excess_returns = returns - float(target_return)
    downside_returns = excess_returns.clip(upper=0.0)
    expanding_mean = excess_returns.expanding(min_periods=min_periods).mean()
    downside_deviation = downside_returns.pow(2).expanding(min_periods=min_periods).mean().pow(0.5)
    downside_deviation = downside_deviation.mask(downside_deviation == 0.0)
    return (expanding_mean / downside_deviation * (annualization_factor ** 0.5)).dropna()


def make_equity_curve_figure(daily: pd.DataFrame, *, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=("Cumulative Realized P/L", "Daily Realized P/L"),
    )

    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["cumulative_pnl"],
            mode="lines",
            name="Cumulative realized P/L",
            line=dict(color="#22c55e", width=2.5),
            hovertemplate="%{x}<br>Cumulative P/L: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if "realized_equity_proxy" in daily.columns:
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily["realized_equity_proxy"],
                mode="lines",
                name="Realized equity proxy",
                line=dict(color="#38bdf8", width=2.5),
                hovertemplate="%{x}<br>Equity proxy: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    bar_colors = ["#22c55e" if value >= 0 else "#ef4444" for value in daily["daily_pnl"]]
    fig.add_trace(
        go.Bar(
            x=daily.index,
            y=daily["daily_pnl"],
            name="Daily realized P/L",
            marker_color=bar_colors,
            hovertemplate="%{x}<br>Daily P/L: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=820,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=80, r=40, t=90, b=60),
    )
    fig.update_yaxes(title_text="P/L ($)", tickprefix="$", separatethousands=True, row=1, col=1)
    fig.update_yaxes(title_text="Daily ($)", tickprefix="$", separatethousands=True, row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def make_market_value_proxy_figure(
    daily: pd.DataFrame,
    *,
    title: str,
    value_column: str = "market_value",
) -> go.Figure:
    """Plot a value-like curve anchored to current account market value."""
    if value_column not in daily.columns:
        raise ValueError(f"daily must include '{value_column}'.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=("Total Portfolio Market Value Proxy", "Daily Realized P/L"),
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily[value_column],
            mode="lines",
            name="Portfolio market value proxy",
            line=dict(color="#38bdf8", width=2.7),
            hovertemplate="%{x}<br>Value: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    bar_colors = ["#22c55e" if value >= 0 else "#ef4444" for value in daily["daily_pnl"]]
    fig.add_trace(
        go.Bar(
            x=daily.index,
            y=daily["daily_pnl"],
            name="Daily realized P/L",
            marker_color=bar_colors,
            hovertemplate="%{x}<br>Daily realized P/L: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=820,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=80, r=40, t=90, b=60),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", tickprefix="$", separatethousands=True, row=1, col=1)
    fig.update_yaxes(title_text="Daily P/L ($)", tickprefix="$", separatethousands=True, row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def build_position_equity_history(
    records: list[dict[str, Any]],
    current_positions: list[dict[str, Any]],
    account_payload: dict[str, Any],
    *,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    use_current_marks_for_open_positions: bool = False,
) -> dict[str, pd.DataFrame]:
    """Reconstruct historical cash and position market-value proxies.

    Position quantities are anchored to current Schwab positions, then transaction
    quantity changes are replayed from the earliest selected date. Option marks
    use observed transaction marks plus current Schwab market values. This avoids
    treating realized P/L as account value, while still working when full
    historical option EOD marks are unavailable.
    """
    transaction_rows = []
    mark_rows = []
    security_meta: dict[str, dict[str, str]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        timestamp = _parse_datetime(record)
        if pd.isna(timestamp):
            continue
        record_type = str(record.get("type") or "")
        if record_type not in POSITION_TYPES:
            continue
        for item in _security_items(record):
            instrument = item.get("instrument") or {}
            key = _normalized_security_key(instrument)
            security_meta.setdefault(
                key,
                {
                    "label": _security_label(instrument),
                    "group": _security_group(instrument),
                    "asset_type": str(instrument.get("assetType") or "UNKNOWN").upper(),
                    "symbol": str(instrument.get("symbol") or instrument.get("uniformSymbol") or ""),
                },
            )
            quantity = _signed_quantity(record_type, item)
            transaction_rows.append(
                {
                    "date": timestamp.normalize().tz_localize(None),
                    "timestamp": timestamp,
                    "key": key,
                    "quantity_change": quantity,
                }
            )
            unit_mark = _unit_mark_from_item(item)
            if unit_mark is not None:
                mark_rows.append(
                    {
                        "date": timestamp.normalize().tz_localize(None),
                        "timestamp": timestamp,
                        "key": key,
                        "unit_mark": unit_mark,
                    }
                )

    current_quantity = defaultdict(float)
    current_unit_marks = {}
    current_position_rows = []
    for position in current_positions or []:
        if not isinstance(position, dict):
            continue
        instrument = position.get("instrument") or {}
        key = _normalized_security_key(instrument)
        long_quantity = float(position.get("longQuantity") or 0.0)
        short_quantity = float(position.get("shortQuantity") or 0.0)
        quantity = long_quantity - short_quantity
        if abs(quantity) <= 1e-9:
            continue
        market_value = float(position.get("marketValue") or 0.0)
        current_quantity[key] += quantity
        current_unit_marks[key] = market_value / quantity
        security_meta.setdefault(
            key,
            {
                "label": _security_label(instrument),
                "group": _security_group(instrument),
                "asset_type": str(instrument.get("assetType") or "UNKNOWN").upper(),
                "symbol": str(instrument.get("symbol") or instrument.get("uniformSymbol") or ""),
            },
        )
        current_position_rows.append(
            {
                "key": key,
                "label": security_meta[key]["label"],
                "group": security_meta[key]["group"],
                "asset_type": security_meta[key]["asset_type"],
                "quantity": quantity,
                "market_value": market_value,
                "unit_mark": current_unit_marks[key],
            }
        )

    if not transaction_rows and not current_quantity:
        raise ValueError("No transaction or current-position security rows available.")

    transaction_frame = pd.DataFrame(transaction_rows)
    inferred_start = transaction_frame["date"].min() if not transaction_frame.empty else pd.Timestamp.today().normalize()
    inferred_end = max(
        transaction_frame["date"].max() if not transaction_frame.empty else pd.Timestamp.today().normalize(),
        pd.Timestamp.today().normalize(),
    )
    resolved_start = pd.Timestamp(start_date).normalize() if start_date is not None else inferred_start
    resolved_end = pd.Timestamp(end_date).normalize() if end_date is not None else inferred_end
    dates = pd.date_range(resolved_start, resolved_end, freq="D")

    all_keys = sorted(set(security_meta) | set(current_quantity))
    change_matrix = pd.DataFrame(0.0, index=dates, columns=all_keys)
    if not transaction_frame.empty:
        changes = (
            transaction_frame[
                (transaction_frame["date"] >= resolved_start)
                & (transaction_frame["date"] <= resolved_end)
            ]
            .pivot_table(
                index="date",
                columns="key",
                values="quantity_change",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(dates, fill_value=0.0)
        )
        change_matrix.loc[:, changes.columns] = changes

    total_changes = change_matrix.sum(axis=0)
    ending_quantity = pd.Series(current_quantity, dtype=float).reindex(all_keys).fillna(0.0)
    starting_quantity = ending_quantity - total_changes
    quantity_history = change_matrix.cumsum().add(starting_quantity, axis="columns")
    quantity_history = quantity_history.loc[:, quantity_history.abs().sum(axis=0) > 1e-9]

    mark_frame = pd.DataFrame(mark_rows)
    current_mark_date = dates[-1]
    for key, unit_mark in current_unit_marks.items():
        mark_frame = pd.concat(
            [
                mark_frame,
                pd.DataFrame(
                    [
                        {
                            "date": current_mark_date,
                            "timestamp": pd.Timestamp(current_mark_date, tz="UTC"),
                            "key": key,
                            "unit_mark": abs(float(unit_mark)),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    mark_history = pd.DataFrame(index=dates, columns=quantity_history.columns, dtype=float)
    if not mark_frame.empty:
        mark_frame = mark_frame[mark_frame["key"].isin(quantity_history.columns)].copy()
        mark_frame["date"] = pd.to_datetime(mark_frame["date"]).dt.normalize()
        mark_frame = mark_frame.sort_values(["key", "timestamp"])
        mark_frame = mark_frame.drop_duplicates(["date", "key"], keep="last")
        observed_marks = mark_frame.pivot(index="date", columns="key", values="unit_mark")
        mark_history.loc[:, observed_marks.columns] = observed_marks.reindex(dates)
    mark_history = mark_history.ffill().bfill()
    if use_current_marks_for_open_positions:
        for key, unit_mark in current_unit_marks.items():
            if key in mark_history.columns:
                mark_history[key] = abs(float(unit_mark))

    position_value = quantity_history * mark_history
    position_value = position_value.where(quantity_history.abs() > 1e-9, 0.0).fillna(0.0)

    labels = {key: security_meta.get(key, {}).get("label", key) for key in position_value.columns}
    groups = {key: security_meta.get(key, {}).get("group", key) for key in position_value.columns}
    position_value_by_position = position_value.rename(columns=labels)
    position_value_by_group = pd.DataFrame(index=position_value.index)
    for key, group in groups.items():
        position_value_by_group[group] = position_value_by_group.get(group, 0.0) + position_value[key]
    position_value_by_group = position_value_by_group.loc[:, position_value_by_group.abs().sum(axis=0) > 1e-9]

    cash_balance = current_cash_balance(account_payload)
    if cash_balance is None:
        liquidation = current_liquidation_value(account_payload)
        ending_position_value = float(position_value.sum(axis=1).iloc[-1])
        cash_balance = float(liquidation or 0.0) - ending_position_value

    cash_changes = defaultdict(float)
    for record in records:
        if not isinstance(record, dict):
            continue
        record_type = record.get("type")
        if record_type in IGNORED_TYPES:
            continue
        timestamp = _parse_datetime(record)
        if pd.isna(timestamp):
            continue
        date = timestamp.normalize().tz_localize(None)
        if resolved_start <= date <= resolved_end:
            cash_changes[date] += float(record.get("netAmount") or 0.0)
    cash_change_series = pd.Series(cash_changes, dtype=float).reindex(dates).fillna(0.0)
    starting_cash = float(cash_balance) - float(cash_change_series.sum())
    cash = starting_cash + cash_change_series.cumsum()

    totals = pd.DataFrame(index=dates)
    totals["cash"] = cash
    totals["position_market_value"] = position_value.sum(axis=1)
    totals["total_account_value"] = totals["cash"] + totals["position_market_value"]
    liquidation = current_liquidation_value(account_payload)
    if liquidation is not None:
        adjustment = float(liquidation) - float(totals["total_account_value"].iloc[-1])
        totals["cash"] = totals["cash"] + adjustment
        totals["total_account_value"] = totals["cash"] + totals["position_market_value"]

    metadata = pd.DataFrame(
        [
            {
                "key": key,
                **security_meta.get(key, {}),
            }
            for key in sorted(security_meta)
        ]
    )

    return {
        "totals": totals,
        "position_value_by_position": position_value_by_position,
        "position_value_by_group": position_value_by_group,
        "quantity_history": quantity_history,
        "mark_history": mark_history,
        "current_positions": pd.DataFrame(current_position_rows),
        "metadata": metadata,
    }


def make_position_equity_figure(
    totals: pd.DataFrame,
    position_value_by_group: pd.DataFrame,
    *,
    title: str,
    max_groups: int = 12,
) -> go.Figure:
    """Plot cash, position market value, total account value, and top groups."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.58, 0.42],
        subplot_titles=("Account Value Components", "Position Market Value by Underlying"),
    )
    fig.add_trace(
        go.Scatter(
            x=totals.index,
            y=totals["total_account_value"],
            mode="lines",
            name="Total account value",
            line=dict(color="#38bdf8", width=3),
            hovertemplate="%{x}<br>Total: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=totals.index,
            y=totals["cash"],
            mode="lines",
            name="Cash",
            line=dict(color="#facc15", width=2),
            hovertemplate="%{x}<br>Cash: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=totals.index,
            y=totals["position_market_value"],
            mode="lines",
            name="Positions ex-cash",
            line=dict(color="#f97316", width=2),
            hovertemplate="%{x}<br>Positions: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if not position_value_by_group.empty:
        group_order = (
            position_value_by_group.abs()
            .max(axis=0)
            .sort_values(ascending=False)
            .index.tolist()
        )
        top_groups = group_order[:max_groups]
        group_frame = position_value_by_group[top_groups].copy()
        if len(group_order) > max_groups:
            group_frame["Other"] = position_value_by_group[group_order[max_groups:]].sum(axis=1)

        for group in group_frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=group_frame.index,
                    y=group_frame[group],
                    mode="lines",
                    name=str(group),
                    hovertemplate=f"{group}<br>%{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=900,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=80, r=40, t=90, b=60),
    )
    fig.update_yaxes(title_text="Value ($)", tickprefix="$", separatethousands=True, row=1, col=1)
    fig.update_yaxes(title_text="Position Value ($)", tickprefix="$", separatethousands=True, row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def despike_isolated_cash_position(
    cash: pd.Series,
    *,
    window: int = 7,
    min_abs_spike: float = 5_000.0,
    scale_multiplier: float = 8.0,
    max_spike_span: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Remove short isolated cash spikes while preserving persistent cash shifts.

    The filter flags points that deviate from a centered rolling median by more
    than both an absolute threshold and local day-to-day noise. Only short
    flagged runs are removed, so real multi-day cash level changes stay intact.
    """
    series = pd.Series(cash, dtype=float).sort_index()
    if series.empty:
        return series.copy(), pd.Series(dtype=bool)

    rolling_median = series.rolling(window, center=True, min_periods=3).median()
    local_move = series.diff().abs().rolling(window, center=True, min_periods=3).median()
    threshold = pd.concat(
        [
            pd.Series(float(min_abs_spike), index=series.index),
            local_move.fillna(0.0).mul(float(scale_multiplier)),
        ],
        axis=1,
    ).max(axis=1)
    raw_mask = (series - rolling_median).abs().gt(threshold).fillna(False)

    spike_mask = raw_mask.copy()
    i = 0
    while i < len(raw_mask):
        if not bool(raw_mask.iloc[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(raw_mask) and bool(raw_mask.iloc[j + 1]):
            j += 1
        if j - i + 1 > int(max_spike_span):
            spike_mask.iloc[i : j + 1] = False
        i = j + 1

    cleaned = series.mask(spike_mask)
    cleaned = cleaned.interpolate(method="time", limit_direction="both")
    return cleaned, spike_mask


def make_cash_position_figure(
    cash: pd.Series,
    cleaned_cash: pd.Series,
    spike_mask: pd.Series,
    *,
    title: str,
    show_raw: bool = True,
    show_spikes: bool = False,
) -> go.Figure:
    """Plot raw and despiked Schwab cash position."""
    cash = pd.Series(cash, dtype=float).sort_index()
    cleaned_cash = pd.Series(cleaned_cash, dtype=float).reindex(cash.index)
    spike_mask = pd.Series(spike_mask, dtype=bool).reindex(cash.index).fillna(False)

    fig = go.Figure()
    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=cash.index,
                y=cash,
                mode="lines",
                name="Raw cash",
                line=dict(color="rgba(148, 163, 184, 0.45)", width=1.2, dash="dot"),
                hovertemplate="%{x}<br>Raw cash: $%{y:,.2f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=cleaned_cash.index,
            y=cleaned_cash,
            mode="lines",
            name="Cash position",
            line=dict(color="#38bdf8", width=2.8),
            hovertemplate="%{x}<br>Cash: $%{y:,.2f}<extra></extra>",
        )
    )
    if show_spikes and spike_mask.any():
        fig.add_trace(
            go.Scatter(
                x=cash.index[spike_mask],
                y=cash.loc[spike_mask],
                mode="markers",
                name="Removed isolated spikes",
                marker=dict(color="#ef4444", size=7, symbol="x"),
                hovertemplate="%{x}<br>Removed raw value: $%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=700,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=80, r=40, t=85, b=60),
    )
    fig.update_yaxes(title_text="Cash ($)", tickprefix="$", separatethousands=True)
    fig.update_xaxes(title_text="Date")
    return fig


def make_market_value_cash_stack_figure(
    market_value_daily: pd.DataFrame,
    cleaned_cash: pd.Series,
    *,
    title: str,
    value_column: str = "market_value",
    value_label: str = "Total portfolio value",
    value_hover_label: str = "Total value",
    sharpe_series: pd.Series | None = None,
    sharpe_label: str = "Rolling Sharpe",
    sortino_series: pd.Series | None = None,
    sortino_label: str = "Rolling Sortino",
) -> go.Figure:
    """Stack total portfolio value, cleaned cash, and optional risk ratios."""
    if value_column not in market_value_daily.columns:
        raise ValueError(f"market_value_daily must include '{value_column}'.")

    cleaned_cash = pd.Series(cleaned_cash, dtype=float).sort_index()
    has_sharpe = sharpe_series is not None and not pd.Series(sharpe_series).dropna().empty
    has_sortino = sortino_series is not None and not pd.Series(sortino_series).dropna().empty
    has_ratios = has_sharpe or has_sortino
    row_count = 3 if has_ratios else 2
    row_heights = [0.50, 0.25, 0.25] if has_ratios else [0.58, 0.42]
    subplot_titles = (
        ("Total Portfolio Market Value Proxy", "Cash Position", "Risk-Adjusted Ratios")
        if has_ratios
        else ("Total Portfolio Market Value Proxy", "Cash Position")
    )
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )
    fig.add_trace(
        go.Scatter(
            x=market_value_daily.index,
            y=market_value_daily[value_column],
            mode="lines",
            name=value_label,
            line=dict(color="#38bdf8", width=2.8),
            hovertemplate=f"%{{x}}<br>{value_hover_label}: $%{{y:,.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=cleaned_cash.index,
            y=cleaned_cash,
            mode="lines",
            name="Cash position",
            line=dict(color="#facc15", width=2.4),
            hovertemplate="%{x}<br>Cash: $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if has_sharpe:
        sharpe_series = pd.Series(sharpe_series, dtype=float).sort_index()
        fig.add_trace(
            go.Scatter(
                x=sharpe_series.index,
                y=sharpe_series,
                mode="lines",
                name=sharpe_label,
                line=dict(color="#22c55e", width=2.4),
                hovertemplate="%{x}<br>Sharpe: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    if has_sortino:
        sortino_series = pd.Series(sortino_series, dtype=float).sort_index()
        fig.add_trace(
            go.Scatter(
                x=sortino_series.index,
                y=sortino_series,
                mode="lines",
                name=sortino_label,
                line=dict(color="#a78bfa", width=2.4),
                hovertemplate="%{x}<br>Sortino: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    if has_ratios:
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.45)", width=1), row=3, col=1)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=980 if has_ratios else 820,
        paper_bgcolor="#05070b",
        plot_bgcolor="#05070b",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        margin=dict(l=80, r=40, t=90, b=60),
    )
    fig.update_yaxes(title_text="Value ($)", tickprefix="$", separatethousands=True, row=1, col=1)
    fig.update_yaxes(title_text="Cash ($)", tickprefix="$", separatethousands=True, row=2, col=1)
    if has_ratios:
        fig.update_yaxes(title_text="Ratio", tickformat=".2f", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
    else:
        fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Schwab raw run directory. Defaults to latest csv_files/schwab_api_raw/*.",
    )
    parser.add_argument("--account-label", default="account_1")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path("csv_files") / "schwab_api_raw"
    run_dir = args.run_dir or _latest_run_dir(raw_root)
    input_path = run_dir / f"{args.account_label}_combined_transactions.json"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records = json.loads(input_path.read_text(encoding="utf-8"))
    daily, events, unmatched = build_realized_curve(records)
    if daily.empty:
        raise RuntimeError("No realized P/L events were produced from the transaction file.")

    output = args.output or run_dir / f"{args.account_label}_realized_equity_curve.html"
    daily_path = output.with_name(output.stem + "_daily.csv")
    events_path = output.with_name(output.stem + "_events.csv")
    unmatched_path = output.with_name(output.stem + "_unmatched_closings.csv")

    write_dataframe(daily_path, daily)
    events.to_csv(events_path, index=False, quoting=csv.QUOTE_MINIMAL)
    unmatched.to_csv(unmatched_path, index=False, quoting=csv.QUOTE_MINIMAL)

    start = daily.index.min()
    end = daily.index.max()
    total = daily["cumulative_pnl"].iloc[-1]
    title = (
        f"Schwab Realized P/L Equity Curve "
        f"({start} to {end}, ending ${total:,.2f})"
    )
    fig = make_equity_curve_figure(daily, title=title)
    fig.write_html(output, include_plotlyjs="cdn")

    print(f"HTML={output}")
    print(f"DAILY_CSV={daily_path}")
    print(f"EVENTS_CSV={events_path}")
    print(f"UNMATCHED_CSV={unmatched_path}")
    print(f"DATE_RANGE={start} to {end}")
    print(f"DAYS={len(daily)}")
    print(f"EVENTS={len(events)}")
    print(f"UNMATCHED_CLOSINGS={len(unmatched)}")
    print(f"ENDING_CUMULATIVE_REALIZED_PL={total:.2f}")


if __name__ == "__main__":
    main()

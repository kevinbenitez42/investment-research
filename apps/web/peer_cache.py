"""Shared Wikipedia membership and daily peer-price caches for the dashboard.

Successful-check timestamps describe source retrieval, not official GICS currency.
Existing daily history is refetched when due: the market-history interface has
no incremental-date API, and adjusted prices can change after corporate actions.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
import pandas as pd

from Quantapp.data import GICSDataClient, build_gics_peer_frames, get_market_history
from Quantapp.data.sources.wikipedia import WIKIPEDIA_SP_MARKET_CAP_INDEX_URLS

_LOCK = RLock()
LEVELS = {'Sector': 'sector', 'Industry Group': 'industry_group',
          'Industry': 'industry', 'Sub-Industry': 'sub_industry'}


@contextmanager
def _file_lock(path, timeout=120):
    """OS-backed lock, released even when a notebook process exits."""
    with open(path, 'a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0')
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another notebook is refreshing peer data; retry shortly')
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        if isinstance(value, pd.DataFrame):
            value.to_csv(staging, index=False)
        else:
            staging.write_text(json.dumps(value, indent=2), encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _stamp(value):
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _fetch_prices(**kwargs):
    # This module owns freshness; bypass the provider's separate 12-hour cache.
    return get_market_history(**kwargs, cache_ttl_seconds=0)


def _validate_companies(frame):
    required = ['Symbol', 'Capitalization', 'Sector', 'Industry Group', 'Industry', 'Sub-Industry', 'GICS Code']
    if len(frame) < 1400 or not set(required).issubset(frame):
        raise ValueError("Incomplete Wikipedia company list")
    if frame[required].isna().any(axis=None) or frame.Symbol.duplicated().any():
        raise ValueError("Missing classifications or duplicate symbols in company list")
    if set(frame.Capitalization) != {'Large Cap', 'Mid Cap', 'Small Cap'}:
        raise ValueError("Missing Wikipedia capitalization table")


def load_companies(root, *, force=False, now=None, fetch=None):
    """Check Wikipedia weekly; retry failures at most hourly unless forced."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    path = root / 'company_data/gics_companies.csv'
    meta_path = path.with_suffix('.meta.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _file_lock(str(path) + '.lock'):
        meta = _read_json(meta_path)
        cached = None
        try:
            cached = pd.read_csv(path)
            _validate_companies(cached)
            if meta.get('fingerprint') != _fingerprint(cached.astype(str).to_dict('records')):
                meta.pop('last_success', None)
        except (OSError, ValueError):
            cached = None
        due = cached is None or now - _stamp(meta.get('last_success')) >= timedelta(days=7)
        retry = now - _stamp(meta.get('last_attempt')) >= timedelta(hours=1)
        if force or (due and retry):
            meta['last_attempt'] = now.isoformat()
            try:
                fresh = (fetch or GICSDataClient(save_path=root).retrieve_companies)()
                _validate_companies(fresh)
                # Canonical CSV roundtrip keeps hashes stable across integer dtypes.
                from io import StringIO
                fresh = pd.read_csv(StringIO(fresh.to_csv(index=False)))
                _save(path, fresh)
                meta.update(last_success=now.isoformat(), error=None,
                            sources=list(WIKIPEDIA_SP_MARKET_CAP_INDEX_URLS.values()),
                            fingerprint=_fingerprint(fresh.astype(str).to_dict('records')),
                            rows=len(fresh))
                cached = fresh
            except Exception as exc:
                meta['error'] = str(exc)
            _save(meta_path, meta)
        if cached is None:
            raise ValueError(f"No valid company list available: {meta.get('error', 'refresh pending')}")
        return cached, meta


def _read_prices(path):
    data = pd.read_csv(path)
    dates = pd.to_datetime(data['Date'], errors='coerce', utc=True).dt.tz_convert(None)
    close = pd.to_numeric(data['Close'], errors='coerce')
    frame = pd.DataFrame({'Close': close.to_numpy()}, index=pd.DatetimeIndex(dates, name='Date'))
    frame = frame.loc[~frame.index.isna() & np.isfinite(frame.Close) & frame.Close.gt(0)]
    return frame.loc[~frame.index.duplicated(keep='last')].sort_index()


def _load_price(root, symbol, period, interval, now, fetch):
    key = _fingerprint([symbol, period, interval, 'adjusted', 1])[:24]
    path = root / 'company_data/peer_prices' / f'{key}.csv'
    meta_path = path.with_suffix('.json')
    meta = _read_json(meta_path)
    try:
        frame = _read_prices(path)
    except (OSError, ValueError, KeyError):
        frame = pd.DataFrame()
    current = not frame.empty and _stamp(meta.get('last_success')).date() == now.date()
    retry = not meta.get('error') or now - _stamp(meta.get('last_attempt')) >= timedelta(hours=1)
    if not current and retry:
        meta['last_attempt'] = now.isoformat()
        try:
            history = fetch(symbols=[symbol], period=period, interval=interval,
                            provider='yfinance', align=False).get(symbol)
            if history is None or history.empty or 'Close' not in history:
                raise ValueError(f'No prices returned for {symbol}')
            history = history[['Close']].copy()
            history.index = pd.to_datetime(history.index, utc=True).tz_convert(None)
            history = history.loc[~history.index.duplicated(keep='last')].sort_index()
            history.Close = pd.to_numeric(history.Close, errors='coerce')
            history = history.loc[np.isfinite(history.Close) & history.Close.gt(0)]
            if len(history) < 2:
                raise ValueError(f'Insufficient prices for {symbol}')
            _save(path, history.rename_axis('Date').reset_index())
            frame = history
            meta.update(last_success=now.isoformat(), error=None, symbol=symbol,
                        source='yfinance', period=period, interval=interval,
                        price_start=str(frame.index.min().date()),
                        price_end=str(frame.index.max().date()))
        except Exception as exc:
            meta['error'] = str(exc)
        _save(meta_path, meta)
    return frame, meta


def refresh_peer_benchmark(root, target, *, period='20y', interval='1d',
                           force=False, now=None, fetch_companies=None, fetch_prices=None,
                           level='Sub-Industry'):
    """Return the current peer basket, a cached/rebuilt index, and audit metadata."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    target = str(target).strip().upper().replace('.', '-').replace('/', '-')
    if not re.fullmatch(r'[A-Z0-9^=-]+', target):
        raise ValueError('Invalid peer lookup symbol')
    path = root / 'company_data/factor_peer_indexes' / f'{target}_{LEVELS[level]}_index.csv'
    meta_path = path.with_suffix('.meta.json')
    with _LOCK:
        previous = _read_json(meta_path)
        company_meta = {}
        try:
            companies, company_meta = load_companies(root, force=force, now=now, fetch=fetch_companies)
            context = build_gics_peer_frames(target, companies=companies)
            symbols = sorted(set(context.symbols(level)) - {target})
            if len(symbols) < 2:
                raise ValueError(f'Only {len(symbols)} {level} peers found for {target}; at least two required')
            classification = {level: str(context.target_row[level]) for level in
                              ('Sector', 'Industry Group', 'Industry', 'Sub-Industry')}
            settings = dict(period=period, interval=interval, weighting='Equal Weight',
                            minimum_daily_constituents=2, version=1, level=level)
            membership = _fingerprint(dict(symbols=symbols, classification=classification, settings=settings))
            prices, checks, missing = {}, {}, []
            for symbol in symbols:
                frame, price_meta = _load_price(root, symbol, period, interval, now, fetch_prices or _fetch_prices)
                if frame.empty:
                    missing.append(symbol)
                    continue
                prices[symbol] = frame.Close
                checks[symbol] = price_meta
            if len(prices) < 2:
                raise ValueError(f'Fewer than two peers have usable prices; unavailable: {missing}')
            signature = _fingerprint([membership, {s: m.get('last_success') for s, m in checks.items()}])
            label = f"{classification[level]} - {level}"
            if previous.get('signature') != signature or not path.exists():
                panel = pd.concat(prices, axis=1).sort_index()
                returns = panel.pct_change(fill_method=None)
                basket = returns.mean(axis=1).where(returns.notna().sum(axis=1) >= 2).dropna()
                if basket.empty:
                    raise ValueError('Peer histories have no overlapping return dates')
                index = (100 * (1 + basket).cumprod()).rename('Close')
                earlier = panel.index[panel.index < index.index[0]]
                if len(earlier):
                    index = pd.concat([pd.Series([100.], index=earlier[-1:], name='Close'), index])
                export = index.to_frame()
                export['Target Symbol'] = target
                export['Benchmark Label'] = label
                export['GICS Level'] = level
                export['GICS Name'] = classification[level]
                export['Weighting'] = 'Equal Weight'
                export['Constituent Count'] = len(prices)
                _save(path, export.rename_axis('Date').reset_index())
            frame = _read_prices(path)
            errors = [f"{s}: {m['error']}" for s, m in checks.items() if m.get('error')]
            if missing:
                errors.append('Excluded peers without prices: ' + ', '.join(missing))
            if company_meta.get('error'):
                errors.insert(0, 'Wikipedia: ' + company_meta['error'])
            meta = dict(target=target, label=label, constituents=sorted(prices),
                        selected_constituents=symbols, excluded_constituents=missing, classification=classification,
                        settings=settings, membership_fingerprint=membership, signature=signature,
                        wikipedia_last_checked=company_meta.get('last_success'),
                        sources=company_meta.get('sources'), price_checks=checks,
                        price_start=str(frame.index.min().date()), price_end=str(frame.index.max().date()),
                        last_success=now.isoformat(), error='; '.join(errors) or None,
                        used_cache=previous.get('signature') == signature)
            _save(meta_path, meta)
            return frame, meta
        except Exception as exc:
            meta = dict(previous, error=str(exc), used_cache=True)
            meta['wikipedia_last_checked'] = company_meta.get('last_success', previous.get('wikipedia_last_checked'))
            # Never substitute a legacy index calculated for different settings.
            if previous.get('settings', {}).get('period') == period and previous.get('settings', {}).get('interval') == interval:
                try:
                    return _read_prices(path), meta
                except (OSError, ValueError, KeyError):
                    pass
            meta['unavailable'] = True
            return pd.DataFrame(), meta


def refresh_gics_benchmarks(root, target, *, period='20y', interval='1d', force=False,
                            now=None, fetch_companies=None, fetch_prices=None):
    """Refresh all four levels, downloading each required company at most once.

    Force bypasses the Wikipedia TTL. Prices remain shared and daily to prevent
    repeated manual clicks from downloading whole price histories again.
    """
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    lock_path = root / 'company_data/peer_refresh.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _file_lock(str(lock_path)):
        try:
            companies, _ = load_companies(root, force=force, now=now, fetch=fetch_companies)
            context = build_gics_peer_frames(target, companies=companies)
            # Sector contains all narrower baskets. Fetch only prices due today.
            due = []
            for symbol in sorted(set(context.symbols('Sector'))):
                key = _fingerprint([symbol, period, interval, 'adjusted', 1])[:24]
                path = root / 'company_data/peer_prices' / f'{key}.csv'
                meta = _read_json(path.with_suffix('.json'))
                if (not path.exists() or _stamp(meta.get('last_success')).date() != now.date()) and (
                    not meta.get('error') or now - _stamp(meta.get('last_attempt')) >= timedelta(hours=1)
                ):
                    due.append(symbol)
            batch = {}
            batch_error = None
            if due:
                try:
                    batch = (fetch_prices or _fetch_prices)(symbols=due, period=period,
                        interval=interval, provider='yfinance', align=False)
                except Exception as exc:
                    batch_error = exc
            def batch_fetch(**kwargs):
                if batch_error:
                    raise batch_error
                return {symbol: batch.get(symbol) for symbol in kwargs['symbols']}
            # Populate shared caches even if an unavailable company interrupts a
            # particular basket; other classification levels can still succeed.
            for symbol in due:
                _load_price(root, symbol, period, interval, now, batch_fetch)
        except Exception:
            # Per-level handling below reports the failure and recovers caches.
            batch_fetch = fetch_prices or _fetch_prices
        return {level: refresh_peer_benchmark(root, target, period=period,
                    interval=interval, now=now, fetch_companies=fetch_companies,
                    fetch_prices=batch_fetch, level=level) for level in LEVELS}


def peer_status(meta):
    checked = meta.get('wikipedia_last_checked')
    text = (f"Peers: {len(meta.get('constituents', []))} | Wikipedia last checked: "
            f"{checked[:10] if checked else 'never'} | Prices through: {meta.get('price_end', 'unavailable')}")
    if meta.get('error'):
        state = 'No usable benchmark. ' if meta.get('unavailable') else 'Using available data; refresh incomplete. '
        return text + ' | ' + state + meta['error']
    return text + (' | Using cache' if meta.get('used_cache') else ' | Updated')

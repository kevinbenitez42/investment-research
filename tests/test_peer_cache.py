"""Offline checks for freshness, membership changes, and failure recovery."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from apps.web.peer_cache import load_companies, refresh_gics_benchmarks, peer_status
from Quantapp.data import GICSDataClient


def companies():
    rows = []
    for i in range(1404):
        rows.append(dict(Symbol=f'X{i}', Capitalization=('Large Cap','Mid Cap','Small Cap')[i % 3],
                         Sector='Other', **{'Industry Group': 'Other', 'Industry': 'Other',
                                           'Sub-Industry': 'Other', 'GICS Code': 1}))
    for i, symbol in enumerate(['PG', 'A', 'B', 'C', 'D', 'E']):
        rows[i].update(Symbol=symbol, Sector='Staples')
        rows[i]['Industry Group'] = 'Household' if i < 5 else 'Food'
        rows[i]['Industry'] = 'Personal' if i < 4 else 'Home'
        rows[i]['Sub-Industry'] = 'Care' if i < 3 else 'Other Care'
    return pd.DataFrame(rows)


def prices(**kwargs):
    return {s: pd.DataFrame({'Close': [100., 101., 99., 105.]},
            index=pd.date_range('2026-09-15', periods=4)) for s in kwargs['symbols']}


class PeerCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
        self.fetch = Mock(side_effect=companies)
        self.prices = Mock(side_effect=prices)

    def refresh(self, **kwargs):
        return refresh_gics_benchmarks(self.root, 'PG', now=kwargs.pop('now', self.now),
            fetch_companies=self.fetch, fetch_prices=self.prices, **kwargs)

    def test_all_levels_share_download_and_second_load_makes_no_queries(self):
        result = self.refresh()
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual(self.prices.call_count, 1)
        self.assertEqual(set(self.prices.call_args.kwargs['symbols']), {'A','B','C','D','E'})
        for level, count in [('Sector',5), ('Industry Group',4), ('Industry',3), ('Sub-Industry',2)]:
            frame, meta = result[level]
            self.assertFalse(frame.empty)
            self.assertEqual(len(meta['constituents']), count)
            self.assertNotIn('PG', meta['constituents'])
            self.assertIsNone(meta['error'])
        again = self.refresh()
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual(self.prices.call_count, 1)
        self.assertTrue(all(meta['used_cache'] for _,meta in again.values()))

    def test_daily_prices_weekly_companies_and_manual_refresh(self):
        self.refresh()
        self.refresh(now=self.now + timedelta(days=1))
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual(self.prices.call_count, 2)
        self.refresh(now=self.now + timedelta(days=7))
        self.assertEqual(self.fetch.call_count, 2)
        self.assertEqual(self.prices.call_count, 3)
        self.refresh(now=self.now + timedelta(days=7), force=True)
        self.assertEqual(self.fetch.call_count, 3)
        self.assertEqual(self.prices.call_count, 3)

    def test_membership_changes_even_with_same_count(self):
        before = self.refresh()['Sub-Industry'][1]
        changed = companies()
        changed.loc[changed.Symbol.eq('B'), 'Symbol'] = 'NEW'
        self.fetch.side_effect = lambda: changed
        after = self.refresh(force=True)['Sub-Industry'][1]
        self.assertNotEqual(before['membership_fingerprint'], after['membership_fingerprint'])
        self.assertEqual(after['constituents'], ['A','NEW'])
        self.assertEqual(self.prices.call_args.kwargs['symbols'], ['NEW'])

    def test_failed_wikipedia_refresh_keeps_success_stamp_and_backs_off(self):
        self.refresh()
        self.fetch.side_effect = RuntimeError('offline')
        result = self.refresh(now=self.now + timedelta(days=8))
        meta = result['Sector'][1]
        self.assertEqual(meta['wikipedia_last_checked'], self.now.isoformat())
        self.assertIn('offline', peer_status(meta))
        calls = self.fetch.call_count
        self.refresh(now=self.now + timedelta(days=8, minutes=1))
        self.assertEqual(self.fetch.call_count, calls)

    def test_failed_prices_keep_cached_series_and_success_dates(self):
        original = self.refresh()
        self.prices.side_effect = RuntimeError('prices offline')
        result = self.refresh(now=self.now + timedelta(days=1))
        pd.testing.assert_frame_equal(original['Sector'][0], result['Sector'][0])
        meta = result['Sector'][1]
        self.assertIn('prices offline', meta['error'])
        self.assertTrue(all(m['last_success'] == self.now.isoformat() for m in meta['price_checks'].values()))

    def test_invalid_refresh_does_not_replace_valid_list(self):
        load_companies(self.root, now=self.now, fetch=self.fetch)
        path = self.root/'company_data/gics_companies.csv'
        before = path.read_bytes()
        frame, meta = load_companies(self.root, now=self.now, force=True,
                                    fetch=lambda: companies().iloc[:10])
        self.assertEqual(path.read_bytes(), before)
        self.assertIn('Incomplete', meta['error'])
        self.assertEqual(meta['last_success'], self.now.isoformat())

    def test_unavailable_symbol_is_explicitly_excluded(self):
        def partial(**kwargs):
            result = prices(**kwargs)
            result.pop('E', None)
            return result
        self.prices.side_effect = partial
        result = self.refresh()
        self.assertEqual(result['Sector'][1]['excluded_constituents'], ['E'])
        self.assertIn('Excluded', result['Sector'][1]['error'])
        self.assertIsNone(result['Sub-Industry'][1]['error'])

    def test_different_period_requires_new_price_coverage(self):
        self.refresh(period='1y')
        self.refresh(period='20y')
        self.assertEqual(self.prices.call_count, 2)

    def test_daily_fetch_bypasses_underlying_provider_cache(self):
        from apps.web.peer_cache import _fetch_prices
        with patch('apps.web.peer_cache.get_market_history', return_value={}) as fetch:
            _fetch_prices(symbols=['PG'])
        self.assertEqual(fetch.call_args.kwargs['cache_ttl_seconds'], 0)

    def test_new_day_after_recent_success_is_not_failure_backoff(self):
        self.refresh(now=self.now.replace(hour=23, minute=59))
        self.refresh(now=(self.now + timedelta(days=1)).replace(hour=0, minute=1))
        self.assertEqual(self.prices.call_count, 2)


class WikipediaCleanupTests(unittest.TestCase):
    def test_alias_and_duplicate_precedence(self):
        frame = pd.DataFrame({'Symbol':['DUP','SHOP'], 'GICS Sector':['Consumer Discretionary']*2,
                              'GICS Sub-Industry':['Specialty Stores']*2})
        reference = pd.DataFrame([{
            'Sub-Industry Name':'Other Specialty Retail', 'Sub-Industry Code':25504040,
            'Sector Code':25, 'Industry Group Name':'Retail', 'Industry Group Code':2550,
            'Industry Name':'Specialty Retail', 'Industry Code':255040}])
        with patch('Quantapp.data.gics_data_client.fetch_wikipedia_tables', return_value=[frame]), \
             patch.object(GICSDataClient, '_load_gics_table', return_value=reference):
            result = GICSDataClient().retrieve_companies()
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.Capitalization), {'Large Cap'})
        self.assertEqual(set(result['GICS Code']), {25504040})

    def test_conflicting_duplicates_rejected(self):
        a = pd.DataFrame({'Symbol':['DUP'], 'GICS Sector':['One'], 'GICS Sub-Industry':['A']})
        b = a.assign(**{'GICS Sub-Industry':'B'})
        with patch('Quantapp.data.gics_data_client.fetch_wikipedia_tables', side_effect=[[a],[b],[a]]):
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                GICSDataClient().retrieve_companies()


if __name__ == '__main__':
    unittest.main()

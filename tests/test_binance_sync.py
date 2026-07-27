"""
tests/test_binance_sync.py — data_sync/binance_sync.py.

Uses a temp SQLite file (never the real I:\\bianceData\\crypto_kline.db) so
running these tests never touches production data.

Run:  python -m unittest tests.test_binance_sync -v
"""
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import data_sync.binance_sync as binance_sync
from data_provider.exceptions import DataProviderNetworkError
from data_sync.binance_sync import sync_all, sync_symbol


class _FakeProvider:
    """Mimics BinanceDataProvider.get_history's pagination contract: returns
    up to `limit` bars starting at `start` (ms), spaced 1 day apart, never
    past `end`. Lets pagination be tested deterministically without network."""

    def __init__(self, bar_span_ms=86_400_000, close_value=100.0):
        self.bar_span_ms = bar_span_ms
        self.close_value = close_value
        self.calls = []

    def get_history(self, symbol, interval, start=None, end=None, limit=None):
        self.calls.append((symbol, interval, start, end, limit))
        # Align to a fixed calendar grid (multiples of bar_span_ms since
        # epoch), matching real Binance behavior: daily candles always open
        # at UTC midnight regardless of the exact `start` ms requested. A
        # naive "step from whatever start we got" fake would double-count
        # bars across pages once the cursor drifts off the grid by +1ms.
        aligned_start = (start // self.bar_span_ms) * self.bar_span_ms
        if aligned_start < start:
            aligned_start += self.bar_span_ms
        rows = []
        t = aligned_start
        while t <= end and len(rows) < limit:
            rows.append(t)
            t += self.bar_span_ms
        if not rows:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return pd.DataFrame({
            "datetime": [pd.Timestamp(ms, unit="ms") for ms in rows],
            "open":   [self.close_value] * len(rows),
            "high":   [self.close_value] * len(rows),
            "low":    [self.close_value] * len(rows),
            "close":  [self.close_value] * len(rows),
            "volume": [1.0] * len(rows),
        })


class _AlwaysFailsProvider:
    def __init__(self):
        self.call_count = 0

    def get_history(self, *args, **kwargs):
        self.call_count += 1
        raise DataProviderNetworkError("simulated network failure")


class TestSyncSymbolPagination(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_crypto_kline.db"
        self._orig_batch_limit = binance_sync._BATCH_LIMIT
        binance_sync._BATCH_LIMIT = 5  # force multiple pages with a small dataset

    def tearDown(self):
        binance_sync._BATCH_LIMIT = self._orig_batch_limit
        self._tmpdir.cleanup()

    def test_multi_page_sync_persists_all_rows(self):
        # 12 days of data, 5-row pages -> 3 pages (5+5+2)
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 12)
        provider = _FakeProvider()

        new_count = sync_symbol("BTCUSDT", "1d", start, end,
                                 db_path=self.db_path, provider=provider)

        self.assertEqual(new_count, 12)
        self.assertGreaterEqual(len(provider.calls), 3)

        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM crypto_kline WHERE symbol='BTCUSDT' AND interval='1d'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 12)

    def test_rerun_upserts_without_new_rows_and_overwrites_values(self):
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)
        provider_v1 = _FakeProvider(close_value=100.0)
        sync_symbol("ETHUSDT", "1d", start, end, db_path=self.db_path, provider=provider_v1)

        provider_v2 = _FakeProvider(close_value=200.0)
        new_count = sync_symbol("ETHUSDT", "1d", start, end,
                                 db_path=self.db_path, provider=provider_v2)

        self.assertEqual(new_count, 0)  # same rows, no new ones added

        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT DISTINCT close FROM crypto_kline WHERE symbol='ETHUSDT' AND interval='1d'"
        ).fetchall()
        conn.close()
        self.assertEqual(rows, [(200.0,)])  # overwritten, not duplicated

    def test_unsupported_interval_raises(self):
        with self.assertRaises(ValueError):
            sync_symbol("BTCUSDT", "2h", datetime(2024, 1, 1), datetime(2024, 1, 2),
                        db_path=self.db_path)

    def test_retry_exhausts_after_three_attempts(self):
        provider = _AlwaysFailsProvider()
        with self.assertRaises(DataProviderNetworkError):
            sync_symbol("BTCUSDT", "1d", datetime(2024, 1, 1), datetime(2024, 1, 2),
                        db_path=self.db_path, provider=provider)
        self.assertEqual(provider.call_count, 3)

    def test_sync_all_isolates_per_symbol_failures(self):
        good = _FakeProvider()

        class _MixedProvider:
            def __init__(self):
                self.n = 0
            def get_history(self, symbol, interval, start=None, end=None, limit=None):
                if symbol == "BADCOIN":
                    raise DataProviderNetworkError("no such symbol")
                return good.get_history(symbol, interval, start=start, end=end, limit=limit)

        import unittest.mock as mock
        with mock.patch("data_sync.binance_sync.BinanceDataProvider", return_value=_MixedProvider()):
            results = sync_all(["BTCUSDT", "BADCOIN"], interval="1d",
                                start_time=datetime(2024, 1, 1), end_time=datetime(2024, 1, 3),
                                db_path=self.db_path)

        self.assertEqual(results["BTCUSDT"], 3)
        self.assertIsNone(results["BADCOIN"])


class TestSyncSymbolLiveSmoke(unittest.TestCase):
    """Hits the real Binance public API — needs network access."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test_crypto_kline_live.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_sync_btc_eth_sol_last_365_days_daily(self):
        end = datetime.now()
        start = end - timedelta(days=365)

        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            new_count = sync_symbol(symbol, "1d", start, end, db_path=self.db_path)
            self.assertGreater(new_count, 350)

            conn = sqlite3.connect(str(self.db_path))
            count = conn.execute(
                "SELECT COUNT(*) FROM crypto_kline WHERE symbol=? AND interval='1d'",
                (symbol,),
            ).fetchone()[0]
            conn.close()
            self.assertGreater(count, 350)


if __name__ == "__main__":
    unittest.main()

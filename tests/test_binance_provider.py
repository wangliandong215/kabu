"""
tests/test_binance_provider.py — smoke tests for BinanceDataProvider.

These call the real Binance public REST API (no API key needed for the
endpoints under test) — they need network access to api.binance.com.

Run:  python -m unittest tests.test_binance_provider -v
"""
import unittest

from data_provider.binance_provider import BinanceDataProvider


class TestBinanceDataProvider(unittest.TestCase):

    def setUp(self):
        self.provider = BinanceDataProvider()

    def test_get_latest_price_btcusdt(self):
        price = self.provider.get_latest_price("BTCUSDT")
        self.assertIsInstance(price, float)
        self.assertGreater(price, 0)

    def test_get_history_btcusdt_daily_100_bars(self):
        df = self.provider.get_history("BTCUSDT", "1d", limit=100)
        self.assertEqual(len(df), 100)
        self.assertEqual(list(df.columns), ["datetime", "open", "high", "low", "close", "volume"])
        for col in ("open", "high", "low", "close", "volume"):
            self.assertEqual(df[col].dtype.kind, "f")

    def test_get_symbols_returns_usdt_trading_pairs(self):
        symbols = self.provider.get_symbols()
        self.assertGreater(len(symbols), 0)
        self.assertIn("BTCUSDT", symbols)
        self.assertTrue(all(s.endswith("USDT") for s in symbols))


if __name__ == "__main__":
    unittest.main()

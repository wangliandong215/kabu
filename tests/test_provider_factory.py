"""
tests/test_provider_factory.py — get_provider() market -> Provider routing.

Run:  python -m unittest tests.test_provider_factory -v
"""
import unittest

from data_provider.binance_provider import BinanceDataProvider
from data_provider.moomoo_provider import MoomooDataProvider
from data_provider.provider_factory import get_provider


class TestGetProvider(unittest.TestCase):

    def test_us_stock_returns_moomoo_provider(self):
        self.assertIsInstance(get_provider("US_STOCK"), MoomooDataProvider)

    def test_hk_stock_returns_moomoo_provider(self):
        self.assertIsInstance(get_provider("HK_STOCK"), MoomooDataProvider)

    def test_jp_stock_returns_moomoo_provider(self):
        self.assertIsInstance(get_provider("JP_STOCK"), MoomooDataProvider)

    def test_crypto_returns_binance_provider(self):
        self.assertIsInstance(get_provider("CRYPTO"), BinanceDataProvider)

    def test_unknown_market_raises(self):
        with self.assertRaises(ValueError):
            get_provider("NOT_A_MARKET")


if __name__ == "__main__":
    unittest.main()

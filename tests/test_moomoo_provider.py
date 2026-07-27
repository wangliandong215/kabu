"""
tests/test_moomoo_provider.py — MoomooDataProvider is a pure routing layer:
these tests prove it forwards to the exact same underlying functions with
the exact same arguments the pre-refactor call sites used to pass directly,
so wiring runner.py/scanner.py/etc. through it changes zero behavior.

No real OpenD connection is exercised — the underlying fetch_kline/get_price
functions are monkeypatched at the module-attribute level (same technique
engine/test_runner_drawdown_halt.py already uses), and the assertions check
the arguments MoomooDataProvider forwarded them, not real market data.

Run:  python -m unittest tests.test_moomoo_provider -v
"""
import unittest

import backtest
import config
import data.fetcher
from data_provider.moomoo_provider import MoomooDataProvider


class TestMoomooDataProviderEquivalence(unittest.TestCase):

    def setUp(self):
        self.provider = MoomooDataProvider(market="US_STOCK")
        self._orig_fetcher_fetch_kline = data.fetcher.fetch_kline
        self._orig_fetcher_get_price = data.fetcher.get_price
        self._orig_backtest_fetch_kline = backtest.fetch_kline
        self._orig_watchlist = config.WATCHLIST

    def tearDown(self):
        data.fetcher.fetch_kline = self._orig_fetcher_fetch_kline
        data.fetcher.get_price = self._orig_fetcher_get_price
        backtest.fetch_kline = self._orig_backtest_fetch_kline
        config.WATCHLIST = self._orig_watchlist

    def test_get_history_without_start_end_uses_fetcher_ktype_bars_path(self):
        # This is the ktype+bars style used by every live-path call site
        # (engine/runner.py, scanner.py, momentum.py, trade.py) -- it must
        # keep going through data/fetcher.py's live_cache, not backtest.py's.
        captured = {}

        def fake_fetch_kline(code, ktype="1d", bars=None):
            captured["args"] = (code, ktype, bars)
            return "LIVE_CACHE_RESULT"

        data.fetcher.fetch_kline = fake_fetch_kline

        result = self.provider.get_history("US.AAPL", interval="K_DAY", limit=120)

        self.assertEqual(captured["args"], ("US.AAPL", "K_DAY", 120))
        self.assertEqual(result, "LIVE_CACHE_RESULT")

    def test_get_history_with_start_end_uses_backtest_cache_path(self):
        # This is the start/end style used by backtest_portfolio.py,
        # engine/hmm_regime.py, engine/hmm_shadow.py -- must keep going
        # through backtest.py's own long-TTL backtest_cache.
        captured = {}

        def fake_fetch_kline(code, start, end):
            captured["args"] = (code, start, end)
            return "BACKTEST_CACHE_RESULT"

        backtest.fetch_kline = fake_fetch_kline

        result = self.provider.get_history("US.AAPL", start="2020-01-01", end="2020-06-01")

        self.assertEqual(captured["args"], ("US.AAPL", "2020-01-01", "2020-06-01"))
        self.assertEqual(result, "BACKTEST_CACHE_RESULT")

    def test_get_latest_price_delegates_to_fetcher_get_price(self):
        data.fetcher.get_price = lambda code: 42.0
        self.assertEqual(self.provider.get_latest_price("US.AAPL"), 42.0)

    def test_get_symbols_filters_watchlist_by_market_prefix(self):
        config.WATCHLIST = ["US.AAPL", "JP.6758", "HK.00700"]

        self.assertEqual(MoomooDataProvider(market="US_STOCK").get_symbols(), ["US.AAPL"])
        self.assertEqual(MoomooDataProvider(market="JP_STOCK").get_symbols(), ["JP.6758"])
        self.assertEqual(MoomooDataProvider(market="HK_STOCK").get_symbols(), ["HK.00700"])


if __name__ == "__main__":
    unittest.main()

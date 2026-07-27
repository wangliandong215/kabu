"""
data_provider/moomoo_provider.py — DataProvider adapter over kabu's existing
moomoo data-fetching code.

This is a thin routing layer, not a reimplementation: it does not change any
behavior of data/fetcher.py or backtest.py. Two call conventions already
coexist in the codebase by design, with separate caches:
    - data/fetcher.py::fetch_kline(code, ktype, bars)  -> live_cache, short TTL
    - backtest.py::fetch_kline(code, start, end)       -> backtest_cache, long TTL
get_history() below picks between them based on whether start/end is given,
so both caching paths are preserved exactly.

Imports of fetch_kline/get_price happen INSIDE each method (not at module
top level) on purpose — engine/runner.py's existing tests monkeypatch
`data.fetcher.fetch_kline` / `data.fetcher.get_price` as module attributes,
which only takes effect on callers that re-import at call time. A top-level
import here would silently break that test-patching pattern.
"""
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

from data_provider.base import DataProvider

_SYMBOL_PREFIX_BY_MARKET = {
    "US_STOCK": "US.",
    "JP_STOCK": "JP.",
    "HK_STOCK": "HK.",
}


class MoomooDataProvider(DataProvider):
    """DataProvider backed by kabu's existing moomoo OpenD data-fetching code."""

    def __init__(self, market: str = "US_STOCK"):
        self._market = market

    def get_history(
        self,
        symbol: str,
        interval: str = "K_DAY",
        start: Optional[Union[str, int, float, datetime, date]] = None,
        end: Optional[Union[str, int, float, datetime, date]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        if start is not None or end is not None:
            import backtest
            return backtest.fetch_kline(symbol, start, end)

        from data.fetcher import fetch_kline
        return fetch_kline(symbol, ktype=interval, bars=limit)

    def get_latest_price(self, symbol: str) -> float:
        from data.fetcher import get_price
        return get_price(symbol)

    def get_symbols(self) -> list:
        import config
        prefix = _SYMBOL_PREFIX_BY_MARKET.get(self._market)
        if prefix is None:
            return list(config.WATCHLIST)
        return [c for c in config.WATCHLIST if c.startswith(prefix)]

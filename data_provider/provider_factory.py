"""
data_provider/provider_factory.py — picks a DataProvider by config.MARKET.

This is what lets strategies/engine code stay data-source-agnostic: they
call get_provider(config.MARKET) once and only ever talk to the returned
DataProvider interface (get_history/get_latest_price/get_symbols), never a
specific exchange SDK directly.

`source` (default "live") lets CRYPTO callers opt into reading the locally
synced crypto_kline table (data_sync/binance_sync.py) instead of hitting the
live Binance API — e.g. backtests, which must be reproducible offline and
must not depend on network access. This is purely additive: every existing
call site (get_provider(config.MARKET) with one positional arg) is
unaffected, since source defaults to the pre-existing "live" behavior.
"""
from data_provider.base import DataProvider
from data_provider.binance_db_provider import BinanceDBProvider
from data_provider.binance_provider import BinanceDataProvider
from data_provider.moomoo_provider import MoomooDataProvider

_MOOMOO_MARKETS = {"US_STOCK", "HK_STOCK", "JP_STOCK"}


def get_provider(market: str, source: str = "live") -> DataProvider:
    if market in _MOOMOO_MARKETS:
        # MoomooDataProvider already dispatches live_cache vs backtest_cache
        # internally based on the get_history() call shape (see its
        # docstring) — no separate `source` handling needed here.
        return MoomooDataProvider(market=market)
    if market == "CRYPTO":
        if source == "db":
            return BinanceDBProvider()
        return BinanceDataProvider()
    raise ValueError(f"Unknown MARKET: {market!r}")

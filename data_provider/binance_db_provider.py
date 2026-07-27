"""
data_provider/binance_db_provider.py — DataProvider reading ONLY from the
local crypto_kline SQLite table (data_sync/binance_sync.py's sync target).

Never calls the Binance API. Built for backtest/analysis workloads that must
be reproducible offline and must not hit the network on every run — the
counterpart to BinanceDataProvider (which always hits the live API and is
what data_sync/binance_sync.py itself uses to populate crypto_kline in the
first place).

If crypto_kline has no rows for the requested symbol/interval, get_history()
raises rather than silently falling back to the live API — callers must run
data_sync.binance_sync.sync_symbol()/sync_all() first.

_DEFAULT_DB_PATH is intentionally duplicated (not imported) from
data_sync/binance_sync.py, matching this codebase's existing "duplicate
rather than cross-import" discipline between sibling modules (see
engine/hmm_regime.py's docstring for the established precedent) — avoids a
data_provider -> data_sync -> data_provider import cycle risk.
"""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from data_provider.base import DataProvider
from data_provider.exceptions import DataProviderParseError

_DEFAULT_DB_PATH = Path(r"I:\bianceData\crypto_kline.db")

_OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def _to_iso(value: Union[str, int, float, datetime, date]) -> str:
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="ms").isoformat()
    return pd.Timestamp(value).isoformat()


class BinanceDBProvider(DataProvider):
    """Read-only DataProvider over the crypto_kline table. No network access."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, interval: str = "1d"):
        self._db_path = Path(db_path or _DEFAULT_DB_PATH)
        self._default_interval = interval

    def get_history(
        self,
        symbol: str,
        interval: Optional[str] = None,
        start: Optional[Union[str, int, float, datetime, date]] = None,
        end: Optional[Union[str, int, float, datetime, date]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        interval = interval or self._default_interval
        conn = sqlite3.connect(str(self._db_path))
        try:
            query = ("SELECT datetime, open, high, low, close, volume FROM crypto_kline "
                      "WHERE symbol=? AND interval=?")
            params = [symbol, interval]
            if start is not None:
                query += " AND datetime >= ?"
                params.append(_to_iso(start))
            if end is not None:
                query += " AND datetime <= ?"
                params.append(_to_iso(end))
            query += " ORDER BY datetime ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            df = pd.read_sql_query(query, conn, params=params)
        except Exception as exc:
            raise DataProviderParseError(f"get_history({symbol}, {interval}) failed: {exc}") from exc
        finally:
            conn.close()

        if df.empty:
            raise ValueError(
                f"No local crypto_kline data for symbol={symbol} interval={interval} "
                f"in [{start}, {end}] — run data_sync.binance_sync.sync_symbol() first."
            )
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df[_OHLCV_COLUMNS].reset_index(drop=True)

    def get_latest_price(self, symbol: str) -> float:
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT close FROM crypto_kline WHERE symbol=? AND interval=? "
                "ORDER BY datetime DESC LIMIT 1",
                (symbol, self._default_interval),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"No local crypto_kline data for symbol={symbol} — run sync first.")
        return float(row[0])

    def get_symbols(self) -> list:
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute("SELECT DISTINCT symbol FROM crypto_kline").fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

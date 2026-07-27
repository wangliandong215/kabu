"""
data_sync/binance_sync.py — sync Binance K-lines into a local SQLite table.

Pulls historical K-lines via data_provider.BinanceDataProvider and upserts
them into the `crypto_kline` table, so backtest/AI-analysis/live code can all
read the same local data instead of hitting the Binance API directly.

Storage: I:\\bianceData\\crypto_kline.db (crypto data root — see kabu's
I:\\kabuData vs I:\\bianceData split; this file is intentionally separate
from engine/trade_tracker.py's stock trade_history.db, never shared with it).

Pagination: Binance's klines endpoint caps each response at _BATCH_LIMIT
rows. sync_symbol() walks forward from start_time in batches, advancing the
cursor to (last returned bar's time + 1ms) each page, until end_time is
reached or a short page signals no more data is available.
"""
import sqlite3
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from data_provider.binance_provider import BinanceDataProvider
from data_provider.exceptions import DataProviderError
from data_provider.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_DB_PATH = Path(r"I:\bianceData\crypto_kline.db")

_SUPPORTED_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}

_BATCH_LIMIT = 1000       # Binance klines max rows per request
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0

# Same rationale as engine/trade_tracker.py's _write_lock: serialize writes
# so overlapping sync_all() symbols never interleave multi-statement commits.
_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crypto_kline (
    symbol   TEXT NOT NULL,
    interval TEXT NOT NULL,
    datetime TEXT NOT NULL,
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   REAL NOT NULL,
    PRIMARY KEY (symbol, interval, datetime)
);
"""


def _get_conn(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    path = Path(db_path or _DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    with _write_lock:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    return conn


def _to_ms(value: Union[str, int, float, datetime, date]) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return int(pd.Timestamp(value).timestamp() * 1000)


def _fetch_page_with_retry(
    provider: BinanceDataProvider, symbol: str, interval: str,
    start_ms: int, end_ms: int,
) -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return provider.get_history(symbol, interval, start=start_ms,
                                         end=end_ms, limit=_BATCH_LIMIT)
        except DataProviderError as exc:
            last_exc = exc
            _log.error("fetch failed (attempt %d/%d) symbol=%s interval=%s — %s",
                       attempt, _MAX_RETRIES, symbol, interval, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc


def _upsert(conn: sqlite3.Connection, symbol: str, interval: str, df: pd.DataFrame) -> None:
    rows = [
        (symbol, interval, row.datetime.isoformat(), float(row.open),
         float(row.high), float(row.low), float(row.close), float(row.volume))
        for row in df.itertuples(index=False)
    ]
    with _write_lock:
        conn.executemany(
            """INSERT INTO crypto_kline
                   (symbol, interval, datetime, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, interval, datetime) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low,
                   close=excluded.close, volume=excluded.volume
            """, rows)
        conn.commit()


def _row_count(conn: sqlite3.Connection, symbol: str, interval: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM crypto_kline WHERE symbol=? AND interval=?",
        (symbol, interval),
    ).fetchone()[0]


def sync_symbol(
    symbol: str,
    interval: str,
    start_time: Union[str, int, float, datetime, date],
    end_time: Union[str, int, float, datetime, date],
    db_path: Optional[Union[str, Path]] = None,
    provider: Optional[BinanceDataProvider] = None,
) -> int:
    """Sync all K-lines for one symbol/interval over [start_time, end_time]
    into the crypto_kline table (upsert — existing rows are overwritten).

    Returns the number of NEW rows added (rows that didn't already exist for
    this symbol+interval before the sync), not the total rows touched.
    """
    if interval not in _SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval!r} (supported: {sorted(_SUPPORTED_INTERVALS)})")

    provider = provider or BinanceDataProvider()
    conn = _get_conn(db_path)

    start_ms = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    _log.info("sync start symbol=%s interval=%s start=%s end=%s", symbol, interval, start_time, end_time)
    t0 = time.time()

    try:
        before = _row_count(conn, symbol, interval)
        cursor_ms = start_ms

        while cursor_ms <= end_ms:
            df = _fetch_page_with_retry(provider, symbol, interval, cursor_ms, end_ms)
            if df is None or df.empty:
                break

            _upsert(conn, symbol, interval, df)

            last_ms = int(df.iloc[-1]["datetime"].timestamp() * 1000)
            if last_ms <= cursor_ms:
                break  # no forward progress — avoid an infinite loop
            cursor_ms = last_ms + 1

            if len(df) < _BATCH_LIMIT:
                break  # short page — no more data available

        new_count = _row_count(conn, symbol, interval) - before
    except Exception as exc:
        _log.error("sync failed symbol=%s interval=%s — %s", symbol, interval, exc)
        raise
    finally:
        conn.close()

    elapsed = time.time() - t0
    _log.info(
        "sync done symbol=%s interval=%s new_rows=%d elapsed=%.2fs updated_at=%s",
        symbol, interval, new_count, elapsed, datetime.now().isoformat(timespec="seconds"),
    )
    return new_count


def sync_all(
    symbols: List[str],
    interval: str = "1d",
    start_time: Optional[Union[str, int, float, datetime, date]] = None,
    end_time: Optional[Union[str, int, float, datetime, date]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[int]]:
    """Sync multiple symbols. One symbol's failure (after retries) is logged
    and recorded as None in the result, not raised — a bad symbol must not
    abort the rest of the batch.

    end_time defaults to now; start_time defaults to 365 days before
    end_time, so sync_all(symbols) alone syncs "last year" of `interval`.
    """
    end_time = end_time if end_time is not None else datetime.now()
    start_time = start_time if start_time is not None else (pd.Timestamp(end_time) - pd.Timedelta(days=365))

    provider = BinanceDataProvider()
    results: Dict[str, Optional[int]] = {}
    for symbol in symbols:
        try:
            results[symbol] = sync_symbol(
                symbol, interval, start_time, end_time,
                db_path=db_path, provider=provider,
            )
        except Exception as exc:
            _log.error("sync_all: %s failed — %s", symbol, exc)
            results[symbol] = None
    return results

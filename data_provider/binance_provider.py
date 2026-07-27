"""
data_provider/binance_provider.py — Binance Spot public market data.

Phase 1: public market-data endpoints only. No API key/secret is required
for anything implemented here.

NOT implemented in this phase (see TODOs at the bottom of the class):
    - order placement / cancellation
    - account queries
    - position queries
    - fund transfers
These all need authenticated endpoints and belong to a later phase.
"""
import json
import time
from datetime import date, datetime
from typing import Optional, Union

import pandas as pd
from binance.error import ClientError, ServerError
from binance.spot import Spot
from requests.exceptions import RequestException

import config_binance
from data_provider.base import DataProvider
from data_provider.exceptions import (
    DataProviderHTTPError,
    DataProviderNetworkError,
    DataProviderParseError,
)
from data_provider.logger import get_logger

_log = get_logger(__name__)

_OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _to_millis(value: Optional[Union[str, int, float, datetime, date]]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return int(pd.Timestamp(value).timestamp() * 1000)


class BinanceDataProvider(DataProvider):
    """DataProvider backed by Binance Spot public REST endpoints (binance-connector)."""

    def __init__(self, base_url: Optional[str] = None):
        self._client = Spot(base_url=base_url or config_binance.BINANCE_BASE_URL)

    # ── DataProvider interface ───────────────────────────────────────────

    def get_history(
        self,
        symbol: str,
        interval: str,
        start: Optional[Union[str, int, float, datetime, date]] = None,
        end: Optional[Union[str, int, float, datetime, date]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        params = {}
        start_ms = _to_millis(start)
        end_ms = _to_millis(end)
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        if limit is not None:
            params["limit"] = limit

        t0 = time.time()
        raw = self._request(
            lambda: self._client.klines(symbol, interval, **params),
            context=f"get_history(symbol={symbol}, interval={interval})",
        )
        elapsed = time.time() - t0
        _log.info(
            "request_time=%s symbol=%s interval=%s elapsed=%.3fs",
            datetime.now().isoformat(timespec="seconds"), symbol, interval, elapsed,
        )

        try:
            df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
            df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            return df[_OHLCV_COLUMNS].reset_index(drop=True)
        except (KeyError, ValueError, TypeError) as e:
            raise DataProviderParseError(f"failed to parse klines for {symbol}: {e}") from e

    def get_latest_price(self, symbol: str) -> float:
        data = self._request(
            lambda: self._client.ticker_price(symbol=symbol),
            context=f"get_latest_price(symbol={symbol})",
        )
        try:
            return float(data["price"])
        except (KeyError, ValueError, TypeError) as e:
            raise DataProviderParseError(f"failed to parse price for {symbol}: {e}") from e

    def get_symbols(self) -> list:
        data = self._request(
            lambda: self._client.exchange_info(),
            context="get_symbols()",
        )
        try:
            return [
                s["symbol"] for s in data["symbols"]
                if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
            ]
        except (KeyError, TypeError) as e:
            raise DataProviderParseError(f"failed to parse exchange info: {e}") from e

    # ── request helper ───────────────────────────────────────────────────

    def _request(self, fn, context: str):
        try:
            return fn()
        except (ClientError, ServerError) as e:
            _log.error("HTTP error in %s: %s", context, e)
            raise DataProviderHTTPError(f"{context}: {e}") from e
        except RequestException as e:
            _log.error("network error in %s: %s", context, e)
            raise DataProviderNetworkError(f"{context}: {e}") from e
        except json.JSONDecodeError as e:
            _log.error("JSON parse error in %s: %s", context, e)
            raise DataProviderParseError(f"{context}: {e}") from e

    # ── Trading — out of scope for Phase 1, do not implement yet ────────

    def place_order(self, *args, **kwargs):
        # TODO(phase 2 — trading): implement once order placement is in scope.
        raise NotImplementedError("place_order is out of scope for Phase 1 (public data only)")

    def cancel_order(self, *args, **kwargs):
        # TODO(phase 2 — trading)
        raise NotImplementedError("cancel_order is out of scope for Phase 1 (public data only)")

    def get_account(self, *args, **kwargs):
        # TODO(phase 2 — trading)
        raise NotImplementedError("get_account is out of scope for Phase 1 (public data only)")

    def get_positions(self, *args, **kwargs):
        # TODO(phase 2 — trading)
        raise NotImplementedError("get_positions is out of scope for Phase 1 (public data only)")

    def transfer(self, *args, **kwargs):
        # TODO(phase 2 — trading)
        raise NotImplementedError("transfer is out of scope for Phase 1 (public data only)")

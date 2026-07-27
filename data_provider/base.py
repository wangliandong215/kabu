"""
data_provider/base.py — common interface implemented by every data provider.

Phase 1 only covers public market data. Trading/account methods
(place_order, cancel_order, get_account, get_positions, transfer) are
deliberately not part of this interface yet — they belong to a later phase
once a provider needs authenticated endpoints.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union
from datetime import date, datetime

import pandas as pd


class DataProvider(ABC):
    """Uniform read-only market-data interface across exchanges/brokers."""

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        interval: str,
        start: Optional[Union[str, int, float, datetime, date]] = None,
        end: Optional[Union[str, int, float, datetime, date]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV history with columns: datetime, open, high, low, close, volume."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Return the current/last traded price for symbol."""
        raise NotImplementedError

    @abstractmethod
    def get_symbols(self) -> list:
        """Return all tradeable symbols this provider exposes."""
        raise NotImplementedError

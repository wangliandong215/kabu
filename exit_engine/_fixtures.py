# -*- coding: utf-8 -*-
"""
exit_engine/_fixtures.py — shared test-only helpers for exit_engine/'s own
test_*.py files. Not itself a test module (no "test_" prefix, so
unittest's discovery never picks it up) — mirrors the role
position_manager/test_*.py's individual `_ctx()` helpers play, centralized
here because ExitEngineContext carries several DataFrame fields that would
otherwise be duplicated eight times, once per signal module's test file.
"""
from datetime import datetime, timedelta

import pandas as pd

from exit_engine.models import ExitEngineContext

DEFAULT_NOW = datetime(2026, 1, 2, 10, 0, 0)


def make_bars(closes, highs=None, lows=None, volumes=None,
              start=datetime(2026, 1, 2, 9, 30, 0), freq_minutes=1):
    """A synthetic OHLCV DataFrame with a moomoo-style `time_key` column,
    one row per `closes` entry, timestamps `freq_minutes` apart starting
    at `start`."""
    n = len(closes)
    times = [start + timedelta(minutes=freq_minutes * i) for i in range(n)]
    return pd.DataFrame({
        "time_key": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
        "open": closes,
        "high": highs if highs is not None else [c + 0.1 for c in closes],
        "low": lows if lows is not None else [c - 0.1 for c in closes],
        "close": closes,
        "volume": volumes if volumes is not None else [1000.0] * n,
    })


def make_context(**overrides) -> ExitEngineContext:
    defaults = dict(
        symbol="US.TEST", trade_id="US.TEST_2026-01-01T00:00:00",
        current_price=100.0, entry_price=100.0, qty=100, entry_time="2026-01-01T00:00:00",
        holding_days=1.0, current_atr=1.0, entry_atr=1.0,
        peak_price=100.0, peak_date="2026-01-01",
        bars_1m=None, bars_5m=None, bars_15m=None, session_bars_1m=None,
        market_regime=1, now=DEFAULT_NOW,
    )
    defaults.update(overrides)
    return ExitEngineContext(**defaults)

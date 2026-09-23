# -*- coding: utf-8 -*-
"""
exit_engine/models.py — data structures for V3.2-A Minute Exit Engine
(分钟级卖出引擎诊断). See exit_engine/__init__.py for the overall
architecture.

ExitEngineContext is the single stable shape every V3.2 signal module
reads from. Nothing here is re-derived by a signal module —
exit_engine/__init__.py::build_context() computes every field once per
symbol per engine/runner.py::run_once() pass, mirroring
position_manager/models.py's PositionManagementContext convention.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

TIER_LOW = "LOW"
TIER_MEDIUM = "MEDIUM"
TIER_HIGH = "HIGH"
TIER_CRITICAL = "CRITICAL"
VALID_TIERS = (TIER_LOW, TIER_MEDIUM, TIER_HIGH, TIER_CRITICAL)


@dataclass
class ExitEngineContext:
    """Everything V3.2-A's signal modules need about one held position,
    for one run_once() pass. `now` is threaded through explicitly (not
    read via datetime.now() inside signal modules) so every module stays
    a pure, deterministic, easily-testable function of this context.

    `bars_1m`/`bars_5m`/`bars_15m` are the raw multi-day intraday windows
    fetch_kline() returns (used by EMA/trend, which are continuous and
    should NOT reset daily). `session_bars_1m` is bars_1m filtered down to
    just the most recent trading session (used by VWAP and the opening/
    afternoon time-of-day signals, which DO reset daily) — see
    exit_engine/__init__.py::_session_bars()."""
    symbol: str
    trade_id: str

    current_price: float
    entry_price: float
    qty: int
    entry_time: str
    holding_days: float

    current_atr: Optional[float]
    entry_atr: Optional[float]

    peak_price: float          # high-water mark since entry (exit_engine's own state_store)
    peak_date: Optional[str]   # calendar date (ISO) peak_price was last raised

    bars_1m: Optional[pd.DataFrame]
    bars_5m: Optional[pd.DataFrame]
    bars_15m: Optional[pd.DataFrame]
    session_bars_1m: Optional[pd.DataFrame]

    market_regime: Optional[int]   # weather_code, passed through read-only

    now: object   # datetime, naive/local — see class docstring


@dataclass
class SignalReading:
    """One signal module's output — pure data, never raises. enabled=False
    means the module's own config flag is off; points/triggered/state
    carry no information in that case (always 0.0/False/None). A module
    that IS enabled but simply didn't fire returns triggered=False,
    points=0.0 — not enabled=False — so pressure_score.aggregate() can
    tell "no opinion" apart from "opinion: no pressure"."""
    name: str
    enabled: bool
    triggered: bool
    points: float
    detail: str
    state: Optional[str] = None        # short machine-readable state label, e.g. "LOST_VWAP"
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExitEngineResult:
    """exit_engine/__init__.py::evaluate()'s output — the full diagnostic
    verdict for one position, one pass. Never fed back into any trading
    decision in V3.2-A — see package docstring."""
    symbol: str
    pressure_score: float
    pressure_tier: str
    exit_confidence: float             # fraction of ENABLED signals that triggered, 0..1
    potential_exit_reason: Optional[str]
    potential_exit_price: Optional[float]
    potential_exit_time: Optional[str]
    signals: List[SignalReading] = field(default_factory=list)

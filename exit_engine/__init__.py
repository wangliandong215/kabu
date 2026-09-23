# -*- coding: utf-8 -*-
"""
exit_engine/ — V3.2-A Minute Exit Engine Diagnostics (分钟级卖出引擎，
Observation Only).

WHAT THIS PACKAGE IS
  Computes, once per engine/runner.py::run_once() pass, for every open
  non-core_etf position: VWAP / EMA / multi-timeframe trend / volume /
  time-of-day / ATR / trailing-stop / break-even signals (each its own
  pure module — vwap_signal.py, ema_signal.py, trend_signal.py,
  volume_signal.py, time_signal.py, atr_signal.py,
  trailing_stop_signal.py, break_even_signal.py), combines them into one
  Exit Pressure Score (pressure_score.py), and appends one row to
  exit_engine_v32_log.jsonl. This is research/diagnostic data for the
  later V3.2-B (per-signal empirical validation) and V3.2-C (multi-signal
  tuning) phases — see the V3.2 spec's section 八 (开发顺序). V3.2-D is
  the earliest phase allowed to let any of this influence a real order,
  and that is a separate, explicit rollout decision — not something a
  config flag in this file can opt into on its own.

WHAT THIS PACKAGE IS NOT (V3.2-A)
  - It never places an order, never calls portfolio.close_position/
    reduce_position, and never feeds risk/guard.py::check_exit_ordered or
    engine/runner.py's own exit-check loop. This is a HARD invariant
    enforced structurally, not just by a config default: evaluate() and
    log_diagnostics() below take no broker/portfolio-mutation handle at
    all, so even with every ENABLE_* flag on, this package is physically
    incapable of selling anything.
  - It does not change Entry, the existing Exit path, Dynamic Position
    Sizing (risk/dynamic_sizing.py), Position Manager (position_manager/),
    Portfolio Risk Manager, QQQ Core Recovery, or Emergency Rebalance.
  - It does not duplicate MFE/MAE/final-pnl/giveback bookkeeping — those
    already live in engine/trade_tracker.py's trades/trade_daily tables.
    This package's log only carries trade_id, so a V3.2-B research script
    can join the two, exactly like research/position_manager_v31_review.py
    already does for V3.1-A.

FAILURE HANDLING
  build_context()/evaluate() never raise on well-formed input — every
  signal module already degrades to enabled=False/triggered=False on bad
  input, and evaluate() wraps each module call in a second try/except as
  defense in depth. log_diagnostics() never lets a logging failure
  propagate — same contract as position_manager/__init__.py::log_decision()
  and regime/__init__.py::_log(). engine/runner.py's caller additionally
  wraps the whole per-symbol block in try/except.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from exit_engine import (atr_signal, break_even_signal, ema_signal,
                          pressure_score, state_store, time_signal,
                          trailing_stop_signal, trend_signal, volume_signal,
                          vwap_signal)
from exit_engine.models import ExitEngineContext, ExitEngineResult, SignalReading

EXIT_ENGINE_LOG_PATH = Path(r"C:\KabuData\portfolio\exit_engine_v32_log.jsonl")

_SIGNAL_MODULES = (vwap_signal, ema_signal, trend_signal, volume_signal,
                    time_signal, atr_signal, trailing_stop_signal, break_even_signal)

__all__ = [
    "build_context", "evaluate", "log_diagnostics",
    "ExitEngineContext", "ExitEngineResult", "SignalReading",
    "EXIT_ENGINE_LOG_PATH",
]


def _holding_days(entry_time_iso: Optional[str], now: datetime) -> float:
    if entry_time_iso:
        try:
            delta = now - datetime.fromisoformat(entry_time_iso)
            return max(0.0, delta.total_seconds() / 86400.0)
        except (ValueError, TypeError):
            pass
    return 0.0


def _session_bars(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Filter intraday bars down to just the most recent trading session —
    the calendar date of the LAST bar in `df`, not a wall-clock/timezone
    computation. VWAP and the opening/afternoon time-of-day signals must
    reset per session, not run over whatever multi-day tail
    fetch_kline("1m") happened to return. Deliberately timezone-free (no
    engine.market_hours dependency): robust across JP/US codes and avoids
    any risk of a circular import with engine/runner.py, which is what
    imports this package."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "time_key" not in df.columns:
            return None
        ts = pd.to_datetime(df["time_key"])
        last_date = ts.iloc[-1].date()
        filtered = df.loc[ts.dt.date == last_date].reset_index(drop=True)
    except Exception:
        return None
    return filtered if not filtered.empty else None


def build_context(code: str, pos: dict, current_price: float,
                   bars_1m: Optional[pd.DataFrame], bars_5m: Optional[pd.DataFrame],
                   bars_15m: Optional[pd.DataFrame], current_atr: Optional[float],
                   trade_id: str, market_regime: Optional[int],
                   now: Optional[datetime] = None) -> Optional[ExitEngineContext]:
    """Assemble one pass's ExitEngineContext for `code`. Returns None on
    current_price<=0, mirroring position_manager.build_context's own
    guard — nothing meaningful can be computed without a live price."""
    if current_price is None or current_price <= 0:
        return None
    if now is None:
        now = datetime.now()

    entry_price = float(pos.get("entry_price", current_price))
    entry_time = pos.get("entry_time", "")
    qty = int(pos.get("qty", 0))
    entry_atr = pos.get("entry_atr")
    today_iso = now.date().isoformat()

    state_store.get_or_init(code, entry_time, entry_price, current_price, today_iso)
    state = state_store.update(code, price=current_price, today_iso=today_iso)
    peak_price = float(state.get("peak_price", max(entry_price, current_price)))
    peak_date = state.get("peak_date")

    return ExitEngineContext(
        symbol=code, trade_id=trade_id,
        current_price=float(current_price), entry_price=entry_price,
        qty=qty, entry_time=entry_time,
        holding_days=_holding_days(entry_time, now),
        current_atr=current_atr, entry_atr=entry_atr,
        peak_price=peak_price, peak_date=peak_date,
        bars_1m=bars_1m, bars_5m=bars_5m, bars_15m=bars_15m,
        session_bars_1m=_session_bars(bars_1m),
        market_regime=market_regime, now=now,
    )


def evaluate(context: ExitEngineContext) -> ExitEngineResult:
    """Pure aggregation — see module docstring. Never raises on a
    well-formed context."""
    signals = []
    for module in _SIGNAL_MODULES:
        try:
            signals.append(module.evaluate(context))
        except Exception as exc:
            signals.append(SignalReading(
                getattr(module, "MODULE_NAME", module.__name__), True, False, 0.0,
                f"signal evaluation failed: {exc}", state="ERROR"))

    score, tier, confidence = pressure_score.aggregate(signals)

    triggered_names = [s.name for s in signals if s.enabled and s.triggered]
    potential_reason = "+".join(triggered_names) if triggered_names else None

    potential_price = None
    trailing = next((s for s in signals if s.name == trailing_stop_signal.MODULE_NAME), None)
    if trailing is not None and trailing.triggered:
        potential_price = trailing.extra.get("trailing_stop_price")
    if potential_price is None:
        atr_reading = next((s for s in signals if s.name == atr_signal.MODULE_NAME), None)
        if atr_reading is not None and atr_reading.triggered:
            potential_price = atr_reading.extra.get("atr_stop_price")

    return ExitEngineResult(
        symbol=context.symbol, pressure_score=score, pressure_tier=tier,
        exit_confidence=confidence, potential_exit_reason=potential_reason,
        potential_exit_price=potential_price,
        potential_exit_time=context.now.isoformat() if triggered_names else None,
        signals=signals,
    )


def log_diagnostics(context: ExitEngineContext, result: ExitEngineResult) -> None:
    """Best-effort JSONL append — a logging failure must never propagate,
    same contract as position_manager/__init__.py::log_decision()."""
    try:
        EXIT_ENGINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": context.symbol,
            "trade_id": context.trade_id,
            "current_price": context.current_price,
            "entry_price": context.entry_price,
            "peak_price": context.peak_price,
            "holding_days": round(context.holding_days, 2),
            "market_regime": context.market_regime,
            "pressure_score": round(result.pressure_score, 2),
            "pressure_tier": result.pressure_tier,
            "exit_confidence": round(result.exit_confidence, 4),
            "potential_exit_reason": result.potential_exit_reason,
            "potential_exit_price": result.potential_exit_price,
            "potential_exit_time": result.potential_exit_time,
            "signals": {
                s.name: {"enabled": s.enabled, "triggered": s.triggered,
                          "points": s.points, "state": s.state, "detail": s.detail}
                for s in result.signals
            },
        }
        with open(EXIT_ENGINE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        import notify.alert as alert
        alert.log(f"exit_engine: failed to write exit_engine_v32_log.jsonl — {exc}")

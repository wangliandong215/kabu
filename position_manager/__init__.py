# -*- coding: utf-8 -*-
"""
position_manager/ — V3.1-A Core Position Management (持仓管理).

WHAT THIS PACKAGE IS
  V3.0-A (risk/dynamic_sizing.py) decides how much to buy AT ENTRY. This
  package decides, for a position already held, what the target position
  SHOULD be right now, given how Confidence/HMM/Volatility/Drawdown have
  moved since entry — see confidence_reduction.py / hmm_reduction.py /
  volatility_reduction.py / drawdown_reduction.py. Each of those is a pure
  function of PositionManagementContext (models.py) -> ModuleSignal;
  evaluate() below is the ONLY place their outputs are combined.

  Combination is multiplicative, not additive: remaining_pct = the product
  of (1 - reduction_i) over every module that triggered. A worked check
  against the V3.1 spec's own example — Confidence 15%, HMM 10%,
  Volatility 10%, Drawdown 5% -> target 65% — comes out to
  0.85*0.90*0.90*0.95 ~= 65.3%, matching almost exactly; the naive
  100% - (15+10+10+5) = 70% the spec explicitly warns against does not.
  This is what keeps multiple simultaneous signals from ever driving
  target below 0% or stacking past what any single position holds.

WHY TARGET IS COMPUTED AGAINST baseline_qty, NOT current qty
  If target were current_qty * remaining, then once a REDUCE actually
  executes (V3.1-A: ACTIVE mode) the SAME unchanged deterioration signal
  would keep shrinking the position every subsequent pass — the exact
  "repeated reduction" bug the spec explicitly calls out. Computing target
  against baseline_qty (position_manager/state_store.py's monotonic
  high-water mark of confirmed size, immune to Position Manager's own past
  REDUCE actions) instead means: once current_qty == target_qty for an
  unchanged signal, the next pass recomputes the identical target and
  action resolves to HOLD, not another REDUCE. See models.py's
  PositionManagementContext docstring and state_store.py.

WHAT THIS PACKAGE IS NOT (V3.1-A)
  - It does not place orders. evaluate() is pure; engine/runner.py decides,
    via config.POSITION_MANAGER_V31_MODE, whether a REDUCE decision is
    actually executed (ACTIVE) or only logged (OBSERVATION, the current
    default) or not computed at all (OFF).
  - It never produces target > current (REDUCE-only in this phase — no
    PYRAMID/EXIT path here; those stay in engine/runner.py's existing 2a/2b
    loops and risk/guard.py respectively, both untouched).
  - It does not call any LLM. See llm_interface.py — config.
    POSITION_LLM_MODE defaults "OFF" and stays inert until a separate,
    explicit rollout decision.
  - It does not touch risk/guard.py's Stop Loss / Take Profit / ATR
    trailing stop, risk/portfolio_risk_manager.py, portfolio/tracker.py's
    position schema, or any Entry/Ranking/Sizing code.

FAILURE HANDLING
  evaluate()/build_context() never raise on well-formed input; log_decision()
  never lets a logging failure propagate — same contract as
  regime/__init__.py::evaluate_and_log()/_log(). engine/runner.py's caller
  wraps the whole per-symbol block in try/except anyway (defense in depth).
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import config
from position_manager import (confidence_reduction, drawdown_reduction,
                               hmm_reduction, state_store, volatility_reduction)
from position_manager.models import (ACTION_HOLD, ACTION_REDUCE, ModuleSignal,
                                      PositionManagementContext,
                                      PositionManagerDecision,
                                      VOLATILITY_NORMAL)

# NOT "position_manager_log.jsonl" — risk/portfolio_position_manager.py (the
# pre-existing, unrelated v2.12 exposure-budget module) already owns that
# exact filename (see its _PM_LOG_PATH). This is a completely different log
# schema for a completely different feature; sharing a path would interleave
# two incompatible JSON shapes into one file.
POSITION_MANAGER_LOG_PATH = Path(r"C:\KabuData\portfolio\position_manager_v31_log.jsonl")

MODE_OFF = "OFF"
MODE_OBSERVATION = "OBSERVATION"
MODE_ACTIVE = "ACTIVE"
VALID_MODES = (MODE_OFF, MODE_OBSERVATION, MODE_ACTIVE)

EXECUTION_EXECUTED = "EXECUTED"
EXECUTION_SKIPPED_OBSERVATION = "SKIPPED_OBSERVATION_MODE"
EXECUTION_SKIPPED_OFF = "SKIPPED_OFF"

_MODULES = (confidence_reduction, hmm_reduction, volatility_reduction, drawdown_reduction)

__all__ = [
    "build_context", "evaluate", "log_decision",
    "PositionManagementContext", "PositionManagerDecision", "ModuleSignal",
    "MODE_OFF", "MODE_OBSERVATION", "MODE_ACTIVE", "VALID_MODES",
    "EXECUTION_EXECUTED", "EXECUTION_SKIPPED_OBSERVATION", "EXECUTION_SKIPPED_OFF",
    "POSITION_MANAGER_LOG_PATH",
]


def _days_held(entry_time_iso: Optional[str]) -> int:
    if entry_time_iso:
        try:
            return (datetime.now() - datetime.fromisoformat(entry_time_iso)).days
        except (ValueError, TypeError):
            pass
    return 0


def build_context(code: str, pos: dict, current_price: float,
                   current_atr: Optional[float],
                   confidence: Optional[float],
                   hmm_state: Optional[str],
                   portfolio,
                   risk_flags: Optional[List[str]] = None) -> Optional[PositionManagementContext]:
    """Assemble one pass's PositionManagementContext for `code`, reading
    Portfolio (already-computed, read-only) plus this module's own
    state_store baselines. Returns None if current_price <= 0 (mirrors
    engine/runner.py's exit loop's own `if price <= 0: continue` guard —
    nothing meaningful can be computed without a live price).

    `confidence`/`hmm_state` are passed in rather than recomputed here —
    callers (engine/runner.py) already have engine.confidence_score.
    gather_and_score()'s result and its embedded hmm_state for this pass,
    and passing them in avoids a second regime_store/confidence_score call
    for the same symbol on the same pass."""
    if current_price is None or current_price <= 0:
        return None

    entry_price = float(pos.get("entry_price", current_price))
    avg_cost = float(pos.get("avg_cost", entry_price))
    qty = int(pos.get("qty", 0))
    entry_time = pos.get("entry_time", "")

    state_store.get_or_init(code, entry_time, entry_price, qty, current_price,
                             confidence, hmm_state)
    state = state_store.update(code, peak_price=current_price, baseline_qty=qty)

    baseline_qty = int(state.get("baseline_qty", qty) or qty)
    peak_price = float(state.get("peak_price", max(entry_price, current_price)))
    confidence_baseline = state.get("confidence_baseline")
    hmm_state_baseline = state.get("hmm_state_baseline")

    current_position_pct = (qty / baseline_qty * 100.0) if baseline_qty > 0 else 100.0
    drawdown_pct = max(0.0, (peak_price - current_price) / peak_price) if peak_price > 0 else 0.0

    atr_pct = None
    if current_atr and current_price > 0:
        atr_pct = float(current_atr) / float(current_price)
    volatility_state = volatility_reduction.classify(atr_pct) if atr_pct is not None else VOLATILITY_NORMAL

    confidence_change = None
    if confidence is not None and confidence_baseline is not None:
        confidence_change = confidence - confidence_baseline
    hmm_state_change = bool(hmm_state and hmm_state_baseline and hmm_state != hmm_state_baseline)

    unrealized_pnl = (current_price - avg_cost) * qty
    unrealized_pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

    sector = config.SECTOR_MAP.get(code, "other")
    portfolio_exposure = portfolio.exposure_pct() if portfolio is not None else None
    sector_exposure = portfolio.sector_exposure_pct(sector) if portfolio is not None else None
    available_cash = portfolio.available_cash() if portfolio is not None else 0.0

    return PositionManagementContext(
        symbol=code,
        current_position=qty,
        current_position_pct=current_position_pct,
        baseline_qty=baseline_qty,
        target_position=qty,
        target_position_pct=current_position_pct,
        entry_price=entry_price,
        current_price=current_price,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        peak_price=peak_price,
        drawdown_pct=drawdown_pct,
        confidence=confidence,
        confidence_baseline=confidence_baseline,
        confidence_change=confidence_change,
        hmm_state=hmm_state,
        hmm_state_baseline=hmm_state_baseline,
        hmm_state_change=hmm_state_change,
        volatility=atr_pct,
        volatility_state=volatility_state,
        atr=current_atr,
        atr_pct=atr_pct,
        trend_state=pos.get("strategy"),
        holding_period=_days_held(entry_time),
        portfolio_exposure=portfolio_exposure,
        sector_exposure=sector_exposure,
        available_cash=available_cash,
        existing_risk_flags=list(risk_flags or []),
    )


def evaluate(context: PositionManagementContext) -> PositionManagerDecision:
    """Pure aggregation — see module docstring for the multiplicative
    combination rationale and the baseline_qty-vs-current_qty rationale.
    Never raises on a well-formed context. action is always HOLD or REDUCE
    in V3.1-A; target_qty is always in [0, current_position]."""
    if context.current_position <= 0 or context.baseline_qty <= 0:
        return PositionManagerDecision(
            symbol=context.symbol, action=ACTION_HOLD,
            current_position_pct=context.current_position_pct,
            target_position_pct=context.current_position_pct,
            delta_pct=0.0, current_qty=context.current_position,
            target_qty=context.current_position, reduction_qty=0,
            triggered_modules=[], reason="no position to manage",
            module_signals=[])

    signals = [module.evaluate(context) for module in _MODULES]
    triggered = [s for s in signals if s.triggered and s.reduction_pct > 0]

    remaining = 1.0
    for s in triggered:
        remaining *= (1.0 - min(max(s.reduction_pct, 0.0), 1.0))
    remaining = max(0.0, min(1.0, remaining))

    target_pct = remaining * 100.0
    # REDUCE-only: never let aggregation suggest growing above current —
    # that would require PYRAMID's own confirmation logic (V3.1-C, out of
    # scope), not a side effect of reduction signals recovering to 0.
    target_qty = max(0, min(int(round(context.baseline_qty * remaining)),
                             context.current_position))
    reduction_qty = max(0, context.current_position - target_qty)
    reduction_frac = (reduction_qty / context.current_position * 100.0
                       if context.current_position else 0.0)

    if reduction_qty > 0 and reduction_frac >= config.PM_MIN_REDUCTION_PCT:
        action = ACTION_REDUCE
    else:
        # Below the noise floor (config.PM_MIN_REDUCTION_PCT) — HOLD, and
        # report target == current so the log doesn't show a phantom
        # sub-threshold "REDUCE" that was never actually recommended.
        action = ACTION_HOLD
        target_qty = context.current_position
        target_pct = context.current_position_pct
        reduction_qty = 0

    triggered_modules = [s.module for s in triggered]
    reason = ("; ".join(f"{s.module}: {s.reason}" for s in triggered)
              if triggered else "no deterioration signals triggered")

    return PositionManagerDecision(
        symbol=context.symbol,
        action=action,
        current_position_pct=context.current_position_pct,
        target_position_pct=target_pct,
        delta_pct=context.current_position_pct - target_pct,
        current_qty=context.current_position,
        target_qty=target_qty,
        reduction_qty=reduction_qty,
        triggered_modules=triggered_modules,
        reason=reason,
        module_signals=signals,
    )


def log_decision(context: PositionManagementContext,
                  decision: PositionManagerDecision, execution: str) -> None:
    """Best-effort JSONL append — a logging failure must never propagate,
    same contract as regime/__init__.py::_log(). `execution` is tracked
    SEPARATELY from decision.action so a REDUCE decision that was only
    observed (not executed) can never be misread from the log as an
    executed trade — see V3.1 spec's explicit Decision-vs-Execution
    requirement."""
    try:
        POSITION_MANAGER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": context.symbol,
            "current_position": context.current_position,
            "current_position_pct": round(context.current_position_pct, 2),
            "target_position": decision.target_qty,
            "target_position_pct": round(decision.target_position_pct, 2),
            "reduction_amount": decision.reduction_qty,
            "confidence_before": context.confidence_baseline,
            "confidence_after": context.confidence,
            "hmm_before": context.hmm_state_baseline,
            "hmm_after": context.hmm_state,
            "volatility_state": context.volatility_state,
            "drawdown_pct": round(context.drawdown_pct, 4),
            "holding_period_days": context.holding_period,
            "triggered_modules": decision.triggered_modules,
            "final_decision": decision.action,
            "reason": decision.reason,
            "execution": execution,
            # Reserved for SHADOW/ACTIVE — always inert placeholders while
            # config.POSITION_LLM_MODE == "OFF" (V3.1-A's only supported value).
            "llm_enabled": config.POSITION_LLM_MODE != "OFF",
            "llm_mode": config.POSITION_LLM_MODE,
            "llm_decision": None,
            "llm_confidence": None,
            "llm_reason": None,
            "deterministic_vs_llm_difference": None,
        }
        with open(POSITION_MANAGER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        import notify.alert as alert
        alert.log(f"position_manager: failed to write position_manager_log.jsonl — {exc}")

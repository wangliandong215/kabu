# -*- coding: utf-8 -*-
"""
position_manager/models.py — data structures for V3.1-A Position Manager
(持仓期间动态仓位管理). See position_manager/__init__.py for the overall
architecture and position_manager/llm_interface.py for the LLM Review Layer
these same shapes are reserved for.

PositionManagementContext is the single stable shape every V3.1 sub-module
(confidence/hmm/volatility/drawdown reduction) reads from, and the one the
future local LLM Review Layer will read from too — see V3.1 spec section
十二. Nothing here is re-derived by a sub-module; position_manager/
__init__.py::build_context() computes every field once per symbol per
engine/runner.py::run_once() pass.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ACTION_HOLD = "HOLD"
ACTION_REDUCE = "REDUCE"
# Reserved for later phases (V3.1-B/C) — never produced by V3.1-A.
ACTION_TAKE_PROFIT = "TAKE_PROFIT"
ACTION_PYRAMID = "PYRAMID"
ACTION_EXIT = "EXIT"

VALID_ACTIONS = (ACTION_HOLD, ACTION_REDUCE, ACTION_TAKE_PROFIT,
                  ACTION_PYRAMID, ACTION_EXIT)

VOLATILITY_NORMAL = "NORMAL"
VOLATILITY_ELEVATED = "ELEVATED"
VOLATILITY_HIGH = "HIGH"


@dataclass
class PositionManagementContext:
    """Everything V3.1's reduction modules — and later the LLM Review Layer
    — need about one held position, for one run_once() pass.

    `baseline_qty` is not in the V3.1 spec's literal field list but is a
    necessary internal reference point: the largest confirmed size this
    holding period has legitimately reached (entry + any pyramid adds),
    monotonically non-decreasing and immune to Position Manager's own past
    REDUCE actions — see position_manager/state_store.py. target_position is
    always computed against this, never against current_position directly,
    which is what makes a repeated REDUCE (same deteriorated signal, cycle
    after cycle) resolve to HOLD once the position is already at target,
    instead of compounding further every pass.

    `confidence_baseline`/`hmm_state_baseline` are likewise not in the
    spec's literal list but are what confidence_change/hmm_state_change are
    computed against — the value first observed by Position Manager for
    this holding period (see state_store's entry_time-keyed reset)."""
    symbol: str

    current_position: int          # shares, live from Portfolio
    current_position_pct: float    # current_position / baseline_qty * 100
    baseline_qty: int

    target_position: int           # filled in after aggregation; == current_position until evaluate() runs
    target_position_pct: float

    entry_price: float
    current_price: float

    unrealized_pnl: float
    unrealized_pnl_pct: float

    peak_price: float
    drawdown_pct: float            # 0..1, (peak_price - current_price) / peak_price

    confidence: Optional[float]            # 0-100, this pass
    confidence_baseline: Optional[float]   # 0-100, first observed this holding period
    confidence_change: Optional[float]     # confidence - confidence_baseline, signed

    hmm_state: Optional[str]
    hmm_state_baseline: Optional[str]
    hmm_state_change: bool                 # hmm_state != hmm_state_baseline

    volatility: Optional[float]        # atr_pct, 0..1
    volatility_state: str              # NORMAL / ELEVATED / HIGH

    atr: Optional[float]
    atr_pct: Optional[float]

    trend_state: Optional[str]         # entry strategy name (pos["strategy"])

    holding_period: int                # calendar days since entry

    portfolio_exposure: Optional[float]   # 0..1
    sector_exposure: Optional[float]      # 0..1

    available_cash: float

    existing_risk_flags: List[str] = field(default_factory=list)
    triggered_position_management_modules: List[str] = field(default_factory=list)


@dataclass
class ModuleSignal:
    """One sub-module's (confidence/hmm/volatility/drawdown) output — pure
    data, consumed only by position_manager/__init__.py::evaluate(). A
    module that doesn't trigger returns reduction_pct=0.0/triggered=False,
    never None, so the aggregator never special-cases "did this run"."""
    module: str
    reduction_pct: float     # 0..1 — how much to shrink target by, this module's opinion
    triggered: bool
    reason: str


@dataclass
class PositionManagerDecision:
    """position_manager/__init__.py::evaluate()'s output — the deterministic,
    LLM-independent decision. `action` is HOLD or REDUCE in V3.1-A; the other
    VALID_ACTIONS values are reserved for later phases and never produced
    here. reduction_qty is always >= 0 and target_position is always <=
    current_position — REDUCE-only, matching the spec's "Position Manager
    绝不能自己PYRAMID/EXIT" constraint for this phase."""
    symbol: str
    action: str
    current_position_pct: float
    target_position_pct: float
    delta_pct: float
    current_qty: int
    target_qty: int
    reduction_qty: int
    triggered_modules: List[str]
    reason: str
    module_signals: List[ModuleSignal] = field(default_factory=list)

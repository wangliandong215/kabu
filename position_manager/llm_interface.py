# -*- coding: utf-8 -*-
"""
position_manager/llm_interface.py — LLM Advisory / Review Layer interface
for V3.1 Position Manager. Reserved now, NOT wired to any real model in
V3.1-A — mirrors regime/llm_provider.py + regime/factory.py's existing
placeholder pattern (same author, same project, days earlier) exactly, so
the shape is proven and consistent rather than invented fresh here.

Architecture this locks in (V3.1 spec section 十一/十七):

    V3.1 Position Manager (deterministic)
            |
    Deterministic Target Position
            |
    LLM Review / Advisory  <-- this module, always optional
            |
    Final Risk Guard   (unchanged, in engine/runner.py / risk/guard.py)
            |
    Order Executor

The LLM NEVER places an order, NEVER decides position size directly, and
NEVER bypasses Risk Guard / Exposure Limit / Sector Limit / Capacity /
Maximum Position — review_position() only ever returns advisory data that a
future ACTIVE-mode integration could choose to fold into the deterministic
decision; V3.1-A does not do that folding anywhere.

Three modes (config.POSITION_LLM_MODE):
  OFF     — get_llm_reviewer() returns None. review_position() short-
            circuits before importing/constructing anything LLM-related.
            The only mode this codebase runs in today.
  SHADOW  — (future) call the LLM, record deterministic vs. LLM decision,
            never let it affect the real decision.
  ACTIVE  — (future, requires a separate tested rollout) LLM recommendation
            may inform the decision, still behind Risk Guard.

In this phase, SHADOW/ACTIVE both resolve to _PlaceholderReviewer, whose
review() always raises NotImplementedError — exactly regime/llm_provider.py
::LocalLLMRegimeProvider's placeholder contract. review_position() catches
that (and any other failure: timeout, invalid JSON, model unavailable,
crash) and returns None, so a stray config change to SHADOW/ACTIVE can
never break a trading pass — see module docstring's "LLM must support OFF"
requirement (V3.1 spec section十四).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import config
from position_manager.models import PositionManagementContext, PositionManagerDecision

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
MODE_ACTIVE = "ACTIVE"
VALID_MODES = (MODE_OFF, MODE_SHADOW, MODE_ACTIVE)

ACTION_HOLD = "HOLD"
ACTION_REDUCE = "REDUCE"
ACTION_STRONG_REDUCE = "STRONG_REDUCE"
ACTION_BLOCK_PYRAMID = "BLOCK_PYRAMID"
ACTION_ALLOW_PYRAMID = "ALLOW_PYRAMID"


@dataclass
class LLMPositionReview:
    """Advisory-only output contract — see module docstring. suggested_
    reduction is a suggestion only; any real decision still has to go back
    through position_manager/__init__.py::evaluate() + Risk Guard, never
    applied directly."""
    action: str
    confidence: float             # 0..1
    reason: str
    suggested_reduction: Optional[float] = None   # 0..1, advisory only
    risk_flags: List[str] = field(default_factory=list)
    model: str = "none"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class _PlaceholderReviewer:
    """No local LLM is implemented, downloaded, or invoked anywhere in this
    codebase in this phase. review() always raises NotImplementedError so
    this class can never silently fabricate an LLMPositionReview — same
    contract as regime/llm_provider.py::LocalLLMRegimeProvider."""

    def review(self, context: PositionManagementContext,
               decision: PositionManagerDecision) -> LLMPositionReview:
        raise NotImplementedError(
            "position_manager LLM Review Layer is a placeholder — no local "
            "LLM is wired in yet. config.POSITION_LLM_MODE must stay 'OFF' "
            "until a real implementation replaces this class's review() "
            "method."
        )


def get_llm_reviewer(mode: Optional[str] = None):
    """mode=OFF (default) -> None, zero overhead, no LLM-related import or
    construction at all. mode=SHADOW/ACTIVE -> a placeholder reviewer whose
    .review() always raises — see _PlaceholderReviewer."""
    mode = mode if mode is not None else config.POSITION_LLM_MODE
    if mode == MODE_OFF:
        return None
    return _PlaceholderReviewer()


def review_position(context: PositionManagementContext,
                     decision: PositionManagerDecision,
                     mode: Optional[str] = None) -> Optional[LLMPositionReview]:
    """The only call site position_manager/__init__.py (or, later,
    engine/runner.py) should ever use. Never raises, never blocks: any
    failure (NotImplementedError today; timeout/invalid JSON/model crash in
    the future) is caught and logged, returning None — the deterministic
    `decision` is always what actually governs behavior in V3.1-A. Returns
    None immediately, with no reviewer constructed, when mode is OFF."""
    import notify.alert as alert

    mode = mode if mode is not None else config.POSITION_LLM_MODE
    reviewer = get_llm_reviewer(mode)
    if reviewer is None:
        return None

    try:
        return reviewer.review(context, decision)
    except Exception as exc:
        alert.log(f"position_manager.llm_interface: review failed for "
                  f"{context.symbol} ({exc.__class__.__name__}: {exc}) — "
                  f"falling back to deterministic decision only")
        return None

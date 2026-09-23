# -*- coding: utf-8 -*-
"""
position_manager/hmm_reduction.py — V3.1-A HMM State Change Reduction.

Reads the per-symbol HMM state engine/regime_store.get_regime_interface()
already durably tracks (HMM_BULL/HMM_SIDEWAYS/HMM_CORRECTION/HMM_BEAR — the
same four-state label set engine/confidence_score.py's _HMM_BASE_SCORE
uses). A state transition NEVER means EXIT here — it only ever produces a
reduction recommendation, which position_manager/__init__.py then combines
with the other three modules; only the Position Manager's aggregated
decision decides target position. See V3.1 spec section 三.2.

config.HMM_TRANSITION_REDUCTION maps (from_state, to_state) -> reduction_pct.
A pair not in that table (including same-state, or a transition back toward
a healthier state) reduces by 0 — recovery is not this module's job (that
would be Pyramid, V3.1-C, out of scope here).
"""
import config
from position_manager.models import ModuleSignal, PositionManagementContext

MODULE_NAME = "hmm"


def evaluate(context: PositionManagementContext) -> ModuleSignal:
    before, after = context.hmm_state_baseline, context.hmm_state

    if before is None or after is None:
        return ModuleSignal(MODULE_NAME, 0.0, False,
                             "hmm state unavailable — no opinion")

    if before == after:
        return ModuleSignal(MODULE_NAME, 0.0, False,
                             f"hmm state unchanged ({after})")

    reduction = config.HMM_TRANSITION_REDUCTION.get((before, after), 0.0)
    if reduction <= 0.0:
        return ModuleSignal(
            MODULE_NAME, 0.0, False,
            f"hmm state {before} -> {after} — not a configured downgrade")

    return ModuleSignal(
        MODULE_NAME, reduction, True,
        f"hmm state transition {before} -> {after}")

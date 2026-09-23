# -*- coding: utf-8 -*-
"""
risk/dynamic_sizing.py — V3.0-A Dynamic Position Sizing (动态仓位), Phase A:
Paper / Research Only.

Product decision (V3.0 spec, 2026-09-23): the V2.9 Confidence Score
(engine/confidence_score.py) was built as a pure observation-only layer —
its own module docstring says nothing may read it to gate/scale a trade.
V3.0 deliberately supersedes that ONE constraint: confidence now decides how
much of a base position to take. Nothing else about that module changes —
this file does not alter how confidence_score is computed, only what a
caller does with the number it already produces.

Scope, per V3.0 spec section 三/九/十三 (do not expand without a separate
product decision):
  - This module ONLY maps confidence -> a [0, 1] multiplier. It knows
    nothing about entry/exit rules, strategy scoring, or risk caps.
  - The multiplier is applied via risk/sizing.py::calculate()'s existing
    `position_scale` parameter — a pre-existing "trims, never expands, and
    is applied AFTER qty_by_tier/qty_by_risk/qty_by_strat are already
    min()-ed together" hook (see that function's docstring). Confidence can
    therefore only ever shrink an order that survived every existing risk
    cap — it can never bypass Portfolio Risk Manager, Portfolio Position
    Manager, the real-cash guard, or any other existing gate, because all of
    those run on the qty this module's output already passed through.
  - Phase A (this file's current callers, see engine/runner.py) is
    restricted to paper/simulated execution only — see runner.py's
    `score_env == "paper"` gate. Lifting that to real money is a separate,
    explicit V3.0-B decision, not something this module controls.

Tier table (V3.0 spec section 二, fixed first cut — do not retune without a
separate product decision; see FORMULA_VERSION below for how a future
retune would be tracked):

    Confidence   Multiplier
    95 - 100     1.0
    85 -  94     0.8
    70 -  84     0.6
    55 -  69     0.4
    <   55       0.0   (skip — position_scale=0 flows through
                         risk/sizing.py::calculate() to qty=0, the same
                         "skip this candidate" path any other zero-qty
                         result already takes in engine/runner.py)

Boundaries are inclusive on the LOWER edge of each tier (>=95, >=85, >=70,
>=55) — i.e. a confidence of exactly 95.0 gets the 1.0 multiplier, not the
0.8 one from the tier below.
"""
from typing import Optional

# Bump any time the tier table itself changes, so logged rows (see
# engine/trade_tracker.py's trade_confidence_score sizing columns) can be
# grouped by which multiplier table produced them.
POSITION_SIZING_FORMULA_VERSION = "v3.0.0"

# (lower_bound_inclusive, multiplier) — checked in the order listed. Kept as
# a single ordered table (not a dict) so the "first tier whose lower bound
# confidence clears" logic below is a plain linear scan.
_CONFIDENCE_TIERS = (
    (95.0, 1.0),
    (85.0, 0.8),
    (70.0, 0.6),
    (55.0, 0.4),
)
_SKIP_MULTIPLIER = 0.0


def confidence_to_position_multiplier(confidence: Optional[float]) -> float:
    """Map a V2.9 Confidence Score (engine.confidence_score, always in
    [0, 100] by construction when not None — see that module's docstring)
    to a position-size multiplier in {1.0, 0.8, 0.6, 0.4, 0.0}.

    confidence=None (confidence_score disabled via config, or its own
    gather_and_score() raised — see engine/pipeline.py::build_candidate_pool)
    returns 1.0, NOT 0.0: an unavailable score is "no opinion", not "bad
    signal" — V3.0-A must never silently shrink or drop a trade just because
    one optional observability input failed to compute. This is a deliberate
    fallback, not an oversight; callers should log confidence=None alongside
    multiplier=1.0 so this fallback path stays auditable (see
    engine/runner.py's sizing log).

    Never raises. A confidence outside [0, 100] (should not happen — see
    module docstring) is clipped rather than rejected, matching
    engine/confidence_score.py's own "clip, don't reject" discipline."""
    if confidence is None:
        return 1.0
    c = max(0.0, min(100.0, float(confidence)))
    for lower_bound, multiplier in _CONFIDENCE_TIERS:
        if c >= lower_bound:
            return multiplier
    return _SKIP_MULTIPLIER

# -*- coding: utf-8 -*-
"""
exit_engine/pressure_score.py — combines every V3.2-A SignalReading into
one Exit Pressure Score + tier.

This is an explicit v1 formula — a simple weighted sum of independently-
enabled signals mapped to 4 fixed tiers (LOW/MEDIUM/HIGH/CRITICAL) — NOT
yet empirically validated. Per the V3.2 spec's development order
(section 八), V3.2-B/C is where the per-signal weights and these tier
thresholds get checked against actual Profit Giveback / Premature-Exit
outcomes; this file is deliberately not tuned for effect yet, only for a
sane, monotonic, always-defined score. See config.py's V3.2-A section for
every weight/threshold constant used here and by the signal modules.
"""
from typing import List, Tuple

import config
from exit_engine.models import TIER_CRITICAL, TIER_HIGH, TIER_LOW, TIER_MEDIUM, SignalReading


def tier_for_score(score: float) -> str:
    if score >= config.EXIT_PRESSURE_CRITICAL_THRESHOLD:
        return TIER_CRITICAL
    if score >= config.EXIT_PRESSURE_HIGH_THRESHOLD:
        return TIER_HIGH
    if score >= config.EXIT_PRESSURE_MEDIUM_THRESHOLD:
        return TIER_MEDIUM
    return TIER_LOW


def aggregate(signals: List[SignalReading]) -> Tuple[float, str, float]:
    """Returns (pressure_score, pressure_tier, exit_confidence).

    pressure_score is the sum of every ENABLED signal's points (a
    disabled signal contributes nothing — this is what makes flipping a
    single ENABLE_* flag off in V3.2-B a clean ablation, not a partial
    one). exit_confidence is the fraction of enabled signals that
    triggered — 0.0 when no signal is enabled at all (nothing to be
    confident about), never a division by zero."""
    enabled = [s for s in signals if s.enabled]
    score = sum(s.points for s in enabled)
    triggered_count = sum(1 for s in enabled if s.triggered)
    confidence = (triggered_count / len(enabled)) if enabled else 0.0
    tier = tier_for_score(score)
    return score, tier, confidence

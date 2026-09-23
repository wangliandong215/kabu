# -*- coding: utf-8 -*-
"""
position_manager/confidence_reduction.py — V3.1-A Confidence Change
Reduction.

Not "confidence < X -> SELL". The decision is driven by current value +
baseline value + magnitude of change together (V3.1 spec section 三.1):
a small drift (e.g. 0.82 -> 0.78) should not trigger anything, while a
sharp deterioration (0.82 -> 0.55) should. Thresholds are configurable
(config.CONF_DROP_MINOR/MODERATE/SEVERE) and do not touch
engine/confidence_score.py's own weights/formula in any way.

Pure function of PositionManagementContext — no I/O, easy to unit test
directly with a constructed context.
"""
import config
from position_manager.models import ModuleSignal, PositionManagementContext

MODULE_NAME = "confidence"


def evaluate(context: PositionManagementContext) -> ModuleSignal:
    change = context.confidence_change

    if change is None or context.confidence_baseline is None:
        return ModuleSignal(MODULE_NAME, 0.0, False,
                             "confidence baseline unavailable — no opinion")

    drop = -change   # positive = deterioration
    if drop < config.CONF_DROP_MINOR:
        return ModuleSignal(
            MODULE_NAME, 0.0, False,
            f"confidence {context.confidence_baseline:.1f} -> "
            f"{context.confidence:.1f} (Δ{change:+.1f}) — within noise band")

    if drop >= config.CONF_DROP_SEVERE:
        reduction = config.CONF_DROP_SEVERE_REDUCTION
        label = "severe deterioration"
    elif drop >= config.CONF_DROP_MODERATE:
        reduction = config.CONF_DROP_MODERATE_REDUCTION
        label = "moderate deterioration"
    else:
        # Between MINOR and MODERATE: linearly interpolate a small reduction
        # rather than jumping straight to the MODERATE tier's full cut, so
        # the boundary at MINOR isn't a step discontinuity.
        span = max(config.CONF_DROP_MODERATE - config.CONF_DROP_MINOR, 1e-9)
        frac = (drop - config.CONF_DROP_MINOR) / span
        reduction = config.CONF_DROP_MODERATE_REDUCTION * frac
        label = "mild deterioration"

    return ModuleSignal(
        MODULE_NAME, reduction, True,
        f"confidence {context.confidence_baseline:.1f} -> "
        f"{context.confidence:.1f} (Δ{change:+.1f}, {label})")

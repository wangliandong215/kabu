# -*- coding: utf-8 -*-
"""
position_manager/volatility_reduction.py — V3.1-A Volatility Reduction.

Not "high volatility -> EXIT" — a graded reduction as atr_pct (ATR / current
price, the same ratio engine/confidence_score.py's volatility component and
the exit loop's trailing-stop math already use — no new ATR implementation
introduced here, see engine/runner.py's exit loop / risk/guard.py for the
existing source) rises through NORMAL -> ELEVATED -> HIGH bands. Entirely
independent of risk/guard.py's stop-loss/take-profit/ATR trailing stop,
which is unchanged. See V3.1 spec section 三.3.
"""
import config
from position_manager.models import (ModuleSignal, PositionManagementContext,
                                       VOLATILITY_ELEVATED, VOLATILITY_HIGH,
                                       VOLATILITY_NORMAL)

MODULE_NAME = "volatility"


def classify(atr_pct: float) -> str:
    if atr_pct >= config.VOL_HIGH_ATR_PCT:
        return VOLATILITY_HIGH
    if atr_pct >= config.VOL_ELEVATED_ATR_PCT:
        return VOLATILITY_ELEVATED
    return VOLATILITY_NORMAL


def evaluate(context: PositionManagementContext) -> ModuleSignal:
    if context.atr_pct is None:
        return ModuleSignal(MODULE_NAME, 0.0, False,
                             "atr_pct unavailable — no opinion")

    state = context.volatility_state
    if state == VOLATILITY_HIGH:
        return ModuleSignal(
            MODULE_NAME, config.VOL_HIGH_REDUCTION, True,
            f"volatility HIGH (atr_pct={context.atr_pct:.2%})")
    if state == VOLATILITY_ELEVATED:
        return ModuleSignal(
            MODULE_NAME, config.VOL_ELEVATED_REDUCTION, True,
            f"volatility ELEVATED (atr_pct={context.atr_pct:.2%})")
    return ModuleSignal(
        MODULE_NAME, 0.0, False,
        f"volatility NORMAL (atr_pct={context.atr_pct:.2%})")

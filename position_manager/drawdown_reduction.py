# -*- coding: utf-8 -*-
"""
position_manager/drawdown_reduction.py — V3.1-A Drawdown Reduction.

drawdown_pct = (peak_price - current_price) / peak_price, where peak_price
is tracked independently in position_manager/state_store.py (its own
per-symbol high-water mark since this holding period began) — this is a
brand-new Position Management layer, deliberately NOT the same thing as
risk/guard.py's Stop Loss / Take Profit / ATR trailing stop (pos["trail_stop"]
etc.), which stays completely untouched. See V3.1 spec section 三.4.

Graded like the other three modules — deeper drawdown tiers produce larger
(but still bounded) reduction recommendations, never a direct EXIT.
"""
import config
from position_manager.models import ModuleSignal, PositionManagementContext

MODULE_NAME = "drawdown"


def evaluate(context: PositionManagementContext) -> ModuleSignal:
    dd = context.drawdown_pct

    if dd >= config.DRAWDOWN_TIER3_PCT:
        return ModuleSignal(
            MODULE_NAME, config.DRAWDOWN_TIER3_REDUCTION, True,
            f"drawdown {dd:.1%} from peak {context.peak_price:.2f} (tier 3)")
    if dd >= config.DRAWDOWN_TIER2_PCT:
        return ModuleSignal(
            MODULE_NAME, config.DRAWDOWN_TIER2_REDUCTION, True,
            f"drawdown {dd:.1%} from peak {context.peak_price:.2f} (tier 2)")
    if dd >= config.DRAWDOWN_TIER1_PCT:
        return ModuleSignal(
            MODULE_NAME, config.DRAWDOWN_TIER1_REDUCTION, True,
            f"drawdown {dd:.1%} from peak {context.peak_price:.2f} (tier 1)")

    return ModuleSignal(
        MODULE_NAME, 0.0, False,
        f"drawdown {dd:.1%} from peak {context.peak_price:.2f} — within tier 1 band")

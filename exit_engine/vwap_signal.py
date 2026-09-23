# -*- coding: utf-8 -*-
"""
exit_engine/vwap_signal.py — V3.2-A VWAP signal.

Is price below the current session's VWAP, and how far? Uses
context.session_bars_1m (VWAP resets every session — see
exit_engine/__init__.py::_session_bars docstring), never context.bars_1m
directly. Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine import indicators
from exit_engine.models import SignalReading

MODULE_NAME = "vwap"

STATE_ABOVE = "ABOVE_VWAP"
STATE_LOST = "LOST_VWAP"
STATE_LOST_WIDE = "LOST_VWAP_WIDE"
STATE_UNKNOWN = "UNKNOWN"


def evaluate(context) -> SignalReading:
    if not config.ENABLE_VWAP_EXIT:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    vwap_now = indicators.compute_vwap(context.session_bars_1m)
    if vwap_now is None or vwap_now <= 0 or context.current_price <= 0:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "insufficient session data for VWAP", state=STATE_UNKNOWN)

    distance_pct = (context.current_price - vwap_now) / vwap_now
    below = context.current_price < vwap_now

    if not below:
        return SignalReading(
            MODULE_NAME, True, False, 0.0,
            f"price {context.current_price:.4f} above VWAP {vwap_now:.4f} ({distance_pct:+.2%})",
            state=STATE_ABOVE, extra={"vwap": vwap_now, "distance_pct": distance_pct})

    points = config.EXIT_VWAP_POINTS_BASE
    state = STATE_LOST
    if distance_pct <= -config.EXIT_VWAP_WIDE_DISTANCE_PCT:
        points += config.EXIT_VWAP_POINTS_WIDE
        state = STATE_LOST_WIDE

    return SignalReading(
        MODULE_NAME, True, True, points,
        f"price {context.current_price:.4f} below VWAP {vwap_now:.4f} ({distance_pct:+.2%})",
        state=state, extra={"vwap": vwap_now, "distance_pct": distance_pct})

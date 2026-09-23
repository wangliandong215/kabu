# -*- coding: utf-8 -*-
"""
exit_engine/atr_signal.py — V3.2-A ATR-based Stop / Volatility Expansion
signal.

Two independent checks:
  ATR stop breach     — price has fallen more than EXIT_ATR_STOP_MULT×ATR
                         below the position's peak price (an ATR-scaled
                         protective stop, not a fixed percentage — avoids
                         the "one stop distance fits every stock" trap the
                         V3.2 spec explicitly calls out).
  Volatility expansion — current ATR has grown well past the ATR observed
                         at entry (the stock has gotten meaningfully
                         choppier since the position was opened).

Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine.models import SignalReading

MODULE_NAME = "atr"

STATE_NORMAL = "NORMAL"
STATE_STOP_BREACH = "ATR_STOP_BREACH"
STATE_EXPANSION = "VOLATILITY_EXPANSION"
STATE_UNKNOWN = "UNKNOWN"


def evaluate(context) -> SignalReading:
    if not config.ENABLE_ATR_EXIT:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    atr = context.current_atr
    if not atr or atr <= 0 or context.current_price <= 0 or context.peak_price <= 0:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "ATR or price unavailable", state=STATE_UNKNOWN)

    stop_price = context.peak_price - config.EXIT_ATR_STOP_MULT * atr
    breached = context.current_price < stop_price

    expansion = None
    if context.entry_atr and context.entry_atr > 0:
        expansion = atr / context.entry_atr
    expanded = expansion is not None and expansion >= config.EXIT_ATR_EXPANSION_RATIO

    points = 0.0
    states = []
    if breached:
        points += config.EXIT_ATR_STOP_POINTS
        states.append(STATE_STOP_BREACH)
    if expanded:
        points += config.EXIT_ATR_EXPANSION_POINTS
        states.append(STATE_EXPANSION)

    triggered = points > 0
    state = "+".join(states) if states else STATE_NORMAL
    detail = (f"price {context.current_price:.4f} vs atr_stop {stop_price:.4f} "
              f"(peak {context.peak_price:.4f}, atr {atr:.4f}"
              + (f", expansion {expansion:.2f}x)" if expansion is not None else ")"))

    return SignalReading(MODULE_NAME, True, triggered, points, detail, state=state,
                          extra={"atr_stop_price": stop_price, "volatility_expansion": expansion})

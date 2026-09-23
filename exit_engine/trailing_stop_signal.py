# -*- coding: utf-8 -*-
"""
exit_engine/trailing_stop_signal.py — V3.2-A Trailing Stop signal.

Computes three candidate hypothetical trailing-stop levels off the
position's peak price — fixed percentage, ATR-based, and profit-based
(only activates once peak gain has cleared a threshold, then locks in a
fraction of it) — and uses whichever is HIGHEST (i.e. tightest/most
protective) as "the" trailing stop this pass. This is what feeds
exit_engine's potential_exit_price when triggered — see
exit_engine/__init__.py::evaluate().

Purely a hypothetical/diagnostic level in V3.2-A: it never becomes a real
order, and it does NOT read or write risk/guard.py's own ATR trailing
stop (pos["trail_stop"]) — two independent numbers on purpose, so this
package can never accidentally perturb the real exit path.

Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine.models import SignalReading

MODULE_NAME = "trailing_stop"

STATE_ABOVE = "ABOVE_TRAILING_STOP"
STATE_BELOW = "BELOW_TRAILING_STOP"
STATE_UNKNOWN = "UNKNOWN"


def evaluate(context) -> SignalReading:
    if not config.ENABLE_TRAILING_STOP:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    if context.peak_price <= 0:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "peak price unavailable", state=STATE_UNKNOWN)

    candidates = {"fixed": context.peak_price * (1.0 - config.EXIT_TRAILING_FIXED_PCT)}

    if context.current_atr and context.current_atr > 0:
        candidates["atr"] = context.peak_price - config.EXIT_TRAILING_ATR_MULT * context.current_atr

    if context.entry_price > 0:
        gain_pct = (context.peak_price - context.entry_price) / context.entry_price
        if gain_pct >= config.EXIT_TRAILING_PROFIT_ACTIVATION_PCT:
            locked_gain_pct = gain_pct * config.EXIT_TRAILING_PROFIT_LOCK_FRACTION
            candidates["profit"] = context.entry_price * (1.0 + locked_gain_pct)

    stop_price = max(candidates.values())
    triggered = context.current_price < stop_price
    points = config.EXIT_TRAILING_STOP_POINTS if triggered else 0.0
    state = STATE_BELOW if triggered else STATE_ABOVE
    detail = (f"price {context.current_price:.4f} vs trailing_stop {stop_price:.4f} "
              f"(candidates={ {k: round(v, 4) for k, v in candidates.items()} })")

    return SignalReading(MODULE_NAME, True, triggered, points, detail, state=state,
                          extra={"trailing_stop_price": stop_price, "candidates": candidates})

# -*- coding: utf-8 -*-
"""
exit_engine/break_even_signal.py — V3.2-A Break Even signal.

Once the position's peak gain has cleared EXIT_BREAK_EVEN_ACTIVATION_PCT,
this signal watches for the price pulling back to at/near the entry cost
(within EXIT_BREAK_EVEN_TOLERANCE_PCT) — a trade that built real profit
and is now giving essentially all of it back. Never triggers before
activation: an ordinary small pullback on a position that was never
meaningfully profitable is not a "gave back its gain" story (see
engine/exit_diagnostics.py's EARLY_FAILURE/TREND_REVERSAL distinction for
the same idea applied post-hoc).

Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine.models import SignalReading

MODULE_NAME = "break_even"

STATE_NOT_ACTIVE = "NOT_ACTIVE"
STATE_PROTECTED = "PROTECTED"
STATE_AT_BREAK_EVEN = "AT_BREAK_EVEN"


def evaluate(context) -> SignalReading:
    if not config.ENABLE_BREAK_EVEN:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    if context.entry_price <= 0:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "entry price unavailable", state=STATE_NOT_ACTIVE)

    peak_gain_pct = (context.peak_price - context.entry_price) / context.entry_price
    current_gain_pct = (context.current_price - context.entry_price) / context.entry_price

    if peak_gain_pct < config.EXIT_BREAK_EVEN_ACTIVATION_PCT:
        return SignalReading(
            MODULE_NAME, True, False, 0.0,
            f"peak gain {peak_gain_pct:+.2%} below activation threshold",
            state=STATE_NOT_ACTIVE,
            extra={"peak_gain_pct": peak_gain_pct, "current_gain_pct": current_gain_pct})

    triggered = current_gain_pct <= config.EXIT_BREAK_EVEN_TOLERANCE_PCT
    points = config.EXIT_BREAK_EVEN_POINTS if triggered else 0.0
    state = STATE_AT_BREAK_EVEN if triggered else STATE_PROTECTED
    detail = f"peak_gain {peak_gain_pct:+.2%}, current_gain {current_gain_pct:+.2%}"

    return SignalReading(MODULE_NAME, True, triggered, points, detail, state=state,
                          extra={"peak_gain_pct": peak_gain_pct, "current_gain_pct": current_gain_pct})

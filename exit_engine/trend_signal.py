# -*- coding: utf-8 -*-
"""
exit_engine/trend_signal.py — V3.2-A Multi-Timeframe Trend signal.

Each of 1m/5m/15m trend direction is independently gated by its own
config flag (ENABLE_1M_TREND_EXIT / ENABLE_5M_TREND_EXIT /
ENABLE_15M_TREND_EXIT), so V3.2-B can validate one timeframe at a time.
Per the V3.2 spec's explicit example, "1m Down" alone is treated as
weaker evidence than "1m Down + 5m Down", which is weaker than all three
agreeing — EXIT_TREND_COMPOUND_BONUS/EXIT_TREND_ALL_DOWN_BONUS add extra
weight on top of the flat per-timeframe point for exactly that reason,
not because any one timeframe's DOWN read is worth more than another's.

Uses the raw (non-session-filtered) bars — trend is continuous, not a
session-reset statistic like VWAP. Pure function, never raises.
"""
import config
from exit_engine import indicators
from exit_engine.models import SignalReading

MODULE_NAME = "trend"

_TIMEFRAMES = (
    ("1m", "ENABLE_1M_TREND_EXIT", "bars_1m"),
    ("5m", "ENABLE_5M_TREND_EXIT", "bars_5m"),
    ("15m", "ENABLE_15M_TREND_EXIT", "bars_15m"),
)


def evaluate(context) -> SignalReading:
    readings = {}
    for label, flag_name, bars_attr in _TIMEFRAMES:
        if not getattr(config, flag_name):
            continue
        df = getattr(context, bars_attr)
        readings[label] = indicators.trend_direction(
            df, config.EXIT_TREND_FAST_SPAN, config.EXIT_TREND_SLOW_SPAN)

    if not readings:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    known = {k: v for k, v in readings.items() if v is not None}
    if not known:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "insufficient bars on every enabled timeframe", state="UNKNOWN")

    down_count = sum(1 for v in known.values() if v == "DOWN")
    points = float(down_count)
    if down_count >= 2:
        points += config.EXIT_TREND_COMPOUND_BONUS
    if len(readings) == 3 and down_count == 3:
        points += config.EXIT_TREND_ALL_DOWN_BONUS

    triggered = down_count > 0
    state = "+".join(f"{k}:{v}" for k, v in readings.items() if v is not None) or "UNKNOWN"
    detail = f"{down_count}/{len(known)} enabled timeframe(s) DOWN ({state})"

    return SignalReading(MODULE_NAME, True, triggered, points, detail, state=state,
                          extra=dict(readings))

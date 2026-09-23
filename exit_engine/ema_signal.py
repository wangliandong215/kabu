# -*- coding: utf-8 -*-
"""
exit_engine/ema_signal.py — V3.2-A EMA Cross / Trend signal.

Fast-vs-slow EMA bearish cross, or price below both, on context.bars_5m.
EMA is continuous (not session-reset) — uses the raw multi-day 5m window,
not session_bars_1m. Pure function of ExitEngineContext, never raises.
"""
import config
from exit_engine import indicators
from exit_engine.models import SignalReading

MODULE_NAME = "ema"

STATE_BULLISH = "BULLISH"
STATE_BELOW_EMA = "BELOW_EMA"
STATE_BEARISH_CROSS = "BEARISH_CROSS"
STATE_UNKNOWN = "UNKNOWN"


def evaluate(context) -> SignalReading:
    if not config.ENABLE_EMA_EXIT:
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    df = context.bars_5m
    close = df["close"].astype(float) if df is not None and "close" in df else None

    fast = indicators.ema_series(close, config.EXIT_EMA_FAST_SPAN) if close is not None else None
    slow = indicators.ema_series(close, config.EXIT_EMA_SLOW_SPAN) if close is not None else None
    if fast is None or slow is None or len(fast) < 2 or len(slow) < 2:
        return SignalReading(MODULE_NAME, True, False, 0.0,
                              "insufficient bars for EMA", state=STATE_UNKNOWN)

    fast_now, fast_prev = fast.iloc[-1], fast.iloc[-2]
    slow_now, slow_prev = slow.iloc[-1], slow.iloc[-2]

    bearish_cross = fast_prev >= slow_prev and fast_now < slow_now
    price_below_both = context.current_price < fast_now and context.current_price < slow_now

    detail = (f"EMA{config.EXIT_EMA_FAST_SPAN}={fast_now:.4f} "
              f"EMA{config.EXIT_EMA_SLOW_SPAN}={slow_now:.4f} price={context.current_price:.4f}")

    if bearish_cross:
        return SignalReading(MODULE_NAME, True, True, config.EXIT_EMA_CROSS_POINTS,
                              detail, state=STATE_BEARISH_CROSS,
                              extra={"ema_fast": float(fast_now), "ema_slow": float(slow_now)})
    if price_below_both:
        return SignalReading(MODULE_NAME, True, True, config.EXIT_EMA_BELOW_POINTS,
                              detail, state=STATE_BELOW_EMA,
                              extra={"ema_fast": float(fast_now), "ema_slow": float(slow_now)})
    return SignalReading(MODULE_NAME, True, False, 0.0, detail, state=STATE_BULLISH,
                          extra={"ema_fast": float(fast_now), "ema_slow": float(slow_now)})

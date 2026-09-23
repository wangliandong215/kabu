# -*- coding: utf-8 -*-
"""
exit_engine/time_signal.py — V3.2-A time-based signals: Opening Weakness,
Afternoon Fade, Time Exit.

Opening Weakness / Afternoon Fade work off context.session_bars_1m (a
session-reset statistic, like VWAP — see exit_engine/__init__.py's
_session_bars docstring), using BAR POSITION within the session (row
count) as a proxy for minutes-since-open instead of wall-clock/timezone
math — robust across JP/US codes and trivially testable with synthetic
bars, at the cost of assuming ~1 row per minute (true for "1m" bars on a
normal trading day; degrades gracefully to "not enough data yet" on a
thin/gappy session rather than misfiring).

Time Exit works off context.holding_days and the peak-price high-water
mark's peak_date (from exit_engine/state_store.py) — "held too long
without the expected gain" and "too long since the last new high" are
independent triggers, either one is enough.

Each of the three checks is independently gated by its own config flag.
Pure function of ExitEngineContext, never raises.
"""
from datetime import date

import config
from exit_engine.models import SignalReading

MODULE_NAME = "time"

STATE_OPENING_WEAKNESS = "OPENING_WEAKNESS"
STATE_AFTERNOON_FADE = "AFTERNOON_FADE"
STATE_TIME_EXIT = "TIME_EXIT"
STATE_NONE = "NONE"


def _opening_weakness(session_df):
    """None = not enough data to judge yet (not yet a "no" verdict)."""
    if (session_df is None or "low" not in session_df
            or len(session_df) < config.EXIT_OPENING_WINDOW_MINUTES):
        return None
    opening = session_df.iloc[:config.EXIT_OPENING_WINDOW_MINUTES]
    opening_low = float(opening["low"].astype(float).min())
    current_price = float(session_df["close"].astype(float).iloc[-1])
    triggered = current_price < opening_low
    return triggered, f"price {current_price:.4f} vs opening_low {opening_low:.4f}"


def _afternoon_fade(session_df):
    if (session_df is None or "high" not in session_df
            or len(session_df) < config.EXIT_AFTERNOON_START_MINUTES):
        return None
    early = session_df.iloc[:config.EXIT_AFTERNOON_START_MINUTES]
    session_high = float(early["high"].astype(float).max())
    if session_high <= 0:
        return None
    current_price = float(session_df["close"].astype(float).iloc[-1])
    drawdown = (session_high - current_price) / session_high
    triggered = drawdown >= config.EXIT_AFTERNOON_FADE_DRAWDOWN_PCT
    return triggered, f"session_high {session_high:.4f} drawdown {drawdown:+.2%}"


def _time_exit(context):
    reasons = []
    triggered = False

    if context.holding_days >= config.EXIT_TIME_MAX_HOLDING_DAYS and context.entry_price > 0:
        gain_pct = (context.current_price - context.entry_price) / context.entry_price
        if gain_pct < config.EXIT_TIME_MIN_EXPECTED_GAIN_PCT:
            triggered = True
            reasons.append(f"held {context.holding_days:.1f}d, gain {gain_pct:+.2%} < expected")

    if context.peak_date:
        try:
            peak_d = date.fromisoformat(context.peak_date)
            days_since_peak = (context.now.date() - peak_d).days
            if days_since_peak >= config.EXIT_TIME_NO_NEW_HIGH_DAYS:
                triggered = True
                reasons.append(f"{days_since_peak}d since last new high")
        except (ValueError, TypeError):
            pass

    detail = "; ".join(reasons) if reasons else "within normal holding window"
    return triggered, detail


def evaluate(context) -> SignalReading:
    if not (config.ENABLE_OPENING_WEAKNESS or config.ENABLE_AFTERNOON_FADE
            or config.ENABLE_TIME_EXIT):
        return SignalReading(MODULE_NAME, False, False, 0.0, "disabled")

    points = 0.0
    states = []
    details = []

    if config.ENABLE_OPENING_WEAKNESS:
        result = _opening_weakness(context.session_bars_1m)
        if result is not None:
            triggered, detail = result
            details.append(f"opening_weakness: {detail}")
            if triggered:
                points += config.EXIT_OPENING_WEAKNESS_POINTS
                states.append(STATE_OPENING_WEAKNESS)

    if config.ENABLE_AFTERNOON_FADE:
        result = _afternoon_fade(context.session_bars_1m)
        if result is not None:
            triggered, detail = result
            details.append(f"afternoon_fade: {detail}")
            if triggered:
                points += config.EXIT_AFTERNOON_FADE_POINTS
                states.append(STATE_AFTERNOON_FADE)

    if config.ENABLE_TIME_EXIT:
        triggered, detail = _time_exit(context)
        details.append(f"time_exit: {detail}")
        if triggered:
            points += config.EXIT_TIME_EXIT_POINTS
            states.append(STATE_TIME_EXIT)

    triggered_any = points > 0
    state = "+".join(states) if states else STATE_NONE
    detail = "; ".join(details) if details else "insufficient data for every enabled check"
    return SignalReading(MODULE_NAME, True, triggered_any, points, detail, state=state)

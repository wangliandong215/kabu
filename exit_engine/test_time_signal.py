# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/time_signal.py — Opening Weakness, Afternoon
Fade, Time Exit.

Run:  python -m unittest exit_engine.test_time_signal -v
"""
import unittest
from datetime import datetime

import config
from exit_engine import time_signal
from exit_engine._fixtures import make_bars, make_context
import test_support


class TimeSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "opening": config.ENABLE_OPENING_WEAKNESS,
            "afternoon": config.ENABLE_AFTERNOON_FADE,
            "time_exit": config.ENABLE_TIME_EXIT,
        }
        config.ENABLE_OPENING_WEAKNESS = False
        config.ENABLE_AFTERNOON_FADE = False
        config.ENABLE_TIME_EXIT = False

    def tearDown(self):
        config.ENABLE_OPENING_WEAKNESS = self._orig["opening"]
        config.ENABLE_AFTERNOON_FADE = self._orig["afternoon"]
        config.ENABLE_TIME_EXIT = self._orig["time_exit"]

    def test_all_flags_off_returns_disabled(self):
        ctx = make_context()
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    # ── Opening Weakness ──────────────────────────────────────────────
    def test_opening_weakness_not_enough_session_bars_is_safe(self):
        config.ENABLE_OPENING_WEAKNESS = True
        ctx = make_context(session_bars_1m=make_bars([100.0] * 5))
        sig = time_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)

    def test_opening_weakness_triggers_when_price_breaks_opening_low(self):
        config.ENABLE_OPENING_WEAKNESS = True
        n = config.EXIT_OPENING_WINDOW_MINUTES
        opening_closes = [100.0] * n
        rest = [95.0] * 5   # breaks below the opening range low
        session = make_bars(opening_closes + rest, lows=[99.5] * n + [90.0] * 5)
        ctx = make_context(session_bars_1m=session)
        sig = time_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertIn(time_signal.STATE_OPENING_WEAKNESS, sig.state)

    def test_opening_weakness_no_trigger_when_price_holds_opening_range(self):
        config.ENABLE_OPENING_WEAKNESS = True
        n = config.EXIT_OPENING_WINDOW_MINUTES
        session = make_bars([100.0] * (n + 5), lows=[99.5] * (n + 5))
        ctx = make_context(session_bars_1m=session)
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)

    # ── Afternoon Fade ────────────────────────────────────────────────
    def test_afternoon_fade_not_enough_session_bars_is_safe(self):
        config.ENABLE_AFTERNOON_FADE = True
        ctx = make_context(session_bars_1m=make_bars([100.0] * 10))
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)

    def test_afternoon_fade_triggers_on_late_day_drawdown(self):
        config.ENABLE_AFTERNOON_FADE = True
        n = config.EXIT_AFTERNOON_START_MINUTES
        early = [100.0] * n
        late = [95.0] * 5   # ~5% drawdown from session high, above the 2% threshold
        session = make_bars(early + late, highs=[100.5] * n + [95.5] * 5)
        ctx = make_context(session_bars_1m=session)
        sig = time_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertIn(time_signal.STATE_AFTERNOON_FADE, sig.state)

    # ── Time Exit ─────────────────────────────────────────────────────
    def test_time_exit_triggers_on_long_hold_low_gain(self):
        config.ENABLE_TIME_EXIT = True
        ctx = make_context(
            entry_price=100.0, current_price=100.5,
            holding_days=config.EXIT_TIME_MAX_HOLDING_DAYS + 1,
            peak_date=None)
        sig = time_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertIn(time_signal.STATE_TIME_EXIT, sig.state)

    def test_time_exit_no_trigger_on_short_hold(self):
        config.ENABLE_TIME_EXIT = True
        ctx = make_context(entry_price=100.0, current_price=100.5, holding_days=1.0,
                            peak_date="2026-01-02", now=datetime(2026, 1, 2, 10, 0, 0))
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)

    def test_time_exit_triggers_on_no_new_high_for_too_long(self):
        config.ENABLE_TIME_EXIT = True
        ctx = make_context(
            entry_price=100.0, current_price=150.0, holding_days=1.0,
            peak_date="2026-01-01", now=datetime(2026, 1, 1 + config.EXIT_TIME_NO_NEW_HIGH_DAYS + 1))
        sig = time_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)

    def test_time_exit_no_trigger_when_gain_meets_expectation(self):
        config.ENABLE_TIME_EXIT = True
        ctx = make_context(
            entry_price=100.0, current_price=110.0,   # +10%, above EXIT_TIME_MIN_EXPECTED_GAIN_PCT
            holding_days=config.EXIT_TIME_MAX_HOLDING_DAYS + 5,
            peak_date="2026-01-01", now=datetime(2026, 1, 2))
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)

    def test_malformed_peak_date_is_safe(self):
        config.ENABLE_TIME_EXIT = True
        ctx = make_context(peak_date="not-a-date", holding_days=1.0,
                            entry_price=100.0, current_price=110.0)
        sig = time_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()

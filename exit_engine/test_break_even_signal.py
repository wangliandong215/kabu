# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/break_even_signal.py.

Run:  python -m unittest exit_engine.test_break_even_signal -v
"""
import unittest

import config
from exit_engine import break_even_signal
from exit_engine._fixtures import make_context
import test_support


class BreakEvenSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = config.ENABLE_BREAK_EVEN
        config.ENABLE_BREAK_EVEN = True

    def tearDown(self):
        config.ENABLE_BREAK_EVEN = self._orig

    def test_disabled_returns_enabled_false(self):
        config.ENABLE_BREAK_EVEN = False
        ctx = make_context()
        sig = break_even_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_zero_entry_price_is_safe(self):
        ctx = make_context(entry_price=0.0)
        sig = break_even_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, break_even_signal.STATE_NOT_ACTIVE)

    def test_not_activated_below_peak_gain_threshold(self):
        ctx = make_context(entry_price=100.0,
                            peak_price=100.0 * (1 + config.EXIT_BREAK_EVEN_ACTIVATION_PCT / 2),
                            current_price=100.0)
        sig = break_even_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, break_even_signal.STATE_NOT_ACTIVE)

    def test_activated_but_still_protected_far_from_cost(self):
        entry = 100.0
        peak = entry * (1 + config.EXIT_BREAK_EVEN_ACTIVATION_PCT * 2)
        ctx = make_context(entry_price=entry, peak_price=peak, current_price=peak)
        sig = break_even_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, break_even_signal.STATE_PROTECTED)

    def test_triggers_when_pulled_back_to_break_even(self):
        entry = 100.0
        peak = entry * (1 + config.EXIT_BREAK_EVEN_ACTIVATION_PCT * 2)
        ctx = make_context(entry_price=entry, peak_price=peak, current_price=entry)
        sig = break_even_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, break_even_signal.STATE_AT_BREAK_EVEN)
        self.assertAlmostEqual(sig.points, config.EXIT_BREAK_EVEN_POINTS)

    def test_tolerance_boundary_included(self):
        entry = 100.0
        peak = entry * (1 + config.EXIT_BREAK_EVEN_ACTIVATION_PCT * 3)
        current = entry * (1 + config.EXIT_BREAK_EVEN_TOLERANCE_PCT)   # exactly at tolerance edge
        ctx = make_context(entry_price=entry, peak_price=peak, current_price=current)
        sig = break_even_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()

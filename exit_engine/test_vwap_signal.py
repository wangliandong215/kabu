# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/vwap_signal.py.

Run:  python -m unittest exit_engine.test_vwap_signal -v
"""
import unittest

import config
from exit_engine import vwap_signal
from exit_engine._fixtures import make_bars, make_context


class VwapSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = config.ENABLE_VWAP_EXIT
        config.ENABLE_VWAP_EXIT = True

    def tearDown(self):
        config.ENABLE_VWAP_EXIT = self._orig

    def _flat_session(self, price=100.0, n=10):
        return make_bars([price] * n, highs=[price] * n, lows=[price] * n, volumes=[1000.0] * n)

    def test_disabled_returns_enabled_false(self):
        config.ENABLE_VWAP_EXIT = False
        ctx = make_context(session_bars_1m=self._flat_session())
        sig = vwap_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.points, 0.0)

    def test_missing_session_bars_degrades_to_unknown(self):
        ctx = make_context(session_bars_1m=None)
        sig = vwap_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_UNKNOWN)

    def test_price_above_vwap_not_triggered(self):
        ctx = make_context(current_price=101.0, session_bars_1m=self._flat_session(100.0))
        sig = vwap_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_ABOVE)

    def test_price_narrowly_below_vwap_triggers_base_points(self):
        ctx = make_context(current_price=99.5, session_bars_1m=self._flat_session(100.0))
        sig = vwap_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_LOST)
        self.assertAlmostEqual(sig.points, config.EXIT_VWAP_POINTS_BASE)

    def test_price_wide_below_vwap_triggers_extra_points(self):
        ctx = make_context(current_price=98.5, session_bars_1m=self._flat_session(100.0))
        sig = vwap_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_LOST_WIDE)
        self.assertAlmostEqual(sig.points,
                                config.EXIT_VWAP_POINTS_BASE + config.EXIT_VWAP_POINTS_WIDE)

    def test_zero_current_price_is_safe(self):
        ctx = make_context(current_price=0.0, session_bars_1m=self._flat_session())
        sig = vwap_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_UNKNOWN)

    def test_empty_session_bars_is_safe(self):
        ctx = make_context(session_bars_1m=make_bars([]))
        sig = vwap_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, vwap_signal.STATE_UNKNOWN)


if __name__ == "__main__":
    unittest.main()

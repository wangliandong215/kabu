# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/atr_signal.py.

Run:  python -m unittest exit_engine.test_atr_signal -v
"""
import unittest

import config
from exit_engine import atr_signal
from exit_engine._fixtures import make_context


class AtrSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = config.ENABLE_ATR_EXIT
        config.ENABLE_ATR_EXIT = True

    def tearDown(self):
        config.ENABLE_ATR_EXIT = self._orig

    def test_disabled_returns_enabled_false(self):
        config.ENABLE_ATR_EXIT = False
        ctx = make_context()
        sig = atr_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_missing_atr_degrades_to_unknown(self):
        ctx = make_context(current_atr=None)
        sig = atr_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, atr_signal.STATE_UNKNOWN)

    def test_zero_atr_degrades_to_unknown(self):
        ctx = make_context(current_atr=0.0)
        sig = atr_signal.evaluate(ctx)
        self.assertEqual(sig.state, atr_signal.STATE_UNKNOWN)

    def test_price_above_atr_stop_does_not_trigger(self):
        ctx = make_context(peak_price=100.0, current_atr=1.0, current_price=99.0,
                            entry_atr=1.0)
        sig = atr_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, atr_signal.STATE_NORMAL)

    def test_price_below_atr_stop_triggers(self):
        # stop = peak - MULT*atr = 100 - 2.5*1 = 97.5
        ctx = make_context(peak_price=100.0, current_atr=1.0, current_price=97.0,
                            entry_atr=1.0)
        sig = atr_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertIn(atr_signal.STATE_STOP_BREACH, sig.state)
        self.assertAlmostEqual(sig.extra["atr_stop_price"], 100.0 - config.EXIT_ATR_STOP_MULT * 1.0)

    def test_volatility_expansion_triggers_independently(self):
        ctx = make_context(peak_price=100.0, current_price=100.0,
                            current_atr=5.0, entry_atr=1.0)   # 5x expansion
        sig = atr_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertIn(atr_signal.STATE_EXPANSION, sig.state)

    def test_both_conditions_sum_points(self):
        # stop = peak - MULT*atr = 100 - 2.5*5 = 87.5 -> price=80 breaches it
        ctx = make_context(peak_price=100.0, current_price=80.0,
                            current_atr=5.0, entry_atr=1.0)
        sig = atr_signal.evaluate(ctx)
        self.assertAlmostEqual(sig.points,
                                config.EXIT_ATR_STOP_POINTS + config.EXIT_ATR_EXPANSION_POINTS)

    def test_missing_entry_atr_skips_expansion_check_only(self):
        ctx = make_context(peak_price=100.0, current_price=99.0,
                            current_atr=1.0, entry_atr=None)
        sig = atr_signal.evaluate(ctx)
        self.assertIsNone(sig.extra["volatility_expansion"])


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/trailing_stop_signal.py.

Run:  python -m unittest exit_engine.test_trailing_stop_signal -v
"""
import unittest

import config
from exit_engine import trailing_stop_signal
from exit_engine._fixtures import make_context


class TrailingStopSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = config.ENABLE_TRAILING_STOP
        config.ENABLE_TRAILING_STOP = True

    def tearDown(self):
        config.ENABLE_TRAILING_STOP = self._orig

    def test_disabled_returns_enabled_false(self):
        config.ENABLE_TRAILING_STOP = False
        ctx = make_context()
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_zero_peak_price_degrades_to_unknown(self):
        ctx = make_context(peak_price=0.0)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, trailing_stop_signal.STATE_UNKNOWN)

    def test_fixed_trailing_stop_used_when_no_atr_or_profit_candidate(self):
        ctx = make_context(peak_price=100.0, entry_price=100.0, current_atr=None,
                            current_price=100.0 * (1 - config.EXIT_TRAILING_FIXED_PCT) - 1)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertAlmostEqual(sig.extra["trailing_stop_price"],
                                100.0 * (1 - config.EXIT_TRAILING_FIXED_PCT))

    def test_price_above_all_candidates_does_not_trigger(self):
        ctx = make_context(peak_price=100.0, entry_price=100.0, current_atr=1.0,
                            current_price=99.9)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, trailing_stop_signal.STATE_ABOVE)

    def test_profit_candidate_activates_only_above_threshold(self):
        # peak gain exactly at the activation threshold -> profit candidate present
        entry = 100.0
        peak = entry * (1 + config.EXIT_TRAILING_PROFIT_ACTIVATION_PCT)
        ctx = make_context(peak_price=peak, entry_price=entry, current_atr=None,
                            current_price=peak)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertIn("profit", sig.extra["candidates"])

    def test_profit_candidate_absent_below_activation_threshold(self):
        entry = 100.0
        peak = entry * (1 + config.EXIT_TRAILING_PROFIT_ACTIVATION_PCT / 2)
        ctx = make_context(peak_price=peak, entry_price=entry, current_atr=None,
                            current_price=peak)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertNotIn("profit", sig.extra["candidates"])

    def test_uses_max_of_candidates_as_stop_price(self):
        ctx = make_context(peak_price=100.0, entry_price=90.0, current_atr=0.5,
                            current_price=95.0)
        sig = trailing_stop_signal.evaluate(ctx)
        self.assertAlmostEqual(sig.extra["trailing_stop_price"], max(sig.extra["candidates"].values()))


if __name__ == "__main__":
    unittest.main()

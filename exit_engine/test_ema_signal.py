# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/ema_signal.py.

Run:  python -m unittest exit_engine.test_ema_signal -v
"""
import unittest

import config
from exit_engine import ema_signal
from exit_engine._fixtures import make_bars, make_context
import test_support


class EmaSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = config.ENABLE_EMA_EXIT
        config.ENABLE_EMA_EXIT = True

    def tearDown(self):
        config.ENABLE_EMA_EXIT = self._orig

    def test_disabled_returns_enabled_false(self):
        config.ENABLE_EMA_EXIT = False
        ctx = make_context(bars_5m=make_bars([100.0] * 30))
        sig = ema_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_insufficient_bars_degrades_to_unknown(self):
        ctx = make_context(bars_5m=make_bars([100.0] * 3))
        sig = ema_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, ema_signal.STATE_UNKNOWN)

    def test_none_bars_degrades_to_unknown(self):
        ctx = make_context(bars_5m=None)
        sig = ema_signal.evaluate(ctx)
        self.assertEqual(sig.state, ema_signal.STATE_UNKNOWN)

    def test_steady_uptrend_price_above_both_emas_is_bullish(self):
        closes = [100.0 + i * 0.5 for i in range(40)]
        ctx = make_context(current_price=closes[-1] + 5, bars_5m=make_bars(closes))
        sig = ema_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, ema_signal.STATE_BULLISH)

    def test_price_dropped_below_both_emas_triggers(self):
        closes = [100.0 + i * 0.5 for i in range(40)]
        ctx = make_context(current_price=50.0, bars_5m=make_bars(closes))
        sig = ema_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, ema_signal.STATE_BELOW_EMA)
        self.assertAlmostEqual(sig.points, config.EXIT_EMA_BELOW_POINTS)

    def test_bearish_cross_triggers_higher_points(self):
        # Rising into a sharp reversal, calibrated (by construction) so the
        # fast EMA crosses below the slow EMA exactly on the last bar.
        closes = [100.0 + i for i in range(25)] + [125.0 - i * 3 for i in range(7)]
        ctx = make_context(current_price=closes[-1], bars_5m=make_bars(closes))
        sig = ema_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, ema_signal.STATE_BEARISH_CROSS)
        self.assertAlmostEqual(sig.points, config.EXIT_EMA_CROSS_POINTS)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()

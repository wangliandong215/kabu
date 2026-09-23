# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/trend_signal.py — multi-timeframe
compounding.

Run:  python -m unittest exit_engine.test_trend_signal -v
"""
import unittest

import config
from exit_engine import trend_signal
from exit_engine._fixtures import make_bars, make_context
import test_support

_UP = [100.0 + i for i in range(30)]
_DOWN = [130.0 - i for i in range(30)]


class TrendSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "1m": config.ENABLE_1M_TREND_EXIT,
            "5m": config.ENABLE_5M_TREND_EXIT,
            "15m": config.ENABLE_15M_TREND_EXIT,
        }
        config.ENABLE_1M_TREND_EXIT = False
        config.ENABLE_5M_TREND_EXIT = False
        config.ENABLE_15M_TREND_EXIT = False

    def tearDown(self):
        config.ENABLE_1M_TREND_EXIT = self._orig["1m"]
        config.ENABLE_5M_TREND_EXIT = self._orig["5m"]
        config.ENABLE_15M_TREND_EXIT = self._orig["15m"]

    def test_all_flags_off_returns_disabled(self):
        ctx = make_context(bars_1m=make_bars(_DOWN))
        sig = trend_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_only_1m_enabled_and_down_scores_one_point(self):
        config.ENABLE_1M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars(_DOWN))
        sig = trend_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.points, 1.0)   # 1 timeframe down, no compound bonus

    def test_1m_and_5m_both_down_scores_more_than_double_single(self):
        config.ENABLE_1M_TREND_EXIT = True
        config.ENABLE_5M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars(_DOWN), bars_5m=make_bars(_DOWN))
        sig = trend_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        expected = 2.0 + config.EXIT_TREND_COMPOUND_BONUS
        self.assertAlmostEqual(sig.points, expected)
        # Explicit spec requirement: 1m+5m Down must score more than a
        # single enabled-and-down timeframe would on its own.
        single_ctx = make_context(bars_1m=make_bars(_DOWN))
        config.ENABLE_5M_TREND_EXIT = False
        single_sig = trend_signal.evaluate(single_ctx)
        self.assertGreater(sig.points, single_sig.points)

    def test_all_three_timeframes_down_scores_highest(self):
        config.ENABLE_1M_TREND_EXIT = True
        config.ENABLE_5M_TREND_EXIT = True
        config.ENABLE_15M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars(_DOWN), bars_5m=make_bars(_DOWN),
                            bars_15m=make_bars(_DOWN))
        sig = trend_signal.evaluate(ctx)
        expected = 3.0 + config.EXIT_TREND_COMPOUND_BONUS + config.EXIT_TREND_ALL_DOWN_BONUS
        self.assertAlmostEqual(sig.points, expected)

    def test_up_trend_does_not_trigger(self):
        config.ENABLE_1M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars(_UP))
        sig = trend_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.points, 0.0)

    def test_insufficient_bars_on_enabled_timeframe_is_safe(self):
        config.ENABLE_1M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars([100.0, 101.0]))
        sig = trend_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, "UNKNOWN")

    def test_missing_bars_for_disabled_timeframe_does_not_affect_result(self):
        config.ENABLE_1M_TREND_EXIT = True
        ctx = make_context(bars_1m=make_bars(_DOWN), bars_5m=None, bars_15m=None)
        sig = trend_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.points, 1.0)


def setUpModule():
    # Keep this suite off the live C:\KabuData state/log files (see test_support.py).
    test_support.isolate_live_state()


def tearDownModule():
    test_support.restore_live_state()


if __name__ == "__main__":
    unittest.main()

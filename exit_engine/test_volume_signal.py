# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/volume_signal.py.

Run:  python -m unittest exit_engine.test_volume_signal -v
"""
import unittest

import config
from exit_engine import volume_signal
from exit_engine._fixtures import make_bars, make_context


class VolumeSignalTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "exhaustion": config.ENABLE_VOLUME_EXHAUSTION,
            "heavy": config.ENABLE_HEAVY_SELLING,
        }
        config.ENABLE_VOLUME_EXHAUSTION = False
        config.ENABLE_HEAVY_SELLING = False

    def tearDown(self):
        config.ENABLE_VOLUME_EXHAUSTION = self._orig["exhaustion"]
        config.ENABLE_HEAVY_SELLING = self._orig["heavy"]

    def test_both_flags_off_returns_disabled(self):
        ctx = make_context(bars_5m=make_bars([100.0] * 25))
        sig = volume_signal.evaluate(ctx)
        self.assertFalse(sig.enabled)

    def test_insufficient_bars_is_safe(self):
        config.ENABLE_HEAVY_SELLING = True
        ctx = make_context(bars_5m=make_bars([100.0] * 3))
        sig = volume_signal.evaluate(ctx)
        self.assertTrue(sig.enabled)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, volume_signal.STATE_UNKNOWN)

    def test_heavy_selling_triggers_on_price_down_volume_spike(self):
        config.ENABLE_HEAVY_SELLING = True
        closes = [100.0] * 21 + [95.0]     # last bar: price down 5%
        volumes = [1000.0] * 21 + [3000.0]  # last bar: volume way above average
        ctx = make_context(bars_5m=make_bars(closes, volumes=volumes))
        sig = volume_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, volume_signal.STATE_HEAVY_SELLING)
        self.assertAlmostEqual(sig.points, config.EXIT_HEAVY_SELLING_POINTS)

    def test_exhaustion_triggers_on_price_up_volume_collapse(self):
        config.ENABLE_VOLUME_EXHAUSTION = True
        closes = [100.0] * 21 + [101.0]    # last bar: price up
        volumes = [1000.0] * 21 + [200.0]  # last bar: volume well below average
        ctx = make_context(bars_5m=make_bars(closes, volumes=volumes))
        sig = volume_signal.evaluate(ctx)
        self.assertTrue(sig.triggered)
        self.assertEqual(sig.state, volume_signal.STATE_EXHAUSTION)
        self.assertAlmostEqual(sig.points, config.EXIT_VOLUME_EXHAUSTION_POINTS)

    def test_heavy_selling_flag_off_does_not_trigger_even_on_matching_data(self):
        config.ENABLE_VOLUME_EXHAUSTION = True
        config.ENABLE_HEAVY_SELLING = False
        closes = [100.0] * 21 + [95.0]
        volumes = [1000.0] * 21 + [3000.0]
        ctx = make_context(bars_5m=make_bars(closes, volumes=volumes))
        sig = volume_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)

    def test_normal_conditions_do_not_trigger(self):
        config.ENABLE_VOLUME_EXHAUSTION = True
        config.ENABLE_HEAVY_SELLING = True
        closes = [100.0 + (i % 2) * 0.1 for i in range(25)]
        ctx = make_context(bars_5m=make_bars(closes))
        sig = volume_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, volume_signal.STATE_NORMAL)

    def test_zero_prior_close_is_safe(self):
        config.ENABLE_HEAVY_SELLING = True
        closes = [0.0] * 22 + [1.0]
        ctx = make_context(bars_5m=make_bars(closes))
        sig = volume_signal.evaluate(ctx)
        self.assertFalse(sig.triggered)
        self.assertEqual(sig.state, volume_signal.STATE_UNKNOWN)


if __name__ == "__main__":
    unittest.main()

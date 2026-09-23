# -*- coding: utf-8 -*-
"""Unit tests for exit_engine/pressure_score.py — aggregation math and
tier-boundary behavior.

Run:  python -m unittest exit_engine.test_pressure_score -v
"""
import unittest

import config
from exit_engine import pressure_score
from exit_engine.models import TIER_CRITICAL, TIER_HIGH, TIER_LOW, TIER_MEDIUM, SignalReading


def _reading(name, enabled=True, triggered=False, points=0.0):
    return SignalReading(name, enabled, triggered, points, "test")


class TierForScoreTestCase(unittest.TestCase):
    def test_below_medium_is_low(self):
        self.assertEqual(pressure_score.tier_for_score(0.0), TIER_LOW)
        self.assertEqual(
            pressure_score.tier_for_score(config.EXIT_PRESSURE_MEDIUM_THRESHOLD - 0.01),
            TIER_LOW)

    def test_at_medium_threshold_is_medium(self):
        self.assertEqual(
            pressure_score.tier_for_score(config.EXIT_PRESSURE_MEDIUM_THRESHOLD),
            TIER_MEDIUM)

    def test_at_high_threshold_is_high(self):
        self.assertEqual(
            pressure_score.tier_for_score(config.EXIT_PRESSURE_HIGH_THRESHOLD),
            TIER_HIGH)

    def test_at_critical_threshold_is_critical(self):
        self.assertEqual(
            pressure_score.tier_for_score(config.EXIT_PRESSURE_CRITICAL_THRESHOLD),
            TIER_CRITICAL)

    def test_far_above_critical_stays_critical(self):
        self.assertEqual(
            pressure_score.tier_for_score(config.EXIT_PRESSURE_CRITICAL_THRESHOLD + 1000),
            TIER_CRITICAL)


class AggregateTestCase(unittest.TestCase):
    def test_all_signals_disabled_yields_zero_score_and_zero_confidence(self):
        signals = [_reading("a", enabled=False), _reading("b", enabled=False)]
        score, tier, confidence = pressure_score.aggregate(signals)
        self.assertEqual(score, 0.0)
        self.assertEqual(tier, TIER_LOW)
        self.assertEqual(confidence, 0.0)

    def test_empty_signal_list_is_safe(self):
        score, tier, confidence = pressure_score.aggregate([])
        self.assertEqual(score, 0.0)
        self.assertEqual(tier, TIER_LOW)
        self.assertEqual(confidence, 0.0)

    def test_disabled_signal_points_are_excluded_from_score(self):
        signals = [
            _reading("a", enabled=True, triggered=True, points=100.0),
            _reading("b", enabled=False, triggered=True, points=999.0),
        ]
        score, _, confidence = pressure_score.aggregate(signals)
        self.assertEqual(score, 100.0)
        self.assertEqual(confidence, 1.0)   # only "a" counts toward the denominator too

    def test_mixed_enabled_disabled_signals_confidence_is_fraction_of_enabled(self):
        signals = [
            _reading("a", enabled=True, triggered=True, points=1.0),
            _reading("b", enabled=True, triggered=False, points=0.0),
            _reading("c", enabled=False, triggered=True, points=5.0),
        ]
        score, tier, confidence = pressure_score.aggregate(signals)
        self.assertEqual(score, 1.0)
        self.assertAlmostEqual(confidence, 0.5)   # 1 of 2 enabled triggered
        self.assertEqual(tier, TIER_LOW)

    def test_sum_of_enabled_points_drives_tier(self):
        signals = [
            _reading("a", enabled=True, triggered=True, points=config.EXIT_PRESSURE_HIGH_THRESHOLD),
        ]
        score, tier, _ = pressure_score.aggregate(signals)
        self.assertEqual(score, config.EXIT_PRESSURE_HIGH_THRESHOLD)
        self.assertEqual(tier, TIER_HIGH)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for research/trade_intelligence_patterns.py — the shared stats +
Observation/Candidate Pattern gating primitives every V3.6-A analysis
module builds on.

Run:  python -m unittest research.test_trade_intelligence_patterns -v
"""
import unittest

import pandas as pd

import config
from research.trade_intelligence_patterns import (
    PatternCandidate, binary_flag_stats, bucket_compare, bucket_label,
    classify_state, _assert_hedged_language,
)


class ClassifyStateTestCase(unittest.TestCase):

    def test_below_observation_floor_returns_none(self):
        self.assertIsNone(classify_state(n_sample=config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N - 1,
                                          effect_size=0.9, precision=0.99))

    def test_between_floors_is_always_observation_regardless_of_effect(self):
        n = config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N
        self.assertLess(n, config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N)
        self.assertEqual(classify_state(n_sample=n, effect_size=0.9, precision=0.99), "OBSERVATION")

    def test_large_n_with_strong_effect_size_is_candidate_pattern(self):
        n = config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N
        self.assertEqual(
            classify_state(n_sample=n, effect_size=config.TRADE_INTELLIGENCE_MIN_CANDIDATE_EFFECT_PCT,
                            precision=None),
            "CANDIDATE_PATTERN")

    def test_large_n_with_strong_precision_is_candidate_pattern(self):
        n = config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N
        self.assertEqual(
            classify_state(n_sample=n, effect_size=None,
                            precision=config.TRADE_INTELLIGENCE_MIN_CANDIDATE_PRECISION),
            "CANDIDATE_PATTERN")

    def test_large_n_with_weak_effect_stays_observation(self):
        n = config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N * 10
        self.assertEqual(classify_state(n_sample=n, effect_size=0.001, precision=None), "OBSERVATION")

    def test_never_returns_a_state_outside_the_known_set(self):
        for n in (0, 1, config.TRADE_INTELLIGENCE_MIN_OBSERVATION_N,
                  config.TRADE_INTELLIGENCE_MIN_CANDIDATE_N, 10_000):
            for effect in (None, -0.9, 0.001, 0.9):
                for precision in (None, 0.0, 0.5, 1.0):
                    self.assertIn(classify_state(n, effect, precision),
                                  (None, "OBSERVATION", "CANDIDATE_PATTERN"))


class BinaryFlagStatsTestCase(unittest.TestCase):

    def test_precision_recall_match_hand_computed_arithmetic(self):
        df = pd.DataFrame({
            "_flag": [1, 1, 1, 0, 0],
            "outcome": ["EARLY_FAILURE", "EARLY_FAILURE", "WINNER", "EARLY_FAILURE", "WINNER"],
        })
        stats = binary_flag_stats(df, "_flag", "outcome", {"EARLY_FAILURE"})
        self.assertEqual(stats["n_flagged"], 3)
        self.assertAlmostEqual(stats["precision"], 2 / 3)          # 2 TP / 3 flagged
        self.assertAlmostEqual(stats["recall"], 2 / 3)             # 2 TP / 3 total EARLY_FAILURE
        self.assertEqual(stats["false_positive_count"], 1)
        self.assertEqual(stats["n_positive_total"], 3)

    def test_no_flagged_rows_returns_none_precision(self):
        df = pd.DataFrame({"_flag": [0, 0], "outcome": ["EARLY_FAILURE", "WINNER"]})
        stats = binary_flag_stats(df, "_flag", "outcome", {"EARLY_FAILURE"})
        self.assertEqual(stats["n_flagged"], 0)
        self.assertIsNone(stats["precision"])


class BucketCompareTestCase(unittest.TestCase):

    def test_groupby_mean_and_count(self):
        df = pd.DataFrame({
            "band": ["A", "A", "B", "B", "B"],
            "value": [1.0, 3.0, 10.0, 20.0, 30.0],
        })
        out = bucket_compare(df, "band", "value")
        row_a = out[out["band"] == "A"].iloc[0]
        row_b = out[out["band"] == "B"].iloc[0]
        self.assertEqual(row_a["n"], 2)
        self.assertAlmostEqual(row_a["value"], 2.0)
        self.assertEqual(row_b["n"], 3)
        self.assertAlmostEqual(row_b["value"], 20.0)

    def test_empty_input_returns_empty_frame(self):
        df = pd.DataFrame({"band": [], "value": []})
        out = bucket_compare(df, "band", "value")
        self.assertTrue(out.empty)


class BucketLabelTestCase(unittest.TestCase):

    def test_two_edge_bucketing(self):
        edges = (40.0, 60.0)
        self.assertEqual(bucket_label(10, edges), "<40")
        self.assertEqual(bucket_label(50, edges), "40-60")
        self.assertEqual(bucket_label(70, edges), ">=60")

    def test_three_edge_bucketing_is_non_overlapping(self):
        edges = (2, 5, 10)
        self.assertEqual(bucket_label(1, edges, suffix="d"), "<2d")
        self.assertEqual(bucket_label(3, edges, suffix="d"), "2-5d")
        self.assertEqual(bucket_label(7, edges, suffix="d"), "5-10d")
        self.assertEqual(bucket_label(15, edges, suffix="d"), ">=10d")

    def test_nan_returns_none(self):
        self.assertIsNone(bucket_label(float("nan"), (1, 2)))


class HedgedLanguageTestCase(unittest.TestCase):

    def test_raises_on_causal_verbs(self):
        with self.assertRaises(ValueError):
            _assert_hedged_language("Low confidence causes early failure.")
        with self.assertRaises(ValueError):
            _assert_hedged_language("This pattern leads to losses.")

    def test_passes_on_hedged_language(self):
        _assert_hedged_language("Association only, not shown to be causal.")  # no raise

    def test_pattern_candidate_construction_enforces_hedged_language(self):
        with self.assertRaises(ValueError):
            PatternCandidate(
                analysis_type="EARLY_FAILURE", pattern_key="x", state="OBSERVATION",
                description="X causes Y.", metric_name="m", metric_value=1.0)

    def test_pattern_candidate_rejects_invalid_state(self):
        with self.assertRaises(ValueError):
            PatternCandidate(
                analysis_type="EARLY_FAILURE", pattern_key="x", state="PRODUCTION_RULE",
                description="Fine, hedged text.", metric_name="m", metric_value=1.0)


if __name__ == "__main__":
    unittest.main()

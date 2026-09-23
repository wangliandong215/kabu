"""
Unit tests for news/scoring/severity.py, confidence.py, news_score.py.

Run:  python -m unittest news.test_scoring -v
"""
import unittest

from news.event_types import Direction, EventType
from news.news_sources import Source
from news.scoring.confidence import compute_confidence
from news.scoring.news_score import compute_news_score
from news.scoring.severity import compute_severity


class TestSeverity(unittest.TestCase):

    def test_bankruptcy_is_max_severity(self):
        self.assertEqual(compute_severity(EventType.BANKRUPTCY), 1.0)

    def test_unknown_is_low_severity(self):
        self.assertLess(compute_severity(EventType.UNKNOWN), 0.5)

    def test_keyword_strength_scales_within_bounds(self):
        base = compute_severity(EventType.EARNINGS_MISS, keyword_strength=1.0)
        boosted = compute_severity(EventType.EARNINGS_MISS, keyword_strength=1.5)
        self.assertGreaterEqual(boosted, base)
        self.assertLessEqual(boosted, 1.0)

    def test_always_clamped_0_to_1(self):
        self.assertGreaterEqual(compute_severity(EventType.BANKRUPTCY, keyword_strength=10.0), 0.0)
        self.assertLessEqual(compute_severity(EventType.BANKRUPTCY, keyword_strength=10.0), 1.0)


class TestConfidence(unittest.TestCase):

    def test_high_trust_source_higher_than_social(self):
        sec_conf = compute_confidence(Source.SEC, keyword_match_count=1)
        social_conf = compute_confidence(Source.SOCIAL, keyword_match_count=1)
        self.assertGreater(sec_conf, social_conf)

    def test_more_keyword_matches_increases_confidence(self):
        one = compute_confidence(Source.REUTERS, keyword_match_count=1)
        three = compute_confidence(Source.REUTERS, keyword_match_count=3)
        self.assertGreater(three, one)

    def test_clamped_0_to_1(self):
        self.assertLessEqual(compute_confidence(Source.SEC, keyword_match_count=100), 1.0)


class TestNewsScore(unittest.TestCase):

    def test_neutral_direction_is_zero(self):
        score = compute_news_score(Direction.NEUTRAL, severity=0.9, confidence=0.9, source=Source.SEC)
        self.assertEqual(score, 0.0)

    def test_unknown_direction_is_zero(self):
        score = compute_news_score(Direction.UNKNOWN, severity=0.9, confidence=0.9, source=Source.SEC)
        self.assertEqual(score, 0.0)

    def test_positive_direction_is_positive(self):
        score = compute_news_score(Direction.POSITIVE, severity=0.5, confidence=0.8, source=Source.REUTERS)
        self.assertGreater(score, 0.0)

    def test_negative_direction_is_negative(self):
        score = compute_news_score(Direction.NEGATIVE, severity=0.5, confidence=0.8, source=Source.REUTERS)
        self.assertLess(score, 0.0)

    def test_not_a_flat_plus_or_minus_one(self):
        # spec section 10: News Score must not simply be +1/-1 by sign alone.
        score = compute_news_score(Direction.NEGATIVE, severity=0.3, confidence=0.3, source=Source.SOCIAL)
        self.assertGreater(score, -1.0)
        self.assertLess(score, 0.0)

    def test_low_severity_and_confidence_yields_small_magnitude(self):
        weak = compute_news_score(Direction.NEGATIVE, severity=0.2, confidence=0.2, source=Source.SOCIAL)
        strong = compute_news_score(Direction.NEGATIVE, severity=0.9, confidence=0.9, source=Source.SEC)
        self.assertGreater(abs(strong), abs(weak))

    def test_clamped_to_unit_range(self):
        score = compute_news_score(Direction.NEGATIVE, severity=999, confidence=999, source=Source.SEC)
        self.assertGreaterEqual(score, -1.0)


if __name__ == "__main__":
    unittest.main()

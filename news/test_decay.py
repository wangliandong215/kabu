"""
Unit tests for news/scoring/decay.py.

Run:  python -m unittest news.test_decay -v
"""
import unittest
from datetime import datetime, timedelta

from news.scoring.decay import decayed_score


class TestDecay(unittest.TestCase):

    def test_no_age_returns_full_score(self):
        now = datetime.now()
        self.assertAlmostEqual(decayed_score(0.8, now, decay_hours=24, as_of=now), 0.8, places=6)

    def test_one_half_life_halves_score(self):
        now = datetime.now()
        event_ts = now - timedelta(hours=24)
        self.assertAlmostEqual(decayed_score(0.8, event_ts, decay_hours=24, as_of=now), 0.4, places=6)

    def test_two_half_lives_quarters_score(self):
        now = datetime.now()
        event_ts = now - timedelta(hours=48)
        self.assertAlmostEqual(decayed_score(0.8, event_ts, decay_hours=24, as_of=now), 0.2, places=6)

    def test_expired_news_score_much_lower_than_fresh(self):
        # V3.4 spec Test 5 — "新闻超过decay window：effective_score应该明显降低"
        now = datetime.now()
        fresh = decayed_score(0.8, now - timedelta(hours=1), decay_hours=24, as_of=now)
        expired = decayed_score(0.8, now - timedelta(hours=24 * 10), decay_hours=24, as_of=now)
        self.assertLess(expired, fresh * 0.1)

    def test_future_timestamp_treated_as_zero_age(self):
        now = datetime.now()
        future = now + timedelta(hours=5)
        self.assertAlmostEqual(decayed_score(0.8, future, decay_hours=24, as_of=now), 0.8, places=6)

    def test_zero_decay_hours_is_safe(self):
        now = datetime.now()
        self.assertEqual(decayed_score(0.8, now, decay_hours=0, as_of=now), 0.0)

    def test_slow_decay_type_retains_more_signal_than_fast_decay_type(self):
        now = datetime.now()
        event_ts = now - timedelta(hours=24 * 5)
        slow = decayed_score(0.8, event_ts, decay_hours=24 * 30, as_of=now)   # e.g. BANKRUPTCY
        fast = decayed_score(0.8, event_ts, decay_hours=24 * 1, as_of=now)    # e.g. UNKNOWN
        self.assertGreater(slow, fast)


if __name__ == "__main__":
    unittest.main()

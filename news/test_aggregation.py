"""
Unit tests for news/scoring/aggregator.py — V3.4 spec Test 4 (duplicate
news must not stack) and Test 6 (multiple distinct events aggregate by
rule, not naive sum).

Run:  python -m unittest news.test_aggregation -v
"""
import unittest
from datetime import datetime

from news.event_types import Direction, EventType
from news.news_event import NewsEvent
from news.news_sources import Source
from news.scoring.aggregator import aggregate


def _event(ticker="US.NVDA", event_type=EventType.EARNINGS_MISS, direction=Direction.NEGATIVE,
           severity=0.5, confidence=0.8, headline="earnings miss reported", ts=None,
           decay_hours=72.0, emergency=False, news_score=None):
    ts = ts or datetime.now()
    score = news_score if news_score is not None else (
        -severity * confidence if direction == Direction.NEGATIVE else severity * confidence)
    return NewsEvent(ticker=ticker, event_type=event_type, direction=direction,
                      severity=severity, confidence=confidence, news_score=score,
                      source=Source.REUTERS, timestamp=ts, headline=headline,
                      decay_hours=decay_hours, emergency=emergency)


class TestAggregation(unittest.TestCase):

    def test_empty_events_returns_zero_signal(self):
        signal = aggregate([])
        self.assertEqual(signal.effective_score, 0.0)
        self.assertFalse(signal.emergency)

    def test_duplicate_headline_does_not_double_count(self):
        # V3.4 spec Test 4 — same story appearing twice must not stack.
        as_of = datetime.now()
        single = aggregate([_event(headline="Company misses earnings estimates")], as_of=as_of)
        duplicated = aggregate([
            _event(headline="Company misses earnings estimates"),
            _event(headline="company MISSES earnings estimates!"),
        ], as_of=as_of)
        self.assertAlmostEqual(single.effective_score, duplicated.effective_score, places=6)

    def test_distinct_events_use_geometric_dampening_not_naive_sum(self):
        # V3.4 spec Test 6 — multiple distinct events, must not be a flat sum.
        as_of = datetime.now()
        e1 = _event(headline="earnings miss", severity=0.5, confidence=0.8, news_score=-0.4)
        e2 = _event(headline="regulatory investigation opened", event_type=EventType.REGULATORY_RISK,
                     severity=0.6, confidence=0.7, news_score=-0.42)
        e3 = _event(headline="supply chain disruption reported", event_type=EventType.SUPPLY_CHAIN,
                     severity=0.4, confidence=0.6, news_score=-0.24)
        signal = aggregate([e1, e2, e3], as_of=as_of)
        naive_sum = -0.4 + -0.42 + -0.24
        self.assertGreater(signal.effective_score, naive_sum)  # dampened, less negative than naive sum
        self.assertGreaterEqual(signal.effective_score, -1.0)

    def test_worst_severity_and_paired_confidence_reported(self):
        as_of = datetime.now()
        mild = _event(headline="minor guidance tweak", event_type=EventType.GUIDANCE_CHANGE,
                       severity=0.3, confidence=0.9, news_score=-0.2)
        severe = _event(headline="bankruptcy filing announced", event_type=EventType.BANKRUPTCY,
                         severity=0.95, confidence=0.4, news_score=-0.3)
        signal = aggregate([mild, severe], as_of=as_of)
        self.assertEqual(signal.worst_severity, 0.95)
        self.assertEqual(signal.confidence, 0.4)

    def test_emergency_flag_is_or_across_events_never_diluted(self):
        as_of = datetime.now()
        calm = _event(headline="routine product update", severity=0.1, confidence=0.5, news_score=0.0)
        emergency = _event(headline="trading halted amid fraud probe", event_type=EventType.FRAUD_RISK,
                            severity=0.95, confidence=0.9, news_score=-0.8, emergency=True)
        signal = aggregate([calm, emergency], as_of=as_of)
        self.assertTrue(signal.emergency)

    def test_expired_event_contributes_much_less_than_fresh(self):
        from datetime import timedelta
        as_of = datetime.now()
        fresh = _event(headline="fresh earnings miss", ts=as_of - timedelta(hours=1),
                        decay_hours=24, news_score=-0.5)
        expired_only = aggregate(
            [_event(headline="stale earnings miss", ts=as_of - timedelta(hours=24 * 30),
                     decay_hours=24, news_score=-0.5)], as_of=as_of)
        fresh_only = aggregate([fresh], as_of=as_of)
        self.assertLess(abs(expired_only.effective_score), abs(fresh_only.effective_score))


if __name__ == "__main__":
    unittest.main()

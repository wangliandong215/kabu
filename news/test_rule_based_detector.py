"""
Unit tests for news/detection/rule_based.py.

Run:  python -m unittest news.test_rule_based_detector -v
"""
import unittest
from datetime import datetime

from news.detection.rule_based import RuleBasedEventDetector
from news.event_types import Direction, EventType
from news.news_item import NewsItem
from news.news_sources import Source


def _item(headline, source=Source.REUTERS, body=""):
    return NewsItem(id="1", ticker="US.NVDA", source=source,
                     timestamp=datetime.now(), headline=headline, body=body)


class TestRuleBasedEventDetector(unittest.TestCase):

    def setUp(self):
        self.detector = RuleBasedEventDetector()

    def test_earnings_beat_headline(self):
        events = self.detector.detect(_item("NVDA Earnings Beat estimates, raised guidance"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.EARNINGS_BEAT)
        self.assertEqual(events[0].direction, Direction.POSITIVE)

    def test_earnings_miss_headline(self):
        events = self.detector.detect(_item("NVDA misses estimates, below expectations"))
        self.assertEqual(events[0].event_type, EventType.EARNINGS_MISS)
        self.assertEqual(events[0].direction, Direction.NEGATIVE)

    def test_bankruptcy_headline(self):
        events = self.detector.detect(_item("Company files for Chapter 11 bankruptcy"))
        self.assertEqual(events[0].event_type, EventType.BANKRUPTCY)
        self.assertTrue(events[0].emergency or events[0].severity > 0.9)

    def test_bankruptcy_outranks_lawsuit_in_same_headline(self):
        events = self.detector.detect(_item("Company sued in class action amid bankruptcy filing"))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.BANKRUPTCY)

    def test_neutral_headline_produces_no_event(self):
        events = self.detector.detect(_item("Company holds annual shareholder meeting"))
        self.assertEqual(events, [])

    def test_fda_approval_is_positive(self):
        events = self.detector.detect(_item("Company receives FDA approval for new drug"))
        self.assertEqual(events[0].event_type, EventType.FDA_APPROVAL)
        self.assertEqual(events[0].direction, Direction.POSITIVE)

    def test_multiple_keyword_matches_raise_confidence(self):
        weak = self.detector.detect(_item("earnings miss reported"))[0]
        strong = self.detector.detect(
            _item("earnings miss, missed earnings, below expectations across the board"))[0]
        self.assertGreaterEqual(strong.confidence, weak.confidence)

    def test_high_trust_source_raises_confidence(self):
        sec_event = self.detector.detect(_item("SEC investigation into accounting practices",
                                                 source=Source.SEC))[0]
        social_event = self.detector.detect(_item("SEC investigation into accounting practices",
                                                    source=Source.SOCIAL))[0]
        self.assertGreater(sec_event.confidence, social_event.confidence)


if __name__ == "__main__":
    unittest.main()

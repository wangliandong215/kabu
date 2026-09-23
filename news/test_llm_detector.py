"""
Unit tests for news/llm/{base,null_provider,mock_provider}.py and
news/detection/llm_detector.py — including V3.4 spec Test 7 (LLM failure
must never propagate / never trigger an emergency).

Run:  python -m unittest news.test_llm_detector -v
"""
import unittest
from datetime import datetime

from news.detection.llm_detector import LLMBackedEventDetector, MODE_OFF, MODE_SHADOW
from news.event_types import EventType
from news.llm.base import EventExtractionResult, LLMEventDetector
from news.llm.mock_provider import MockLLMEventDetector
from news.llm.null_provider import NullLLMEventDetector
from news.news_item import NewsItem
from news.news_sources import Source


def _item(headline="NVDA files for Chapter 11 bankruptcy"):
    return NewsItem(id="1", ticker="US.NVDA", source=Source.REUTERS,
                     timestamp=datetime.now(), headline=headline)


class _RaisingProvider(LLMEventDetector):
    def extract_event(self, news):
        raise TimeoutError("model backend unreachable")


class _GarbageProvider(LLMEventDetector):
    def extract_event(self, news):
        return EventExtractionResult(ticker=news.ticker, event_type="BANKRUPTCY",
                                      direction="NEGATIVE", severity=5.7, confidence=-2.0)


class TestNullLLMEventDetector(unittest.TestCase):

    def test_never_raises_and_returns_unknown(self):
        result = NullLLMEventDetector().extract_event(_item())
        self.assertEqual(result.event_type, "UNKNOWN")
        self.assertEqual(result.confidence, 0.0)


class TestMockLLMEventDetector(unittest.TestCase):

    def test_detects_bankruptcy_like_rule_based(self):
        result = MockLLMEventDetector().extract_event(_item())
        self.assertEqual(result.event_type, EventType.BANKRUPTCY.value)

    def test_neutral_headline_returns_unknown(self):
        result = MockLLMEventDetector().extract_event(_item("Company holds annual meeting"))
        self.assertEqual(result.event_type, "UNKNOWN")


class TestLLMBackedEventDetector(unittest.TestCase):

    def test_mode_off_returns_no_events(self):
        detector = LLMBackedEventDetector(provider=MockLLMEventDetector(), mode=MODE_OFF)
        self.assertEqual(detector.detect(_item()), [])

    def test_shadow_mode_with_mock_provider_produces_validated_event(self):
        detector = LLMBackedEventDetector(provider=MockLLMEventDetector(), mode=MODE_SHADOW)
        events = detector.detect(_item())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.BANKRUPTCY)

    def test_provider_raising_never_propagates_and_returns_empty(self):
        # V3.4 spec Test 7 — LLM timeout/unavailable must not raise or
        # trigger anything; pipeline just proceeds without this event.
        detector = LLMBackedEventDetector(provider=_RaisingProvider(), mode=MODE_SHADOW)
        events = detector.detect(_item())
        self.assertEqual(events, [])

    def test_garbage_llm_output_is_clamped_by_validation_not_rejected(self):
        detector = LLMBackedEventDetector(provider=_GarbageProvider(), mode=MODE_SHADOW)
        events = detector.detect(_item())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].severity, 1.0)
        self.assertEqual(events[0].confidence, 0.0)
        # confidence was clamped to 0 -> never emergency regardless of type/severity.
        self.assertFalse(events[0].emergency)


if __name__ == "__main__":
    unittest.main()

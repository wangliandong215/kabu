"""
Unit tests for news/integration/risk_adapter.py::NewsEngine.

Run:  python -m unittest news.test_risk_adapter -v
"""
import unittest
from datetime import datetime

from news.detection.llm_detector import LLMBackedEventDetector, MODE_SHADOW
from news.event_types import EventType
from news.ingestion.mock import MockNewsSource
from news.integration.risk_adapter import NewsEngine
from news.news_item import NewsItem
from news.news_sources import Source


def _item(ticker, headline, item_id="1"):
    return NewsItem(id=item_id, ticker=ticker, source=Source.REUTERS,
                     timestamp=datetime.now(), headline=headline)


class TestNewsEngine(unittest.TestCase):

    def test_no_source_returns_empty_state_for_all_tickers(self):
        engine = NewsEngine(source=None)
        state = engine.build_news_events_state(["US.NVDA", "US.AMD"])
        self.assertEqual(state, {"US.NVDA": [], "US.AMD": []})

    def test_builds_events_per_ticker(self):
        source = MockNewsSource([
            _item("US.NVDA", "NVDA files for Chapter 11 bankruptcy", "1"),
            _item("US.AMD", "AMD earnings beat estimates", "2"),
        ])
        engine = NewsEngine(source=source)
        state = engine.build_news_events_state(["US.NVDA", "US.AMD"])
        self.assertEqual(len(state["US.NVDA"]), 1)
        self.assertEqual(state["US.NVDA"][0].event_type, EventType.BANKRUPTCY)
        self.assertEqual(state["US.AMD"][0].event_type, EventType.EARNINGS_BEAT)

    def test_ticker_with_no_news_gets_empty_list(self):
        source = MockNewsSource([_item("US.NVDA", "NVDA files for bankruptcy")])
        engine = NewsEngine(source=source)
        state = engine.build_news_events_state(["US.NVDA", "US.MSFT"])
        self.assertEqual(state["US.MSFT"], [])

    def test_source_fetch_failure_degrades_to_empty_state(self):
        class _BrokenSource:
            def fetch(self, tickers):
                raise ConnectionError("feed down")

        engine = NewsEngine(source=_BrokenSource())
        state = engine.build_news_events_state(["US.NVDA"])
        self.assertEqual(state, {"US.NVDA": []})

    def test_llm_detector_can_be_combined_with_rule_based(self):
        source = MockNewsSource([_item("US.NVDA", "NVDA files for Chapter 11 bankruptcy")])
        engine = NewsEngine(source=source,
                             llm_detector=LLMBackedEventDetector(mode=MODE_SHADOW))
        state = engine.build_news_events_state(["US.NVDA"])
        # rule-based + mock-llm both fire on the same headline -> 2 raw events
        # (aggregation/dedup happens downstream in risk/news_event_risk.py,
        # not here).
        self.assertEqual(len(state["US.NVDA"]), 2)


if __name__ == "__main__":
    unittest.main()

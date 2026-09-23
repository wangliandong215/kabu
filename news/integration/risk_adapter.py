"""
news/integration/risk_adapter.py -- V3.4's "V3.3 Adapter" (spec section
21/22): the one place that turns ingestion+detection+validation output
into the Dict[str, List[NewsEvent]] risk/news_event_risk.py reads from
PortfolioState.news_events. Everything upstream (ingestion/detection/
validation/scoring) is pure and independently testable; this module is
the only thing that touches the V3.3 data shape.

Deliberately NOT wired into engine/runner.py yet (spec section 27's
"第一阶段禁止事项" + config.NEWS_ENGINE_ENABLED defaults to False) -- even
once a caller starts building/attaching news_events,
risk/news_event_risk.py stays a no-op (DATA_UNAVAILABLE-safe ALLOW) until
a human flips NEWS_ENGINE_ENABLED on deliberately. See that module's
docstring for the full authority chain.
"""
from typing import Dict, List, Optional

from news.detection.llm_detector import LLMBackedEventDetector
from news.detection.rule_based import RuleBasedEventDetector
from news.ingestion.base import NewsSource
from news.news_event import NewsEvent
from news.news_item import NewsItem


class NewsEngine:
    def __init__(self, source: Optional[NewsSource] = None,
                 rule_detector: Optional[RuleBasedEventDetector] = None,
                 llm_detector: Optional[LLMBackedEventDetector] = None):
        self._source = source
        self._rule_detector = rule_detector or RuleBasedEventDetector()
        # None => rule-based only. Pass an LLMBackedEventDetector() to
        # also run the (mock/null, per config.NEWS_ENGINE_LLM_MODE) LLM path.
        self._llm_detector = llm_detector

    def build_events_for_item(self, item: NewsItem) -> List[NewsEvent]:
        events = list(self._rule_detector.detect(item))
        if self._llm_detector is not None:
            events.extend(self._llm_detector.detect(item))
        return events

    def build_news_events_state(self, tickers: List[str]) -> Dict[str, List[NewsEvent]]:
        """Fetch -> detect for every ticker. Never raises -- a per-source
        or per-item failure degrades to an empty list for that ticker,
        same fail-open convention as every other V3.3/V3.4 module; one bad
        NewsItem must never take down the whole pass."""
        state: Dict[str, List[NewsEvent]] = {t: [] for t in tickers}
        if self._source is None:
            return state

        try:
            items = self._source.fetch(tickers)
        except Exception:
            return state

        for item in items:
            try:
                events = self.build_events_for_item(item)
            except Exception:
                continue
            state.setdefault(item.ticker, []).extend(events)

        return state

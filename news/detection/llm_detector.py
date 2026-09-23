"""
news/detection/llm_detector.py -- orchestrates an LLMEventDetector
provider the same way risk/llm_advisor.py orchestrates an LLMProvider:
pick the provider for config.NEWS_ENGINE_LLM_MODE, call it, and NEVER let
a raise/timeout/bad-JSON/model-error propagate -- LLM failure means "no
event from this path this pass", not "assume the worst" (spec section 20).

extract_event()'s raw EventExtractionResult always passes through
news/validation/event_validator.py before becoming a NewsEvent -- this is
the only code path from an LLM's output to something
risk/news_event_risk.py can see.
"""
from typing import List, Optional

import config
from news.llm.base import LLMEventDetector
from news.llm.mock_provider import MockLLMEventDetector
from news.llm.null_provider import NullLLMEventDetector
from news.news_event import NewsEvent
from news.news_item import NewsItem
from news.validation.event_validator import validate_event

MODE_OFF = "OFF"
MODE_SHADOW = "SHADOW"
VALID_MODES = (MODE_OFF, MODE_SHADOW)


def get_llm_event_detector(mode: Optional[str] = None) -> LLMEventDetector:
    mode = mode if mode is not None else config.NEWS_ENGINE_LLM_MODE
    if mode == MODE_SHADOW:
        return MockLLMEventDetector()
    return NullLLMEventDetector()


class LLMBackedEventDetector:
    def __init__(self, provider: Optional[LLMEventDetector] = None, mode: Optional[str] = None):
        self._mode = mode if mode is not None else config.NEWS_ENGINE_LLM_MODE
        self._provider = provider

    def detect(self, news: NewsItem) -> List[NewsEvent]:
        """Never raises. Empty list on disabled mode, provider failure, or
        an unrecoverable result (see validate_event())."""
        if self._mode not in VALID_MODES or self._mode == MODE_OFF:
            return []

        provider = self._provider or get_llm_event_detector(self._mode)
        try:
            result = provider.extract_event(news)
        except Exception as exc:
            import notify.alert as alert
            alert.log(f"news.llm_detector: extract_event failed for {news.ticker} "
                      f"({exc.__class__.__name__}: {exc}) — event unavailable, "
                      f"pipeline continues without it")
            return []

        candidate = {
            "ticker": result.ticker or news.ticker,
            "event_type": result.event_type,
            "direction": result.direction,
            "severity": result.severity,
            "confidence": result.confidence,
            "source": news.source,
            "timestamp": news.timestamp,
            "headline": news.headline,
            "summary": result.summary,
            "evidence": result.evidence,
            "news_id": news.id,
        }
        event = validate_event(candidate)
        return [event] if event is not None else []

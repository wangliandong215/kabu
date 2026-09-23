"""
news/llm/null_provider.py -- default LLMEventDetector when
config.NEWS_ENGINE_LLM_MODE == "OFF". Always succeeds, always returns
UNKNOWN/zero-confidence -- never raises, never fabricates a real
classification. Mirrors risk/llm/null_provider.py's "OFF still runs
completely normally" contract.
"""
from news.llm.base import EventExtractionResult, LLMEventDetector
from news.news_item import NewsItem


class NullLLMEventDetector(LLMEventDetector):
    def extract_event(self, news: NewsItem) -> EventExtractionResult:
        return EventExtractionResult(
            ticker=news.ticker, event_type="UNKNOWN", direction="UNKNOWN",
            severity=0.0, confidence=0.0,
            summary="LLM event extraction disabled (NEWS_ENGINE_LLM_MODE=OFF)",
            model_name="none",
        )

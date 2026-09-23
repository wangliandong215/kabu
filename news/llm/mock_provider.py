"""
news/llm/mock_provider.py -- canned, deterministic LLMEventDetector for
tests and for exercising SHADOW mode before any real backend exists. Not
wired to any real model -- reuses news/detection/rule_based.py's keyword
signal so the "Mock LLM" pipeline (spec section 26's second pipeline:
Mock News -> LLM Interface -> EventExtractionResult -> Validation ->
NewsEvent) is exercisable end-to-end, while still going through the full
raw/unvalidated-output contract (tests can also construct an
EventExtractionResult directly with out-of-range values to exercise
validation's clamping -- see news/test_llm_detector.py).
"""
from news.detection.rule_based import RuleBasedEventDetector
from news.llm.base import EventExtractionResult, LLMEventDetector
from news.news_item import NewsItem


class MockLLMEventDetector(LLMEventDetector):
    def extract_event(self, news: NewsItem) -> EventExtractionResult:
        detected = RuleBasedEventDetector().detect(news)
        if not detected:
            return EventExtractionResult(
                ticker=news.ticker, event_type="UNKNOWN", direction="UNKNOWN",
                severity=0.0, confidence=0.0,
                summary="[mock-llm] no signal", model_name="mock-llm",
            )
        ev = detected[0]
        return EventExtractionResult(
            ticker=news.ticker, event_type=ev.event_type.value, direction=ev.direction.value,
            severity=ev.severity, confidence=ev.confidence,
            summary=f"[mock-llm] {ev.event_type.value}", evidence=ev.evidence,
            model_name="mock-llm",
        )

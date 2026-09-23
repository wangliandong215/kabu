"""
news/llm/base.py -- V3.4 LLM Provider abstraction (spec sections 16/17).

Mirrors risk/llm/base.py's ABC pattern exactly -- a swappable provider so a
future LocalLLMEventDetector/OllamaEventDetector/OpenAIEventDetector can be
dropped in without news/detection/llm_detector.py, NewsEvent, scoring, or
risk/news_event_risk.py ever changing.

extract_event()'s output is always routed through
news/validation/event_validator.py before it can become a NewsEvent --
nothing here can construct a NewsEvent directly (spec section 18: LLM
output must never reach the Risk Engine unvalidated).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from news.news_item import NewsItem


@dataclass
class EventExtractionResult:
    """Raw, UNVALIDATED LLM output -- event_type/direction are plain
    strings (may not be valid enum members), severity/confidence may be
    out of [0,1] range. See
    news/validation/event_validator.py::validate_event() for the only
    path from this to a NewsEvent."""
    ticker: str
    event_type: str = "UNKNOWN"
    direction: str = "UNKNOWN"
    severity: float = 0.0
    confidence: float = 0.0
    summary: str = ""
    evidence: str = ""
    model_name: str = "none"
    model_version: str = ""
    prompt_version: str = "v1"
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMEventDetector(ABC):
    @abstractmethod
    def extract_event(self, news: NewsItem) -> EventExtractionResult:
        """Implementations may raise (timeout, unavailable, invalid JSON,
        model error) -- news/detection/llm_detector.py catches everything
        and treats a raise the same as 'no event available' (spec section
        20: LLM failure must never trigger an emergency exit)."""
        raise NotImplementedError

"""
news/news_event.py -- V3.4's structured Event Signal (spec section 6). The
one object that crosses from "information" into
risk/news_event_risk.py's Risk Engine sub-check.

Nothing upstream should construct a NewsEvent directly except
news/validation/event_validator.py::validate_event() -- that is the single
schema/range/enum validation boundary (spec section 18) both the rule-based
detector and the LLM path must go through.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from news.event_types import Direction, EventType
from news.news_sources import Source


@dataclass
class NewsEvent:
    ticker: str
    event_type: EventType
    direction: Direction

    severity: float      # 0.0..1.0
    confidence: float    # 0.0..1.0
    news_score: float    # -1.0..+1.0, undecayed

    source: Source
    timestamp: datetime

    headline: str = ""
    summary: str = ""
    evidence: str = ""

    decay_hours: float = 24.0
    emergency: bool = False
    news_id: str = ""     # links back to the originating NewsItem.id

    def age_hours(self, as_of: Optional[datetime] = None) -> float:
        as_of = as_of or datetime.now()
        return max(0.0, (as_of - self.timestamp).total_seconds() / 3600.0)

    def effective_score(self, as_of: Optional[datetime] = None) -> float:
        """news_score after time decay -- see news/scoring/decay.py.
        Imported lazily to avoid a decay.py <-> news_event.py import
        cycle (decay.py keys its default-hours table off EventType, not
        NewsEvent)."""
        from news.scoring.decay import decayed_score
        return decayed_score(self.news_score, self.timestamp, self.decay_hours, as_of)

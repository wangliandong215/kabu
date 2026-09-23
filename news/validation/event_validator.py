"""
news/validation/event_validator.py -- V3.4 spec section 18's single
Schema / Range / Enum / Business-rule validation boundary. EVERY candidate
event -- rule-based or LLM -- passes through validate_event() before
becoming a NewsEvent; nothing else in this package is allowed to construct
a NewsEvent directly (see news/news_event.py's docstring).

This is what makes V3.4 spec Test 1 (severity=5.7, confidence=-2) and
Test 2 (event_type=UNKNOWN never emergency) safe: bad input is
clamped/downgraded here, never passed through to
risk/news_event_risk.py unvalidated.
"""
from datetime import datetime
from numbers import Number
from typing import Any, Dict, Optional, Type, TypeVar

from news.event_types import Direction, EventType, is_emergency_candidate
from news.news_event import NewsEvent
from news.news_sources import Source
from news.scoring import decay as decay_mod
from news.scoring.news_score import compute_news_score

E = TypeVar("E")


def _safe_enum(enum_cls: Type[E], value: Any, default: E) -> E:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    if not isinstance(value, Number) or isinstance(value, bool):
        return default
    return max(lo, min(hi, float(value)))


def validate_event(candidate: Dict[str, Any]) -> Optional[NewsEvent]:
    """candidate is a plain dict (from news/detection/rule_based.py or
    from an LLM's raw EventExtractionResult via
    news/detection/llm_detector.py) with keys: ticker, event_type,
    direction, severity, confidence, source, timestamp, headline, summary,
    evidence, decay_hours (optional), news_id (optional).

    Returns None only when the candidate is unrecoverable (no ticker) --
    every other field is clamped/defaulted rather than rejected, per spec
    section 18 ("必须拒绝或安全归一化", never silently pass through)."""
    ticker = candidate.get("ticker")
    if not ticker:
        return None

    event_type = _safe_enum(EventType, candidate.get("event_type"), EventType.UNKNOWN)
    direction = _safe_enum(Direction, candidate.get("direction"), Direction.UNKNOWN)
    source = _safe_enum(Source, candidate.get("source"), Source.OTHER)
    severity = _clamp_float(candidate.get("severity"), 0.0, 1.0, default=0.0)
    confidence = _clamp_float(candidate.get("confidence"), 0.0, 1.0, default=0.0)

    timestamp = candidate.get("timestamp")
    if not isinstance(timestamp, datetime):
        timestamp = datetime.now()

    decay_hours = candidate.get("decay_hours")
    default_decay = decay_mod.DEFAULT_DECAY_HOURS.get(
        event_type, decay_mod.DEFAULT_DECAY_HOURS[EventType.UNKNOWN])
    if not isinstance(decay_hours, Number) or isinstance(decay_hours, bool) or decay_hours <= 0:
        decay_hours = default_decay

    news_score = compute_news_score(direction, severity, confidence, source)
    emergency = is_emergency_candidate(event_type, severity, confidence)

    return NewsEvent(
        ticker=str(ticker),
        event_type=event_type,
        direction=direction,
        severity=severity,
        confidence=confidence,
        news_score=news_score,
        source=source,
        timestamp=timestamp,
        headline=str(candidate.get("headline", "")),
        summary=str(candidate.get("summary", "")),
        evidence=str(candidate.get("evidence", "")),
        decay_hours=float(decay_hours),
        emergency=emergency,
        news_id=str(candidate.get("news_id", "")),
    )

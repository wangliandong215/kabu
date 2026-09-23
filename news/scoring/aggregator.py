"""
news/scoring/aggregator.py -- V3.4 Event Aggregation (spec section 12):
multiple events on the same ticker must not simply sum ("不能简单
score1+score2+score3 造成无限叠加"), and the same story re-appearing must
not double-count (spec Test 4).

Rules, in order:
  1. Dedup by (ticker, normalized headline) -- keep only the
     highest-confidence copy of each distinct story.
  2. Decay every remaining event to `as_of` (spec section 11's Time Decay,
     applied here rather than by the caller so "effective score" always
     means "decayed AND aggregated").
  3. Geometric-dampened sum: sort by |decayed score| descending, weight
     1, 0.5, 0.25, ... so a 4th or 5th corroborating event keeps adding
     signal but with rapidly diminishing effect, then clamp to [-1, 1].
  4. Severity/confidence aggregate as "worst case wins" (max severity,
     paired confidence) -- simple and explainable per spec section 12's
     "第一版可以采用简单、可解释的规则".
  5. Emergency is OR'd across all events -- a single confirmed emergency
     candidate must never be diluted by averaging with unrelated news.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from news.news_event import NewsEvent
from news.news_item import normalize_headline
from news.scoring.decay import decayed_score


@dataclass
class AggregatedNewsSignal:
    ticker: str
    effective_score: float = 0.0     # -1..+1, decayed + aggregated
    worst_severity: float = 0.0
    confidence: float = 0.0           # confidence of the worst-severity event
    emergency: bool = False
    contributing_events: List[NewsEvent] = field(default_factory=list)
    as_of: datetime = field(default_factory=datetime.now)


def _dedup(events: List[NewsEvent]) -> List[NewsEvent]:
    best: Dict[str, NewsEvent] = {}
    for ev in events:
        key = f"{ev.ticker}:{normalize_headline(ev.headline)}"
        current = best.get(key)
        if current is None or ev.confidence > current.confidence:
            best[key] = ev
    return list(best.values())


def aggregate(events: List[NewsEvent], as_of: Optional[datetime] = None) -> AggregatedNewsSignal:
    as_of = as_of or datetime.now()
    if not events:
        return AggregatedNewsSignal(ticker="", as_of=as_of)

    ticker = events[0].ticker
    unique_events = _dedup(events)

    decayed = [(ev, decayed_score(ev.news_score, ev.timestamp, ev.decay_hours, as_of))
               for ev in unique_events]
    # Keep an emergency-flagged event even if its decayed score rounds to
    # ~0 -- emergency status itself does not decay away silently.
    contributing = [(ev, s) for ev, s in decayed if abs(s) > 1e-6 or ev.emergency]

    contributing.sort(key=lambda pair: abs(pair[1]), reverse=True)
    total = 0.0
    weight = 1.0
    for _, s in contributing:
        total += s * weight
        weight *= 0.5
    effective_score = max(-1.0, min(1.0, total))

    worst_event = max(unique_events, key=lambda ev: ev.severity)
    emergency = any(ev.emergency for ev in unique_events)

    return AggregatedNewsSignal(
        ticker=ticker,
        effective_score=effective_score,
        worst_severity=worst_event.severity,
        confidence=worst_event.confidence,
        emergency=emergency,
        contributing_events=unique_events,
        as_of=as_of,
    )

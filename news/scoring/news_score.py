"""
news/scoring/news_score.py -- V3.4 News Score (spec section 10): unified
-1.0..+1.0 "Effective News Score" combining Direction / Severity /
Confidence / Source Reliability. Deliberately NOT a flat
Positive=+1/Negative=-1 (spec explicitly forbids that) -- magnitude scales
with how severe, how certain, and how reliable the source is. Time decay
(spec section 11) is applied separately, later, by
news/scoring/decay.py -- this function returns the UNDECAYED score as of
the moment the event happened.
"""
from news.event_types import Direction
from news.news_sources import RELIABILITY, Source


def compute_news_score(direction: Direction, severity: float, confidence: float,
                        source: Source) -> float:
    if direction in (Direction.NEUTRAL, Direction.UNKNOWN):
        return 0.0
    sign = 1.0 if direction == Direction.POSITIVE else -1.0
    reliability = RELIABILITY.get(source, RELIABILITY[Source.OTHER])
    magnitude = max(0.0, min(1.0, severity)) * max(0.0, min(1.0, confidence)) * reliability
    return max(-1.0, min(1.0, sign * magnitude))

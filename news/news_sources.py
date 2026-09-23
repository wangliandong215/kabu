"""
news/news_sources.py -- V3.4 Source taxonomy (spec section 3) + per-source
reliability weight used by news/scoring/news_score.py and confidence.py.
Phase 1 never fetches from any of these except via news/ingestion/mock.py
-- Source is a data-model field only until a real source is wired (spec
section 27, "暂时不要直接接Reuters/Bloomberg/Social API").
"""
from enum import Enum
from typing import Dict


class Source(str, Enum):
    REUTERS = "REUTERS"
    BLOOMBERG = "BLOOMBERG"
    SEC = "SEC"
    EARNINGS = "EARNINGS"
    FDA = "FDA"
    MACRO = "MACRO"
    INSIDER = "INSIDER"
    SOCIAL = "SOCIAL"
    OTHER = "OTHER"


# How much weight this source's headlines get in news/scoring/news_score.py
# and confidence.py. Reference values only (spec section 8/9 caution
# against hard-coding examples as final rules) -- tune via measurement,
# not guesswork, once real sources exist.
RELIABILITY: Dict[Source, float] = {
    Source.SEC: 1.00,
    Source.FDA: 1.00,
    Source.EARNINGS: 0.95,
    Source.REUTERS: 0.90,
    Source.BLOOMBERG: 0.90,
    Source.MACRO: 0.80,
    Source.INSIDER: 0.75,
    Source.OTHER: 0.60,
    Source.SOCIAL: 0.35,
}

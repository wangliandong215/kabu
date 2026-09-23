"""
news/scoring/confidence.py -- V3.4 Confidence (spec section 9): "系统对于
事件识别/分类结果正确的置信程度", strictly separate from Severity. Phase 1
derives it from source reliability plus how many distinct keyword hits
corroborated the classification -- not from the same inputs as severity,
so a severe-but-uncertain event and a mild-but-certain one stay
distinguishable (spec's own example).
"""
from news.news_sources import RELIABILITY, Source

_MATCH_BOOST_PER_EXTRA_KEYWORD = 0.05
_MAX_MATCH_BOOST = 0.20


def compute_confidence(source: Source, keyword_match_count: int = 1) -> float:
    base = RELIABILITY.get(source, RELIABILITY[Source.OTHER])
    extra_matches = max(0, keyword_match_count - 1)
    boost = min(_MAX_MATCH_BOOST, _MATCH_BOOST_PER_EXTRA_KEYWORD * extra_matches)
    return max(0.0, min(1.0, base + boost))
